"""
trading/strategies/ai_signal.py
────────────────────────────────
AI Signal Strategy — uses the LightGBM model to pick stocks daily.

Flow (runs once per trading day at 09:30 ET):
  1. Refresh feature store if stale (> 20 hours old)
  2. Score all stocks → top-N picks from LightGBM
  3. Exit positions that hit any exit rule (see below)
  4. Enter new positions for tickers in top-N not already owned
  5. Log every trade to the SQLite journal with full thesis

Exit rules (in priority order):
  STOP_LOSS       gain <= -stop_loss_pct           default -3%
  PROFIT_TARGET   gain >= profit_target_pct        default +5%
  TRAIL_STOP      price falls 3% below peak price  (locks in profits)
  TIME_STOP       held > max_holding_days          default 10 days
  SIGNAL_EXIT     dropped out of model top picks   (thesis invalidated)

Parameters (all tunable via params.yaml → ai_signal section):
  budget            Total USD across all AI-signal positions   ($2,000)
  max_positions     Maximum simultaneous holdings              (5)
  profit_target     Sell when gain >= this fraction            (0.05)
  stop_loss         Sell when loss >= this fraction            (0.03)
  max_holding_days  Force-exit after this many days            (10)
  pipeline_mode     Feature refresh scope                      ("top100")
  min_confidence    Only enter HIGH or MED picks               ("MED")

State : state/ai_signal_state.json  (positions, last_scored)
Journal: state/trade_journal.db     (full trade history, grades)
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from trading.client import AlpacaClient

# Each worker thread gets its own AlpacaClient (and therefore its own
# requests.Session).  A single shared session is NOT thread-safe — urllib3's
# connection pool can deadlock when hit by 10 threads simultaneously.
_thread_local = threading.local()


def _thread_get_price(ticker: str) -> float:
    """Fetch the latest price using a per-thread AlpacaClient."""
    if not hasattr(_thread_local, "client"):
        _thread_local.client = AlpacaClient()
    return _thread_local.client.get_latest_price(ticker)
from trading.journal import TradeJournal
from trading.logger import get_logger
from trading.portfolio.kelly import KellySizer
from trading.portfolio.optimizer import HRPOptimizer
from trading.reasoning.claude_analyst import ClaudeAnalyst
from trading.reasoning.signal_aggregator import SignalAggregator
from trading.risk.manager import RiskManager
from trading.strategies.base import BaseStrategy

logger = get_logger(__name__)

_STATE_FILE    = Path("state/ai_signal_state.json")
_FEATURE_STORE = Path("state/features/all_features.parquet")
_STALE_HOURS   = 20


class AISignalStrategy(BaseStrategy):
    """
    Daily strategy that buys the LightGBM model's top stock picks and
    exits on profit target, stop-loss, trailing stop, time stop, or signal
    invalidation. Every trade is logged to the SQLite journal with thesis.
    """

    def __init__(
        self,
        # ── Dynamic sizing (live equity) ─────────────────────────────────────
        budget_pct:            float = 0.90,    # fraction of live equity to deploy
        min_position_dollars:  float = 2_000.0, # min $ per slot → sets max_positions
        max_positions_cap:     int   = 20,       # hard ceiling on simultaneous positions
        # ── Legacy overrides (set > 0 to pin a fixed value) ──────────────────
        budget:                float = 0.0,      # 0 = use budget_pct
        max_positions:         int   = 0,        # 0 = compute from min_position_dollars
        # ── Exit / entry rules ────────────────────────────────────────────────
        profit_target:         float = 0.05,
        stop_loss:             float = 0.03,
        max_holding_days:      int   = 10,
        pipeline_mode:         str   = "top100",
        min_confidence:        str   = "MED",
        # ── Risk parameters ───────────────────────────────────────────────────
        daily_loss_limit:      float = 0.02,
        max_drawdown:          float = 0.10,
        max_position_pct:      float = 0.30,
        correlation_threshold: float = 0.85,
    ) -> None:
        # Dynamic sizing config
        self.budget_pct           = budget_pct
        self.min_position_dollars = min_position_dollars
        self.max_positions_cap    = max_positions_cap
        self._fixed_budget        = budget        # 0 = dynamic
        self._fixed_max_positions = max_positions  # 0 = dynamic

        # Working values — updated each cycle by _update_dynamic_sizing()
        self.budget        = budget if budget > 0 else 2_000.0
        self.max_positions = max_positions if max_positions > 0 else 5

        self.profit_target    = profit_target
        self.stop_loss        = stop_loss
        self.max_holding_days = max_holding_days
        self.pipeline_mode    = pipeline_mode
        self.min_confidence   = min_confidence

        self._client     = AlpacaClient()
        self._journal    = TradeJournal()
        self._aggregator = SignalAggregator()
        self._analyst    = ClaudeAnalyst()
        self._hrp        = HRPOptimizer()
        self._kelly      = KellySizer()
        self._risk       = RiskManager(
            daily_loss_limit      = daily_loss_limit,
            max_drawdown          = max_drawdown,
            max_position_pct      = max_position_pct,
            correlation_threshold = correlation_threshold,
        )
        self._state      = self._load_state()

        # Single Telegram instance — avoids re-initialising on every alert
        try:
            from trading.alerts.telegram import TelegramBot
            self._telegram = TelegramBot()
        except Exception:
            self._telegram = None

    # ── BaseStrategy ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "AISignal"

    # ── Dynamic sizing ────────────────────────────────────────────────────────

    def _update_dynamic_sizing(self, equity: float) -> None:
        """
        Recompute self.budget and self.max_positions from live equity.
        Called once per cycle, immediately after the equity fetch.

        Logic:
          budget        = equity × budget_pct   (unless _fixed_budget > 0)
          max_positions = floor(budget / min_position_dollars),
                          clamped to [1, max_positions_cap]
                          (unless _fixed_max_positions > 0)
        """
        if equity <= 0:
            return

        # Budget
        if self._fixed_budget > 0:
            self.budget = self._fixed_budget
        else:
            self.budget = round(equity * self.budget_pct, 2)

        # Max positions
        if self._fixed_max_positions > 0:
            self.max_positions = self._fixed_max_positions
        else:
            dynamic = max(1, int(self.budget / self.min_position_dollars))
            self.max_positions = min(dynamic, self.max_positions_cap)

        per_slot = self.budget / self.max_positions if self.max_positions else self.budget
        logger.info(
            f"{self.name}: sizing — equity=${equity:,.0f} "
            f"→ budget=${self.budget:,.0f} | {self.max_positions} positions "
            f"| ~${per_slot:,.0f}/slot"
        )

    # ── Cycle ─────────────────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info(f"{self.name}: starting cycle")

        # ── Risk gate + dynamic sizing ────────────────────────────────────────
        try:
            equity = float(self._client.get_account().get("equity", 0))
            self._update_dynamic_sizing(equity)
            self._risk.update_daily_reference(equity)
            halted, halt_reason = self._risk.check_halt(equity)
            if halted:
                logger.warning(f"{self.name}: RISK HALT — {halt_reason} — skipping entries")
                self._send_cycle_alert(
                    f"RISK HALT — {halt_reason}\n"
                    f"  Equity: ${equity:,.2f}\n"
                    f"  All new entries blocked until drawdown recovers.",
                    emoji="🛑",
                )
        except Exception as exc:
            logger.warning(f"{self.name}: risk check failed — {exc}")
            equity = 0.0
            halted = False

        self._refresh_features_if_stale()
        picks = self._get_model_picks()
        self._manage_exits(picks)

        try:
            if picks and not halted:
                self._enter_new_positions(picks, equity)
            elif halted:
                logger.warning(f"{self.name}: new entries blocked by risk manager")
            else:
                logger.info(f"{self.name}: no valid picks — regime gate or empty store")
        except Exception as exc:
            logger.error(f"{self.name}: entry phase failed — {exc}", exc_info=True)
        finally:
            # Always persist state — even if entries partially failed
            self._save_state()

        n_pos = len(self._state["positions"])
        logger.info(f"{self.name}: cycle complete — {n_pos} positions")

        # ── Daily cycle summary ───────────────────────────────────────────────
        self._send_cycle_summary(equity, picks, n_pos)

    def status(self) -> dict:
        pos = self._state["positions"]
        stats = self._journal.get_performance_stats(days=30)
        return {
            "strategy":      self.name,
            "positions":     len(pos),
            "tickers":       list(pos.keys()),
            "last_scored":   self._state.get("last_scored", "never"),
            "budget_used":   round(sum(p["budget_used"] for p in pos.values()), 2),
            "budget_total":  self.budget,
            "30d_trades":    stats.get("total_trades", 0),
            "30d_win_rate":  stats.get("win_rate", 0),
            "30d_pnl":       stats.get("total_pnl", 0),
        }

    # ── Step 1 — Feature refresh ──────────────────────────────────────────────

    def _refresh_features_if_stale(self) -> None:
        if _FEATURE_STORE.exists():
            age_h = (
                datetime.now() -
                datetime.fromtimestamp(_FEATURE_STORE.stat().st_mtime)
            ).total_seconds() / 3600
            if age_h < _STALE_HOURS:
                logger.info(f"{self.name}: features fresh ({age_h:.1f}h) — skipping refresh")
                return

        logger.info(f"{self.name}: refreshing features ({self.pipeline_mode})...")
        try:
            from trading.data.pipeline import FeaturePipeline
            stats = FeaturePipeline(use_signals=True).run(mode=self.pipeline_mode)
            logger.info(f"{self.name}: features refreshed — {stats['ok']} tickers, {stats['rows']:,} rows")
        except Exception as exc:
            logger.warning(f"{self.name}: feature refresh failed — {exc}")

    # ── Step 2 — Model picks ──────────────────────────────────────────────────

    def _get_model_picks(self) -> list[dict]:
        try:
            from trading.models.lightgbm_model import LightGBMPredictor
            all_picks = LightGBMPredictor().score(top_n=self.max_positions * 2)
        except Exception as exc:
            logger.warning(f"{self.name}: scoring failed — {exc}")
            return []

        conf_order = {"HIGH": 3, "MED": 2, "LOW": 1}
        min_level  = conf_order.get(self.min_confidence, 2)
        picks = [p for p in all_picks if conf_order.get(p.get("confidence", "LOW"), 1) >= min_level]
        picks = picks[:self.max_positions * 2]   # gather extras for LLM to filter

        self._state["last_scored"] = datetime.now().isoformat()
        logger.info(f"{self.name}: {len(all_picks)} picks → {len(picks)} pass {self.min_confidence}+ filter")

        # ── LLM reasoning layer ───────────────────────────────────────────────
        if picks:
            logger.info(f"{self.name}: gathering signals for {len(picks)} candidates...")
            signals_map = self._aggregator.gather_many(picks)
            picks = self._analyst.screen_picks(picks, signals_map)
            logger.info(f"{self.name}: {len(picks)} picks approved by Claude")

        return picks[:self.max_positions]

    # ── Step 3 — Exit management ──────────────────────────────────────────────

    def _manage_exits(self, current_picks: list[dict]) -> None:
        positions    = self._state["positions"]
        if not positions:
            return

        pick_tickers = {p["ticker"] for p in current_picks}
        to_exit: list[tuple[str, int, str]] = []

        # Fetch all exit prices in parallel
        held_tickers = list(positions.keys())
        exit_prices: dict[str, float] = self._fetch_prices_parallel(held_tickers)

        for ticker, pos in list(positions.items()):
            price = exit_prices.get(ticker)
            if price is None:
                continue

            entry        = pos["entry_price"]
            gain         = (price - entry) / entry
            entry_date   = datetime.fromisoformat(pos["entry_date"])
            holding_days = (datetime.now() - entry_date).days

            # Update peak & trailing floor in journal
            trail_floor = self._journal.update_peak(ticker, price)

            # ── Exit rules (evaluated in priority order) ──────────────────────
            if gain <= -self.stop_loss:
                reason = "STOP_LOSS"
            elif gain >= self.profit_target:
                reason = "PROFIT_TARGET"
            elif trail_floor > entry and price < trail_floor:
                # Trailing stop: only active once price has risen above entry
                reason = "TRAIL_STOP"
                logger.info(
                    f"{self.name}: {ticker} TRAIL_STOP — "
                    f"price ${price:.2f} < floor ${trail_floor:.2f}"
                )
            elif holding_days >= self.max_holding_days:
                reason = "TIME_STOP"
                logger.info(
                    f"{self.name}: {ticker} TIME_STOP — held {holding_days}d "
                    f">= {self.max_holding_days}d max"
                )
            elif current_picks and ticker not in pick_tickers:
                reason = "SIGNAL_EXIT"
            else:
                continue  # hold

            to_exit.append((ticker, pos["shares"], reason))

        for ticker, shares, reason in to_exit:
            try:
                self._client.place_market_order(symbol=ticker, qty=shares, side="sell")
                price = self._get_price(ticker)
                closed = self._journal.close_trade(
                    ticker=ticker, exit_price=price, exit_reason=reason
                )
                pnl = closed["pnl"] if closed else 0.0
                logger.info(
                    f"{self.name}: SELL {shares}x {ticker} @ ${price:.2f} "
                    f"| P&L ${pnl:+.2f} | {reason}"
                )
                self._send_sell_alert(ticker, shares, price, pnl, reason)
                del positions[ticker]
            except Exception as exc:
                logger.error(f"{self.name}: sell failed for {ticker} — {exc}")

    # ── Step 4 — Entry ────────────────────────────────────────────────────────

    def _enter_new_positions(self, picks: list[dict], equity: float = 0.0) -> None:
        positions  = self._state["positions"]
        slots_open = self.max_positions - len(positions)
        if slots_open <= 0:
            logger.info(f"{self.name}: portfolio full ({self.max_positions} positions)")
            return

        budget_used = sum(p["budget_used"] for p in positions.values())
        budget_left = self.budget - budget_used
        if budget_left < 10:
            logger.info(f"{self.name}: budget exhausted (${budget_left:.2f} left)")
            return

        # Cap budget_left to actual available buying power so we never
        # attempt orders we can't fill (other strategies occupy cash too).
        try:
            buying_power = float(self._client.get_buying_power())
            if buying_power < budget_left:
                logger.info(
                    f"{self.name}: buying power ${buying_power:,.0f} < "
                    f"strategy budget ${budget_left:,.0f} — capping to buying power"
                )
                budget_left = buying_power
        except Exception as exc:
            logger.warning(f"{self.name}: could not fetch buying power — {exc}")

        # Filter to only new tickers, then apply per-pick risk checks
        existing_tickers = list(positions.keys())
        new_picks = []
        for p in picks:
            if p["ticker"] in positions:
                continue
            allowed, reason = self._risk.check_can_enter(
                p["ticker"], equity, existing_tickers
            )
            if allowed:
                new_picks.append(p)
                existing_tickers.append(p["ticker"])  # update for next iteration
            else:
                logger.info(f"{self.name}: risk block {p['ticker']} — {reason}")
        new_picks = new_picks[:slots_open]
        if not new_picks:
            return

        new_tickers = [p["ticker"] for p in new_picks]

        # ── Step 4a: Fetch live prices for all new tickers (parallel) ───────
        live_prices: dict[str, float] = self._fetch_prices_parallel(new_tickers)

        # ── Step 4b: HRP weights (with timeout guard) ────────────────────────
        # HRP reads 20 parquet files from state/bars/. On OneDrive or slow
        # drives this can hang indefinitely.  We run it on a daemon thread
        # with an 8-second timeout; if it doesn't finish we fall back to
        # equal-weight allocation so the rest of the cycle is never blocked.
        logger.info(f"{self.name}: step 4b — running HRP allocation for {len(new_tickers)} tickers...")
        _hrp_result: list = [None]
        def _run_hrp() -> None:
            try:
                _hrp_result[0] = self._hrp.allocate_dollars(new_tickers, budget_left)
            except Exception as exc:
                logger.warning(f"{self.name}: HRP error — {exc}")
        _hrp_thread = threading.Thread(target=_run_hrp, daemon=True)
        _hrp_thread.start()
        _hrp_thread.join(timeout=8)
        if _hrp_result[0] is not None:
            hrp_alloc = _hrp_result[0]
            logger.info(f"{self.name}: step 4b done — HRP complete")
        else:
            logger.warning(f"{self.name}: step 4b — HRP timed out, using equal-weight fallback")
            per = budget_left / len(new_tickers)
            hrp_alloc = {t: {"dollars": per, "shares": 0, "weight": round(1/len(new_tickers), 4), "price": 0}
                         for t in new_tickers}

        # ── Step 4c: Kelly fractions ──────────────────────────────────────────
        macro_mult = self._get_macro_mult()
        kelly_sizes = self._kelly.size_all(
            picks         = new_picks,
            total_budget  = budget_left,
            prices        = live_prices,
            stop_loss     = self.stop_loss,
            profit_target = self.profit_target,
            macro_mult    = macro_mult,
        )

        # ── Blend: average HRP and Kelly dollar allocations ───────────────────
        per_slot   = budget_left / len(new_tickers)   # equal-weight fallback
        blended: dict[str, dict] = {}
        for ticker in new_tickers:
            hrp_d   = hrp_alloc.get(ticker, {}).get("dollars", per_slot)
            kelly_d = kelly_sizes.get(ticker, {}).get("dollars", hrp_d)
            dollars = (hrp_d + kelly_d) / 2
            # ── Concentration cap ─────────────────────────────────────────────
            dollars = self._risk.cap_position_dollars(dollars, equity)
            price   = live_prices.get(ticker, 0)
            shares  = int(dollars / price) if price > 0 else 0
            blended[ticker] = {"dollars": round(dollars, 2), "shares": shares, "price": price}

        logger.info(
            f"{self.name}: HRP+Kelly sizing — "
            + " | ".join(f"{t} ${v['dollars']:.0f} ({v['shares']}sh @ ${v['price']:.2f})"
                         for t, v in blended.items())
        )

        entered = 0
        for pick in new_picks:
            ticker = pick["ticker"]
            sizing = blended.get(ticker, {})

            try:
                price  = sizing.get("price", 0)    # already fetched live above
                shares = sizing.get("shares", 0)   # computed from live price
                if price <= 0:
                    logger.warning(f"{self.name}: {ticker} — no live price, skipping")
                    continue
                if shares < 1:
                    logger.info(f"{self.name}: {ticker} @ ${price:.2f} — insufficient budget for 1 share (${sizing.get('dollars',0):.0f} allocated), skipping")
                    continue

                self._client.place_market_order(symbol=ticker, qty=shares, side="buy")
                logger.info(f"{self.name}: ORDER SENT — BUY {shares}x {ticker} @ ${price:.2f}")

                # Build thesis — prefer Claude's reasoning, fall back to LightGBM
                features    = pick.get("top_features", [])
                llm_thesis  = pick.get("llm_thesis", "")
                llm_conv    = pick.get("llm_conviction", "")
                lgbm_thesis = (
                    f"LightGBM {pick.get('confidence','?')} | "
                    f"pred={pick.get('pred_pct','?')} | "
                    f"drivers: {', '.join(features[:3])}"
                )
                thesis = (
                    f"[Claude {llm_conv}] {llm_thesis} | {lgbm_thesis}"
                    if llm_thesis else lgbm_thesis
                )

                trade_id = self._journal.open_trade(
                    strategy    = self.name,
                    ticker      = ticker,
                    shares      = shares,
                    entry_price = price,
                    thesis      = thesis,
                    pred_return = pick.get("pred_return", 0.0),
                    confidence  = pick.get("confidence", "?"),
                )

                positions[ticker] = {
                    "trade_id":    trade_id,
                    "entry_price": price,
                    "shares":      shares,
                    "entry_date":  datetime.now().isoformat(),
                    "pred_return": pick.get("pred_return", 0),
                    "confidence":  pick.get("confidence", "?"),
                    "budget_used": round(price * shares, 2),
                    "key_features": features,
                }
                logger.info(
                    f"{self.name}: BUY {shares}x {ticker} @ ${price:.2f} "
                    f"| {pick.get('pred_pct','?')} | {pick.get('confidence','?')} "
                    f"| journal #{trade_id}"
                )
                self._send_buy_alert(ticker, shares, price, pick, thesis)
                slots_open -= 1
                entered    += 1

            except Exception as exc:
                err = str(exc)
                if "insufficient buying power" in err.lower() or "40310000" in err:
                    logger.warning(
                        f"{self.name}: buying power exhausted after {entered} entries — stopping"
                    )
                    break
                logger.error(f"{self.name}: buy failed for {ticker} — {exc}")

        if entered == 0:
            logger.info(f"{self.name}: no new entries this cycle")

    # ── Telegram alerts ───────────────────────────────────────────────────────

    def _send_buy_alert(self, ticker, shares, price, pick, thesis) -> None:
        if not self._telegram:
            return
        try:
            self._telegram.send_buy_alert(
                ticker     = ticker,
                shares     = shares,
                price      = price,
                reasoning  = thesis,
                confidence = pick.get("confidence", ""),
                source     = "LightGBM + Claude",
            )
        except Exception as exc:
            logger.warning(f"{self.name}: buy alert failed — {exc}")

    def _send_sell_alert(self, ticker, shares, price, pnl, reason) -> None:
        if not self._telegram:
            return
        try:
            self._telegram.send_sell_alert(
                ticker    = ticker,
                shares    = shares,
                price     = price,
                pnl       = pnl,
                reasoning = reason,
                urgency   = "HIGH" if reason == "STOP_LOSS" else "NORMAL",
            )
        except Exception as exc:
            logger.warning(f"{self.name}: sell alert failed — {exc}")

    def _send_cycle_alert(self, body: str, emoji: str = "ℹ️") -> None:
        """Send a plain-text one-off alert (risk halt, macro gate, etc.)."""
        if not self._telegram:
            return
        try:
            self._telegram._send_message(
                f"{emoji} AISignal — {datetime.now().strftime('%d %b %H:%M IST')}\n\n"
                f"{body}"
            )
        except Exception as exc:
            logger.warning(f"{self.name}: cycle alert failed — {exc}")

    def _send_cycle_summary(self, equity: float, picks: list, n_pos: int) -> None:
        """Send a brief daily cycle summary so you always hear from the bot."""
        if not self._telegram:
            return
        try:
            macro_mult = self._get_macro_mult()
            regime_str = (
                "RISK-OFF (macro gate active)" if macro_mult < 0.5
                else "CAUTION (reduced sizing)" if macro_mult < 1.0
                else "NORMAL"
            )

            pos_tickers = list(self._state["positions"].keys())
            pos_str = ", ".join(pos_tickers) if pos_tickers else "none"

            msg = (
                f"AISignal Daily Cycle\n"
                f"  {datetime.now().strftime('%d %b %Y %H:%M IST')}\n\n"
                f"  Equity:    ${equity:,.2f}\n"
                f"  Regime:    {regime_str}\n"
                f"  Picks:     {len(picks)} approved by model\n"
                f"  Positions: {n_pos} open ({pos_str})\n"
            )

            if not picks and macro_mult < 0.5:
                msg += "\n  No trades entered — macro gate suppressing entries (VIX high)."
            elif not picks:
                msg += "\n  No trades entered — model found no high-confidence picks today."

            self._telegram._send_message(msg)
        except Exception as exc:
            logger.warning(f"{self.name}: cycle summary alert failed — {exc}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fetch_prices_parallel(
        self, tickers: list[str], timeout: float = 25.0
    ) -> dict[str, float]:
        """
        Fetch latest prices for *tickers* in parallel using per-thread clients.

        Key design decisions:
        - Each thread creates its own AlpacaClient (thread-local) to avoid
          sharing a requests.Session, which deadlocks urllib3's connection pool.
        - We do NOT use `with ThreadPoolExecutor` because __exit__ calls
          shutdown(wait=True), which blocks forever if threads are hung.
        - Instead we call shutdown(wait=False, cancel_futures=True) ourselves
          so stuck threads are abandoned immediately after the timeout.
        """
        if not tickers:
            return {}

        prices: dict[str, float] = {}
        executor = ThreadPoolExecutor(max_workers=min(10, len(tickers)))
        try:
            fut_map = {executor.submit(_thread_get_price, t): t for t in tickers}
            for fut in as_completed(fut_map, timeout=timeout):
                t = fut_map[fut]
                try:
                    prices[t] = fut.result()
                except Exception as exc:
                    logger.warning(f"{self.name}: price unavailable for {t} — {exc}")
        except TimeoutError:
            completed = len(prices)
            missing   = [t for t in tickers if t not in prices]
            logger.warning(
                f"{self.name}: price fetch timed out after {timeout:.0f}s "
                f"({completed}/{len(tickers)} fetched) — skipping {missing}"
            )
        finally:
            # cancel_futures=True drops queued-but-not-started work;
            # wait=False means we don't block on threads that are mid-request.
            executor.shutdown(wait=False, cancel_futures=True)

        logger.info(
            f"{self.name}: prices fetched — "
            + ", ".join(f"{t}=${v:.2f}" for t, v in prices.items())
        ) if prices else None
        return prices

    def _get_price(self, ticker: str) -> float:
        return self._client.get_latest_price(ticker)

    def _get_macro_mult(self) -> float:
        """Return position-size multiplier from macro regime (0.25–1.0)."""
        try:
            from trading.signals.macro import MacroData
            return MacroData().get_position_size_multiplier()
        except Exception:
            return 1.0

    # ── State persistence ─────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if _STATE_FILE.exists():
            try:
                return json.loads(_STATE_FILE.read_text())
            except Exception:
                pass
        return {"positions": {}, "last_scored": None}

    def _save_state(self) -> None:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(self._state, indent=2))
