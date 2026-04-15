"""
trading/risk/manager.py
────────────────────────
Portfolio-level risk manager — sits above individual strategies.

Four rules (all tunable via params.yaml → risk section):

  DAILY_LOSS_LIMIT
    If portfolio value drops > daily_loss_limit (default 2%) since today's
    market-open snapshot → block all new entries for the rest of the day.
    Resets automatically at the next trading day.

  MAX_DRAWDOWN
    If portfolio value drops > max_drawdown (default 10%) from its all-time
    peak → trigger a HALT: block new entries and send a Telegram alert.
    Auto-clears when drawdown recovers below the threshold.

  CONCENTRATION
    Cap each new position so it cannot exceed max_position_pct (default 30%)
    of total portfolio equity. Applied silently — sizes are reduced, not blocked.

  CORRELATION
    Block entry into a new ticker if its 60-day return series correlates
    > correlation_threshold (default 0.85) with any existing position.
    Prevents doubling up on effectively the same trade.

State: state/risk_state.json
    peak_equity       float   all-time high equity seen
    day_open_equity   float   equity at the start of today
    day_open_date     str     ISO date of last day-open snapshot
    halt_active       bool    True while MAX_DRAWDOWN halt is in force
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trading.logger import get_logger

logger = get_logger(__name__)

_STATE_FILE = Path("state/risk_state.json")
_BARS_DIR   = Path("state/bars")
_LOOKBACK   = 60   # days of returns used for correlation


class RiskManager:
    """
    Call order in AISignalStrategy.run():

        risk = RiskManager(...)
        risk.update_daily_reference(equity)          # step 0: snapshot

        halted, reason = risk.check_halt(equity)     # step 1: max-drawdown gate
        if halted: skip entries

        # per-pick:
        allowed, reason = risk.check_can_enter(ticker, equity, existing_tickers)
        dollars = risk.cap_position_dollars(dollars, equity)
    """

    def __init__(
        self,
        daily_loss_limit:       float = 0.02,   # 2%  — pause day's entries
        max_drawdown:           float = 0.10,   # 10% — emergency halt
        max_position_pct:       float = 0.30,   # 30% — concentration cap
        correlation_threshold:  float = 0.85,   # 85% — block correlated entry
    ) -> None:
        self.daily_loss_limit      = daily_loss_limit
        self.max_drawdown          = max_drawdown
        self.max_position_pct      = max_position_pct
        self.correlation_threshold = correlation_threshold
        self._state = self._load_state()

    # ── Public API ────────────────────────────────────────────────────────────

    def update_daily_reference(self, equity: float) -> None:
        """
        Call once at the start of each cycle.
        If it's a new trading day, refresh the day-open snapshot and update
        the all-time peak equity.
        """
        today = date.today().isoformat()
        if self._state.get("day_open_date") != today:
            self._state["day_open_date"]  = today
            self._state["day_open_equity"] = equity
            logger.info(f"RiskManager: new day snapshot — open equity ${equity:,.2f}")

        # Always update peak (monotonically increasing)
        if equity > self._state.get("peak_equity", 0):
            self._state["peak_equity"] = equity
            logger.debug(f"RiskManager: new equity peak ${equity:,.2f}")

        self._save_state()

    def check_halt(self, equity: float) -> tuple[bool, str]:
        """
        Check MAX_DRAWDOWN rule.
        Returns (halted: bool, reason: str).
        """
        peak = self._state.get("peak_equity", equity)
        if peak <= 0:
            return False, ""

        drawdown = (peak - equity) / peak
        threshold = self.max_drawdown

        if drawdown >= threshold:
            if not self._state.get("halt_active"):
                # First time crossing the threshold — alert
                self._state["halt_active"] = True
                self._save_state()
                self._send_halt_alert(drawdown, equity, peak)
                logger.warning(
                    f"RiskManager: MAX_DRAWDOWN HALT — "
                    f"drawdown {drawdown:.1%} >= {threshold:.0%} | "
                    f"equity ${equity:,.2f} peak ${peak:,.2f}"
                )
            return True, (
                f"MAX_DRAWDOWN: portfolio down {drawdown:.1%} from peak "
                f"(limit {threshold:.0%})"
            )

        # Drawdown recovered — clear halt
        if self._state.get("halt_active"):
            self._state["halt_active"] = False
            self._save_state()
            logger.info(f"RiskManager: drawdown recovered to {drawdown:.1%} — halt cleared")

        return False, ""

    def check_daily_loss(self, equity: float) -> tuple[bool, str]:
        """
        Check DAILY_LOSS_LIMIT rule.
        Returns (blocked: bool, reason: str).
        """
        day_open = self._state.get("day_open_equity", equity)
        if day_open <= 0:
            return False, ""

        day_loss = (day_open - equity) / day_open
        if day_loss >= self.daily_loss_limit:
            reason = (
                f"DAILY_LOSS: down {day_loss:.1%} today "
                f"(limit {self.daily_loss_limit:.0%}) — "
                f"no new entries until tomorrow"
            )
            logger.warning(f"RiskManager: {reason}")
            return True, reason

        return False, ""

    def check_can_enter(
        self,
        ticker:           str,
        equity:           float,
        existing_tickers: list[str],
    ) -> tuple[bool, str]:
        """
        Run all entry-level checks for a new ticker.
        Returns (allowed: bool, reason: str).

        Checks (in order):
          1. Daily loss limit
          2. Correlation with existing positions
        """
        # 1. Daily loss limit
        blocked, reason = self.check_daily_loss(equity)
        if blocked:
            return False, reason

        # 2. Correlation check
        if existing_tickers:
            corr = self._max_correlation(ticker, existing_tickers)
            if corr >= self.correlation_threshold:
                reason = (
                    f"CORRELATION: {ticker} is {corr:.0%} correlated with "
                    f"existing positions (limit {self.correlation_threshold:.0%})"
                )
                logger.info(f"RiskManager: blocked — {reason}")
                return False, reason

        return True, ""

    def cap_position_dollars(self, dollars: float, equity: float) -> float:
        """
        Apply CONCENTRATION rule: cap dollars so position <= max_position_pct
        of total portfolio equity. Returns (possibly reduced) dollars.
        """
        if equity <= 0:
            return dollars
        cap = equity * self.max_position_pct
        if dollars > cap:
            logger.info(
                f"RiskManager: concentration cap — ${dollars:.0f} → ${cap:.0f} "
                f"({self.max_position_pct:.0%} of ${equity:,.0f})"
            )
            return cap
        return dollars

    def get_status(self, equity: float) -> dict:
        """
        Return a full risk-status dict for the dashboard.

        Keys:
            drawdown_pct    float   current drawdown from peak (0–1)
            peak_equity     float   all-time high equity
            day_loss_pct    float   intraday loss from market open (0–1)
            day_open_equity float   equity at market open today
            halt_active     bool    MAX_DRAWDOWN halt in force
            daily_blocked   bool    DAILY_LOSS_LIMIT in force today
            status          str     "GREEN" | "YELLOW" | "RED"
        """
        peak     = self._state.get("peak_equity", equity) or equity
        day_open = self._state.get("day_open_equity", equity) or equity

        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        day_loss = (day_open - equity) / day_open if day_open > 0 else 0.0

        halt_active   = bool(self._state.get("halt_active"))
        daily_blocked = day_loss >= self.daily_loss_limit

        if halt_active or drawdown >= self.max_drawdown:
            status = "RED"
        elif daily_blocked or drawdown >= self.max_drawdown * 0.5:
            status = "YELLOW"
        else:
            status = "GREEN"

        return {
            "drawdown_pct":    round(drawdown, 4),
            "peak_equity":     round(peak, 2),
            "day_loss_pct":    round(max(day_loss, 0.0), 4),
            "day_open_equity": round(day_open, 2),
            "halt_active":     halt_active,
            "daily_blocked":   daily_blocked,
            "status":          status,
            # Limits (for display)
            "max_drawdown":       self.max_drawdown,
            "daily_loss_limit":   self.daily_loss_limit,
            "max_position_pct":   self.max_position_pct,
            "corr_threshold":     self.correlation_threshold,
        }

    # ── Correlation helper ────────────────────────────────────────────────────

    def _max_correlation(
        self,
        new_ticker:       str,
        existing_tickers: list[str],
    ) -> float:
        """
        Return the maximum absolute Pearson correlation between *new_ticker*'s
        60-day return series and any ticker in *existing_tickers*.
        Returns 0.0 if data is unavailable (fail-open).
        """
        new_ret = self._load_returns(new_ticker)
        if new_ret is None or len(new_ret) < 20:
            return 0.0

        max_corr = 0.0
        for ticker in existing_tickers:
            existing_ret = self._load_returns(ticker)
            if existing_ret is None:
                continue
            aligned = pd.concat(
                [new_ret.rename("a"), existing_ret.rename("b")], axis=1
            ).dropna()
            if len(aligned) < 20:
                continue
            c = float(aligned["a"].corr(aligned["b"]))
            max_corr = max(max_corr, abs(c))
            logger.debug(f"RiskManager: corr({new_ticker}, {ticker}) = {c:.2f}")

        return max_corr

    @staticmethod
    def _load_returns(ticker: str) -> pd.Series | None:
        path = _BARS_DIR / f"{ticker}.parquet"
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            return df["close"].pct_change().dropna().iloc[-_LOOKBACK:]
        except Exception:
            return None

    # ── Telegram alert ────────────────────────────────────────────────────────

    def _send_halt_alert(self, drawdown: float, equity: float, peak: float) -> None:
        try:
            from trading.alerts.telegram import TelegramBot
            TelegramBot().send_error_alert(
                component="RiskManager",
                error=(
                    f"MAX DRAWDOWN HALT TRIGGERED\n"
                    f"Drawdown: {drawdown:.1%} (limit {self.max_drawdown:.0%})\n"
                    f"Equity: ${equity:,.2f}  Peak: ${peak:,.2f}\n"
                    f"New entries blocked until drawdown recovers."
                ),
            )
        except Exception as exc:
            logger.warning(f"RiskManager: could not send halt alert — {exc}")

    # ── State I/O ─────────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if _STATE_FILE.exists():
            try:
                return json.loads(_STATE_FILE.read_text())
            except Exception:
                pass
        return {
            "peak_equity":     0.0,
            "day_open_equity": 0.0,
            "day_open_date":   None,
            "halt_active":     False,
        }

    def _save_state(self) -> None:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(self._state, indent=2))
