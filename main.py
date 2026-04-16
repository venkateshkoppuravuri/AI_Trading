"""
main.py — single entry point for the AI Trading Bot.

Usage:
  python main.py                         # run all 3 strategies on schedule
  python main.py --strategy trailing     # run trailing stop only
  python main.py --strategy copy         # run copy trading only
  python main.py --strategy wheel        # run wheel only
  python main.py --once                  # run one cycle and exit (good for cron)

Schedules:
  Trailing Stop  : every 5 minutes   | market hours only
  Copy Trading   : daily at 9:35am   | Mon-Fri
  Wheel Strategy : every 15 minutes  | market hours only

Parameters:
  Edit params.yaml in the project root to tune strategies without
  touching Python code. CLI flags always take precedence over params.yaml.
"""

import argparse
import signal
import threading
import time
from pathlib import Path

import schedule

from trading.config import get_settings
from trading.logger import get_logger
from trading.market import is_market_hours
from trading.strategies import (
    AISignalStrategy,
    CopyTradingStrategy,
    TrailingStopStrategy,
    WheelStrategy,
)
from trading.strategies.base import BaseStrategy

# ── Week 1: Alerts & Signals (lazy-imported so missing keys don't crash) ──────
def _get_telegram():
    """Return TelegramBot instance for scheduler-level alerts (silent init)."""
    try:
        import os
        from trading.alerts.telegram import TelegramBot
        # Suppress duplicate "initialised" log — strategy already logs its own
        import logging
        tg_logger = logging.getLogger("trading.alerts.telegram")
        prev_level = tg_logger.level
        tg_logger.setLevel(logging.WARNING)
        bot = TelegramBot()
        tg_logger.setLevel(prev_level)
        return bot
    except Exception as exc:
        logger.warning(f"Telegram unavailable: {exc}")
        return None

def _get_macro():
    """Return MacroData instance (or None if import fails)."""
    try:
        from trading.signals.macro import MacroData
        return MacroData()
    except Exception as exc:
        logger.warning(f"MacroData unavailable: {exc}")
        return None

logger = get_logger(__name__)

# ── Hard-coded fallback defaults (overridden by params.yaml) ──────────────────

_TRAILING_DEFAULTS: dict = dict(symbol="TSLA", initial_shares=10)
_COPY_DEFAULTS:     dict = dict(trade_budget=1_000.0, max_positions=10, top_n=3)
_WHEEL_DEFAULTS:    dict = dict(symbol="TSLA", contracts=1)
_AI_DEFAULTS:       dict = dict(
    # dynamic sizing — overridden each cycle from live equity
    budget_pct=0.90, min_position_dollars=2_000.0, max_positions_cap=20,
    budget=0.0, max_positions=0,          # 0 = use dynamic values above
    # exit rules
    profit_target=0.05, stop_loss=0.03, max_holding_days=10,
    pipeline_mode="top100", min_confidence="MED",
    # risk
    daily_loss_limit=0.02, max_drawdown=0.10,
    max_position_pct=0.30, correlation_threshold=0.85,
)


# ── params.yaml loader ────────────────────────────────────────────────────────

def _load_params() -> dict:
    """Load params.yaml from the project root. Returns {} if missing or invalid."""
    params_file = Path(__file__).parent / "params.yaml"
    if not params_file.exists():
        return {}
    try:
        import yaml  # pyyaml
        with open(params_file) as f:
            data = yaml.safe_load(f)
        return data or {}
    except Exception as exc:
        logger.warning(f"Could not load params.yaml — using defaults: {exc}")
        return {}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI Trading Bot — Alpaca paper trading strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # all 3 strategies
  python main.py --strategy trailing      # trailing stop only
  python main.py --strategy copy          # copy trading only
  python main.py --strategy wheel         # wheel strategy only
  python main.py --once                   # one cycle then exit
  python main.py --symbol AAPL --shares 5 # custom symbol / shares
        """,
    )
    p.add_argument(
        "--strategy",
        choices=["all", "trailing", "copy", "wheel", "ai"],
        default="all",
        help="Which strategy to run (default: all)",
    )
    p.add_argument("--symbol",        default=None,  help="Override stock symbol")
    p.add_argument("--shares",        type=int,   default=None, help="Override initial shares (trailing)")
    p.add_argument("--contracts",     type=int,   default=None, help="Override contracts count (wheel)")
    p.add_argument("--budget",        type=float, default=None, help="Override trade budget (copy)")
    p.add_argument("--max-positions", type=int,   default=None, dest="max_positions",
                   help="Override max positions (copy)")
    p.add_argument("--once", action="store_true",
                   help="Run one cycle and exit (useful for cron jobs)")
    return p.parse_args()


# ── Strategy factory ──────────────────────────────────────────────────────────

def _build_strategies(args: argparse.Namespace) -> list[BaseStrategy]:
    params = _load_params()
    which  = args.strategy
    strats: list[BaseStrategy] = []

    # ── Trailing Stop ─────────────────────────────────────────────────────────
    if which in ("all", "trailing"):
        kwargs = {**_TRAILING_DEFAULTS, **params.get("trailing_stop", {})}
        if args.symbol: kwargs["symbol"]         = args.symbol
        if args.shares: kwargs["initial_shares"] = args.shares
        # ladder_levels comes from YAML as [[20.0, 10], [30.0, 20]]; convert to list of tuples
        if "ladder_levels" in kwargs:
            kwargs["ladder_levels"] = [tuple(lv) for lv in kwargs["ladder_levels"]]
        strats.append(TrailingStopStrategy(**kwargs))

    # ── Copy Trading ──────────────────────────────────────────────────────────
    if which in ("all", "copy"):
        kwargs = {**_COPY_DEFAULTS, **params.get("copy_trading", {})}
        if args.budget:        kwargs["trade_budget"]  = args.budget
        if args.max_positions: kwargs["max_positions"] = args.max_positions
        strats.append(CopyTradingStrategy(**kwargs))

    # ── Wheel ─────────────────────────────────────────────────────────────────
    if which in ("all", "wheel"):
        kwargs = {**_WHEEL_DEFAULTS, **params.get("wheel", {})}
        if args.symbol:    kwargs["symbol"]    = args.symbol
        if args.contracts: kwargs["contracts"] = args.contracts
        strats.append(WheelStrategy(**kwargs))

    # ── AI Signal ─────────────────────────────────────────────────────────────
    if which in ("all", "ai"):
        kwargs = {**_AI_DEFAULTS, **params.get("ai_signal", {}), **params.get("risk", {})}
        if args.budget:        kwargs["budget"]        = args.budget
        if args.max_positions: kwargs["max_positions"] = args.max_positions
        strats.append(AISignalStrategy(**kwargs))

    return strats


# ── Job wrappers ──────────────────────────────────────────────────────────────

def _make_job(strategy: BaseStrategy, telegram=None):
    """Return a scheduler-safe callable that runs one strategy cycle."""
    def job() -> None:
        try:
            strategy.run()
            logger.info(f"{strategy.name}: {strategy.status()}")
        except Exception as exc:
            logger.error(f"{strategy.name}: Unhandled error — {exc}", exc_info=True)
            if telegram:
                try:
                    telegram.send_error_alert(strategy.name, str(exc))
                except Exception:
                    pass
    return job


def _make_market_job(strategy: BaseStrategy, telegram=None):
    """Wrap _make_job with a market-hours guard — skips when market is closed."""
    inner = _make_job(strategy, telegram)
    def guarded() -> None:
        if not is_market_hours():
            return
        inner()
    return guarded


# ── Scheduler setup ───────────────────────────────────────────────────────────

def _setup_schedule(strategies: list[BaseStrategy], telegram=None) -> None:
    for s in strategies:
        if isinstance(s, TrailingStopStrategy):
            schedule.every(5).minutes.do(_make_market_job(s, telegram))
            logger.info(f"Scheduled {s.name} every 5 min (market hours)")

        elif isinstance(s, CopyTradingStrategy):
            for day in ("monday", "tuesday", "wednesday", "thursday", "friday"):
                getattr(schedule.every(), day).at("09:35").do(_make_job(s, telegram))
            logger.info(f"Scheduled {s.name} daily at 09:35 ET (Mon-Fri)")

        elif isinstance(s, WheelStrategy):
            schedule.every(15).minutes.do(_make_market_job(s, telegram))
            logger.info(f"Scheduled {s.name} every 15 min (market hours)")

        elif isinstance(s, AISignalStrategy):
            for day in ("monday", "tuesday", "wednesday", "thursday", "friday"):
                getattr(schedule.every(), day).at("09:30").do(_make_job(s, telegram))
            logger.info(f"Scheduled {s.name} daily at 09:30 ET (Mon-Fri)")


# ── Graceful shutdown ─────────────────────────────────────────────────────────

_stop_event = threading.Event()


def _shutdown(signum, frame) -> None:  # noqa: ARG001
    logger.info(f"Signal {signum} received — shutting down gracefully...")
    _stop_event.set()


# ── Main ──────────────────────────────────────────────────────────────────────

def _send_daily_summary(telegram, client) -> None:
    """Fetch portfolio data from Alpaca and send daily P&L summary to Telegram."""
    try:
        account   = client.get_account()
        positions = client.get_positions()
        equity    = float(account.get("equity", 0))
        prev_eq   = float(account.get("last_equity", equity))
        day_pnl   = equity - prev_eq
        day_pct   = (day_pnl / prev_eq * 100) if prev_eq else 0.0

        # Find top mover by unrealized P&L
        top_mover = None
        if positions:
            best = max(positions, key=lambda p: float(p.get("unrealized_pl", 0)))
            if float(best.get("unrealized_pl", 0)) > 0:
                top_mover = best.get("symbol")

        telegram.send_daily_summary(
            portfolio_value=equity,
            day_pnl=day_pnl,
            day_pnl_pct=day_pct,
            positions=positions,
            top_mover=top_mover,
        )
        logger.info("Daily summary sent to Telegram.")
    except Exception as exc:
        logger.warning(f"Daily summary failed: {exc}")


def _send_weekly_report(telegram) -> None:
    """Send weekly trade journal report to Telegram."""
    try:
        from trading.journal import TradeJournal
        report = TradeJournal().format_weekly_report()
        if telegram:
            telegram._send_message(report)
        logger.info("Weekly journal report sent.")
    except Exception as exc:
        logger.warning(f"Weekly report failed: {exc}")


def main() -> None:
    args     = _parse_args()
    settings = get_settings()   # fail fast if .env is missing
    strats   = _build_strategies(args)

    if not strats:
        logger.error("No strategies selected — exiting.")
        return

    # ── Macro check (non-blocking — failures are warnings) ────────────────────
    telegram = _get_telegram()   # used for scheduler-level alerts only
    macro    = _get_macro()

    if macro:
        try:
            regime = macro.get_market_regime()
            multiplier = macro.get_position_size_multiplier()
            logger.info(f"Macro regime: {regime}  |  Size multiplier: {multiplier}x")
        except Exception as exc:
            logger.warning(f"Macro check failed: {exc}")

    logger.info("=" * 56)
    logger.info("  AI Trading Bot starting")
    logger.info(f"  Account : {settings.base_url}")
    for s in strats:
        logger.info(f"  Strategy: {s.name}")
    logger.info("=" * 56)

    if telegram:
        try:
            telegram.send_bot_started()
        except Exception:
            pass

    # ── One-shot mode (--once) ────────────────────────────────────────────────
    if args.once:
        if not is_market_hours():
            from datetime import datetime
            from zoneinfo import ZoneInfo
            et = datetime.now(ZoneInfo("America/New_York"))
            logger.warning(
                f"Market is CLOSED right now ({et.strftime('%H:%M ET %a')}).\n"
                f"  Orders placed now are queued as day orders and fill at next open (9:30 AM ET).\n"
                f"  Run again during market hours (9:30 AM – 4:00 PM ET Mon–Fri) for live fills."
            )
        logger.info("--once mode: running one cycle per strategy then exiting")
        for s in strats:
            try:
                s.run()
                logger.info(f"{s.name}: {s.status()}")
            except Exception as exc:
                logger.error(f"{s.name}: {exc}", exc_info=True)
        return

    # ── Continuous scheduler mode ─────────────────────────────────────────────
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    _setup_schedule(strats, telegram)

    # Daily P&L summary at 13:30 UTC (= 1:30 AM IST = ~30 min after US market close)
    try:
        from trading.client import AlpacaClient
        _client = AlpacaClient()
        schedule.every().day.at("13:30").do(_send_daily_summary, telegram, _client)
        logger.info("Daily P&L summary scheduled at 13:30 UTC (1:30 AM IST)")
    except Exception as exc:
        logger.warning(f"Could not schedule daily summary: {exc}")

    # Weekly journal report every Friday at 13:45 UTC (after market close)
    try:
        schedule.every().friday.at("13:45").do(_send_weekly_report, telegram)
        logger.info("Weekly journal report scheduled every Friday at 13:45 UTC")
    except Exception as exc:
        logger.warning(f"Could not schedule weekly report: {exc}")

    logger.info("Running initial cycle for all strategies...")
    for s in strats:
        _make_job(s, telegram)()

    def _scheduler_loop() -> None:
        while not _stop_event.is_set():
            schedule.run_pending()
            time.sleep(15)

    t = threading.Thread(target=_scheduler_loop, daemon=True, name="scheduler")
    t.start()

    logger.info("All strategies running. Press Ctrl+C or send SIGTERM to stop.\n")
    _stop_event.wait()

    # ── Shutdown ──────────────────────────────────────────────────────────────
    if telegram:
        try:
            telegram.send_bot_stopped()
        except Exception:
            pass
    logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
