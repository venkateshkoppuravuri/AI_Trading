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
    CopyTradingStrategy,
    TrailingStopStrategy,
    WheelStrategy,
)
from trading.strategies.base import BaseStrategy

logger = get_logger(__name__)

# ── Hard-coded fallback defaults (overridden by params.yaml) ──────────────────

_TRAILING_DEFAULTS: dict = dict(symbol="TSLA", initial_shares=10)
_COPY_DEFAULTS:     dict = dict(trade_budget=1_000.0, max_positions=10, top_n=3)
_WHEEL_DEFAULTS:    dict = dict(symbol="TSLA", contracts=1)


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
        choices=["all", "trailing", "copy", "wheel"],
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

    return strats


# ── Job wrappers ──────────────────────────────────────────────────────────────

def _make_job(strategy: BaseStrategy):
    """Return a scheduler-safe callable that runs one strategy cycle."""
    def job() -> None:
        try:
            strategy.run()
            logger.info(f"{strategy.name}: {strategy.status()}")
        except Exception as exc:
            logger.error(f"{strategy.name}: Unhandled error — {exc}", exc_info=True)
    return job


def _make_market_job(strategy: BaseStrategy):
    """Wrap _make_job with a market-hours guard — skips when market is closed."""
    inner = _make_job(strategy)
    def guarded() -> None:
        if not is_market_hours():
            return
        inner()
    return guarded


# ── Scheduler setup ───────────────────────────────────────────────────────────

def _setup_schedule(strategies: list[BaseStrategy]) -> None:
    for s in strategies:
        if isinstance(s, TrailingStopStrategy):
            schedule.every(5).minutes.do(_make_market_job(s))
            logger.info(f"Scheduled {s.name} every 5 min (market hours)")

        elif isinstance(s, CopyTradingStrategy):
            for day in ("monday", "tuesday", "wednesday", "thursday", "friday"):
                getattr(schedule.every(), day).at("09:35").do(_make_job(s))
            logger.info(f"Scheduled {s.name} daily at 09:35 ET (Mon-Fri)")

        elif isinstance(s, WheelStrategy):
            schedule.every(15).minutes.do(_make_market_job(s))
            logger.info(f"Scheduled {s.name} every 15 min (market hours)")


# ── Graceful shutdown ─────────────────────────────────────────────────────────

_stop_event = threading.Event()


def _shutdown(signum, frame) -> None:  # noqa: ARG001
    logger.info(f"Signal {signum} received — shutting down gracefully...")
    _stop_event.set()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args     = _parse_args()
    settings = get_settings()   # fail fast if .env is missing
    strats   = _build_strategies(args)

    if not strats:
        logger.error("No strategies selected — exiting.")
        return

    logger.info("=" * 56)
    logger.info("  AI Trading Bot starting")
    logger.info(f"  Account : {settings.base_url}")
    for s in strats:
        logger.info(f"  Strategy: {s.name}")
    logger.info("=" * 56)

    # ── One-shot mode (--once) ────────────────────────────────────────────────
    if args.once:
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

    _setup_schedule(strats)

    logger.info("Running initial cycle for all strategies...")
    for s in strats:
        _make_job(s)()

    def _scheduler_loop() -> None:
        while not _stop_event.is_set():
            schedule.run_pending()
            time.sleep(15)

    t = threading.Thread(target=_scheduler_loop, daemon=True, name="scheduler")
    t.start()

    logger.info("All strategies running. Press Ctrl+C or send SIGTERM to stop.\n")
    _stop_event.wait()
    logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
