"""
backtest.py
────────────
CLI entry point for the full-pipeline backtesting engine.

Usage:
    python backtest.py                        # test window from feature store
    python backtest.py --start 2025-01-01     # custom start date
    python backtest.py --end   2025-12-31     # custom end date
    python backtest.py --capital 5000         # different starting capital
    python backtest.py --positions 3          # fewer max positions
    python backtest.py --confidence HIGH      # only HIGH-confidence picks
    python backtest.py --save                 # save CSV + HTML report
    python backtest.py --save --start 2025-01-01 --capital 5000

The backtest uses:
  • Already-trained LightGBM model  (state/models/lgbm_model.pkl)
  • Cached OHLCV bars               (state/bars/*.parquet)
  • Feature store                   (state/features/all_features.parquet)

No live API calls are made. No Claude LLM filtering is applied
(results reflect LightGBM + rule-based exits only).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI Signal full-pipeline backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--start",       default=None,  metavar="YYYY-MM-DD",
                   help="Backtest start date (default: test-window start)")
    p.add_argument("--end",         default=None,  metavar="YYYY-MM-DD",
                   help="Backtest end date (default: latest date in store)")
    p.add_argument("--capital",     type=float, default=2_000.0,
                   help="Starting capital in USD (default: 2000)")
    p.add_argument("--positions",   type=int,   default=5,
                   help="Max simultaneous positions (default: 5)")
    p.add_argument("--stop",        type=float, default=0.03,
                   help="Stop-loss fraction (default: 0.03 = 3%%)")
    p.add_argument("--target",      type=float, default=0.05,
                   help="Profit-target fraction (default: 0.05 = 5%%)")
    p.add_argument("--trail",       type=float, default=0.03,
                   help="Trailing-stop distance from peak (default: 0.03)")
    p.add_argument("--hold",        type=int,   default=10,
                   help="Max holding days (time stop, default: 10)")
    p.add_argument("--confidence",  default="MED", choices=["HIGH", "MED", "LOW"],
                   help="Minimum model confidence (default: MED)")
    p.add_argument("--test-pct",    type=float, default=0.20,
                   help="Fraction of dates used as test window (default: 0.20)")
    p.add_argument("--save",        action="store_true",
                   help="Save equity CSV, trades CSV, metrics JSON, HTML chart")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from trading.backtest.engine import BacktestEngine
    from trading.backtest.result import compute_spy_return

    print(f"\n  AI Signal Strategy — Backtester")
    print(f"  Capital ${args.capital:,.0f} | "
          f"Max {args.positions} positions | "
          f"Stop {args.stop:.0%} | Target {args.target:.0%} | "
          f"Trail {args.trail:.0%} | Hold {args.hold}d | "
          f"Conf ≥ {args.confidence}")
    if args.start or args.end:
        print(f"  Date range: {args.start or 'feature-store start'} → "
              f"{args.end or 'feature-store end'}")
    print()

    engine = BacktestEngine(
        initial_capital  = args.capital,
        max_positions    = args.positions,
        stop_loss        = args.stop,
        profit_target    = args.target,
        trail_pct        = args.trail,
        max_holding_days = args.hold,
        min_confidence   = args.confidence,
        test_pct         = args.test_pct,
        start_date       = args.start,
        end_date         = args.end,
    )

    result = engine.run()

    # ── SPY benchmark ─────────────────────────────────────────────────────────
    spy_return = None
    if result.equity.index.size >= 2:
        spy_return = compute_spy_return(
            result.equity.index[0],
            result.equity.index[-1],
        )
        if spy_return is None:
            print("  (SPY benchmark not available — "
                  "run the pipeline to fetch SPY bars if needed)")

    # ── Print summary ─────────────────────────────────────────────────────────
    result.print_summary(benchmark_return=spy_return)

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.save:
        out_dir = result.save_report()
        print(f"\n  Reports saved to: {out_dir}/")
        print("  Open the .html file in a browser for the interactive chart.")
    else:
        print("\n  Tip: add --save to export equity CSV, trades CSV, "
              "metrics JSON, and an interactive HTML chart.")


if __name__ == "__main__":
    main()
