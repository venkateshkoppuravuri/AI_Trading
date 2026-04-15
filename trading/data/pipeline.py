"""
trading/data/pipeline.py
────────────────────────
Orchestrates the full data → feature pipeline.

Usage (CLI)::

    python -m trading.data.pipeline              # quick mode  (~2 min,  watchlist only)
    python -m trading.data.pipeline --mode top100  # top 100 S&P 500 (~15 min)
    python -m trading.data.pipeline --mode full    # all S&P 500     (~45 min)
    python -m trading.data.pipeline --mode quick --no-signals  # skip Finnhub/FRED

Usage (import)::

    from trading.data.pipeline import FeaturePipeline
    pipeline = FeaturePipeline()
    pipeline.run(mode="quick")          # builds feature store
    df = pipeline.load_all_features()   # load combined feature parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from trading.data.features import FeatureEngine
from trading.data.historical import HistoricalData
from trading.data.universe import get_sp500_tickers, get_watchlist_tickers
from trading.logger import get_logger

logger = get_logger(__name__)

_COMBINED_PATH = Path("state/features/all_features.parquet")


class FeaturePipeline:
    """
    End-to-end pipeline: universe → bars → features → parquet store.

    Parameters
    ----------
    use_signals : bool
        Whether to call Finnhub / FRED / EDGAR for external signals.
        Disable for fast back-tests that only need technical features.
    days : int
        How many calendar days of history to fetch (default 504 ≈ 2 years).
    """

    def __init__(self, use_signals: bool = True, days: int = 504) -> None:
        self._historical = HistoricalData()
        self._days       = days

        if use_signals:
            insider, news, macro = self._init_signals()
        else:
            insider = news = macro = None

        self._engine = FeatureEngine(
            insider_scraper=insider,
            news_sentiment=news,
            macro_data=macro,
        )

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self, mode: str = "quick") -> dict[str, int]:
        """
        Build the feature store for *mode* universe.

        Modes:
          quick   — politician watchlist tickers (~15–25 tickers, ~2 min)
          top100  — top 100 S&P 500 by market cap (~15 min)
          full    — all S&P 500 (~45 min, ~500 API calls)

        Returns a dict with run statistics.
        """
        tickers = self._get_tickers(mode)
        logger.info(f"Pipeline [{mode}]: {len(tickers)} tickers | {self._days} days history")

        # ── Fetch OHLCV bars ──────────────────────────────────────────────────
        t0   = time.time()
        bars = self._historical.get_bulk_bars(tickers, self._days)
        logger.info(f"Bars fetched: {len(bars)}/{len(tickers)} in {time.time()-t0:.1f}s")

        # ── Compute features ──────────────────────────────────────────────────
        all_frames: list[pd.DataFrame] = []
        ok = errors = skipped = 0

        for ticker in tickers:
            df = bars.get(ticker, pd.DataFrame())
            if df.empty:
                skipped += 1
                continue
            try:
                feat = self._engine.compute_and_save(df, ticker)
                if not feat.empty:
                    all_frames.append(feat)
                    ok += 1
                else:
                    skipped += 1
            except Exception as exc:
                logger.warning(f"{ticker}: feature error — {exc}")
                errors += 1

        # ── Save combined parquet ─────────────────────────────────────────────
        if all_frames:
            combined = pd.concat(all_frames, ignore_index=False)
            _COMBINED_PATH.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(_COMBINED_PATH)
            rows = len(combined)
            logger.info(
                f"Feature store: {rows:,} rows × {combined.shape[1]} cols → "
                f"{_COMBINED_PATH}"
            )
        else:
            rows = 0
            logger.warning("No features were computed — check API keys and connectivity")

        elapsed = time.time() - t0
        stats = {
            "mode":    mode,
            "tickers": len(tickers),
            "ok":      ok,
            "skipped": skipped,
            "errors":  errors,
            "rows":    rows,
            "elapsed": round(elapsed, 1),
        }
        logger.info(
            f"Pipeline done in {elapsed:.1f}s — "
            f"{ok} ok | {skipped} skipped | {errors} errors"
        )
        return stats

    def load_all_features(self) -> pd.DataFrame:
        """Load the combined feature store. Returns empty DataFrame if not built yet."""
        if not _COMBINED_PATH.exists():
            logger.warning(
                "Feature store not found. Run: python -m trading.data.pipeline"
            )
            return pd.DataFrame()
        return pd.read_parquet(_COMBINED_PATH)

    def get_latest_snapshot(self) -> pd.DataFrame:
        """
        Return one row per ticker — the most recent feature vector for each stock.
        Useful for model scoring: 'which stocks look best right now?'
        """
        df = self.load_all_features()
        if df.empty:
            return df
        df = df.reset_index()
        # Keep the most recent date for each ticker
        return (
            df.sort_values("date")
            .groupby("ticker")
            .last()
            .reset_index()
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _get_tickers(mode: str) -> list[str]:
        if mode == "quick":
            tickers = get_watchlist_tickers()
            if not tickers:
                logger.warning("Watchlist empty — using top-20 fallback")
                from trading.data.universe import FALLBACK_TICKERS
                tickers = FALLBACK_TICKERS[:20]
            return tickers
        elif mode == "top100":
            sp500 = get_sp500_tickers()
            return sp500[:100]
        elif mode == "full":
            return get_sp500_tickers()
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use: quick | top100 | full")

    @staticmethod
    def _init_signals():
        insider = news = macro = None
        try:
            from trading.signals.insider_trades import InsiderTradesScraper
            insider = InsiderTradesScraper()
        except Exception as exc:
            logger.warning(f"InsiderScraper unavailable: {exc}")
        try:
            from trading.signals.news_sentiment import NewsSentiment
            news = NewsSentiment()
        except Exception as exc:
            logger.warning(f"NewsSentiment unavailable: {exc}")
        try:
            from trading.signals.macro import MacroData
            macro = MacroData()
        except Exception as exc:
            logger.warning(f"MacroData unavailable: {exc}")
        return insider, news, macro


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the AI Trading feature store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  quick    Politician watchlist tickers only  (~2 min,  recommended for first run)
  top100   Top 100 S&P 500 by market cap     (~15 min)
  full     All ~500 S&P 500 stocks           (~45 min)

Examples:
  python -m trading.data.pipeline
  python -m trading.data.pipeline --mode top100
  python -m trading.data.pipeline --mode full --no-signals
        """,
    )
    p.add_argument(
        "--mode",
        choices=["quick", "top100", "full"],
        default="quick",
        help="Universe size (default: quick)",
    )
    p.add_argument(
        "--no-signals",
        action="store_true",
        dest="no_signals",
        help="Skip external signals (Finnhub / FRED / EDGAR) — technical features only",
    )
    p.add_argument(
        "--days",
        type=int,
        default=504,
        help="Days of history to fetch (default: 504 ≈ 2 years)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args     = _parse_args()
    pipeline = FeaturePipeline(use_signals=not args.no_signals, days=args.days)
    stats    = pipeline.run(mode=args.mode)

    print(f"\nDone in {stats['elapsed']}s")
    print(f"  Tickers  : {stats['tickers']}")
    print(f"  OK       : {stats['ok']}")
    print(f"  Skipped  : {stats['skipped']}")
    print(f"  Errors   : {stats['errors']}")
    print(f"  Rows     : {stats['rows']:,}")
    print(f"\nFeature store: state/features/all_features.parquet")
