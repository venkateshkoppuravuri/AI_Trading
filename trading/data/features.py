"""
trading/data/features.py
────────────────────────
Feature engineering pipeline.

Inputs  : daily OHLCV DataFrame from HistoricalData
Outputs : feature DataFrame — one row per trading day, ready for LightGBM

Technical features (price/volume only):
  ret_1d, ret_5d, ret_20d, ret_60d   momentum
  rsi_14                              Relative Strength Index
  macd_hist                           MACD histogram (normalised by price)
  bb_pct                              Bollinger %B  (0 = lower band, 1 = upper)
  atr_pct                             Average True Range as % of price
  vol_ratio_20                        Today volume / 20-day avg volume
  vol_trend_5                         5-day volume % change
  gap_pct                             Overnight gap (open vs prior close)
  high_52w_pct                        % below 52-week high  (≤ 0)
  low_52w_pct                         % above 52-week low   (≥ 0)

External signals (latest snapshot, constant across all rows for a given ticker):
  insider_score    InsiderTradesScraper  0–100
  sentiment_score  NewsSentiment (Finnhub)  -1 to +1
  analyst_upside   Analyst target vs current price  (fraction, e.g. 0.15 = +15%)
  macro_regime     MacroData encoded: BULL=1 NEUTRAL=0 BEAR=-1 HIGH_FEAR=-2

Label (for LightGBM training — NaN on the last 5 rows):
  ret_5d_fwd       5-day forward return
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from trading.logger import get_logger

if TYPE_CHECKING:
    from trading.signals.insider_trades import InsiderTradesScraper
    from trading.signals.macro import MacroData
    from trading.signals.news_sentiment import NewsSentiment

logger = get_logger(__name__)

_FEATURE_DIR = Path("state/features")

# Minimum rows needed to compute 52-week features (252 trading days)
_MIN_ROWS = 60


class FeatureEngine:
    """
    Compute features for one ticker given its OHLCV history.

    All external signal classes are optional — pass ``None`` and the engine
    falls back to neutral defaults so the pipeline never crashes.

    Usage::

        engine = FeatureEngine(insider_scraper, news_sentiment, macro_data)
        df_feat = engine.compute(ohlcv_df, "AAPL", current_price=172.5)
        engine.compute_and_save(ohlcv_df, "AAPL")   # also writes to parquet
    """

    def __init__(
        self,
        insider_scraper: "InsiderTradesScraper | None" = None,
        news_sentiment:  "NewsSentiment | None"        = None,
        macro_data:      "MacroData | None"            = None,
    ) -> None:
        self._insider = insider_scraper
        self._news    = news_sentiment
        self._macro   = macro_data
        _FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public ────────────────────────────────────────────────────────────────

    def compute(
        self,
        df: pd.DataFrame,
        ticker: str,
        current_price: float | None = None,
    ) -> pd.DataFrame:
        """
        Compute all features for *ticker*.

        Parameters
        ----------
        df            : OHLCV DataFrame indexed by date (from HistoricalData).
        ticker        : Stock symbol — used for external signal lookups.
        current_price : Override latest close when computing analyst upside.
                        Defaults to the last close in *df*.

        Returns
        -------
        Feature DataFrame with the same date index.
        Rows with NaN in core features are dropped.
        """
        if df.empty or len(df) < _MIN_ROWS:
            logger.debug(f"{ticker}: too few rows ({len(df)}) — skipping")
            return pd.DataFrame()

        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        volume = df["volume"].astype(float)

        f = pd.DataFrame(index=df.index)

        # ── Momentum ──────────────────────────────────────────────────────────
        f["ret_1d"]  = close.pct_change(1)
        f["ret_5d"]  = close.pct_change(5)
        f["ret_20d"] = close.pct_change(20)
        f["ret_60d"] = close.pct_change(60)

        # ── RSI 14 ────────────────────────────────────────────────────────────
        f["rsi_14"] = _rsi(close, 14)

        # ── MACD histogram (normalised by price to be scale-invariant) ────────
        ema12      = close.ewm(span=12, adjust=False).mean()
        ema26      = close.ewm(span=26, adjust=False).mean()
        macd_line  = ema12 - ema26
        sig_line   = macd_line.ewm(span=9, adjust=False).mean()
        f["macd_hist"] = (macd_line - sig_line) / close.replace(0, np.nan)

        # ── Bollinger %B ─────────────────────────────────────────────────────
        sma20  = close.rolling(20).mean()
        std20  = close.rolling(20).std(ddof=0)
        upper  = sma20 + 2 * std20
        lower  = sma20 - 2 * std20
        band_w = (upper - lower).replace(0, np.nan)
        f["bb_pct"] = (close - lower) / band_w

        # ── ATR % ─────────────────────────────────────────────────────────────
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        f["atr_pct"] = tr.rolling(14).mean() / close.replace(0, np.nan)

        # ── Volume ────────────────────────────────────────────────────────────
        vol_ma20         = volume.rolling(20).mean().replace(0, np.nan)
        f["vol_ratio_20"] = volume / vol_ma20
        f["vol_trend_5"]  = volume.pct_change(5)

        # ── Overnight gap ─────────────────────────────────────────────────────
        open_px         = df["open"].astype(float)
        f["gap_pct"]     = (open_px - prev_close) / prev_close.replace(0, np.nan)

        # ── 52-week high / low distance ───────────────────────────────────────
        high_52w          = close.rolling(252, min_periods=60).max()
        low_52w           = close.rolling(252, min_periods=60).min()
        f["high_52w_pct"] = (close - high_52w) / high_52w.replace(0, np.nan)
        f["low_52w_pct"]  = (close - low_52w)  / low_52w.replace(0, np.nan)

        # ── Forward return label ──────────────────────────────────────────────
        f["ret_5d_fwd"] = close.pct_change(5).shift(-5)

        # ── External signals (point-in-time snapshot, constant per ticker) ────
        cp = current_price or float(close.iloc[-1])
        f["insider_score"]   = self._get_insider_score(ticker)
        f["sentiment_score"] = self._get_sentiment(ticker, cp)
        f["analyst_upside"]  = self._get_analyst_upside(ticker, cp)
        f["macro_regime"]    = self._get_macro_regime()

        f["ticker"] = ticker

        # Drop rows missing any core technical feature
        core_cols = ["ret_1d", "ret_5d", "rsi_14", "macd_hist", "bb_pct", "atr_pct"]
        return f.dropna(subset=core_cols)

    def compute_and_save(
        self,
        df: pd.DataFrame,
        ticker: str,
        current_price: float | None = None,
    ) -> pd.DataFrame:
        """Compute features and write to ``state/features/{ticker}.parquet``."""
        features = self.compute(df, ticker, current_price)
        if not features.empty:
            path = _FEATURE_DIR / f"{ticker}.parquet"
            features.to_parquet(path)
            logger.debug(f"{ticker}: {len(features)} feature rows saved → {path.name}")
        return features

    # ── Signal helpers ────────────────────────────────────────────────────────

    def _get_insider_score(self, ticker: str) -> float:
        if self._insider is None:
            return 50.0
        try:
            return float(self._insider.get_insider_score(ticker))
        except Exception:
            return 50.0

    def _get_sentiment(self, ticker: str, price: float) -> float:
        if self._news is None:
            return 0.0
        try:
            sig = self._news.get_full_signal(ticker, price)
            return float(sig.get("sentiment_score", 0.0))
        except Exception:
            return 0.0

    def _get_analyst_upside(self, ticker: str, price: float) -> float:
        if self._news is None:
            return 0.0
        try:
            sig = self._news.get_full_signal(ticker, price)
            upside_pct = float(sig.get("upside_pct", 0.0))
            return upside_pct / 100.0   # store as fraction: 0.15 = +15%
        except Exception:
            return 0.0

    def _get_macro_regime(self) -> float:
        if self._macro is None:
            return 0.0
        try:
            regime = self._macro.get_market_regime()
            return {"BULL": 1.0, "NEUTRAL": 0.0, "BEAR": -1.0, "HIGH_FEAR": -2.0}.get(
                regime, 0.0
            )
        except Exception:
            return 0.0


# ── Pure technical helpers ────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))
