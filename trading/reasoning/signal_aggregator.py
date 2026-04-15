"""
trading/reasoning/signal_aggregator.py
───────────────────────────────────────
Collects all available signals for a ticker into a single flat dict
that ClaudeAnalyst can reason over.

Sources (all optional — missing sources return neutral defaults):
  • LightGBM rank + prediction + top features
  • SEC EDGAR insider trades score
  • Finnhub news sentiment + analyst consensus + upside + earnings date
  • FRED macro regime + VIX + yield curve
  • Current price from Alpaca

Usage::

    from trading.reasoning.signal_aggregator import SignalAggregator
    agg = SignalAggregator()
    signals = agg.gather("AXON", lgbm_pick={...})
    # {current_price, lgbm_rank, lgbm_pred_pct, insider_score, ...}

    signals_map = agg.gather_many(picks_list)
    # {"AXON": {...}, "CF": {...}, ...}
"""

from __future__ import annotations

import time

from trading.client import AlpacaClient
from trading.logger import get_logger

logger = get_logger(__name__)

_CALL_DELAY = 0.5   # seconds between Finnhub calls (60/min free tier)


class SignalAggregator:
    """
    Gathers all signals for candidate tickers from every available source.
    Gracefully handles missing API keys and network failures.
    """

    def __init__(self) -> None:
        self._client   = AlpacaClient()
        self._insider  = self._init_insider()
        self._news     = self._init_news()
        self._macro    = self._init_macro()
        self._macro_cache: dict | None = None   # fetch once per session

    # ── Public ────────────────────────────────────────────────────────────────

    def gather(self, ticker: str, lgbm_pick: dict | None = None) -> dict:
        """
        Return a flat signal dict for *ticker*.

        lgbm_pick : the dict from LightGBMPredictor.score() for this ticker.
                    If None, LightGBM fields are set to neutral defaults.
        """
        signals: dict = {"ticker": ticker}

        # ── Price ─────────────────────────────────────────────────────────────
        try:
            signals["current_price"] = self._client.get_latest_price(ticker)
        except Exception:
            signals["current_price"] = 0.0

        # ── LightGBM ──────────────────────────────────────────────────────────
        if lgbm_pick:
            signals["lgbm_rank"]         = lgbm_pick.get("rank", "N/A")
            signals["lgbm_pred_pct"]     = lgbm_pick.get("pred_pct", "N/A")
            signals["lgbm_pred_return"]  = lgbm_pick.get("pred_return", 0.0)
            signals["lgbm_confidence"]   = lgbm_pick.get("confidence", "LOW")
            signals["lgbm_top_features"] = lgbm_pick.get("top_features", [])
        else:
            signals.update({
                "lgbm_rank": "N/A", "lgbm_pred_pct": "N/A",
                "lgbm_pred_return": 0.0, "lgbm_confidence": "LOW",
                "lgbm_top_features": [],
            })

        # ── Macro (cached — one FRED call per session) ────────────────────────
        macro = self._get_macro()
        signals["macro_regime"]  = macro.get("regime",      "NEUTRAL")
        signals["vix"]           = macro.get("vix",         "N/A")
        signals["yield_curve"]   = macro.get("yield_curve", "N/A")
        signals["fed_rate"]      = macro.get("fed_rate",    "N/A")

        # ── Insider trades ────────────────────────────────────────────────────
        signals["insider_score"] = self._get_insider_score(ticker)

        # ── News + analyst ────────────────────────────────────────────────────
        news = self._get_news_signals(ticker, signals["current_price"])
        signals.update(news)

        return signals

    def gather_many(
        self,
        picks: list[dict],
        delay: float = _CALL_DELAY,
    ) -> dict[str, dict]:
        """
        Gather signals for all tickers in *picks*.
        Returns {ticker: signals_dict}.
        Respects Finnhub rate limit via *delay* between calls.
        """
        result: dict[str, dict] = {}
        for i, pick in enumerate(picks):
            ticker = pick["ticker"]
            result[ticker] = self.gather(ticker, lgbm_pick=pick)
            if i < len(picks) - 1:
                time.sleep(delay)
        return result

    # ── Signal fetchers ───────────────────────────────────────────────────────

    def _get_macro(self) -> dict:
        if self._macro_cache is not None:
            return self._macro_cache
        try:
            data = self._macro.get_all()
            self._macro_cache = data
            return data
        except Exception as exc:
            logger.debug(f"Macro unavailable: {exc}")
            self._macro_cache = {}
            return {}

    def _get_insider_score(self, ticker: str) -> int:
        if self._insider is None:
            return 50
        try:
            return int(self._insider.get_insider_score(ticker))
        except Exception:
            return 50

    def _get_news_signals(self, ticker: str, price: float) -> dict:
        defaults = {
            "sentiment_score":     0.0,
            "analyst_consensus":   "N/A",
            "analyst_upside_pct":  0.0,
            "days_to_earnings":    "N/A",
        }
        if self._news is None:
            return defaults
        try:
            sig = self._news.get_full_signal(ticker, price)
            return {
                "sentiment_score":    round(float(sig.get("sentiment_score", 0.0)), 3),
                "analyst_consensus":  sig.get("analyst_consensus", "N/A"),
                "analyst_upside_pct": round(float(sig.get("upside_pct", 0.0)), 1),
                "days_to_earnings":   sig.get("days_to_earnings", "N/A"),
            }
        except Exception as exc:
            logger.debug(f"News signals unavailable for {ticker}: {exc}")
            return defaults

    # ── Initialisers ──────────────────────────────────────────────────────────

    @staticmethod
    def _init_insider():
        try:
            from trading.signals.insider_trades import InsiderTradesScraper
            return InsiderTradesScraper()
        except Exception as exc:
            logger.debug(f"InsiderScraper unavailable: {exc}")
            return None

    @staticmethod
    def _init_news():
        try:
            from trading.signals.news_sentiment import NewsSentiment
            return NewsSentiment()
        except Exception as exc:
            logger.debug(f"NewsSentiment unavailable: {exc}")
            return None

    @staticmethod
    def _init_macro():
        try:
            from trading.signals.macro import MacroData
            return MacroData()
        except Exception as exc:
            logger.debug(f"MacroData unavailable: {exc}")
            return None
