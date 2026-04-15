"""
trading/signals/news_sentiment.py
───────────────────────────────────
News sentiment and analyst ratings from Finnhub (free tier).

Free tier limits:
  • 60 API calls/minute
  • Company news: last 1 year
  • Analyst ratings: current consensus
  • Earnings calendar: next 3 months

Signals:
  • news_sentiment_score  (-1.0 to +1.0) — positive/negative news flow
  • analyst_consensus     — Strong Buy | Buy | Hold | Sell | Strong Sell
  • analyst_target_price  — consensus price target
  • upside_pct            — % upside to analyst target
  • earnings_date         — next earnings (risk event)

Setup (one time, free):
  1. Go to https://finnhub.io → Sign up (free)
  2. Copy your API key
  3. Add to .env: FINNHUB_API_KEY=your_key

Usage:
    news = NewsSentiment()
    score = news.get_sentiment_score("NVDA")
    ratings = news.get_analyst_ratings("NVDA")
"""

import os
import json
import time
from datetime import date, timedelta

import requests

from trading.config import get_settings
from trading.logger import get_logger

logger = get_logger(__name__)

_CACHE_TTL     = 3_600   # 1 hour
_FINNHUB_BASE  = "https://finnhub.io/api/v1"


class NewsSentiment:
    """
    Fetches news sentiment and analyst ratings from Finnhub.
    Works without API key (very limited) — add FINNHUB_API_KEY for full access.
    """

    def __init__(self) -> None:
        settings         = get_settings()
        self._cache_file = settings.state_dir / "news_sentiment_cache.json"
        self._api_key    = os.getenv("FINNHUB_API_KEY", "")
        self._session    = requests.Session()
        self._session.headers.update({"X-Finnhub-Token": self._api_key})
        self._cache: dict = self._load_cache()

        if not self._api_key:
            logger.warning(
                "FINNHUB_API_KEY not set — news sentiment limited. "
                "Get a free key at https://finnhub.io"
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_sentiment_score(self, ticker: str) -> dict:
        """
        Get news sentiment score for a ticker.
        Returns dict with: score, label, positive_count, negative_count, articles
        """
        cache_key = f"sentiment_{ticker}"
        cached    = self._cache.get(cache_key)
        if cached and time.time() - cached.get("ts", 0) < _CACHE_TTL:
            return cached["data"]

        result = self._fetch_news_sentiment(ticker)
        self._cache[cache_key] = {"ts": time.time(), "data": result}
        self._save_cache()
        return result

    def get_analyst_ratings(self, ticker: str) -> dict:
        """
        Get analyst consensus and price target.
        Returns: consensus, target_price, upside_pct, strong_buy, buy, hold, sell
        """
        cache_key = f"ratings_{ticker}"
        cached    = self._cache.get(cache_key)
        if cached and time.time() - cached.get("ts", 0) < _CACHE_TTL:
            return cached["data"]

        result = self._fetch_analyst_ratings(ticker)
        self._cache[cache_key] = {"ts": time.time(), "data": result}
        self._save_cache()
        return result

    def get_earnings_date(self, ticker: str) -> str | None:
        """Return next earnings date for ticker (YYYY-MM-DD) or None."""
        cache_key = f"earnings_{ticker}"
        cached    = self._cache.get(cache_key)
        if cached and time.time() - cached.get("ts", 0) < _CACHE_TTL * 6:
            return cached["data"]

        result = self._fetch_earnings_date(ticker)
        self._cache[cache_key] = {"ts": time.time(), "data": result}
        self._save_cache()
        return result

    def get_full_signal(self, ticker: str, current_price: float) -> dict:
        """
        Get all signals for a ticker in one call.
        Used by the evidence packet assembler.
        """
        sentiment = self.get_sentiment_score(ticker)
        ratings   = self.get_analyst_ratings(ticker)
        earnings  = self.get_earnings_date(ticker)

        # Days to earnings (risk event)
        dte = None
        if earnings:
            try:
                dte = (date.fromisoformat(earnings) - date.today()).days
            except Exception:
                pass

        return {
            "ticker":             ticker,
            "news_score":         sentiment.get("score", 0.0),
            "news_label":         sentiment.get("label", "NEUTRAL"),
            "news_positive":      sentiment.get("positive_count", 0),
            "news_negative":      sentiment.get("negative_count", 0),
            "analyst_consensus":  ratings.get("consensus", "Hold"),
            "analyst_target":     ratings.get("target_price"),
            "analyst_upside_pct": (
                round((ratings["target_price"] - current_price) / current_price * 100, 1)
                if ratings.get("target_price") and current_price
                else None
            ),
            "strong_buy_count":   ratings.get("strong_buy", 0),
            "buy_count":          ratings.get("buy", 0),
            "hold_count":         ratings.get("hold", 0),
            "sell_count":         ratings.get("sell", 0),
            "earnings_date":      earnings,
            "days_to_earnings":   dte,
        }

    # ── Fetchers ──────────────────────────────────────────────────────────────

    def _fetch_news_sentiment(self, ticker: str) -> dict:
        """Fetch and score recent news from Finnhub."""
        if not self._api_key:
            return {"score": 0.0, "label": "NEUTRAL", "positive_count": 0,
                    "negative_count": 0, "articles": []}
        try:
            end   = date.today().isoformat()
            start = (date.today() - timedelta(days=7)).isoformat()
            r = self._session.get(
                f"{_FINNHUB_BASE}/company-news",
                params={"symbol": ticker, "from": start, "to": end},
                timeout=10,
            )
            if r.status_code != 200:
                return _empty_sentiment()

            articles   = r.json()[:20]  # last 20 articles
            pos, neg   = 0, 0
            scored_articles = []

            for a in articles:
                headline = a.get("headline", "").lower()
                summary  = a.get("summary", "").lower()
                text     = headline + " " + summary

                pos_words = sum(1 for w in _POS_WORDS if w in text)
                neg_words = sum(1 for w in _NEG_WORDS if w in text)

                if pos_words > neg_words:
                    pos += 1
                    sentiment = "positive"
                elif neg_words > pos_words:
                    neg += 1
                    sentiment = "negative"
                else:
                    sentiment = "neutral"

                scored_articles.append({
                    "headline":  a.get("headline", ""),
                    "sentiment": sentiment,
                    "source":    a.get("source", ""),
                    "datetime":  a.get("datetime", 0),
                })

            total = len(articles) or 1
            score = round((pos - neg) / total, 3)

            return {
                "score":          score,
                "label":          _sentiment_label(score),
                "positive_count": pos,
                "negative_count": neg,
                "articles":       scored_articles[:5],
            }

        except Exception as exc:
            logger.debug(f"News sentiment fetch failed for {ticker}: {exc}")
            return _empty_sentiment()

    def _fetch_analyst_ratings(self, ticker: str) -> dict:
        """Fetch analyst consensus from Finnhub."""
        if not self._api_key:
            return _empty_ratings()
        try:
            # Recommendation trends
            r = self._session.get(
                f"{_FINNHUB_BASE}/stock/recommendation",
                params={"symbol": ticker},
                timeout=10,
            )
            if r.status_code != 200:
                return _empty_ratings()

            data = r.json()
            if not data:
                return _empty_ratings()

            latest     = data[0]  # most recent period
            strong_buy = latest.get("strongBuy", 0)
            buy        = latest.get("buy", 0)
            hold       = latest.get("hold", 0)
            sell       = latest.get("sell", 0)
            strong_sell = latest.get("strongSell", 0)

            total     = strong_buy + buy + hold + sell + strong_sell or 1
            bull_pct  = (strong_buy + buy) / total

            if bull_pct >= 0.6:    consensus = "Strong Buy"
            elif bull_pct >= 0.45: consensus = "Buy"
            elif bull_pct >= 0.35: consensus = "Hold"
            else:                  consensus = "Sell"

            # Price target
            target_price = self._fetch_price_target(ticker)

            return {
                "consensus":   consensus,
                "target_price": target_price,
                "strong_buy":  strong_buy,
                "buy":         buy,
                "hold":        hold,
                "sell":        sell + strong_sell,
                "bull_pct":    round(bull_pct, 3),
                "period":      latest.get("period", ""),
            }

        except Exception as exc:
            logger.debug(f"Analyst ratings fetch failed for {ticker}: {exc}")
            return _empty_ratings()

    def _fetch_price_target(self, ticker: str) -> float | None:
        """Fetch consensus price target from Finnhub."""
        try:
            r = self._session.get(
                f"{_FINNHUB_BASE}/stock/price-target",
                params={"symbol": ticker},
                timeout=10,
            )
            if r.status_code == 200:
                return r.json().get("targetMean")
        except Exception:
            pass
        return None

    def _fetch_earnings_date(self, ticker: str) -> str | None:
        """Fetch next earnings date from Finnhub."""
        if not self._api_key:
            return None
        try:
            end   = (date.today() + timedelta(days=90)).isoformat()
            start = date.today().isoformat()
            r = self._session.get(
                f"{_FINNHUB_BASE}/calendar/earnings",
                params={"symbol": ticker, "from": start, "to": end},
                timeout=10,
            )
            if r.status_code != 200:
                return None
            items = r.json().get("earningsCalendar", [])
            if items:
                return items[0].get("date")
        except Exception:
            pass
        return None

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _load_cache(self) -> dict:
        if self._cache_file.exists():
            try:
                return json.loads(self._cache_file.read_text())
            except Exception:
                pass
        return {}

    def _save_cache(self) -> None:
        try:
            self._cache_file.write_text(
                json.dumps(self._cache, indent=2, default=str)
            )
        except Exception as exc:
            logger.warning(f"Could not save news cache: {exc}")


# ── Helpers ───────────────────────────────────────────────────────────────────

_POS_WORDS = [
    "beat", "beats", "record", "growth", "surge", "rally", "upgrade",
    "strong", "profit", "revenue", "earnings beat", "raised guidance",
    "buyback", "dividend", "partnership", "deal", "contract", "win",
]

_NEG_WORDS = [
    "miss", "misses", "decline", "loss", "layoff", "cut", "downgrade",
    "warning", "lawsuit", "investigation", "recall", "default", "debt",
    "missed guidance", "lowered", "weak", "disappoints", "fraud",
]


def _sentiment_label(score: float) -> str:
    if score > 0.3:  return "VERY POSITIVE"
    if score > 0.1:  return "POSITIVE"
    if score < -0.3: return "VERY NEGATIVE"
    if score < -0.1: return "NEGATIVE"
    return "NEUTRAL"


def _empty_sentiment() -> dict:
    return {"score": 0.0, "label": "NEUTRAL", "positive_count": 0,
            "negative_count": 0, "articles": []}


def _empty_ratings() -> dict:
    return {"consensus": "Hold", "target_price": None, "strong_buy": 0,
            "buy": 0, "hold": 0, "sell": 0, "bull_pct": 0.0, "period": ""}
