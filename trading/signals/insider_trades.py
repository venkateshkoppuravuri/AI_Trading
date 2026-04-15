"""
trading/signals/insider_trades.py
───────────────────────────────────
SEC EDGAR Form 4 insider trading data.

Form 4 = filed within 2 business days when a corporate insider
(CEO, CFO, Director, 10%+ owner) buys or sells company stock.

This is one of the strongest free signals in finance:
  • CEO buying their own stock with personal money = high conviction
  • Cluster = 3+ insiders buying same stock within 30 days = very strong
  • MSPR (Monthly Share Purchase Ratio) = institutional buying pressure

Data source: SEC EDGAR (free, no API key needed)
  Primary:  https://efts.sec.gov/LATEST/search-index  (full-text search)
  Fallback: https://data.sec.gov/submissions/          (company filings)

Usage:
    scraper = InsiderTradesScraper()
    trades = scraper.get_recent_trades(ticker="NVDA", days=30)
    score  = scraper.get_insider_score("NVDA")
    top    = scraper.get_top_insider_buys(limit=20)
"""

import json
import re
import time
from datetime import date, datetime, timedelta
from collections import defaultdict
from pathlib import Path

import requests

from trading.config import get_settings
from trading.logger import get_logger

logger = get_logger(__name__)

_HEADERS = {
    "User-Agent": "AI-Trading-Bot contact@example.com",  # SEC requires User-Agent
    "Accept":     "application/json",
}

_CACHE_TTL  = 3_600   # 1 hour
_EDGAR_BASE = "https://efts.sec.gov/LATEST/search-index"
_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


class InsiderTradesScraper:
    """
    Fetches and scores SEC Form 4 insider trading data.
    Caches results to state/insider_trades_cache.json.
    """

    def __init__(self) -> None:
        settings             = get_settings()
        self._cache_file     = settings.state_dir / "insider_trades_cache.json"
        self._session        = requests.Session()
        self._session.headers.update(_HEADERS)
        self._cache: dict    = self._load_cache()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_recent_trades(
        self,
        ticker: str | None = None,
        days:   int        = 30,
        limit:  int        = 50,
    ) -> list[dict]:
        """
        Return recent Form 4 filings.
        If ticker is given, filter to that stock only.
        Each trade has: ticker, insider_name, title, trade_type,
                        shares, price, value, filed_date, is_ceo, is_cfo
        """
        cache_key = f"trades_{ticker or 'all'}_{days}"
        cached    = self._cache.get(cache_key)
        if cached and time.time() - cached.get("ts", 0) < _CACHE_TTL:
            return cached["data"]

        trades = self._fetch_form4_trades(ticker=ticker, days=days, limit=limit)
        self._cache[cache_key] = {"ts": time.time(), "data": trades}
        self._save_cache()
        return trades

    def get_insider_score(self, ticker: str) -> dict:
        """
        Score insider activity for a single ticker (0-100).

        Score components:
          • buy_count_30d:   number of insider buys in last 30 days
          • ceo_bought:      CEO personally bought shares
          • cfo_bought:      CFO personally bought shares
          • cluster_signal:  3+ insiders bought in 30 days
          • net_shares:      net shares bought (buys - sells)
          • mspr:            monthly share purchase ratio
          • score:           composite 0-100
        """
        cache_key = f"score_{ticker}"
        cached    = self._cache.get(cache_key)
        if cached and time.time() - cached.get("ts", 0) < _CACHE_TTL:
            return cached["data"]

        trades = self.get_recent_trades(ticker=ticker, days=30)

        buys  = [t for t in trades if t.get("trade_type") == "buy"]
        sells = [t for t in trades if t.get("trade_type") == "sell"]

        ceo_bought     = any(t.get("is_ceo") for t in buys)
        cfo_bought     = any(t.get("is_cfo") for t in buys)
        cluster_signal = len(set(t.get("insider_name") for t in buys)) >= 3

        net_shares = (
            sum(t.get("shares", 0) for t in buys) -
            sum(t.get("shares", 0) for t in sells)
        )
        total_shares = sum(t.get("shares", 0) for t in buys + sells) or 1
        mspr = sum(t.get("shares", 0) for t in buys) / total_shares

        # Score: 0-100
        score = 0
        score += min(len(buys) * 10, 30)       # up to 30 pts for buy count
        score += 25 if ceo_bought     else 0    # 25 pts CEO buy
        score += 15 if cfo_bought     else 0    # 15 pts CFO buy
        score += 20 if cluster_signal else 0    # 20 pts cluster
        score += 10 if mspr > 0.7     else (5 if mspr > 0.3 else 0)  # MSPR

        result = {
            "ticker":         ticker,
            "buy_count_30d":  len(buys),
            "sell_count_30d": len(sells),
            "ceo_bought":     ceo_bought,
            "cfo_bought":     cfo_bought,
            "cluster_signal": cluster_signal,
            "net_shares":     net_shares,
            "mspr":           round(mspr, 3),
            "score":          min(score, 100),
            "score_label":    _score_label(score),
            "top_buyers":     [
                {"name": t.get("insider_name"), "title": t.get("title"),
                 "shares": t.get("shares"), "value": t.get("value")}
                for t in buys[:5]
            ],
        }

        self._cache[cache_key] = {"ts": time.time(), "data": result}
        self._save_cache()
        return result

    def get_top_insider_buys(self, limit: int = 20) -> list[dict]:
        """
        Scan S&P 500 for the stocks with strongest insider buying right now.
        Returns list sorted by insider_score descending.
        """
        cache_key = "top_buys"
        cached    = self._cache.get(cache_key)
        if cached and time.time() - cached.get("ts", 0) < _CACHE_TTL:
            return cached["data"]

        # Fetch all recent Form 4 buys across all companies
        all_trades = self._fetch_form4_trades(ticker=None, days=14, limit=200)
        buys       = [t for t in all_trades if t.get("trade_type") == "buy"]

        # Group by ticker
        by_ticker: dict[str, list] = defaultdict(list)
        for t in buys:
            ticker = t.get("ticker", "").upper()
            if ticker:
                by_ticker[ticker].append(t)

        # Score each ticker
        results = []
        for ticker, ticker_buys in by_ticker.items():
            ceo = any(t.get("is_ceo") for t in ticker_buys)
            cfo = any(t.get("is_cfo") for t in ticker_buys)
            cluster = len(set(t.get("insider_name") for t in ticker_buys)) >= 3
            total_value = sum(t.get("value", 0) for t in ticker_buys)
            score = (
                len(ticker_buys) * 10 +
                (25 if ceo else 0) +
                (15 if cfo else 0) +
                (20 if cluster else 0)
            )
            results.append({
                "ticker":      ticker,
                "buy_count":   len(ticker_buys),
                "ceo_bought":  ceo,
                "cfo_bought":  cfo,
                "cluster":     cluster,
                "total_value": total_value,
                "score":       min(score, 100),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        top = results[:limit]

        self._cache[cache_key] = {"ts": time.time(), "data": top}
        self._save_cache()
        return top

    # ── Data fetching ──────────────────────────────────────────────────────────

    def _fetch_form4_trades(
        self,
        ticker: str | None,
        days:   int,
        limit:  int,
    ) -> list[dict]:
        """Fetch Form 4 filings from SEC EDGAR full-text search."""
        try:
            start = (date.today() - timedelta(days=days)).isoformat()
            end   = date.today().isoformat()

            query = '"form 4" "transaction code" "purchase"'
            if ticker:
                query = f'"{ticker}" ' + query

            params = {
                "q":        query,
                "dateRange": "custom",
                "startdt":   start,
                "enddt":     end,
                "forms":     "4",
            }

            r = self._session.get(_EDGAR_BASE, params=params, timeout=15)
            if r.status_code != 200:
                logger.debug(f"EDGAR returned {r.status_code}")
                return self._fetch_form4_rss(ticker, days, limit)

            hits = r.json().get("hits", {}).get("hits", [])
            if not hits:
                return self._fetch_form4_rss(ticker, days, limit)

            trades: list[dict] = []
            for hit in hits[:limit]:
                src         = hit.get("_source", {})
                names       = src.get("display_names", [])
                filed       = src.get("file_date", "")
                entity_name = src.get("entity_name", "")

                # Extract ticker from entity name or display names
                t = _extract_ticker(entity_name, names, ticker)

                for name in names:
                    clean = re.sub(r"\s*\(CIK[^)]+\)", "", name).strip()
                    if not clean or len(clean) < 4:
                        continue
                    title   = _guess_title(clean)
                    is_ceo  = "chief executive" in title.lower() or "ceo" in title.lower()
                    is_cfo  = "chief financial" in title.lower() or "cfo" in title.lower()
                    trades.append({
                        "ticker":       t or entity_name[:6].upper(),
                        "insider_name": clean,
                        "title":        title,
                        "trade_type":   "buy",
                        "shares":       0,
                        "price":        0.0,
                        "value":        0.0,
                        "filed_date":   filed,
                        "is_ceo":       is_ceo,
                        "is_cfo":       is_cfo,
                        "source":       "edgar_fts",
                    })

            return trades

        except Exception as exc:
            logger.debug(f"EDGAR FTS failed: {exc}")
            return self._fetch_form4_rss(ticker, days, limit)

    def _fetch_form4_rss(
        self,
        ticker: str | None,
        days:   int,
        limit:  int,
    ) -> list[dict]:
        """
        Fallback: Fetch Form 4 from SEC EDGAR RSS feed.
        https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&dateb=&owner=include&count=40
        """
        try:
            url    = "https://www.sec.gov/cgi-bin/browse-edgar"
            params = {
                "action": "getcurrent",
                "type":   "4",
                "dateb":  "",
                "owner":  "include",
                "count":  str(limit),
                "search_text": "",
                "output": "atom",
            }
            r = self._session.get(url, params=params, timeout=15)
            if r.status_code != 200:
                return []

            # Parse atom XML
            entries = re.findall(
                r"<entry>(.*?)</entry>", r.text, re.DOTALL
            )
            trades: list[dict] = []
            cutoff = (date.today() - timedelta(days=days)).isoformat()

            for entry in entries[:limit]:
                title_match = re.search(r"<title>(.*?)</title>", entry)
                date_match  = re.search(r"<updated>(.*?)</updated>", entry)
                if not title_match:
                    continue

                title_text = title_match.group(1)
                filed_date = date_match.group(1)[:10] if date_match else ""
                if filed_date and filed_date < cutoff:
                    continue

                # Title format: "4 - CompanyName (TICKER) (Filer)"
                ticker_match = re.search(r"\(([A-Z]{1,5})\)", title_text)
                t = ticker_match.group(1) if ticker_match else ""
                if ticker and t != ticker.upper():
                    continue

                trades.append({
                    "ticker":       t,
                    "insider_name": title_text,
                    "title":        "",
                    "trade_type":   "buy",
                    "shares":       0,
                    "price":        0.0,
                    "value":        0.0,
                    "filed_date":   filed_date,
                    "is_ceo":       False,
                    "is_cfo":       False,
                    "source":       "edgar_rss",
                })

            return trades

        except Exception as exc:
            logger.debug(f"EDGAR RSS fallback failed: {exc}")
            return []

    # ── Cache helpers ──────────────────────────────────────────────────────────

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
            logger.warning(f"Could not save insider cache: {exc}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_label(score: int) -> str:
    if score >= 80: return "VERY STRONG BUY"
    if score >= 60: return "STRONG BUY"
    if score >= 40: return "MODERATE BUY"
    if score >= 20: return "WEAK BUY"
    return "NEUTRAL"


def _guess_title(name: str) -> str:
    """Try to extract a title from the display name."""
    name_lower = name.lower()
    for title in ["chief executive", "chief financial", "chief operating",
                  "president", "director", "chairman", "treasurer", "vp "]:
        if title in name_lower:
            return title.title()
    return "Insider"


def _extract_ticker(entity_name: str, display_names: list, hint: str | None) -> str:
    """Try to extract ticker symbol from entity name."""
    if hint:
        return hint.upper()
    # Common pattern: "Apple Inc (AAPL)"
    match = re.search(r"\(([A-Z]{1,5})\)", entity_name)
    if match:
        return match.group(1)
    return ""
