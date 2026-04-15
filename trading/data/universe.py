"""
trading/data/universe.py
────────────────────────
S&P 500 universe loader.

Primary  : fetch current tickers from Wikipedia (free, no API key).
Fallback : curated list of 150 most liquid S&P 500 stocks.
Cache    : state/universe.json, TTL 7 days.
"""

import json
from datetime import datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path

import requests

from trading.logger import get_logger

logger = get_logger(__name__)

_CACHE_FILE = Path("state/universe.json")
_CACHE_TTL  = timedelta(days=7)

# Top-150 S&P 500 by market cap — used when Wikipedia is unavailable
FALLBACK_TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "LLY", "JPM", "V",
    "UNH", "XOM", "MA", "AVGO", "JNJ", "HD", "PG", "COST", "MRK", "ABBV",
    "CVX", "BAC", "WMT", "CRM", "NFLX", "AMD", "KO", "PEP", "TMO", "ACN",
    "LIN", "ABT", "MCD", "PM", "CSCO", "ORCL", "NKE", "DHR", "TXN", "WFC",
    "INTC", "ADBE", "COP", "MS", "NEE", "RTX", "INTU", "SPGI", "AMGN", "BMY",
    "CAT", "QCOM", "IBM", "HON", "GS", "ISRG", "ELV", "GILD", "MDT", "AMAT",
    "DE", "BA", "REGN", "AXP", "SBUX", "TJX", "PLD", "BLK", "SYK", "VRTX",
    "MMC", "MDLZ", "CB", "ZTS", "LRCX", "ADI", "CI", "ETN", "MO", "SO",
    "DUK", "PGR", "GE", "NOC", "ADP", "SHW", "PANW", "KLAC", "MCO", "CME",
    "BSX", "MMM", "ITW", "EOG", "HUM", "APD", "WELL", "MSI", "FCX", "CSX",
    "NSC", "USB", "EMR", "AON", "ROST", "ORLY", "MNST", "AZO", "IDXX", "FTNT",
    "HCA", "DXCM", "ODFL", "CTAS", "GD", "EW", "BIIB", "MCHP", "CTSH", "AFL",
    "KEYS", "TDG", "VRSK", "FAST", "SRE", "PRU", "ANSS", "CPRT", "DLTR", "PAYX",
    "CDW", "HLT", "MAR", "IQV", "TMUS", "CARR", "OTIS", "GEHC", "DECK", "BLDR",
]


# Tickers that cause Alpaca IEX feed 400 errors (class shares, recent renames, etc.)
_ALPACA_UNSUPPORTED = frozenset(["BRK-B", "BF-B", "BRK/B", "BF/B"])


def get_sp500_tickers() -> list[str]:
    """Return S&P 500 tickers. Wikipedia first, fallback list if unavailable."""
    cached = _load_cache()
    if cached:
        return cached

    tickers = _fetch_wikipedia()
    if tickers:
        tickers = _filter_unsupported(tickers)
        _save_cache(tickers)
        logger.info(f"Universe: {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers

    logger.warning("Wikipedia unavailable — using 150-stock fallback list")
    return FALLBACK_TICKERS


def _filter_unsupported(tickers: list[str]) -> list[str]:
    """Remove tickers that Alpaca's IEX feed rejects."""
    filtered = [t for t in tickers if t not in _ALPACA_UNSUPPORTED and "-" not in t]
    removed  = len(tickers) - len(filtered)
    if removed:
        logger.debug(f"Universe: filtered {removed} unsupported tickers")
    return filtered


def get_watchlist_tickers() -> list[str]:
    """Return tickers from the politician watchlist (for quick/test runs)."""
    watchlist_file = Path("state/politician_watchlist.json")
    if not watchlist_file.exists():
        return []
    try:
        data = json.loads(watchlist_file.read_text())
        # Format: list of {politician, tickers: [...], ...}
        politicians = data if isinstance(data, list) else data.get("politicians", [])
        tickers: list[str] = []
        for pol in politicians:
            tickers.extend(pol.get("tickers", pol.get("watchlist", [])))
        return sorted(set(tickers))
    except Exception as exc:
        logger.warning(f"Could not load watchlist: {exc}")
        return []


# ── Internal ──────────────────────────────────────────────────────────────────

class _TableParser(HTMLParser):
    """Extracts ticker symbols from the first column of the S&P 500 Wikipedia table."""

    def __init__(self) -> None:
        super().__init__()
        self.tickers: list[str] = []
        self._in_td    = False
        self._col_idx  = 0
        self._capture  = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._col_idx = 0
        elif tag == "td":
            self._in_td   = True
            self._capture = self._col_idx == 0

    def handle_endtag(self, tag: str) -> None:
        if tag == "td":
            self._in_td  = False
            self._capture = False
            self._col_idx += 1

    def handle_data(self, data: str) -> None:
        if not self._capture:
            return
        ticker = data.strip().replace(".", "-")  # BRK.B → BRK-B (Alpaca format)
        if ticker and ticker.isupper() and 1 <= len(ticker) <= 5:
            self.tickers.append(ticker)


def _fetch_wikipedia() -> list[str]:
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (compatible; AI-Trading-Bot/1.0)"},
            timeout=20,
        )
        resp.raise_for_status()
        parser = _TableParser()
        parser.feed(resp.text)
        tickers = parser.tickers
        if len(tickers) >= 400:
            return tickers
        logger.warning(f"Wikipedia returned only {len(tickers)} tickers — too few, skipping")
        return []
    except Exception as exc:
        logger.warning(f"Wikipedia fetch failed: {exc}")
        return []


def _load_cache() -> list[str] | None:
    try:
        if not _CACHE_FILE.exists():
            return None
        data = json.loads(_CACHE_FILE.read_text())
        age  = datetime.now() - datetime.fromisoformat(data["saved_at"])
        if age > _CACHE_TTL:
            return None
        return data["tickers"]
    except Exception:
        return None


def _save_cache(tickers: list[str]) -> None:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(
        {"saved_at": datetime.now().isoformat(), "tickers": tickers},
        indent=2,
    ))
