"""
trading/data/historical.py
──────────────────────────
Fetch daily OHLCV bars from the Alpaca free data API.

Endpoint : GET https://data.alpaca.markets/v2/stocks/bars
Feed     : iex (free tier — adequate for daily bars)
Cache    : state/bars/{SYMBOL}.parquet, refreshed every 8 hours
"""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests as _requests

from trading.config import get_settings
from trading.logger import get_logger

logger = get_logger(__name__)

_CACHE_DIR  = Path("state/bars")
_CACHE_TTL  = timedelta(hours=8)
_BATCH_SIZE = 50        # symbols per request — keeps URL params manageable
_CALL_DELAY = 0.3       # seconds between batch calls (respect rate limits)


class HistoricalData:
    """
    Download and cache daily OHLCV bars for US stocks via Alpaca.

    Usage::

        hd = HistoricalData()
        df = hd.get_bars("AAPL")          # single symbol → DataFrame
        bulk = hd.get_bulk_bars(["AAPL", "MSFT", "TSLA"])  # many symbols
    """

    def __init__(self) -> None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        s = get_settings()
        self._session = _requests.Session()
        self._session.headers.update({
            "APCA-API-KEY-ID":     s.api_key,
            "APCA-API-SECRET-KEY": s.api_secret,
        })
        self._data_url = s.data_url

    # ── Public ────────────────────────────────────────────────────────────────

    def get_bars(self, symbol: str, days: int = 504) -> pd.DataFrame:
        """
        Return daily bars for *symbol* as a DataFrame indexed by date.
        ``days=504`` ≈ 2 trading years of history.
        Returns empty DataFrame if the symbol has no data.
        """
        cached = self._load(symbol)
        if cached is not None:
            return cached

        result = self._fetch([symbol], days)
        df = result.get(symbol, pd.DataFrame())
        if not df.empty:
            self._save(symbol, df)
        return df

    def get_bulk_bars(
        self,
        symbols: list[str],
        days: int = 504,
    ) -> dict[str, pd.DataFrame]:
        """
        Return daily bars for many symbols.
        Cached symbols are served instantly; missing ones are batched and fetched.
        Returns a dict mapping symbol → DataFrame (empty DF if no data found).
        """
        results: dict[str, pd.DataFrame] = {}
        missing: list[str] = []

        for sym in symbols:
            hit = self._load(sym)
            if hit is not None:
                results[sym] = hit
            else:
                missing.append(sym)

        if missing:
            logger.info(
                f"HistoricalData: {len(results)} cached | {len(missing)} to fetch"
                f" in {-(-len(missing) // _BATCH_SIZE)} batches"
            )

        for i in range(0, len(missing), _BATCH_SIZE):
            batch   = missing[i : i + _BATCH_SIZE]
            fetched = self._fetch(batch, days)
            for sym, df in fetched.items():
                if not df.empty:
                    self._save(sym, df)
                    results[sym] = df
            if i + _BATCH_SIZE < len(missing):
                time.sleep(_CALL_DELAY)

        return results

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fetch(self, symbols: list[str], days: int) -> dict[str, pd.DataFrame]:
        start = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).strftime("%Y-%m-%d")

        params: dict = {
            "symbols":    ",".join(symbols),
            "timeframe":  "1Day",
            "start":      start,
            "adjustment": "split",   # adjust for splits
            "feed":       "iex",     # free tier
            "limit":      10_000,
        }

        raw_bars: dict[str, list] = {s: [] for s in symbols}
        page_token: str | None = None

        while True:
            if page_token:
                params["page_token"] = page_token

            try:
                resp = self._session.get(
                    f"{self._data_url}/stocks/bars",
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
                body = resp.json()
            except Exception as exc:
                # 400 usually means one bad ticker in the batch — retry individually
                if len(symbols) > 1 and getattr(getattr(exc, "response", None), "status_code", 0) == 400:
                    logger.debug(f"Batch 400 error — retrying {len(symbols)} symbols one-by-one")
                    merged: dict[str, pd.DataFrame] = {}
                    for sym in symbols:
                        try:
                            merged.update(self._fetch([sym], days))
                        except Exception:
                            pass
                    return merged
                logger.warning(f"Bars fetch error for {symbols[:3]}: {exc}")
                break

            for sym, bars in body.get("bars", {}).items():
                raw_bars.setdefault(sym, []).extend(bars)

            page_token = body.get("next_page_token")
            if not page_token:
                break

        dfs: dict[str, pd.DataFrame] = {}
        for sym, bars in raw_bars.items():
            if not bars:
                continue
            df = pd.DataFrame(bars)
            df["t"] = pd.to_datetime(df["t"]).dt.date
            df = (
                df.rename(columns={
                    "t": "date", "o": "open", "h": "high",
                    "l": "low",  "c": "close", "v": "volume",
                })
                [["date", "open", "high", "low", "close", "volume"]]
                .sort_values("date")
                .set_index("date")
            )
            dfs[sym] = df

        return dfs

    def _path(self, symbol: str) -> Path:
        return _CACHE_DIR / f"{symbol}.parquet"

    def _load(self, symbol: str) -> Optional[pd.DataFrame]:
        p = self._path(symbol)
        if not p.exists():
            return None
        age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
        if age > _CACHE_TTL:
            return None
        try:
            return pd.read_parquet(p)
        except Exception:
            return None

    def _save(self, symbol: str, df: pd.DataFrame) -> None:
        try:
            df.to_parquet(self._path(symbol))
        except Exception as exc:
            logger.warning(f"Could not cache bars for {symbol}: {exc}")
