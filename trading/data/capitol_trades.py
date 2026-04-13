"""
trading/data/capitol_trades.py
───────────────────────────────
Congressional stock trade data aggregator.

Data source priority (tries each in order until one returns usable data):

  1. Senate Stock Watcher  — s3 JSON updated daily with real senator PTR data
     https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json

  2. House Stock Watcher   — s3 JSON updated daily with real House member PTR data
     https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json

  3. Senate eFD (EFTS)    — SEC EDGAR full-text search for periodic-transaction-report filings
     https://efts.sec.gov/LATEST/search-index

  4. House Disclosures CSV — disclosures.house.gov annual PTR filing list
     https://disclosures.house.gov/public_disc/financial-pdfs/{year}FD.txt

  5. Watchlist fallback    — state/politician_watchlist.json  (user-editable)

Why not Capitol Trades?
  capitoltrades.com is a client-side Next.js SPA — no data is embedded in the
  HTML and the backend API is not publicly documented or accessible.

Usage:
    scraper = CapitolTradesScraper()
    politician, trades = scraper.get_top_politician_trades()
"""

import csv
import io
import json
import re
import time
from collections import defaultdict
from datetime import date, timedelta

import requests

from trading.config import get_settings
from trading.exceptions import ScraperError
from trading.logger import get_logger

logger = get_logger(__name__)

_CACHE_TTL = 3_600   # seconds — 1 hour

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}

# ── Senate Stock Watcher S3 ───────────────────────────────────────────────────
_SENATE_S3 = (
    "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com"
    "/aggregate/all_transactions.json"
)

# ── House Stock Watcher S3 ────────────────────────────────────────────────────
_HOUSE_S3 = (
    "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com"
    "/data/all_transactions.json"
)

# ── Default watchlist written on first run (user can edit the file) ───────────
_DEFAULT_WATCHLIST: list[dict] = [
    {
        "politician": "Tommy Tuberville",
        "chamber":    "senate",
        "party":      "R",
        "tickers":    ["LMT", "RTX", "NOC", "GD", "XOM", "CVX"],
        "trade_type": "purchase",
        "note":       "Very active Senate trader — defense, energy, commodities",
    },
    {
        "politician": "Nancy Pelosi",
        "chamber":    "house",
        "party":      "D",
        "tickers":    ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"],
        "trade_type": "purchase",
        "note":       "Famous for large tech-sector purchases and calls",
    },
    {
        "politician": "Josh Gottheimer",
        "chamber":    "house",
        "party":      "D",
        "tickers":    ["MSFT", "AAPL", "GOOGL", "META", "CRM"],
        "trade_type": "purchase",
        "note":       "One of the most active House traders; tech focus",
    },
]


class CapitolTradesScraper:
    """
    Aggregate congressional stock trade disclosures from public government
    S3 data feeds.  Falls back through SEC EFTS → House CSV → user watchlist.
    """

    def __init__(self) -> None:
        settings             = get_settings()
        self._cache_file     = settings.state_dir / "capitol_trades_cache.json"
        self._watchlist_file = settings.state_dir / "politician_watchlist.json"
        self._session        = requests.Session()
        self._session.headers.update(_HEADERS)
        self._ensure_watchlist()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_recent_trades(self, limit: int = 200) -> list[dict]:
        """
        Return recent congressional stock trades from the best available source.

        Each trade dict has keys:
          politician, ticker, trade_type, traded_date, filed_date, source

        Raises ScraperError only if all five sources fail.
        """
        for fetch, label in [
            (self._fetch_senate_s3,    "Senate Stock Watcher S3"),
            (self._fetch_house_s3,     "House Stock Watcher S3"),
            (self._fetch_efts_trades,  "SEC EFTS"),
            (self._fetch_house_csv,    "House Disclosures CSV"),
        ]:
            try:
                trades = fetch(limit)
                if trades:
                    logger.info(f"Data source: {label} — {len(trades)} trades")
                    return trades
            except Exception as exc:
                logger.debug(f"{label} failed: {exc}")

        # Watchlist fallback
        trades = self._trades_from_watchlist()
        if trades:
            logger.warning(
                "All live sources unavailable — using watchlist fallback. "
                f"Edit {self._watchlist_file} to customise."
            )
            return trades

        raise ScraperError(
            "No congressional trade data available. "
            "All live sources failed and the watchlist is empty."
        )

    def rank_politicians(self, trades: list[dict]) -> list[dict]:
        """Score politicians by activity + recency. Returns sorted list."""
        scores: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "buys": 0, "sells": 0, "tickers": set(), "latest": ""}
        )
        for t in trades:
            name = t.get("politician", "").strip()
            if not name or len(name) < 3:
                continue
            scores[name]["count"] += 1
            ttype = t.get("trade_type", "").lower()
            if any(w in ttype for w in ("buy", "purchase")):
                scores[name]["buys"] += 1
            elif any(w in ttype for w in ("sell", "sale")):
                scores[name]["sells"] += 1
            ticker = t.get("ticker", "").strip()
            if ticker and 1 <= len(ticker) <= 5 and re.match(r"^[A-Z]+$", ticker):
                scores[name]["tickers"].add(ticker)
            d = t.get("filed_date") or t.get("traded_date") or ""
            if d > scores[name]["latest"]:
                scores[name]["latest"] = d

        ranked = [
            {
                "politician":  name,
                "score":       d["count"] * 10 + d["buys"] * 5,
                "trade_count": d["count"],
                "buys":        d["buys"],
                "sells":       d["sells"],
                "tickers":     sorted(d["tickers"]),
                "latest":      d["latest"],
            }
            for name, d in scores.items()
        ]
        ranked.sort(key=lambda x: (x["score"], x["latest"]), reverse=True)
        return ranked

    def get_top_politician_trades(self) -> tuple[str | None, list[dict]]:
        """
        Return (politician_name, their_trades) for the #1 most active trader.
        Convenience wrapper around get_top_n_politician_trades(1).
        """
        results = self.get_top_n_politician_trades(n=1)
        if not results:
            return None, []
        return results[0]

    def get_top_n_politician_trades(
        self, n: int = 3
    ) -> list[tuple[str, list[dict]]]:
        """
        Return the top-N most active politicians as a list of
        (politician_name, their_trades) tuples, ordered by activity score.
        Results are cached for _CACHE_TTL seconds.
        """
        cached = self._load_cache()
        if cached:
            logger.info(f"Using cached trade data (age {cached['age']}s)")
            all_trades = cached["all_trades"]
            ranked     = cached["ranked"]
        else:
            try:
                all_trades = self.get_recent_trades(limit=200)
            except ScraperError as exc:
                logger.error(f"Trade fetch failed: {exc}")
                return []

            ranked = self.rank_politicians(all_trades)
            if not ranked:
                logger.warning("No politicians found in trade data")
                return []

            self._save_cache(all_trades, ranked)

        results: list[tuple[str, list[dict]]] = []
        for entry in ranked[:n]:
            pol    = entry["politician"]
            trades = [t for t in all_trades if t.get("politician", "").strip() == pol]
            logger.info(
                f"Top-{n} trader: {pol} | trades={entry['trade_count']} | "
                f"buys={entry['buys']} | tickers={entry['tickers'][:5]}"
            )
            results.append((pol, trades))

        return results

    def print_leaderboard(self, n: int = 10) -> None:
        """Print a ranked leaderboard of the most active congressional traders."""
        try:
            trades = self.get_recent_trades(limit=200)
        except ScraperError as exc:
            print(f"Could not fetch leaderboard: {exc}")
            return
        ranked = self.rank_politicians(trades)
        w = 68
        print(f"\n{'=' * w}")
        print("  Congressional Trader Leaderboard")
        print(f"{'=' * w}")
        for i, p in enumerate(ranked[:n], 1):
            tickers = ", ".join(p["tickers"][:5]) or "—"
            print(
                f"  {i:2}. {p['politician']:<30} "
                f"trades={p['trade_count']:3}  buys={p['buys']:2}  [{tickers}]"
            )
        print(f"{'=' * w}\n")

    # ── Source 1: Senate Stock Watcher S3 ─────────────────────────────────────

    def _fetch_senate_s3(self, limit: int) -> list[dict]:
        """
        Senate Periodic Transaction Reports from the Senate Stock Watcher project.
        JSON array of transactions with keys: senator, ticker, type,
        transaction_date, disclosure_date, amount, asset_type.
        """
        r = self._session.get(_SENATE_S3, timeout=20)
        r.raise_for_status()
        data = r.json()

        # The response is either a list or {"transactions": [...]}
        rows = data if isinstance(data, list) else data.get("transactions", [])

        cutoff = _ninety_days_ago()
        trades: list[dict] = []
        for row in rows:
            # Filter to recent stock purchases/sales only
            asset = row.get("asset_type", "").lower()
            if "stock" not in asset and "equit" not in asset and "etf" not in asset:
                # allow empty asset_type through (some rows omit it)
                if asset:
                    continue

            disclosure = row.get("disclosure_date", "") or ""
            traded     = row.get("transaction_date", "") or ""
            ref_date   = disclosure or traded
            if ref_date and ref_date < cutoff:
                continue

            ticker = (row.get("ticker") or "").strip().upper()
            if not ticker or len(ticker) > 5 or not re.match(r"^[A-Z]+$", ticker):
                continue

            ttype_raw = (row.get("type") or "").lower()
            if "purchase" in ttype_raw or "buy" in ttype_raw:
                ttype = "purchase"
            elif "sale" in ttype_raw or "sell" in ttype_raw:
                ttype = "sale"
            else:
                ttype = ttype_raw or "purchase"

            senator = (
                row.get("senator")
                or row.get("name")
                or row.get("politician")
                or ""
            ).strip()
            if not senator:
                continue

            trades.append({
                "politician":  senator,
                "ticker":      ticker,
                "trade_type":  ttype,
                "traded_date": traded,
                "filed_date":  disclosure,
                "amount":      row.get("amount", ""),
                "source":      "senate_s3",
            })
            if len(trades) >= limit:
                break

        return trades

    # ── Source 2: House Stock Watcher S3 ──────────────────────────────────────

    def _fetch_house_s3(self, limit: int) -> list[dict]:
        """
        House Periodic Transaction Reports from the House Stock Watcher project.
        JSON array with keys: representative, ticker, type,
        transaction_date, disclosure_date, amount, asset_type.
        """
        r = self._session.get(_HOUSE_S3, timeout=20)
        r.raise_for_status()
        data = r.json()

        rows = data if isinstance(data, list) else data.get("transactions", [])

        cutoff = _ninety_days_ago()
        trades: list[dict] = []
        for row in rows:
            asset = row.get("asset_type", "").lower()
            if "stock" not in asset and "equit" not in asset and "etf" not in asset:
                if asset:
                    continue

            disclosure = row.get("disclosure_date", "") or ""
            traded     = row.get("transaction_date", "") or ""
            ref_date   = disclosure or traded
            if ref_date and ref_date < cutoff:
                continue

            ticker = (row.get("ticker") or "").strip().upper()
            if not ticker or len(ticker) > 5 or not re.match(r"^[A-Z]+$", ticker):
                continue

            ttype_raw = (row.get("type") or "").lower()
            if "purchase" in ttype_raw or "buy" in ttype_raw:
                ttype = "purchase"
            elif "sale" in ttype_raw or "sell" in ttype_raw:
                ttype = "sale"
            else:
                ttype = ttype_raw or "purchase"

            rep = (
                row.get("representative")
                or row.get("senator")
                or row.get("name")
                or ""
            ).strip()
            if not rep:
                continue

            trades.append({
                "politician":  rep,
                "ticker":      ticker,
                "trade_type":  ttype,
                "traded_date": traded,
                "filed_date":  disclosure,
                "amount":      row.get("amount", ""),
                "source":      "house_s3",
            })
            if len(trades) >= limit:
                break

        return trades

    # ── Source 3: SEC EFTS full-text search ───────────────────────────────────

    def _fetch_efts_trades(self, limit: int) -> list[dict]:
        """
        Query SEC EDGAR full-text search for recent PTR (Periodic Transaction
        Report) filings.  Tickers are not in the index so we return name-only
        records; the copy-trading strategy won't act on them without tickers.
        """
        url    = "https://efts.sec.gov/LATEST/search-index"
        params = {
            "q":        '"periodic transaction report" "purchase"',
            "dateRange": "custom",
            "startdt":   _ninety_days_ago(),
            "enddt":     _today(),
        }
        r = self._session.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return []

        hits = r.json().get("hits", {}).get("hits", [])
        if not hits:
            return []

        trades: list[dict] = []
        for hit in hits[:limit]:
            src  = hit.get("_source", {})
            date_val = src.get("period_ending", "")
            for raw_name in src.get("display_names", []):
                clean = re.sub(r"\s*\(CIK[^)]+\)", "", raw_name).strip()
                if any(w in clean for w in ("Inc", "Corp", "LLC", "Ltd", "Trust")):
                    continue
                if not clean or len(clean) < 4:
                    continue
                # No ticker available — only useful for leaderboard, not buy signals
                trades.append({
                    "politician":  clean,
                    "ticker":      "",
                    "trade_type":  "purchase",
                    "traded_date": date_val,
                    "filed_date":  date_val,
                    "source":      "efts",
                })

        return trades

    # ── Source 4: House Financial Disclosures CSV + Watchlist hybrid ─────────

    def _fetch_house_csv(self, limit: int) -> list[dict]:
        """
        Hybrid source: rank politicians by real PTR filing frequency from the
        House Clerk financial disclosure CSV, then cross-reference each active
        politician against the user's watchlist to obtain actionable tickers.

        Result: genuine activity data (who is actually filing trade reports)
        combined with known tickers for those politicians.
        """
        # Try current and prior year (the current year list builds over time)
        for year in (_current_year(), _current_year() - 1):
            url = (
                f"https://disclosures-clerk.house.gov"
                f"/public_disc/financial-pdfs/{year}FD.txt"
            )
            try:
                r = self._session.get(url, timeout=20)
            except Exception:
                continue
            if r.status_code != 200:
                continue

            # Count PTR filings per politician; track most recent date
            ptr_count: dict[str, int] = defaultdict(int)
            ptr_latest: dict[str, str] = defaultdict(str)
            reader = csv.DictReader(io.StringIO(r.text), delimiter="\t")
            for row in reader:
                if "P" not in row.get("FilingType", ""):
                    continue
                first = row.get("First", "").strip()
                last  = row.get("Last",  "").strip()
                name  = f"{first} {last}".strip()
                if not name or name == " ":
                    continue
                ptr_count[name] += 1
                d = row.get("FilingDate", "")
                if d > ptr_latest[name]:
                    ptr_latest[name] = d

            if not ptr_count:
                continue

            # Sort by PTR count descending (most active traders first)
            ranked = sorted(ptr_count.items(), key=lambda kv: kv[1], reverse=True)

            # Cross-reference with watchlist for tickers
            watchlist = self._load_watchlist()
            wl_by_last: dict[str, dict] = {}
            for entry in watchlist:
                pol_name = entry.get("politician", "")
                last_n   = pol_name.split()[-1].lower() if pol_name else ""
                if last_n:
                    wl_by_last[last_n] = entry

            trades: list[dict] = []
            for name, count in ranked:
                last_n = name.split()[-1].lower()
                entry  = wl_by_last.get(last_n)
                if not entry or not entry.get("tickers"):
                    continue

                filed = ptr_latest.get(name, _today())
                for ticker in entry["tickers"]:
                    if not ticker:
                        continue
                    trades.append({
                        "politician":  entry.get("politician", name),
                        "ticker":      ticker,
                        "trade_type":  entry.get("trade_type", "purchase"),
                        "traded_date": filed,
                        "filed_date":  filed,
                        "ptr_count":   count,
                        "source":      "house_fd_hybrid",
                    })
                    if len(trades) >= limit:
                        break
                if len(trades) >= limit:
                    break

            if trades:
                logger.info(
                    f"House FD hybrid: {len(ptr_count)} politicians filed PTRs in {year} | "
                    f"matched {len(set(t['politician'] for t in trades))} watchlist entries"
                )
                return trades

        return []

    # ── Source 5: User-editable watchlist ─────────────────────────────────────

    def _trades_from_watchlist(self) -> list[dict]:
        """Convert politician_watchlist.json into synthetic trade records."""
        watchlist = self._load_watchlist()
        if not watchlist:
            return []

        today  = _today()
        trades: list[dict] = []
        for entry in watchlist:
            politician = entry.get("politician", "")
            for ticker in entry.get("tickers", []):
                if not ticker:
                    continue
                trades.append({
                    "politician":  politician,
                    "ticker":      ticker,
                    "trade_type":  entry.get("trade_type", "purchase"),
                    "traded_date": today,
                    "filed_date":  today,
                    "source":      "watchlist",
                })
        return trades

    # ── Watchlist file helpers ─────────────────────────────────────────────────

    def _ensure_watchlist(self) -> None:
        if not self._watchlist_file.exists():
            with open(self._watchlist_file, "w") as f:
                json.dump(_DEFAULT_WATCHLIST, f, indent=2)
            logger.info(f"Created default watchlist: {self._watchlist_file}")

    def _load_watchlist(self) -> list[dict]:
        try:
            with open(self._watchlist_file) as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"Could not load watchlist: {exc}")
            return []

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _load_cache(self) -> dict | None:
        if not self._cache_file.exists():
            return None
        try:
            with open(self._cache_file) as f:
                cache = json.load(f)
            age = time.time() - cache.get("timestamp", 0)
            if age < _CACHE_TTL:
                cache["age"] = int(age)
                return cache
        except Exception:
            pass
        return None

    def _save_cache(self, all_trades: list[dict], ranked: list[dict]) -> None:
        try:
            with open(self._cache_file, "w") as f:
                json.dump(
                    {
                        "timestamp":  time.time(),
                        "all_trades": all_trades,
                        "ranked":     ranked[:20],
                        # legacy key — kept so old get_top_politician_trades callers
                        # reading cache directly still work during transition
                        "politician": ranked[0]["politician"] if ranked else "",
                        "trades":     [
                            t for t in all_trades
                            if t.get("politician") == (ranked[0]["politician"] if ranked else "")
                        ],
                    },
                    f,
                    indent=2,
                )
        except Exception as exc:
            logger.warning(f"Could not save cache: {exc}")


# ── Date helpers ──────────────────────────────────────────────────────────────

def _today() -> str:
    return date.today().isoformat()


def _ninety_days_ago() -> str:
    return (date.today() - timedelta(days=90)).isoformat()


def _current_year() -> int:
    return date.today().year
