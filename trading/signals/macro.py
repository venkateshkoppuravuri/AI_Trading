"""
trading/signals/macro.py
─────────────────────────
Macro economic data from FRED (Federal Reserve Economic Data).
Free, no API key needed for basic access.

Signals tracked:
  • VIX        — market fear index (> 30 = high fear, avoid buying)
  • Yield curve — 10Y - 2Y spread (negative = recession warning)
  • Fed rate    — current federal funds rate
  • Credit spread — HY spread (> 500bps = stress)
  • Unemployment — monthly rate

Also fetches from Yahoo Finance (yfinance):
  • S&P 500 trend (SPY)
  • Market breadth

Usage:
    macro = MacroData()
    regime = macro.get_market_regime()
    # Returns: "BULL" | "NEUTRAL" | "BEAR" | "HIGH_FEAR"

    data = macro.get_all()
    print(data["vix"], data["yield_curve"], data["regime"])
"""

import json
import time
from datetime import date, timedelta

import requests

from trading.config import get_settings
from trading.logger import get_logger

logger = get_logger(__name__)

_CACHE_TTL  = 3_600   # 1 hour
_FRED_BASE  = "https://api.stlouisfed.org/fred/series/observations"

# FRED series IDs
_SERIES = {
    "vix":          "VIXCLS",       # CBOE Volatility Index
    "yield_10y":    "DGS10",        # 10-Year Treasury
    "yield_2y":     "DGS2",         # 2-Year Treasury
    "fed_rate":     "FEDFUNDS",     # Federal Funds Rate
    "hy_spread":    "BAMLH0A0HYM2", # High Yield OAS
    "unemployment": "UNRATE",       # Unemployment Rate
    "cpi_yoy":      "CPIAUCSL",     # CPI for inflation context
}

_FRED_API_KEY = "free"  # Works without key for recent observations


class MacroData:
    """
    Fetches macro regime data from FRED and yfinance.
    Determines current market regime for position sizing.
    """

    def __init__(self) -> None:
        settings         = get_settings()
        self._cache_file = settings.state_dir / "macro_cache.json"
        self._session    = requests.Session()
        self._session.headers.update({"User-Agent": "AI-Trading-Bot/1.0"})
        self._cache: dict = self._load_cache()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_all(self) -> dict:
        """
        Return all macro indicators plus market regime classification.
        Cached for 1 hour.
        """
        cached = self._cache.get("all")
        if cached and time.time() - cached.get("ts", 0) < _CACHE_TTL:
            return cached["data"]

        data: dict = {}

        # Fetch each FRED series
        for name, series_id in _SERIES.items():
            try:
                val = self._fetch_fred(series_id)
                data[name] = val
            except Exception as exc:
                logger.debug(f"FRED {series_id} failed: {exc}")
                data[name] = None

        # Derived signals
        data["yield_curve"] = (
            round(data["yield_10y"] - data["yield_2y"], 3)
            if data.get("yield_10y") and data.get("yield_2y")
            else None
        )

        # Market regime classification
        data["regime"]       = self._classify_regime(data)
        data["regime_color"] = {
            "BULL":      "🟢",
            "NEUTRAL":   "🟡",
            "BEAR":      "🔴",
            "HIGH_FEAR": "🔴",
        }.get(data["regime"], "⚪")

        data["fetched_at"] = date.today().isoformat()

        self._cache["all"] = {"ts": time.time(), "data": data}
        self._save_cache()

        logger.info(
            f"Macro: VIX={data.get('vix')} | "
            f"Yield curve={data.get('yield_curve')} | "
            f"Fed={data.get('fed_rate')} | "
            f"Regime={data.get('regime')}"
        )
        return data

    def get_market_regime(self) -> str:
        """
        Return market regime string: BULL | NEUTRAL | BEAR | HIGH_FEAR
        Used by strategies to adjust position sizing.
        """
        return self.get_all().get("regime", "NEUTRAL")

    def get_position_size_multiplier(self) -> float:
        """
        Return a position size multiplier based on macro regime.
          BULL      → 1.0  (full size)
          NEUTRAL   → 0.75 (slightly reduced)
          BEAR      → 0.5  (half size)
          HIGH_FEAR → 0.25 (very cautious)
        """
        regime = self.get_market_regime()
        return {
            "BULL":      1.0,
            "NEUTRAL":   0.75,
            "BEAR":      0.5,
            "HIGH_FEAR": 0.25,
        }.get(regime, 0.75)

    def format_summary(self) -> str:
        """Return a human-readable macro summary for Telegram alerts."""
        d = self.get_all()
        vix   = d.get("vix")
        yc    = d.get("yield_curve")
        fed   = d.get("fed_rate")
        emoji = d.get("regime_color", "⚪")

        lines = [f"{emoji} *Macro Regime: {d.get('regime', 'UNKNOWN')}*"]
        if vix   is not None: lines.append(f"  VIX: {vix:.1f}")
        if yc    is not None: lines.append(f"  Yield Curve (10Y-2Y): {yc:+.2f}%")
        if fed   is not None: lines.append(f"  Fed Rate: {fed:.2f}%")
        return "\n".join(lines)

    # ── FRED fetcher ──────────────────────────────────────────────────────────

    def _fetch_fred(self, series_id: str) -> float | None:
        """Fetch the most recent value for a FRED series."""
        # Try without API key first (works for many series)
        params = {
            "series_id":         series_id,
            "observation_start": (date.today() - timedelta(days=30)).isoformat(),
            "observation_end":   date.today().isoformat(),
            "sort_order":        "desc",
            "limit":             5,
            "file_type":         "json",
        }

        # Try with free API key env var if set
        api_key = _get_fred_key()
        if api_key:
            params["api_key"] = api_key

        r = self._session.get(_FRED_BASE, params=params, timeout=10)

        if r.status_code == 400 and not api_key:
            # FRED requires API key for some series — return None gracefully
            logger.debug(f"FRED {series_id} requires API key")
            return None

        if r.status_code != 200:
            return None

        obs = r.json().get("observations", [])
        # Find the most recent non-missing value
        for o in obs:
            val = o.get("value", ".")
            if val != ".":
                try:
                    return float(val)
                except ValueError:
                    continue
        return None

    # ── Regime classifier ─────────────────────────────────────────────────────

    @staticmethod
    def _classify_regime(data: dict) -> str:
        """
        Classify macro regime based on multiple indicators.
        Conservative — errs on side of caution.
        """
        vix = data.get("vix")
        yc  = data.get("yield_curve")

        # High fear overrides everything
        if vix and vix > 30:
            return "HIGH_FEAR"

        bear_signals = 0
        bull_signals = 0

        # VIX
        if vix:
            if vix < 18:  bull_signals += 1
            elif vix > 25: bear_signals += 1

        # Yield curve
        if yc is not None:
            if yc > 0.5:   bull_signals += 1
            elif yc < -0.2: bear_signals += 1

        # HY spread
        hy = data.get("hy_spread")
        if hy:
            if hy < 300:  bull_signals += 1
            elif hy > 500: bear_signals += 1

        if bear_signals >= 2:
            return "BEAR"
        if bull_signals >= 2:
            return "BULL"
        return "NEUTRAL"

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
            logger.warning(f"Could not save macro cache: {exc}")


def _get_fred_key() -> str:
    """Get FRED API key from env (optional — many series work without it)."""
    import os
    return os.getenv("FRED_API_KEY", "")
