"""
trading/market.py
─────────────────
Market hours checks and shared formatting utilities.
All functions are pure and stateless.
"""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")


# ── Market hours ──────────────────────────────────────────────────────────────

def is_market_hours() -> bool:
    """Return True if NYSE / Nasdaq is currently open (local ET check, no API call)."""
    now = datetime.now(_ET)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    open_time  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_time <= now <= close_time


def next_expiry_range(min_dte: int, max_dte: int) -> tuple[str, str]:
    """
    Return an ISO-date range (min_date, max_date) for options contract filtering.

    Example:
        min_date, max_date = next_expiry_range(14, 28)
        # ("2026-04-25", "2026-05-09")
    """
    today    = date.today()
    min_date = (today + timedelta(days=min_dte)).isoformat()
    max_date = (today + timedelta(days=max_dte)).isoformat()
    return min_date, max_date


# ── Formatting helpers ────────────────────────────────────────────────────────

def format_currency(value: float) -> str:
    """Format a float as a dollar string, e.g. 1234.5 → '$1,234.50'."""
    return f"${value:,.2f}"


def pct_change(old: float, new: float) -> float:
    """Return percent change from *old* to *new*.  Returns 0.0 if old is zero."""
    if old == 0:
        return 0.0
    return ((new - old) / old) * 100
