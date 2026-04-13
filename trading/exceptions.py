"""
trading/exceptions.py
─────────────────────
Custom exception hierarchy for the trading bot.
All exceptions derive from TradingError so callers can catch broadly
or narrowly depending on context.
"""


class TradingError(Exception):
    """Base class for all trading bot errors."""

    def __init__(self, message: str, symbol: str | None = None) -> None:
        super().__init__(message)
        self.symbol = symbol


class ConfigurationError(TradingError):
    """Missing or invalid environment / configuration value."""


class APIError(TradingError):
    """Wraps an HTTP error returned by the Alpaca API."""

    def __init__(self, message: str, status_code: int = 0,
                 response_body: str = "", symbol: str | None = None) -> None:
        super().__init__(message, symbol)
        self.status_code = status_code
        self.response_body = response_body

    def __str__(self) -> str:
        return f"{super().__str__()} (HTTP {self.status_code})"


class RateLimitError(APIError):
    """HTTP 429 — Alpaca rate limit exceeded."""


class AuthenticationError(APIError):
    """HTTP 401 / 403 — invalid API credentials or forbidden action."""


class MarketClosedError(TradingError):
    """Action attempted outside of market hours."""


class InsufficientFundsError(TradingError):
    """Not enough buying power to place the requested order."""


class OrderError(TradingError):
    """Order placement or cancellation failed."""


class PriceUnavailableError(TradingError):
    """Could not retrieve a live price for the given symbol."""


class ScraperError(TradingError):
    """Capitol Trades (or other data source) fetch/parse failed."""
