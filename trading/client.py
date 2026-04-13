"""
trading/client.py
─────────────────
Alpaca REST API client with automatic retry + typed exceptions.

All public methods raise exceptions from trading.exceptions — callers never
see raw requests.HTTPError or requests.ConnectionError.

Retry policy (applied to every _get / _post / _delete call):
  • Retryable status codes  : 429, 500, 502, 503, 504
  • Non-retryable 4xx       : raised immediately as APIError / AuthenticationError
  • Max attempts            : settings.retry_max_attempts  (default 3)
  • Backoff                 : settings.retry_backoff_base ** attempt  seconds
    (attempt 1 → 2 s, attempt 2 → 4 s)
"""

import time
import functools
import logging
from typing import Any

import requests

from trading.config import get_settings
from trading.exceptions import (
    APIError,
    AuthenticationError,
    RateLimitError,
    OrderError,
    PriceUnavailableError,
)
from trading.logger import get_logger

logger = get_logger(__name__)

# HTTP status codes that warrant a retry
_RETRYABLE = {429, 500, 502, 503, 504}


# ── Retry decorator ───────────────────────────────────────────────────────────

def _with_retry(method):
    """
    Decorator applied to _get / _post / _delete.
    Retries on transient errors; raises typed APIError on final failure.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        settings = get_settings()
        last_exc: Exception | None = None

        for attempt in range(settings.retry_max_attempts):
            try:
                return method(self, *args, **kwargs)

            except requests.HTTPError as exc:
                status = exc.response.status_code
                body   = exc.response.text[:500]

                if status == 401:
                    raise AuthenticationError(
                        "Invalid API credentials", status_code=401, response_body=body
                    ) from exc
                if status == 403:
                    raise AuthenticationError(
                        f"Forbidden: {body}", status_code=403, response_body=body
                    ) from exc
                if status not in _RETRYABLE:
                    raise APIError(
                        f"API error: {body}", status_code=status, response_body=body
                    ) from exc

                # Retryable
                wait = settings.retry_backoff_base ** attempt
                logger.warning(
                    f"HTTP {status} on attempt {attempt + 1}/"
                    f"{settings.retry_max_attempts} — retrying in {wait:.1f}s"
                )
                last_exc = exc
                time.sleep(wait)

            except (requests.ConnectionError, requests.Timeout) as exc:
                wait = settings.retry_backoff_base ** attempt
                logger.warning(
                    f"Network error on attempt {attempt + 1}/"
                    f"{settings.retry_max_attempts}: {exc} — retrying in {wait:.1f}s"
                )
                last_exc = exc
                time.sleep(wait)

        # Exhausted all attempts
        raise APIError(f"Request failed after {settings.retry_max_attempts} attempts: {last_exc}") from last_exc

    return wrapper


# ── Client ────────────────────────────────────────────────────────────────────

class AlpacaClient:
    """
    Thin REST wrapper around the Alpaca Broker API.

    All methods are safe to call concurrently from different threads —
    the requests.Session is not shared (a new session is created per instance).
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._base  = settings.base_url.rstrip("/")
        self._data  = settings.data_url.rstrip("/")
        self._timeout = settings.request_timeout

        self._session = requests.Session()
        self._session.headers.update({
            "APCA-API-KEY-ID":     settings.api_key,
            "APCA-API-SECRET-KEY": settings.api_secret,
            "Content-Type":        "application/json",
            "Accept":              "application/json",
        })

    # ── Internal HTTP helpers ─────────────────────────────────────────────────

    @_with_retry
    def _get(self, path: str, params: dict | None = None, base: str | None = None) -> Any:
        url = (base or self._base) + path
        r   = self._session.get(url, params=params, timeout=self._timeout)
        r.raise_for_status()
        return r.json() if r.content else {}

    @_with_retry
    def _post(self, path: str, payload: dict) -> Any:
        r = self._session.post(self._base + path, json=payload, timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    @_with_retry
    def _delete(self, path: str) -> bool:
        r = self._session.delete(self._base + path, timeout=self._timeout)
        if r.status_code == 204:
            return True
        r.raise_for_status()
        return True

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        """Return full account info: status, portfolio_value, cash, buying_power …"""
        return self._get("/account")

    def get_buying_power(self) -> float:
        return float(self.get_account().get("buying_power", 0))

    def get_cash(self) -> float:
        return float(self.get_account().get("cash", 0))

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_positions(self) -> list[dict]:
        """Return all open positions."""
        return self._get("/positions")

    def get_position(self, symbol: str) -> dict | None:
        """Return position for *symbol*, or None if not held."""
        try:
            return self._get(f"/positions/{symbol}")
        except APIError as exc:
            if exc.status_code == 404:
                return None
            raise

    # ── Orders ────────────────────────────────────────────────────────────────

    def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        params: dict = {"status": "open", "limit": 100}
        if symbol:
            params["symbols"] = symbol
        return self._get("/orders", params=params)

    def get_order(self, order_id: str) -> dict:
        return self._get(f"/orders/{order_id}")

    def place_market_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        time_in_force: str = "day",
    ) -> dict:
        """Place a market order. side = 'buy' | 'sell'."""
        logger.info(f"MARKET {side.upper()} {qty}x {symbol}")
        try:
            return self._post("/orders", {
                "symbol":        symbol,
                "qty":           str(qty),
                "side":          side,
                "type":          "market",
                "time_in_force": time_in_force,
            })
        except APIError as exc:
            raise OrderError(
                f"Market {side} {qty}x {symbol} failed: {exc}", symbol=symbol
            ) from exc

    def place_limit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        limit_price: float,
        time_in_force: str = "day",
    ) -> dict:
        logger.info(f"LIMIT {side.upper()} {qty}x {symbol} @ ${limit_price:.2f}")
        try:
            return self._post("/orders", {
                "symbol":        symbol,
                "qty":           str(qty),
                "side":          side,
                "type":          "limit",
                "limit_price":   str(round(limit_price, 2)),
                "time_in_force": time_in_force,
            })
        except APIError as exc:
            raise OrderError(
                f"Limit {side} {qty}x {symbol} @ {limit_price} failed: {exc}", symbol=symbol
            ) from exc

    def place_stop_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        stop_price: float,
        time_in_force: str = "gtc",
    ) -> dict:
        logger.info(f"STOP {side.upper()} {qty}x {symbol} @ ${stop_price:.2f}")
        try:
            return self._post("/orders", {
                "symbol":        symbol,
                "qty":           str(qty),
                "side":          side,
                "type":          "stop",
                "stop_price":    str(round(stop_price, 2)),
                "time_in_force": time_in_force,
            })
        except APIError as exc:
            raise OrderError(
                f"Stop {side} {qty}x {symbol} @ {stop_price} failed: {exc}", symbol=symbol
            ) from exc

    def cancel_order(self, order_id: str) -> bool:
        logger.info(f"Cancelling order {order_id}")
        return self._delete(f"/orders/{order_id}")

    def cancel_all_orders(self) -> bool:
        logger.info("Cancelling all open orders")
        return self._delete("/orders")

    def cancel_orders_for_symbol(self, symbol: str) -> list[str]:
        """Cancel every open order for *symbol*. Returns list of cancelled IDs."""
        cancelled: list[str] = []
        for order in self.get_open_orders(symbol=symbol):
            try:
                self.cancel_order(order["id"])
                cancelled.append(order["id"])
            except (APIError, OrderError) as exc:
                logger.warning(f"Could not cancel order {order['id']}: {exc}")
        return cancelled

    # ── Price data ────────────────────────────────────────────────────────────

    def get_latest_price(self, symbol: str) -> float:
        """
        Return the latest trade price for *symbol*.
        Raises PriceUnavailableError if the price cannot be determined.
        """
        try:
            data = self._get(f"/stocks/{symbol}/trades/latest", base=self._data)
            price = data.get("trade", {}).get("p")
            if price is not None:
                return float(price)
        except APIError:
            pass

        # Fallback: quote mid-price
        try:
            data  = self._get(f"/stocks/{symbol}/quotes/latest", base=self._data)
            quote = data.get("quote", {})
            ask   = float(quote.get("ap") or 0)
            bid   = float(quote.get("bp") or 0)
            if ask and bid:
                return (ask + bid) / 2.0
        except APIError:
            pass

        raise PriceUnavailableError(
            f"Could not retrieve a price for {symbol}", symbol=symbol
        )

    # ── Options ───────────────────────────────────────────────────────────────

    def get_options_contracts(
        self,
        underlying_symbol: str,
        option_type: str | None = None,
        expiration_date_gte: str | None = None,
        expiration_date_lte: str | None = None,
        strike_price_gte: float | None = None,
        strike_price_lte: float | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Return options contracts for *underlying_symbol*.
        option_type: 'call' | 'put'
        """
        params: dict = {
            "underlying_symbols": underlying_symbol,
            "status": "active",
            "limit":  limit,
        }
        if option_type:
            params["type"] = option_type
        if expiration_date_gte:
            params["expiration_date_gte"] = expiration_date_gte
        if expiration_date_lte:
            params["expiration_date_lte"] = expiration_date_lte
        if strike_price_gte is not None:
            params["strike_price_gte"] = str(strike_price_gte)
        if strike_price_lte is not None:
            params["strike_price_lte"] = str(strike_price_lte)

        result = self._get("/options/contracts", params=params)
        return result.get("option_contracts", [])

    def get_option_quote(self, contract_symbol: str) -> dict | None:
        """
        Return the latest quote for an options contract, or None on failure.
        Public API — replaces the previous private _get() call in WheelStrategy.
        """
        try:
            data = self._get(
                f"/stocks/{contract_symbol}/quotes/latest",
                base=self._data,
            )
            return data.get("quote")
        except APIError as exc:
            logger.debug(f"Option quote unavailable for {contract_symbol}: {exc}")
            return None

    def place_options_order(
        self,
        symbol: str,
        qty: int,
        position_intent: str,
        order_type: str = "market",
        limit_price: float | None = None,
        time_in_force: str = "day",
    ) -> dict:
        """
        Place an options order.

        position_intent: 'buy_to_open' | 'sell_to_open' | 'buy_to_close' | 'sell_to_close'

        Alpaca requires side='buy'|'sell' as a separate field from position_intent.
        This method derives side automatically from position_intent.
        """
        side = "buy" if position_intent.startswith("buy") else "sell"
        logger.info(f"OPTIONS {position_intent.upper()} {qty}x {symbol}")
        payload: dict = {
            "symbol":          symbol,
            "qty":             str(qty),
            "side":            side,
            "type":            order_type,
            "time_in_force":   time_in_force,
            "asset_class":     "us_option",
            "position_intent": position_intent,
        }
        if limit_price is not None:
            payload["limit_price"] = str(round(limit_price, 2))
        try:
            return self._post("/orders", payload)
        except APIError as exc:
            raise OrderError(
                f"Options {position_intent} {qty}x {symbol} failed: {exc}", symbol=symbol
            ) from exc

    # ── Market clock ──────────────────────────────────────────────────────────

    def get_clock(self) -> dict:
        """Return market clock: is_open, next_open, next_close."""
        return self._get("/clock")

    def is_market_open(self) -> bool:
        """API-based market open check (authoritative but uses a network call)."""
        try:
            return bool(self.get_clock().get("is_open", False))
        except APIError:
            return False
