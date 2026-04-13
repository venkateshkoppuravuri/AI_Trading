"""
trading/strategies/trailing_stop.py
─────────────────────────────────────
Trailing Stop Strategy:

  1. Buy *initial_shares* at market price on first run.
  2. Stop-loss floor: if price drops *stop_loss_pct*% from entry → sell all.
  3. Trailing floor: once price rises *trail_activate_pct*% above entry,
     set floor = current_high × (1 − trail_distance_pct / 100).
     The floor only moves UP, never down.
  4. Ladder buys: buy extra shares on dips below entry
     (configurable levels, default −20% → 10 shares, −30% → 20 shares).

State is persisted to state/trailing_stop_{symbol}.json so it survives restarts.
"""

import json
from datetime import datetime

from trading.client import AlpacaClient
from trading.config import get_settings
from trading.exceptions import OrderError, PriceUnavailableError
from trading.logger import get_logger
from trading.market import format_currency, pct_change
from trading.strategies.base import BaseStrategy

logger = get_logger(__name__)


class TrailingStopStrategy(BaseStrategy):
    def __init__(
        self,
        symbol: str,
        initial_shares: int = 10,
        stop_loss_pct: float = 10.0,
        trail_activate_pct: float = 10.0,
        trail_distance_pct: float = 5.0,
        ladder_levels: list[tuple[float, int]] | None = None,
    ) -> None:
        self.symbol              = symbol.upper()
        self.initial_shares      = initial_shares
        self.stop_loss_pct       = stop_loss_pct
        self.trail_activate_pct  = trail_activate_pct
        self.trail_distance_pct  = trail_distance_pct
        self.ladder_levels       = ladder_levels or [(20.0, 10), (30.0, 20)]

        self._client     = AlpacaClient()
        self._state_file = get_settings().state_dir / f"trailing_stop_{self.symbol}.json"
        self._state      = self._load_state()

    # ── BaseStrategy interface ─────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"TrailingStop[{self.symbol}]"

    def run(self) -> None:
        """Enter on IDLE, monitor on IN_TRADE."""
        current = self._state["status"]
        if current == "IDLE":
            self._enter_trade()
        elif current == "IN_TRADE":
            self._monitor()
        else:
            logger.info(f"{self.name}: status={current}. Reset state file to restart.")

    def status(self) -> dict:
        s = self._state
        payload: dict = {
            "strategy": self.name,
            "status":   s["status"],
        }
        if s["status"] == "IN_TRADE":
            try:
                price = self._client.get_latest_price(self.symbol)
            except PriceUnavailableError:
                price = None
            payload.update({
                "current_price":   price,
                "entry_price":     s["entry_price"],
                "highest_price":   s["highest_price"],
                "floor_price":     s["floor_price"],
                "trailing_active": s["trailing_active"],
                "total_shares":    s["total_shares"],
                "ladders_hit":     s["ladder_triggered"],
                "unrealised_pct":  (
                    round(pct_change(s["entry_price"], price), 2)
                    if price and s["entry_price"] else None
                ),
            })
        return payload

    # ── State helpers ──────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if self._state_file.exists():
            with open(self._state_file) as f:
                return json.load(f)
        return {
            "status":           "IDLE",
            "entry_price":      None,
            "entry_shares":     0,
            "highest_price":    None,
            "floor_price":      None,
            "trailing_active":  False,
            "ladder_triggered": [],
            "total_shares":     0,
            "last_updated":     None,
        }

    def _save_state(self) -> None:
        self._state["last_updated"] = datetime.now().isoformat()
        with open(self._state_file, "w") as f:
            json.dump(self._state, f, indent=2)

    # ── Trade lifecycle ────────────────────────────────────────────────────────

    def _enter_trade(self) -> None:
        try:
            price = self._client.get_latest_price(self.symbol)
        except PriceUnavailableError as exc:
            logger.error(f"{self.name}: Cannot enter — {exc}")
            return

        logger.info(
            f"{self.name}: Entering | BUY {self.initial_shares}x {self.symbol} "
            f"@ {format_currency(price)}"
        )
        try:
            self._client.place_market_order(self.symbol, self.initial_shares, "buy")
        except OrderError as exc:
            logger.error(f"{self.name}: Entry order failed — {exc}")
            return

        floor = round(price * (1 - self.stop_loss_pct / 100), 2)
        self._state.update({
            "status":           "IN_TRADE",
            "entry_price":      price,
            "entry_shares":     self.initial_shares,
            "highest_price":    price,
            "floor_price":      floor,
            "trailing_active":  False,
            "ladder_triggered": [],
            "total_shares":     self.initial_shares,
        })
        self._save_state()
        logger.info(
            f"{self.name}: Trade entered | entry={format_currency(price)} | "
            f"floor={format_currency(floor)}"
        )

    def _monitor(self) -> None:
        try:
            price = self._client.get_latest_price(self.symbol)
        except PriceUnavailableError as exc:
            logger.warning(f"{self.name}: {exc} — skipping cycle")
            return

        entry        = self._state["entry_price"]
        floor        = self._state["floor_price"]
        highest      = self._state["highest_price"]
        total_shares = self._state["total_shares"]
        change_pct   = pct_change(entry, price)

        logger.info(
            f"{self.name}: price={format_currency(price)} | "
            f"entry={format_currency(entry)} | chg={change_pct:+.2f}% | "
            f"floor={format_currency(floor)} | shares={total_shares}"
        )

        # 1. Floor hit → exit
        if price <= floor:
            logger.warning(
                f"{self.name}: Floor hit! {format_currency(price)} <= {format_currency(floor)}"
            )
            self._exit_trade(price, reason="Stop-loss / trailing floor")
            return

        # 2. Track new high
        if price > highest:
            self._state["highest_price"] = price
            highest = price

        # 3. Trail the floor upward
        gain_from_entry = pct_change(entry, highest)
        if gain_from_entry >= self.trail_activate_pct:
            new_floor = round(highest * (1 - self.trail_distance_pct / 100), 2)
            if new_floor > floor:
                logger.info(
                    f"{self.name}: Trail floor raised "
                    f"{format_currency(floor)} -> {format_currency(new_floor)}"
                )
                self._state["floor_price"]    = new_floor
                self._state["trailing_active"] = True

        # 4. Ladder buys on dip
        drop_pct = pct_change(entry, price)
        for (level, extra) in self.ladder_levels:
            if drop_pct <= -level and level not in self._state["ladder_triggered"]:
                logger.info(
                    f"{self.name}: Ladder buy at -{level}% | "
                    f"buying {extra} shares @ {format_currency(price)}"
                )
                try:
                    self._client.place_market_order(self.symbol, extra, "buy")
                    self._state["ladder_triggered"].append(level)
                    self._state["total_shares"] += extra
                except OrderError as exc:
                    logger.error(f"{self.name}: Ladder buy failed — {exc}")

        self._save_state()

    def _exit_trade(self, price: float, reason: str = "") -> None:
        shares = self._state["total_shares"]
        entry  = self._state["entry_price"]
        pnl    = (price - entry) * shares

        try:
            self._client.place_market_order(self.symbol, shares, "sell")
        except OrderError as exc:
            logger.error(f"{self.name}: Exit order failed — {exc}")
            return

        logger.info(
            f"{self.name}: CLOSED | reason={reason} | "
            f"entry={format_currency(entry)} | exit={format_currency(price)} | "
            f"shares={shares} | P/L={format_currency(pnl)} "
            f"({pct_change(entry, price):+.2f}%)"
        )
        self._state.update({
            "status":          "STOPPED",
            "entry_price":     None,
            "highest_price":   None,
            "floor_price":     None,
            "trailing_active": False,
            "total_shares":    0,
        })
        self._save_state()
