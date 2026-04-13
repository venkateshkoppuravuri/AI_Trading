"""
trading/strategies/wheel.py
────────────────────────────
The Wheel Strategy — systematic premium income from options.

  STAGE 1 — Sell Cash-Secured Put (CSP):
    Strike ~10% below current price, 2–4 weeks to expiry.
    Collect premium. Repeat until assigned.
    On assignment → buy shares at strike → move to Stage 2.

  STAGE 2 — Sell Covered Call (CC):
    Strike ~10% above cost basis, 2–4 weeks to expiry.
    Collect premium. Repeat until called away.
    On assignment → shares sold at profit → back to Stage 1.

  Early close rule: buy-to-close any contract at 50% profit to free capital.
  Daily summary: printed at 3:55pm ET.
"""

import json
from datetime import datetime

from trading.client import AlpacaClient
from trading.config import get_settings
from trading.exceptions import OrderError, PriceUnavailableError
from trading.logger import get_logger
from trading.market import format_currency, next_expiry_range, pct_change
from trading.strategies.base import BaseStrategy

logger = get_logger(__name__)


class WheelStrategy(BaseStrategy):
    def __init__(
        self,
        symbol: str = "TSLA",
        strike_pct_below: float = 10.0,
        strike_pct_above: float = 10.0,
        min_dte: int = 14,
        max_dte: int = 28,
        early_close_pct: float = 50.0,
        contracts: int = 1,
    ) -> None:
        self.symbol           = symbol.upper()
        self.strike_pct_below = strike_pct_below
        self.strike_pct_above = strike_pct_above
        self.min_dte          = min_dte
        self.max_dte          = max_dte
        self.early_close_pct  = early_close_pct
        self.contracts        = contracts
        self.shares_per_contract = 100

        self._client     = AlpacaClient()
        self._state_file = get_settings().state_dir / f"wheel_{self.symbol}.json"
        self._state      = self._load_state()

    # ── BaseStrategy interface ─────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"Wheel[{self.symbol}]"

    def run(self) -> None:
        self._state["last_run"] = datetime.now().isoformat()
        stage = self._state["stage"]
        logger.info(f"{self.name}: stage={stage}")

        match stage:
            case "STAGE_1":
                self._sell_put()
            case "WAITING_PUT":
                self._monitor_contract()
            case "STAGE_2":
                self._sell_call()
            case "WAITING_CALL":
                self._monitor_contract()
            case "IDLE":
                logger.info(f"{self.name}: Idle. Set stage to STAGE_1 to begin.")
            case _:
                logger.warning(f"{self.name}: Unknown stage '{stage}'")

        self._save_state()

        # Daily summary at 3:55pm ET
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        if now.hour == 15 and now.minute >= 55 and self._state.get("daily_summary_sent") != today:
            self._log_summary()
            self._state["daily_summary_sent"] = today
            self._save_state()

    def status(self) -> dict:
        s = self._state
        return {
            "strategy":        self.name,
            "stage":           s["stage"],
            "contract":        s.get("current_contract"),
            "contract_side":   s.get("contract_side"),
            "cost_basis":      s.get("cost_basis"),
            "shares_held":     s.get("shares_held", 0),
            "total_premium":   s.get("total_premium", 0.0),
            "cycles":          s.get("cycles", 0),
            "last_run":        s.get("last_run"),
        }

    # ── State helpers ──────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if self._state_file.exists():
            with open(self._state_file) as f:
                return json.load(f)
        return {
            "stage":              "STAGE_1",
            "current_contract":   None,
            "contract_side":      None,
            "premium_collected":  0.0,
            "cost_basis":         None,
            "shares_held":        0,
            "total_premium":      0.0,
            "cycles":             0,
            "last_run":           None,
            "daily_summary_sent": "",
        }

    def _save_state(self) -> None:
        with open(self._state_file, "w") as f:
            json.dump(self._state, f, indent=2)

    # ── Stage 1: Sell Cash-Secured Put ────────────────────────────────────────

    def _sell_put(self) -> None:
        try:
            price = self._client.get_latest_price(self.symbol)
        except PriceUnavailableError as exc:
            logger.error(f"{self.name}: {exc}")
            return

        target_strike = round(price * (1 - self.strike_pct_below / 100), 2)
        cash_needed   = target_strike * self.shares_per_contract * self.contracts

        if self._client.get_cash() < cash_needed:
            logger.warning(
                f"{self.name}: Insufficient cash for CSP — need "
                f"{format_currency(cash_needed)}"
            )
            return

        contract = self._find_contract("put", target_strike)
        if not contract:
            return

        symbol = contract.get("symbol", "")
        strike = float(contract.get("strike_price", target_strike))
        logger.info(
            f"{self.name}: STAGE 1 | Selling CSP | "
            f"strike={format_currency(strike)} | contract={symbol} | "
            f"expiry={contract.get('expiration_date')}"
        )

        premium = self._mid_price(symbol) or 0.0
        try:
            order = self._client.place_options_order(symbol, self.contracts, position_intent="sell_to_open")
            total_premium = premium * self.shares_per_contract * self.contracts
            self._state.update({
                "stage":             "WAITING_PUT",
                "current_contract":  symbol,
                "contract_side":     "put",
                "premium_collected": total_premium,
            })
            self._state["total_premium"] += total_premium
            logger.info(
                f"{self.name}: Put sold | est. premium={format_currency(total_premium)} | "
                f"order={order.get('id')}"
            )
        except OrderError as exc:
            logger.error(f"{self.name}: Failed to sell put — {exc}")

    # ── Stage 2: Sell Covered Call ────────────────────────────────────────────

    def _sell_call(self) -> None:
        cost_basis = self._state.get("cost_basis")
        if not cost_basis:
            pos = self._client.get_position(self.symbol)
            if not pos:
                logger.warning(f"{self.name}: No position for covered call")
                return
            cost_basis = float(pos.get("avg_entry_price", 0))
            self._state["cost_basis"]  = cost_basis
            self._state["shares_held"] = int(float(pos.get("qty", 0)))

        target_strike = round(cost_basis * (1 + self.strike_pct_above / 100), 2)
        # Safety: never sell call below cost basis
        if target_strike <= cost_basis:
            target_strike = round(cost_basis * 1.05, 2)
            logger.warning(f"{self.name}: Adjusted call strike to {format_currency(target_strike)}")

        contract = self._find_contract("call", target_strike)
        if not contract:
            return

        symbol = contract.get("symbol", "")
        strike = float(contract.get("strike_price", target_strike))
        logger.info(
            f"{self.name}: STAGE 2 | Selling CC | "
            f"cost_basis={format_currency(cost_basis)} | "
            f"strike={format_currency(strike)} | contract={symbol} | "
            f"expiry={contract.get('expiration_date')}"
        )

        premium = self._mid_price(symbol) or 0.0
        try:
            order = self._client.place_options_order(symbol, self.contracts, position_intent="sell_to_open")
            total_premium = premium * self.shares_per_contract * self.contracts
            self._state.update({
                "stage":             "WAITING_CALL",
                "current_contract":  symbol,
                "contract_side":     "call",
                "premium_collected": total_premium,
            })
            self._state["total_premium"] += total_premium
            logger.info(
                f"{self.name}: Call sold | est. premium={format_currency(total_premium)} | "
                f"order={order.get('id')}"
            )
        except OrderError as exc:
            logger.error(f"{self.name}: Failed to sell call — {exc}")

    # ── Monitor open contract ─────────────────────────────────────────────────

    def _monitor_contract(self) -> None:
        symbol = self._state.get("current_contract")
        side   = self._state.get("contract_side")
        if not symbol:
            return

        current_price = self._mid_price(symbol)
        sold_price    = (
            self._state.get("premium_collected", 0)
            / (self.shares_per_contract * self.contracts)
            if self.shares_per_contract * self.contracts
            else 0
        )

        if sold_price and current_price is not None:
            profit_pct = ((sold_price - current_price) / sold_price) * 100
            logger.info(
                f"{self.name}: Contract {symbol} | sold@{format_currency(sold_price)} | "
                f"now@{format_currency(current_price)} | profit={profit_pct:.1f}%"
            )
            if profit_pct >= self.early_close_pct:
                logger.info(f"{self.name}: 50% profit target reached — closing early")
                self._close_early(symbol, side, current_price)
                return

        self._check_assignment(side)

    def _close_early(self, symbol: str, side: str, current_price: float) -> None:
        try:
            self._client.place_options_order(symbol, self.contracts, position_intent="buy_to_close")
            logger.info(f"{self.name}: Closed early | {symbol}")
            next_stage = "STAGE_1" if side == "put" else "STAGE_2"
            self._state.update({"stage": next_stage, "current_contract": None})
        except OrderError as exc:
            logger.error(f"{self.name}: Early close failed — {exc}")

    def _check_assignment(self, side: str | None) -> None:
        pos = self._client.get_position(self.symbol)

        if side == "put" and self._state["stage"] == "WAITING_PUT":
            if pos and int(float(pos.get("qty", 0))) >= self.shares_per_contract:
                cost_basis = float(pos.get("avg_entry_price", 0))
                logger.info(
                    f"{self.name}: PUT ASSIGNED | "
                    f"now own {pos.get('qty')} shares @ {format_currency(cost_basis)}"
                )
                self._state.update({
                    "stage":            "STAGE_2",
                    "current_contract": None,
                    "cost_basis":       cost_basis,
                    "shares_held":      int(float(pos.get("qty", 0))),
                })

        elif side == "call" and self._state["stage"] == "WAITING_CALL":
            if not pos or int(float(pos.get("qty", 0))) == 0:
                logger.info(f"{self.name}: CALL EXERCISED | shares called away — wheel cycle complete")
                self._state.update({
                    "stage":            "STAGE_1",
                    "current_contract": None,
                    "cost_basis":       None,
                    "shares_held":      0,
                    "cycles":           self._state.get("cycles", 0) + 1,
                })

    # ── Options helpers ───────────────────────────────────────────────────────

    def _find_contract(self, option_type: str, target_strike: float) -> dict | None:
        date_min, date_max = next_expiry_range(self.min_dte, self.max_dte)
        contracts = self._client.get_options_contracts(
            underlying_symbol   = self.symbol,
            option_type         = option_type,
            expiration_date_gte = date_min,
            expiration_date_lte = date_max,
            strike_price_gte    = round(target_strike * 0.90, 2),
            strike_price_lte    = round(target_strike * 1.10, 2),
            limit               = 20,
        )
        if not contracts:
            logger.warning(
                f"{self.name}: No {option_type} contracts found near "
                f"{format_currency(target_strike)}"
            )
            return None
        return min(contracts, key=lambda c: abs(float(c.get("strike_price", 0)) - target_strike))

    def _mid_price(self, contract_symbol: str) -> float | None:
        quote = self._client.get_option_quote(contract_symbol)
        if not quote:
            return None
        ask = float(quote.get("ap") or 0)
        bid = float(quote.get("bp") or 0)
        if ask and bid:
            return round((ask + bid) / 2, 2)
        return None

    # ── Summary ───────────────────────────────────────────────────────────────

    def _log_summary(self) -> None:
        s = self._state
        try:
            acc = self._client.get_account()
            portfolio_value = float(acc.get("portfolio_value", 0))
        except Exception:
            portfolio_value = 0.0

        logger.info(
            f"\n{self.name} — Daily Summary\n"
            f"  Stage            : {s['stage']}\n"
            f"  Open Contract    : {s.get('current_contract', 'None')}\n"
            f"  Cost Basis/Share : {format_currency(s.get('cost_basis') or 0)}\n"
            f"  Shares Held      : {s.get('shares_held', 0)}\n"
            f"  Cycles Completed : {s.get('cycles', 0)}\n"
            f"  Total Premium    : {format_currency(s.get('total_premium', 0.0))}\n"
            f"  Portfolio Value  : {format_currency(portfolio_value)}"
        )
