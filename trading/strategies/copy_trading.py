"""
trading/strategies/copy_trading.py
────────────────────────────────────
Copy Trading Strategy — mirrors stock trades of the TOP-N most active US
politicians as reported through public STOCK Act disclosures.

  • On each run: fetch the top-N active traders, copy any NEW disclosed trades.
  • Budget is split evenly across the N politicians (e.g. $333 each if $1,000 / 3).
  • Each politician gets their own isolated position bucket so P/L is tracked
    per-politician and max_positions is enforced independently.
  • BUY  → buy proportionally from that politician's budget slice.
  • SELL → sell our position if we hold the stock under that politician's bucket.
  • Each trade is fingerprinted to prevent double-execution.

State migration: automatically upgrades the old single-politician JSON format
to the new multi-politician format on first load.
"""

import json
from datetime import datetime

from trading.client import AlpacaClient
from trading.config import get_settings
from trading.data.capitol_trades import CapitolTradesScraper
from trading.exceptions import OrderError, PriceUnavailableError, ScraperError
from trading.logger import get_logger
from trading.market import format_currency
from trading.strategies.base import BaseStrategy

logger = get_logger(__name__)


class CopyTradingStrategy(BaseStrategy):
    def __init__(
        self,
        trade_budget: float = 1_000.0,
        max_positions: int = 10,
        top_n: int = 3,
        scraper: CapitolTradesScraper | None = None,
    ) -> None:
        self.trade_budget   = trade_budget
        self.max_positions  = max_positions          # per-politician
        self.top_n          = top_n
        self.budget_per_pol = round(trade_budget / top_n, 2)

        self._client     = AlpacaClient()
        self._scraper    = scraper or CapitolTradesScraper()
        self._state_file = get_settings().state_dir / "copy_trading.json"
        self._state      = self._load_state()

    # ── BaseStrategy interface ─────────────────────────────────────────────────

    @property
    def name(self) -> str:
        following = self._state.get("following") or []
        if following:
            last_names = ", ".join(p.split()[-1] for p in following[:3])
            return f"CopyTrading[{last_names}]"
        return "CopyTrading[?]"

    def run(self) -> None:
        logger.info(f"{self.name}: Starting run (top_n={self.top_n}, budget_per_pol=${self.budget_per_pol})")
        self._state["last_run"] = datetime.now().isoformat()

        try:
            pol_trades_list = self._scraper.get_top_n_politician_trades(self.top_n)
        except ScraperError as exc:
            logger.error(f"{self.name}: Scraper failed — {exc}")
            self._save_state()
            return

        if not pol_trades_list:
            logger.warning(f"{self.name}: No politicians identified — aborting")
            self._save_state()
            return

        new_following = [pol for pol, _ in pol_trades_list]
        old_following = self._state.get("following") or []
        if new_following != old_following:
            logger.info(f"{self.name}: Following updated: {new_following} (was: {old_following})")
            self._state["following"] = new_following

        # Ensure every politician has a state bucket
        for pol in new_following:
            if pol not in self._state["politicians"]:
                self._state["politicians"][pol] = {"positions": {}, "seen_trades": []}

        try:
            buying_power = self._client.get_buying_power()
        except Exception as exc:
            logger.error(f"{self.name}: Cannot read buying power — {exc}")
            return

        total_new = 0
        for politician, trades in pol_trades_list:
            pol_state  = self._state["politicians"][politician]
            new_count  = 0
            pol_max    = max(1, self.max_positions // self.top_n)

            if not trades:
                logger.info(f"{self.name}: No trades found for {politician}")
                continue

            for trade in trades:
                fp = self._fingerprint(trade)
                if fp in pol_state["seen_trades"]:
                    continue

                ticker     = trade.get("ticker", "").strip().upper()
                trade_type = trade.get("trade_type", "").lower()

                # Skip invalid or non-stock instruments
                if not ticker or len(ticker) > 5 or "/" in ticker or "option" in ticker.lower():
                    pol_state["seen_trades"].append(fp)
                    continue

                if "buy" in trade_type or "purchase" in trade_type:
                    # Cross-politician deduplication: skip if ANY other politician
                    # already holds or has ordered this ticker — prevents duplicate
                    # orders when multiple politicians trade the same stock (e.g.
                    # both Gottheimer and Pelosi buy GOOGL).
                    already_held = any(
                        ticker in self._state["politicians"].get(other_pol, {}).get("positions", {})
                        for other_pol in self._state["politicians"]
                        if other_pol != politician
                    )
                    if already_held:
                        logger.info(
                            f"{self.name}: [{politician}] Skipping {ticker} — "
                            f"already held by another politician's bucket"
                        )
                        pol_state["seen_trades"].append(fp)
                        continue

                    self._copy_buy(
                        politician, ticker, trade,
                        buying_power, self.budget_per_pol,
                        pol_max, pol_state,
                    )
                    new_count += 1
                elif "sell" in trade_type or "sale" in trade_type:
                    self._copy_sell(politician, ticker, trade, pol_state)
                    new_count += 1

                pol_state["seen_trades"].append(fp)

            total_new += new_count
            logger.info(
                f"{self.name}: [{politician}] {new_count} new trade(s) | "
                f"positions: {len(pol_state['positions'])}"
            )

        logger.info(
            f"{self.name}: Run complete — {total_new} total new trade(s) across "
            f"{len(new_following)} politicians"
        )
        self._save_state()

    def status(self) -> dict:
        following    = self._state.get("following") or []
        politicians  = self._state.get("politicians", {})

        per_pol = {}
        for pol in following:
            bucket = politicians.get(pol, {})
            tickers = list(bucket.get("positions", {}).keys())
            per_pol[pol] = {
                "positions":   len(tickers),
                "tickers":     tickers,
                "seen_trades": len(bucket.get("seen_trades", [])),
            }

        total_positions = sum(d["positions"] for d in per_pol.values())

        return {
            "strategy":        self.name,
            "following":       following,
            "top_n":           self.top_n,
            "budget_per_pol":  self.budget_per_pol,
            "total_positions": total_positions,
            "per_politician":  per_pol,
            "last_run":        self._state.get("last_run"),
        }

    # ── State helpers ──────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if self._state_file.exists():
            with open(self._state_file) as f:
                data = json.load(f)

            # ── Migrate old single-politician format ───────────────────────────
            if "positions" in data and not isinstance(data.get("following"), list):
                old_pol  = data.get("following") or "Unknown"
                old_pos  = data.get("positions", {})
                old_seen = data.get("seen_trades", [])
                data = {
                    "following":   [old_pol] if old_pol != "Unknown" else [],
                    "politicians": {
                        old_pol: {"positions": old_pos, "seen_trades": old_seen}
                    } if old_pol != "Unknown" else {},
                    "last_run": data.get("last_run"),
                }
                logger.info("Migrated copy_trading state → multi-politician format")

            return data

        return {"following": [], "politicians": {}, "last_run": None}

    def _save_state(self) -> None:
        with open(self._state_file, "w") as f:
            json.dump(self._state, f, indent=2)

    @staticmethod
    def _fingerprint(trade: dict) -> str:
        return "|".join([
            trade.get("politician", ""),
            trade.get("ticker", ""),
            trade.get("traded_date", ""),
            trade.get("trade_type", ""),
            trade.get("trade_size", ""),
        ])

    # ── Trade execution ────────────────────────────────────────────────────────

    def _copy_buy(
        self,
        politician: str,
        ticker: str,
        trade: dict,
        buying_power: float,
        budget: float,
        max_pos: int,
        pol_state: dict,
    ) -> None:
        if len(pol_state["positions"]) >= max_pos:
            logger.warning(
                f"{self.name}: [{politician}] Max positions ({max_pos}) reached — skipping {ticker}"
            )
            return

        if ticker in pol_state["positions"]:
            logger.info(f"{self.name}: [{politician}] Already holding {ticker} — skipping")
            return

        try:
            price = self._client.get_latest_price(ticker)
        except PriceUnavailableError as exc:
            logger.warning(f"{self.name}: [{politician}] {exc} — skipping {ticker}")
            return

        effective_budget = min(budget, buying_power * 0.9)
        shares = max(1, int(effective_budget // price))
        cost   = shares * price

        if cost > buying_power:
            logger.warning(
                f"{self.name}: [{politician}] Insufficient buying power for {ticker} — "
                f"need {format_currency(cost)}, have {format_currency(buying_power)}"
            )
            return

        logger.info(
            f"{self.name}: [{politician}] Copying BUY {ticker} "
            f"(filed: {trade.get('filed_date', '?')}) → "
            f"{shares} shares @ ~{format_currency(price)}"
        )
        try:
            order = self._client.place_market_order(ticker, shares, "buy")
            pol_state["positions"][ticker] = {
                "shares":      shares,
                "avg_price":   price,
                "copied_from": politician,
                "copied_date": datetime.now().isoformat(),
                "order_id":    order.get("id"),
            }
            logger.info(
                f"{self.name}: [{politician}] Bought {shares}x {ticker} | "
                f"order={order.get('id')}"
            )
        except OrderError as exc:
            logger.error(f"{self.name}: [{politician}] Buy failed for {ticker} — {exc}")

    def _copy_sell(
        self,
        politician: str,
        ticker: str,
        trade: dict,
        pol_state: dict,
    ) -> None:
        if ticker not in pol_state["positions"]:
            logger.info(f"{self.name}: [{politician}] Not holding {ticker} — no sell needed")
            return

        alpaca_pos = self._client.get_position(ticker)
        if not alpaca_pos:
            logger.info(
                f"{self.name}: [{politician}] No Alpaca position in {ticker} — removing from state"
            )
            del pol_state["positions"][ticker]
            return

        shares = int(float(alpaca_pos.get("qty", 0)))
        try:
            price = self._client.get_latest_price(ticker)
        except PriceUnavailableError:
            price = pol_state["positions"][ticker].get("avg_price", 0)

        cost_basis = pol_state["positions"][ticker].get("avg_price", 0) * shares
        sale_value = price * shares

        logger.info(
            f"{self.name}: [{politician}] Copying SELL {ticker} "
            f"(filed: {trade.get('filed_date', '?')}) → "
            f"{shares} shares @ ~{format_currency(price)}"
        )
        try:
            self._client.place_market_order(ticker, shares, "sell")
            logger.info(
                f"{self.name}: [{politician}] Sold {shares}x {ticker} | "
                f"P/L: {format_currency(sale_value - cost_basis)}"
            )
            del pol_state["positions"][ticker]
        except OrderError as exc:
            logger.error(f"{self.name}: [{politician}] Sell failed for {ticker} — {exc}")
