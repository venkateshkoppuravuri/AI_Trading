"""
trading/alerts/telegram.py
───────────────────────────
Telegram bot for trade alerts and confirmations.

Features:
  • Instant trade alerts (BUY/SELL/HOLD) with full reasoning
  • Confirmation mode — bot asks you, you reply /confirm or /skip
  • Daily P&L summary at market close (1:30 AM IST)
  • /status command — see all positions and P&L
  • /portfolio command — full portfolio breakdown
  • /stop command — emergency stop the bot

Setup (one time):
  1. Open Telegram, message @BotFather → /newbot → copy the token
  2. Message your new bot once, then run:
       python -c "from trading.alerts.telegram import TelegramBot; TelegramBot.get_chat_id()"
  3. Add to .env:
       TELEGRAM_BOT_TOKEN=your_token
       TELEGRAM_CHAT_ID=your_chat_id

Usage:
    from trading.alerts.telegram import TelegramBot
    bot = TelegramBot()
    bot.send_buy_alert("NVDA", shares=3, price=450.0, reasoning="CFO bought $2M...")
    order_confirmed = bot.ask_confirmation("BUY NVDA", timeout_seconds=120)
"""

import os
import time
import threading
from datetime import datetime
from typing import Optional

import requests

from trading.config import get_settings
from trading.logger import get_logger
from trading.market import format_currency

logger = get_logger(__name__)

# ── Telegram API base ─────────────────────────────────────────────────────────
_TG_BASE = "https://api.telegram.org/bot{token}/{method}"


class TelegramBot:
    """
    Lightweight Telegram notifier — no async, no webhook, just clean HTTP calls.
    Uses long-polling only for confirmation replies (not a persistent listener).
    """

    def __init__(self) -> None:
        self._token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self._token and self._chat_id)

        if not self._enabled:
            logger.warning(
                "Telegram not configured — add TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID to .env to enable alerts."
            )
        else:
            logger.info("Telegram bot initialised ✓")

        # Track last update_id for polling
        self._last_update_id: int = 0
        self._pending_confirmations: dict[str, threading.Event] = {}
        self._confirmation_results:  dict[str, bool]            = {}

    # ── Public alert methods ──────────────────────────────────────────────────

    def send_buy_alert(
        self,
        ticker:     str,
        shares:     int,
        price:      float,
        reasoning:  str,
        confidence: str = "",   # "HIGH" | "MED" | "LOW" or empty
        source:     str = "strategy",
    ) -> None:
        """Send a BUY signal alert."""
        cost     = shares * price
        conf_str = f"  Confidence: {confidence}\n" if confidence else ""
        # Use plain text to avoid Markdown parse errors on reasoning strings
        msg = (
            f"BUY {ticker}\n\n"
            f"  Shares: {shares}\n"
            f"  Price:  {format_currency(price)}\n"
            f"  Cost:   {format_currency(cost)}\n"
            f"{conf_str}"
            f"  Source: {source}\n\n"
            f"Reasoning:\n{reasoning[:300]}"
        )
        self._send_message(msg)

    def send_sell_alert(
        self,
        ticker:    str,
        shares:    int,
        price:     float,
        pnl:       float,
        reasoning: str,
        urgency:   str = "NORMAL",
    ) -> None:
        """Send a SELL signal alert."""
        label = "SELL" if urgency != "HIGH" else "STOP-LOSS SELL"
        sign  = "+" if pnl >= 0 else ""
        msg = (
            f"{label} {ticker}\n\n"
            f"  Shares: {shares}\n"
            f"  Price:  {format_currency(price)}\n"
            f"  P/L:    {sign}{format_currency(pnl)}\n"
            f"  Reason: {reasoning}"
        )
        self._send_message(msg)

    def send_opportunity_alert(
        self,
        ticker:    str,
        reasoning: str,
        signals:   dict,
    ) -> None:
        """Send a new opportunity scan alert."""
        signal_lines = "\n".join(
            f"  • {k}: {v}" for k, v in signals.items()
        )
        msg = (
            f"💡 *NEW OPPORTUNITY — {ticker}*\n\n"
            f"*Signals:*\n{signal_lines}\n\n"
            f"*Reasoning:*\n{reasoning}"
        )
        self._send(msg)

    def send_daily_summary(
        self,
        portfolio_value: float,
        day_pnl:         float,
        day_pnl_pct:     float,
        positions:       list[dict],
        top_mover:       Optional[str] = None,
    ) -> None:
        """Send daily P&L summary at market close."""
        sign  = "+" if day_pnl >= 0 else ""
        emoji = "📈" if day_pnl >= 0 else "📉"

        pos_lines = ""
        for p in sorted(positions, key=lambda x: float(x.get("unrealized_pl", 0)), reverse=True)[:8]:
            sym    = p.get("symbol", "?")
            pl     = float(p.get("unrealized_pl", 0))
            pl_pct = float(p.get("unrealized_plpc", 0)) * 100
            s      = "+" if pl >= 0 else ""
            pos_lines += f"  {sym:<6} {s}{pl:>+7.2f}  ({s}{pl_pct:.1f}%)\n"

        msg = (
            f"{emoji} *Daily Summary — {datetime.now().strftime('%d %b %Y')}*\n\n"
            f"  Portfolio:  {format_currency(portfolio_value)}\n"
            f"  Day P/L:    {sign}{format_currency(day_pnl)} ({sign}{day_pnl_pct:.2f}%)\n\n"
            f"*Positions:*\n```\n{pos_lines}```"
        )
        if top_mover:
            msg += f"\n🏆 Top mover: *{top_mover}*"
        self._send(msg)

    def send_error_alert(self, component: str, error: str) -> None:
        """Send a critical error alert."""
        msg = (
            f"⚠️ *ERROR — {component}*\n\n"
            f"`{error[:500]}`\n\n"
            f"_{datetime.now().strftime('%H:%M:%S IST')}_"
        )
        self._send(msg)

    def send_bot_started(self) -> None:
        """Notify when the bot starts."""
        msg = (
            f"AI Trading Bot Started\n\n"
            f"  Time: {datetime.now().strftime('%d %b %Y %H:%M IST')}\n"
            f"  Strategies: Trailing Stop | Copy Trading | Wheel | AI Signal\n"
            f"  Market opens in ~10 min\n\n"
            f"Commands: /status /portfolio /stop"
        )
        self._send_message(msg)

    def send_bot_stopped(self) -> None:
        """Notify when the bot stops."""
        msg = (
            f"AI Trading Bot Stopped\n\n"
            f"  Time: {datetime.now().strftime('%d %b %Y %H:%M IST')}\n"
            f"  Market closed. See you tomorrow at 6:50 PM IST."
        )
        self._send_message(msg)

    def ask_confirmation(
        self,
        action:          str,
        timeout_seconds: int = 120,
    ) -> bool:
        """
        Ask user to confirm an action via Telegram.
        Returns True if confirmed, False if skipped or timed out.

        Sends: "Confirm BUY NVDA 3 shares? Reply /confirm or /skip (2 min timeout)"
        """
        if not self._enabled:
            return True  # Auto-confirm if Telegram not configured

        key = f"{action}_{int(time.time())}"
        event = threading.Event()
        self._pending_confirmations[key] = event
        self._confirmation_results[key]  = False

        msg = (
            f"⏳ *Confirmation Required*\n\n"
            f"Action: *{action}*\n\n"
            f"Reply /confirm to execute or /skip to pass\n"
            f"_(Times out in {timeout_seconds}s)_"
        )
        self._send(msg)

        # Start polling for reply in background
        t = threading.Thread(
            target=self._poll_for_reply,
            args=(key, event, timeout_seconds),
            daemon=True,
        )
        t.start()

        confirmed = event.wait(timeout=timeout_seconds + 5)
        result    = self._confirmation_results.pop(key, False)

        if not confirmed:
            self._send(f"⏰ *Timeout* — {action} skipped (no reply in {timeout_seconds}s)")

        self._pending_confirmations.pop(key, None)
        return result

    # ── Command listener ──────────────────────────────────────────────────────

    def _poll_for_reply(
        self,
        key:             str,
        event:           threading.Event,
        timeout_seconds: int,
    ) -> None:
        """Poll Telegram for /confirm or /skip reply."""
        deadline = time.time() + timeout_seconds
        while time.time() < deadline and not event.is_set():
            updates = self._get_updates()
            for update in updates:
                text = (
                    update.get("message", {})
                          .get("text", "")
                          .strip()
                          .lower()
                )
                if text in ("/confirm", "confirm", "yes", "y"):
                    self._confirmation_results[key] = True
                    event.set()
                    self._send("✅ *Confirmed* — executing trade")
                    return
                elif text in ("/skip", "skip", "no", "n"):
                    self._confirmation_results[key] = False
                    event.set()
                    self._send("⏭ *Skipped* — trade cancelled")
                    return
            time.sleep(2)

    # ── Core HTTP helpers ─────────────────────────────────────────────────────

    def _send_message(self, text: str) -> bool:
        """Send a plain-text message (no markdown). Safe for any string."""
        if not self._enabled:
            logger.debug(f"[Telegram disabled] {text[:100]}")
            return False
        try:
            url = _TG_BASE.format(token=self._token, method="sendMessage")
            resp = requests.post(
                url,
                json={"chat_id": self._chat_id, "text": text},
                timeout=10,
            )
            if not resp.ok:
                logger.warning(f"Telegram send failed: {resp.text[:200]}")
                return False
            return True
        except Exception as exc:
            logger.warning(f"Telegram error: {exc}")
            return False

    def _send(self, text: str) -> bool:
        """Send a Markdown message, falling back to plain text on parse error."""
        if not self._enabled:
            logger.debug(f"[Telegram disabled] {text[:100]}")
            return False
        try:
            url = _TG_BASE.format(token=self._token, method="sendMessage")
            resp = requests.post(
                url,
                json={
                    "chat_id":    self._chat_id,
                    "text":       text,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
            if resp.ok:
                return True
            # Parse error → retry as plain text
            if resp.status_code == 400:
                return self._send_message(text)
            logger.warning(f"Telegram send failed: {resp.text[:200]}")
            return False
        except Exception as exc:
            logger.warning(f"Telegram error: {exc}")
            return False

    def _get_updates(self) -> list[dict]:
        """Fetch new messages via long-polling."""
        try:
            url = _TG_BASE.format(token=self._token, method="getUpdates")
            resp = requests.get(
                url,
                params={"offset": self._last_update_id + 1, "timeout": 2},
                timeout=5,
            )
            if not resp.ok:
                return []
            updates = resp.json().get("result", [])
            if updates:
                self._last_update_id = updates[-1]["update_id"]
            return updates
        except Exception:
            return []

    # ── Setup helper ──────────────────────────────────────────────────────────

    @staticmethod
    def get_chat_id() -> None:
        """
        Run this once after messaging your bot to find your chat_id.
        Usage: python -c "from trading.alerts.telegram import TelegramBot; TelegramBot.get_chat_id()"
        """
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        if not token:
            print("ERROR: TELEGRAM_BOT_TOKEN not set in .env")
            return
        url  = _TG_BASE.format(token=token, method="getUpdates")
        resp = requests.get(url, timeout=10)
        data = resp.json()
        msgs = data.get("result", [])
        if not msgs:
            print("No messages found. Send any message to your bot first, then run this again.")
            return
        for m in msgs:
            chat = m.get("message", {}).get("chat", {})
            print(f"Chat ID: {chat.get('id')}  |  Name: {chat.get('first_name')} {chat.get('last_name', '')}")
        print("\nAdd the Chat ID to your .env as TELEGRAM_CHAT_ID=<id>")

    def test(self) -> bool:
        """Send a test message to verify configuration."""
        return self._send(
            "🧪 *AI Trading Bot — Test Message*\n\n"
            "Telegram alerts are configured correctly!\n"
            f"_{datetime.now().strftime('%d %b %Y %H:%M:%S')}_"
        )
