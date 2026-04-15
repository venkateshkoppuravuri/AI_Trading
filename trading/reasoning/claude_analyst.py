"""
trading/reasoning/claude_analyst.py
────────────────────────────────────
Claude LLM reasoning layer — synthesizes all signals into a trade decision.

For each candidate stock, Claude receives:
  • LightGBM rank + predicted 5-day return + top SHAP features
  • Insider trading score (CEO/CFO buying = bullish)
  • News sentiment score + analyst consensus + upside to target
  • Macro regime (BULL/NEUTRAL/BEAR/HIGH_FEAR) + VIX + yield curve
  • Current price context

Claude returns a structured decision:
  {
    "decision":   "BUY" | "SKIP" | "WAIT",
    "conviction": "HIGH" | "MED" | "LOW",
    "thesis":     "...",    # 2–3 sentence human reasoning
    "risks":      "...",    # key downside risks
    "price_target": float,  # 5-day price target
  }

Requires: ANTHROPIC_API_KEY in .env
Model: claude-haiku-4-5-20251001 (fast + cheap for screening)
       Upgrades to claude-sonnet-4-6 for HIGH-conviction final check.

If API key is missing or call fails, falls back to auto-approving the
LightGBM signal so the rest of the bot keeps running.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from trading.logger import get_logger

load_dotenv(Path(__file__).parent.parent.parent / ".env")

logger = get_logger(__name__)

_SCREENING_MODEL = "claude-haiku-4-5-20251001"   # fast, cheap — screens all picks
_DEEP_MODEL      = "claude-sonnet-4-6"            # thorough — used only for final pick

_SYSTEM_PROMPT = """You are a quantitative trading assistant working alongside a LightGBM
model. The LightGBM rank and predicted return are your PRIMARY signal — they are based on
2 years of S&P 500 data and are statistically validated.

Your role is to flag CLEAR RED FLAGS only. Approve the trade unless you see:
  - Earnings announcement within 3 days (binary risk event)
  - Strongly negative news sentiment (score < -0.5)
  - Macro regime is BEAR or HIGH_FEAR
  - Analyst consensus is Strong Sell with negative upside

If none of those red flags are present, return BUY.
WAIT = needs 1-2 more days of data before deciding.
SKIP = hard no due to a specific identified risk.

Default bias: BUY when LightGBM rank <= 5 and confidence is HIGH or MED.

Respond ONLY with a JSON object — no markdown, no explanation outside the JSON:
{
  "decision":     "BUY" | "SKIP" | "WAIT",
  "conviction":   "HIGH" | "MED" | "LOW",
  "thesis":       "1-2 sentence explanation",
  "risks":        "main downside risk or NONE",
  "price_target": <float — your 5-day price target>
}"""


class ClaudeAnalyst:
    """
    LLM reasoning layer that approves or rejects LightGBM picks.

    Usage::

        analyst = ClaudeAnalyst()
        decision = analyst.analyse(ticker="AXON", signals={...})
        # {"decision": "BUY", "conviction": "HIGH", "thesis": "...", ...}

        approved = analyst.screen_picks(picks_list)   # filter a list of picks
    """

    def __init__(self) -> None:
        self._api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._enabled = bool(self._api_key)
        if not self._enabled:
            logger.warning(
                "ClaudeAnalyst: ANTHROPIC_API_KEY not set — "
                "reasoning disabled, all LightGBM picks auto-approved"
            )

    # ── Public ────────────────────────────────────────────────────────────────

    def analyse(self, ticker: str, signals: dict) -> dict:
        """
        Analyse signals for one ticker and return a decision dict.

        Parameters
        ----------
        ticker  : e.g. "AXON"
        signals : output of SignalAggregator.gather(ticker)

        Returns
        -------
        dict with keys: decision, conviction, thesis, risks, price_target
        Falls back to auto-BUY with LOW conviction if the API is unavailable.
        """
        if not self._enabled:
            return self._fallback(ticker, signals)

        # Use Sonnet only for rank #1, Haiku for everything else
        # Sonnet overthinks with neutral/missing signals — Haiku is more decisive
        lgbm_rank = signals.get("lgbm_rank", 99)
        model     = _DEEP_MODEL if lgbm_rank == 1 else _SCREENING_MODEL

        prompt = self._build_prompt(ticker, signals)
        try:
            result = self._call_api(model, prompt)
            result["model_used"] = model
            logger.info(
                f"ClaudeAnalyst: {ticker} → {result['decision']} "
                f"({result['conviction']}) via {model.split('-')[1]}"
            )
            return result
        except Exception as exc:
            logger.warning(f"ClaudeAnalyst: API error for {ticker} — {exc} — auto-approving")
            return self._fallback(ticker, signals)

    def screen_picks(self, picks: list[dict], signals_map: dict[str, dict]) -> list[dict]:
        """
        Filter a list of LightGBM picks through Claude reasoning.

        Parameters
        ----------
        picks        : list of dicts from LightGBMPredictor.score()
        signals_map  : ticker → signals dict from SignalAggregator

        Returns
        -------
        Filtered + enriched list — only BUY decisions, with thesis attached.
        """
        approved: list[dict] = []
        for pick in picks:
            ticker  = pick["ticker"]
            signals = signals_map.get(ticker, {})
            decision = self.analyse(ticker, signals)

            if decision["decision"] == "BUY":
                enriched = {**pick}
                enriched["llm_thesis"]      = decision.get("thesis", "")
                enriched["llm_risks"]       = decision.get("risks", "")
                enriched["llm_conviction"]  = decision.get("conviction", "LOW")
                enriched["llm_target"]      = decision.get("price_target")
                enriched["llm_model"]       = decision.get("model_used", "fallback")
                approved.append(enriched)
            else:
                logger.info(
                    f"ClaudeAnalyst: {ticker} REJECTED ({decision['decision']}) "
                    f"— {decision.get('thesis', '')[:80]}"
                )

        logger.info(
            f"ClaudeAnalyst: {len(picks)} picks → {len(approved)} approved by LLM"
        )
        return approved

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(ticker: str, signals: dict) -> str:
        price   = signals.get("current_price", 0)
        regime  = signals.get("macro_regime", "UNKNOWN")
        vix     = signals.get("vix", "N/A")
        yc      = signals.get("yield_curve", "N/A")

        lgbm_rank   = signals.get("lgbm_rank", "N/A")
        lgbm_pred   = signals.get("lgbm_pred_pct", "N/A")
        lgbm_conf   = signals.get("lgbm_confidence", "N/A")
        lgbm_feat   = signals.get("lgbm_top_features", [])

        insider     = signals.get("insider_score", 50)
        sentiment   = signals.get("sentiment_score", 0)
        analyst_con = signals.get("analyst_consensus", "N/A")
        analyst_up  = signals.get("analyst_upside_pct", "N/A")
        days_earn   = signals.get("days_to_earnings", "N/A")

        return f"""Analyse {ticker} at ${price:.2f} for a 5-day swing trade.

QUANTITATIVE MODEL (LightGBM):
  Rank          : #{lgbm_rank} out of ~100 S&P 500 stocks
  Predicted 5d  : {lgbm_pred}
  Confidence    : {lgbm_conf}
  Key drivers   : {', '.join(lgbm_feat) if lgbm_feat else 'N/A'}

INSIDER ACTIVITY (SEC EDGAR Form 4):
  Insider score : {insider}/100  (>70 = heavy insider buying, <30 = selling)

NEWS & ANALYST SIGNALS (Finnhub):
  Sentiment     : {sentiment:+.2f}  (-1 = very negative, +1 = very positive)
  Analyst view  : {analyst_con}
  Upside to target: {analyst_up}%
  Days to earnings: {days_earn}  (earnings < 5 days = elevated risk)

MACRO ENVIRONMENT (FRED):
  Regime        : {regime}
  VIX           : {vix}  (>30 = fear, <20 = calm)
  Yield curve   : {yc}   (negative = recession risk)

Should we BUY {ticker} right now for a 5-day hold? Respond in JSON only."""

    def _call_api(self, model: str, prompt: str) -> dict:
        import anthropic
        client   = anthropic.Anthropic(api_key=self._api_key)
        message  = client.messages.create(
            model      = model,
            max_tokens = 400,
            system     = _SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)

    @staticmethod
    def _fallback(ticker: str, signals: dict) -> dict:
        """Auto-approve with LOW conviction when API unavailable."""
        price = signals.get("current_price", 0)
        pred  = signals.get("lgbm_pred_pct", "?")
        return {
            "decision":    "BUY",
            "conviction":  "LOW",
            "thesis":      f"LightGBM signal {pred} — LLM reasoning unavailable, auto-approved.",
            "risks":       "No LLM risk analysis available.",
            "price_target": round(price * 1.02, 2) if price else None,
            "model_used":  "fallback",
        }
