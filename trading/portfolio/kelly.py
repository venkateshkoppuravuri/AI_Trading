"""
trading/portfolio/kelly.py
───────────────────────────
Kelly Criterion position sizer.

Kelly formula: f* = (p * b - q) / b
  p  = win probability
  b  = average win / average loss ratio (reward:risk)
  q  = 1 - p (loss probability)
  f* = fraction of capital to risk

In practice we use FRACTIONAL Kelly (half-Kelly by default) to reduce
variance. Full Kelly is mathematically optimal but causes huge drawdowns
in real markets. Half-Kelly gives ~75% of the return with ~50% the variance.

Sources for p and b:
  1. Live journal history (best — real outcomes from this bot)
  2. LightGBM predicted return + stop-loss/profit-target config (prior)
  3. Conservative defaults if no data available

Usage::

    from trading.portfolio.kelly import KellySizer
    ks = KellySizer()

    # Size for one pick using journal history
    fraction = ks.kelly_fraction("AXON", pred_return=0.039)
    dollars  = ks.size_position("AXON", pred_return=0.039, budget=400)
    # e.g. 0.72 → $288

    # Size all picks at once
    sizes = ks.size_all(picks, total_budget=2000)
    # {"AXON": {"dollars": 520, "shares": 1, "fraction": 0.26}, ...}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading.logger import get_logger

logger = get_logger(__name__)

_JOURNAL_DB    = Path("state/trade_journal.db")
_HALF_KELLY    = 0.5    # fractional Kelly multiplier (0.5 = half-Kelly)
_MAX_FRACTION  = 0.40   # never bet more than 40% of budget on one stock
_MIN_FRACTION  = 0.05   # never bet less than 5% (minimum meaningful position)
_MIN_TRADES    = 10     # need at least this many closed trades to use journal stats


class KellySizer:
    """
    Position sizer using fractional Kelly Criterion.

    Combines:
      • Historical win rate and payoff from the trade journal (when available)
      • LightGBM predicted return as the forward-looking edge estimate
      • Macro regime multiplier (from MacroData) to reduce size in bad markets
    """

    def __init__(self, kelly_fraction: float = _HALF_KELLY) -> None:
        self._kf      = kelly_fraction
        self._journal = self._load_journal_stats()

    # ── Public ────────────────────────────────────────────────────────────────

    def kelly_fraction(
        self,
        ticker:          str,
        pred_return:     float,
        stop_loss:       float = 0.03,
        profit_target:   float = 0.05,
        macro_mult:      float = 1.0,
    ) -> float:
        """
        Return the Kelly fraction (0–MAX_FRACTION) for a single position.

        Parameters
        ----------
        ticker        : stock symbol (used to look up ticker-specific history)
        pred_return   : LightGBM predicted 5-day return (e.g. 0.039)
        stop_loss     : downside exit level (e.g. 0.03 = 3%)
        profit_target : upside exit level (e.g. 0.05 = 5%)
        macro_mult    : position size multiplier from MacroData (0.25–1.0)

        Returns
        -------
        Fraction of budget to allocate (float, MIN_FRACTION–MAX_FRACTION)
        """
        p, b = self._estimate_edge(ticker, pred_return, stop_loss, profit_target)
        q    = 1.0 - p

        # Raw Kelly
        raw = (p * b - q) / b if b > 0 else 0.0

        # Apply fractional Kelly + macro multiplier
        f = raw * self._kf * macro_mult

        # Clamp
        f = max(_MIN_FRACTION, min(_MAX_FRACTION, f))

        logger.debug(
            f"Kelly {ticker}: p={p:.2f} b={b:.2f} raw={raw:.2f} "
            f"→ f={f:.2f} (×{macro_mult} macro)"
        )
        return round(f, 4)

    def size_position(
        self,
        ticker:        str,
        pred_return:   float,
        budget:        float,
        price:         float,
        stop_loss:     float = 0.03,
        profit_target: float = 0.05,
        macro_mult:    float = 1.0,
    ) -> dict:
        """
        Return dollar amount and share count for one position.

        Returns
        -------
        {"dollars": float, "shares": int, "fraction": float}
        """
        fraction = self.kelly_fraction(
            ticker, pred_return, stop_loss, profit_target, macro_mult
        )
        dollars  = budget * fraction
        shares   = int(dollars / price) if price > 0 else 0
        return {
            "dollars":  round(dollars, 2),
            "shares":   shares,
            "fraction": fraction,
        }

    def size_all(
        self,
        picks:        list[dict],
        total_budget: float,
        prices:       dict[str, float] | None = None,
        stop_loss:    float = 0.03,
        profit_target: float = 0.05,
        macro_mult:   float = 1.0,
    ) -> dict[str, dict]:
        """
        Size all picks, normalising so total allocated ≤ total_budget.

        Returns
        -------
        dict mapping ticker → {"dollars": float, "shares": int, "fraction": float}
        """
        if not picks:
            return {}

        # Raw Kelly fractions
        raw: dict[str, float] = {}
        for p in picks:
            ticker = p["ticker"]
            raw[ticker] = self.kelly_fraction(
                ticker,
                pred_return   = p.get("pred_return", 0.02),
                stop_loss     = stop_loss,
                profit_target = profit_target,
                macro_mult    = macro_mult,
            )

        # Normalise so fractions sum to ≤ 1.0
        total_raw = sum(raw.values())
        if total_raw > 1.0:
            raw = {t: f / total_raw for t, f in raw.items()}

        result: dict[str, dict] = {}
        for ticker, fraction in raw.items():
            dollars = total_budget * fraction
            price   = (prices or {}).get(ticker, 0)
            shares  = int(dollars / price) if price > 0 else 0
            result[ticker] = {
                "dollars":  round(dollars, 2),
                "shares":   shares,
                "fraction": round(fraction, 4),
            }

        logger.info(
            "Kelly sizing: " +
            " | ".join(
                f"{t} {v['fraction']:.1%} (${v['dollars']:.0f})"
                for t, v in result.items()
            )
        )
        return result

    # ── Edge estimation ───────────────────────────────────────────────────────

    def _estimate_edge(
        self,
        ticker:        str,
        pred_return:   float,
        stop_loss:     float,
        profit_target: float,
    ) -> tuple[float, float]:
        """
        Estimate (win_probability p, payoff_ratio b) for Kelly formula.

        Priority:
          1. Ticker-specific journal history (most accurate)
          2. Overall journal history (good prior)
          3. LightGBM prediction-based prior (no history yet)
        """
        # Try ticker-specific history
        if ticker in self._journal.get("by_ticker", {}):
            stats = self._journal["by_ticker"][ticker]
            if stats["n"] >= 3:
                return stats["win_rate"], stats["payoff"]

        # Try overall journal history
        overall = self._journal.get("overall", {})
        if overall.get("n", 0) >= _MIN_TRADES:
            return overall["win_rate"], overall["payoff"]

        # Fall back to prediction-based prior
        return self._prediction_prior(pred_return, stop_loss, profit_target)

    @staticmethod
    def _prediction_prior(
        pred_return:   float,
        stop_loss:     float,
        profit_target: float,
    ) -> tuple[float, float]:
        """
        Derive p and b from the LightGBM predicted return.

        p  : sigmoid of (pred_return / stop_loss) — higher pred → higher win prob
        b  : profit_target / stop_loss (fixed reward:risk ratio from config)
        """
        # Sigmoid centred at 0: pred=0 → p=0.50, pred=stop_loss → p=0.73
        x  = pred_return / stop_loss if stop_loss > 0 else 0.0
        p  = 1.0 / (1.0 + np.exp(-2.0 * x))
        p  = float(np.clip(p, 0.45, 0.75))   # realistic bounds
        b  = profit_target / stop_loss if stop_loss > 0 else 1.67
        return p, b

    # ── Journal stats loader ──────────────────────────────────────────────────

    @staticmethod
    def _load_journal_stats() -> dict:
        """
        Load closed trade outcomes from the SQLite journal.
        Returns {"overall": {win_rate, payoff, n}, "by_ticker": {...}}.
        """
        if not _JOURNAL_DB.exists():
            return {}
        try:
            import sqlite3
            conn = sqlite3.connect(_JOURNAL_DB)
            df   = pd.read_sql_query(
                "SELECT ticker, pnl, pnl_pct FROM trades WHERE status='CLOSED'", conn
            )
            conn.close()

            if df.empty:
                return {}

            def _stats(sub: pd.DataFrame) -> dict:
                wins    = sub[sub["pnl"] > 0]
                losses  = sub[sub["pnl"] <= 0]
                win_r   = len(wins) / len(sub)
                avg_win = wins["pnl_pct"].mean()  if len(wins)   > 0 else profit_target
                avg_loss= losses["pnl_pct"].abs().mean() if len(losses) > 0 else stop_loss
                payoff  = avg_win / avg_loss if avg_loss > 0 else 1.67
                return {"win_rate": round(win_r, 3), "payoff": round(payoff, 3), "n": len(sub)}

            profit_target = 0.05
            stop_loss     = 0.03

            overall   = _stats(df)
            by_ticker = {t: _stats(g) for t, g in df.groupby("ticker")}
            return {"overall": overall, "by_ticker": by_ticker}

        except Exception as exc:
            logger.debug(f"KellySizer: could not load journal stats — {exc}")
            return {}
