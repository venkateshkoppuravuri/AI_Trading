"""
trading/backtest/engine.py
───────────────────────────
Full-pipeline backtesting engine.

Simulates the AISignalStrategy day by day on historical data:
  1. LightGBM scores every ticker for each date in the test window
  2. Confidence filter (same percentile thresholds as live)
  3. Check exits for open positions using day OHLCV:
       STOP_LOSS     — day low  <= entry × (1 - stop_loss)
       PROFIT_TARGET — day high >= entry × (1 + profit_target)
       TRAIL_STOP    — day low  <  peak  × (1 - trail_pct)
       TIME_STOP     — days held >= max_holding_days
       SIGNAL_EXIT   — ticker dropped from model top-N
  4. Enter approved picks at next-day open price
  5. Record equity mark-to-market each day

Walk-forward integrity:
  The model is trained on the oldest 65 % of the feature-store dates.
  The backtest uses only the test window (last 20 %) so there is zero
  overlap between training data and simulated trades.

No Claude LLM calls are made during backtest (too slow and costly).
The results represent the LightGBM + rule-based layer only.
"""

from __future__ import annotations

import pickle
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading.logger import get_logger
from trading.models.lightgbm_model import FEATURE_COLS

logger = get_logger(__name__)

_FEATURE_STORE = Path("state/features/all_features.parquet")
_MODEL_PATH    = Path("state/models/lgbm_model.pkl")
_BARS_DIR      = Path("state/bars")


class BacktestEngine:
    """
    Replays the AI Signal strategy on the held-out test window.

    Parameters
    ----------
    initial_capital   : Starting portfolio cash (default $2 000)
    max_positions     : Max simultaneous holdings (default 5)
    stop_loss         : Hard stop fraction (default 0.03 = 3 %)
    profit_target     : Take-profit fraction (default 0.05 = 5 %)
    trail_pct         : Trailing stop distance from peak (default 0.03)
    max_holding_days  : Time-stop after this many days (default 10)
    min_confidence    : "HIGH" | "MED" | "LOW" (default "MED")
    test_pct          : Fraction of dates used for testing (default 0.20)
    start_date        : Override test-window start (YYYY-MM-DD string or date)
    end_date          : Override test-window end
    """

    def __init__(
        self,
        initial_capital:  float = 2_000.0,
        max_positions:    int   = 5,
        stop_loss:        float = 0.03,
        profit_target:    float = 0.05,
        trail_pct:        float = 0.03,
        max_holding_days: int   = 10,
        min_confidence:   str   = "MED",
        test_pct:         float = 0.20,
        start_date:       str | date | None = None,
        end_date:         str | date | None = None,
    ) -> None:
        self.initial_capital  = initial_capital
        self.max_positions    = max_positions
        self.stop_loss        = stop_loss
        self.profit_target    = profit_target
        self.trail_pct        = trail_pct
        self.max_holding_days = max_holding_days
        self.min_confidence   = min_confidence
        self.test_pct         = test_pct

        # Parse date overrides
        self.start_date = _parse_date(start_date)
        self.end_date   = _parse_date(end_date)

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self) -> "BacktestResult":  # noqa: F821
        from trading.backtest.result import BacktestResult

        logger.info("Backtest: loading feature store and model…")
        features = self._load_features()
        model    = self._load_model()
        bars     = self._load_all_bars()

        if features.empty:
            raise RuntimeError("Feature store is empty — run the pipeline first.")
        if model is None:
            raise RuntimeError("No trained model — run: python -m trading.models.lightgbm_model --train")

        # ── Determine test window ─────────────────────────────────────────────
        all_dates = sorted(features.index.unique())
        test_start_idx = int(len(all_dates) * (1.0 - self.test_pct))
        default_start  = all_dates[test_start_idx]

        sim_start = self.start_date or default_start
        sim_end   = self.end_date   or all_dates[-1]

        sim_dates = [d for d in all_dates if sim_start <= d <= sim_end]
        if len(sim_dates) < 5:
            raise RuntimeError(
                f"Only {len(sim_dates)} dates in simulation window "
                f"({sim_start} → {sim_end}). Need at least 5."
            )

        logger.info(
            f"Backtest: {len(sim_dates)} trading days "
            f"({sim_start} → {sim_end}) | "
            f"capital ${self.initial_capital:,.0f} | "
            f"max {self.max_positions} positions"
        )

        # ── Precompute confidence thresholds (same logic as live scoring) ─────
        available = [c for c in FEATURE_COLS if c in features.columns]
        all_preds = model.predict(features[available].fillna(0))
        pos_preds = all_preds[all_preds > 0]
        hi_thresh = float(np.percentile(pos_preds, 75)) if len(pos_preds) else 0.02
        md_thresh = float(np.percentile(pos_preds, 40)) if len(pos_preds) else 0.01
        conf_min  = {"HIGH": hi_thresh, "MED": md_thresh, "LOW": 0.0}
        min_pred  = conf_min.get(self.min_confidence, md_thresh)

        # ── Simulation state ──────────────────────────────────────────────────
        cash      = self.initial_capital
        positions: dict[str, dict] = {}   # ticker → position state
        equity_curve: dict[date, float] = {}
        trades: list[dict] = []

        for day_i, today in enumerate(sim_dates):
            # ── Score picks for today ─────────────────────────────────────────
            try:
                snap = features.loc[today]
            except KeyError:
                equity_curve[today] = _mark_to_market(cash, positions, bars, today)
                continue

            if isinstance(snap, pd.Series):
                snap = snap.to_frame().T   # single ticker on this date

            picks = self._score_snapshot(model, snap, available, min_pred, hi_thresh, md_thresh)
            pick_tickers = {p["ticker"] for p in picks}

            # ── Check exits ───────────────────────────────────────────────────
            for ticker in list(positions.keys()):
                pos      = positions[ticker]
                day_bars = _get_day_bars(bars, ticker, today)
                if day_bars is None:
                    continue

                entry       = pos["entry_price"]
                shares      = pos["shares"]
                days_held   = (today - pos["entry_date"]).days
                day_low     = day_bars["low"]
                day_high    = day_bars["high"]
                day_close   = day_bars["close"]

                # Update trailing peak
                pos["peak_price"] = max(pos["peak_price"], day_high)
                trail_floor       = pos["peak_price"] * (1.0 - self.trail_pct)

                # Evaluate exit rules in priority order
                stop_level   = entry * (1.0 - self.stop_loss)
                profit_level = entry * (1.0 + self.profit_target)

                if day_low <= stop_level:
                    exit_price  = stop_level
                    exit_reason = "STOP_LOSS"
                elif day_high >= profit_level:
                    exit_price  = profit_level
                    exit_reason = "PROFIT_TARGET"
                elif trail_floor > entry and day_low < trail_floor:
                    # Trail stop only activates once price has risen enough
                    # that trail_floor > entry (peak must be > entry / 0.97)
                    # — mirrors live strategy, prevents same-day trigger
                    exit_price  = trail_floor
                    exit_reason = "TRAIL_STOP"
                elif days_held >= self.max_holding_days:
                    exit_price  = day_close
                    exit_reason = "TIME_STOP"
                elif pick_tickers and ticker not in pick_tickers and days_held >= 2:
                    # Require 2-day minimum hold before signal exit — prevents
                    # churning on daily model noise (model predicts 5-day returns)
                    exit_price  = day_close
                    exit_reason = "SIGNAL_EXIT"
                else:
                    continue  # hold

                pnl          = (exit_price - entry) * shares
                pnl_pct      = (exit_price - entry) / entry
                cash        += exit_price * shares
                trades.append({
                    "ticker":       ticker,
                    "entry_price":  round(entry, 4),
                    "exit_price":   round(exit_price, 4),
                    "shares":       shares,
                    "entry_date":   pos["entry_date"],
                    "exit_date":    today,
                    "holding_days": days_held,
                    "pnl":          round(pnl, 4),
                    "pnl_pct":      round(pnl_pct, 4),
                    "exit_reason":  exit_reason,
                    "pred_return":  pos.get("pred_return", 0),
                    "confidence":   pos.get("confidence", "?"),
                })
                del positions[ticker]

            # ── Enter new positions (at tomorrow's open) ──────────────────────
            slots = self.max_positions - len(positions)
            if slots > 0 and day_i + 1 < len(sim_dates):
                next_day  = sim_dates[day_i + 1]
                new_picks = [p for p in picks if p["ticker"] not in positions][:slots]

                for pick in new_picks:
                    ticker     = pick["ticker"]
                    next_bars  = _get_day_bars(bars, ticker, next_day)
                    if next_bars is None:
                        continue

                    entry_price = next_bars["open"]
                    if entry_price <= 0:
                        continue

                    dollars = self.initial_capital / self.max_positions
                    shares  = int(dollars / entry_price)
                    if shares < 1:
                        continue
                    cost = shares * entry_price
                    if cost > cash:
                        shares = int(cash * 0.95 / entry_price)
                        cost   = shares * entry_price
                    if shares < 1:
                        continue

                    cash -= cost
                    positions[ticker] = {
                        "entry_price": entry_price,
                        "shares":      shares,
                        "entry_date":  next_day,
                        "peak_price":  entry_price,
                        "pred_return": pick["pred_return"],
                        "confidence":  pick["confidence"],
                    }

            # ── Mark to market ────────────────────────────────────────────────
            equity_curve[today] = _mark_to_market(cash, positions, bars, today)

        # ── Close any remaining open positions at last known close ────────────
        last_day = sim_dates[-1]
        for ticker, pos in list(positions.items()):
            last_bars = _get_day_bars(bars, ticker, last_day)
            exit_price = last_bars["close"] if last_bars else pos["entry_price"]
            pnl = (exit_price - pos["entry_price"]) * pos["shares"]
            trades.append({
                "ticker":       ticker,
                "entry_price":  round(pos["entry_price"], 4),
                "exit_price":   round(exit_price, 4),
                "shares":       pos["shares"],
                "entry_date":   pos["entry_date"],
                "exit_date":    last_day,
                "holding_days": (last_day - pos["entry_date"]).days,
                "pnl":          round(pnl, 4),
                "pnl_pct":      round((exit_price - pos["entry_price"]) / pos["entry_price"], 4),
                "exit_reason":  "END_OF_BACKTEST",
                "pred_return":  pos.get("pred_return", 0),
                "confidence":   pos.get("confidence", "?"),
            })

        logger.info(
            f"Backtest complete: {len(trades)} trades | "
            f"final equity ${list(equity_curve.values())[-1]:,.2f}"
        )
        return BacktestResult(
            equity_curve    = equity_curve,
            trades          = trades,
            initial_capital = self.initial_capital,
            params = {
                "stop_loss":        self.stop_loss,
                "profit_target":    self.profit_target,
                "trail_pct":        self.trail_pct,
                "max_holding_days": self.max_holding_days,
                "max_positions":    self.max_positions,
                "min_confidence":   self.min_confidence,
                "start_date":       str(sim_start),
                "end_date":         str(sim_end),
            },
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _score_snapshot(
        self,
        model,
        snap:      pd.DataFrame,
        available: list[str],
        min_pred:  float,
        hi_thresh: float,
        md_thresh: float,
    ) -> list[dict]:
        """Score a single-date feature snapshot, return ranked pick list."""
        try:
            X    = snap[available].fillna(0)
            pred = model.predict(X)

            snap      = snap.copy()
            snap["_pred"] = pred
            snap       = snap[snap["_pred"] >= min_pred]

            results = []
            for _, row in snap.nlargest(self.max_positions * 2, "_pred").iterrows():
                p    = float(row["_pred"])
                conf = "HIGH" if p >= hi_thresh else "MED" if p >= md_thresh else "LOW"
                results.append({
                    "ticker":      row["ticker"] if "ticker" in snap.columns else row.name,
                    "pred_return": round(p, 4),
                    "confidence":  conf,
                })
            return results
        except Exception as exc:
            logger.debug(f"Backtest: scoring failed on {exc}")
            return []

    @staticmethod
    def _load_features() -> pd.DataFrame:
        if not _FEATURE_STORE.exists():
            return pd.DataFrame()
        df = pd.read_parquet(_FEATURE_STORE)
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index).date
        return df

    @staticmethod
    def _load_model() -> Any | None:
        if not _MODEL_PATH.exists():
            return None
        with open(_MODEL_PATH, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _load_all_bars() -> dict[str, pd.DataFrame]:
        bars: dict[str, pd.DataFrame] = {}
        for path in _BARS_DIR.glob("*.parquet"):
            try:
                df = pd.read_parquet(path)
                # Normalise date index
                if df.index.dtype == "object" or str(df.index.dtype).startswith("datetime"):
                    df.index = pd.to_datetime(df.index).date
                elif "date" in df.columns:
                    df = df.set_index("date")
                    df.index = pd.to_datetime(df.index).date
                # Ensure required columns exist
                for col in ("open", "high", "low", "close"):
                    if col not in df.columns and "o" in df.columns:
                        df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
                        break
                bars[path.stem] = df
            except Exception as exc:
                logger.debug(f"Backtest: could not load bars for {path.stem}: {exc}")
        logger.info(f"Backtest: loaded bars for {len(bars)} tickers")
        return bars


# ── Module helpers ────────────────────────────────────────────────────────────

def _get_day_bars(bars: dict, ticker: str, day: date) -> dict | None:
    """Return {open, high, low, close} for ticker on day, or None."""
    df = bars.get(ticker)
    if df is None:
        return None
    try:
        row = df.loc[day]
        return {
            "open":  float(row["open"])  if "open"  in df.columns else float(row["close"]),
            "high":  float(row["high"])  if "high"  in df.columns else float(row["close"]),
            "low":   float(row["low"])   if "low"   in df.columns else float(row["close"]),
            "close": float(row["close"]),
        }
    except (KeyError, Exception):
        return None


def _mark_to_market(
    cash:      float,
    positions: dict,
    bars:      dict,
    today:     date,
) -> float:
    """Return total portfolio value: cash + sum of position market values."""
    total = cash
    for ticker, pos in positions.items():
        day_bars = _get_day_bars(bars, ticker, today)
        price    = day_bars["close"] if day_bars else pos["entry_price"]
        total   += price * pos["shares"]
    return total


def _parse_date(d: str | date | None) -> date | None:
    if d is None:
        return None
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d))
