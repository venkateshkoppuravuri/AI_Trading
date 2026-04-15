"""
trading/models/lightgbm_model.py
─────────────────────────────────
LightGBM model that predicts 5-day forward stock returns.

Training  : walk-forward validation (no lookahead bias)
Predicting: rank stocks by predicted return → top picks for next 5 days
Explaining: SHAP values per prediction (which features drove the call)

Usage (CLI)::

    python -m trading.models.lightgbm_model --train     # train + backtest
    python -m trading.models.lightgbm_model --score     # score today's stocks
    python -m trading.models.lightgbm_model --train --score  # both

Usage (import)::

    from trading.models.lightgbm_model import LightGBMPredictor
    model = LightGBMPredictor()
    model.train()
    rankings = model.score()     # list of {ticker, pred_return, confidence, shap}
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading.logger import get_logger

logger = get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_MODEL_DIR      = Path("state/models")
_MODEL_PATH     = _MODEL_DIR / "lgbm_model.pkl"
_METRICS_PATH   = _MODEL_DIR / "backtest_metrics.json"
_FEATURE_STORE  = Path("state/features/all_features.parquet")

# ── Feature columns (all technical + external signals) ────────────────────────
FEATURE_COLS: list[str] = [
    "ret_1d", "ret_5d", "ret_20d", "ret_60d",
    "rsi_14", "macd_hist", "bb_pct", "atr_pct",
    "vol_ratio_20", "vol_trend_5",
    "gap_pct", "high_52w_pct", "low_52w_pct",
    "insider_score", "sentiment_score", "analyst_upside", "macro_regime",
]
LABEL_COL = "ret_5d_fwd"


class LightGBMPredictor:
    """
    Predicts 5-day forward return for each stock using LightGBM.

    Walk-forward training ensures no lookahead bias:
    - Training window : oldest 70% of dates in the feature store
    - Test window     : most recent 30% of dates
    - Model saved to  : state/models/lgbm_model.pkl
    """

    def __init__(self) -> None:
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._model: Any = None   # LightGBM Booster, loaded lazily

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self) -> dict:
        """
        Train the LightGBM model on the feature store.
        Returns backtest metrics dict.
        Saves the trained model + metrics to state/models/.
        """
        import lightgbm as lgb

        df = self._load_features()
        if df.empty:
            raise RuntimeError(
                "Feature store is empty. Run: python -m trading.data.pipeline"
            )

        df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
        logger.info(f"Training data: {len(df):,} rows × {len(FEATURE_COLS)} features")

        # ── Walk-forward split — 3 windows ───────────────────────────────────
        # Train   0%–65% : model learns from historical data
        # Val    65%–80% : early stopping (same regime as train — no leakage)
        # Test   80%–100%: held-out evaluation only (not used for training)
        #
        # Keeping val WITHIN the training era prevents early stopping from
        # firing on a regime-changed test period (e.g. tariff crash).
        dates      = sorted(df.index.unique())
        n          = len(dates)
        train_end  = dates[int(n * 0.65)]
        val_end    = dates[int(n * 0.80)]
        test_start = dates[int(n * 0.80) + 1]

        train_df = df[df.index <= train_end]
        val_df   = df[(df.index > train_end) & (df.index <= val_end)]
        test_df  = df[df.index >= test_start]

        X_train, y_train = train_df[FEATURE_COLS], train_df[LABEL_COL]
        X_val,   y_val   = val_df[FEATURE_COLS],   val_df[LABEL_COL]
        X_test,  y_test  = test_df[FEATURE_COLS],  test_df[LABEL_COL]

        logger.info(
            f"Train: {len(train_df):,} rows (up to {train_end}) | "
            f"Val: {len(val_df):,} rows | "
            f"Test: {len(test_df):,} rows (from {test_start})"
        )

        # ── Adapt complexity to dataset size ─────────────────────────────────
        n_rows = len(train_df)
        if n_rows < 10_000:
            n_estimators = 200
            num_leaves   = 15
            early_stop   = None
            logger.info(f"Small dataset ({n_rows:,} rows) — no early stopping")
        elif n_rows < 100_000:
            n_estimators = 500
            num_leaves   = 31
            early_stop   = 100
        else:
            n_estimators = 1000
            num_leaves   = 63
            early_stop   = 50

        params = {
            "objective":         "regression",
            "metric":            "rmse",
            "boosting_type":     "gbdt",
            "num_leaves":        num_leaves,
            "max_depth":         -1,
            "learning_rate":     0.05,
            "n_estimators":      n_estimators,
            "min_child_samples": max(10, n_rows // 500),
            "feature_fraction":  0.8,
            "bagging_fraction":  0.8,
            "bagging_freq":      5,
            "reg_alpha":         0.1,
            "reg_lambda":        0.1,
            "random_state":      42,
            "verbose":           -1,
            "n_jobs":            -1,
        }

        callbacks = [lgb.log_evaluation(0)]
        if early_stop and not val_df.empty:
            callbacks.insert(0, lgb.early_stopping(early_stop, verbose=False))

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],   # validate against SAME-ERA data
            callbacks=callbacks,
        )

        # ── Backtest metrics ──────────────────────────────────────────────────
        y_pred     = model.predict(X_test)
        metrics    = self._compute_metrics(y_test, y_pred, test_df)
        metrics["trained_at"]   = datetime.now().isoformat()
        metrics["train_rows"]   = len(train_df)
        metrics["test_rows"]    = len(test_df)
        metrics["n_features"]   = len(FEATURE_COLS)
        metrics["best_iter"]    = model.best_iteration_ or params["n_estimators"]

        # ── Save ──────────────────────────────────────────────────────────────
        with open(_MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        _METRICS_PATH.write_text(json.dumps(metrics, indent=2))

        self._model = model
        self._log_results(metrics, model)
        return metrics

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, top_n: int = 10) -> list[dict]:
        """
        Score today's stocks and return the top-N ranked by predicted return.

        Each entry in the returned list::

            {
                "rank":         1,
                "ticker":       "NVDA",
                "pred_return":  0.034,        # predicted 5-day return (+3.4%)
                "confidence":   "HIGH",       # HIGH / MED / LOW
                "top_features": [...],        # top 3 SHAP drivers
            }

        Scoring is gated by macro regime:
          HIGH_FEAR → no output (market too dangerous, wait for calm)
          BEAR      → only tickers with pred_return > 2% pass through
          NEUTRAL / BULL → normal scoring
        """
        model = self._load_model()
        df    = self._load_features()
        if df.empty:
            logger.warning("Feature store empty — run the pipeline first")
            return []

        # ── Macro regime gate ─────────────────────────────────────────────────
        regime, size_mult = self._get_macro_regime()
        logger.info(f"Macro regime: {regime}  |  Size multiplier: {size_mult}x")

        # Latest feature vector per ticker (most recent date)
        df = df.reset_index()
        latest = (
            df.sort_values("date")
            .groupby("ticker")
            .last()
            .reset_index()
        )

        available = [c for c in FEATURE_COLS if c in latest.columns]
        X = latest[available].fillna(0)

        predictions = model.predict(X)
        latest      = latest.copy()
        latest["pred_return"] = predictions

        # ── SHAP explanations ─────────────────────────────────────────────────
        shap_top = self._compute_shap_top(model, X)

        # ── Regime gate: filter or warn ───────────────────────────────────────
        if regime == "HIGH_FEAR":
            logger.warning(
                "Macro regime is HIGH_FEAR (VIX > 30) — scoring suppressed. "
                "Wait for VIX to drop below 30 before taking new positions."
            )
            return []
        min_pred = 0.02 if regime == "BEAR" else 0.0   # BEAR: only strong signals pass

        # ── Build ranked output ───────────────────────────────────────────────
        latest = latest[latest["pred_return"] >= min_pred]
        top    = latest.nlargest(top_n, "pred_return")

        if top.empty:
            logger.warning(f"No tickers passed the {regime} regime filter (min pred={min_pred*100:.0f}%)")
            return []

        # Confidence thresholds: relative to the spread of predictions
        preds_arr = predictions[predictions > 0]
        hi_thresh = float(np.percentile(preds_arr, 75)) if len(preds_arr) else 0.02
        md_thresh = float(np.percentile(preds_arr, 40)) if len(preds_arr) else 0.01

        results: list[dict] = []
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            pred = float(row["pred_return"])
            conf = "HIGH" if pred >= hi_thresh else "MED" if pred >= md_thresh else "LOW"
            results.append({
                "rank":         rank,
                "ticker":       row["ticker"],
                "pred_return":  round(pred, 4),
                "pred_pct":     f"{pred*100:+.1f}%",
                "confidence":   conf,
                "regime":       regime,
                "size_mult":    size_mult,
                "top_features": shap_top.get(int(row.name), []),
            })

        self._print_scoreboard(results)
        return results

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _get_macro_regime() -> tuple[str, float]:
        """Return (regime, position_size_multiplier). Defaults to NEUTRAL/1.0 on failure."""
        try:
            from trading.signals.macro import MacroData
            macro = MacroData()
            return macro.get_market_regime(), macro.get_position_size_multiplier()
        except Exception:
            return "NEUTRAL", 1.0

    @staticmethod
    def _load_features() -> pd.DataFrame:
        if not _FEATURE_STORE.exists():
            return pd.DataFrame()
        df = pd.read_parquet(_FEATURE_STORE)
        # Ensure date index
        if "date" in df.columns:
            df = df.set_index("date")
        return df

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not _MODEL_PATH.exists():
            raise RuntimeError(
                "No trained model found. Run: python -m trading.models.lightgbm_model --train"
            )
        with open(_MODEL_PATH, "rb") as f:
            self._model = pickle.load(f)
        return self._model

    @staticmethod
    def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray, test_df: pd.DataFrame) -> dict:
        """Directional accuracy and top-5 simulated Sharpe."""
        # Directional accuracy
        correct_dir = ((y_pred > 0) == (y_true > 0)).mean()

        # Simulated top-5 strategy: each day buy the 5 highest predicted stocks
        test_with_pred = test_df[["ticker", LABEL_COL]].copy()
        test_with_pred["pred"] = y_pred

        daily_returns: list[float] = []
        for date, group in test_with_pred.groupby(level=0):
            if len(group) < 2:
                continue
            top5 = group.nlargest(5, "pred")
            daily_returns.append(float(top5[LABEL_COL].mean()))

        if daily_returns:
            arr    = np.array(daily_returns)
            sharpe = (arr.mean() / (arr.std() + 1e-9)) * np.sqrt(252 / 5)
            mean_r = float(arr.mean())
            win_rt = float((arr > 0).mean())
        else:
            sharpe = mean_r = win_rt = 0.0

        rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))

        return {
            "directional_accuracy": round(float(correct_dir), 4),
            "top5_simulated_sharpe": round(sharpe, 3),
            "top5_mean_5d_return":   round(mean_r, 4),
            "top5_win_rate":         round(win_rt, 4),
            "rmse":                  round(rmse, 6),
        }

    @staticmethod
    def _compute_shap_top(model, X: pd.DataFrame) -> dict[int, list[str]]:
        """Return top-3 SHAP feature names per row (index → list)."""
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            vals      = explainer.shap_values(X)
            result: dict[int, list[str]] = {}
            for i, row_vals in enumerate(vals):
                abs_v  = np.abs(row_vals)
                top3   = np.argsort(abs_v)[::-1][:3]
                result[i] = [X.columns[j] for j in top3]
            return result
        except Exception as exc:
            logger.debug(f"SHAP unavailable: {exc}")
            return {}

    @staticmethod
    def _log_results(metrics: dict, model) -> None:
        logger.info("=" * 52)
        logger.info("  LightGBM Training Complete")
        best = metrics.get('best_iter') or params.get('n_estimators', '?')
        logger.info(f"  Iterations trained : {best}")
        logger.info(f"  Directional acc    : {metrics['directional_accuracy']*100:.1f}%")
        logger.info(f"  Top-5 Sharpe       : {metrics['top5_simulated_sharpe']:.2f}")
        logger.info(f"  Top-5 win rate     : {metrics['top5_win_rate']*100:.1f}%")
        logger.info(f"  Top-5 mean 5d ret  : {metrics['top5_mean_5d_return']*100:+.2f}%")
        logger.info(f"  RMSE               : {metrics['rmse']:.6f}")
        logger.info("=" * 52)

        logger.info("Feature importances (top 10):")
        imp = pd.Series(
            model.feature_importances_,
            index=FEATURE_COLS[:len(model.feature_importances_)],
        ).sort_values(ascending=False)
        for feat, score in imp.head(10).items():
            logger.info(f"  {feat:<22} {score:>6.0f}")

    @staticmethod
    def _print_scoreboard(results: list[dict]) -> None:
        if not results:
            return
        regime    = results[0].get("regime", "?")
        size_mult = results[0].get("size_mult", 1.0)
        print("\n" + "=" * 60)
        print(f"  Stock Rankings — Predicted 5-Day Return")
        print(f"  Macro: {regime}  |  Position size: {size_mult}x of normal")
        print("=" * 60)
        print(f"  {'Rank':<5} {'Ticker':<7} {'Pred Return':<13} {'Conf':<6} {'Key Drivers'}")
        print("-" * 60)
        for r in results:
            drivers = ", ".join(r["top_features"][:2]) if r["top_features"] else "-"
            print(
                f"  {r['rank']:<5} {r['ticker']:<7} {r['pred_pct']:<13} "
                f"{r['confidence']:<6} {drivers}"
            )
        print("=" * 60 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LightGBM stock return predictor")
    p.add_argument("--train",  action="store_true", help="Train model on feature store")
    p.add_argument("--score",  action="store_true", help="Score stocks with trained model")
    p.add_argument("--top",    type=int, default=10, help="Top-N stocks to show (default 10)")
    return p.parse_args()


if __name__ == "__main__":
    args  = _parse_args()
    model = LightGBMPredictor()

    if not args.train and not args.score:
        print("Specify --train, --score, or both.")
        print("  python -m trading.models.lightgbm_model --train")
        print("  python -m trading.models.lightgbm_model --score")
        print("  python -m trading.models.lightgbm_model --train --score")
    else:
        if args.train:
            model.train()
        if args.score:
            model.score(top_n=args.top)
