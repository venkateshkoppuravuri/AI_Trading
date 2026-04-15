#!/usr/bin/env bash
# scripts/entrypoint.sh — startup script for the trading bot container
# Runs at container start (Railway / Render / docker-compose bot service)
#
# Steps:
#   1. Ensure state directory tree exists (safe even if volume already has data)
#   2. If feature store is missing → run the data pipeline (first boot)
#   3. If LightGBM model is missing → train it
#   4. Start the trading bot

set -euo pipefail

echo "======================================================"
echo "  AI Trading Bot — container starting"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "======================================================"

# ── 1. Directory structure ────────────────────────────────────────────────────
mkdir -p state/bars state/features state/models \
         state/journal state/backtest state/risk logs

# ── 2. Seed feature store if this is a fresh volume ──────────────────────────
if [ ! -f "state/features/all_features.parquet" ]; then
    echo ""
    echo ">>> Feature store not found. Running data pipeline (first boot)..."
    echo "    This may take 5-10 minutes on first run."
    python -m trading.data.pipeline || {
        echo "WARNING: Pipeline failed — bot will start but may not trade."
    }
fi

# ── 3. Train model if missing ─────────────────────────────────────────────────
if [ ! -f "state/models/lgbm_model.pkl" ]; then
    echo ""
    echo ">>> LightGBM model not found. Training now..."
    python -m trading.models.lightgbm_model --train || {
        echo "WARNING: Model training failed — bot will start but AI picks disabled."
    }
fi

# ── 4. Launch bot ─────────────────────────────────────────────────────────────
echo ""
echo ">>> Starting trading bot (python main.py --strategy ai)"
echo ""
exec python main.py --strategy ai
