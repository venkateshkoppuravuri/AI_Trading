#!/usr/bin/env bash
# scripts/seed_pipeline.sh — one-time data seeding script
#
# Run this ONCE after first deploy to populate state/ with bars,
# features, and a trained LightGBM model.
#
# Usage (local Docker):
#   docker compose run --rm bot bash scripts/seed_pipeline.sh
#
# Usage (Railway one-off command):
#   railway run bash scripts/seed_pipeline.sh
#
# Usage (Render shell):
#   bash scripts/seed_pipeline.sh

set -euo pipefail

echo "======================================================"
echo "  AI Trading — state seeding"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "======================================================"

mkdir -p state/bars state/features state/models \
         state/journal state/backtest state/risk logs

# ── Step 1: Fetch OHLCV bars + build feature store ───────────────────────────
echo ""
echo "[1/2] Running data pipeline (fetches bars + builds features)..."
python -m trading.data.pipeline
echo "  Pipeline complete."

# ── Step 2: Train LightGBM model ─────────────────────────────────────────────
echo ""
echo "[2/2] Training LightGBM model..."
python -m trading.models.lightgbm_model --train
echo "  Training complete."

echo ""
echo "======================================================"
echo "  Seeding done. You can now start the bot."
echo "======================================================"
