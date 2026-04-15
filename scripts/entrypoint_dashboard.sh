#!/usr/bin/env bash
# scripts/entrypoint_dashboard.sh — startup script for the Streamlit dashboard
# Runs at container start (Railway / Render / docker-compose dashboard service)

set -euo pipefail

echo "======================================================"
echo "  AI Trading Dashboard — container starting"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "======================================================"

mkdir -p state/bars state/features state/models \
         state/journal state/backtest state/risk logs

# PORT is set by Railway/Render automatically; default 8501 for local
PORT="${PORT:-8501}"

echo ">>> Launching Streamlit on port $PORT"
exec python -m streamlit run streamlit_app.py \
    --server.port "$PORT" \
    --server.address "0.0.0.0" \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false
