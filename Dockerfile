# ── AI Trading Bot — Docker image ────────────────────────────────────────────
# Builds a single image that can run either the trading bot or the Streamlit
# dashboard depending on the CMD override.
#
# Build:   docker build -t ai-trading .
# Bot:     docker run --env-file .env -v trading_state:/app/state ai-trading
# Dash:    docker run --env-file .env -v trading_state:/app/state -p 8501:8501 \
#              ai-trading python -m streamlit run streamlit_app.py \
#              --server.port 8501 --server.address 0.0.0.0 --server.headless true

FROM python:3.11-slim

# System build deps (needed by LightGBM, scipy, pyarrow wheel builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 dos2unix \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy only the metadata first so the layer is cached unless deps change
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -e .

# ── Copy application source ───────────────────────────────────────────────────
COPY . .

# ── Create runtime directories (volume will overlay state/ on first mount) ────
RUN mkdir -p state/bars state/features state/models \
             state/journal state/backtest state/risk \
    && mkdir -p logs

# ── Fix line endings (Windows checkout → Linux container) ────────────────────
RUN find scripts/ -name "*.sh" -exec dos2unix {} \; 2>/dev/null || true \
 && chmod +x scripts/*.sh 2>/dev/null || true

# ── Default: run the AI Signal trading bot ───────────────────────────────────
CMD ["bash", "scripts/entrypoint.sh"]
