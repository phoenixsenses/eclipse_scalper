# ============================================================================
# Eclipse Scalper — Production Docker Image
# Multi-stage build: deps install + slim runtime
# ============================================================================

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for native extensions (numpy, pandas, ccxt)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

LABEL maintainer="eclipse-scalper-team"
LABEL description="Eclipse Scalper — Binance Futures USDT-M Perpetual Scalping Bot"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create runtime directories
RUN mkdir -p logs logs/health state reports data

# Copy application code
COPY . .

# Environment defaults (override via docker-compose or .env)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SCALPER_DRY_RUN=1 \
    LOG_LEVEL=INFO

# Health check: heartbeat file mtime (primary) + dashboard API (secondary)
# Heartbeat file is updated every guardian cycle (~15s). Stale > 60s = unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "\
import os, sys, time; \
hb = 'logs/health/heartbeat.json'; \
ok = os.path.exists(hb) and (time.time() - os.path.getmtime(hb)) < 60; \
sys.exit(0 if ok else 1)" || exit 1

# Expose dashboard port
EXPOSE 8000

# Entrypoint: run deploy checklist then start bot
COPY scripts/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "main.py"]
