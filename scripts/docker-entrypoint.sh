#!/bin/bash
# ============================================================================
# Eclipse Scalper — Docker Entrypoint
# Runs deploy checklist before starting the bot.
# ============================================================================
set -e

echo "============================================"
echo "  Eclipse Scalper — Container Startup"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================"

# Ensure required directories exist
mkdir -p logs logs/health state reports data

# Run deploy checklist (non-fatal — log warnings but continue)
if [ -f scripts/deploy_checklist.sh ]; then
    echo ""
    echo "[ENTRYPOINT] Running deploy checklist..."
    bash scripts/deploy_checklist.sh || {
        echo "[ENTRYPOINT] WARNING: Deploy checklist reported issues (see above)"
        echo "[ENTRYPOINT] Continuing startup anyway..."
    }
    echo ""
fi

# Import check
echo "[ENTRYPOINT] Import check..."
python -c "import execution; import bot; import risk; import exchanges; print('  [OK] All core modules import successfully')" || {
    echo "  [FAIL] Core module import failed — aborting"
    exit 1
}

echo ""
echo "[ENTRYPOINT] Starting: $@"
echo "============================================"

exec "$@"
