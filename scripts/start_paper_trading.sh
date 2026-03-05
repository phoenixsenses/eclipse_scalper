#!/usr/bin/env bash
# =============================================================================
# Eclipse Scalper — Paper Trading Startup Script (Bash)
#
# Usage:
#   bash scripts/start_paper_trading.sh
#   bash scripts/start_paper_trading.sh --env .env.paper.dual
#   bash scripts/start_paper_trading.sh --env .env.paper --skip-validation
#   bash scripts/start_paper_trading.sh --no-watchdog
#
# IMPORTANT — Why not `python main.py`?
#   main.py removes SCALPER_DRY_RUN from the environment unless --dry-run is
#   passed. Using execution.bootstrap directly preserves SCALPER_DRY_RUN=1
#   from the .env.paper file, guaranteeing paper mode.
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
ENV_FILE=".env.paper"
SKIP_VALIDATION=0
NO_WATCHDOG=0
WATCHDOG_PID=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env) ENV_FILE="$2"; shift 2 ;;
        --skip-validation) SKIP_VALIDATION=1; shift ;;
        --no-watchdog) NO_WATCHDOG=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Cleanup on exit ───────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "================================================="
    echo "  Eclipse Scalper — Session Ended"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================="
    if [[ -n "$WATCHDOG_PID" ]]; then
        echo "Stopping watchdog (PID: $WATCHDOG_PID)..."
        kill "$WATCHDOG_PID" 2>/dev/null || true
        wait "$WATCHDOG_PID" 2>/dev/null || true
        echo "  Watchdog stopped."
    fi
    echo "Trade log: ${ENTRY_TRADE_LOG_DB:-data/paper_trades.db}"
    echo ""
}
trap cleanup EXIT INT TERM

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "================================================="
echo "  Eclipse Scalper — Paper Trading Mode"
echo "================================================="
echo "  Config  : $ENV_FILE"
echo "  Date    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================="
echo ""

# ── Load .env file ────────────────────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: Config file not found: $ENV_FILE" >&2
    echo "Create it by copying: cp .env.example $ENV_FILE" >&2
    exit 1
fi

echo "Loading $ENV_FILE ..."
# Export each non-comment, non-blank var=value line
set -a
# Use a subshell to parse safely, then re-export
while IFS= read -r line; do
    # Skip blanks and comments
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// }" ]] && continue
    # Must contain =
    [[ "$line" != *"="* ]] && continue
    # Extract name and value (split on first =)
    varname="${line%%=*}"
    varvalue="${line#*=}"
    # Trim whitespace from name
    varname="${varname// /}"
    # Strip inline comments (unquoted # preceded by space)
    varvalue="${varvalue%%  #*}"
    varvalue="${varvalue%% #*}"
    # Strip surrounding quotes
    varvalue="${varvalue#\"}"
    varvalue="${varvalue%\"}"
    varvalue="${varvalue#\'}"
    varvalue="${varvalue%\'}"
    # Export
    export "$varname=$varvalue" 2>/dev/null || true
done < "$ENV_FILE"
set +a

# Verify SCALPER_DRY_RUN=1
if [[ "${SCALPER_DRY_RUN:-}" != "1" ]]; then
    echo ""
    echo "ERROR: SCALPER_DRY_RUN is not '1' after loading $ENV_FILE" >&2
    echo "       Current value: '${SCALPER_DRY_RUN:-<not set>}'" >&2
    echo "       Set SCALPER_DRY_RUN=1 in $ENV_FILE" >&2
    exit 1
fi
echo "  Paper mode confirmed: SCALPER_DRY_RUN=1"

# ── Validate environment ──────────────────────────────────────────────────────
if [[ "$SKIP_VALIDATION" -eq 0 ]]; then
    echo ""
    echo "Running environment validation..."
    python -m tools.validate_env --env "$ENV_FILE"
    validation_rc=$?
    if [[ $validation_rc -ne 0 ]]; then
        echo ""
        echo "Validation failed. Fix the FAIL items above before starting." >&2
        exit 1
    fi
else
    echo "  Validation skipped (--skip-validation)"
fi

# ── Start data collection watchdog (background) ───────────────────────────────
if [[ "$NO_WATCHDOG" -eq 0 ]]; then
    echo ""
    echo "Starting data collection watchdog..."
    python tools/collection_watchdog.py &
    WATCHDOG_PID=$!
    echo "  Watchdog started (PID: $WATCHDOG_PID)"
else
    echo "  Watchdog skipped (--no-watchdog)"
fi

# ── Start paper trading ───────────────────────────────────────────────────────
echo ""
echo "Starting paper trading (CTRL+C to stop)..."
echo ""

# Use execution.bootstrap directly — NOT python main.py
# (main.py removes SCALPER_DRY_RUN unless --dry-run is passed)
python -m execution.bootstrap
