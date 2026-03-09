#!/bin/bash
# ============================================================================
# Eclipse Scalper — Pre-Deploy Verification Checklist
# ============================================================================
# Runs a sequence of checks before deploying the bot to production.
# Exit code: 0 = all checks passed, 1 = one or more checks failed.
# ============================================================================

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

pass_check() {
    echo "  [PASS] $1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

fail_check() {
    echo "  [FAIL] $1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

warn_check() {
    echo "  [WARN] $1"
    WARN_COUNT=$((WARN_COUNT + 1))
}

# Find repo root (script lives in scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================"
echo "  Eclipse Scalper — Deploy Checklist"
echo "  Repo: $REPO_ROOT"
echo "  Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================"
echo ""

# --------------------------------------------------------------------------
# 1. Python version >= 3.10
# --------------------------------------------------------------------------
echo "[1/7] Python version check"
if command -v python3 &>/dev/null; then
    PY_CMD=python3
elif command -v python &>/dev/null; then
    PY_CMD=python
else
    fail_check "Python not found in PATH"
    PY_CMD=""
fi

if [ -n "$PY_CMD" ]; then
    PY_VERSION=$($PY_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    PY_MAJOR=$($PY_CMD -c "import sys; print(sys.version_info.major)" 2>/dev/null)
    PY_MINOR=$($PY_CMD -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
    if [ "$PY_MAJOR" -ge 3 ] 2>/dev/null && [ "$PY_MINOR" -ge 10 ] 2>/dev/null; then
        pass_check "Python $PY_VERSION (>= 3.10)"
    else
        fail_check "Python $PY_VERSION is below required 3.10"
    fi
fi

# --------------------------------------------------------------------------
# 2. Required environment variables
# --------------------------------------------------------------------------
echo ""
echo "[2/7] Environment variables"

if [ -n "$BINANCE_API_KEY" ]; then
    pass_check "BINANCE_API_KEY is set"
else
    fail_check "BINANCE_API_KEY is NOT set"
fi

if [ -n "$BINANCE_API_SECRET" ]; then
    pass_check "BINANCE_API_SECRET is set"
else
    fail_check "BINANCE_API_SECRET is NOT set"
fi

# --------------------------------------------------------------------------
# 3. Compile check on critical files
# --------------------------------------------------------------------------
echo ""
echo "[3/7] Compile checks (py_compile)"

CRITICAL_FILES=(
    "execution/order_router.py"
    "execution/reconcile.py"
    "execution/guardian.py"
    "execution/kill_switch.py"
    "execution/entry_loop.py"
)

for f in "${CRITICAL_FILES[@]}"; do
    FULL_PATH="$REPO_ROOT/$f"
    if [ ! -f "$FULL_PATH" ]; then
        fail_check "$f — file not found"
        continue
    fi
    if $PY_CMD -m py_compile "$FULL_PATH" 2>/dev/null; then
        pass_check "$f compiles OK"
    else
        fail_check "$f has syntax errors"
    fi
done

# --------------------------------------------------------------------------
# 4. Runtime safety tests
# --------------------------------------------------------------------------
echo ""
echo "[4/7] Runtime safety tests"

TESTS_DIR="$REPO_ROOT/tests/runtime"
if [ -d "$TESTS_DIR" ]; then
    if $PY_CMD -m pytest "$TESTS_DIR" -q --timeout=60 2>/dev/null; then
        pass_check "Runtime tests passed"
    else
        fail_check "Runtime tests failed (or pytest not installed)"
    fi
else
    warn_check "tests/runtime/ directory not found — skipping"
fi

# --------------------------------------------------------------------------
# 5. Disk space check (warn if < 500MB free)
# --------------------------------------------------------------------------
echo ""
echo "[5/7] Disk space"

# Works on Linux/macOS/Git-Bash-on-Windows
FREE_KB=$(df -k "$REPO_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')
if [ -n "$FREE_KB" ] && [ "$FREE_KB" -gt 0 ] 2>/dev/null; then
    FREE_MB=$((FREE_KB / 1024))
    if [ "$FREE_MB" -ge 500 ]; then
        pass_check "Disk space: ${FREE_MB}MB free"
    else
        warn_check "Disk space low: ${FREE_MB}MB free (< 500MB)"
    fi
else
    warn_check "Could not determine free disk space"
fi

# --------------------------------------------------------------------------
# 6. Required directories
# --------------------------------------------------------------------------
echo ""
echo "[6/7] Required directories"

for dir_name in "logs" "state"; do
    DIR_PATH="$REPO_ROOT/$dir_name"
    if [ -d "$DIR_PATH" ]; then
        pass_check "$dir_name/ exists"
    else
        fail_check "$dir_name/ directory missing — create it before deploy"
    fi
done

# --------------------------------------------------------------------------
# 7. Summary
# --------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  SUMMARY"
echo "    Passed : $PASS_COUNT"
echo "    Failed : $FAIL_COUNT"
echo "    Warnings: $WARN_COUNT"
echo "============================================"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "  RESULT: FAIL — fix $FAIL_COUNT issue(s) before deploying."
    exit 1
else
    echo "  RESULT: PASS — all critical checks OK."
    exit 0
fi
