#!/bin/bash
# ============================================================================
# Eclipse Scalper — Restore Script
# ============================================================================
# Restores state from a backup created by backup.sh.
#
# Usage:
#   bash scripts/restore.sh /path/to/backups/eclipse_backup_20260310_120000
#
# Safety:
#   - Creates a pre-restore backup before overwriting
#   - Validates backup contents before restoring
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_SRC="${1:-}"

if [ -z "$BACKUP_SRC" ]; then
    echo "Usage: bash scripts/restore.sh /path/to/backup/directory"
    echo ""
    echo "Available backups:"
    ls -dt "$REPO_ROOT/backups"/eclipse_backup_* 2>/dev/null | head -10 || echo "  (none found)"
    exit 1
fi

if [ ! -d "$BACKUP_SRC" ]; then
    echo "ERROR: Backup directory not found: $BACKUP_SRC"
    exit 1
fi

echo "============================================"
echo "  Eclipse Scalper — Restore"
echo "  Source: $BACKUP_SRC"
echo "  Dest:   $REPO_ROOT"
echo "  Time:   $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================"

# Validate backup
if [ ! -f "$BACKUP_SRC/backup_meta.json" ]; then
    echo "ERROR: No backup_meta.json found — invalid backup directory"
    exit 1
fi

echo ""
echo "Backup metadata:"
cat "$BACKUP_SRC/backup_meta.json"
echo ""

# Pre-restore safety backup
echo "[0/3] Creating pre-restore safety backup..."
SAFETY_DIR="$REPO_ROOT/backups/pre_restore_$(date -u '+%Y%m%d_%H%M%S')"
mkdir -p "$SAFETY_DIR"
[ -f "$REPO_ROOT/data/paper_trades.db" ] && cp "$REPO_ROOT/data/paper_trades.db" "$SAFETY_DIR/" 2>/dev/null || true
[ -f "$REPO_ROOT/state/brain.json" ] && cp "$REPO_ROOT/state/brain.json" "$SAFETY_DIR/" 2>/dev/null || true
[ -f "$REPO_ROOT/logs/kill_switch_state.json" ] && cp "$REPO_ROOT/logs/kill_switch_state.json" "$SAFETY_DIR/" 2>/dev/null || true
echo "  Safety backup: $SAFETY_DIR"

# Restore files
if [ -f "$BACKUP_SRC/paper_trades.db" ]; then
    echo "[1/3] Restoring paper_trades.db..."
    mkdir -p "$REPO_ROOT/data"
    cp "$BACKUP_SRC/paper_trades.db" "$REPO_ROOT/data/paper_trades.db"
    echo "  OK"
else
    echo "[1/3] paper_trades.db not in backup — skipping"
fi

if [ -f "$BACKUP_SRC/brain.json" ]; then
    echo "[2/3] Restoring brain.json..."
    mkdir -p "$REPO_ROOT/state"
    cp "$BACKUP_SRC/brain.json" "$REPO_ROOT/state/brain.json"
    # Validate JSON
    python -c "import json; json.load(open('$REPO_ROOT/state/brain.json'))" 2>/dev/null && echo "  OK (valid JSON)" || echo "  WARNING: brain.json may be corrupt"
else
    echo "[2/3] brain.json not in backup — skipping"
fi

if [ -f "$BACKUP_SRC/kill_switch_state.json" ]; then
    echo "[3/3] Restoring kill_switch_state.json..."
    mkdir -p "$REPO_ROOT/logs"
    cp "$BACKUP_SRC/kill_switch_state.json" "$REPO_ROOT/logs/kill_switch_state.json"
    echo "  OK"
else
    echo "[3/3] kill_switch_state.json not in backup — skipping"
fi

echo ""
echo "============================================"
echo "  Restore complete!"
echo "  Safety backup: $SAFETY_DIR"
echo "  Run 'python -c \"import brain.persistence\"' to validate"
echo "============================================"
