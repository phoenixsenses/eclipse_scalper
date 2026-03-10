#!/bin/bash
# ============================================================================
# Eclipse Scalper — Database Backup Script
# ============================================================================
# Backs up critical state files:
#   - data/paper_trades.db (SQLite)
#   - state/brain.json
#   - logs/kill_switch_state.json
#
# Usage:
#   bash scripts/backup.sh                    # Manual backup
#   bash scripts/backup.sh /path/to/dest      # Custom destination
#
# Cron example (every 6 hours):
#   0 */6 * * * cd /app && bash scripts/backup.sh
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_DIR="${1:-$REPO_ROOT/backups}"
TIMESTAMP=$(date -u '+%Y%m%d_%H%M%S')
BACKUP_NAME="eclipse_backup_${TIMESTAMP}"
DEST="$BACKUP_DIR/$BACKUP_NAME"

echo "============================================"
echo "  Eclipse Scalper — Backup"
echo "  Source: $REPO_ROOT"
echo "  Dest:   $DEST"
echo "  Time:   $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================"

mkdir -p "$DEST"

# 1. SQLite backup (safe, uses .backup command)
DB_FILE="$REPO_ROOT/data/paper_trades.db"
if [ -f "$DB_FILE" ]; then
    echo "[1/4] Backing up paper_trades.db..."
    sqlite3 "$DB_FILE" ".backup '$DEST/paper_trades.db'" 2>/dev/null || {
        # Fallback: plain copy if sqlite3 not available
        cp "$DB_FILE" "$DEST/paper_trades.db"
        echo "  (used file copy — sqlite3 not available)"
    }
    echo "  OK — $(du -h "$DEST/paper_trades.db" | cut -f1)"
else
    echo "[1/4] paper_trades.db not found — skipping"
fi

# 2. Brain state
BRAIN_FILE="$REPO_ROOT/state/brain.json"
if [ -f "$BRAIN_FILE" ]; then
    echo "[2/4] Backing up brain.json..."
    cp "$BRAIN_FILE" "$DEST/brain.json"
    echo "  OK — $(du -h "$DEST/brain.json" | cut -f1)"
else
    echo "[2/4] brain.json not found — skipping"
fi

# 3. Kill switch state
KS_FILE="$REPO_ROOT/logs/kill_switch_state.json"
if [ -f "$KS_FILE" ]; then
    echo "[3/4] Backing up kill_switch_state.json..."
    cp "$KS_FILE" "$DEST/kill_switch_state.json"
    echo "  OK"
else
    echo "[3/4] kill_switch_state.json not found — skipping"
fi

# 4. Metadata
echo "[4/4] Writing backup metadata..."
cat > "$DEST/backup_meta.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "source": "$REPO_ROOT",
    "files": [
        "paper_trades.db",
        "brain.json",
        "kill_switch_state.json"
    ]
}
EOF

# Cleanup old backups (keep last 10)
echo ""
echo "Cleaning old backups (keeping last 10)..."
ls -dt "$BACKUP_DIR"/eclipse_backup_* 2>/dev/null | tail -n +11 | xargs rm -rf 2>/dev/null || true

TOTAL_SIZE=$(du -sh "$DEST" 2>/dev/null | cut -f1)
echo ""
echo "============================================"
echo "  Backup complete: $DEST"
echo "  Total size: $TOTAL_SIZE"
echo "============================================"
