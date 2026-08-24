"""Log rotation manager for Eclipse Scalper JSONL files.

Compresses old JSONL logs with gzip and archives them.
Designed to run periodically (e.g. daily via cron or guardian hook).

Usage:
    python -m monitoring.log_rotation [--dry-run] [--max-age-days 180]

Environment:
    LOG_ROTATION_MAX_SIZE_MB   — rotate when file exceeds this (default 50)
    LOG_ROTATION_MAX_AGE_DAYS  — delete archives older than this (default 180)
    LOG_ROTATION_DRY_RUN       — 1 to preview without acting
"""
from __future__ import annotations

import gzip
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

LOGS_DIR = Path("logs")
ARCHIVE_DIR = LOGS_DIR / "archive"

# Files to manage — only JSONL that grow unbounded
MANAGED_PATTERNS = [
    "*.jsonl",
]

# Sub-directories to also scan
MANAGED_SUBDIRS = [
    "health",
]

DEFAULT_MAX_SIZE_MB = 50
DEFAULT_MAX_AGE_DAYS = 180


def _size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0


def _file_age_days(path: Path) -> float:
    try:
        return (time.time() - path.stat().st_mtime) / 86400
    except Exception:
        return 0.0


def find_rotatable(max_size_mb: float = DEFAULT_MAX_SIZE_MB) -> List[Tuple[Path, float]]:
    """Find JSONL files exceeding max_size_mb."""
    results = []
    dirs = [LOGS_DIR] + [LOGS_DIR / d for d in MANAGED_SUBDIRS]
    for d in dirs:
        if not d.is_dir():
            continue
        for pattern in MANAGED_PATTERNS:
            for f in d.glob(pattern):
                if f.is_file() and not f.name.endswith(".gz"):
                    sz = _size_mb(f)
                    if sz >= max_size_mb:
                        results.append((f, sz))
    return sorted(results, key=lambda x: -x[1])


def rotate_file(path: Path, dry_run: bool = False) -> Path | None:
    """Compress a JSONL file to archive/{name}.{timestamp}.jsonl.gz.

    Returns the archive path on success, None on skip/error.
    """
    if not path.is_file():
        return None

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_name = f"{path.stem}.{ts}{path.suffix}.gz"
    archive_path = ARCHIVE_DIR / archive_name

    if dry_run:
        print(f"  [DRY-RUN] would rotate {path} ({_size_mb(path):.1f} MB) -> {archive_path}")
        return archive_path

    try:
        with open(path, "rb") as f_in, gzip.open(archive_path, "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)

        # Truncate the original (don't delete — processes may hold handles)
        with open(path, "w", encoding="utf-8") as f:
            f.write("")

        print(f"  rotated {path} ({_size_mb(archive_path):.1f} MB compressed) -> {archive_path}")
        return archive_path
    except Exception as e:
        print(f"  ERROR rotating {path}: {e}")
        return None


def cleanup_old_archives(max_age_days: float = DEFAULT_MAX_AGE_DAYS, dry_run: bool = False) -> int:
    """Delete archives older than max_age_days. Returns count deleted."""
    if not ARCHIVE_DIR.is_dir():
        return 0
    deleted = 0
    for f in ARCHIVE_DIR.glob("*.gz"):
        if not f.is_file():
            continue
        age = _file_age_days(f)
        if age > max_age_days:
            if dry_run:
                print(f"  [DRY-RUN] would delete {f.name} (age={age:.0f}d)")
            else:
                try:
                    f.unlink()
                    print(f"  deleted {f.name} (age={age:.0f}d)")
                    deleted += 1
                except Exception as e:
                    print(f"  ERROR deleting {f.name}: {e}")
    return deleted


def run_rotation(
    max_size_mb: float = DEFAULT_MAX_SIZE_MB,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    dry_run: bool = False,
) -> dict:
    """Run full rotation cycle. Returns summary dict."""
    print(f"Log rotation: max_size={max_size_mb}MB, max_age={max_age_days}d, dry_run={dry_run}")

    rotatable = find_rotatable(max_size_mb)
    rotated = 0
    for path, sz in rotatable:
        result = rotate_file(path, dry_run=dry_run)
        if result is not None:
            rotated += 1

    deleted = cleanup_old_archives(max_age_days, dry_run=dry_run)

    summary = {
        "checked_at": time.time(),
        "files_found": len(rotatable),
        "files_rotated": rotated,
        "archives_deleted": deleted,
        "dry_run": dry_run,
    }
    print(f"Log rotation complete: {rotated} rotated, {deleted} archives deleted")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Eclipse Scalper log rotation")
    parser.add_argument("--dry-run", action="store_true", help="Preview without acting")
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=float(os.getenv("LOG_ROTATION_MAX_SIZE_MB", str(DEFAULT_MAX_SIZE_MB))),
        help=f"Rotate files larger than this (default {DEFAULT_MAX_SIZE_MB})",
    )
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=float(os.getenv("LOG_ROTATION_MAX_AGE_DAYS", str(DEFAULT_MAX_AGE_DAYS))),
        help=f"Delete archives older than this (default {DEFAULT_MAX_AGE_DAYS})",
    )
    args = parser.parse_args()

    if os.getenv("LOG_ROTATION_DRY_RUN", "").strip().lower() in ("1", "true", "yes"):
        args.dry_run = True

    run_rotation(
        max_size_mb=args.max_size_mb,
        max_age_days=args.max_age_days,
        dry_run=args.dry_run,
    )
