from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import time
from pathlib import Path


def _load_dotenv_best_effort() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    root = Path(__file__).resolve().parents[1]
    env_paper = root / ".env.paper"
    env_default = root / ".env"
    if env_paper.exists():
        load_dotenv(dotenv_path=env_paper, override=False)
    elif env_default.exists():
        load_dotenv(dotenv_path=env_default, override=False)


_load_dotenv_best_effort()


def _send_alert(msg: str) -> None:
    try:
        from notifications.telegram import Notifier  # type: ignore
        import asyncio
    except Exception:
        return
    token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("ECLIPSE_TG_BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("ECLIPSE_TG_CHAT_ID")
    if not token or not chat:
        return
    try:
        asyncio.run(Notifier(token=token, chat_id=chat).speak(msg, priority="normal", silent=True))
    except Exception:
        pass


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SQLite maintenance: checkpoint, backup, disk-space checks.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--trade-db", default="data/paper_trades.db")
    p.add_argument("--backup-dir", default="data/backups")
    p.add_argument("--keep", type=int, default=7)
    p.add_argument("--min-free-gb", type=float, default=10.0)
    p.add_argument("--max-wal-mb", type=float, default=2048.0)
    return p.parse_args()


def _checkpoint(db: Path) -> str:
    conn = sqlite3.connect(str(db), check_same_thread=False)
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        conn.commit()
        return str(row)
    finally:
        conn.close()


def _backup_file(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = dst_dir / f"{src.stem}_{ts}{src.suffix}"
    shutil.copy2(src, dst)
    return dst


def _prune_backups(dst_dir: Path, stem: str, keep: int) -> int:
    files = sorted(dst_dir.glob(f"{stem}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    removed = 0
    for f in files[int(max(1, keep)) :]:
        try:
            f.unlink(missing_ok=True)
            removed += 1
        except Exception:
            pass
    return removed


def _wal_path(db: Path) -> Path:
    return Path(str(db) + "-wal")


def main() -> int:
    args = _args()
    db = Path(args.db)
    trade_db = Path(args.trade_db)
    bdir = Path(args.backup_dir)

    if not db.exists():
        print(f"[db_maintenance] missing db: {db}")
        return 2

    ck = _checkpoint(db)
    print(f"[db_maintenance] checkpoint {db}: {ck}")
    try:
        b1 = _backup_file(db, bdir)
        print(f"[db_maintenance] backup: {b1}")
    except Exception as e:
        msg = f"DB MAINTENANCE ALERT: backup failed db={db} err={type(e).__name__}: {e}"
        print(msg)
        _send_alert(msg)
        return 2
    removed = _prune_backups(bdir, db.stem, int(args.keep))
    if removed:
        print(f"[db_maintenance] pruned {removed} old backups for {db.stem}")

    if trade_db.exists():
        try:
            b2 = _backup_file(trade_db, bdir)
            print(f"[db_maintenance] backup: {b2}")
        except Exception as e:
            msg = f"DB MAINTENANCE ALERT: backup failed db={trade_db} err={type(e).__name__}: {e}"
            print(msg)
            _send_alert(msg)
            return 2
        removed2 = _prune_backups(bdir, trade_db.stem, int(args.keep))
        if removed2:
            print(f"[db_maintenance] pruned {removed2} old backups for {trade_db.stem}")

    du = shutil.disk_usage(str(Path(".").resolve()))
    free_gb = float(du.free) / (1024.0 ** 3)
    print(f"[db_maintenance] disk_free_gb={free_gb:.2f}")
    if free_gb < float(args.min_free_gb):
        msg = f"DB MAINTENANCE ALERT: low disk space free={free_gb:.2f}GB threshold={float(args.min_free_gb):.2f}GB"
        print(msg)
        _send_alert(msg)
        return 1

    wal = _wal_path(db)
    wal_mb = (float(wal.stat().st_size) / (1024.0 ** 2)) if wal.exists() else 0.0
    print(f"[db_maintenance] wal_size_mb={wal_mb:.2f}")
    if wal_mb > float(args.max_wal_mb):
        msg = (
            "DB MAINTENANCE ALERT: WAL too large "
            f"path={wal} size_mb={wal_mb:.2f} threshold_mb={float(args.max_wal_mb):.2f}"
        )
        print(msg)
        _send_alert(msg)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
