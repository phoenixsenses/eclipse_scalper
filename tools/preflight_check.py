from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from tools.run_summary import build_run_summary

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preflight gate before paper/live bootstrap.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--trade-db", default="data/paper_trades.db")
    p.add_argument("--max-db-stale-sec", type=float, default=1800.0)
    p.add_argument("--min-free-gb", type=float, default=2.0)
    p.add_argument("--out-json", default="reports/PREFLIGHT_CHECK.json")
    p.add_argument("--out-md", default="reports/PREFLIGHT_CHECK.md")
    return p.parse_args()


def _check_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path), encoding="utf-8") as f:
            f.write("ok")
            tmp = Path(f.name)
        tmp.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _read_max_ts(db: Path) -> int:
    if not db.exists():
        return 0
    conn = sqlite3.connect(str(db), check_same_thread=False)
    try:
        cols = conn.execute("PRAGMA table_info(mark_prices)").fetchall()
        names = {str(r[1]).lower(): str(r[1]) for r in cols if len(r) > 1}
        ts_col = (
            names.get("ts_ms")
            or names.get("timestamp_ms")
            or names.get("ts_utc")
            or names.get("ts")
            or names.get("timestamp")
            or names.get("event_ts")
            or names.get("time")
        )
        if not ts_col:
            return 0
        cur = conn.execute(f"SELECT MAX({ts_col}) FROM mark_prices")
        row = cur.fetchone()
        if not row or row[0] is None:
            return 0
        raw = row[0]
        try:
            v = float(raw)
        except Exception:
            try:
                # Text timestamp fallback.
                from datetime import datetime

                s = str(raw).replace("Z", "+00:00")
                v = float(datetime.fromisoformat(s).timestamp())
            except Exception:
                return 0
        if v < 10_000_000_000:
            return int(v * 1000.0)
        return int(v)
    except Exception:
        return 0
    finally:
        conn.close()


def main() -> int:
    args = _parse_args()
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    failures: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, Any] = {}

    dry = str(os.getenv("SCALPER_DRY_RUN", "")).strip()
    checks["SCALPER_DRY_RUN"] = dry
    if dry != "1":
        failures.append("SCALPER_DRY_RUN must be 1 for paper startup")

    active_symbols = str(os.getenv("ACTIVE_SYMBOLS", "")).strip()
    checks["ACTIVE_SYMBOLS"] = active_symbols
    if not active_symbols:
        warnings.append("ACTIVE_SYMBOLS is empty; fallback config will be used")

    token = str(os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("ECLIPSE_TG_BOT_TOKEN") or "").strip()
    chat = str(os.getenv("TELEGRAM_CHAT_ID") or os.getenv("ECLIPSE_TG_CHAT_ID") or "").strip()
    checks["telegram_token_present"] = bool(token)
    checks["telegram_chat_present"] = bool(chat)
    if bool(token) != bool(chat):
        failures.append("Telegram token/chat mismatch: provide both or neither")

    db = Path(str(args.db))
    checks["db_exists"] = db.exists()
    if not db.exists():
        failures.append(f"Missing DB: {db}")
        db_age_sec = float("inf")
    else:
        max_ts = _read_max_ts(db)
        if max_ts <= 0:
            warnings.append("Could not read mark_prices MAX(ts); freshness unknown")
            db_age_sec = float("inf")
        else:
            db_age_sec = max(0.0, (time.time() * 1000.0 - float(max_ts)) / 1000.0)
        checks["db_age_sec"] = db_age_sec
        if db_age_sec > float(args.max_db_stale_sec):
            failures.append(f"DB stale: age_sec={db_age_sec:.1f} > max_db_stale_sec={float(args.max_db_stale_sec):.1f}")

    for d in (Path("logs"), Path("reports"), Path("data")):
        ok = _check_writable(d)
        checks[f"writable_{d.name}"] = bool(ok)
        if not ok:
            failures.append(f"Directory not writable: {d}")

    du = shutil.disk_usage(str(Path(".").resolve()))
    free_gb = float(du.free) / (1024.0 ** 3)
    checks["disk_free_gb"] = free_gb
    if free_gb < float(args.min_free_gb):
        failures.append(f"Low disk space: free_gb={free_gb:.2f} < min_free_gb={float(args.min_free_gb):.2f}")

    payload = {
        "ok": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
        "checks": checks,
    }
    payload["run_summary"] = build_run_summary(
        run_type="preflight_check",
        inputs={
            "db": str(args.db),
            "trade_db": str(args.trade_db),
            "max_db_stale_sec": float(args.max_db_stale_sec),
            "min_free_gb": float(args.min_free_gb),
        },
        metrics={
            "ok": bool(payload["ok"]),
            "failure_count": len(failures),
            "warning_count": len(warnings),
        },
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    out_json.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    md = ["# Preflight Check", ""]
    md.append(f"- ok: `{int(payload['ok'])}`")
    if failures:
        md.append("- failures:")
        for x in failures:
            md.append(f"  - {x}")
    if warnings:
        md.append("- warnings:")
        for x in warnings:
            md.append(f"  - {x}")
    md.append("")
    md.append("## Checks")
    for k in sorted(checks.keys()):
        md.append(f"- `{k}`: `{checks[k]}`")
    md.extend(["", "## Run Summary", f"- `{payload.get('run_summary', {})}`"])
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"preflight_check: wrote {out_md}")
    if failures:
        for x in failures:
            print(f"preflight_check: FAIL {x}")
        return 1
    for x in warnings:
        print(f"preflight_check: WARN {x}")
    print("preflight_check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
