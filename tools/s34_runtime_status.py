from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "microstructure.db"
WINDOW_PATH = ROOT / "reports" / "runtime_validation" / "s34_liq_restore_window.json"
COLLECTOR_HEARTBEAT = ROOT / "logs" / "collector_heartbeat.json"
COLLECTOR_HEALTH = ROOT / "logs" / "health" / "collector.json"
WATCHDOG_STATUS = ROOT / "reports" / "WATCHDOG_STATUS.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _scalar(con: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> Any:
    row = con.execute(sql, params).fetchone()
    return row[0] if row else None


def _db_counts(db_path: Path, since_ms: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        for table in ("agg_trades", "mark_prices", "liquidations"):
            max_ts = _scalar(con, f"SELECT MAX(ts_ms) FROM {table}")
            total = None
            by_symbol = []
            if table == "liquidations":
                total = _scalar(con, f"SELECT COUNT(*) FROM {table} WHERE ts_ms >= ?", (since_ms,))
                by_symbol = con.execute(
                    f"SELECT symbol, COUNT(*), MAX(ts_ms) FROM {table} WHERE ts_ms >= ? GROUP BY symbol ORDER BY COUNT(*) DESC LIMIT 20",
                    (since_ms,),
                ).fetchall()
            out[table] = {
                "count_since_window": (None if total is None else int(total or 0)),
                "latest_ts_ms": int(max_ts or 0),
                "latest_ts_utc": (
                    datetime.fromtimestamp(float(max_ts) / 1000.0, timezone.utc).isoformat()
                    if max_ts
                    else None
                ),
                "by_symbol_since_window": [
                    {
                        "symbol": str(sym),
                        "count": int(count or 0),
                        "latest_ts_utc": (
                            datetime.fromtimestamp(float(latest) / 1000.0, timezone.utc).isoformat()
                            if latest
                            else None
                        ),
                    }
                    for sym, count, latest in by_symbol
                ],
            }
    finally:
        con.close()
    return out


def init_window(args: argparse.Namespace) -> int:
    now = _utc_now()
    payload = {
        "created_at_utc": now.isoformat(),
        "window_start_utc": args.start_utc or now.isoformat(),
        "milestones": {
            "24h_due_utc": None,
            "72h_due_utc": None,
        },
        "purpose": "S34 liquidation transport restored runtime validation",
        "requirements": {
            "collector_alive": True,
            "watchdog_overall": "GREEN",
            "liquidations_increasing": True,
            "rest_fallback_active": False,
            "liquidation_transport_available": True,
        },
    }
    start = _parse_utc(str(payload["window_start_utc"])) or now
    payload["milestones"]["24h_due_utc"] = datetime.fromtimestamp(start.timestamp() + 24 * 3600, timezone.utc).isoformat()
    payload["milestones"]["72h_due_utc"] = datetime.fromtimestamp(start.timestamp() + 72 * 3600, timezone.utc).isoformat()
    _write_json(WINDOW_PATH, payload)
    print(json.dumps(payload, indent=2))
    return 0


def status(args: argparse.Namespace) -> int:
    now = _utc_now()
    window = _read_json(WINDOW_PATH)
    start = _parse_utc(str(window.get("window_start_utc") or ""))
    if start is None:
        start = now
    since_ms = int(start.timestamp() * 1000)
    elapsed_h = max(0.0, (now - start).total_seconds() / 3600.0)
    remaining_24 = max(0.0, 24.0 - elapsed_h)
    remaining_72 = max(0.0, 72.0 - elapsed_h)

    heartbeat = _read_json(COLLECTOR_HEARTBEAT)
    health = _read_json(COLLECTOR_HEALTH)
    watchdog = _read_json(WATCHDOG_STATUS)
    counts = _db_counts(Path(args.db), since_ms)

    hb_ts = _parse_utc(str(heartbeat.get("ts_utc") or ""))
    hb_age = None if hb_ts is None else max(0.0, (now - hb_ts).total_seconds())

    result = {
        "ts_utc": now.isoformat(),
        "window_start_utc": start.isoformat(),
        "elapsed_hours": round(elapsed_h, 3),
        "remaining_24h": round(remaining_24, 3),
        "remaining_72h": round(remaining_72, 3),
        "collector": {
            "heartbeat_age_sec": None if hb_age is None else round(hb_age, 1),
            "connected": bool(heartbeat.get("connected")),
            "last_message_ts_utc": heartbeat.get("last_message_ts_utc"),
            "rows_written_since_start": heartbeat.get("rows_written_since_start") or {},
            "rest_fallback_active": bool(heartbeat.get("rest_fallback_active")),
            "liquidation_stream_mode": heartbeat.get("liquidation_stream_mode"),
            "liquidation_transport_available": bool(health.get("liquidation_transport_available")),
            "health_status": health.get("status"),
        },
        "watchdog_overall": watchdog.get("overall"),
        "watchdog_issues": watchdog.get("issues") or [],
        "db_counts_since_window": counts,
    }
    ok = (
        result["collector"]["connected"]
        and result["collector"]["liquidation_transport_available"]
        and result["collector"]["health_status"] == "ok"
        and result["watchdog_overall"] == "GREEN"
        and counts.get("liquidations", {}).get("count_since_window", 0) > 0
    )
    result["overall"] = "GREEN" if ok else "YELLOW"
    print(json.dumps(result, indent=2))
    return 0 if ok else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="S34 restored liquidation runtime status")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_init = sub.add_parser("init-window")
    p_init.add_argument("--start-utc", default="", help="Override window start timestamp.")
    p_status = sub.add_parser("status")
    p_status.add_argument("--db", default=str(DEFAULT_DB), help="Microstructure DB path.")
    args = parser.parse_args()
    if args.cmd == "init-window":
        return init_window(args)
    if args.cmd == "status":
        return status(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
