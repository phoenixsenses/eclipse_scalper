from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prototype audit of websocket path vs DB-read path latency.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--collector-heartbeat", default="logs/collector_heartbeat.json")
    p.add_argument("--out-md", default="reports/WS_VS_DB_LATENCY_PROTOTYPE.md")
    p.add_argument("--out-json", default="reports/WS_VS_DB_LATENCY_PROTOTYPE.json")
    return p.parse_args()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if x != x:
            return float(default)
        return x
    except Exception:
        return float(default)


def _db_latest_ts_ms(db_path: Path, symbol: str) -> int:
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        row = conn.execute(
            "SELECT MAX(ts_ms) FROM agg_trades WHERE symbol=?",
            (str(symbol).upper(),),
        ).fetchone()
        return int(_safe_float((row[0] if row else 0), 0.0))
    except Exception:
        return 0
    finally:
        conn.close()


def main() -> int:
    args = _parse_args()
    out_md = Path(str(args.out_md))
    out_json = Path(str(args.out_json))
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    hb_path = Path(str(args.collector_heartbeat))
    hb = {}
    if hb_path.exists():
        try:
            hb = json.loads(hb_path.read_text(encoding="utf-8"))
        except Exception:
            hb = {}

    now_ms = int(time.time() * 1000.0)
    db_ts = _db_latest_ts_ms(Path(str(args.db)), str(args.symbol))
    db_lag_sec = ((now_ms - db_ts) / 1000.0) if db_ts > 0 else None
    progress_lag = hb.get("progress_lag_sec", None)
    ws_connected = bool(hb.get("connected", False))

    # Rough improvement estimate: DB-read lag minus collector progress lag.
    est_gain_sec = None
    if db_lag_sec is not None and progress_lag is not None:
        est_gain_sec = max(0.0, float(db_lag_sec) - float(progress_lag))

    payload = {
        "status": "ok",
        "symbol": str(args.symbol).upper(),
        "db_path": str(args.db),
        "collector_heartbeat_path": str(args.collector_heartbeat),
        "collector_connected": ws_connected,
        "collector_progress_lag_sec": progress_lag,
        "db_latest_ts_ms": db_ts,
        "db_lag_sec": db_lag_sec,
        "estimated_ws_bypass_gain_sec": est_gain_sec,
        "notes": (
            "Estimated gain is approximate. "
            "A true WS path requires in-process feature computation directly from stream callbacks."
        ),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# WS vs DB Latency Prototype",
        "",
        f"- symbol: `{payload['symbol']}`",
        f"- collector_connected: `{int(bool(ws_connected))}`",
        f"- collector_progress_lag_sec: `{payload['collector_progress_lag_sec']}`",
        f"- db_lag_sec: `{payload['db_lag_sec']}`",
        f"- estimated_ws_bypass_gain_sec: `{payload['estimated_ws_bypass_gain_sec']}`",
        "",
        "## Interpretation",
        "- If `db_lag_sec` is consistently > 2s, feature staleness is materially high for microstructure triggers.",
        "- If estimated gain is large, prioritize direct-WS feature pipeline prototype.",
    ]
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"prototype_ws_vs_db_latency: wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

