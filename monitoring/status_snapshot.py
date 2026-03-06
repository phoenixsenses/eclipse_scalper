from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from tools.paper_trade_summary import generate_summary


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _query_one(db_path: str, sql: str) -> Any:
    p = Path(str(db_path))
    if not p.exists():
        return None
    conn = sqlite3.connect(str(p))
    try:
        row = conn.execute(sql).fetchone()
        return (row[0] if row else None)
    finally:
        conn.close()


def _latest_micro_ts(db_path: str = "data/microstructure.db") -> float | None:
    q = "SELECT MAX(ts_ms) FROM mark_prices"
    v = _query_one(db_path, q)
    if v is None:
        return None
    fv = float(v)
    return (fv / 1000.0) if fv > 1e12 else fv


def collect_pnl(db_path: str = "data/paper_trades.db") -> Dict[str, Any]:
    p = Path(str(db_path))
    if not p.exists():
        return {"ok": False, "reason": "paper_trades_db_missing"}
    s = generate_summary(str(p), days=1)
    t = generate_summary(str(p), days=0)
    return {
        "ok": True,
        "today_trades": int(s.get("total_trades", 0) or 0),
        "today_win_rate": float(s.get("win_rate", 0.0) or 0.0),
        "today_pnl_bps": float(s.get("total_pnl_bps", 0.0) or 0.0),
        "max_drawdown_bps": float(t.get("max_drawdown_bps", 0.0) or 0.0),
        "total_trades": int(t.get("total_trades", 0) or 0),
        "total_win_rate": float(t.get("win_rate", 0.0) or 0.0),
        "total_pnl_bps": float(t.get("total_pnl_bps", 0.0) or 0.0),
    }


def collect_last_decisions(journal_path: str = "logs/execution_journal.jsonl", limit: int = 5) -> List[Dict[str, Any]]:
    p = Path(str(journal_path))
    if not p.exists():
        return []
    # Read only the tail of the file to avoid loading GBs into memory.
    _TAIL_BYTES = 256 * 1024  # 256 KB is enough for recent decisions
    try:
        file_size = p.stat().st_size
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            if file_size > _TAIL_BYTES:
                fh.seek(file_size - _TAIL_BYTES)
                fh.readline()  # skip partial first line
            tail_lines = fh.readlines()
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for raw in reversed(tail_lines):
        if len(out) >= int(limit):
            break
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        evt = str(obj.get("event") or "")
        if evt in ("entry.blocked", "entry.submitted", "position.closed", "exit.risk_action", "alpha_gate_check"):
            data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
            out.append(
                {
                    "ts": str(obj.get("ts") or ""),
                    "event": evt,
                    "symbol": str(obj.get("symbol") or data.get("symbol") or ""),
                    "reason": str(data.get("reason") or data.get("risk_reason") or ""),
                    "level": str(obj.get("level") or ""),
                }
            )
    return out


def collect_open_positions(brain_path: str = "state/brain.json") -> Dict[str, Any]:
    p = Path(str(brain_path))
    if not p.exists():
        return {"ok": False, "count": 0, "positions": []}
    try:
        obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {"ok": False, "count": 0, "positions": []}
    pos = obj.get("positions")
    if not isinstance(pos, dict):
        return {"ok": False, "count": 0, "positions": []}
    rows = []
    for sym, v in pos.items():
        if not isinstance(v, dict):
            continue
        qty = float(v.get("size") or v.get("qty") or 0.0)
        if abs(qty) <= 0:
            continue
        rows.append(
            {
                "symbol": str(sym),
                "side": str(v.get("side") or ""),
                "qty": qty,
                "entry_price": float(v.get("entry_price") or 0.0),
                "entry_ts": float(v.get("entry_ts") or 0.0),
            }
        )
    return {"ok": True, "count": len(rows), "positions": rows}


def collect_config_flags() -> Dict[str, Any]:
    keys = [
        "ENTRY_REGIME",
        "ENTRY_REGIME_RISK_ENABLED",
        "EXIT_SCRATCH_ENABLED",
        "ENTRY_TRADE_LOGGER_ENABLED",
        "EXIT_TRADE_LOGGER_ENABLED",
        "NOTIFY_ENABLED",
        "NOTIFY_TRADE_ALERTS",
        "NOTIFY_RISK_ALERTS",
        "NOTIFY_DAILY_SUMMARY",
        "NOTIFY_HEARTBEAT",
    ]
    return {k: str(os.getenv(k, "")) for k in keys}


def collect_diag(micro_db: str = "data/microstructure.db") -> Dict[str, Any]:
    now = time.time()
    ts = _latest_micro_ts(micro_db)
    if ts is None:
        return {"ok": False, "micro_db": micro_db, "feed_age_sec": None}
    return {
        "ok": True,
        "micro_db": micro_db,
        "last_tick_utc": _utc(ts),
        "feed_age_sec": int(max(0.0, now - ts)),
    }


def collect_kill_switch_status() -> Dict[str, Any]:
    ks_path = Path("state/kill_switch.json")
    if not ks_path.exists():
        return {"active": False, "source": "no_file"}
    try:
        obj = json.loads(ks_path.read_text(encoding="utf-8", errors="replace"))
        return {
            "active": bool(obj.get("halted", False)),
            "reason": str(obj.get("reason") or ""),
            "ts": float(obj.get("ts") or 0),
            "source": "state_file",
        }
    except Exception:
        return {"active": False, "source": "read_error"}


def collect_status() -> Dict[str, Any]:
    pnl = collect_pnl()
    diag = collect_diag()
    cfg = collect_config_flags()
    positions = collect_open_positions()
    decisions = collect_last_decisions(limit=5)
    kill_switch = collect_kill_switch_status()
    return {
        "pnl": pnl,
        "diag": diag,
        "config": cfg,
        "positions": positions,
        "last_decisions": decisions,
        "kill_switch": kill_switch,
    }


def render_status_text() -> str:
    s = collect_status()
    pnl = s["pnl"]
    diag = s["diag"]
    cfg = s["config"]
    positions = s["positions"]
    decisions = s["last_decisions"]
    ks = s["kill_switch"]

    lines = ["Eclipse Scalper - Health Check", "============================="]

    # Kill switch
    if ks.get("active"):
        lines.append(f"KILL SWITCH: ACTIVE — {ks.get('reason', 'unknown')}")
    else:
        lines.append("Kill switch: inactive")

    # Data feed
    if diag.get("ok"):
        age = int(diag.get("feed_age_sec", 0))
        freshness = "FRESH" if age < 30 else ("DEGRADED" if age < 120 else "STALE")
        lines.append(f"Data feed: {freshness} | Last tick: {diag.get('last_tick_utc')} | age={age}s")
    else:
        lines.append("Data feed: UNAVAILABLE")

    # PnL
    if pnl.get("ok"):
        lines.append(
            f"PnL today: {float(pnl.get('today_pnl_bps', 0.0)):+.2f} bps | Trades: {int(pnl.get('today_trades', 0))} | Win: {float(pnl.get('today_win_rate', 0.0))*100.0:.1f}%"
        )
        lines.append(
            f"Total: {float(pnl.get('total_pnl_bps', 0.0)):+.2f} bps | Trades: {int(pnl.get('total_trades', 0))} | DD: -{float(pnl.get('max_drawdown_bps', 0.0)):.2f} bps"
        )
    else:
        lines.append("PnL: unavailable")

    # Positions
    pos_count = int(positions.get("count", 0))
    if pos_count > 0:
        lines.append(f"Open positions: {pos_count}")
        for p in positions.get("positions", [])[:5]:
            lines.append(f"  {p.get('symbol')} {p.get('side')} qty={p.get('qty')} entry={p.get('entry_price')}")
    else:
        lines.append("Open positions: 0")

    # Last decisions
    if decisions:
        lines.append("Recent decisions:")
        for d in decisions[:3]:
            lines.append(f"  [{d.get('event')}] {d.get('symbol')} — {d.get('reason', '')}"[:80])

    # Config flags
    lines.append("Flags: regime={ENTRY_REGIME} risk={ENTRY_REGIME_RISK_ENABLED} scratch={EXIT_SCRATCH_ENABLED} notify={NOTIFY_ENABLED}".format(**cfg))
    return "\n".join(lines)


if __name__ == "__main__":
    print(render_status_text())

