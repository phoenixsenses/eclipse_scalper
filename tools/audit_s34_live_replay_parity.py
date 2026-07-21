from __future__ import annotations

import copy
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_shadow_paper_runner import DEFAULT_RISK, _evaluate_trade, _risk_payload, _rule_from_trade

DB = f"file:{(ROOT / 'data' / 'microstructure.db').as_posix()}?mode=ro"
TRADES_PATH = ROOT / "reports" / "research" / "s34" / "S34_SHADOW_PAPER_TRADES.json"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_LIVE_REPLAY_PARITY_AUDIT.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_LIVE_REPLAY_PARITY_AUDIT.md"

RECENT_IDS = {"P187", "P188", "P189", "P191", "P192", "P206", "P217", "P326", "P328", "P330"}
EXCLUDED_IDS = {"P013", "P056"}


def load_trades() -> list[dict[str, Any]]:
    payload = json.loads(TRADES_PATH.read_text(encoding="utf-8"))
    return payload.get("trades", []) if isinstance(payload, dict) else []


def reset_for_replay(live: dict[str, Any]) -> dict[str, Any]:
    trade = copy.deepcopy(live)
    rule = _rule_from_trade(trade)
    trade["status"] = "OPEN"
    trade["closed_at_utc"] = None
    trade["exit_ts_ms"] = None
    trade["exit_ts_utc"] = None
    trade["exit_reference_price"] = None
    trade["exit_price"] = None
    trade["exit_fill"] = None
    trade["exit_reason"] = None
    trade["gross_bps"] = None
    trade["mid_to_mid_bps"] = None
    trade["executable_gross_bps"] = None
    trade["entry_adverse_bps"] = None
    trade["exit_adverse_bps"] = None
    trade["mark_to_fill_cost_bps"] = None
    trade["spread_cost_bps"] = None
    trade["entry_spread_bps"] = None
    trade["exit_spread_bps"] = None
    trade["fee_cost_bps"] = None
    trade["net_bps"] = None
    trade["gross_usdt"] = None
    trade["net_usdt"] = None
    trade["be_active"] = False
    trade["be_activated_ts_ms"] = None
    trade["be_activated_ts_utc"] = None
    trade["last_evaluated_mark_ts_ms"] = None
    trade["last_evaluated_mark_ts_utc"] = None
    trade["risk"] = _risk_payload(rule, DEFAULT_RISK)
    return trade


def replay_trade(conn: sqlite3.Connection, live: dict[str, Any]) -> dict[str, Any]:
    replay = reset_for_replay(live)
    rule = replay.get("rule") or {}
    end_ms = int(replay["entry_ts_ms"]) + int(rule.get("max_horizon_sec") or 3600) * 1000
    replay = _evaluate_trade(conn, replay, end_ms)
    return replay


def cmp_float(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def audit_row(conn: sqlite3.Connection, live: dict[str, Any]) -> dict[str, Any]:
    replay = replay_trade(conn, live)
    diffs = {
        "exit_ts_ms": None if live.get("exit_ts_ms") == replay.get("exit_ts_ms") else [live.get("exit_ts_ms"), replay.get("exit_ts_ms")],
        "exit_reason": None if live.get("exit_reason") == replay.get("exit_reason") else [live.get("exit_reason"), replay.get("exit_reason")],
        "entry_price_diff": cmp_float(live.get("entry_price"), replay.get("entry_price")),
        "exit_price_diff": cmp_float(live.get("exit_price"), replay.get("exit_price")),
        "gross_bps_diff": cmp_float(live.get("gross_bps"), replay.get("gross_bps")),
        "net_bps_diff": cmp_float(live.get("net_bps"), replay.get("net_bps")),
        "entry_adverse_diff": cmp_float(live.get("entry_adverse_bps"), replay.get("entry_adverse_bps")),
        "exit_adverse_diff": cmp_float(live.get("exit_adverse_bps"), replay.get("exit_adverse_bps")),
    }
    ok = True
    reasons = []
    if diffs["exit_ts_ms"] is not None:
        ok = False
        reasons.append("exit_ts_mismatch")
    if diffs["exit_reason"] is not None:
        ok = False
        reasons.append("exit_reason_mismatch")
    for key in ("entry_price_diff", "exit_price_diff", "gross_bps_diff", "net_bps_diff", "entry_adverse_diff", "exit_adverse_diff"):
        value = diffs[key]
        if value is not None and abs(float(value)) > 1e-6:
            ok = False
            reasons.append(key)
    return {
        "trade_id": live.get("trade_id"),
        "rule": (live.get("rule") or {}).get("name"),
        "symbol": live.get("symbol"),
        "live": {
            "entry_ts_utc": live.get("entry_ts_utc"),
            "exit_ts_utc": live.get("exit_ts_utc"),
            "exit_reason": live.get("exit_reason"),
            "entry_price": live.get("entry_price"),
            "exit_price": live.get("exit_price"),
            "gross_bps": live.get("gross_bps"),
            "net_bps": live.get("net_bps"),
        },
        "replay": {
            "entry_ts_utc": replay.get("entry_ts_utc"),
            "exit_ts_utc": replay.get("exit_ts_utc"),
            "exit_reason": replay.get("exit_reason"),
            "entry_price": replay.get("entry_price"),
            "exit_price": replay.get("exit_price"),
            "gross_bps": replay.get("gross_bps"),
            "net_bps": replay.get("net_bps"),
        },
        "diffs": diffs,
        "ok": ok,
        "reasons": reasons,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def write_report(rows: list[dict[str, Any]]) -> None:
    OUT_JSON.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=True), encoding="utf-8")
    ok_count = sum(1 for row in rows if row["ok"])
    lines = [
        "# S34 Live vs Replay Parity Audit",
        "",
        "Scope: replay selected closed live paper trades through the current runner evaluation path using the stored live signal/trade snapshot.",
        "",
        f"- Audited trades: `{len(rows)}`",
        f"- Exact parity: `{ok_count}/{len(rows)}`",
        "",
        "| Trade | Rule | Live Exit | Replay Exit | Live Net | Replay Net | Net Diff | Verdict | Reasons |",
        "|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        live = row["live"]
        replay = row["replay"]
        net_diff = row["diffs"]["net_bps_diff"]
        lines.append(
            f"| {row['trade_id']} | {row['rule']} | {live['exit_reason']} {live['exit_ts_utc']} | "
            f"{replay['exit_reason']} {replay['exit_ts_utc']} | {float(live['net_bps'] or 0):+.6f} | "
            f"{float(replay['net_bps'] or 0):+.6f} | {fmt(net_diff)} | "
            f"{'OK' if row['ok'] else 'MISMATCH'} | {', '.join(row['reasons'])} |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "A mismatch here means the live journal and current replay path do not agree for the same stored signal/trade snapshot. That does not automatically mean the trade PnL is wrong, but it blocks using research replay and live paper as interchangeable evidence until explained.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    trades = [
        t
        for t in load_trades()
        if t.get("status") == "CLOSED"
        and str(t.get("trade_id") or "") not in EXCLUDED_IDS
        and str(t.get("trade_id") or "") in RECENT_IDS
    ]
    conn = sqlite3.connect(DB, uri=True)
    conn.execute("pragma query_only=1")
    rows = [audit_row(conn, trade) for trade in trades]
    write_report(rows)
    print(json.dumps({"audited": len(rows), "ok": sum(1 for row in rows if row["ok"]), "out_md": str(OUT_MD)}, indent=2))
    conn.close()


if __name__ == "__main__":
    main()
