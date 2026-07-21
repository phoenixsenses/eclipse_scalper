"""S34 v11 forward integrity and operator governance.

Observation/risk only. No live executor/config/order logic changes.

Outputs:
- Forward Sample Integrity Score (valid_N, partial/invalid reasons).
- Operator decision-journal enforcement.
- Execution Truth Ledger snapshot from telemetry.
- Fee-tier verification status.
- Kill-switch drill/readiness contract.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB = ROOT / "data" / "microstructure.db"
ENV = ROOT / ".env"
SHADOW_LEDGER = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_LEDGER.jsonl"
MGMT_LEDGER = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_FORWARD_MANAGEMENT_LEDGER.jsonl"
MGMT_READOUT = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_FORWARD_MANAGEMENT_READOUT.json"
V10 = ROOT / "reports" / "research" / "s34" / "S34_V10_OPERATIONAL_RISK_SUITE.json"
LIVE_STATE = ROOT / "runtime" / "s34_v_engine_live_state.json"
DECISION_JOURNAL = ROOT / "reports" / "research" / "s34" / "S34_OPERATOR_DECISION_JOURNAL.jsonl"
TELEMETRY = ROOT / "logs" / "telemetry.jsonl"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V11_FORWARD_GOVERNANCE.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V11_FORWARD_GOVERNANCE.md"

RULE = "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def parse_iso_ms(text: str | None) -> int | None:
    if not text:
        return None
    t = str(text)
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    return int(datetime.fromisoformat(t).timestamp() * 1000)


def age_sec(ts_ms: int | None) -> float | None:
    if ts_ms is None:
        return None
    return round((now_ms() - int(ts_ms)) / 1000.0, 1)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def tail_jsonl(path: Path, n: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    buf: deque[str] = deque(maxlen=n)
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            buf.append(line)
    rows = []
    for line in buf:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def parse_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def latest_ts(conn: sqlite3.Connection, table: str, symbol: str) -> int | None:
    try:
        row = conn.execute(f"SELECT MAX(ts_ms) FROM {table} WHERE symbol=?", (symbol,)).fetchone()
    except sqlite3.Error:
        return None
    return int(row[0]) if row and row[0] is not None else None


def row_quality(row: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = []
    if row.get("observation_status") != "CLOSED":
        reasons.append("not_closed")
    if row.get("sim_status") != "FILLED":
        reasons.append("not_filled")
    if row.get("net_bps") is None:
        reasons.append("missing_net_bps")
    if not row.get("dissipation_observer"):
        reasons.append("missing_dissipation")
    atomic = row.get("atomicity_gap_observer") or {}
    if atomic.get("status") not in {"OBSERVED", "NO_FILL_OR_NO_ENTRY"}:
        reasons.append("atomicity_incomplete")
    if row.get("sample_type") == "FORWARD_OOS" and reasons:
        return "INVALID", reasons
    if reasons:
        return "PARTIAL", reasons
    return "VALID", []


def forward_sample_integrity(shadow: list[dict[str, Any]], mgmt: list[dict[str, Any]], readout: dict[str, Any], db: Path) -> dict[str, Any]:
    shadow_ids = {r.get("observation_id") for r in shadow if r.get("observation_id")}
    mgmt_ids = {r.get("source_observation_id") for r in mgmt if r.get("source_observation_id")}
    cards = []
    counts: dict[str, int] = {"VALID": 0, "PARTIAL": 0, "INVALID": 0}
    for row in mgmt:
        quality, reasons = row_quality(row)
        counts[quality] = counts.get(quality, 0) + 1
        cards.append(
            {
                "source_observation_id": row.get("source_observation_id"),
                "signal_utc": row.get("signal_utc"),
                "sample_type": row.get("sample_type"),
                "quality": quality,
                "reasons": reasons,
            }
        )
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        book_age = age_sec(latest_ts(conn, "book_ticker", "ETHUSDT"))
        liq_age = age_sec(latest_ts(conn, "liquidations", "ETHUSDT"))
        mark_age = age_sec(latest_ts(conn, "mark_prices", "ETHUSDT"))
    forward_cards = [c for c in cards if c.get("sample_type") == "FORWARD_OOS"]
    valid_forward = [c for c in forward_cards if c["quality"] == "VALID"]
    return {
        "status": "NO_FORWARD_OOS_YET" if not forward_cards else ("OK" if len(valid_forward) == len(forward_cards) else "HAS_INVALID_FORWARD_ROWS"),
        "source_shadow_n": len(shadow),
        "management_n": len(mgmt),
        "missing_in_management_n": len(shadow_ids - mgmt_ids),
        "extra_management_n": len(mgmt_ids - shadow_ids),
        "quality_counts_all": counts,
        "forward_n": len(forward_cards),
        "valid_forward_n": len(valid_forward),
        "valid_forward_ratio": None if not forward_cards else round(len(valid_forward) / len(forward_cards), 3),
        "frozen_start_utc": readout.get("frozen_start_utc"),
        "data_freshness_sec": {"book": book_age, "liquidations": liq_age, "mark": mark_age},
        "latest_cards": cards[-10:],
    }


def governance_status(v10: dict[str, Any], journal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    modes = v10.get("risk_budget_modes") or {}
    current = modes.get("CURRENT_ENV") or {}
    balanced = modes.get("BALANCED") or {}
    current_notional = float(current.get("notional") or 0.0)
    balanced_notional = float(balanced.get("notional") or 0.0)
    oversize_vs_balanced = current_notional / balanced_notional if balanced_notional > 0 else None
    real_decisions = [r for r in journal_rows if r.get("event") != "decision_template" and r.get("decision")]
    latest = real_decisions[-1] if real_decisions else None
    latest_ms = parse_iso_ms(latest.get("ts_utc")) if latest else None
    latest_age_h = None if latest_ms is None else round((now_ms() - latest_ms) / 3_600_000.0, 2)
    decision_required = bool(oversize_vs_balanced and oversize_vs_balanced > 1.0 and (latest is None or (latest_age_h is not None and latest_age_h > 24.0)))
    return {
        "status": "DECISION_REQUIRED" if decision_required else "DECISION_CURRENT",
        "current_notional": current_notional,
        "balanced_notional": balanced_notional,
        "oversize_vs_balanced": round(oversize_vs_balanced, 1) if oversize_vs_balanced else None,
        "journal_rows": len(journal_rows),
        "real_decision_rows": len(real_decisions),
        "latest_decision": latest,
        "latest_decision_age_hours": latest_age_h,
        "required_action": "operator must log REDUCE_MARGIN / DISARM / ARMED_KEEP_SIZE rationale" if decision_required else "none",
    }


def execution_truth_ledger(telemetry_rows: list[dict[str, Any]]) -> dict[str, Any]:
    order_events = []
    for row in telemetry_rows:
        if str(row.get("event")) != "order.create":
            continue
        data = row.get("data") or {}
        text = json.dumps(data, sort_keys=True, ensure_ascii=True)
        if "S34_V_ENGINE" in text or RULE in text:
            order_events.append(row)
    cards = []
    for row in order_events[-20:]:
        data = row.get("data") or {}
        order = data.get("order") or {}
        info = order.get("info") or {}
        cards.append(
            {
                "ts": row.get("ts"),
                "symbol": row.get("symbol") or data.get("symbol"),
                "intent": data.get("intent"),
                "order_id": data.get("order_id") or order.get("id"),
                "client_id": info.get("clientOrderId") or info.get("clientAlgoId") or order.get("clientOrderId"),
                "type": info.get("type") or info.get("orderType") or order.get("type"),
                "side": info.get("side") or order.get("side"),
                "reduce_only": info.get("reduceOnly") or order.get("reduceOnly"),
                "fee": order.get("fee"),
            }
        )
    return {
        "status": "NO_S34_REAL_ORDER_TELEMETRY_FOUND" if not cards else "HAS_REAL_ORDER_TELEMETRY",
        "scanned_tail_rows": len(telemetry_rows),
        "s34_order_event_n": len(order_events),
        "latest_order_cards": cards,
        "missing_fields_to_instrument": [
            "local_send_ts",
            "exchange_ack_ts",
            "fill_ts",
            "stop_send_ts",
            "stop_ack_ts",
            "realized_fee_bps",
            "realized_stop_slippage_bps",
        ],
    }


def fee_tier_verification(env: dict[str, str], truth: dict[str, Any]) -> dict[str, Any]:
    fees = []
    for card in truth.get("latest_order_cards") or []:
        fee = card.get("fee")
        if fee:
            fees.append(fee)
    return {
        "status": "ACTUAL_FEE_TIER_UNKNOWN" if not fees else "HAS_ORDER_FEE_SAMPLES",
        "assumed_maker_fee_bps": float(env.get("S34_V_ENGINE_MAKER_FEE_BPS", 2.0)),
        "assumed_taker_fee_bps": float(env.get("S34_V_ENGINE_TAKER_FEE_BPS", 3.05)),
        "fee_samples": fees[-10:],
        "read": "No S34 real order fee samples in telemetry; provision pocket cannot be promoted until actual fee tier is verified.",
    }


def kill_switch_drill(env: dict[str, str], live_state: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / str(env.get("S34_LIVE_KILL_SWITCH_FILE", "runtime/KILL_SWITCH"))
    active = isinstance(live_state.get("active"), dict)
    return {
        "status": "DRY_RUN_CONTRACT_ONLY",
        "path": str(path),
        "exists_now": path.exists(),
        "would_block_new_entries_if_created": True,
        "would_auto_close_active_position": False,
        "active_position_in_state": active,
        "operator_commands": {"create": f"New-Item {path} -ItemType File", "clear": f"Remove-Item {path}"},
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    shadow = load_jsonl(args.shadow_ledger)
    mgmt = load_jsonl(args.management_ledger)
    readout = load_json(args.management_readout, {})
    v10 = load_json(args.v10_json, {})
    env = parse_env(args.env)
    live_state = load_json(args.live_state, {})
    journal = load_jsonl(args.decision_journal)
    telemetry = tail_jsonl(args.telemetry, int(args.telemetry_tail))
    truth = execution_truth_ledger(telemetry)
    return {
        "generated_at_utc": utc_now(),
        "mode": "OBSERVATION_GOVERNANCE_ONLY_NO_LIVE_CHANGE",
        "forward_sample_integrity": forward_sample_integrity(shadow, mgmt, readout, args.db),
        "operator_governance": governance_status(v10, journal),
        "execution_truth_ledger": truth,
        "fee_tier_verification": fee_tier_verification(env, truth),
        "kill_switch_drill": kill_switch_drill(env, live_state),
        "final_read": "Forward validation is decision-ready only after valid_forward_N reaches the frozen gate and operator risk decisions are journaled.",
    }


def render_md(report: dict[str, Any]) -> str:
    integ = report["forward_sample_integrity"]
    gov = report["operator_governance"]
    truth = report["execution_truth_ledger"]
    fee = report["fee_tier_verification"]
    drill = report["kill_switch_drill"]
    lines = [
        "# S34 v11 Forward Governance",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Mode: `{report['mode']}`",
        "",
        "## Forward Sample Integrity",
        "",
        f"- status: `{integ['status']}`",
        f"- source shadow N: `{integ['source_shadow_n']}`",
        f"- management N: `{integ['management_n']}`",
        f"- forward N / valid forward N: `{integ['forward_n']}` / `{integ['valid_forward_n']}`",
        f"- missing in management: `{integ['missing_in_management_n']}`",
        f"- extra management: `{integ['extra_management_n']}`",
        f"- quality counts all: `{integ['quality_counts_all']}`",
        f"- data freshness sec: `{integ['data_freshness_sec']}`",
        "",
        "## Operator Governance",
        "",
        f"- status: `{gov['status']}`",
        f"- oversize vs BALANCED: `{gov['oversize_vs_balanced']}x`",
        f"- real decision rows: `{gov['real_decision_rows']}`",
        f"- latest decision age h: `{gov['latest_decision_age_hours']}`",
        f"- required action: {gov['required_action']}",
        "",
        "## Execution Truth Ledger",
        "",
        f"- status: `{truth['status']}`",
        f"- scanned telemetry tail rows: `{truth['scanned_tail_rows']}`",
        f"- S34 order event N: `{truth['s34_order_event_n']}`",
        f"- missing fields to instrument: `{truth['missing_fields_to_instrument']}`",
        "",
        "## Fee Tier Verification",
        "",
        f"- status: `{fee['status']}`",
        f"- assumed maker/taker bps: `{fee['assumed_maker_fee_bps']}` / `{fee['assumed_taker_fee_bps']}`",
        f"- read: {fee['read']}",
        "",
        "## Kill Switch Drill",
        "",
        f"- status: `{drill['status']}`",
        f"- path: `{drill['path']}` exists=`{drill['exists_now']}`",
        f"- would block new entries: `{drill['would_block_new_entries_if_created']}`",
        f"- would auto-close active position: `{drill['would_auto_close_active_position']}`",
        "",
        "## Final Read",
        "",
        report["final_read"],
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S34 v11 forward integrity and governance.")
    p.add_argument("--db", type=Path, default=DB)
    p.add_argument("--env", type=Path, default=ENV)
    p.add_argument("--shadow-ledger", type=Path, default=SHADOW_LEDGER)
    p.add_argument("--management-ledger", type=Path, default=MGMT_LEDGER)
    p.add_argument("--management-readout", type=Path, default=MGMT_READOUT)
    p.add_argument("--v10-json", type=Path, default=V10)
    p.add_argument("--live-state", type=Path, default=LIVE_STATE)
    p.add_argument("--decision-journal", type=Path, default=DECISION_JOURNAL)
    p.add_argument("--telemetry", type=Path, default=TELEMETRY)
    p.add_argument("--telemetry-tail", type=int, default=5000)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")
    args.out_md.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
