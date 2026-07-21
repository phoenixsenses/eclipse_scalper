"""S34 V Engine management alert builder.

Default behavior is file/telemetry output only. Telegram delivery requires
explicit --notify. This is observation/risk notification, not action.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
READOUT = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_FORWARD_MANAGEMENT_READOUT.json"
AUDIT = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_EXECUTION_MANAGEMENT_AUDIT.json"
V9 = ROOT / "reports" / "research" / "s34" / "S34_V9_KILL_FORWARD_REVIEW.json"
V11 = ROOT / "reports" / "research" / "s34" / "S34_V11_FORWARD_GOVERNANCE.json"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_MANAGEMENT_ALERTS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_MANAGEMENT_ALERTS.md"
TELEMETRY = ROOT / "logs" / "telemetry.jsonl"
STATE = ROOT / "runtime" / "s34_v_engine_management_alert_state.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def severity(alerts: list[dict[str, Any]]) -> str:
    if any(a.get("severity") == "critical" for a in alerts):
        return "critical"
    if any(a.get("severity") == "warning" for a in alerts):
        return "warning"
    return "info"


def build_alerts(
    readout: dict[str, Any],
    audit: dict[str, Any],
    v9: dict[str, Any] | None = None,
    v11: dict[str, Any] | None = None,
) -> dict[str, Any]:
    alerts: list[dict[str, Any]] = []
    sizing = readout.get("tail_aware_sizing_monitor") or {}
    atomicity = readout.get("atomicity_gap_monitor") or {}
    regime = readout.get("regime_degradation_monitor") or {}
    kills = readout.get("explicit_kill_criteria") or {}
    gap = audit.get("gap_through") or {}
    tail = audit.get("tail_frequency") or {}
    stop_math = audit.get("stop_budget_math") or {}
    v9 = v9 or {}
    v11 = v11 or {}
    tick = v9.get("tick_level_atomicity_scan") or {}
    kill_design = (v9.get("kill_criteria_redesign") or {}).get("recommended_kill_rule") or {}
    gov = v11.get("operator_governance") or {}

    if sizing.get("status") == "ALERT_OVERSIZE_OPERATOR_ACTION_REQUIRED":
        alerts.append(
            {
                "code": "S34_OVERSIZE",
                "severity": "critical",
                "message": (
                    f"S34 live env planned margin ${((sizing.get('planned_live_size_from_env') or {}).get('planned_margin_usdt'))} "
                    f"vs tail-budget ${sizing.get('max_tail_budget_margin_usdt')} "
                    f"(oversize {sizing.get('oversize_multiple')}x)."
                ),
                "recommendation": "operator reduce margin to budget or disarm; no automatic action taken",
            }
        )
    if gap and float(gap.get("gap_plus_fee_bps") or 0.0) > 0:
        alerts.append(
            {
                "code": "S34_STOP_GAP_THROUGH",
                "severity": "warning",
                "message": (
                    f"Configured stop {gap.get('current_stop_nominal_bps')} bps realized worst "
                    f"{gap.get('current_stop_research_max_loss_bps')} bps."
                ),
                "recommendation": "treat stop as partial protection; size remains primary control",
            }
        )
    if atomicity.get("status") == "ALERT_ADVERSE_IN_GAP":
        alerts.append(
            {
                "code": "S34_ATOMICITY_GAP",
                "severity": "warning",
                "message": f"Adverse move observed inside fill-to-stop gap; worst {atomicity.get('worst_adverse_bps')} bps.",
                "recommendation": "document atomic bracket requirement; operator sign-off required for live logic change",
            }
        )
    if tick.get("status") == "NO_TICK_CATASTROPHIC_GAP_FOUND":
        alerts.append(
            {
                "code": "S34_ATOMICITY_SCAN_OK_SMALL_N",
                "severity": "info",
                "message": f"Tick scan found no catastrophic 2s gap; worst {tick.get('worst_tick_adverse_bps')} bps.",
                "recommendation": "continue observation; absence in 23 fills is not proof of zero risk",
            }
        )
    if regime.get("status") == "TRIGGERED":
        alerts.append(
            {
                "code": "S34_REGIME_DEGRADATION",
                "severity": "warning",
                "message": f"Regime degradation triggers: {regime.get('triggers')}",
                "recommendation": "operator pause/scale-down review",
            }
        )
    triggered = kills.get("triggered") or []
    if triggered:
        alerts.append(
            {
                "code": "S34_KILL_CRITERIA",
                "severity": "critical" if "OPERATOR_SIZE_REVIEW_REQUIRED" in triggered else "warning",
                "message": f"Kill/pause recommendations triggered: {triggered}",
                "recommendation": "recommendation only; executor not changed",
            }
        )
    if gov.get("status") == "DECISION_REQUIRED":
        alerts.append(
            {
                "code": "S34_OPERATOR_DECISION_REQUIRED",
                "severity": "critical",
                "message": f"Oversize vs BALANCED {gov.get('oversize_vs_balanced')}x with no real decision journal row.",
                "recommendation": gov.get("required_action") or "operator must log risk decision",
            }
        )

    payload = {
        "generated_at_utc": utc_now(),
        "mode": "NOTIFY_ONLY_NO_ACTION",
        "severity": severity(alerts),
        "alerts": alerts,
        "dashboard_line": readout.get("dashboard_line"),
        "tail_context": {
            "large_loss_rate": tail.get("large_loss_rate"),
            "probabilities": tail.get("probabilities"),
            "planned_stop_loss_pct_equity_research": stop_math.get("planned_stop_loss_pct_equity_research"),
        },
        "kill_design": kill_design,
        "operator_governance": gov,
    }
    return payload


def alert_signature(payload: dict[str, Any]) -> str:
    parts = []
    for alert in payload.get("alerts") or []:
        parts.append(f"{alert.get('severity')}:{alert.get('code')}:{alert.get('message')}")
    return "|".join(sorted(parts)) or "NO_ALERTS"


def apply_dedup(payload: dict[str, Any], state_path: Path, *, emit_unchanged: bool) -> dict[str, Any]:
    state = load_json(state_path)
    sig = alert_signature(payload)
    previous = state.get("last_signature")
    changed = sig != previous
    payload["alert_signature"] = sig
    payload["state_changed"] = bool(changed)
    payload["delivery_status"] = "EMIT_STATE_CHANGED" if changed or emit_unchanged else "DEDUP_SUPPRESSED_UNCHANGED"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "last_signature": sig,
                "last_emit_utc": payload["generated_at_utc"] if changed or emit_unchanged else state.get("last_emit_utc"),
                "last_seen_utc": payload["generated_at_utc"],
                "last_delivery_status": payload["delivery_status"],
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    return payload


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine Management Alerts",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        f"Mode: `{payload['mode']}`",
        "",
        f"Severity: `{payload['severity']}`",
        "",
        f"Delivery: `{payload.get('delivery_status')}` state_changed=`{payload.get('state_changed')}`",
        "",
        "| Severity | Code | Message | Recommendation |",
        "| --- | --- | --- | --- |",
    ]
    for alert in payload.get("alerts") or []:
        lines.append(
            f"| {alert.get('severity')} | `{alert.get('code')}` | {alert.get('message')} | {alert.get('recommendation')} |"
        )
    if not payload.get("alerts"):
        lines.append("| info | `S34_NO_ALERTS` | No management alerts. | Continue observation. |")
    lines.extend(["", "## Dashboard Line", "", f"`{json.dumps(payload.get('dashboard_line'), sort_keys=True)}`", ""])
    return "\n".join(lines)


def append_telemetry(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": payload["generated_at_utc"],
        "event": "s34.v_engine.management_alerts",
        "severity": payload["severity"],
        "alert_count": len(payload.get("alerts") or []),
        "alerts": payload.get("alerts") or [],
        "delivery_status": payload.get("delivery_status"),
        "state_changed": payload.get("state_changed"),
        "mode": payload["mode"],
    }
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n")


async def notify_telegram(text: str) -> bool:
    token = os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("ECLIPSE_TG_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("ECLIPSE_TG_CHAT_ID")
    if not token or not chat_id:
        return False
    from notifications.telegram import Notifier

    notifier = Notifier(token, chat_id)
    return bool(await notifier.speak(text, priority="critical", silent=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build S34 management alerts. Telegram requires explicit --notify.")
    p.add_argument("--readout", type=Path, default=READOUT)
    p.add_argument("--audit", type=Path, default=AUDIT)
    p.add_argument("--v9", type=Path, default=V9)
    p.add_argument("--v11", type=Path, default=V11)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    p.add_argument("--telemetry", type=Path, default=TELEMETRY)
    p.add_argument("--state", type=Path, default=STATE)
    p.add_argument("--emit-unchanged", action="store_true")
    p.add_argument("--notify", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_alerts(load_json(args.readout), load_json(args.audit), load_json(args.v9), load_json(args.v11))
    payload = apply_dedup(payload, args.state, emit_unchanged=bool(args.emit_unchanged))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")
    md = render_md(payload)
    args.out_md.write_text(md, encoding="utf-8")
    append_telemetry(args.telemetry, payload)
    should_deliver = payload.get("delivery_status") == "EMIT_STATE_CHANGED"
    if args.notify and should_deliver:
        sent = asyncio.run(notify_telegram(md[:3500]))
        payload["telegram_sent"] = bool(sent)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")
    print(md)
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Appended {args.telemetry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
