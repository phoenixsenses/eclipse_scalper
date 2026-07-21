"""S34 V Engine live-vs-mirror divergence report.

Read-only diagnostic. It explains cases where the V0.2 shadow mirror has a
signal but the live executor did not open/track an order.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MIRROR_LEDGER = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_LEDGER.csv"
LIVE_STATE = ROOT / "runtime" / "s34_v_engine_live_state.json"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_LIVE_MIRROR_DIVERGENCE.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_LIVE_MIRROR_DIVERGENCE.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def load_mirror_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def row_key(row: dict[str, Any]) -> str:
    return f"{row.get('signal_ts_ms')}:{row.get('bucket')}:{row.get('vdepth_bps')}"


def explain_latest(mirror_rows: list[dict[str, str]], live_state: dict[str, Any]) -> dict[str, Any]:
    latest = mirror_rows[-1] if mirror_rows else {}
    active = live_state.get("active") if isinstance(live_state.get("active"), dict) else None
    status = live_state.get("status") if isinstance(live_state.get("status"), dict) else {}
    active_event_id = str(active.get("event_id") or "") if active else ""
    signal_ts = str(latest.get("signal_ts_ms") or "")
    live_matches_latest = bool(active_event_id and signal_ts and signal_ts in active_event_id)
    reason = "UNKNOWN"
    if not latest:
        reason = "NO_MIRROR_ROWS"
    elif live_matches_latest:
        reason = "LIVE_ACTIVE_MATCHES_LATEST_MIRROR"
    elif active:
        reason = "LIVE_ACTIVE_DIFFERENT_EVENT"
    elif status.get("last_missed_signal"):
        missed = status.get("last_missed_signal") or {}
        if str(missed.get("anchor_utc")) == str(latest.get("signal_utc")) or str(missed.get("event_id", "")).endswith(signal_ts):
            reason = "LIVE_REPORTED_MISSED_FRESHNESS"
        else:
            reason = "LIVE_NO_ACTIVE_WITH_PRIOR_MISSED_SIGNAL"
    elif status.get("new_entry_blocked_by"):
        reason = f"LIVE_BLOCKED_BY_{status.get('new_entry_blocked_by')}"
    elif latest.get("decision") == "OBSERVE_ONLY_NO_ORDER":
        reason = "MIRROR_IS_OBSERVE_ONLY_LIVE_NOT_LINKED_TO_THIS_ROW"
    return {
        "reason": reason,
        "latest_mirror": {
            "observation_id": latest.get("observation_id"),
            "signal_utc": latest.get("signal_utc"),
            "observation_status": latest.get("observation_status"),
            "sim_status": latest.get("sim_status"),
            "maker_fill_utc": latest.get("maker_fill_utc"),
            "entry_price": latest.get("entry_price"),
            "exit_utc": latest.get("exit_utc"),
            "net_bps": latest.get("net_bps"),
            "decision": latest.get("decision"),
            "notes": latest.get("notes"),
        },
        "live": {
            "active": active,
            "active_status": status.get("active_status"),
            "new_entry_blocked_by": status.get("new_entry_blocked_by"),
            "last_missed_signal": status.get("last_missed_signal"),
            "last_signal_scan": status.get("last_signal_scan"),
            "orders_count": len(live_state.get("orders") or []),
            "reconciliation": live_state.get("reconciliation"),
        },
    }


def render_md(report: dict[str, Any]) -> str:
    latest = report["latest"]
    lm = latest["latest_mirror"]
    lv = latest["live"]
    lines = [
        "# S34 V Engine Live/Mirror Divergence",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Read-only diagnostic. No live state, order logic, config, or executor process is changed.",
        "",
        "## Latest Explanation",
        "",
        f"- reason: `{latest['reason']}`",
        f"- mirror signal: `{lm.get('signal_utc')}`",
        f"- mirror status: `{lm.get('observation_status')}` / `{lm.get('sim_status')}`",
        f"- mirror decision: `{lm.get('decision')}`",
        f"- live active status: `{lv.get('active_status')}`",
        f"- live blocked by: `{lv.get('new_entry_blocked_by')}`",
        "",
        "## Latest Mirror",
        "",
        "```json",
        json.dumps(lm, indent=2, sort_keys=True),
        "```",
        "",
        "## Live Diagnostic",
        "",
        "```json",
        json.dumps(lv, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def run(mirror_ledger: Path, live_state_path: Path) -> dict[str, Any]:
    mirror_rows = load_mirror_rows(mirror_ledger)
    live_state = load_json(live_state_path, {})
    return {
        "generated_at_utc": utc_now(),
        "research_only": True,
        "live_touched": False,
        "mirror_ledger": str(mirror_ledger),
        "live_state": str(live_state_path),
        "mirror_rows": len(mirror_rows),
        "latest": explain_latest(mirror_rows, live_state),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build S34 V Engine live/mirror divergence report.")
    p.add_argument("--mirror-ledger", type=Path, default=MIRROR_LEDGER)
    p.add_argument("--live-state", type=Path, default=LIVE_STATE)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args.mirror_ledger, args.live_state)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
