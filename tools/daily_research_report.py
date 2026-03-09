from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.check_event_lanes import check_gate


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if out != out:
            return default
        return out
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _format_bps_from_net(net: float | None) -> str:
    if net is None:
        return "n/a"
    return f"{net * 10000.0:+.3f} bps"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.2f}%"


@dataclass
class PromotionPocket:
    name: str
    baseline_path: Path | None
    alt_path: Path


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a daily research monitor summary.")
    p.add_argument("--db", default="data/microstructure.db", help="Microstructure DB path for live event lane check.")
    p.add_argument("--symbol", default="ETHUSDT", help="Symbol for event lane gate snapshot.")
    p.add_argument("--date", default="", help="Report date in YYYY-MM-DD. Defaults to local date.")
    p.add_argument("--out", default="", help="Markdown output path. Defaults to reports/DAILY_<date>.md.")
    p.add_argument("--telemetry-path", default="logs/telemetry.jsonl", help="Telemetry JSONL path for regime recovery prep.")
    p.add_argument("--recovery-lookback-min", type=int, default=180, help="Lookback window for recovery signals.")
    p.add_argument("--event-lookback-min", type=int, default=60, help="Lookback window for event lane gate.")
    p.add_argument("--event-bucket-sec", type=int, default=5, help="Bucket seconds for event lane gate.")
    p.add_argument("--event-stale-after-sec", type=int, default=60, help="Stale threshold for event lane gate.")
    return p.parse_args()


def _default_report_date(raw: str) -> str:
    if raw:
        return str(raw)
    return datetime.now().strftime("%Y-%m-%d")


def _default_output_path(raw: str, report_date: str) -> Path:
    if raw:
        return Path(raw)
    return Path("reports") / f"DAILY_{report_date}.md"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_recent_jsonl(path: Path, since_min: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    cutoff = datetime.now().timestamp() - max(1, int(since_min)) * 60
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            ts = _safe_float(item.get("ts") or item.get("time"), 0.0)
            if ts < cutoff:
                continue
            rows.append(item)
    return rows


def build_event_lane_snapshot(
    *,
    db: str,
    symbol: str,
    lookback_min: int,
    bucket_sec: int,
    stale_after_sec: int,
) -> dict[str, Any]:
    try:
        result = check_gate(
            db=db,
            symbol=symbol,
            lookback_min=lookback_min,
            bucket_sec=bucket_sec,
            stale_after_sec=stale_after_sec,
        )
    except Exception as exc:
        return {
            "status": "UNKNOWN",
            "summary": f"event lane check failed: {exc}",
            "raw": {"symbol": symbol, "gate": "UNKNOWN", "reason": "exception"},
        }

    gate = str(result.get("gate") or "UNKNOWN").upper()
    if gate == "ALLOWED":
        status = "CLEAR"
    elif gate == "BLOCKED":
        status = "BLOCKED"
    else:
        status = "UNKNOWN"
    blocked = result.get("blocked_lanes") or []
    summary = f"gate={gate} reason={result.get('reason', 'n/a')}"
    if blocked:
        summary += f" blocked_lanes={','.join(str(x) for x in blocked)}"
    return {"status": status, "summary": summary, "raw": result}


def build_recovery_snapshot(*, telemetry_path: Path, lookback_min: int) -> dict[str, Any]:
    rows = _load_recent_jsonl(telemetry_path, lookback_min)
    belief_rows = [r for r in rows if str(r.get("event") or "") == "execution.belief_state"]
    blocked_rows = [r for r in rows if str(r.get("event") or "") == "entry.blocked"]
    guard_rows = [r for r in rows if str(r.get("event") or "") == "exit.telemetry_guard"]
    latest = belief_rows[-1] if belief_rows else None
    if latest is None:
        return {
            "status": "UNKNOWN",
            "summary": "no recent execution.belief_state telemetry",
            "raw": {
                "telemetry_path": str(telemetry_path),
                "belief_events": 0,
                "entry_blocked_events": len(blocked_rows),
                "guard_events": len(guard_rows),
            },
        }

    data = latest.get("data") or {}
    runtime_gate_degraded = bool(data.get("runtime_gate_degraded"))
    allow_entries = bool(data.get("allow_entries", True))
    guard_mode = str(data.get("guard_mode") or "UNKNOWN").upper()
    stage = str(data.get("guard_recovery_stage") or "UNKNOWN")
    reason = str(data.get("runtime_gate_reason") or data.get("guard_unlock_conditions") or "n/a")

    if runtime_gate_degraded or not allow_entries:
        status = "HOLD"
    elif guard_mode in {"GREEN", "READY", "CLEAR"}:
        status = "READY"
    else:
        status = "WATCH"

    summary = (
        f"guard_mode={guard_mode} allow_entries={allow_entries} "
        f"stage={stage} reason={reason}"
    )
    return {
        "status": status,
        "summary": summary,
        "raw": {
            "telemetry_path": str(telemetry_path),
            "belief_events": len(belief_rows),
            "entry_blocked_events": len(blocked_rows),
            "guard_events": len(guard_rows),
            "guard_mode": guard_mode,
            "allow_entries": allow_entries,
            "guard_recovery_stage": stage,
            "runtime_gate_reason": reason,
            "latest_ts": latest.get("ts"),
        },
    }


def _promotion_pockets() -> list[PromotionPocket]:
    return [
        PromotionPocket("Pocket B", Path("reports/ETH_POCKET_B_7D_BASELINE_SPLIT2.json"), Path("reports/ETH_POCKET_B_7D_PASSIVE_THEN_TAKER.json")),
        PromotionPocket("Pocket C", None, Path("reports/ETH_POCKET_C_7D_PASSIVE_THEN_TAKER.json")),
        PromotionPocket("Soft", Path("reports/ETH_POCKET_SOFT_7D_BASELINE.json"), Path("reports/ETH_POCKET_SOFT_7D_PASSIVE_THEN_TAKER.json")),
        PromotionPocket("Mid", Path("reports/ETH_POCKET_MID_7D_BASELINE.json"), Path("reports/ETH_POCKET_MID_7D_PASSIVE_THEN_TAKER.json")),
        PromotionPocket("Tight-mid", Path("reports/ETH_POCKET_TIGHTMID_7D_BASELINE.json"), Path("reports/ETH_POCKET_TIGHTMID_7D_PASSIVE_THEN_TAKER.json")),
    ]


def build_pocket_promotion_snapshot() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    promotable: list[str] = []
    observe_only: list[str] = []
    no_go: list[str] = []
    for pocket in _promotion_pockets():
        alt = _read_json(pocket.alt_path)
        base = _read_json(pocket.baseline_path) if pocket.baseline_path else None
        if alt is None:
            missing.append(str(pocket.alt_path))
            continue
        alt_pass = _safe_int(alt.get("pass_count"))
        alt_net = None
        per_split = alt.get("per_split") or []
        if per_split:
            alt_net = _safe_float(per_split[0].get("filled_avg_net_mean"), 0.0)
            alt_fill = _safe_float(per_split[0].get("attempt_fill_rate_mean"), 0.0)
        else:
            alt_fill = None
        base_pass = _safe_int(base.get("pass_count")) if base else None
        base_net = None
        if base and (base.get("per_split") or []):
            base_net = _safe_float(base["per_split"][0].get("filled_avg_net_mean"), 0.0)
        classification = "observe_only"
        if alt_pass == 3 and (alt_net or 0.0) > 0.0:
            classification = "promotable"
            promotable.append(pocket.name)
        elif alt_pass == 0 or (alt_net is not None and alt_net < 0.0):
            classification = "no_go"
            no_go.append(pocket.name)
        else:
            observe_only.append(pocket.name)
        rows.append(
            {
                "pocket": pocket.name,
                "baseline_pass": f"{base_pass}/3" if base_pass is not None else "n/a",
                "alt_pass": f"{alt_pass}/3",
                "baseline_net_bps": _format_bps_from_net(base_net),
                "alt_net_bps": _format_bps_from_net(alt_net),
                "alt_fill_rate": _format_pct(alt_fill),
                "classification": classification,
            }
        )

    required_promotable = {"Pocket B", "Pocket C", "Tight-mid"}
    promotable_set = set(promotable)
    if missing:
        status = "INCOMPLETE"
        summary = f"missing promotion artifacts={len(missing)}"
    elif required_promotable.issubset(promotable_set) and "Mid" in no_go:
        status = "GO_EXPERIMENTAL"
        summary = "tighter ETH 60s pocket family is promotable; keep softer pockets out of rollout"
    elif promotable:
        status = "WATCH"
        summary = "some pockets are promotable, but family boundary is not yet fully clean"
    else:
        status = "NO_GO"
        summary = "no promotable tighter family pocket found"

    return {
        "status": status,
        "summary": summary,
        "rows": rows,
        "missing": missing,
        "promotable": promotable,
        "observe_only": observe_only,
        "no_go": no_go,
    }


def build_daily_research_report(
    *,
    report_date: str,
    event_lane: dict[str, Any],
    recovery: dict[str, Any],
    promotion: dict[str, Any],
) -> str:
    event_raw = event_lane.get("raw") or {}
    recovery_raw = recovery.get("raw") or {}
    lines = [
        f"# Daily Research Report - {report_date}",
        "",
        "## Headline",
        f"- event lanes: `{event_lane['status']}`",
        f"- regime recovery prep: `{recovery['status']}`",
        f"- pocket promotion checklist: `{promotion['status']}`",
        "",
        "## Event Lane Gate",
        f"- symbol: `{event_raw.get('symbol', 'n/a')}`",
        f"- status: `{event_lane['status']}`",
        f"- summary: {event_lane['summary']}",
        f"- pocket: `{event_raw.get('pocket', 'n/a')}`",
        f"- profile: `{event_raw.get('profile', 'n/a')}`",
        "",
        "## Regime Recovery Prep",
        f"- status: `{recovery['status']}`",
        f"- summary: {recovery['summary']}",
        f"- belief events in window: `{recovery_raw.get('belief_events', 0)}`",
        f"- recent blocked entries: `{recovery_raw.get('entry_blocked_events', 0)}`",
        f"- recent guard events: `{recovery_raw.get('guard_events', 0)}`",
        "",
        "## Pocket Promotion Checklist",
        f"- status: `{promotion['status']}`",
        f"- summary: {promotion['summary']}",
        f"- promotable: `{', '.join(promotion.get('promotable') or ['none'])}`",
        f"- observe_only: `{', '.join(promotion.get('observe_only') or ['none'])}`",
        f"- no_go: `{', '.join(promotion.get('no_go') or ['none'])}`",
        "",
        "| Pocket | Baseline pass | Passive-then-taker pass | Baseline net | Passive-then-taker net | Fill rate | Class |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in promotion.get("rows") or []:
        lines.append(
            f"| {row['pocket']} | {row['baseline_pass']} | {row['alt_pass']} | "
            f"{row['baseline_net_bps']} | {row['alt_net_bps']} | {row['alt_fill_rate']} | {row['classification']} |"
        )
    if promotion.get("missing"):
        lines.extend(
            [
                "",
                "## Missing Inputs",
                *[f"- `{path}`" for path in promotion["missing"]],
            ]
        )
    lines.extend(
        [
            "",
            "## Next Action",
            f"- gate focus: `{event_lane['status']}`",
            f"- recovery focus: `{recovery['status']}`",
            f"- promotion focus: `{promotion['status']}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_daily_research_payload(
    *,
    report_date: str,
    event_lane: dict[str, Any],
    recovery: dict[str, Any],
    promotion: dict[str, Any],
) -> dict[str, Any]:
    return {
        "report_date": report_date,
        "headline": {
            "event_lanes": event_lane.get("status"),
            "regime_recovery_prep": recovery.get("status"),
            "pocket_promotion_checklist": promotion.get("status"),
        },
        "event_lane": event_lane,
        "recovery": recovery,
        "promotion": promotion,
    }


def run_once(args: argparse.Namespace) -> int:
    report_date = _default_report_date(args.date)
    out_path = _default_output_path(args.out, report_date)
    event_lane = build_event_lane_snapshot(
        db=str(args.db),
        symbol=str(args.symbol),
        lookback_min=int(args.event_lookback_min),
        bucket_sec=int(args.event_bucket_sec),
        stale_after_sec=int(args.event_stale_after_sec),
    )
    recovery = build_recovery_snapshot(
        telemetry_path=Path(args.telemetry_path),
        lookback_min=int(args.recovery_lookback_min),
    )
    promotion = build_pocket_promotion_snapshot()
    md = build_daily_research_report(
        report_date=report_date,
        event_lane=event_lane,
        recovery=recovery,
        promotion=promotion,
    )
    payload = build_daily_research_payload(
        report_date=report_date,
        event_lane=event_lane,
        recovery=recovery,
        promotion=promotion,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"daily_research_report: wrote {out_path}")
    print(f"daily_research_report: wrote {json_path}")
    return 0


def main() -> int:
    return run_once(_args())


if __name__ == "__main__":
    raise SystemExit(main())
