from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.run_summary import build_run_summary


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _headline_for_row(row: Dict[str, Any]) -> str:
    return str(row.get("headline") or row.get("lane") or "unknown")


def _select_focus_lanes(lanes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fresh_non_quiet = [
        row
        for row in lanes
        if str(row.get("freshness_status") or "") == "fresh" and str(row.get("level") or "quiet") != "quiet"
    ]
    if fresh_non_quiet:
        return fresh_non_quiet[:3]
    severe_any = [row for row in lanes if str(row.get("level") or "quiet") == "severe"]
    return severe_any[:3]


def build_merged_banner_policy_payload(
    *,
    effective_json: str,
    out_json: str,
    out_md: str,
) -> Dict[str, Any]:
    effective = _load_json(effective_json)
    lanes = [dict(row) for row in (effective.get("lanes") or []) if isinstance(row, dict)]
    lanes.sort(key=lambda row: float(row.get("effective_priority_score") or 0.0), reverse=True)
    focus_lanes = _select_focus_lanes(lanes)
    top = dict(effective.get("effective_top_event") or {})
    summary = dict(effective.get("summary") or {})

    merge_active = len(focus_lanes) >= 2
    banner_mode = "merged" if merge_active else "single"
    parts = [str(row.get("lane") or "unknown") for row in focus_lanes[:3]]
    merged_title = " + ".join(parts) if parts else str(top.get("lane") or "unknown")
    headline = (
        f"Research events merged={merged_title} action={str(top.get('recommended_action') or 'monitor_only')}"
        if merge_active
        else f"Research event top={str(top.get('lane') or 'unknown')} action={str(top.get('recommended_action') or 'monitor_only')}"
    )
    reasons: List[str] = []
    if int(summary.get("degraded_lane_count") or 0) > 0:
        reasons.append("overlap_suppression_active")
    if int(summary.get("noisy_lane_count") or 0) > 0:
        reasons.append("persistence_guard_active")
    if merge_active:
        reasons.append("multiple_fresh_high_priority_lanes")

    payload = {
        "effective_json": str(effective_json),
        "summary": {
            "banner_mode": banner_mode,
            "focus_lane_count": len(focus_lanes),
            "focus_lanes": [str(row.get("lane") or "") for row in focus_lanes],
            "top_lane": str(top.get("lane") or ""),
            "top_action": str(top.get("recommended_action") or "monitor_only"),
        },
        "banner": {
            "headline": headline,
            "recommended_action": str(top.get("recommended_action") or "monitor_only"),
            "top_lane": str(top.get("lane") or ""),
            "banner_mode": banner_mode,
            "focus_lanes": [str(row.get("lane") or "") for row in focus_lanes],
            "reasons": reasons,
            "operator_note": (
                "Show one combined banner for the current high-priority lanes."
                if merge_active
                else "Show the top lane banner without merge."
            ),
        },
        "focus_rows": [
            {
                "lane": str(row.get("lane") or ""),
                "level": str(row.get("level") or "quiet"),
                "freshness_status": str(row.get("freshness_status") or ""),
                "recommended_action": str(row.get("recommended_action") or "monitor_only"),
                "effective_display_mode": str(row.get("effective_display_mode") or "keep"),
                "effective_priority_score": float(row.get("effective_priority_score") or 0.0),
                "headline": _headline_for_row(row),
            }
            for row in focus_lanes
        ],
    }
    payload["run_summary"] = build_run_summary(
        run_type="event_merged_banner_policy",
        inputs={"effective_json": str(effective_json)},
        metrics={
            "banner_mode": banner_mode,
            "focus_lane_count": len(focus_lanes),
            "top_lane": str(top.get("lane") or ""),
            "top_action": str(top.get("recommended_action") or "monitor_only"),
        },
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a merged operator banner policy from the effective event watchboard.")
    p.add_argument("--effective-json", default="reports/EVENT_WATCHBOARD_EFFECTIVE.json")
    p.add_argument("--out-json", default="reports/EVENT_MERGED_BANNER_POLICY.json")
    p.add_argument("--out-md", default="reports/EVENT_MERGED_BANNER_POLICY.md")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    payload = build_merged_banner_policy_payload(
        effective_json=str(args.effective_json),
        out_json=str(args.out_json),
        out_md=str(args.out_md),
    )
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# EVENT MERGED BANNER POLICY",
        "",
        f"banner_mode={payload['summary']['banner_mode']}",
        f"focus_lanes={','.join(payload['summary']['focus_lanes'])}",
        f"headline={payload['banner']['headline']}",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
