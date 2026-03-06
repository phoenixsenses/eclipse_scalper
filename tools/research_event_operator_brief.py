from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.run_summary import build_run_summary


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_operator_brief_payload(
    *,
    watchboard_json: str,
    trend_json: str,
    overlap_json: str,
    consolidation_json: str,
    out_json: str,
    out_md: str,
) -> Dict[str, Any]:
    watchboard = _load_json(watchboard_json)
    trend = _load_json(trend_json)
    overlap = _load_json(overlap_json)
    consolidation = _load_json(consolidation_json)

    lanes = watchboard.get("lanes") or []
    severe_lanes = [lane for lane in lanes if str(lane.get("level") or lane.get("state_level") or "") == "severe"]
    stale_lanes = [lane for lane in lanes if str(lane.get("freshness_status") or "") == "stale"]
    top_event = watchboard.get("top_event") or {}
    trend_summary = trend.get("summary") or {}
    lane_deltas = list(trend.get("lane_deltas") or [])
    strongest_delta = lane_deltas[0] if lane_deltas else {}
    strongest_overlap_rows = list(overlap.get("strongest_overlaps") or [])
    strongest_overlap = strongest_overlap_rows[0] if strongest_overlap_rows else {}
    consolidation_decisions = list(consolidation.get("decisions") or [])
    suppression_candidates = [
        row for row in consolidation_decisions if str(row.get("recommendation") or "") == "candidate_suppress_secondary"
    ]
    primary_suppression = suppression_candidates[0] if suppression_candidates else {}

    brief_lines: List[str] = []
    top_lane = str((watchboard.get("summary") or {}).get("top_lane") or "")
    top_action = str(top_event.get("recommended_action") or "monitor_only")
    trend_name = str(trend_summary.get("trend") or "flat")
    brief_lines.append(f"top lane {top_lane or 'unknown'}, action {top_action}, trend {trend_name}.")
    if strongest_delta:
        brief_lines.append(
            "strongest lane delta: "
            + f"{str(strongest_delta.get('lane') or 'unknown')} {str(strongest_delta.get('trend') or 'flat')} "
            + f"({float(strongest_delta.get('delta_priority_score') or 0.0):+.2f})."
        )
    if strongest_overlap:
        brief_lines.append(
            "strongest overlap: "
            + f"{str(strongest_overlap.get('lane_a') or 'unknown')} + {str(strongest_overlap.get('lane_b') or 'unknown')} "
            + f"(jaccard={float(strongest_overlap.get('jaccard') or 0.0):.2f}, "
            + f"coactive_count={int(strongest_overlap.get('coactive_count') or 0)})."
        )
    if primary_suppression:
        brief_lines.append(
            "suppression candidate: "
            + f"{str(primary_suppression.get('secondary_lane') or 'unknown')} "
            + f"behind {str(primary_suppression.get('lane_a') or 'unknown')} / {str(primary_suppression.get('lane_b') or 'unknown')}."
        )
    if severe_lanes:
        brief_lines.append("severe lanes: " + ", ".join(str(lane.get("lane") or "unknown") for lane in severe_lanes) + ".")
    if stale_lanes:
        brief_lines.append("stale lanes: " + ", ".join(str(lane.get("lane") or "unknown") for lane in stale_lanes) + ".")
    if not severe_lanes and not stale_lanes:
        brief_lines.append("no severe or stale lanes.")

    payload = {
        "watchboard_json": str(watchboard_json),
        "trend_json": str(trend_json),
        "overlap_json": str(overlap_json),
        "consolidation_json": str(consolidation_json),
        "summary": {
            "top_lane": top_lane,
            "top_action": top_action,
            "trend": trend_name,
            "severe_lane_count": len(severe_lanes),
            "stale_lane_count": len(stale_lanes),
            "strongest_delta_lane": str(strongest_delta.get("lane") or ""),
            "strongest_delta_trend": str(strongest_delta.get("trend") or "flat"),
            "strongest_overlap_pair": (
                f"{str(strongest_overlap.get('lane_a') or '')}::{str(strongest_overlap.get('lane_b') or '')}"
                if strongest_overlap
                else ""
            ),
            "suppression_candidate_count": len(suppression_candidates),
            "primary_suppression_lane": str(primary_suppression.get("secondary_lane") or ""),
        },
        "brief": {
            "headline": f"Research events top={top_lane or 'unknown'} action={top_action} trend={trend_name}",
            "operator_note": " ".join(brief_lines),
            "top_event": {
                "lane": top_lane,
                "action": top_action,
                "headline": str(top_event.get("headline") or ""),
            },
            "strongest_delta": {
                "lane": str(strongest_delta.get("lane") or ""),
                "trend": str(strongest_delta.get("trend") or "flat"),
                "delta_priority_score": float(strongest_delta.get("delta_priority_score") or 0.0),
            },
            "strongest_overlap": {
                "lane_a": str(strongest_overlap.get("lane_a") or ""),
                "lane_b": str(strongest_overlap.get("lane_b") or ""),
                "jaccard": float(strongest_overlap.get("jaccard") or 0.0),
                "coactive_count": int(strongest_overlap.get("coactive_count") or 0),
            },
            "primary_suppression": {
                "secondary_lane": str(primary_suppression.get("secondary_lane") or ""),
                "lane_a": str(primary_suppression.get("lane_a") or ""),
                "lane_b": str(primary_suppression.get("lane_b") or ""),
                "recommendation": str(primary_suppression.get("recommendation") or ""),
            },
            "severe_lanes": [str(lane.get("lane") or "") for lane in severe_lanes],
            "stale_lanes": [str(lane.get("lane") or "") for lane in stale_lanes],
        },
    }
    payload["run_summary"] = build_run_summary(
        run_type="research_event_operator_brief",
        inputs={
            "watchboard_json": watchboard_json,
            "trend_json": trend_json,
            "overlap_json": overlap_json,
            "consolidation_json": consolidation_json,
        },
        metrics=payload["summary"],
        artifacts={
            "json": out_json,
            "md": out_md,
            "watchboard_json": watchboard_json,
            "trend_json": trend_json,
            "overlap_json": overlap_json,
            "consolidation_json": consolidation_json,
        },
    )
    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an operator-facing brief from watchboard and trend outputs.")
    p.add_argument("--watchboard-json", default="reports/RESEARCH_EVENT_WATCHBOARD.json")
    p.add_argument("--trend-json", default="reports/RESEARCH_EVENT_WATCHBOARD_TREND_FROM_HISTORY.json")
    p.add_argument("--overlap-json", default="reports/EVENT_LANE_OVERLAP.json")
    p.add_argument("--consolidation-json", default="reports/EVENT_LANE_CONSOLIDATION.json")
    p.add_argument("--out-json", default="reports/RESEARCH_EVENT_OPERATOR_BRIEF.json")
    p.add_argument("--out-md", default="reports/RESEARCH_EVENT_OPERATOR_BRIEF.md")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    payload = build_operator_brief_payload(
        watchboard_json=str(args.watchboard_json),
        trend_json=str(args.trend_json),
        overlap_json=str(args.overlap_json),
        consolidation_json=str(args.consolidation_json),
        out_json=str(args.out_json),
        out_md=str(args.out_md),
    )
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# RESEARCH EVENT OPERATOR BRIEF",
        "",
        payload["brief"]["headline"],
        "",
        payload["brief"]["operator_note"],
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
