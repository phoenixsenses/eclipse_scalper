from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.run_summary import build_run_summary


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _lane_stats_map(overlap: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("lane") or ""): row
        for row in (overlap.get("lane_stats") or [])
        if isinstance(row, dict) and str(row.get("lane") or "")
    }


def _pick_secondary(lane_a: str, lane_b: str, stats: Dict[str, Dict[str, Any]]) -> str:
    a = stats.get(lane_a, {})
    b = stats.get(lane_b, {})
    a_tuple = (int(a.get("top_count") or 0), int(a.get("active_count") or 0), lane_a)
    b_tuple = (int(b.get("top_count") or 0), int(b.get("active_count") or 0), lane_b)
    return lane_a if a_tuple < b_tuple else lane_b


def _decision_for_pair(row: Dict[str, Any], stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    lane_a = str(row.get("lane_a") or "")
    lane_b = str(row.get("lane_b") or "")
    jaccard = float(row.get("jaccard") or 0.0)
    coactive_count = int(row.get("coactive_count") or 0)
    secondary_lane = _pick_secondary(lane_a, lane_b, stats)
    if jaccard >= 0.85 and coactive_count >= 2:
        recommendation = "candidate_suppress_secondary"
        reason = (
            f"{lane_a} and {lane_b} move almost together; consider suppressing {secondary_lane} "
            "when the stronger companion lane is already active."
        )
    elif jaccard >= 0.60 and coactive_count >= 2:
        recommendation = "review_for_merge"
        reason = f"{lane_a} and {lane_b} overlap materially; review whether they encode the same market context."
    else:
        recommendation = "keep_separate"
        reason = f"{lane_a} and {lane_b} do not overlap enough to justify consolidation."
    return {
        "lane_a": lane_a,
        "lane_b": lane_b,
        "jaccard": jaccard,
        "coactive_count": coactive_count,
        "secondary_lane": secondary_lane,
        "recommendation": recommendation,
        "reason": reason,
    }


def build_consolidation_payload(
    *,
    watchboard_json: str,
    overlap_json: str,
    top_n: int,
    out_json: str,
    out_md: str,
) -> Dict[str, Any]:
    watchboard = _load_json(watchboard_json)
    overlap = _load_json(overlap_json)
    stats = _lane_stats_map(overlap)
    strongest = list(overlap.get("strongest_overlaps") or [])[: max(int(top_n), 0)]
    decisions = [_decision_for_pair(row, stats) for row in strongest if isinstance(row, dict)]
    recommendation_counts: Dict[str, int] = {}
    for row in decisions:
        key = str(row.get("recommendation") or "keep_separate")
        recommendation_counts[key] = recommendation_counts.get(key, 0) + 1
    top_event = dict(watchboard.get("top_event") or {})
    summary = {
        "top_lane": str((watchboard.get("summary") or {}).get("top_lane") or top_event.get("lane") or ""),
        "top_overlap_pair": str((overlap.get("summary") or {}).get("top_overlap_pair") or ""),
        "decision_count": len(decisions),
        "recommendation_counts": recommendation_counts,
    }
    payload = {
        "watchboard_json": str(watchboard_json),
        "overlap_json": str(overlap_json),
        "summary": summary,
        "decisions": decisions,
    }
    payload["run_summary"] = build_run_summary(
        run_type="event_lane_consolidation",
        inputs={"watchboard_json": str(watchboard_json), "overlap_json": str(overlap_json), "top_n": int(top_n)},
        metrics={
            "decision_count": len(decisions),
            "candidate_suppress_secondary_count": recommendation_counts.get("candidate_suppress_secondary", 0),
            "review_for_merge_count": recommendation_counts.get("review_for_merge", 0),
        },
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Turn lane overlap into keep/review/suppress consolidation decisions.")
    p.add_argument("--watchboard-json", default="reports/RESEARCH_EVENT_WATCHBOARD.json")
    p.add_argument("--overlap-json", default="reports/EVENT_LANE_OVERLAP.json")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--out-json", default="reports/EVENT_LANE_CONSOLIDATION.json")
    p.add_argument("--out-md", default="reports/EVENT_LANE_CONSOLIDATION.md")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    payload = build_consolidation_payload(
        watchboard_json=str(args.watchboard_json),
        overlap_json=str(args.overlap_json),
        top_n=int(args.top_n),
        out_json=str(args.out_json),
        out_md=str(args.out_md),
    )
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# EVENT LANE CONSOLIDATION",
        "",
        f"summary={json.dumps(payload['summary'], ensure_ascii=True, sort_keys=True)}",
        "",
        "| lane_a | lane_b | jaccard | coactive_count | secondary_lane | recommendation |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in payload["decisions"]:
        lines.append(
            f"| {row['lane_a']} | {row['lane_b']} | {float(row['jaccard']):.4f} | {int(row['coactive_count'])} | "
            f"{row['secondary_lane']} | {row['recommendation']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
