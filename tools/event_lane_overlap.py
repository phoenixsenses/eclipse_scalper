from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tools.run_summary import build_run_summary


_LEVEL_RANK = {"quiet": 0, "elevated": 1, "severe": 2}


def _load_history_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _level_rank(level: Any) -> int:
    return _LEVEL_RANK.get(str(level or "quiet"), 0)


def _active_lanes(row: Dict[str, Any], min_level: str) -> List[str]:
    active: List[str] = []
    min_rank = _level_rank(min_level)
    for lane in _row_lanes(row):
        if not isinstance(lane, dict):
            continue
        name = str(lane.get("lane") or "")
        if not name:
            continue
        if _level_rank(lane.get("level")) >= min_rank:
            active.append(name)
    return sorted(set(active))


def _lane_meta(row: Dict[str, Any], lane_name: str) -> Dict[str, Any]:
    for lane in _row_lanes(row):
        if isinstance(lane, dict) and str(lane.get("lane") or "") == lane_name:
            return lane
    return {}


def _row_lanes(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    lanes = [lane for lane in (row.get("lanes") or []) if isinstance(lane, dict)]
    if lanes:
        return lanes
    top_event = dict(row.get("top_event") or {})
    top_lane = str(row.get("top_lane") or top_event.get("lane") or "")
    if not top_lane:
        return []
    return [
        {
            "lane": top_lane,
            "level": str(top_event.get("level") or (row.get("banner") or {}).get("top_level") or "quiet"),
            "freshness_status": "stale",
            "recommended_action": str(top_event.get("recommended_action") or "monitor_only"),
            "priority_score": 0.0,
        }
    ]


def build_overlap_payload(
    *,
    history_rows: Sequence[Dict[str, Any]],
    history_jsonl: str,
    min_level: str,
    top_n: int,
    out_json: str,
    out_md: str,
) -> Dict[str, Any]:
    lanes = sorted(
        {
            str(lane.get("lane") or "")
            for row in history_rows
            for lane in _row_lanes(row)
            if isinstance(lane, dict) and str(lane.get("lane") or "")
        }
    )
    active_sets: List[Tuple[Dict[str, Any], List[str]]] = [(row, _active_lanes(row, min_level)) for row in history_rows]
    active_snapshot_count = sum(1 for _, active in active_sets if active)
    lane_stats: List[Dict[str, Any]] = []
    for lane_name in lanes:
        active_count = 0
        fresh_active_count = 0
        top_count = 0
        for row, active in active_sets:
            if lane_name not in active:
                continue
            active_count += 1
            if str(row.get("top_lane") or "") == lane_name:
                top_count += 1
            meta = _lane_meta(row, lane_name)
            if str(meta.get("freshness_status") or "stale") == "fresh":
                fresh_active_count += 1
        lane_stats.append(
            {
                "lane": lane_name,
                "active_count": active_count,
                "active_rate": (active_count / len(history_rows)) if history_rows else 0.0,
                "fresh_active_count": fresh_active_count,
                "top_count": top_count,
            }
        )
    lane_stats.sort(key=lambda row: (float(row["active_rate"]), int(row["active_count"]), row["lane"]), reverse=True)

    pairwise: List[Dict[str, Any]] = []
    for idx, lane_a in enumerate(lanes):
        for lane_b in lanes[idx + 1 :]:
            coactive_count = 0
            lane_a_count = 0
            lane_b_count = 0
            for _, active in active_sets:
                has_a = lane_a in active
                has_b = lane_b in active
                if has_a:
                    lane_a_count += 1
                if has_b:
                    lane_b_count += 1
                if has_a and has_b:
                    coactive_count += 1
            union_count = lane_a_count + lane_b_count - coactive_count
            pairwise.append(
                {
                    "lane_a": lane_a,
                    "lane_b": lane_b,
                    "coactive_count": coactive_count,
                    "coactive_rate": (coactive_count / len(history_rows)) if history_rows else 0.0,
                    "jaccard": (coactive_count / union_count) if union_count else 0.0,
                }
            )
    pairwise.sort(key=lambda row: (float(row["jaccard"]), int(row["coactive_count"]), row["lane_a"], row["lane_b"]), reverse=True)
    strongest_overlaps = pairwise[: max(int(top_n), 0)]
    redundancy_notes: List[str] = []
    for row in strongest_overlaps:
        if int(row["coactive_count"]) >= 2 and float(row["jaccard"]) >= 0.6:
            redundancy_notes.append(
                f"{row['lane_a']} and {row['lane_b']} co-activate frequently "
                f"(jaccard={float(row['jaccard']):.2f}, coactive_count={int(row['coactive_count'])})."
            )
    summary = {
        "available_rows": len(history_rows),
        "used_rows": len(history_rows),
        "lane_count": len(lanes),
        "active_lane_count": sum(1 for row in lane_stats if int(row["active_count"]) > 0),
        "active_snapshot_count": active_snapshot_count,
        "min_level": str(min_level),
        "top_overlap_pair": (
            f"{strongest_overlaps[0]['lane_a']}::{strongest_overlaps[0]['lane_b']}" if strongest_overlaps else ""
        ),
    }
    payload = {
        "history_jsonl": str(history_jsonl),
        "summary": summary,
        "lane_stats": lane_stats,
        "pairwise": pairwise,
        "strongest_overlaps": strongest_overlaps,
        "redundancy_notes": redundancy_notes,
    }
    payload["run_summary"] = build_run_summary(
        run_type="event_lane_overlap",
        inputs={"history_jsonl": str(history_jsonl), "min_level": str(min_level), "top_n": int(top_n)},
        metrics={
            "available_rows": len(history_rows),
            "lane_count": len(lanes),
            "active_snapshot_count": active_snapshot_count,
            "top_overlap_pair": summary["top_overlap_pair"],
        },
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze overlap between research event lanes from snapshot history.")
    p.add_argument("--history-jsonl", default="reports/RESEARCH_EVENT_WATCHBOARD_HISTORY.jsonl")
    p.add_argument("--min-level", default="elevated")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--out-json", default="reports/EVENT_LANE_OVERLAP.json")
    p.add_argument("--out-md", default="reports/EVENT_LANE_OVERLAP.md")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    history_path = Path(str(args.history_jsonl))
    rows = _load_history_rows(history_path)
    payload = build_overlap_payload(
        history_rows=rows,
        history_jsonl=str(args.history_jsonl),
        min_level=str(args.min_level),
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
        "# EVENT LANE OVERLAP",
        "",
        f"history_jsonl={payload['history_jsonl']}",
        f"summary={json.dumps(payload['summary'], ensure_ascii=True, sort_keys=True)}",
        "",
        "| lane | active_count | active_rate | fresh_active_count | top_count |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["lane_stats"]:
        lines.append(
            f"| {row['lane']} | {int(row['active_count'])} | {float(row['active_rate']):.4f} | "
            f"{int(row['fresh_active_count'])} | {int(row['top_count'])} |"
        )
    lines.extend(
        [
            "",
            "| lane_a | lane_b | coactive_count | coactive_rate | jaccard |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in payload["strongest_overlaps"]:
        lines.append(
            f"| {row['lane_a']} | {row['lane_b']} | {int(row['coactive_count'])} | "
            f"{float(row['coactive_rate']):.4f} | {float(row['jaccard']):.4f} |"
        )
    if payload["redundancy_notes"]:
        lines.extend(["", "## Redundancy Notes", ""])
        lines.extend(f"- {note}" for note in payload["redundancy_notes"])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
