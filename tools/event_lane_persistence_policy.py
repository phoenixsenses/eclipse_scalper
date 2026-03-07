from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.event_watchboard_trend_from_history import _load_history
from tools.run_summary import build_run_summary


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _history_lane_sequence(rows: List[Dict[str, Any]]) -> List[str]:
    sequence: List[str] = []
    for row in rows:
        top_event = row.get("top_event") or {}
        lane = str(row.get("top_lane") or top_event.get("lane") or "")
        sequence.append(lane)
    return sequence


def build_persistence_policy_payload(
    *,
    history_rows: List[Dict[str, Any]],
    history_path: str,
    last_n: int,
    out_json: str,
    out_md: str,
) -> Dict[str, Any]:
    rows = history_rows[-max(1, int(last_n)) :]
    sequence = _history_lane_sequence(rows)
    lanes = sorted({lane for lane in sequence if lane})
    lane_stats: List[Dict[str, Any]] = []
    total_flips = 0

    for idx in range(1, len(sequence)):
        if sequence[idx] and sequence[idx - 1] and sequence[idx] != sequence[idx - 1]:
            total_flips += 1

    for lane in lanes:
        hits = sum(1 for item in sequence if item == lane)
        longest_streak = 0
        current_streak = 0
        transitions_involved = 0
        for idx, item in enumerate(sequence):
            if item == lane:
                current_streak += 1
                longest_streak = max(longest_streak, current_streak)
                if idx > 0 and sequence[idx - 1] and sequence[idx - 1] != lane:
                    transitions_involved += 1
                if idx + 1 < len(sequence) and sequence[idx + 1] and sequence[idx + 1] != lane:
                    transitions_involved += 1
            else:
                current_streak = 0
        hit_rate = hits / len(sequence) if sequence else 0.0
        noisy = hits >= 2 and longest_streak <= 1 and transitions_involved >= 2
        lane_stats.append(
            {
                "lane": lane,
                "top_hits": hits,
                "hit_rate": hit_rate,
                "longest_streak": longest_streak,
                "transitions_involved": transitions_involved,
                "is_noisy": noisy,
                "recommended_min_persist_snapshots": 2 if noisy else 1,
                "recommended_cooldown_snapshots": 1 if noisy else 0,
                "recommendation": "stabilize_banner" if noisy else "keep_immediate",
            }
        )

    lane_stats.sort(
        key=lambda row: (
            bool(row["is_noisy"]),
            int(row["transitions_involved"]),
            -int(row["longest_streak"]),
            str(row["lane"]),
        ),
        reverse=True,
    )
    noisy_lanes = [row for row in lane_stats if bool(row["is_noisy"])]
    summary = {
        "history_path": str(history_path),
        "available_rows": len(history_rows),
        "used_rows": len(rows),
        "sequence_length": len(sequence),
        "latest_top_lane": sequence[-1] if sequence else "",
        "flip_count": total_flips,
        "noisy_lane_count": len(noisy_lanes),
        "primary_noisy_lane": str(noisy_lanes[0]["lane"]) if noisy_lanes else "",
    }
    payload = {
        "history_path": str(history_path),
        "last_n": int(last_n),
        "summary": summary,
        "lanes": lane_stats,
    }
    payload["run_summary"] = build_run_summary(
        run_type="event_lane_persistence_policy",
        inputs={"history_path": str(history_path), "last_n": int(last_n)},
        metrics={
            "used_rows": len(rows),
            "flip_count": total_flips,
            "noisy_lane_count": len(noisy_lanes),
            "primary_noisy_lane": str(noisy_lanes[0]["lane"]) if noisy_lanes else "",
        },
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build lane persistence and cooldown policy from watchboard history.")
    p.add_argument("--history-jsonl", default="reports/RESEARCH_EVENT_WATCHBOARD_HISTORY.jsonl")
    p.add_argument("--last-n", type=int, default=24)
    p.add_argument("--out-json", default="reports/EVENT_LANE_PERSISTENCE_POLICY.json")
    p.add_argument("--out-md", default="reports/EVENT_LANE_PERSISTENCE_POLICY.md")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    history_path = Path(str(args.history_jsonl))
    history_rows = _load_history(history_path)
    payload = build_persistence_policy_payload(
        history_rows=history_rows,
        history_path=str(history_path),
        last_n=int(args.last_n),
        out_json=str(args.out_json),
        out_md=str(args.out_md),
    )
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# EVENT LANE PERSISTENCE POLICY",
        "",
        f"summary={json.dumps(payload['summary'], ensure_ascii=True, sort_keys=True)}",
        "",
        "| lane | hit_rate | longest_streak | transitions_involved | is_noisy | min_persist | cooldown |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in payload["lanes"]:
        lines.append(
            f"| {row['lane']} | {_safe_float(row['hit_rate']):.4f} | {int(row['longest_streak'])} | {int(row['transitions_involved'])} | {str(bool(row['is_noisy'])).lower()} | {int(row['recommended_min_persist_snapshots'])} | {int(row['recommended_cooldown_snapshots'])} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
