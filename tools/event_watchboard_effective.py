from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.run_summary import build_run_summary


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _apply_display_mode(row: Dict[str, Any], mode: str) -> Dict[str, Any]:
    updated = dict(row)
    updated["effective_display_mode"] = str(mode)
    if mode == "degrade":
        updated["effective_priority_score"] = float(updated.get("priority_score") or 0.0) - 80.0
    elif mode == "collapse":
        updated["effective_priority_score"] = float(updated.get("priority_score") or 0.0) - 120.0
    elif mode == "hide":
        updated["effective_priority_score"] = -1.0
    else:
        updated["effective_priority_score"] = float(updated.get("priority_score") or 0.0)
    return updated


def build_effective_watchboard_payload(
    *,
    watchboard_json: str,
    suppression_json: str,
    persistence_json: str,
    out_json: str,
    out_md: str,
) -> Dict[str, Any]:
    watchboard = _load_json(watchboard_json)
    suppression = _load_json(suppression_json)
    persistence = _load_json(persistence_json)
    suppress_map = {
        str(row.get("secondary_lane") or ""): str(row.get("display_mode") or "keep")
        for row in (suppression.get("rules") or [])
        if isinstance(row, dict) and str(row.get("secondary_lane") or "")
    }
    persistence_map = {
        str(row.get("lane") or ""): row
        for row in (persistence.get("lanes") or [])
        if isinstance(row, dict) and str(row.get("lane") or "")
    }
    raw_lanes = [dict(row) for row in (watchboard.get("lanes") or []) if isinstance(row, dict)]
    effective_lanes = []
    for row in raw_lanes:
        lane = str(row.get("lane") or "")
        updated = _apply_display_mode(row, suppress_map.get(lane, "keep"))
        persistence_row = persistence_map.get(lane, {})
        updated["persistence_recommendation"] = str(persistence_row.get("recommendation") or "keep_immediate")
        updated["recommended_min_persist_snapshots"] = int(persistence_row.get("recommended_min_persist_snapshots") or 1)
        updated["recommended_cooldown_snapshots"] = int(persistence_row.get("recommended_cooldown_snapshots") or 0)
        updated["is_noisy"] = bool(persistence_row.get("is_noisy"))
        effective_lanes.append(updated)
    visible_lanes = [row for row in effective_lanes if str(row.get("effective_display_mode") or "keep") != "hide"]
    visible_lanes.sort(key=lambda row: float(row.get("effective_priority_score") or 0.0), reverse=True)
    top_effective = visible_lanes[0] if visible_lanes else {}
    persistence_summary = persistence.get("summary") or {}
    summary = {
        "raw_top_lane": str((watchboard.get("summary") or {}).get("top_lane") or ""),
        "effective_top_lane": str(top_effective.get("lane") or ""),
        "hidden_lane_count": sum(1 for row in effective_lanes if str(row.get("effective_display_mode") or "") == "hide"),
        "degraded_lane_count": sum(1 for row in effective_lanes if str(row.get("effective_display_mode") or "") == "degrade"),
        "collapsed_lane_count": sum(1 for row in effective_lanes if str(row.get("effective_display_mode") or "") == "collapse"),
        "noisy_lane_count": int(persistence_summary.get("noisy_lane_count") or 0),
        "primary_noisy_lane": str(persistence_summary.get("primary_noisy_lane") or ""),
    }
    payload = {
        "watchboard_json": str(watchboard_json),
        "suppression_json": str(suppression_json),
        "persistence_json": str(persistence_json),
        "summary": summary,
        "effective_top_event": {
            "lane": str(top_effective.get("lane") or ""),
            "level": str(top_effective.get("level") or "quiet"),
            "recommended_action": str(top_effective.get("recommended_action") or "monitor_only"),
            "effective_display_mode": str(top_effective.get("effective_display_mode") or "keep"),
            "persistence_recommendation": str(top_effective.get("persistence_recommendation") or "keep_immediate"),
            "recommended_min_persist_snapshots": int(top_effective.get("recommended_min_persist_snapshots") or 1),
            "recommended_cooldown_snapshots": int(top_effective.get("recommended_cooldown_snapshots") or 0),
        },
        "lanes": effective_lanes,
    }
    payload["run_summary"] = build_run_summary(
        run_type="event_watchboard_effective",
        inputs={
            "watchboard_json": str(watchboard_json),
            "suppression_json": str(suppression_json),
            "persistence_json": str(persistence_json),
        },
        metrics=summary,
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply suppression policy to a raw watchboard and produce an effective view.")
    p.add_argument("--watchboard-json", default="reports/RESEARCH_EVENT_WATCHBOARD.json")
    p.add_argument("--suppression-json", default="reports/EVENT_LANE_SUPPRESSION_POLICY.json")
    p.add_argument("--persistence-json", default="reports/EVENT_LANE_PERSISTENCE_POLICY.json")
    p.add_argument("--out-json", default="reports/EVENT_WATCHBOARD_EFFECTIVE.json")
    p.add_argument("--out-md", default="reports/EVENT_WATCHBOARD_EFFECTIVE.md")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    payload = build_effective_watchboard_payload(
        watchboard_json=str(args.watchboard_json),
        suppression_json=str(args.suppression_json),
        persistence_json=str(args.persistence_json),
        out_json=str(args.out_json),
        out_md=str(args.out_md),
    )
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# EVENT WATCHBOARD EFFECTIVE",
        "",
        f"summary={json.dumps(payload['summary'], ensure_ascii=True, sort_keys=True)}",
        "",
        "| lane | level | recommended_action | effective_display_mode | effective_priority_score |",
        "|---|---|---|---|---:|",
    ]
    for row in payload["lanes"]:
        lines.append(
            f"| {row['lane']} | {row['level']} | {row['recommended_action']} | {row['effective_display_mode']} | {float(row['effective_priority_score']):.2f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
