from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.run_summary import build_run_summary


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _severity_score(level: str) -> float:
    return {"quiet": 0.0, "elevated": 100.0, "severe": 200.0}.get(str(level), 0.0)


def _freshness_bonus(status: str) -> float:
    return 25.0 if str(status) == "fresh" else 0.0


def _snapshot_score(payload: Dict[str, Any]) -> float:
    top = dict(payload.get("top_event") or {})
    lanes = list(payload.get("lanes") or [])
    top_score = 0.0
    for lane in lanes:
        if str(lane.get("lane")) == str(top.get("lane")):
            top_score = _safe_float(lane.get("priority_score"))
            break
    if top_score > 0.0:
        return top_score
    return _severity_score(top.get("level")) + _freshness_bonus(top.get("freshness_status"))


def _trend_label(delta: float) -> str:
    if delta >= 50.0:
        return "rising_fast"
    if delta > 0.0:
        return "rising"
    if delta <= -50.0:
        return "falling_fast"
    if delta < 0.0:
        return "falling"
    return "flat"


def _lane_scores(payload: Dict[str, Any]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for lane in list(payload.get("lanes") or []):
        lane_name = str(lane.get("lane") or "")
        if not lane_name:
            continue
        scores[lane_name] = _safe_float(lane.get("priority_score"))
    return scores


def build_trend_payload(*, snapshots: List[Dict[str, Any]], source_paths: List[str], out_json: str, out_md: str) -> Dict[str, Any]:
    points: List[Dict[str, Any]] = []
    for idx, payload in enumerate(snapshots):
        top = dict(payload.get("top_event") or {})
        summary = dict(payload.get("summary") or {})
        score = _snapshot_score(payload)
        points.append(
            {
                "index": idx,
                "source": str(source_paths[idx]) if idx < len(source_paths) else "",
                "top_lane": str(summary.get("top_lane") or top.get("lane") or ""),
                "top_level": str(top.get("level") or "quiet"),
                "top_recommended_action": str(top.get("recommended_action") or "monitor_only"),
                "priority_score": score,
            }
        )
    first = points[0] if points else {}
    last = points[-1] if points else {}
    delta = _safe_float(last.get("priority_score")) - _safe_float(first.get("priority_score"))
    lane_deltas: List[Dict[str, Any]] = []
    if snapshots:
        start_scores = _lane_scores(snapshots[0])
        end_scores = _lane_scores(snapshots[-1])
        lane_names = sorted(set(start_scores) | set(end_scores))
        for lane_name in lane_names:
            start_score = _safe_float(start_scores.get(lane_name))
            end_score = _safe_float(end_scores.get(lane_name))
            lane_delta = end_score - start_score
            lane_deltas.append(
                {
                    "lane": lane_name,
                    "start_priority_score": start_score,
                    "end_priority_score": end_score,
                    "delta_priority_score": lane_delta,
                    "trend": _trend_label(lane_delta),
                }
            )
        lane_deltas.sort(key=lambda row: abs(_safe_float(row.get("delta_priority_score"))), reverse=True)
    payload = {
        "summary": {
            "snapshot_count": len(points),
            "start_top_lane": str(first.get("top_lane") or ""),
            "end_top_lane": str(last.get("top_lane") or ""),
            "delta_priority_score": delta,
            "trend": _trend_label(delta),
        },
        "latest": dict(last),
        "points": points,
        "lane_deltas": lane_deltas,
    }
    payload["run_summary"] = build_run_summary(
        run_type="event_watchboard_trend",
        inputs={"sources": list(source_paths)},
        metrics={
            "snapshot_count": len(points),
            "delta_priority_score": delta,
            "trend": payload["summary"]["trend"],
        },
        artifacts={"json": out_json, "md": out_md},
    )
    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize trend across research event watchboard snapshots.")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out-json", default="reports/RESEARCH_EVENT_WATCHBOARD_TREND.json")
    p.add_argument("--out-md", default="reports/RESEARCH_EVENT_WATCHBOARD_TREND.md")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    source_paths = [str(x) for x in args.inputs]
    snapshots = [_load_json(Path(p)) for p in source_paths]
    payload = build_trend_payload(
        snapshots=snapshots,
        source_paths=source_paths,
        out_json=str(args.out_json),
        out_md=str(args.out_md),
    )
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# RESEARCH EVENT WATCHBOARD TREND",
        "",
        f"snapshot_count={payload['summary']['snapshot_count']}",
        f"start_top_lane={payload['summary']['start_top_lane']}",
        f"end_top_lane={payload['summary']['end_top_lane']}",
        f"delta_priority_score={float(payload['summary']['delta_priority_score']):.2f}",
        f"trend={payload['summary']['trend']}",
        "",
        "| index | top_lane | top_level | top_recommended_action | priority_score |",
        "|---:|---|---|---|---:|",
    ]
    for row in payload["points"]:
        lines.append(
            f"| {int(row['index'])} | {row['top_lane']} | {row['top_level']} | {row['top_recommended_action']} | {float(row['priority_score']):.2f} |"
        )
    if payload["lane_deltas"]:
        lines.extend(
            [
                "",
                "| lane | start_priority_score | end_priority_score | delta_priority_score | trend |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in payload["lane_deltas"]:
            lines.append(
                f"| {row['lane']} | {float(row['start_priority_score']):.2f} | {float(row['end_priority_score']):.2f} | "
                f"{float(row['delta_priority_score']):.2f} | {row['trend']} |"
            )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
