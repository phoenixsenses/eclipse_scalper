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


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rank_lanes(by_lane: Dict[str, Any]) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for lane, section in by_lane.items():
        if not isinstance(section, dict):
            continue
        ranked.append(
            {
                "lane": str(lane),
                "tagged_n": int(section.get("tagged_n", 0) or 0),
                "delta_avg_net": _safe_float(section.get("delta_avg_net")),
                "delta_p90_net": _safe_float(section.get("delta_p90_net")),
            }
        )
    ranked.sort(key=lambda row: (row["delta_avg_net"], row["delta_p90_net"], row["tagged_n"]), reverse=True)
    return ranked


def _summarize_side(section: Dict[str, Any]) -> Dict[str, Any]:
    by_lane = section.get("by_lane", {}) if isinstance(section, dict) else {}
    ranked = _rank_lanes(by_lane if isinstance(by_lane, dict) else {})
    positive = [row for row in ranked if row["tagged_n"] > 0 and row["delta_avg_net"] > 0.0]
    negative = sorted(
        [row for row in ranked if row["tagged_n"] > 0 and row["delta_avg_net"] < 0.0],
        key=lambda row: (row["delta_avg_net"], row["delta_p90_net"]),
    )
    best = positive[0] if positive else (ranked[0] if ranked else {"lane": "", "tagged_n": 0, "delta_avg_net": 0.0, "delta_p90_net": 0.0})
    worst = negative[0] if negative else {"lane": "", "tagged_n": 0, "delta_avg_net": 0.0, "delta_p90_net": 0.0}
    return {
        "available": bool(section.get("available")),
        "rows_total": int(section.get("rows_total", 0) or 0),
        "best_positive_lane": best,
        "worst_negative_lane": worst,
        "positive_lane_count": int(len(positive)),
        "negative_lane_count": int(len(negative)),
        "ranked": ranked,
    }


def build_payload(*, forward_json: str, out_json: str) -> Dict[str, Any]:
    payload = _load_json(forward_json)
    event_ctx = payload.get("event_lane_context_impact", {}) if isinstance(payload, dict) else {}
    discovery = _summarize_side(event_ctx.get("discovery", {}) if isinstance(event_ctx, dict) else {})
    validation = _summarize_side(event_ctx.get("validation", {}) if isinstance(event_ctx, dict) else {})

    recommendation = "keep_lane_context_descriptive"
    if validation.get("best_positive_lane", {}).get("lane") and _safe_float(validation.get("best_positive_lane", {}).get("delta_avg_net")) > 0.0:
        recommendation = "test_event_conditioned_filter"

    out = {
        "source_forward_json": str(forward_json),
        "discovery": discovery,
        "validation": validation,
        "recommendation": str(recommendation),
    }
    out["run_summary"] = build_run_summary(
        run_type="summarize_event_signal_bridge",
        inputs={"forward_json": str(forward_json)},
        metrics={
            "discovery_positive_lane_count": int(discovery.get("positive_lane_count", 0) or 0),
            "validation_positive_lane_count": int(validation.get("positive_lane_count", 0) or 0),
        },
        artifacts={"json": str(out_json)},
    )
    return out


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize event/signal overlap from validate_micro_edge_forward output.")
    p.add_argument("--forward-json", required=True)
    p.add_argument("--out-json", default="reports/EVENT_SIGNAL_BRIDGE_SUMMARY.json")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    payload = build_payload(forward_json=str(args.forward_json), out_json=str(args.out_json))
    out_path = Path(str(args.out_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
