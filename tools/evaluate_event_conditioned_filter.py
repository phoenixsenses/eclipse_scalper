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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ranked(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = section.get("ranked", []) if isinstance(section, dict) else []
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "lane": str(row.get("lane", "")),
                "tagged_n": _safe_int(row.get("tagged_n")),
                "delta_avg_net": _safe_float(row.get("delta_avg_net")),
                "delta_p90_net": _safe_float(row.get("delta_p90_net")),
            }
        )
    return out


def build_payload(*, bridge_json: str, out_json: str, min_tagged_n: int = 3) -> Dict[str, Any]:
    payload = _load_json(bridge_json)
    discovery = payload.get("discovery", {}) if isinstance(payload, dict) else {}
    validation = payload.get("validation", {}) if isinstance(payload, dict) else {}

    disc_ranked = _ranked(discovery)
    valid_ranked = _ranked(validation)

    valid_positive = [r for r in valid_ranked if r["tagged_n"] >= int(min_tagged_n) and r["delta_avg_net"] > 0.0]
    valid_negative = [r for r in valid_ranked if r["tagged_n"] >= int(min_tagged_n) and r["delta_avg_net"] < 0.0]
    disc_positive = [r for r in disc_ranked if r["tagged_n"] >= int(min_tagged_n) and r["delta_avg_net"] > 0.0]
    disc_negative = [r for r in disc_ranked if r["tagged_n"] >= int(min_tagged_n) and r["delta_avg_net"] < 0.0]

    primary_allow = valid_positive[0] if valid_positive else {}
    tentative_allow = {}
    for row in disc_positive:
        if row.get("lane") != primary_allow.get("lane"):
            tentative_allow = row
            break

    block_lanes: List[Dict[str, Any]] = []
    for row in valid_negative:
        lane = str(row.get("lane", ""))
        disc_match = next((d for d in disc_negative if str(d.get("lane", "")) == lane), None)
        block_lanes.append(
            {
                "lane": lane,
                "validation_delta_avg_net": float(row.get("delta_avg_net", 0.0) or 0.0),
                "validation_tagged_n": int(row.get("tagged_n", 0) or 0),
                "confirmed_in_discovery": bool(disc_match is not None),
                "discovery_delta_avg_net": float(disc_match.get("delta_avg_net", 0.0) or 0.0) if disc_match else 0.0,
            }
        )

    recommendation = "keep_descriptive_only"
    if primary_allow:
        recommendation = "test_allow_filter"
    if block_lanes:
        recommendation = "test_allow_and_block_filters" if primary_allow else "test_block_filter"

    summary = {
        "primary_allow_lane": str(primary_allow.get("lane", "")),
        "tentative_allow_lane": str(tentative_allow.get("lane", "")),
        "block_lane_count": int(len(block_lanes)),
        "recommendation": str(recommendation),
    }
    out = {
        "source_bridge_json": str(bridge_json),
        "summary": summary,
        "filter_candidate": {
            "min_tagged_n": int(min_tagged_n),
            "allow_lanes": [str(primary_allow.get("lane", ""))] if primary_allow else [],
            "tentative_allow_lanes": [str(tentative_allow.get("lane", ""))] if tentative_allow else [],
            "block_lanes": block_lanes,
        },
        "run_summary": build_run_summary(
            run_type="evaluate_event_conditioned_filter",
            inputs={"bridge_json": str(bridge_json), "min_tagged_n": int(min_tagged_n)},
            metrics=summary,
            artifacts={"json": str(out_json)},
        ),
    }
    return out


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate event-conditioned filter candidates from bridge summary.")
    p.add_argument("--bridge-json", required=True)
    p.add_argument("--min-tagged-n", type=int, default=3)
    p.add_argument("--out-json", default="reports/EVENT_CONDITIONED_FILTER.json")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    payload = build_payload(
        bridge_json=str(args.bridge_json),
        out_json=str(args.out_json),
        min_tagged_n=int(args.min_tagged_n),
    )
    out_path = Path(str(args.out_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
