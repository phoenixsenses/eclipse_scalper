from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.evaluate_event_conditioned_forward import build_payload as build_forward_payload
from tools.run_summary import build_run_summary


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_filter(path: Path, allow_lanes: List[str], block_lanes: List[str]) -> None:
    payload = {
        "filter_candidate": {
            "allow_lanes": allow_lanes,
            "block_lanes": [{"lane": lane} for lane in block_lanes],
        }
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def build_payload(*, debug_jsonl: str, filter_json: str, out_json: str) -> Dict[str, Any]:
    filter_payload = _load_json(filter_json)
    candidate = filter_payload.get("filter_candidate", {}) if isinstance(filter_payload, dict) else {}
    primary_allow = [str(x) for x in candidate.get("allow_lanes", []) if str(x)]
    tentative_allow = [str(x) for x in candidate.get("tentative_allow_lanes", []) if str(x)]
    block_lanes = [str(item.get("lane", "")) for item in candidate.get("block_lanes", []) if str(item.get("lane", ""))]

    variants = [
        {"name": "baseline_like", "allow": [], "block": []},
        {"name": "block_only", "allow": [], "block": block_lanes},
        {"name": "primary_allow_block", "allow": primary_allow, "block": block_lanes},
        {"name": "allow_union_block", "allow": sorted(set(primary_allow + tentative_allow)), "block": block_lanes},
        {"name": "allow_union_only", "allow": sorted(set(primary_allow + tentative_allow)), "block": []},
    ]

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = out_path.parent / f".tmp_event_filter_grid_{out_path.stem}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    best_name = ""
    best_delta = float("-inf")
    best_kept_ratio = 0.0
    try:
        for variant in variants:
            filt_path = tmp_root / f"{variant['name']}.filter.json"
            eval_path = tmp_root / f"{variant['name']}.eval.json"
            _write_filter(filt_path, allow_lanes=list(variant["allow"]), block_lanes=list(variant["block"]))
            result = build_forward_payload(debug_jsonl=debug_jsonl, filter_json=str(filt_path), out_json=str(eval_path))
            row = {
                "variant": str(variant["name"]),
                "allow_lanes": list(variant["allow"]),
                "block_lanes": list(variant["block"]),
                "validation_delta_avg_net": float(result["validation"]["delta_avg_net"]),
                "validation_delta_p90_net": float(result["validation"]["delta_p90_net"]),
                "validation_kept_ratio": float(result["validation"]["kept_ratio"]),
                "validation_filtered_n": int(result["validation"]["filtered"]["n"]),
                "recommendation": str(result["recommendation"]),
            }
            rows.append(row)
            score = row["validation_delta_avg_net"]
            if score > best_delta:
                best_name = row["variant"]
                best_delta = score
                best_kept_ratio = row["validation_kept_ratio"]
    finally:
        for child in tmp_root.glob("*"):
            child.unlink(missing_ok=True)
        tmp_root.rmdir()

    rows.sort(key=lambda r: (r["validation_delta_avg_net"], r["validation_kept_ratio"]), reverse=True)
    payload = {
        "source_debug_jsonl": str(debug_jsonl),
        "source_filter_json": str(filter_json),
        "variant_count": int(len(rows)),
        "best_variant": str(best_name),
        "best_validation_delta_avg_net": float(best_delta if best_delta != float("-inf") else 0.0),
        "best_validation_kept_ratio": float(best_kept_ratio),
        "rows": rows,
    }
    payload["run_summary"] = build_run_summary(
        run_type="evaluate_event_conditioned_forward_grid",
        inputs={"debug_jsonl": str(debug_jsonl), "filter_json": str(filter_json)},
        metrics={
            "variant_count": int(len(rows)),
            "best_validation_delta_avg_net": float(payload["best_validation_delta_avg_net"]),
            "best_validation_kept_ratio": float(payload["best_validation_kept_ratio"]),
        },
        artifacts={"json": str(out_json)},
    )
    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search small event-conditioned forward filter variants.")
    p.add_argument("--debug-jsonl", required=True)
    p.add_argument("--filter-json", required=True)
    p.add_argument("--out-json", default="reports/EVENT_CONDITIONED_FORWARD_GRID.json")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    payload = build_payload(debug_jsonl=str(args.debug_jsonl), filter_json=str(args.filter_json), out_json=str(args.out_json))
    out_path = Path(str(args.out_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
