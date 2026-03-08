from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tools.analyze_micro_edge_regimes import group_key, group_stats, load_debug_rows, summarize
from tools.run_summary import build_run_summary
from tools.validate_micro_edge_forward import _event_lane_tagged_rows, _selection


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _selected_rows(rows: List[Dict[str, Any]], fields: List[str], discover_frac: float, top_k: int, min_n_discovery: int, min_n_validation: int) -> Dict[str, Any]:
    rows = sorted(rows, key=lambda r: float(r.get("ts_bucket") or 0.0))
    n_total = len(rows)
    cut = max(1, min(n_total - 1, int(n_total * float(discover_frac)))) if n_total > 1 else n_total
    disc = rows[:cut]
    valid = rows[cut:]

    disc_all = group_stats(disc, group_fields=fields)
    valid_all = group_stats(valid, group_fields=fields)
    disc_pass = [g for g in disc_all if int(g.get("n", 0) or 0) >= int(min_n_discovery)]
    valid_n_by_group = {str(g["group"]): int(g.get("n", 0) or 0) for g in valid_all}
    disc_pass.sort(key=lambda r: float(r.get("avg_net", 0.0) or 0.0), reverse=True)
    top_disc = disc_pass[: max(1, int(top_k))]
    top_keys = {str(g["group"]) for g in top_disc if int(valid_n_by_group.get(str(g["group"]), 0)) >= int(min_n_validation)}
    return {
        "discovery": _selection(disc, top_keys, fields),
        "validation": _selection(valid, top_keys, fields),
    }


def _apply_filter(rows: List[Dict[str, Any]], allow: Set[str], block: Set[str]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    lane_map = _event_lane_tagged_rows(rows)
    lane_membership = {
        lane: {id(row) for row in tagged}
        for lane, tagged in lane_map.items()
    }
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        row_id = id(row)
        row_lanes = {lane for lane, members in lane_membership.items() if row_id in members}
        if allow and not (row_lanes & allow):
            continue
        if block and (row_lanes & block):
            continue
        filtered.append(row)
    return filtered


def _section_payload(baseline: List[Dict[str, Any]], filtered: List[Dict[str, Any]]) -> Dict[str, Any]:
    base_sm = summarize(baseline)
    filt_sm = summarize(filtered)
    base_n = int(base_sm.get("n", 0) or 0)
    filt_n = int(filt_sm.get("n", 0) or 0)
    return {
        "baseline": {
            "n": base_n,
            "avg_net": float(base_sm.get("avg_net", 0.0) or 0.0),
            "p90_net": float(base_sm.get("p90_net", 0.0) or 0.0),
        },
        "filtered": {
            "n": filt_n,
            "avg_net": float(filt_sm.get("avg_net", 0.0) or 0.0),
            "p90_net": float(filt_sm.get("p90_net", 0.0) or 0.0),
        },
        "delta_avg_net": float(filt_sm.get("avg_net", 0.0) or 0.0) - float(base_sm.get("avg_net", 0.0) or 0.0),
        "delta_p90_net": float(filt_sm.get("p90_net", 0.0) or 0.0) - float(base_sm.get("p90_net", 0.0) or 0.0),
        "kept_ratio": (float(filt_n) / float(base_n)) if base_n else 0.0,
    }


def build_payload(*, debug_jsonl: str, filter_json: str, out_json: str) -> Dict[str, Any]:
    rows = load_debug_rows(Path(debug_jsonl))
    filter_payload = _load_json(filter_json)
    allow = {str(x) for x in filter_payload.get("filter_candidate", {}).get("allow_lanes", []) if str(x)}
    block = {str(item.get("lane", "")) for item in filter_payload.get("filter_candidate", {}).get("block_lanes", []) if str(item.get("lane", ""))}
    fields = ["regime_spread_bin", "regime_intensity_bin", "regime_vol_bin", "regime_imb_bin"]

    selected = _selected_rows(rows, fields=fields, discover_frac=0.6, top_k=5, min_n_discovery=10, min_n_validation=10)
    disc_sel = selected["discovery"]
    valid_sel = selected["validation"]
    disc_filtered = _apply_filter(disc_sel, allow=allow, block=block)
    valid_filtered = _apply_filter(valid_sel, allow=allow, block=block)

    discovery = _section_payload(disc_sel, disc_filtered)
    validation = _section_payload(valid_sel, valid_filtered)
    recommendation = "keep_baseline"
    if validation["delta_avg_net"] > 0.0 and validation["filtered"]["n"] > 0:
        recommendation = "test_filter_in_rank_pipeline"

    payload = {
        "source_debug_jsonl": str(debug_jsonl),
        "source_filter_json": str(filter_json),
        "allow_lanes": sorted(allow),
        "block_lanes": sorted(block),
        "discovery": discovery,
        "validation": validation,
        "recommendation": recommendation,
    }
    payload["run_summary"] = build_run_summary(
        run_type="evaluate_event_conditioned_forward",
        inputs={"debug_jsonl": str(debug_jsonl), "filter_json": str(filter_json)},
        metrics={
            "validation_delta_avg_net": float(validation["delta_avg_net"]),
            "validation_kept_ratio": float(validation["kept_ratio"]),
        },
        artifacts={"json": str(out_json)},
    )
    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate event-conditioned filters on selected forward rows.")
    p.add_argument("--debug-jsonl", required=True)
    p.add_argument("--filter-json", required=True)
    p.add_argument("--out-json", default="reports/EVENT_CONDITIONED_FORWARD.json")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    payload = build_payload(
        debug_jsonl=str(args.debug_jsonl),
        filter_json=str(args.filter_json),
        out_json=str(args.out_json),
    )
    out_path = Path(str(args.out_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
