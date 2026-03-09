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


def _top_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    rows = payload.get("ranking", [])
    if isinstance(rows, list) and rows:
        top = rows[0]
        if isinstance(top, dict):
            return top
    return {}


def build_payload(*, baseline_json: str, filtered_json: str, out_json: str) -> Dict[str, Any]:
    baseline = _load_json(baseline_json)
    filtered = _load_json(filtered_json)
    base_top = _top_row(baseline)
    filt_top = _top_row(filtered)

    delta_npa = _safe_float(filt_top.get("npa_core")) - _safe_float(base_top.get("npa_core"))
    delta_score = _safe_float(filt_top.get("score_raw_core")) - _safe_float(base_top.get("score_raw_core"))
    delta_fill = _safe_float(filt_top.get("attempt_fill_rate")) - _safe_float(base_top.get("attempt_fill_rate"))
    kept_ratio = _safe_float(filt_top.get("event_filter_kept_ratio"), 1.0)

    recommendation = "keep_baseline"
    if delta_npa > 0.0 and kept_ratio >= 0.50:
        recommendation = "test_event_block_v1_in_rank_pipeline"
    elif delta_npa > 0.0:
        recommendation = "quality_up_but_coverage_thin"

    payload = {
        "source_baseline_json": str(baseline_json),
        "source_filtered_json": str(filtered_json),
        "baseline_top": {
            "symbol": str(base_top.get("symbol", "")),
            "rule": str(base_top.get("rule", "")),
            "npa_core": _safe_float(base_top.get("npa_core")),
            "score_raw_core": _safe_float(base_top.get("score_raw_core")),
            "attempt_fill_rate": _safe_float(base_top.get("attempt_fill_rate")),
        },
        "filtered_top": {
            "symbol": str(filt_top.get("symbol", "")),
            "rule": str(filt_top.get("rule", "")),
            "npa_core": _safe_float(filt_top.get("npa_core")),
            "score_raw_core": _safe_float(filt_top.get("score_raw_core")),
            "attempt_fill_rate": _safe_float(filt_top.get("attempt_fill_rate")),
            "event_block_lanes": list(filt_top.get("event_block_lanes", [])),
            "event_filter_kept_ratio": kept_ratio,
        },
        "delta": {
            "npa_core": delta_npa,
            "score_raw_core": delta_score,
            "attempt_fill_rate": delta_fill,
        },
        "recommendation": str(recommendation),
    }
    payload["run_summary"] = build_run_summary(
        run_type="summarize_rank_event_filter",
        inputs={"baseline_json": str(baseline_json), "filtered_json": str(filtered_json)},
        metrics={
            "delta_npa_core": delta_npa,
            "delta_score_raw_core": delta_score,
            "filtered_kept_ratio": kept_ratio,
        },
        artifacts={"json": str(out_json)},
    )
    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize baseline vs event-filtered rank outputs.")
    p.add_argument("--baseline-json", required=True)
    p.add_argument("--filtered-json", required=True)
    p.add_argument("--out-json", default="reports/RANK_EVENT_FILTER_SUMMARY.json")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    payload = build_payload(
        baseline_json=str(args.baseline_json),
        filtered_json=str(args.filtered_json),
        out_json=str(args.out_json),
    )
    out_path = Path(str(args.out_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
