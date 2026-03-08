from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
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


def _ranking_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("ranking", [])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _row_key(row: Dict[str, Any]) -> str:
    return (
        f"{str(row.get('symbol', ''))}|{str(row.get('rule', ''))}|"
        f"{int(_safe_float(row.get('horizon_sec', 0), 0.0))}|"
        f"{_safe_float(row.get('min_imbalance')):.6f}|"
        f"{_safe_float(row.get('min_trade_intensity')):.6f}|"
        f"{_safe_float(row.get('max_spread')):.9f}"
    )


def _median_or_zero(values: List[float]) -> float:
    return float(median(values)) if values else 0.0


def build_payload(*, baseline_json: str, filtered_json: str, out_json: str) -> Dict[str, Any]:
    baseline = _load_json(baseline_json)
    filtered = _load_json(filtered_json)
    baseline_rows = _ranking_rows(baseline)
    filtered_rows = _ranking_rows(filtered)
    baseline_by_key = {_row_key(row): row for row in baseline_rows}
    filtered_by_key = {_row_key(row): row for row in filtered_rows}
    common_keys = sorted(set(baseline_by_key) & set(filtered_by_key))

    compared: List[Dict[str, Any]] = []
    for key in common_keys:
        base = baseline_by_key[key]
        filt = filtered_by_key[key]
        compared.append(
            {
                "key": key,
                "symbol": str(filt.get("symbol", "")),
                "rule": str(filt.get("rule", "")),
                "delta_npa_core": _safe_float(filt.get("npa_core")) - _safe_float(base.get("npa_core")),
                "delta_score_raw_core": _safe_float(filt.get("score_raw_core")) - _safe_float(base.get("score_raw_core")),
                "delta_attempt_fill_rate": _safe_float(filt.get("attempt_fill_rate")) - _safe_float(base.get("attempt_fill_rate")),
                "filtered_kept_ratio": _safe_float(filt.get("event_filter_kept_ratio"), 1.0),
            }
        )

    improved = [row for row in compared if _safe_float(row.get("delta_npa_core")) > 0.0]
    degraded = [row for row in compared if _safe_float(row.get("delta_npa_core")) < 0.0]
    best_tradeoff = sorted(
        compared,
        key=lambda row: (_safe_float(row.get("delta_npa_core")) * max(0.0, _safe_float(row.get("filtered_kept_ratio"), 1.0))),
        reverse=True,
    )

    recommendation = "keep_baseline"
    if improved and (len(improved) / max(1, len(compared))) >= 0.5:
        recommendation = "test_event_block_v1_more_broadly"
    elif improved:
        recommendation = "mixed_candidate_set_result"

    payload = {
        "source_baseline_json": str(baseline_json),
        "source_filtered_json": str(filtered_json),
        "common_count": int(len(compared)),
        "improved_count": int(len(improved)),
        "degraded_count": int(len(degraded)),
        "median_delta_npa_core": _median_or_zero([_safe_float(row.get("delta_npa_core")) for row in compared]),
        "median_delta_score_raw_core": _median_or_zero([_safe_float(row.get("delta_score_raw_core")) for row in compared]),
        "median_filtered_kept_ratio": _median_or_zero([_safe_float(row.get("filtered_kept_ratio"), 1.0) for row in compared]),
        "best_tradeoff_row": dict(best_tradeoff[0]) if best_tradeoff else {},
        "recommendation": str(recommendation),
        "rows": compared,
    }
    payload["run_summary"] = build_run_summary(
        run_type="summarize_rank_event_filter_set",
        inputs={"baseline_json": str(baseline_json), "filtered_json": str(filtered_json)},
        metrics={
            "common_count": int(len(compared)),
            "improved_count": int(len(improved)),
            "median_delta_npa_core": payload["median_delta_npa_core"],
        },
        artifacts={"json": str(out_json)},
    )
    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize baseline vs event-filtered rank outputs across a candidate set.")
    p.add_argument("--baseline-json", required=True)
    p.add_argument("--filtered-json", required=True)
    p.add_argument("--out-json", default="reports/RANK_EVENT_FILTER_SET_SUMMARY.json")
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
