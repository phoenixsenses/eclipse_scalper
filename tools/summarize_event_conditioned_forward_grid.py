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


def _variant_score(row: Dict[str, Any]) -> float:
    delta = _safe_float(row.get("validation_delta_avg_net"))
    kept = _safe_float(row.get("validation_kept_ratio"))
    return (delta * 10000.0) * max(0.0, kept)


def build_payload(*, grid_json: str, out_json: str) -> Dict[str, Any]:
    payload = _load_json(grid_json)
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    variants: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        enriched = dict(row)
        enriched["tradeoff_score"] = _variant_score(row)
        variants.append(enriched)
    variants.sort(key=lambda row: (_safe_float(row.get("tradeoff_score")), _safe_float(row.get("validation_delta_avg_net"))), reverse=True)
    best_tradeoff = variants[0] if variants else {}
    best_quality = max(variants, key=lambda row: _safe_float(row.get("validation_delta_avg_net")), default={})

    recommendation = "keep_baseline"
    if best_tradeoff and str(best_tradeoff.get("variant", "")) != "baseline_like":
        recommendation = "test_tradeoff_variant_in_rank_pipeline"

    out = {
        "source_grid_json": str(grid_json),
        "best_tradeoff_variant": {
            "variant": str(best_tradeoff.get("variant", "")),
            "tradeoff_score": _safe_float(best_tradeoff.get("tradeoff_score")),
            "validation_delta_avg_net": _safe_float(best_tradeoff.get("validation_delta_avg_net")),
            "validation_kept_ratio": _safe_float(best_tradeoff.get("validation_kept_ratio")),
        },
        "best_quality_variant": {
            "variant": str(best_quality.get("variant", "")),
            "validation_delta_avg_net": _safe_float(best_quality.get("validation_delta_avg_net")),
            "validation_kept_ratio": _safe_float(best_quality.get("validation_kept_ratio")),
        },
        "recommendation": str(recommendation),
        "rows": variants,
    }
    out["run_summary"] = build_run_summary(
        run_type="summarize_event_conditioned_forward_grid",
        inputs={"grid_json": str(grid_json)},
        metrics={
            "best_tradeoff_score": _safe_float(out["best_tradeoff_variant"].get("tradeoff_score")),
            "variant_count": int(len(variants)),
        },
        artifacts={"json": str(out_json)},
    )
    return out


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize event-conditioned forward grid into a tradeoff recommendation.")
    p.add_argument("--grid-json", required=True)
    p.add_argument("--out-json", default="reports/EVENT_CONDITIONED_FORWARD_GRID_SUMMARY.json")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    payload = build_payload(grid_json=str(args.grid_json), out_json=str(args.out_json))
    out_path = Path(str(args.out_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
