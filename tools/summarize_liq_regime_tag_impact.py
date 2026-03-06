from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.run_summary import build_run_summary


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _fmt(v: Any) -> str:
    return f"{_safe_float(v):+.6e}"


def _impact_summary(block: Dict[str, Any]) -> Dict[str, Any]:
    tagged = block.get("tagged") if isinstance(block.get("tagged"), dict) else {}
    normal = block.get("normal") if isinstance(block.get("normal"), dict) else {}
    tagged_avg = _safe_float(tagged.get("avg_net"))
    normal_avg = _safe_float(normal.get("avg_net"))
    tagged_p90 = _safe_float(tagged.get("p90_net"))
    normal_p90 = _safe_float(normal.get("p90_net"))
    delta_avg = tagged_avg - normal_avg
    delta_p90 = tagged_p90 - normal_p90
    tagged_n = int(tagged.get("n", 0) or 0)
    normal_n = int(normal.get("n", 0) or 0)
    return {
        "available": bool(block.get("available", False)),
        "tagged": {
            "n": tagged_n,
            "avg_net": tagged_avg,
            "p90_net": tagged_p90,
        },
        "normal": {
            "n": normal_n,
            "avg_net": normal_avg,
            "p90_net": normal_p90,
        },
        "delta_avg_net": delta_avg,
        "delta_p90_net": delta_p90,
        "sample_warning": tagged_n == 0 or normal_n == 0,
    }


def _recommendation(discovery: Dict[str, Any], validation: Dict[str, Any]) -> str:
    if not discovery.get("available") or not validation.get("available"):
        return "Next action: missing tagged-vs-normal data. Re-run forward validation with liquidation regime tags."
    if discovery.get("sample_warning") or validation.get("sample_warning"):
        return "Next action: tagged sample is too small. Expand the debug window before judging usefulness."
    disc_delta = _safe_float(discovery.get("delta_avg_net"))
    valid_delta = _safe_float(validation.get("delta_avg_net"))
    if disc_delta > 0.0 and valid_delta > 0.0:
        return "Next action: tagged regime outperforms in both slices. Treat it as a useful downstream filter."
    if disc_delta > 0.0 and valid_delta <= 0.0:
        return "Next action: discovery edge does not survive validation. Keep as annotation, not as a trading gate."
    if disc_delta <= 0.0 and valid_delta <= 0.0:
        return "Next action: tagged regime underperforms in both slices. Archive it as a descriptive tag only."
    return "Next action: mixed result. Inspect horizon, costs, and sample size before changing model logic."


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize tagged-vs-normal liquidation regime impact from forward validation JSON.")
    p.add_argument("--in", dest="in_path", required=True, help="Path to validate_micro_edge_forward JSON.")
    p.add_argument("--out-json", default="", help="Optional summary JSON path.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    in_path = Path(str(args.in_path))
    if not in_path.exists():
        print(f"missing input: {in_path}")
        return 2
    try:
        payload = json.loads(in_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"invalid json: {exc}")
        return 2

    impact = payload.get("liquidation_regime_tag_impact")
    if not isinstance(impact, dict):
        print("invalid payload: missing liquidation_regime_tag_impact")
        return 2

    discovery = _impact_summary(impact.get("discovery") if isinstance(impact.get("discovery"), dict) else {})
    validation = _impact_summary(impact.get("validation") if isinstance(impact.get("validation"), dict) else {})
    recommendation = _recommendation(discovery, validation)

    print(f"source={in_path}")
    print("slice tagged_n normal_n tagged_avg normal_avg delta_avg tagged_p90 normal_p90 delta_p90")
    for name, block in (("discovery", discovery), ("validation", validation)):
        print(
            f"{name:<10} "
            f"{int(block['tagged']['n']):>8d} "
            f"{int(block['normal']['n']):>8d} "
            f"{_fmt(block['tagged']['avg_net']):>12} "
            f"{_fmt(block['normal']['avg_net']):>12} "
            f"{_fmt(block['delta_avg_net']):>12} "
            f"{_fmt(block['tagged']['p90_net']):>12} "
            f"{_fmt(block['normal']['p90_net']):>12} "
            f"{_fmt(block['delta_p90_net']):>12}"
        )
    print(recommendation)

    if str(args.out_json).strip():
        out_json = Path(str(args.out_json))
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_payload = {
            "source": str(in_path),
            "discovery": discovery,
            "validation": validation,
            "recommendation": recommendation,
        }
        out_payload["run_summary"] = build_run_summary(
            run_type="summarize_liq_regime_tag_impact",
            inputs={"source": str(in_path)},
            metrics={
                "discovery_delta_avg_net": float(discovery["delta_avg_net"]),
                "validation_delta_avg_net": float(validation["delta_avg_net"]),
                "discovery_tagged_n": int(discovery["tagged"]["n"]),
                "validation_tagged_n": int(validation["tagged"]["n"]),
            },
            artifacts={"json": str(out_json)},
        )
        out_json.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
