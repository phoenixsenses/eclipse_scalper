from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.analyze_micro_edge_regimes import enrich_liq_regime_tags, load_debug_rows, summarize
from tools.run_summary import build_run_summary


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _slice_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    tagged = [r for r in rows if str(r.get("liq_regime_tag", "")) == "high_liq_reversal"]
    normal = [r for r in rows if str(r.get("liq_regime_tag", "")) != "high_liq_reversal"]
    tagged_sm = summarize(tagged)
    normal_sm = summarize(normal)
    return {
        "rows_total": int(len(rows)),
        "tagged": {
            "n": int(tagged_sm.get("n", 0) or 0),
            "avg_net": float(tagged_sm.get("avg_net", 0.0) or 0.0),
            "p90_net": float(tagged_sm.get("p90_net", 0.0) or 0.0),
            "break_even_bps_total": float(tagged_sm.get("break_even_bps_total", 0.0) or 0.0),
        },
        "normal": {
            "n": int(normal_sm.get("n", 0) or 0),
            "avg_net": float(normal_sm.get("avg_net", 0.0) or 0.0),
            "p90_net": float(normal_sm.get("p90_net", 0.0) or 0.0),
            "break_even_bps_total": float(normal_sm.get("break_even_bps_total", 0.0) or 0.0),
        },
        "delta_avg_net": float(tagged_sm.get("avg_net", 0.0) or 0.0) - float(normal_sm.get("avg_net", 0.0) or 0.0),
        "delta_p90_net": float(tagged_sm.get("p90_net", 0.0) or 0.0) - float(normal_sm.get("p90_net", 0.0) or 0.0),
    }


def _recommendation(summary: Dict[str, Any]) -> str:
    tagged_n = int(summary["tagged"]["n"])
    normal_n = int(summary["normal"]["n"])
    if tagged_n == 0:
        return "Next action: no tagged signals overlap this debug surface. Treat liquidation regime as an orthogonal annotation."
    if tagged_n < 25:
        return "Next action: tagged overlap is sparse. Expand time window before changing signal logic."
    if _safe_float(summary["delta_avg_net"]) > 0.0 and _safe_float(summary["delta_p90_net"]) > 0.0:
        return "Next action: tagged signals look stronger. Use liquidation regime as a downstream filter candidate."
    if _safe_float(summary["delta_avg_net"]) < 0.0 and _safe_float(summary["delta_p90_net"]) < 0.0:
        return "Next action: tagged signals look weaker. Use liquidation regime as a caution flag, not as a boost."
    if normal_n == 0:
        return "Next action: only tagged signals are present. Compare against a broader baseline run."
    return "Next action: mixed tag effect. Inspect side split, costs, and adverse selection before acting."


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize signal behavior under liquidation regime tags from debug JSONL.")
    p.add_argument("--debug", required=True, help="Path to micro-edge debug JSONL.")
    p.add_argument("--rule", default="high_liq_reversal_regime")
    p.add_argument("--out-json", default="")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    debug_path = Path(str(args.debug))
    if not debug_path.exists():
        print(f"missing debug: {debug_path}")
        return 2
    rows = load_debug_rows(debug_path)
    if not rows:
        print(f"no rows: {debug_path}")
        return 2
    enrich_liq_regime_tags(rows, rule_name=str(args.rule))
    overall = _slice_summary(rows)
    recommendation = _recommendation(overall)

    print(f"debug={debug_path}")
    print("slice rows tagged_n normal_n tagged_avg normal_avg delta_avg tagged_p90 normal_p90 delta_p90")
    print(
        "overall "
        f"{int(overall['rows_total']):>5d} "
        f"{int(overall['tagged']['n']):>8d} "
        f"{int(overall['normal']['n']):>8d} "
        f"{_safe_float(overall['tagged']['avg_net']):+12.6e} "
        f"{_safe_float(overall['normal']['avg_net']):+12.6e} "
        f"{_safe_float(overall['delta_avg_net']):+12.6e} "
        f"{_safe_float(overall['tagged']['p90_net']):+12.6e} "
        f"{_safe_float(overall['normal']['p90_net']):+12.6e} "
        f"{_safe_float(overall['delta_p90_net']):+12.6e}"
    )
    print(recommendation)

    if str(args.out_json).strip():
        out_json = Path(str(args.out_json))
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "debug": str(debug_path),
            "rule": str(args.rule),
            "overall": overall,
            "recommendation": recommendation,
        }
        payload["run_summary"] = build_run_summary(
            run_type="summarize_liq_tag_signal_behavior",
            inputs={"debug": str(debug_path), "rule": str(args.rule)},
            metrics={
                "rows_total": int(overall["rows_total"]),
                "tagged_n": int(overall["tagged"]["n"]),
                "normal_n": int(overall["normal"]["n"]),
                "delta_avg_net": float(overall["delta_avg_net"]),
                "delta_p90_net": float(overall["delta_p90_net"]),
            },
            artifacts={"json": str(out_json)},
        )
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
