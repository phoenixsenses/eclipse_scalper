from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from tools.execution_diagnostics import compute_execution_diagnostics, _load_rows
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


def _compute_state(diag: Dict[str, Any]) -> tuple[str, list[str]]:
    rows = _safe_int(diag.get("rows"))
    p95 = _safe_float(diag.get("latency_fill_delay_sec_p95"))
    p50 = _safe_float(diag.get("latency_fill_delay_sec_p50"))
    corr = _safe_float(diag.get("latency_impact_vs_net_corr"))
    fill_rate = _safe_float(diag.get("fill_rate"))
    reasons: list[str] = []
    if rows <= 0:
        return ("quiet", ["no_trade_rows"])
    if p95 >= 10.0:
        reasons.append("very_high_p95_fill_delay")
    if p50 >= 4.0:
        reasons.append("high_p50_fill_delay")
    if corr <= -0.20:
        reasons.append("latency_negatively_correlated_with_net")
    if fill_rate <= 0.20:
        reasons.append("low_fill_rate_under_latency")
    if reasons:
        return ("severe", reasons)
    if p95 >= 5.0:
        reasons.append("elevated_p95_fill_delay")
    if p50 >= 2.0:
        reasons.append("elevated_p50_fill_delay")
    if corr <= -0.05:
        reasons.append("mild_negative_latency_net_correlation")
    if fill_rate <= 0.40:
        reasons.append("soft_fill_rate_pressure")
    if reasons:
        return ("elevated", reasons)
    return ("quiet", ["no_runtime_threshold_hit"])


def _recommended_action(level: str) -> str:
    if level == "severe":
        return "escalate_monitoring"
    if level == "elevated":
        return "show_caution"
    return "monitor_only"


def build_state_payload(*, source: str, diag: Dict[str, Any], out_json: str, out_md: str) -> Dict[str, Any]:
    level, reasons = _compute_state(diag)
    recommended_action = _recommended_action(level)
    payload = {
        "source": str(source),
        "state": {
            "level": level,
            "reasons": reasons,
        },
        "dashboard_summary": (
            f"latency stress {level}, p95={_safe_float(diag.get('latency_fill_delay_sec_p95')):.2f}s, "
            f"p50={_safe_float(diag.get('latency_fill_delay_sec_p50')):.2f}s, fill_rate={_safe_float(diag.get('fill_rate')):.2%}."
        ),
        "notification_text": (
            f"[latency-stress] level={level} p95={_safe_float(diag.get('latency_fill_delay_sec_p95')):.2f}s "
            f"p50={_safe_float(diag.get('latency_fill_delay_sec_p50')):.2f}s "
            f"fill_rate={_safe_float(diag.get('fill_rate')):.4f} "
            f"corr={_safe_float(diag.get('latency_impact_vs_net_corr')):+.4f} "
            f"action={recommended_action}"
        ),
        "recommended_action": recommended_action,
        "card": {
            "headline": f"Latency stress {level}",
            "operator_note": {
                "quiet": "Latency is within current monitoring thresholds.",
                "elevated": "Show caution for execution latency quality.",
                "severe": "Escalate latency monitoring and inspect runtime execution path.",
            }[level],
            "rows": _safe_int(diag.get("rows")),
            "fill_rate": _safe_float(diag.get("fill_rate")),
            "latency_fill_delay_sec_p50": _safe_float(diag.get("latency_fill_delay_sec_p50")),
            "latency_fill_delay_sec_p95": _safe_float(diag.get("latency_fill_delay_sec_p95")),
            "latency_impact_vs_net_corr": _safe_float(diag.get("latency_impact_vs_net_corr")),
        },
        "summary_snapshot": {
            "rows": _safe_int(diag.get("rows")),
            "fill_rate": _safe_float(diag.get("fill_rate")),
            "queue_competition_score": _safe_float(diag.get("queue_competition_score")),
            "toxicity_score": _safe_float(diag.get("toxicity_score")),
            "adverse_selection_bps_mean": _safe_float(diag.get("adverse_selection_bps_mean")),
            "latency_fill_delay_sec_p50": _safe_float(diag.get("latency_fill_delay_sec_p50")),
            "latency_fill_delay_sec_p95": _safe_float(diag.get("latency_fill_delay_sec_p95")),
            "latency_impact_vs_net_corr": _safe_float(diag.get("latency_impact_vs_net_corr")),
        },
    }
    payload["run_summary"] = build_run_summary(
        run_type="latency_stress_state",
        inputs={"source": str(source)},
        metrics={
            "rows": _safe_int(diag.get("rows")),
            "state_level": level,
            "fill_rate": _safe_float(diag.get("fill_rate")),
            "latency_fill_delay_sec_p95": _safe_float(diag.get("latency_fill_delay_sec_p95")),
        },
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    return payload


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a runtime-ready latency-stress state payload from execution diagnostics.")
    p.add_argument("--in", dest="in_path", default="data/live/papertrades_live.parquet")
    p.add_argument("--out-json", default="reports/LATENCY_STRESS_STATE.json")
    p.add_argument("--out-md", default="reports/LATENCY_STRESS_STATE.md")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    diag = compute_execution_diagnostics(_load_rows(Path(str(args.in_path))))
    payload = build_state_payload(
        source=str(args.in_path),
        diag=diag,
        out_json=str(args.out_json),
        out_md=str(args.out_md),
    )
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# LATENCY STRESS STATE",
        "",
        f"state={payload['state']['level']}",
        f"dashboard_summary={payload['dashboard_summary']}",
        f"notification_text={payload['notification_text']}",
        f"recommended_action={payload['recommended_action']}",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
