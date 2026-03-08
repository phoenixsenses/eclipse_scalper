from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from tools.run_summary import build_run_summary
from tools.toxicity_report import build_toxicity_report, _load as _load_rows


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


def _choose_top_side(sides: Dict[str, Any]) -> str:
    best_side = ""
    best_score = -1.0
    for side, stats in sorted(sides.items()):
        score = _safe_float(stats.get("toxicity_score")) * 100.0 + _safe_float(stats.get("adverse_bps_mean"))
        if score > best_score:
            best_score = score
            best_side = str(side)
    return best_side


def _compute_state(rows: int, top_stats: Dict[str, Any]) -> tuple[str, list[str]]:
    if rows <= 0:
        return ("quiet", ["no_trade_rows"])
    tox = _safe_float(top_stats.get("toxicity_score"))
    adv = _safe_float(top_stats.get("adverse_bps_mean"))
    pnl = _safe_float(top_stats.get("pnl_bps_mean"))
    reasons: list[str] = []
    if tox >= 1.5:
        reasons.append("extreme_toxicity_score")
    if adv >= 2.0:
        reasons.append("high_adverse_selection")
    if pnl < 0.0 and tox >= 1.0:
        reasons.append("negative_pnl_under_toxicity")
    if reasons:
        return ("severe", reasons)
    if tox >= 0.8:
        reasons.append("elevated_toxicity_score")
    if adv >= 1.0:
        reasons.append("moderate_adverse_selection")
    if pnl < 0.0:
        reasons.append("negative_pnl_bias")
    if reasons:
        return ("elevated", reasons)
    return ("quiet", ["no_runtime_threshold_hit"])


def _recommended_action(level: str) -> str:
    if level == "severe":
        return "reduce_passive_aggression"
    if level == "elevated":
        return "show_caution"
    return "monitor_only"


def build_state_payload(*, source: str, report_payload: Dict[str, Any], out_json: str, out_md: str) -> Dict[str, Any]:
    sides = dict(report_payload.get("sides") or {})
    top_side = _choose_top_side(sides)
    top_stats = dict(sides.get(top_side) or {})
    rows = _safe_int(report_payload.get("rows"))
    level, reasons = _compute_state(rows, top_stats)
    recommended_action = _recommended_action(level)
    payload = {
        "source": str(source),
        "rows": rows,
        "top_side": top_side,
        "state": {
            "level": level,
            "reasons": reasons,
        },
        "dashboard_summary": (
            f"fill toxicity {level}, top_side {top_side or 'none'}, "
            f"toxicity={_safe_float(top_stats.get('toxicity_score')):.3f}, adverse={_safe_float(top_stats.get('adverse_bps_mean')):.3f}bps."
        ),
        "notification_text": (
            f"[fill-toxicity] level={level} top_side={top_side or 'none'} "
            f"toxicity={_safe_float(top_stats.get('toxicity_score')):.4f} "
            f"adverse_bps={_safe_float(top_stats.get('adverse_bps_mean')):.4f} "
            f"pnl_bps={_safe_float(top_stats.get('pnl_bps_mean')):+.4f} action={recommended_action}"
        ),
        "recommended_action": recommended_action,
        "card": {
            "headline": f"Fill toxicity {level}",
            "operator_note": {
                "quiet": "No immediate fill-toxicity concern.",
                "elevated": "Show caution for passive execution quality.",
                "severe": "Escalate monitoring and consider reducing passive aggression.",
            }[level],
            "top_side": top_side,
            "rows": rows,
            "toxicity_score": _safe_float(top_stats.get("toxicity_score")),
            "adverse_bps_mean": _safe_float(top_stats.get("adverse_bps_mean")),
            "pnl_bps_mean": _safe_float(top_stats.get("pnl_bps_mean")),
        },
        "summary_snapshot": {
            "rows": rows,
            "sides": sides,
        },
    }
    payload["run_summary"] = build_run_summary(
        run_type="fill_toxicity_state",
        inputs={"source": str(source)},
        metrics={
            "rows": rows,
            "side_count": len(sides),
            "state_level": level,
            "top_side": top_side,
            "top_toxicity_score": _safe_float(top_stats.get("toxicity_score")),
        },
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    return payload


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a runtime-ready fill-toxicity state payload from trade logs.")
    p.add_argument("--in", dest="in_path", default="data/live/papertrades_live.parquet")
    p.add_argument("--out-json", default="reports/FILL_TOXICITY_STATE.json")
    p.add_argument("--out-md", default="reports/FILL_TOXICITY_STATE.md")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    report_payload = build_toxicity_report(_load_rows(Path(str(args.in_path))))
    payload = build_state_payload(
        source=str(args.in_path),
        report_payload=report_payload,
        out_json=str(args.out_json),
        out_md=str(args.out_md),
    )
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# FILL TOXICITY STATE",
        "",
        f"state={payload['state']['level']}",
        f"top_side={payload['top_side']}",
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
