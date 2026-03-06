from __future__ import annotations

"""Analyze fee vs adverse-selection cost breakdown for passive pockets.

For each pocket in a rank JSON, fits a linear model:
    NPA = b0 + b1*maker_fee_bps + b2*adv_mult

  b0  = raw edge proxy at zero cost (NPA intercept)
  b1  = fee sensitivity  (NPA change per bps of maker fee)
  b2  = adv sensitivity  (NPA change per unit of adv_mult)

Derives:
  - fee_drag_bps / adverse_drag_bps at reference conditions
  - break-even fee and break-even adv_mult
  - "dominator": which component blocks profitability most

Usage:
  python -m tools.analyze_cost_breakdown \\
    --rank-json reports/RANK_V3_costgrid.json \\
    --out-md    reports/COST_BREAKDOWN_V1.md \\
    --out-json  reports/COST_BREAKDOWN_V1.json
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tools.run_summary import build_run_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────

def _parse_eval_key(key: str) -> Optional[Tuple[float, float]]:
    """Parse 'fee=1.000|adv=1.200' → (1.0, 1.2)."""
    m = re.match(r"fee=([\d.]+)\|adv=([\d.]+)", str(key))
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


# ─────────────────────────────────────────────
# Linear model fit
# ─────────────────────────────────────────────

def _fit_linear(
    data_points: List[Tuple[float, float, float]],
) -> Tuple[Optional[Tuple[float, float, float]], str]:
    """Fit NPA = b0 + b1*fee + b2*adv via least squares.

    Returns ((b0, b1, b2), quality_flag) where quality_flag ∈ {"ok", "degenerate"}.
    """
    if len(data_points) < 3:
        return None, "too_few_points"
    X = np.array([[1.0, fee, adv] for fee, adv, _ in data_points], dtype=float)
    y = np.array([npa for _, _, npa in data_points], dtype=float)
    if np.linalg.matrix_rank(X) < 3:
        # Degenerate (e.g. single adv level — can still fit 2-param model)
        X2 = np.array([[1.0, fee] for fee, adv, _ in data_points], dtype=float)
        if np.linalg.matrix_rank(X2) < 2:
            return None, "degenerate"
        b2, _, _, _ = np.linalg.lstsq(X2, y, rcond=None)
        return (float(b2[0]), float(b2[1]), 0.0), "degenerate_no_adv_variation"
    b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return (float(b[0]), float(b[1]), float(b[2])), "ok"


# ─────────────────────────────────────────────
# Per-pocket analysis
# ─────────────────────────────────────────────

def _analyze_pocket(pocket: Dict[str, Any]) -> Dict[str, Any]:
    evals = pocket.get("evals", {})

    # Collect (fee, adv, npa) data points
    data_points: List[Tuple[float, float, float]] = []
    per_eval: List[Dict[str, Any]] = []

    for key, ev in evals.items():
        parsed = _parse_eval_key(key)
        if parsed is None:
            continue
        fee_bps, adv_mult = parsed
        npa = float(ev.get("median_net_per_attempt", 0.0))
        data_points.append((fee_bps, adv_mult, npa))
        per_eval.append({
            "eval_key": key,
            "maker_fee_bps": fee_bps,
            "passive_adverse_mult": adv_mult,
            "npa": round(npa, 10),
            "npa_bps": round(npa * 10_000.0, 6),
            "fee_drag_bps_roundtrip": round(2.0 * fee_bps, 4),
        })

    if not data_points:
        return {
            "symbol": pocket.get("symbol"),
            "horizon_sec": pocket.get("horizon_sec"),
            "min_imbalance": pocket.get("min_imbalance"),
            "min_trade_intensity": pocket.get("min_trade_intensity"),
            "max_spread": pocket.get("max_spread"),
            "rule": pocket.get("rule"),
            "error": "no_evals",
            "per_eval": [],
        }

    data_points.sort(key=lambda x: (x[0], x[1]))
    per_eval.sort(key=lambda x: (x["maker_fee_bps"], x["passive_adverse_mult"]))

    # Reference: fee≈1.0, adv≈1.0
    fees = sorted(set(f for f, _, _ in data_points))
    advs = sorted(set(a for _, a, _ in data_points))
    ref_fee = min(fees, key=lambda x: abs(x - 1.0))
    ref_adv = min(advs, key=lambda x: abs(x - 1.0))

    ref_pts = [(f, a, n) for f, a, n in data_points
               if abs(f - ref_fee) < 1e-6 and abs(a - ref_adv) < 1e-6]
    current_npa = float(ref_pts[0][2]) if ref_pts else 0.0

    # Fit linear model
    fit, fit_quality = _fit_linear(data_points)
    if fit is None:
        # Simple fallback: intercept only
        b0 = current_npa + (2.0 * ref_fee + ref_adv) / 10_000.0
        b1 = -2.0 / 10_000.0
        b2 = 0.0
        fit_quality = "fallback"
    else:
        b0, b1, b2 = fit

    raw_edge_proxy_bps = b0 * 10_000.0

    # Drag at reference conditions (in NPA units → bps equivalent)
    fee_drag_npa = -b1 * ref_fee   # positive = cost
    adv_drag_npa = -b2 * ref_adv
    total_drag_npa = fee_drag_npa + adv_drag_npa
    fee_drag_bps = fee_drag_npa * 10_000.0
    adv_drag_bps = adv_drag_npa * 10_000.0

    # Dominator
    if total_drag_npa > 1e-12:
        fee_share = fee_drag_npa / total_drag_npa
        dominator = "fee" if fee_share > 0.6 else ("adverse" if fee_share < 0.4 else "balanced")
    else:
        dominator = "unknown"

    # Break-even calculations
    be_fee: Optional[float] = None
    be_adv: Optional[float] = None
    if abs(b1) > 1e-12:
        be_fee = max(0.0, -(b0 + b2 * ref_adv) / b1)
    if abs(b2) > 1e-12:
        be_adv = max(0.0, -(b0 + b1 * ref_fee) / b2)

    gap_fee = (ref_fee - be_fee) if be_fee is not None else None
    gap_adv = (ref_adv - be_adv) if be_adv is not None else None

    return {
        "symbol": pocket.get("symbol"),
        "horizon_sec": pocket.get("horizon_sec"),
        "min_imbalance": pocket.get("min_imbalance"),
        "min_trade_intensity": pocket.get("min_trade_intensity"),
        "max_spread": pocket.get("max_spread"),
        "rule": pocket.get("rule"),
        "n_evals": len(data_points),
        "fit_quality": fit_quality,
        "raw_edge_proxy": round(b0, 10),
        "raw_edge_proxy_bps": round(raw_edge_proxy_bps, 6),
        "fee_sensitivity": round(b1, 12),
        "adv_sensitivity": round(b2, 12),
        "ref_fee_bps": ref_fee,
        "ref_adv_mult": ref_adv,
        "current_npa": round(current_npa, 10),
        "current_npa_bps": round(current_npa * 10_000.0, 6),
        "fee_drag_bps_implied": round(fee_drag_bps, 6),
        "adverse_drag_bps_implied": round(adv_drag_bps, 6),
        "dominator": dominator,
        "break_even_fee_bps": round(be_fee, 6) if be_fee is not None else None,
        "break_even_adv_mult": round(be_adv, 6) if be_adv is not None else None,
        "gap_to_breakeven_fee_bps": round(gap_fee, 6) if gap_fee is not None else None,
        "gap_to_breakeven_adv_mult": round(gap_adv, 6) if gap_adv is not None else None,
        "per_eval": per_eval,
    }


# ─────────────────────────────────────────────
# Ranking / sorting
# ─────────────────────────────────────────────

def _sort_key(p: Dict[str, Any]) -> Tuple[float, float]:
    npa = p.get("current_npa", 0.0) or 0.0
    if npa > 0:
        return (-1e9, -npa)   # profitable pockets first
    gap = p.get("gap_to_breakeven_fee_bps")
    if gap is None:
        return (1e9, 0.0)
    return (abs(gap), gap)


# ─────────────────────────────────────────────
# Markdown rendering
# ─────────────────────────────────────────────

def _render_md(pockets: List[Dict[str, Any]], source_json: str) -> str:
    lines = [
        "# COST BREAKDOWN ANALYSIS",
        "",
        f"source: {source_json}",
        f"generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary Table",
        "",
        "| rank | symbol | h | min_imb | min_int | max_spr | dominator "
        "| raw_edge_bps | fee_drag_bps | adv_drag_bps | cur_npa_bps "
        "| be_fee_bps | be_adv_mult | gap_fee | gap_adv |",
        "|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, p in enumerate(pockets, 1):
        def _fmt(v, fmt=".3f"):
            return f"{v:{fmt}}" if v is not None else "n/a"
        lines.append(
            f"| {i} | {p.get('symbol')} | {p.get('horizon_sec')} "
            f"| {p.get('min_imbalance'):.2f} | {p.get('min_trade_intensity'):.0f} "
            f"| {p.get('max_spread'):.6f} | {p.get('dominator', '?')} "
            f"| {_fmt(p.get('raw_edge_proxy_bps'), '+.4f')} "
            f"| {_fmt(p.get('fee_drag_bps_implied'), '.4f')} "
            f"| {_fmt(p.get('adverse_drag_bps_implied'), '.4f')} "
            f"| {_fmt(p.get('current_npa_bps'), '+.4f')} "
            f"| {_fmt(p.get('break_even_fee_bps'))} "
            f"| {_fmt(p.get('break_even_adv_mult'))} "
            f"| {_fmt(p.get('gap_to_breakeven_fee_bps'), '+.4f')} "
            f"| {_fmt(p.get('gap_to_breakeven_adv_mult'), '+.4f')} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- **fee**: round-trip fee drag (2 × maker_fee_bps) accounts for >60% of cost.",
        "  → Negotiate maker rebate tier or reduce holding frequency.",
        "- **adverse**: adverse selection accounts for >60% of cost.",
        "  → Apply `--mitigation-profile anti_adverse_v1` to tighten signal filters.",
        "- **balanced**: both components contribute. Address the larger gap first.",
        "- **gap_fee < 0** (negative): pocket is already profitable on fee dimension.",
        "- **gap_adv < 0** (negative): pocket is already profitable on adverse dimension.",
        "",
        "## Closest to Break-Even — Fee Dimension",
        "",
        "*(sorted by |gap_to_breakeven_fee_bps|, smallest first)*",
        "",
    ]
    by_fee = sorted(
        [p for p in pockets if p.get("gap_to_breakeven_fee_bps") is not None],
        key=lambda p: abs(p["gap_to_breakeven_fee_bps"]),
    )
    for p in by_fee[:5]:
        g = p["gap_to_breakeven_fee_bps"]
        be = p["break_even_fee_bps"]
        lines.append(
            f"- **{p.get('symbol')}** h={p.get('horizon_sec')} "
            f"imb>={p.get('min_imbalance'):.2f} int>={p.get('min_trade_intensity'):.0f}: "
            f"gap={g:+.4f} bps (break-even at fee≤{be:.4f} bps, "
            f"adv_mult={p.get('ref_adv_mult'):.1f}x)"
        )

    lines += ["", "## Closest to Break-Even — Adverse Dimension", "", ]
    by_adv = sorted(
        [p for p in pockets if p.get("gap_to_breakeven_adv_mult") is not None],
        key=lambda p: abs(p["gap_to_breakeven_adv_mult"]),
    )
    for p in by_adv[:5]:
        g = p["gap_to_breakeven_adv_mult"]
        be = p["break_even_adv_mult"]
        lines.append(
            f"- **{p.get('symbol')}** h={p.get('horizon_sec')} "
            f"imb>={p.get('min_imbalance'):.2f} int>={p.get('min_trade_intensity'):.0f}: "
            f"gap={g:+.4f} mult (break-even at adv_mult≤{be:.4f}, "
            f"fee={p.get('ref_fee_bps'):.2f} bps)"
        )

    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze fee vs adverse cost breakdown per pocket from a rank JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rank-json", required=True,
                   help="Rank JSON produced by rank_passive_pockets_forward (e.g. reports/RANK_V3_costgrid.json)")
    p.add_argument("--out-md",   default="reports/COST_BREAKDOWN_V1.md",   help="Output markdown report")
    p.add_argument("--out-json", default="reports/COST_BREAKDOWN_V1.json", help="Output JSON report")
    return p.parse_args()


def main() -> int:
    args = _args()
    rank_path = Path(args.rank_json)
    if not rank_path.exists():
        log.error("rank-json not found: %s", rank_path)
        return 2

    raw = json.loads(rank_path.read_text(encoding="utf-8"))
    ranking = raw.get("ranking", [])
    if not ranking:
        log.error("No pockets in ranking JSON (empty 'ranking' list)")
        return 2

    log.info("Analyzing %d pockets from %s", len(ranking), rank_path)
    analyzed = [_analyze_pocket(p) for p in ranking]
    analyzed.sort(key=_sort_key)

    def _top5_gap(field_gap: str, field_be: str) -> List[Dict[str, Any]]:
        sortable = [p for p in analyzed if p.get(field_gap) is not None]
        sortable.sort(key=lambda p: abs(p[field_gap]))
        return [
            {
                "symbol": p.get("symbol"),
                "horizon_sec": p.get("horizon_sec"),
                field_gap: p.get(field_gap),
                field_be: p.get(field_be),
            }
            for p in sortable[:5]
        ]

    out = {
        "tool": "analyze_cost_breakdown",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_json": str(rank_path),
        "n_pockets": len(analyzed),
        "pockets": analyzed,
        "closest_to_breakeven_fee": _top5_gap("gap_to_breakeven_fee_bps", "break_even_fee_bps"),
        "closest_to_breakeven_adv": _top5_gap("gap_to_breakeven_adv_mult", "break_even_adv_mult"),
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out["run_summary"] = build_run_summary(
        run_type="analyze_cost_breakdown",
        inputs={"source_json": str(rank_path)},
        metrics={"n_pockets": len(analyzed)},
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("wrote %s", out_json)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_md(analyzed, str(rank_path)), encoding="utf-8")
    log.info("wrote %s", out_md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
