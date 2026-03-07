from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.analyze_micro_edge_regimes import enrich_liq_regime_tags, group_key, group_stats, load_debug_rows, summarize
from tools.run_summary import build_run_summary


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward validation for micro-edge regime subsets.")
    p.add_argument("--debug", required=True)
    p.add_argument(
        "--group-by",
        default="regime_spread_bin,regime_intensity_bin,regime_vol_bin,regime_imb_bin",
    )
    p.add_argument("--discover-frac", type=float, default=0.60)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--min-n", type=int, default=None, help="Optional legacy alias that sets both min-n-discovery and min-n-validation.")
    p.add_argument("--min-n-discovery", type=int, default=30)
    p.add_argument("--min-n-validation", type=int, default=30)
    p.add_argument("--min-select-frac", type=float, default=0.01)
    p.add_argument("--allow-small-selection", action="store_true")
    p.add_argument("--no-relax", action="store_true")
    p.add_argument("--relax-floor", type=int, default=20)
    p.add_argument("--collapse-avg-eps", type=float, default=0.0002)
    p.add_argument("--collapse-p90-eps", type=float, default=0.0002)
    p.add_argument("--collapse-select-ratio", type=float, default=0.5)
    p.add_argument("--collapse-n-ratio", type=float, default=0.5)
    p.add_argument("--collapse-mode", choices=["strict", "balanced"], default="balanced")
    p.add_argument("--out-json", default=None)
    return p.parse_args(argv)


def _fmt_num(x: Any) -> str:
    try:
        return f"{float(x):+.6f}"
    except Exception:
        return "-"


def _selection(rows: List[Dict[str, Any]], keys: set[str], fields: List[str]) -> List[Dict[str, Any]]:
    return [r for r in rows if group_key(r, fields) in keys]


def _quantile(vals: List[float], q: float) -> Optional[float]:
    if not vals:
        return None
    q = max(0.0, min(1.0, float(q)))
    ordered = sorted(float(v) for v in vals)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    if hi == lo:
        return ordered[lo]
    w = pos - lo
    return ordered[lo] * (1.0 - w) + ordered[hi] * w


def _liq_signal_score(row: Dict[str, Any]) -> Optional[float]:
    try:
        if row.get("v2_liq_reversal_signal") is not None:
            return abs(float(row.get("v2_liq_reversal_signal")))
        li = row.get("liq_imbalance")
        lr = row.get("liq_rate_per_sec")
        if li is None or lr is None:
            return None
        return abs(float(li)) * max(0.0, float(lr))
    except Exception:
        return None


def _liquidation_impact(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored: List[tuple[float, Dict[str, Any]]] = []
    for row in rows:
        score = _liq_signal_score(row)
        if score is None:
            continue
        scored.append((float(score), row))
    if not scored:
        return {"available": False, "count": 0}
    vals = [s for s, _ in scored]
    threshold = _quantile(vals, 0.75)
    if threshold is None:
        return {"available": False, "count": len(scored)}
    active = [row for score, row in scored if float(score) >= float(threshold) and float(score) > 0.0]
    inactive = [row for score, row in scored if not (float(score) >= float(threshold) and float(score) > 0.0)]
    active_sm = summarize(active)
    inactive_sm = summarize(inactive)
    return {
        "available": True,
        "count": len(scored),
        "threshold_q75": float(threshold),
        "active": {
            "n": int(active_sm.get("n", 0) or 0),
            "avg_net": float(active_sm.get("avg_net", 0.0) or 0.0),
            "p90_net": float(active_sm.get("p90_net", 0.0) or 0.0),
        },
        "inactive": {
            "n": int(inactive_sm.get("n", 0) or 0),
            "avg_net": float(inactive_sm.get("avg_net", 0.0) or 0.0),
            "p90_net": float(inactive_sm.get("p90_net", 0.0) or 0.0),
        },
    }


def _liq_regime_tag_impact(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    enrich_liq_regime_tags(rows, rule_name="high_liq_reversal_regime")
    tagged = [r for r in rows if str(r.get("liq_regime_tag", "")) == "high_liq_reversal"]
    normal = [r for r in rows if str(r.get("liq_regime_tag", "")) != "high_liq_reversal"]
    tagged_sm = summarize(tagged)
    normal_sm = summarize(normal)
    return {
        "available": bool(rows),
        "tagged": {
            "n": int(tagged_sm.get("n", 0) or 0),
            "avg_net": float(tagged_sm.get("avg_net", 0.0) or 0.0),
            "p90_net": float(tagged_sm.get("p90_net", 0.0) or 0.0),
        },
        "normal": {
            "n": int(normal_sm.get("n", 0) or 0),
            "avg_net": float(normal_sm.get("avg_net", 0.0) or 0.0),
            "p90_net": float(normal_sm.get("p90_net", 0.0) or 0.0),
        },
    }


def _event_lane_context_impact(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"available": False, "rows_total": 0, "lane_count": 0, "by_lane": {}}

    spreads = [abs(float(r.get("spread") or 0.0)) for r in rows if r.get("spread") is not None]
    intensities = [max(0.0, float(r.get("trade_intensity") or 0.0)) for r in rows if r.get("trade_intensity") is not None]
    abs_rets = [abs(float(r.get("ret_1") or 0.0)) for r in rows if r.get("ret_1") is not None]
    abs_imbalances = [abs(float(r.get("imbalance") or 0.0)) for r in rows if r.get("imbalance") is not None]

    spread_q90 = float(_quantile(spreads, 0.90) or 0.0)
    spread_q75 = float(_quantile(spreads, 0.75) or 0.0)
    spread_q50 = float(_quantile(spreads, 0.50) or 0.0)
    intensity_q75 = float(_quantile(intensities, 0.75) or 0.0)
    intensity_q60 = float(_quantile(intensities, 0.60) or 0.0)
    intensity_q50 = float(_quantile(intensities, 0.50) or 0.0)
    intensity_q40 = float(_quantile(intensities, 0.40) or 0.0)
    intensity_q25 = float(_quantile(intensities, 0.25) or 0.0)
    intensity_q10 = float(_quantile(intensities, 0.10) or 0.0)
    abs_ret_q90 = float(_quantile(abs_rets, 0.90) or 0.0)
    abs_ret_q75 = float(_quantile(abs_rets, 0.75) or 0.0)
    abs_ret_q50 = float(_quantile(abs_rets, 0.50) or 0.0)
    abs_imb_q90 = float(_quantile(abs_imbalances, 0.90) or 0.0)
    abs_imb_q75 = float(_quantile(abs_imbalances, 0.75) or 0.0)

    def _tagged_rows(pred) -> List[Dict[str, Any]]:
        return [r for r in rows if pred(r)]

    def _lane_summary(tagged: List[Dict[str, Any]]) -> Dict[str, Any]:
        normal = [r for r in rows if r not in tagged]
        tagged_sm = summarize(tagged)
        normal_sm = summarize(normal)
        return {
            "tagged_n": int(tagged_sm.get("n", 0) or 0),
            "tagged_rate": (float(tagged_sm.get("n", 0) or 0) / float(len(rows))) if rows else 0.0,
            "tagged_avg_net": float(tagged_sm.get("avg_net", 0.0) or 0.0),
            "tagged_p90_net": float(tagged_sm.get("p90_net", 0.0) or 0.0),
            "normal_n": int(normal_sm.get("n", 0) or 0),
            "normal_avg_net": float(normal_sm.get("avg_net", 0.0) or 0.0),
            "normal_p90_net": float(normal_sm.get("p90_net", 0.0) or 0.0),
            "delta_avg_net": float(tagged_sm.get("avg_net", 0.0) or 0.0) - float(normal_sm.get("avg_net", 0.0) or 0.0),
            "delta_p90_net": float(tagged_sm.get("p90_net", 0.0) or 0.0) - float(normal_sm.get("p90_net", 0.0) or 0.0),
        }

    def _spread_stress(r: Dict[str, Any]) -> bool:
        return float(r.get("spread") or 0.0) >= spread_q75 and float(r.get("trade_intensity") or 0.0) <= max(1.0, intensity_q25 * 1.5)

    def _return_shock(r: Dict[str, Any]) -> bool:
        return abs(float(r.get("ret_1") or 0.0)) >= abs_ret_q75 and float(r.get("spread") or 0.0) <= max(spread_q75, 0.0)

    def _volatility_burst(r: Dict[str, Any]) -> bool:
        return abs(float(r.get("ret_1") or 0.0)) >= abs_ret_q75 and float(r.get("trade_intensity") or 0.0) >= intensity_q40

    def _volume_vacuum(r: Dict[str, Any]) -> bool:
        return float(r.get("trade_intensity") or 0.0) <= intensity_q25 and float(r.get("spread") or 0.0) >= spread_q50 and abs(float(r.get("ret_1") or 0.0)) <= abs_ret_q50

    def _book_proxy_pressure(r: Dict[str, Any]) -> bool:
        return abs(float(r.get("imbalance") or 0.0)) >= abs_imb_q75 and float(r.get("trade_intensity") or 0.0) >= intensity_q50 and float(r.get("spread") or 0.0) >= spread_q50

    lane_defs = {
        "spread_stress": _spread_stress,
        "return_shock": _return_shock,
        "volatility_burst": _volatility_burst,
        "volume_vacuum": _volume_vacuum,
        "book_proxy_pressure": _book_proxy_pressure,
    }

    by_lane: Dict[str, Any] = {}
    for lane, pred in lane_defs.items():
        tagged = _tagged_rows(pred)
        by_lane[lane] = _lane_summary(tagged)

    ranked = sorted(
        (
            {
                "lane": lane,
                "tagged_n": int(section.get("tagged_n", 0) or 0),
                "delta_avg_net": float(section.get("delta_avg_net", 0.0) or 0.0),
                "delta_p90_net": float(section.get("delta_p90_net", 0.0) or 0.0),
            }
            for lane, section in by_lane.items()
        ),
        key=lambda item: (item["tagged_n"] > 0, item["delta_avg_net"], item["delta_p90_net"]),
        reverse=True,
    )
    return {
        "available": True,
        "rows_total": int(len(rows)),
        "lane_count": int(len(by_lane)),
        "top_lane_by_delta_avg_net": ranked[0]["lane"] if ranked else "",
        "by_lane": by_lane,
    }


def _relax_threshold(groups: List[Dict[str, Any]], start_min_n: int, floor: int, no_relax: bool) -> tuple[int, bool]:
    cur = int(start_min_n)
    if any(int(g.get("n", 0)) >= cur for g in groups):
        return cur, False
    if no_relax:
        return cur, False
    f = int(max(1, floor))
    while cur > f:
        cur -= 1
        if any(int(g.get("n", 0)) >= cur for g in groups):
            return cur, True
    return cur, True


def evaluate_collapse_flags(
    *,
    disc_sm: Dict[str, Any],
    valid_sm: Dict[str, Any],
    disc_frac: float,
    valid_frac: float,
    avg_eps: float,
    p90_eps: float,
    select_ratio: float,
    n_ratio: float,
    mode: str,
) -> tuple[bool, Dict[str, bool], Dict[str, float]]:
    disc_p90 = float(disc_sm.get("p90_net", 0.0) or 0.0)
    valid_p90 = float(valid_sm.get("p90_net", 0.0) or 0.0)
    disc_avg = float(disc_sm.get("avg_net", 0.0) or 0.0)
    valid_avg = float(valid_sm.get("avg_net", 0.0) or 0.0)
    disc_n = float(disc_sm.get("n", 0) or 0)
    valid_n = float(valid_sm.get("n", 0) or 0)
    select_thr = float(disc_frac) * float(select_ratio)
    n_thr = float(disc_n) * float(n_ratio)

    flags = {
        "p90_sign_flip": (disc_p90 > 0.0 and valid_p90 < 0.0),
        "avg_net_drop": (valid_avg <= (disc_avg - float(avg_eps))),
        "p90_drop": (valid_p90 <= (disc_p90 - float(p90_eps))),
        "selection_frac_drop": (valid_frac < select_thr),
        "n_drop": (valid_n < n_thr),
    }
    m = str(mode).lower()
    if m == "strict":
        collapse = any(flags.values())
    else:
        collapse = bool(flags["p90_sign_flip"]) or (bool(flags["avg_net_drop"]) and bool(flags["p90_drop"]))
    values = {
        "disc_avg_net": disc_avg,
        "valid_avg_net": valid_avg,
        "disc_p90_net": disc_p90,
        "valid_p90_net": valid_p90,
        "avg_eps": float(avg_eps),
        "p90_eps": float(p90_eps),
        "disc_select_frac": float(disc_frac),
        "valid_select_frac": float(valid_frac),
        "select_ratio": float(select_ratio),
        "select_threshold": float(select_thr),
        "disc_n": float(disc_n),
        "valid_n": float(valid_n),
        "n_ratio": float(n_ratio),
        "n_threshold": float(n_thr),
    }
    return collapse, flags, values


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    path = Path(str(args.debug))
    if not path.exists():
        print(f"error: debug file not found: {path}")
        return 2
    fields = [x.strip() for x in str(args.group_by).split(",") if x.strip()]
    rows = load_debug_rows(path)
    rows = sorted(rows, key=lambda r: float(r.get("ts_bucket") or 0.0))
    n_total = len(rows)
    cut = max(1, min(n_total - 1, int(n_total * float(args.discover_frac)))) if n_total > 1 else n_total
    disc = rows[:cut]
    valid = rows[cut:]

    min_disc = int(args.min_n_discovery)
    min_val = int(args.min_n_validation)
    if args.min_n is not None:
        min_disc = int(args.min_n)
        min_val = int(args.min_n)
    disc_all = group_stats(disc, group_fields=fields)
    valid_all = group_stats(valid, group_fields=fields)
    min_disc_eff, relaxed_disc = _relax_threshold(disc_all, min_disc, int(args.relax_floor), bool(args.no_relax))
    min_val_eff, relaxed_val = _relax_threshold(valid_all, min_val, int(args.relax_floor), bool(args.no_relax))
    disc_pass = [g for g in disc_all if int(g.get("n", 0)) >= int(min_disc_eff)]
    valid_n_by_group = {str(g["group"]): int(g.get("n", 0)) for g in valid_all}
    disc_pass.sort(key=lambda r: float(r.get("avg_net", 0.0)), reverse=True)
    top_disc = disc_pass[: max(1, int(args.top_k))]
    top_keys = {
        str(g["group"])
        for g in top_disc
        if int(valid_n_by_group.get(str(g["group"]), 0)) >= int(min_val_eff)
    }

    disc_sel = _selection(disc, top_keys, fields)
    valid_sel = _selection(valid, top_keys, fields)

    disc_frac = (len(disc_sel) / len(disc)) if disc else 0.0
    valid_frac = (len(valid_sel) / len(valid)) if valid else 0.0
    disc_sm = summarize(disc_sel)
    valid_sm = summarize(valid_sel)
    rejected_small_disc = disc_frac < float(args.min_select_frac)
    rejected_small_valid = valid_frac < float(args.min_select_frac)
    rejected_small = (rejected_small_disc or rejected_small_valid) and (not bool(args.allow_small_selection))
    collapse, collapse_flags, collapse_vals = evaluate_collapse_flags(
        disc_sm=disc_sm,
        valid_sm=valid_sm,
        disc_frac=disc_frac,
        valid_frac=valid_frac,
        avg_eps=float(args.collapse_avg_eps),
        p90_eps=float(args.collapse_p90_eps),
        select_ratio=float(args.collapse_select_ratio),
        n_ratio=float(args.collapse_n_ratio),
        mode=str(args.collapse_mode),
    )
    disc_liq = _liquidation_impact(disc_sel)
    valid_liq = _liquidation_impact(valid_sel)
    disc_liq_tag = _liq_regime_tag_impact(disc_sel)
    valid_liq_tag = _liq_regime_tag_impact(valid_sel)
    disc_event_ctx = _event_lane_context_impact(disc_sel)
    valid_event_ctx = _event_lane_context_impact(valid_sel)

    print(
        f"validate_micro_edge_forward debug={path} total={n_total} discover={len(disc)} valid={len(valid)} "
        f"discover_frac={args.discover_frac}"
    )
    print(
        f"groups_total_discovery={len(disc_all)} groups_total_validation={len(valid_all)} "
        f"groups_passing_min_n_discovery={len(disc_pass)} groups_passing_min_n_validation="
        f"{sum(1 for g in valid_all if int(g.get('n',0)) >= int(min_val_eff))}"
    )
    print(
        f"min_n_discovery={min_disc} effective={min_disc_eff} relaxed={'yes' if relaxed_disc else 'no'} "
        f"min_n_validation={min_val} effective={min_val_eff} relaxed={'yes' if relaxed_val else 'no'}"
    )
    if relaxed_disc:
        print(f"RELAX min_n_discovery from {min_disc} to {min_disc_eff}")
    if relaxed_val:
        print(f"RELAX min_n_validation from {min_val} to {min_val_eff}")
    print(
        f"selection groups={len(top_keys)} selection_fraction_discovery={disc_frac:.4f} "
        f"selection_fraction_validation={valid_frac:.4f} min_select_frac={float(args.min_select_frac):.4f}"
    )
    if rejected_small:
        if rejected_small_disc:
            print("REJECT small_selection_fraction slice=discovery")
        if rejected_small_valid:
            print("REJECT small_selection_fraction slice=validation")

    print("DISCOVERY_METRICS")
    print(
        f"n={int(disc_sm['n'])} avg_net={_fmt_num(disc_sm['avg_net'])} p90_net={_fmt_num(disc_sm['p90_net'])} "
        f"p90<0={'YES' if bool(disc_sm['p90_net_negative']) else 'NO'}"
    )
    print("VALIDATION_METRICS")
    print(
        f"n={int(valid_sm['n'])} avg_net={_fmt_num(valid_sm['avg_net'])} p90_net={_fmt_num(valid_sm['p90_net'])} "
        f"p90<0={'YES' if bool(valid_sm['p90_net_negative']) else 'NO'}"
    )
    print("DISCOVERY_VS_VALIDATION")
    print(
        f"avg_net: discovery={_fmt_num(disc_sm['avg_net'])} validation={_fmt_num(valid_sm['avg_net'])} "
        f"delta={_fmt_num(float(valid_sm.get('avg_net',0.0))-float(disc_sm.get('avg_net',0.0)))}"
    )
    print(
        f"p90_net: discovery={_fmt_num(disc_sm['p90_net'])} validation={_fmt_num(valid_sm['p90_net'])} "
        f"delta={_fmt_num(float(valid_sm.get('p90_net',0.0))-float(disc_sm.get('p90_net',0.0)))}"
    )
    print(
        f"n: discovery={int(disc_sm['n'])} validation={int(valid_sm['n'])} "
        f"select_frac: discovery={disc_frac:.4f} validation={valid_frac:.4f}"
    )
    print("FLAGS")
    print(
        f"collapse_mode={args.collapse_mode} collapse_avg_eps={float(args.collapse_avg_eps):.6f} "
        f"collapse_p90_eps={float(args.collapse_p90_eps):.6f} collapse_select_ratio={float(args.collapse_select_ratio):.3f} "
        f"collapse_n_ratio={float(args.collapse_n_ratio):.3f}"
    )
    for k in ["p90_sign_flip", "avg_net_drop", "p90_drop", "selection_frac_drop", "n_drop"]:
        print(f"flag_{k}={'YES' if bool(collapse_flags.get(k)) else 'NO'}")
    print(
        f"measured disc_avg={_fmt_num(collapse_vals['disc_avg_net'])} valid_avg={_fmt_num(collapse_vals['valid_avg_net'])} "
        f"disc_p90={_fmt_num(collapse_vals['disc_p90_net'])} valid_p90={_fmt_num(collapse_vals['valid_p90_net'])}"
    )
    print(
        f"measured disc_select={collapse_vals['disc_select_frac']:.4f} valid_select={collapse_vals['valid_select_frac']:.4f} "
        f"select_threshold={collapse_vals['select_threshold']:.4f} disc_n={int(collapse_vals['disc_n'])} "
        f"valid_n={int(collapse_vals['valid_n'])} n_threshold={collapse_vals['n_threshold']:.2f}"
    )
    if collapse:
        print("FLAG validation_collapse_detected")
    else:
        print("OK validation_noncollapsed")

    print("TOP_DISCOVERY_GROUPS")
    for g in top_disc:
        print(
            f"group={g['group']} n={int(g['n'])} avg_net={_fmt_num(g['avg_net'])} "
            f"p90_net={_fmt_num(g['p90_net'])} p90<0={'YES' if bool(g['p90_net_negative']) else 'NO'}"
        )
    if bool(disc_liq.get("available")) or bool(valid_liq.get("available")):
        print("LIQUIDATION_IMPACT")
        for name, section in (("discovery", disc_liq), ("validation", valid_liq)):
            if not bool(section.get("available")):
                print(f"{name}: unavailable")
                continue
            active = section.get("active", {})
            inactive = section.get("inactive", {})
            print(
                f"{name}: threshold_q75={float(section.get('threshold_q75', 0.0)):.6f} "
                f"active_n={int(active.get('n', 0) or 0)} active_avg_net={_fmt_num(active.get('avg_net'))} "
                f"active_p90_net={_fmt_num(active.get('p90_net'))} inactive_n={int(inactive.get('n', 0) or 0)} "
                f"inactive_avg_net={_fmt_num(inactive.get('avg_net'))} inactive_p90_net={_fmt_num(inactive.get('p90_net'))}"
            )
    print("EVENT_LANE_CONTEXT")
    for name, section in (("discovery", disc_event_ctx), ("validation", valid_event_ctx)):
        top_lane = str(section.get("top_lane_by_delta_avg_net", ""))
        by_lane = section.get("by_lane", {}) if isinstance(section, dict) else {}
        top = by_lane.get(top_lane, {}) if isinstance(by_lane, dict) else {}
        print(
            f"{name}: top_lane={top_lane or '-'} tagged_n={int(top.get('tagged_n', 0) or 0)} "
            f"delta_avg_net={_fmt_num(top.get('delta_avg_net'))} delta_p90_net={_fmt_num(top.get('delta_p90_net'))}"
        )

    payload = {
        "debug": str(path),
        "group_by": fields,
        "discover_frac": float(args.discover_frac),
        "counts": {
            "total": int(n_total),
            "discovery": int(len(disc)),
            "validation": int(len(valid)),
            "selected_discovery": int(len(disc_sel)),
            "selected_validation": int(len(valid_sel)),
            "top_groups": int(len(top_keys)),
        },
        "thresholds": {
            "min_n_discovery": int(min_disc_eff),
            "min_n_validation": int(min_val_eff),
            "min_select_frac": float(args.min_select_frac),
        },
        "discovery": disc_sm,
        "validation": valid_sm,
        "collapse": {
            "detected": bool(collapse),
            "flags": collapse_flags,
            "values": collapse_vals,
        },
        "liquidation_impact": {
            "discovery": disc_liq,
            "validation": valid_liq,
        },
        "liquidation_regime_tag_impact": {
            "discovery": disc_liq_tag,
            "validation": valid_liq_tag,
        },
        "event_lane_context_impact": {
            "discovery": disc_event_ctx,
            "validation": valid_event_ctx,
        },
        "run_summary": build_run_summary(
            run_type="validate_micro_edge_forward",
            inputs={
                "debug": str(path),
                "group_by": fields,
                "discover_frac": float(args.discover_frac),
                "top_k": int(args.top_k),
            },
            metrics={
                "total": int(n_total),
                "selected_discovery": int(len(disc_sel)),
                "selected_validation": int(len(valid_sel)),
                "collapse_detected": int(bool(collapse)),
            },
            artifacts={"json": str(args.out_json)} if args.out_json else {},
        ),
    }
    if args.out_json:
        out_path = Path(str(args.out_json))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
