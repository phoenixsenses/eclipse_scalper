from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.analyze_micro_edge_regimes import group_key, group_stats, load_debug_rows, summarize


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
    return p.parse_args(argv)


def _fmt_num(x: Any) -> str:
    try:
        return f"{float(x):+.6f}"
    except Exception:
        return "-"


def _selection(rows: List[Dict[str, Any]], keys: set[str], fields: List[str]) -> List[Dict[str, Any]]:
    return [r for r in rows if group_key(r, fields) in keys]


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
