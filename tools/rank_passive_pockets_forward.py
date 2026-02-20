from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import median, pstdev
from typing import Any, Dict, List, Tuple

from tools.validate_passive_pocket_forward import validate_pocket_forward


def _parse_list(raw: str) -> List[str]:
    out: List[str] = []
    for tok in str(raw or "").replace(";", ",").split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _parse_float_list(raw: str) -> List[float]:
    return [float(x) for x in _parse_list(raw)]


def _parse_seed_list(raw: str) -> str:
    return ",".join(_parse_list(raw))


_FIELD_ALIASES: Dict[str, List[str]] = {
    "symbol": ["symbol"],
    "horizon_sec": ["horizon_sec", "horizon", "h", "horizonsec"],
    "min_imbalance": ["min_imbalance", "min_imb", "imb>=", "imb", "minimbalance"],
    "min_trade_intensity": ["min_trade_intensity", "min_int", "int>=", "trade_intensity", "mintradeintensity"],
    "max_spread": ["max_spread", "spr<=", "spr", "spread", "maxspread"],
    "pass": ["pass", "PASS"],
}


def _canon(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name or "").strip().lower())


def _find_col_idx(headers: List[str], aliases: List[str]) -> int:
    canon_headers = [_canon(h) for h in headers]
    wanted = {_canon(a) for a in aliases}
    for i, h in enumerate(canon_headers):
        if h in wanted:
            return i
    return -1


def _is_sep_row(parts: List[str]) -> bool:
    if not parts:
        return True
    for p in parts:
        t = p.strip()
        if not t:
            continue
        if not re.fullmatch(r"[-:]+", t):
            return False
    return True


def _to_float(x: str) -> float:
    s = str(x or "").strip().replace("%", "").replace("+", "")
    return float(s)


def _pass_yes(raw: str) -> bool:
    return _canon(raw) in {"yes", "true", "1", "y"}


def _dedupe_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for c in candidates:
        k = (
            str(c.get("symbol", "")),
            int(c.get("horizon_sec", 0)),
            float(c.get("min_imbalance", 0.0)),
            float(c.get("min_trade_intensity", 0.0)),
            float(c.get("max_spread", 0.0)),
        )
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
    return uniq


def _parse_candidates_from_md(path: Path, debug: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    stats: Dict[str, Any] = {
        "path": str(path),
        "total_rows_seen": 0,
        "table_rows_seen": 0,
        "rows_with_pass_yes": 0,
        "candidates_parsed": 0,
        "candidates_unique": 0,
        "rows_skipped_missing_fields": 0,
        "skipped_examples": [],
        "parsed_examples": [],
    }
    if not path.exists():
        stats["error"] = "missing_file"
        return [], stats

    lines = path.read_text(encoding="utf-8").splitlines()
    stats["total_rows_seen"] = len(lines)
    headers: List[str] = []
    idx_map: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []

    for ln in lines:
        s = ln.strip()
        if not s.startswith("|"):
            continue
        parts = [p.strip() for p in s.strip("|").split("|")]
        if not parts:
            continue
        stats["table_rows_seen"] += 1
        if _is_sep_row(parts):
            continue

        # header detection / re-header handling
        parts_canon = [_canon(p) for p in parts]
        if "symbol" in parts_canon and ("horizonsec" in parts_canon or "horizon" in parts_canon or "h" in parts_canon):
            headers = parts
            idx_map = {k: _find_col_idx(headers, v) for k, v in _FIELD_ALIASES.items()}
            continue

        if not headers:
            # fallback: expected positional schema from classic sweep output
            # symbol|horizon|min_imbalance|min_trade_intensity|max_spread|...|pass
            if len(parts) >= 6:
                headers = ["symbol", "horizon_sec", "min_imbalance", "min_trade_intensity", "max_spread"] + [f"col{i}" for i in range(5, len(parts) - 1)] + ["pass"]
                idx_map = {k: _find_col_idx(headers, v) for k, v in _FIELD_ALIASES.items()}
            else:
                stats["rows_skipped_missing_fields"] += 1
                if len(stats["skipped_examples"]) < 5:
                    stats["skipped_examples"].append({"row": parts, "reason": "no_header_and_too_few_cols"})
                continue

        needed = ["symbol", "horizon_sec", "min_imbalance", "min_trade_intensity", "max_spread", "pass"]
        missing = [k for k in needed if idx_map.get(k, -1) < 0]
        if missing:
            stats["rows_skipped_missing_fields"] += 1
            if len(stats["skipped_examples"]) < 5:
                stats["skipped_examples"].append({"row": parts, "reason": f"missing_cols={missing}"})
            continue

        try:
            pass_raw = parts[idx_map["pass"]]
            if _pass_yes(pass_raw):
                stats["rows_with_pass_yes"] += 1
            else:
                continue
            cand = {
                "symbol": str(parts[idx_map["symbol"]]).strip(),
                "horizon_sec": int(float(_to_float(parts[idx_map["horizon_sec"]]))),
                "min_imbalance": float(_to_float(parts[idx_map["min_imbalance"]])),
                "min_trade_intensity": float(_to_float(parts[idx_map["min_trade_intensity"]])),
                "max_spread": float(_to_float(parts[idx_map["max_spread"]])),
            }
            out.append(cand)
            if len(stats["parsed_examples"]) < 5:
                stats["parsed_examples"].append(cand)
        except Exception as exc:
            stats["rows_skipped_missing_fields"] += 1
            if len(stats["skipped_examples"]) < 5:
                stats["skipped_examples"].append({"row": parts, "reason": f"parse_error={exc}"})

    stats["candidates_parsed"] = len(out)
    uniq = _dedupe_candidates(out)
    stats["candidates_unique"] = len(uniq)
    if debug:
        print(
            "[parse] "
            f"path={path} total_rows_seen={stats['total_rows_seen']} table_rows_seen={stats['table_rows_seen']} "
            f"rows_with_pass_yes={stats['rows_with_pass_yes']} candidates_parsed={stats['candidates_parsed']} "
            f"candidates_unique={stats['candidates_unique']} rows_skipped_missing_fields={stats['rows_skipped_missing_fields']}"
        )
        for ex in stats["parsed_examples"]:
            print(f"[parse] parsed_example={ex}")
        for ex in stats["skipped_examples"]:
            print(f"[parse] skipped_example={ex}")
    return uniq, stats


def _combo_pass(row: Dict[str, Any], min_n: int) -> bool:
    return bool(
        int(row.get("filled_n", 0) or 0) >= int(min_n)
        and float(row.get("filled_avg_net", 0.0)) >= 0.000005
        and float(row.get("filled_p90_net", 0.0)) >= 0.00005
    )


def _split_seed_stability(per_combo: List[Dict[str, Any]]) -> float:
    by_split: Dict[int, List[float]] = {}
    for r in per_combo:
        sid = int(r.get("split", 0) or 0)
        by_split.setdefault(sid, []).append(float(r.get("filled_avg_net", 0.0)))
    if not by_split:
        return 0.0
    stds = []
    for vals in by_split.values():
        stds.append(pstdev(vals) if len(vals) > 1 else 0.0)
    return sum(stds) / len(stds)


def _aggregate_eval(res: Dict[str, Any], min_n: int) -> Dict[str, Any]:
    per_combo = list(res.get("per_combo", []))
    if not per_combo:
        return {
            "pass_rate": 0.0,
            "median_filled_avg_net": 0.0,
            "worst_split_avg_net": 0.0,
            "worst_split_p90_net": 0.0,
            "stability_std": 0.0,
            "median_net_per_attempt": 0.0,
            "attempt_fill_rate_median": 0.0,
            "attempts_per_min_median": 0.0,
            "rows": 0,
        }
    flags = [_combo_pass(r, min_n=min_n) for r in per_combo]
    pass_rate = sum(1 for f in flags if f) / len(flags)
    med_avg = float(median(float(r.get("filled_avg_net", 0.0)) for r in per_combo))
    by_split: Dict[int, List[Dict[str, Any]]] = {}
    for r in per_combo:
        sid = int(r.get("split", 0) or 0)
        by_split.setdefault(sid, []).append(r)
    split_avg = []
    split_p90 = []
    for grp in by_split.values():
        split_avg.append(sum(float(x.get("filled_avg_net", 0.0)) for x in grp) / len(grp))
        split_p90.append(sum(float(x.get("filled_p90_net", 0.0)) for x in grp) / len(grp))
    return {
        "pass_rate": pass_rate,
        "median_filled_avg_net": med_avg,
        "worst_split_avg_net": min(split_avg) if split_avg else 0.0,
        "worst_split_p90_net": min(split_p90) if split_p90 else 0.0,
        "stability_std": _split_seed_stability(per_combo),
        "median_net_per_attempt": float(median(float(r.get("net_per_attempt", 0.0)) for r in per_combo)),
        "attempt_fill_rate_median": float(median(float(r.get("attempt_fill_rate", 0.0)) for r in per_combo)),
        "attempts_per_min_median": float(median(float(r.get("attempts_per_min", 0.0)) for r in per_combo)),
        "median_effective_min_n": int(median(int(r.get("effective_min_n", 0) or 0) for r in per_combo)),
        "rows": len(per_combo),
    }


def _fee_score(agg: Dict[str, Any]) -> float:
    pass_rate = float(agg.get("pass_rate", 0.0))
    med_bps = float(agg.get("median_filled_avg_net", 0.0)) * 10000.0
    worst_bps = max(0.0, float(agg.get("worst_split_avg_net", 0.0)) * 10000.0)
    return pass_rate * med_bps * worst_bps


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank passive pockets by forward robustness.")
    p.add_argument("--candidates-md", required=True)
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--lookback-min", type=int, default=1440)
    p.add_argument("--bucket-sec", type=int, default=1)
    p.add_argument("--rule", default="intensity_spike_imbalance_cont")
    p.add_argument("--rules", default="", help="Optional comma list to compare rules on identical pockets.")
    p.add_argument("--side", default="auto")
    p.add_argument("--splits", type=int, default=4)
    p.add_argument("--seeds", default="11,22,33,44,55")
    p.add_argument("--min-n", type=int, default=50)
    p.add_argument("--min-n-frac", type=float, default=0.0)
    p.add_argument("--maker-fee-bps-grid", default="0.5,1.0,1.5")
    p.add_argument("--passive-adverse-mult-grid", default="0.8,1.0,1.2")
    p.add_argument("--v2-min-score", type=float, default=0.0)
    p.add_argument("--v2-min-persistence", type=float, default=0.0)
    p.add_argument("--v2-min-confidence", type=float, default=0.0)
    p.add_argument("--passive-profile-in", default="state/passive_realistic_profiles.json")
    p.add_argument("--out-md", default="reports/PASSIVE_POCKET_RANKING.md")
    p.add_argument("--out-json", default="reports/PASSIVE_POCKET_RANKING.json")
    p.add_argument("--debug-parse", action="store_true")
    p.add_argument("--min-attempt-fill-rate", type=float, default=0.10,
                   help="Skip pockets whose core-eval attempt_fill_rate_median is below this threshold.")
    p.add_argument("--max-insufficient-fill-rate", type=float, default=0.50,
                   help="Skip pockets whose core-eval insufficient_fill_rate exceeds this threshold.")
    p.add_argument("--min-intensity-strong", type=float, default=0.0, help="Pre-attempt gate: skip when trade_intensity < this.")
    p.add_argument("--min-imbalance-strong", type=float, default=0.0, help="Pre-attempt gate: skip when |imbalance| < this.")
    p.add_argument("--max-spread-tight", type=float, default=0.0, help="Pre-attempt gate: skip when spread > this.")
    p.add_argument("--pass-threshold", type=float, default=0.5, help="Minimum pass_rate required for robust_core/stress flags.")
    return p.parse_args()


def main() -> int:
    args = _args()
    md_paths = [Path(x) for x in _parse_list(args.candidates_md)]
    candidates: List[Dict[str, Any]] = []
    parse_stats_all: List[Dict[str, Any]] = []
    for p in md_paths:
        parsed, st = _parse_candidates_from_md(p, debug=bool(args.debug_parse))
        parse_stats_all.append(st)
        candidates.extend(parsed)
    candidates = _dedupe_candidates(candidates)
    total_rows_seen = sum(int(s.get("total_rows_seen", 0)) for s in parse_stats_all)
    table_rows_seen = sum(int(s.get("table_rows_seen", 0)) for s in parse_stats_all)
    rows_with_pass_yes = sum(int(s.get("rows_with_pass_yes", 0)) for s in parse_stats_all)
    candidates_parsed = sum(int(s.get("candidates_parsed", 0)) for s in parse_stats_all)
    rows_skipped_missing = sum(int(s.get("rows_skipped_missing_fields", 0)) for s in parse_stats_all)
    print(
        "candidate_parse "
        f"total_rows_seen={total_rows_seen} table_rows_seen={table_rows_seen} "
        f"rows_with_pass_yes={rows_with_pass_yes} candidates_parsed={candidates_parsed} "
        f"candidates_unique={len(candidates)} rows_skipped_missing_fields={rows_skipped_missing}"
    )
    if bool(args.debug_parse):
        shown = 0
        for st in parse_stats_all:
            for ex in st.get("skipped_examples", []):
                print(f"[parse] skipped_row={ex}")
                shown += 1
                if shown >= 8:
                    break
            if shown >= 8:
                break
    if not candidates:
        print(
            "ERROR no candidates parsed from candidates-md. "
            "Check markdown headers and PASS values or use --debug-parse for diagnostics."
        )
        return 2
    fee_grid = _parse_float_list(args.maker_fee_bps_grid)
    adverse_grid = _parse_float_list(args.passive_adverse_mult_grid)
    rules = _parse_list(args.rules) if str(args.rules).strip() else [str(args.rule)]
    seed_str = _parse_seed_list(args.seeds)

    scored: List[Dict[str, Any]] = []
    for c in candidates:
        for rule_name in rules:
            pocket_evals: Dict[str, Any] = {}
            for fee in fee_grid:
                for adv in adverse_grid:
                    key = f"fee={fee:.3f}|adv={adv:.3f}"
                    res = validate_pocket_forward(
                        db=str(args.db),
                        symbol=str(c["symbol"]),
                        lookback_min=int(args.lookback_min),
                        bucket_sec=int(args.bucket_sec),
                        horizon_sec=int(c["horizon_sec"]),
                        rule=str(rule_name),
                        side=str(args.side),
                        min_imbalance=float(c["min_imbalance"]),
                        min_trade_intensity=float(c["min_trade_intensity"]),
                        max_spread=float(c["max_spread"]),
                        splits=int(args.splits),
                        seeds=seed_str,
                        min_n=int(args.min_n),
                        min_n_frac=float(args.min_n_frac),
                        maker_fee_bps=float(fee),
                        passive_profile_in=str(args.passive_profile_in),
                        passive_adverse_mult=float(adv),
                        v2_min_score=float(args.v2_min_score),
                        v2_min_persistence=float(args.v2_min_persistence),
                        v2_min_confidence=float(args.v2_min_confidence),
                        min_intensity_strong=float(args.min_intensity_strong),
                        min_imbalance_strong=float(args.min_imbalance_strong),
                        max_spread_tight=float(args.max_spread_tight),
                    )
                    pocket_evals[key] = _aggregate_eval(res, min_n=int(args.min_n))
                    pocket_evals[key]["rows_total"] = int(res.get("rows_total", 0))
                    pocket_evals[key]["pass_count"] = int(res.get("pass_count", 0))
                    pocket_evals[key]["pass_rate_raw"] = float(res.get("pass_rate", 0.0))
                    pocket_evals[key]["insufficient_fill_rate"] = float(res.get("insufficient_fill_rate", 0.0))

            def get_eval(fee: float, adv: float) -> Dict[str, Any]:
                return pocket_evals.get(f"fee={fee:.3f}|adv={adv:.3f}", {})

            fee_min = min(fee_grid)
            fee_max = max(fee_grid)
            adv_max = max(adverse_grid)
            adv_one = min(adverse_grid, key=lambda x: abs(x - 1.0))
            fee_one = min(fee_grid, key=lambda x: abs(x - 1.0))

            core_eval = get_eval(fee_one, adv_one)
            stress_eval = get_eval(fee_one, adv_max)

            # NPA-based scoring (net_per_attempt; capacity-honest metric)
            core_npa = float(core_eval.get("median_net_per_attempt", 0.0))
            stress_npa = float(stress_eval.get("median_net_per_attempt", 0.0))
            stab_bps = float(core_eval.get("stability_std", 0.0)) * 10000.0
            npa_bps = core_npa * 10000.0
            insufficient_fill_rate = float(core_eval.get("insufficient_fill_rate", 0.0))
            capacity_penalty = 1.0 + 0.5 * insufficient_fill_rate
            base_score = max(0.0, npa_bps) / (1.0 + max(0.0, stab_bps))

            # Robustness: edge must be positive at both core and stress conditions
            pass_rate_core = float(core_eval.get("pass_rate", 0.0))
            pass_rate_stress = float(stress_eval.get("pass_rate", 0.0))
            robust_core = (pass_rate_core >= float(args.pass_threshold)) and (core_npa > 0.0)
            robust_stress = (pass_rate_stress >= float(args.pass_threshold)) and (stress_npa > 0.0)
            final_score = (base_score / capacity_penalty) if (robust_core and robust_stress) else 0.0

            # Raw score fields — always populated regardless of robustness gate
            score_raw_core = float(get_eval(fee_min, adv_one).get("median_filled_avg_net", 0.0))
            score_raw_stress = float(get_eval(fee_max, adv_max).get("median_filled_avg_net", 0.0))
            score_raw_min = min(
                (float(e.get("median_filled_avg_net", 0.0)) for e in pocket_evals.values()),
                default=0.0,
            )

            # Capacity filter: skip pockets that cannot generate meaningful flow
            core_afr = float(core_eval.get("attempt_fill_rate_median", 0.0))
            core_eff_min = int(core_eval.get("median_effective_min_n", 0) or 0)
            if core_afr < float(args.min_attempt_fill_rate):
                print(
                    f"[cap_filter] skip symbol={c['symbol']} rule={rule_name} h={c['horizon_sec']} "
                    f"attempt_fill_rate={core_afr:.4f} < threshold={args.min_attempt_fill_rate:.4f} "
                    f"effective_min_n_median={core_eff_min}"
                )
                continue
            if insufficient_fill_rate > float(args.max_insufficient_fill_rate):
                print(
                    f"[cap_filter] skip symbol={c['symbol']} rule={rule_name} h={c['horizon_sec']} "
                    f"insufficient_fill_rate={insufficient_fill_rate:.4f} > threshold={args.max_insufficient_fill_rate:.4f} "
                    f"effective_min_n_median={core_eff_min}. "
                    "Hint: Consider lowering --min-n-frac (e.g. 0.0005) or increasing lookback/splitting."
                )
                continue

            scored.append(
                {
                    **c,
                    "rule": str(rule_name),
                    "score": float(final_score),
                    "stability_std_bps": float(stab_bps),
                    "insufficient_fill_rate": float(insufficient_fill_rate),
                    "robust_core": bool(robust_core),
                    "robust_stress": bool(robust_stress),
                    "score_raw_core": score_raw_core,
                    "score_raw_stress": score_raw_stress,
                    "score_raw_min": score_raw_min,
                    "net_per_attempt": core_npa,
                    "npa_core": core_npa,
                    "npa_stress": stress_npa,
                    "pass_rate_core": pass_rate_core,
                    "pass_rate_stress": pass_rate_stress,
                    "attempt_fill_rate": core_afr,
                    "attempts_per_min": float(core_eval.get("attempts_per_min_median", 0.0)),
                    "best_fee_survive": max(
                        [f for f in fee_grid if float(get_eval(f, adv_one).get("pass_rate", 0.0)) > 0.0] or [0.0]
                    ),
                    "evals": pocket_evals,
                }
            )

    # Primary sort: score (gated); secondary: score_raw_core so zero-gated pockets remain comparable
    scored.sort(
        key=lambda r: (
            float(r.get("score", 0.0)),
            float(r.get("pass_rate_stress", 0.0)),
            float(r.get("npa_stress", 0.0)),
            float(r.get("score_raw_core", 0.0)),
        ),
        reverse=True,
    )

    top10 = scored[:10]
    print("top 10 pockets overall")
    for i, r in enumerate(top10, start=1):
        print(
            f"{i:2d}. {r['symbol']} rule={r['rule']} h={r['horizon_sec']} imb>={r['min_imbalance']:.2f} "
            f"int>={r['min_trade_intensity']:.0f} spr<={r['max_spread']:.6f} "
            f"score={r['score']:.6e} npa={r.get('net_per_attempt', 0.0):.6e} "
            f"pass_core={r.get('pass_rate_core', 0.0):.2%} pass_stress={r.get('pass_rate_stress', 0.0):.2%} "
            f"afr={r.get('attempt_fill_rate', 0.0):.2%} "
            f"score_raw_core={r.get('score_raw_core', 0.0):.6e}"
        )
    for sym in sorted(set(r["symbol"] for r in scored)):
        print(f"top 5 {sym}")
        cnt = 0
        for r in scored:
            if r["symbol"] != sym:
                continue
            print(
                f" - rule={r['rule']} h={r['horizon_sec']} imb>={r['min_imbalance']:.2f} int>={r['min_trade_intensity']:.0f} "
                f"spr<={r['max_spread']:.6f} score={r['score']:.6e} npa={r.get('net_per_attempt', 0.0):.6e} "
                f"pass_core={r.get('pass_rate_core', 0.0):.2%} pass_stress={r.get('pass_rate_stress', 0.0):.2%} "
                f"score_raw_core={r.get('score_raw_core', 0.0):.6e}"
            )
            cnt += 1
            if cnt >= 5:
                break
    survive = sum(1 for r in scored if float(r["evals"].get("fee=1.000|adv=1.000", {}).get("pass_rate", 0.0)) >= 0.5)
    print(f"pockets survive fee>=1.0 with pass_rate>=0.5: {survive}")

    out_json = Path(str(args.out_json))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"count": len(scored), "ranking": scored}, indent=2), encoding="utf-8")

    out_md = Path(str(args.out_md))
    lines = [
        "# PASSIVE_POCKET_RANKING",
        "",
        f"candidates={len(candidates)} ranked={len(scored)}",
        f"candidate_parse total_rows_seen={total_rows_seen} table_rows_seen={table_rows_seen} rows_with_pass_yes={rows_with_pass_yes} candidates_parsed={candidates_parsed} candidates_unique={len(candidates)} rows_skipped_missing_fields={rows_skipped_missing}",
        f"fee_grid={fee_grid} adverse_mult_grid={adverse_grid}",
        f"pass_threshold={float(args.pass_threshold):.3f}",
        "",
        "| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | stability_std_bps | best_fee_survive | insufficient_fill_rate |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, r in enumerate(scored, start=1):
        lines.append(
            f"| {idx} | {r['symbol']} | {r['rule']} | {r['horizon_sec']} | {r['min_imbalance']:.2f} | {r['min_trade_intensity']:.0f} | {r['max_spread']:.6f} | "
            f"{r['score']:.6e} | {r['robust_core']} | {r['robust_stress']} | "
            f"{r.get('pass_rate_core', 0.0):.2%} | {r.get('pass_rate_stress', 0.0):.2%} | {r.get('npa_core', 0.0):+.6e} | {r.get('npa_stress', 0.0):+.6e} | "
            f"{r.get('attempt_fill_rate', 0.0):.2%} | {r.get('attempts_per_min', 0.0):.2f} | "
            f"{r.get('score_raw_core', 0.0):+.8f} | {r.get('score_raw_stress', 0.0):+.8f} | {r.get('score_raw_min', 0.0):+.8f} | "
            f"{r['stability_std_bps']:.3f} | {r['best_fee_survive']:.2f} | {float(r.get('insufficient_fill_rate', 0.0)):.2%} |"
        )
    lines += [
        "",
        f"survive_fee1_passrate_ge_0.5={survive}",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
