from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List

from config.costs import DEFAULT_MAKER_FEE_BPS
from tools.rank_passive_pockets_forward import _parse_candidates_from_md
from tools.run_summary import build_run_summary
from tools.validate_passive_pocket_forward import validate_pocket_forward


def _parse_list(raw: str) -> List[str]:
    out: List[str] = []
    for tok in str(raw or "").replace(";", ",").split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _parse_int_list(raw: str) -> List[int]:
    return [int(x) for x in _parse_list(raw)]


def _parse_float_list(raw: str) -> List[float]:
    return [float(x) for x in _parse_list(raw)]


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate min_n_frac against real capacity/forward behavior.")
    p.add_argument("--candidates-md", required=True)
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--lookback-min", type=int, default=10080)
    p.add_argument("--bucket-sec", type=int, default=1)
    p.add_argument("--horizons", default="")
    p.add_argument("--rule", default="micro_edge_v3_passive_alpha")
    p.add_argument("--side", default="auto")
    p.add_argument("--splits", type=int, default=6)
    p.add_argument("--seeds", default="7,11,22,33,44,55,66,77,88")
    p.add_argument("--min-n", type=int, default=50)
    p.add_argument("--min-n-frac-grid", default="0.0001,0.00025,0.0005,0.00075,0.001,0.002,0.003")
    p.add_argument("--maker-fee-bps", type=float, default=float(DEFAULT_MAKER_FEE_BPS))
    p.add_argument("--passive-adverse-mult", type=float, default=1.2)
    p.add_argument("--passive-profile-in", default="state/passive_realistic_profiles.json")
    p.add_argument("--out-md", default="reports/CAPACITY_THRESHOLD_CALIBRATION.md")
    p.add_argument("--out-json", default="reports/CAPACITY_THRESHOLD_CALIBRATION.json")
    return p.parse_args()


def main() -> int:
    args = _args()
    md_paths = [Path(x) for x in _parse_list(args.candidates_md)]
    candidates: List[Dict[str, Any]] = []
    for p in md_paths:
        parsed, _ = _parse_candidates_from_md(p, debug=False)
        candidates.extend(parsed)
    # dedupe by canonical key
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for c in candidates:
        k = (c["symbol"], int(c["horizon_sec"]), float(c["min_imbalance"]), float(c["min_trade_intensity"]), float(c["max_spread"]))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
    candidates = uniq

    horizon_filter = set(_parse_int_list(args.horizons)) if str(args.horizons).strip() else set()
    if horizon_filter:
        candidates = [c for c in candidates if int(c.get("horizon_sec", 0)) in horizon_filter]

    grid = _parse_float_list(args.min_n_frac_grid)
    summary_rows: List[Dict[str, Any]] = []
    for frac in grid:
        candidate_stats: List[Dict[str, Any]] = []
        for c in candidates:
            res = validate_pocket_forward(
                db=str(args.db),
                symbol=str(c["symbol"]),
                lookback_min=int(args.lookback_min),
                bucket_sec=int(args.bucket_sec),
                horizon_sec=int(c["horizon_sec"]),
                rule=str(args.rule),
                side=str(args.side),
                min_imbalance=float(c["min_imbalance"]),
                min_trade_intensity=float(c["min_trade_intensity"]),
                max_spread=float(c["max_spread"]),
                splits=int(args.splits),
                seeds=str(args.seeds),
                min_n=int(args.min_n),
                min_n_frac=float(frac),
                maker_fee_bps=float(args.maker_fee_bps),
                passive_profile_in=str(args.passive_profile_in),
                passive_adverse_mult=float(args.passive_adverse_mult),
            )
            per_combo = list(res.get("per_combo", []))
            if per_combo:
                med_attempts_per_min = float(median(float(r.get("attempts_per_min", 0.0)) for r in per_combo))
                med_attempt_fill = float(median(float(r.get("attempt_fill_rate", 0.0)) for r in per_combo))
                med_npa = float(median(float(r.get("net_per_attempt", 0.0)) for r in per_combo))
            else:
                med_attempts_per_min = 0.0
                med_attempt_fill = 0.0
                med_npa = 0.0
            candidate_stats.append(
                {
                    "pass_rate": float(res.get("pass_rate", 0.0)),
                    "insufficient_fill_rate": float(res.get("insufficient_fill_rate", 1.0)),
                    "attempts_per_min": med_attempts_per_min,
                    "attempt_fill_rate": med_attempt_fill,
                    "net_per_attempt": med_npa,
                    "effective_min_n_median": int(res.get("effective_min_n_median", 0) or 0),
                    "frac_min_component_median": int(res.get("frac_min_component_median", 0) or 0),
                }
            )
        if candidate_stats:
            cap_pass_pct = sum(1 for s in candidate_stats if float(s["insufficient_fill_rate"]) <= 0.5) / len(candidate_stats)
            med_apm = float(median(float(s["attempts_per_min"]) for s in candidate_stats))
            med_afr = float(median(float(s["attempt_fill_rate"]) for s in candidate_stats))
            med_npa = float(median(float(s["net_per_attempt"]) for s in candidate_stats))
            med_eff_min = int(median(int(s["effective_min_n_median"]) for s in candidate_stats))
            med_frac_comp = int(median(int(s["frac_min_component_median"]) for s in candidate_stats))
            dominance_mode = "frac_component" if med_frac_comp > int(args.min_n) else "min_n"
            pass_rates = sorted(float(s["pass_rate"]) for s in candidate_stats)
            n = len(pass_rates)
            p25 = pass_rates[int(max(0, round((n - 1) * 0.25)))]
            p50 = pass_rates[int(max(0, round((n - 1) * 0.50)))]
            p75 = pass_rates[int(max(0, round((n - 1) * 0.75)))]
        else:
            cap_pass_pct = med_apm = med_afr = med_npa = p25 = p50 = p75 = 0.0
            med_eff_min = med_frac_comp = 0
            dominance_mode = "none"
        summary_rows.append(
            {
                "min_n_frac": float(frac),
                "candidate_count": int(len(candidates)),
                "capacity_pass_pct": float(cap_pass_pct),
                "median_attempts_per_min": float(med_apm),
                "median_attempt_fill_rate": float(med_afr),
                "median_net_per_attempt": float(med_npa),
                "median_effective_min_n": int(med_eff_min),
                "dominance_mode": str(dominance_mode),
                "forward_pass_rate_p25": float(p25),
                "forward_pass_rate_p50": float(p50),
                "forward_pass_rate_p75": float(p75),
            }
        )

    out_json = Path(str(args.out_json))
    payload = {
        "inputs": {
            "db": args.db,
            "lookback_min": int(args.lookback_min),
            "bucket_sec": int(args.bucket_sec),
            "rule": args.rule,
            "side": args.side,
            "splits": int(args.splits),
            "seeds": str(args.seeds),
            "min_n": int(args.min_n),
            "maker_fee_bps": float(args.maker_fee_bps),
            "passive_adverse_mult": float(args.passive_adverse_mult),
        },
        "rows": summary_rows,
    }
    payload["run_summary"] = build_run_summary(
        run_type="calibrate_capacity_thresholds",
        inputs={"candidates_md": str(args.candidates_md), "db": str(args.db), "min_n": int(args.min_n), "grid": grid},
        metrics={"candidate_count": len(candidates), "row_count": len(summary_rows)},
        artifacts={"json": str(out_json), "md": str(args.out_md)},
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md = Path(str(args.out_md))
    lines = [
        "# CAPACITY_THRESHOLD_CALIBRATION",
        "",
        f"candidates={len(candidates)} rule={args.rule} fee={args.maker_fee_bps} adverse={args.passive_adverse_mult}",
        "",
        "| min_n_frac | candidate_count | capacity_pass_pct | median_attempts_per_min | median_attempt_fill_rate | median_net_per_attempt | median_effective_min_n | dominance_mode | pass_rate_p25 | pass_rate_p50 | pass_rate_p75 | note |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for r in summary_rows:
        note = (
            "rows may look identical while min_n dominates effective_min_n"
            if str(r.get("dominance_mode")) == "min_n"
            else "frac component influences effective_min_n"
        )
        lines.append(
            f"| {r['min_n_frac']:.6f} | {r['candidate_count']} | {r['capacity_pass_pct']:.2%} | {r['median_attempts_per_min']:.2f} | "
            f"{r['median_attempt_fill_rate']:.2%} | {r['median_net_per_attempt']:+.6e} | {r['median_effective_min_n']} | {r['dominance_mode']} | "
            f"{r['forward_pass_rate_p25']:.2%} | {r['forward_pass_rate_p50']:.2%} | {r['forward_pass_rate_p75']:.2%} | {note} |"
        )
    lines.extend(["", "## Run Summary", f"- `{payload['run_summary']}`"])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
