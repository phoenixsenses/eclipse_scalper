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


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capacity triage per pocket under current min_n/min_n_frac settings.")
    p.add_argument("--candidates-md", required=True)
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--lookback-min", type=int, default=10080)
    p.add_argument("--bucket-sec", type=int, default=1)
    p.add_argument("--rule", default="micro_edge_v3_passive_alpha")
    p.add_argument("--side", default="auto")
    p.add_argument("--splits", type=int, default=6)
    p.add_argument("--seeds", default="7,11,22,33,44,55,66,77,88")
    p.add_argument("--min-n", type=int, default=50)
    p.add_argument("--min-n-frac", type=float, default=0.0005)
    p.add_argument("--maker-fee-bps", type=float, default=float(DEFAULT_MAKER_FEE_BPS))
    p.add_argument("--passive-adverse-mult", type=float, default=1.2)
    p.add_argument("--mitigation-profile", default="baseline", choices=["baseline", "anti_adverse_v2", "anti_adverse_v3"])
    p.add_argument("--max-volatility-extreme", type=float, default=None)
    p.add_argument("--vol-quantile-reject", type=float, default=None)
    p.add_argument("--passive-profile-in", default="state/passive_realistic_profiles.json")
    p.add_argument("--out-json", default="reports/TRIAGE_CAPACITY.json")
    return p.parse_args()


def main() -> int:
    args = _args()
    max_volatility_extreme = None if args.max_volatility_extreme is None else float(args.max_volatility_extreme)
    vol_quantile_reject = None if args.vol_quantile_reject is None else float(args.vol_quantile_reject)
    if str(args.mitigation_profile) == "anti_adverse_v2":
        if max_volatility_extreme is None:
            max_volatility_extreme = 0.0020
        vol_quantile_reject = None
    elif str(args.mitigation_profile) == "anti_adverse_v3":
        if vol_quantile_reject is None:
            vol_quantile_reject = 0.01
        max_volatility_extreme = None

    candidates: List[Dict[str, Any]] = []
    for p in _parse_list(args.candidates_md):
        parsed, _ = _parse_candidates_from_md(Path(p), debug=False)
        candidates.extend(parsed)
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for c in candidates:
        k = (c["symbol"], int(c["horizon_sec"]), float(c["min_imbalance"]), float(c["min_trade_intensity"]), float(c["max_spread"]))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
    candidates = uniq

    rows: List[Dict[str, Any]] = []
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
            min_n_frac=float(args.min_n_frac),
            maker_fee_bps=float(args.maker_fee_bps),
            passive_profile_in=str(args.passive_profile_in),
            passive_adverse_mult=float(args.passive_adverse_mult),
            max_volatility_extreme=float(max_volatility_extreme) if max_volatility_extreme is not None else 0.0,
            vol_quantile_reject=float(vol_quantile_reject) if vol_quantile_reject is not None else 0.0,
        )
        per = list(res.get("per_combo", []))
        if per:
            med_apm = float(median(float(r.get("attempts_per_min", 0.0)) for r in per))
            med_eff = int(median(int(r.get("effective_min_n", 0) or 0) for r in per))
            med_fill = float(median(float(r.get("filled_n", 0.0)) for r in per))
            fail_prob = sum(1 for r in per if str(r.get("fail_reason", "")) == "insufficient_fills") / len(per)
            exp_fills_split = med_fill
        else:
            med_apm = 0.0
            med_eff = 0
            exp_fills_split = 0.0
            fail_prob = 1.0
        rows.append(
            {
                "symbol": c["symbol"],
                "horizon_sec": int(c["horizon_sec"]),
                "min_imbalance": float(c["min_imbalance"]),
                "min_trade_intensity": float(c["min_trade_intensity"]),
                "max_spread": float(c["max_spread"]),
                "attempts_per_min": med_apm,
                "effective_min_n_median": med_eff,
                "expected_fills_per_split": exp_fills_split,
                "prob_fail_insufficient_fills": fail_prob,
            }
        )

    rows.sort(key=lambda r: (float(r["prob_fail_insufficient_fills"]), -float(r["attempts_per_min"])))
    print("triage_capacity")
    print(
        "gate_config "
        f"mitigation_profile={args.mitigation_profile} "
        f"max_volatility_extreme={max_volatility_extreme} "
        f"vol_quantile_reject={vol_quantile_reject}"
    )
    print("symbol h imb int spr attempts/min eff_min_n exp_fills/split p_fail_insufficient")
    for r in rows[:20]:
        print(
            f"{r['symbol']:8} {int(r['horizon_sec']):3d} {float(r['min_imbalance']):.2f} {float(r['min_trade_intensity']):.0f} "
            f"{float(r['max_spread']):.6f} {float(r['attempts_per_min']):6.2f} {int(r['effective_min_n_median']):5d} "
            f"{float(r['expected_fills_per_split']):7.2f} {float(r['prob_fail_insufficient_fills']):.2%}"
        )

    out = Path(str(args.out_json))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "inputs": vars(args),
                "gate_config": {
                    "mitigation_profile": str(args.mitigation_profile),
                    "max_volatility_extreme": max_volatility_extreme,
                    "vol_quantile_reject": vol_quantile_reject,
                },
                "rows": rows,
                "run_summary": build_run_summary(
                    run_type="triage_capacity",
                    inputs={"candidates_md": str(args.candidates_md), "db": str(args.db), "mitigation_profile": str(args.mitigation_profile)},
                    metrics={"candidate_count": len(candidates), "row_count": len(rows)},
                    artifacts={"json": str(out)},
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
