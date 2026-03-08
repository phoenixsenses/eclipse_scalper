from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from config.costs import DEFAULT_MAKER_FEE_BPS
from tools.run_summary import build_run_summary

def _parse_list(raw: str) -> List[str]:
    out: List[str] = []
    for tok in str(raw or "").replace(";", ",").split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _parse_float_list(raw: str) -> List[float]:
    vals = [float(x) for x in _parse_list(raw)]
    # preserve input order but remove exact duplicates for deterministic sweep.
    uniq: List[float] = []
    seen = set()
    for v in vals:
        if v in seen:
            continue
        seen.add(v)
        uniq.append(v)
    return uniq


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if cp.returncode == 0:
            out = cp.stdout.strip()
            return out or None
    except Exception:
        return None
    return None


def _stable_hash(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _extract_summary(rank_json_path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(rank_json_path.read_text(encoding="utf-8"))
    except Exception:
        return {"count": 0, "top": None}
    ranking = payload.get("ranking") or []
    liq_impact = payload.get("liquidation_scoring_impact") or {}
    top = ranking[0] if ranking else None
    top_summary = None
    if isinstance(top, dict):
        top_summary = {
            "symbol": top.get("symbol"),
            "rule": top.get("rule"),
            "horizon_sec": top.get("horizon_sec"),
            "score": top.get("score"),
            "score_raw_core": top.get("score_raw_core"),
            "npa_core": top.get("npa_core"),
            "pass_rate_core": top.get("pass_rate_core"),
            "failure_reason_top": top.get("failure_reason_top"),
        }
    return {
        "count": int(payload.get("count", 0) or 0),
        "top": top_summary,
        "liquidation_scoring_impact": {
            "available": bool(liq_impact.get("available", False)),
            "count": int(liq_impact.get("count", 0) or 0),
            "positive_delta_score_count": int(liq_impact.get("positive_delta_score_count", 0) or 0),
            "avg_delta_score_raw_core": liq_impact.get("avg_delta_score_raw_core"),
            "avg_delta_npa_core": liq_impact.get("avg_delta_npa_core"),
            "avg_delta_pass_rate_core": liq_impact.get("avg_delta_pass_rate_core"),
        },
    }


def _invoke_rank(argv: List[str]) -> int:
    cp = subprocess.run([sys.executable, "-m", "tools.rank_passive_pockets_forward", *argv], check=False)
    return int(cp.returncode)


def _build_rank_args(
    *,
    db: str,
    lookback_min: int,
    bucket_sec: int,
    candidates_md: str,
    splits: int,
    seeds: str,
    min_n: int,
    min_n_frac: float,
    rule: str,
    side: str,
    mitigation_profile: str,
    maker_fee_bps: float,
    passive_adverse_mult: float,
    vol_quantile_reject: float,
    out_md: Path,
    out_json: Path,
    research_mode: bool,
) -> List[str]:
    args = [
        "--db",
        str(db),
        "--lookback-min",
        str(int(lookback_min)),
        "--bucket-sec",
        str(int(bucket_sec)),
        "--candidates-md",
        str(candidates_md),
        "--splits",
        str(int(splits)),
        "--seeds",
        str(seeds),
        "--min-n",
        str(int(min_n)),
        "--min-n-frac",
        str(float(min_n_frac)),
        "--rule",
        str(rule),
        "--side",
        str(side),
        "--mitigation-profile",
        str(mitigation_profile),
        "--maker-fee-bps-grid",
        str(float(maker_fee_bps)),
        "--passive-adverse-mult-grid",
        str(float(passive_adverse_mult)),
        "--vol-quantile-reject",
        str(float(vol_quantile_reject)),
        "--out-md",
        str(out_md),
        "--out-json",
        str(out_json),
    ]
    if bool(research_mode):
        args.append("--research-mode")
    return args


def _plan_runs(
    maker_fee_bps: Iterable[float],
    passive_adverse_mult: Iterable[float],
    vol_quantile_reject: Iterable[float],
) -> List[Tuple[float, float, float]]:
    grid = list(itertools.product(maker_fee_bps, passive_adverse_mult, vol_quantile_reject))
    grid.sort(key=lambda t: (float(t[0]), float(t[1]), float(t[2])))
    return grid


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run sweep grid over tools.rank_passive_pockets_forward.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--lookback-min", type=int, default=10080)
    p.add_argument("--bucket-sec", type=int, default=1)
    p.add_argument("--candidates-md", required=True)
    p.add_argument("--splits", type=int, default=3)
    p.add_argument("--seeds", default="7,11,22,33,44,55,66,77,88")
    p.add_argument("--min-n", type=int, default=20)
    p.add_argument("--min-n-frac", type=float, default=0.00010)
    p.add_argument("--rule", default="micro_edge_v3_passive_alpha")
    p.add_argument("--side", default="auto")
    p.add_argument("--mitigation-profile", default="anti_adverse_v3")
    p.add_argument("--maker-fee-bps-grid", default=f"{float(DEFAULT_MAKER_FEE_BPS)}")
    p.add_argument("--passive-adverse-mult-grid", default="1.0,1.2,1.5")
    p.add_argument("--vol-quantile-reject-grid", default="0.01")
    p.add_argument("--research-mode", action="store_true")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--registry", default="reports/RUN_RANK_SWEEP_REGISTRY.jsonl")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _args()
    maker_fee_vals = _parse_float_list(args.maker_fee_bps_grid)
    adverse_vals = _parse_float_list(args.passive_adverse_mult_grid)
    vol_q_vals = _parse_float_list(args.vol_quantile_reject_grid)
    planned = _plan_runs(maker_fee_vals, adverse_vals, vol_q_vals)
    if not planned:
        print("no planned runs")
        return 2

    reports_dir = Path(str(args.reports_dir))
    runs_root = reports_dir / "_runs"
    registry_path = Path(str(args.registry))
    runs_root.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    git_commit = _git_commit()

    print(f"planned_runs={len(planned)}")
    rc_any = 0
    for fee, adv, volq in planned:
        run_args = {
            "db": str(args.db),
            "lookback_min": int(args.lookback_min),
            "bucket_sec": int(args.bucket_sec),
            "candidates_md": str(args.candidates_md),
            "splits": int(args.splits),
            "seeds": str(args.seeds),
            "min_n": int(args.min_n),
            "min_n_frac": float(args.min_n_frac),
            "rule": str(args.rule),
            "side": str(args.side),
            "mitigation_profile": str(args.mitigation_profile),
            "maker_fee_bps": float(fee),
            "passive_adverse_mult": float(adv),
            "vol_quantile_reject": float(volq),
            "research_mode": bool(args.research_mode),
        }
        run_hash = _stable_hash(run_args)
        run_id = f"rank_{run_hash[:12]}"
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        out_md = run_dir / "rank.md"
        out_json = run_dir / "rank.json"
        cli_args = _build_rank_args(
            db=run_args["db"],
            lookback_min=run_args["lookback_min"],
            bucket_sec=run_args["bucket_sec"],
            candidates_md=run_args["candidates_md"],
            splits=run_args["splits"],
            seeds=run_args["seeds"],
            min_n=run_args["min_n"],
            min_n_frac=run_args["min_n_frac"],
            rule=run_args["rule"],
            side=run_args["side"],
            mitigation_profile=run_args["mitigation_profile"],
            maker_fee_bps=run_args["maker_fee_bps"],
            passive_adverse_mult=run_args["passive_adverse_mult"],
            vol_quantile_reject=run_args["vol_quantile_reject"],
            out_md=out_md,
            out_json=out_json,
            research_mode=run_args["research_mode"],
        )

        cmd_preview = f"{sys.executable} -m tools.rank_passive_pockets_forward " + " ".join(cli_args)
        print(f"[{run_id}] fee={fee} adv={adv} volq={volq} -> {out_json}")
        if bool(args.dry_run):
            print(f"DRY_RUN {cmd_preview}")
            continue

        rank_rc = _invoke_rank(cli_args)
        if rank_rc != 0:
            rc_any = rank_rc
        summary = _extract_summary(out_json) if out_json.exists() else {"count": 0, "top": None}
        reg = {
            "run_id": run_id,
            "timestamp_utc": _utc_now_iso(),
            "git_commit": git_commit,
            "args": run_args,
            "outputs": {"md": str(out_md), "json": str(out_json)},
            "summary": summary,
            "returncode": int(rank_rc),
        }
        reg["run_summary"] = build_run_summary(
            run_type="run_rank_sweep",
            inputs=run_args,
            metrics={
                "returncode": int(rank_rc),
                "ranking_count": int(summary.get("count", 0) or 0),
            },
            artifacts={"registry": str(registry_path), "md": str(out_md), "json": str(out_json)},
        )
        with registry_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(reg, ensure_ascii=True) + "\n")

    if bool(args.dry_run):
        print("dry_run_complete")
        return 0
    print(f"registry={registry_path}")
    return int(rc_any)


if __name__ == "__main__":
    raise SystemExit(main())
