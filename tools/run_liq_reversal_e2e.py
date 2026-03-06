from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from tools.run_summary import build_run_summary


def _parse_list(raw: str) -> List[str]:
    out: List[str] = []
    for tok in str(raw or "").replace(";", ",").split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _run_tool(module: str, args: List[str]) -> int:
    cp = subprocess.run([sys.executable, "-m", module, *args], check=False)
    return int(cp.returncode)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rank_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    ranking = payload.get("ranking") or []
    top = ranking[0] if isinstance(ranking, list) and ranking else {}
    return {
        "count": int(payload.get("count", 0) or 0),
        "top": {
            "symbol": top.get("symbol"),
            "rule": top.get("rule"),
            "horizon_sec": top.get("horizon_sec"),
            "score": top.get("score"),
            "score_raw_core": top.get("score_raw_core"),
            "npa_core": top.get("npa_core"),
            "pass_rate_core": top.get("pass_rate_core"),
            "attempt_fill_rate": top.get("attempt_fill_rate"),
            "failure_reason_top": top.get("failure_reason_top"),
        }
        if isinstance(top, dict) and top
        else None,
    }


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run end-to-end liquidation reversal research chain.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--lookback-min", type=int, default=1440)
    p.add_argument("--coverage-lookbacks-min", default="60,1440,10080")
    p.add_argument("--bucket-sec", type=int, default=5)
    p.add_argument("--horizons-sec", default="30,60")
    p.add_argument("--min-imbalances", default="0.25,0.35")
    p.add_argument("--min-trade-intensities", default="100,200")
    p.add_argument("--max-spreads", default="0.00035")
    p.add_argument("--splits", type=int, default=2)
    p.add_argument("--seeds", default="7,11,22")
    p.add_argument("--min-n", type=int, default=20)
    p.add_argument("--min-n-frac", type=float, default=0.00001)
    p.add_argument("--rule", default="high_liq_reversal_regime")
    p.add_argument("--reports-prefix", default="reports/LIQ_REVERSAL_E2E")
    return p.parse_args()


def main() -> int:
    args = _args()
    prefix = Path(str(args.reports_prefix))
    prefix.parent.mkdir(parents=True, exist_ok=True)

    coverage_json = prefix.with_name(prefix.name + "_COVERAGE.json")
    coverage_md = prefix.with_name(prefix.name + "_COVERAGE.md")
    candidates_json = prefix.with_name(prefix.name + "_CANDIDATES.json")
    candidates_md = prefix.with_name(prefix.name + "_CANDIDATES.md")
    rank_baseline_json = prefix.with_name(prefix.name + "_RANK_BASELINE.json")
    rank_baseline_md = prefix.with_name(prefix.name + "_RANK_BASELINE.md")
    rank_v5_json = prefix.with_name(prefix.name + "_RANK_V5.json")
    rank_v5_md = prefix.with_name(prefix.name + "_RANK_V5.md")
    rank_v6_json = prefix.with_name(prefix.name + "_RANK_V6.json")
    rank_v6_md = prefix.with_name(prefix.name + "_RANK_V6.md")
    out_json = prefix.with_suffix(".json")
    out_md = prefix.with_suffix(".md")

    rc = _run_tool(
        "tools.liquidation_rule_coverage",
        [
            "--db",
            str(args.db),
            "--symbol",
            str(args.symbol),
            "--lookbacks-min",
            str(args.coverage_lookbacks_min),
            "--bucket-sec",
            str(int(args.bucket_sec)),
            "--rule",
            str(args.rule),
            "--out-json",
            str(coverage_json),
            "--out-md",
            str(coverage_md),
        ],
    )
    if rc != 0:
        return rc

    rc = _run_tool(
        "tools.generate_liq_reversal_candidates",
        [
            "--symbols",
            str(args.symbol),
            "--horizons-sec",
            str(args.horizons_sec),
            "--min-imbalances",
            str(args.min_imbalances),
            "--min-trade-intensities",
            str(args.min_trade_intensities),
            "--max-spreads",
            str(args.max_spreads),
            "--rule",
            str(args.rule),
            "--out-json",
            str(candidates_json),
            "--out-md",
            str(candidates_md),
        ],
    )
    if rc != 0:
        return rc

    common_rank_args = [
        "--db",
        str(args.db),
        "--lookback-min",
        str(int(args.lookback_min)),
        "--bucket-sec",
        str(int(args.bucket_sec)),
        "--candidates-md",
        str(candidates_md),
        "--rule",
        str(args.rule),
        "--side",
        "auto",
        "--splits",
        str(int(args.splits)),
        "--seeds",
        str(args.seeds),
        "--min-n",
        str(int(args.min_n)),
        "--min-n-frac",
        str(float(args.min_n_frac)),
        "--min-attempt-fill-rate",
        "0.0",
    ]

    rc = _run_tool(
        "tools.rank_passive_pockets_forward",
        [
            *common_rank_args,
            "--mitigation-profile",
            "baseline",
            "--out-json",
            str(rank_baseline_json),
            "--out-md",
            str(rank_baseline_md),
        ],
    )
    if rc != 0:
        return rc

    rc = _run_tool(
        "tools.rank_passive_pockets_forward",
        [
            *common_rank_args,
            "--mitigation-profile",
            "anti_adverse_v5",
            "--out-json",
            str(rank_v5_json),
            "--out-md",
            str(rank_v5_md),
        ],
    )
    if rc != 0:
        return rc

    rc = _run_tool(
        "tools.rank_passive_pockets_forward",
        [
            *common_rank_args,
            "--mitigation-profile",
            "anti_adverse_v6",
            "--out-json",
            str(rank_v6_json),
            "--out-md",
            str(rank_v6_md),
        ],
    )
    if rc != 0:
        return rc

    coverage = _load_json(coverage_json)
    candidates = _load_json(candidates_json)
    rank_baseline = _load_json(rank_baseline_json)
    rank_v5 = _load_json(rank_v5_json)
    rank_v6 = _load_json(rank_v6_json)

    summary = {
        "symbol": str(args.symbol).upper(),
        "rule": str(args.rule),
        "coverage": {
            "windows": int(len(coverage.get("results", []) or [])),
            "max_rule_fire_count": int(max((int(r.get("rule_fire_count", 0) or 0) for r in coverage.get("results", []) or []), default=0)),
            "max_rule_given_liq_rate": float(max((float(r.get("rule_given_liq_rate", 0.0) or 0.0) for r in coverage.get("results", []) or []), default=0.0)),
        },
        "candidate_surface": {
            "count": int(candidates.get("count", 0) or 0),
        },
        "rank_baseline": _rank_summary(rank_baseline),
        "rank_v5": _rank_summary(rank_v5),
        "rank_v6": _rank_summary(rank_v6),
        "decision": {
            "baseline_tradeable": bool(int(rank_baseline.get("count", 0) or 0) > 0),
            "v5_tradeable": bool(int(rank_v5.get("count", 0) or 0) > 0),
            "v6_tradeable": bool(int(rank_v6.get("count", 0) or 0) > 0),
            "next_step": (
                "change_execution_style"
                if int(rank_baseline.get("count", 0) or 0) == 0 and int(rank_v5.get("count", 0) or 0) == 0 and int(rank_v6.get("count", 0) or 0) == 0
                else "inspect_ranked_pockets"
            ),
        },
    }

    payload = {
        "symbol": str(args.symbol).upper(),
        "rule": str(args.rule),
        "lookback_min": int(args.lookback_min),
        "bucket_sec": int(args.bucket_sec),
        "coverage_json": str(coverage_json),
        "candidates_json": str(candidates_json),
        "rank_baseline_json": str(rank_baseline_json),
        "rank_v5_json": str(rank_v5_json),
        "rank_v6_json": str(rank_v6_json),
        "summary": summary,
    }
    payload["run_summary"] = build_run_summary(
        run_type="run_liq_reversal_e2e",
        inputs={
            "db": str(args.db),
            "symbol": str(args.symbol).upper(),
            "lookback_min": int(args.lookback_min),
            "coverage_lookbacks_min": _parse_list(args.coverage_lookbacks_min),
            "bucket_sec": int(args.bucket_sec),
            "horizons_sec": _parse_list(args.horizons_sec),
            "min_imbalances": _parse_list(args.min_imbalances),
            "min_trade_intensities": _parse_list(args.min_trade_intensities),
            "max_spreads": _parse_list(args.max_spreads),
            "splits": int(args.splits),
            "seeds": _parse_list(args.seeds),
            "min_n": int(args.min_n),
            "min_n_frac": float(args.min_n_frac),
            "rule": str(args.rule),
        },
        metrics={
            "coverage_windows": int(summary["coverage"]["windows"]),
            "candidate_count": int(summary["candidate_surface"]["count"]),
            "baseline_rank_count": int(summary["rank_baseline"]["count"]),
            "v5_rank_count": int(summary["rank_v5"]["count"]),
            "v6_rank_count": int(summary["rank_v6"]["count"]),
        },
        artifacts={
            "json": str(out_json),
            "md": str(out_md),
            "coverage_json": str(coverage_json),
            "candidates_json": str(candidates_json),
            "rank_baseline_json": str(rank_baseline_json),
            "rank_v5_json": str(rank_v5_json),
            "rank_v6_json": str(rank_v6_json),
        },
    )

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# LIQUIDATION REVERSAL E2E",
        "",
        f"symbol={payload['symbol']} rule={payload['rule']} lookback_min={payload['lookback_min']} bucket_sec={payload['bucket_sec']}",
        "",
        f"coverage_windows={summary['coverage']['windows']} max_rule_fire_count={summary['coverage']['max_rule_fire_count']} max_rule_given_liq_rate={summary['coverage']['max_rule_given_liq_rate']:.2%}",
        f"candidate_count={summary['candidate_surface']['count']}",
        f"baseline_rank_count={summary['rank_baseline']['count']}",
        f"v5_rank_count={summary['rank_v5']['count']}",
        f"v6_rank_count={summary['rank_v6']['count']}",
        f"next_step={summary['decision']['next_step']}",
        "",
        "## Top Results",
        "",
        f"- baseline_top={summary['rank_baseline']['top']}",
        f"- v5_top={summary['rank_v5']['top']}",
        f"- v6_top={summary['rank_v6']['top']}",
        "",
        "## Artifacts",
        f"- coverage_json={coverage_json}",
        f"- candidates_json={candidates_json}",
        f"- rank_baseline_json={rank_baseline_json}",
        f"- rank_v5_json={rank_v5_json}",
        f"- rank_v6_json={rank_v6_json}",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
