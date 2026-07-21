"""S34 v5 permutation-null and artifact detector.

Research-only. Builds a reusable gauntlet for candidate signals:
non-overlap / holdout / real-cost / cross-asset / N gate / permutation-null.

This intentionally does not touch live, paper, runtime state, or executor files.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_funding_nonoverlap import Z_THRESHOLDS, build_trades


POOL_JSON = ROOT / "reports" / "research" / "s34" / "S34_ABSORPTION_SYNC_2X2_POOL.json"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V5_PERMUTATION_ARTIFACT_DETECTOR.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V5_PERMUTATION_ARTIFACT_DETECTOR.md"
DEFAULT_DB = ROOT / "data" / "microstructure.db"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def metrics(vals: list[float]) -> dict[str, Any]:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if not xs:
        return {"n": 0, "sum": 0.0, "mean": None, "median": None, "win_rate": None, "t3r": 0.0, "max_loss": None}
    ordered = sorted(xs, reverse=True)
    return {
        "n": len(xs),
        "sum": round(sum(xs), 1),
        "mean": round(sum(xs) / len(xs), 2),
        "median": round(median(xs), 2),
        "win_rate": round(sum(1 for v in xs if v > 0.0) / len(xs), 3),
        "t3r": round(sum(ordered[3:]) if len(ordered) > 3 else sum(xs), 1),
        "max_loss": round(min(xs), 1),
    }


def pctile(vals: list[float], q: float) -> float | None:
    xs = sorted(v for v in vals if math.isfinite(v))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def month_split(rows: list[dict[str, Any]], hold_months: set[str] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []
    months = sorted({str(r.get("month")) for r in rows if r.get("month") is not None})
    if hold_months is None:
        hold_n = max(1, len(months) // 3)
        hold_months = set(months[-hold_n:])
    cal = [r for r in rows if str(r.get("month")) not in hold_months]
    hold = [r for r in rows if str(r.get("month")) in hold_months]
    return cal, hold


def split_by_time(rows: list[dict[str, Any]], holdout_frac: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sub = sorted(rows, key=lambda r: int(r.get("ts_ms") or r.get("entry_ts_ms") or 0))
    if not sub:
        return [], []
    cut_idx = max(0, min(len(sub) - 1, int(len(sub) * (1.0 - holdout_frac))))
    cut_ts = int(sub[cut_idx].get("ts_ms") or sub[cut_idx].get("entry_ts_ms") or 0)
    return [r for r in sub if int(r.get("ts_ms") or r.get("entry_ts_ms") or 0) < cut_ts], [
        r for r in sub if int(r.get("ts_ms") or r.get("entry_ts_ms") or 0) >= cut_ts
    ]


def label_shuffle_null(
    population_vals: list[float],
    selected_n: int,
    *,
    rng: random.Random,
    n_perm: int,
) -> list[float]:
    if selected_n <= 0 or selected_n > len(population_vals):
        return []
    out = []
    for _ in range(n_perm):
        out.append(sum(rng.sample(population_vals, selected_n)))
    return out


def sign_flip_null(vals: list[float], *, rng: random.Random, n_perm: int) -> list[float]:
    out = []
    for _ in range(n_perm):
        out.append(sum(v if rng.random() >= 0.5 else -v for v in vals))
    return out


def p_right(null_vals: list[float], real: float) -> float | None:
    if not null_vals:
        return None
    return (sum(1 for v in null_vals if v >= real) + 1.0) / (len(null_vals) + 1.0)


@dataclass
class Candidate:
    name: str
    family: str
    rows: list[dict[str, Any]]
    value_key: str
    filter_fn: Callable[[dict[str, Any]], bool]
    null_mode: str
    non_overlap: bool
    real_cost: bool
    beta_control: str
    min_n: int = 40


def evaluate_candidate(candidate: Candidate, *, rng: random.Random, n_perm: int) -> dict[str, Any]:
    cal_rows, hold_rows = month_split(candidate.rows)
    if not hold_rows:
        cal_rows, hold_rows = split_by_time(candidate.rows, 0.30)
    hold_selected = [r for r in hold_rows if candidate.filter_fn(r) and finite(r.get(candidate.value_key)) is not None]
    cal_selected = [r for r in cal_rows if candidate.filter_fn(r) and finite(r.get(candidate.value_key)) is not None]
    hold_vals = [float(r[candidate.value_key]) for r in hold_selected]
    cal_vals = [float(r[candidate.value_key]) for r in cal_selected]
    pop_vals = [float(r[candidate.value_key]) for r in hold_rows if finite(r.get(candidate.value_key)) is not None]
    real = sum(hold_vals)
    if candidate.null_mode == "label_shuffle":
        null_vals = label_shuffle_null(pop_vals, len(hold_vals), rng=rng, n_perm=n_perm)
    elif candidate.null_mode == "sign_flip":
        null_vals = sign_flip_null(hold_vals, rng=rng, n_perm=n_perm)
    else:
        raise ValueError(f"unknown null_mode {candidate.null_mode}")
    null_p95 = pctile(null_vals, 0.95)
    pval = p_right(null_vals, real)
    symbols = sorted({str(r.get("symbol")) for r in hold_selected if r.get("symbol")})
    by_symbol = {
        sym: metrics([float(r[candidate.value_key]) for r in hold_selected if str(r.get("symbol")) == sym])
        for sym in symbols
    }
    cross_asset_ok = len([sym for sym, m in by_symbol.items() if int(m["n"]) >= 20 and float(m["sum"]) > 0.0]) >= 2
    n_ok = len(hold_vals) >= candidate.min_n
    hold_positive = real > 0.0 and metrics(hold_vals)["t3r"] > 0.0
    perm_ok = pval is not None and pval <= 0.05 and null_p95 is not None and real > null_p95
    cal_ok = sum(cal_vals) > 0.0 and metrics(cal_vals)["t3r"] > 0.0
    gauntlet = {
        "non_overlap": bool(candidate.non_overlap),
        "holdout_positive_sum_t3r": bool(hold_positive),
        "cal_positive_sum_t3r": bool(cal_ok),
        "real_cost": bool(candidate.real_cost),
        "cross_asset": bool(cross_asset_ok),
        "beta_control": candidate.beta_control,
        "n_ge_min": bool(n_ok),
        "permutation_p_le_0_05": bool(perm_ok),
    }
    verdict = "PASS" if all(v is True or v == "PASS" for v in gauntlet.values()) else "ARTIFACT"
    return {
        "name": candidate.name,
        "family": candidate.family,
        "verdict": verdict,
        "null_mode": candidate.null_mode,
        "min_n": candidate.min_n,
        "cal": metrics(cal_vals),
        "hold": metrics(hold_vals),
        "hold_population": metrics(pop_vals),
        "by_symbol_hold": by_symbol,
        "permutation": {
            "n_perm": n_perm,
            "real_sum": round(real, 1),
            "null_p50": round(pctile(null_vals, 0.50), 1) if null_vals else None,
            "null_p95": round(null_p95, 1) if null_p95 is not None else None,
            "p_right": round(pval, 4) if pval is not None else None,
        },
        "gauntlet": gauntlet,
    }


def load_pool_candidates(pool_path: Path) -> list[Candidate]:
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    hold_months = set((payload.get("split") or {}).get("holdout_months") or [])
    for r in rows:
        if "month" not in r and r.get("entry_ts_ms"):
            r["month"] = datetime.fromtimestamp(int(r["entry_ts_ms"]) / 1000, tz=timezone.utc).strftime("%Y-%m")
        if hold_months and str(r.get("month")) in hold_months:
            r["_forced_hold"] = True
    return [
        Candidate(
            name="cascade_fade_all_signflip",
            family="cascade",
            rows=rows,
            value_key="net_bps",
            filter_fn=lambda r: True,
            null_mode="sign_flip",
            non_overlap=False,
            real_cost=True,
            beta_control="FAIL:not_beta_controlled",
            min_n=40,
        ),
        Candidate(
            name="sync_gate_label_shuffle",
            family="cascade_sync",
            rows=rows,
            value_key="net_bps",
            filter_fn=lambda r: str(r.get("sync_gate")) == "sync",
            null_mode="label_shuffle",
            non_overlap=False,
            real_cost=True,
            beta_control="FAIL:not_beta_controlled",
            min_n=40,
        ),
        Candidate(
            name="deep_bid_absorption_label_shuffle",
            family="cascade_absorption",
            rows=rows,
            value_key="net_bps",
            filter_fn=lambda r: str(r.get("bid_depth_gate")) == "deep_bid",
            null_mode="label_shuffle",
            non_overlap=False,
            real_cost=True,
            beta_control="FAIL:not_beta_controlled",
            min_n=40,
        ),
        Candidate(
            name="sync_plus_deep_bid_label_shuffle",
            family="cascade_confluence",
            rows=rows,
            value_key="net_bps",
            filter_fn=lambda r: str(r.get("sync_gate")) == "sync" and str(r.get("bid_depth_gate")) == "deep_bid",
            null_mode="label_shuffle",
            non_overlap=False,
            real_cost=True,
            beta_control="FAIL:not_beta_controlled",
            min_n=40,
        ),
    ]


def load_funding_candidates(db: Path, cost: float) -> list[Candidate]:
    out: list[Candidate] = []
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        for z_thr in Z_THRESHOLDS:
            rows = []
            for sym in SYMBOLS:
                for t in build_trades(conn, sym, z_thr, cost):
                    rows.append({"symbol": sym, "ts_ms": t["ts_ms"], "side": t["side"], "z": t["z"], "net": t["net"]})
            out.append(
                Candidate(
                    name=f"funding_nonoverlap_z{z_thr:g}_all_signflip",
                    family="funding_nonoverlap",
                    rows=rows,
                    value_key="net",
                    filter_fn=lambda r: True,
                    null_mode="sign_flip",
                    non_overlap=True,
                    real_cost=True,
                    beta_control="FAIL:side_split_not_consistent",
                    min_n=40,
                )
            )
            out.append(
                Candidate(
                    name=f"funding_nonoverlap_z{z_thr:g}_eth_only_signflip",
                    family="funding_nonoverlap",
                    rows=rows,
                    value_key="net",
                    filter_fn=lambda r: str(r.get("symbol")) == "ETHUSDT",
                    null_mode="sign_flip",
                    non_overlap=True,
                    real_cost=True,
                    beta_control="FAIL:single_asset",
                    min_n=40,
                )
            )
    return out


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 v5 Permutation Null + Artifact Detector",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "`RESEARCH_ONLY_NO_LIVE_NO_PAPER` - no live executor, runtime state, or paper state was touched.",
        "",
        f"- permutations per candidate: `{report['config']['n_perm']}`",
        f"- seed: `{report['config']['seed']}`",
        "",
        "## Verdict Summary",
        "",
        "| Candidate | Verdict | Null | Hold | Null p95 | p-right | Gauntlet failures |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for r in report["candidates"]:
        failures = [k for k, v in r["gauntlet"].items() if not (v is True or v == "PASS")]
        lines.append(
            f"| `{r['name']}` | `{r['verdict']}` | `{r['null_mode']}` | "
            f"N={r['hold']['n']} sum={r['hold']['sum']} T3R={r['hold']['t3r']} | "
            f"{r['permutation']['null_p95']} | {r['permutation']['p_right']} | `{', '.join(failures) or 'none'}` |"
        )
    lines.extend(["", "## Detail", ""])
    for r in report["candidates"]:
        lines.extend(
            [
                f"### {r['name']}",
                "",
                f"- verdict: `{r['verdict']}`",
                f"- family: `{r['family']}`",
                f"- cal: `{r['cal']}`",
                f"- hold: `{r['hold']}`",
                f"- permutation: `{r['permutation']}`",
                f"- gauntlet: `{r['gauntlet']}`",
                f"- by_symbol_hold: `{r['by_symbol_hold']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Read",
            "",
            "- `label_shuffle` asks whether a filter selects better outcomes than a same-size random subset of holdout events.",
            "- `sign_flip` asks whether the selected P&L is larger than a no-directional-edge sign-randomized null.",
            "- A permutation win alone is not enough: the gauntlet also requires non-overlap, holdout, cost, cross-asset, beta control, and N discipline.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Run S34 v5 permutation-null artifact detector.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--pool-json", type=Path, default=POOL_JSON)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--seed", type=int, default=3405)
    p.add_argument("--funding-cost-bps-rt", type=float, default=8.0)
    args = p.parse_args()
    rng = random.Random(int(args.seed))
    candidates = load_pool_candidates(args.pool_json) + load_funding_candidates(args.db, float(args.funding_cost_bps_rt))
    results = [evaluate_candidate(c, rng=rng, n_perm=int(args.n_perm)) for c in candidates]
    report = {
        "generated_at_utc": utc_now(),
        "mode": "RESEARCH_ONLY_NO_LIVE_NO_PAPER",
        "config": {
            "n_perm": int(args.n_perm),
            "seed": int(args.seed),
            "pool_json": str(args.pool_json),
            "funding_cost_bps_rt": float(args.funding_cost_bps_rt),
        },
        "candidates": results,
        "summary": {
            "pass": sum(1 for r in results if r["verdict"] == "PASS"),
            "artifact": sum(1 for r in results if r["verdict"] == "ARTIFACT"),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md = render_md(report)
    args.out_md.write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
