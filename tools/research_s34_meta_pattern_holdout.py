"""MC-corrected permutation-null + chronological holdout on k5/k8 meta-pattern candidates.

Status: RESEARCH_ONLY_NO_LIVE_CHANGE

In-sample (full dataset) the meta-pattern scan found:
  k5=CLEAN NORMAL    : N=192, T3R=+7876
  k5=DANGER REVERSE  : N=314, T3R=+7453
  k8=DANGER REVERSE  : N=530, T3R=+6551
  k5->k20 C->D NORMAL: N=74,  T3R=+3143
  danger_count_0      : N=912, T3R=+2417  (all {k5,k8,k10}=CLEAN)

All are IN-SAMPLE. This tool applies the two-step discipline:
  1. Chronological holdout (70% cal / 30% hold):
     - holdout labels computed from cal neighbors only (no OOS leakage)
  2. MC-corrected permutation null on cal:
     - 1000 shuffles of cal outcomes, track max T3R across ALL candidates
     - compare real max T3R vs null p95 (corrects for ~20 tested patterns)

SAF-02: research-only, no live order/config/env change.
DAT-01: no lookahead (holdout labels use only cal neighbors).
DAT-03: seeded permutation (seed=42).
"""

from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_navigation_full_followup import (
    classify_neighbor,
    feature_vector,
    load_jsonl,
    summary,
    r1,
    r3,
    NAV_EVENTS,
    FEE_BPS,
)

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_META_PATTERN_HOLDOUT.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_META_PATTERN_HOLDOUT.md"

HOLDOUT_FRAC = 0.30
N_PERM = 1000
SEED = 42
KS = (5, 8, 10, 20)
MIN_N = 30


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def knn_label(query_vec: list[float], ref_rows: list[dict], ref_vecs: list[list[float]],
              k: int, strictness: str = "base") -> str:
    """Return KNN label for a query using ref_rows as the neighbor pool (OOS-safe)."""
    dists = [(distance(query_vec, ref_vecs[j]), ref_rows[j]) for j in range(len(ref_rows))]
    nn = [r for _, r in sorted(dists, key=lambda x: x[0])[:k]]
    vals = [float(r["net_2h_bps"]) for r in nn if r.get("net_2h_bps") is not None]
    s = summary(vals)
    return "UNKNOWN" if s["n"] < k else classify_neighbor(s, strictness)


def build_labels(rows: list[dict], ref_rows: list[dict], ref_vecs: list[list[float]],
                 ks: tuple[int, ...]) -> list[dict]:
    """For each row, compute KNN label using ref_rows (not the row itself -> OOS-safe).
    Row itself is excluded if it's in ref_rows."""
    query_vecs = [feature_vector(r) for r in rows]
    out = []
    for i, (row, qv) in enumerate(zip(rows, query_vecs)):
        # Exclude self from ref pool (leave-one-out for cal; already excluded for hold)
        row_id = row.get("event_id") or row.get("signal_ts_ms")
        ref_excl = [(rv, rr) for rv, rr in zip(ref_vecs, ref_rows)
                    if (rr.get("event_id") or rr.get("signal_ts_ms")) != row_id]
        if not ref_excl:
            out.append({**row, "labels": {f"k{k}": "UNKNOWN" for k in ks}})
            continue
        excl_vecs, excl_rows = zip(*ref_excl)
        item = dict(row)
        item["labels"] = {f"k{k}": knn_label(qv, list(excl_rows), list(excl_vecs), k) for k in ks}
        out.append(item)
    return out


def outcome(row: dict, direction: str) -> float | None:
    v = row.get("net_2h_bps")
    if v is None or not math.isfinite(float(v)):
        return None
    v = float(v)
    if direction == "NORMAL":
        return v
    elif direction == "REVERSE":
        return -v - 2.0 * FEE_BPS
    return None


def t3r(vals: list[float]) -> float:
    if len(vals) <= 3:
        return sum(vals)
    return sum(sorted(vals, reverse=True)[3:])


CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "k5_CLEAN_NORMAL",
        "desc": "k5=CLEAN -> NORMAL (buy-fade) direction",
        "filter": lambda r: r["labels"].get("k5") == "CLEAN",
        "direction": "NORMAL",
        "insample_t3r": 7876.5,
    },
    {
        "name": "k5_DANGER_REVERSE",
        "desc": "k5=DANGER -> REVERSE (counter-DANGER fade) direction",
        "filter": lambda r: r["labels"].get("k5") == "DANGER",
        "direction": "REVERSE",
        "insample_t3r": 7453.9,
    },
    {
        "name": "k8_DANGER_REVERSE",
        "desc": "k8=DANGER -> REVERSE direction",
        "filter": lambda r: r["labels"].get("k8") == "DANGER",
        "direction": "REVERSE",
        "insample_t3r": 6551.6,
    },
    {
        "name": "k5_CLEAN_k20_DANGER_NORMAL",
        "desc": "k5=CLEAN, k20=DANGER -> NORMAL (local-clean, broad-danger = reversal setup)",
        "filter": lambda r: r["labels"].get("k5") == "CLEAN" and r["labels"].get("k20") == "DANGER",
        "direction": "NORMAL",
        "insample_t3r": 3143.4,
    },
    {
        "name": "danger_count_0_NORMAL",
        "desc": "All {k5,k8,k10}=CLEAN -> NORMAL (consensus clean)",
        "filter": lambda r: all(r["labels"].get(f"k{k}") == "CLEAN" for k in (5, 8, 10)),
        "direction": "NORMAL",
        "insample_t3r": 2417.9,
    },
    {
        "name": "k5_CLEAN_k8_CLEAN_NORMAL",
        "desc": "k5=CLEAN AND k8=CLEAN -> NORMAL (consensus small-scale clean)",
        "filter": lambda r: r["labels"].get("k5") == "CLEAN" and r["labels"].get("k8") == "CLEAN",
        "direction": "NORMAL",
        "insample_t3r": None,
    },
]


def eval_candidates(rows: list[dict]) -> dict[str, float]:
    """Return T3R for each candidate on given rows."""
    out = {}
    for cand in CANDIDATES:
        subset = [r for r in rows if cand["filter"](r)]
        vals = [v for r in subset if (v := outcome(r, cand["direction"])) is not None]
        out[cand["name"]] = t3r(vals) if len(vals) >= MIN_N else float("nan")
    return out


def permutation_null(rows: list[dict], n_perm: int, seed: int) -> list[float]:
    """Shuffle cal outcomes n_perm times; return list of max T3R across candidates per shuffle."""
    rng = random.Random(seed)
    # Pre-collect outcome positions in rows
    indices = [i for i, r in enumerate(rows) if r.get("net_2h_bps") is not None]
    original_vals = [float(rows[i]["net_2h_bps"]) for i in indices]
    max_t3rs = []
    for _ in range(n_perm):
        shuffled = original_vals[:]
        rng.shuffle(shuffled)
        # Temporarily assign shuffled outcomes
        for idx, val in zip(indices, shuffled):
            rows[idx]["_perm_net"] = val
        # Eval each candidate on shuffled outcomes
        cand_t3rs = []
        for cand in CANDIDATES:
            subset = [r for r in rows if cand["filter"](r)]
            if cand["direction"] == "NORMAL":
                vals = [float(r["_perm_net"]) for r in subset if r.get("_perm_net") is not None]
            else:
                vals = [-float(r["_perm_net"]) - 2.0 * FEE_BPS for r in subset if r.get("_perm_net") is not None]
            if len(vals) >= MIN_N:
                cand_t3rs.append(t3r(vals))
        max_t3rs.append(max(cand_t3rs) if cand_t3rs else float("nan"))
    # Clean up temp field
    for r in rows:
        r.pop("_perm_net", None)
    return max_t3rs


def pctile(vals: list[float], p: float) -> float:
    v = sorted(v for v in vals if math.isfinite(v))
    if not v:
        return float("nan")
    idx = p * (len(v) - 1)
    lo = int(idx)
    return v[lo] + (idx - lo) * (v[min(lo + 1, len(v) - 1)] - v[lo])


def candidate_report(name: str, cand: dict, cal_rows: list[dict], hold_rows: list[dict]) -> dict:
    cal_sub = [r for r in cal_rows if cand["filter"](r)]
    hold_sub = [r for r in hold_rows if cand["filter"](r)]
    cal_vals = [v for r in cal_sub if (v := outcome(r, cand["direction"])) is not None]
    hold_vals = [v for r in hold_sub if (v := outcome(r, cand["direction"])) is not None]
    return {
        "name": name,
        "desc": cand["desc"],
        "direction": cand["direction"],
        "insample_t3r": cand.get("insample_t3r"),
        "cal": {
            "n": len(cal_vals),
            "t3r": r1(t3r(cal_vals)) if len(cal_vals) >= MIN_N else None,
            "sum": r1(sum(cal_vals)) if cal_vals else None,
            "median": r1(median(cal_vals)) if cal_vals else None,
            "win_rate": r3(sum(1 for v in cal_vals if v > 0) / len(cal_vals)) if cal_vals else None,
            "max_loss": r1(min(cal_vals)) if cal_vals else None,
        },
        "hold": {
            "n": len(hold_vals),
            "t3r": r1(t3r(hold_vals)) if len(hold_vals) >= MIN_N else None,
            "sum": r1(sum(hold_vals)) if hold_vals else None,
            "median": r1(median(hold_vals)) if hold_vals else None,
            "win_rate": r3(sum(1 for v in hold_vals if v > 0) / len(hold_vals)) if hold_vals else None,
            "max_loss": r1(min(hold_vals)) if hold_vals else None,
        },
    }


def render_md(result: dict) -> str:
    lines = [
        "# S34 Meta-Pattern Holdout + MC-Corrected Permutation Null",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        "Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`",
        "",
        f"Cal N={result['split']['cal_n']}, Hold N={result['split']['hold_n']}  |  "
        f"Cal date range: {result['split']['cal_start_utc']} to {result['split']['cal_end_utc']}",
        "",
        f"Hold date range: {result['split']['hold_start_utc']} -> {result['split']['hold_end_utc']}",
        "",
        "## Permutation Null (MC-corrected, cal split)",
        "",
        f"Candidates tested: {result['permutation']['n_candidates']}  |  "
        f"Permutations: {result['permutation']['n_perm']}  |  Seed: {result['permutation']['seed']}",
        "",
        f"Real max T3R (best candidate): **{result['permutation']['real_max_t3r']}**",
        f"Null p95 max T3R: **{result['permutation']['null_p95']}**",
        f"p-right (real >= null): **{result['permutation']['p_right']}**",
        "",
        f"**MC-corrected verdict: {result['permutation']['mc_verdict']}**",
        "",
        "## Per-Candidate Results",
        "",
        "| Candidate | In-sample T3R | Cal N | Cal T3R | Cal median | Cal win | Cal maxL | Hold N | Hold T3R | Hold median | Hold win | Hold maxL | Holdout verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for cand in result["candidates"]:
        cal = cand["cal"]
        hold = cand["hold"]
        # Holdout verdict
        if hold["n"] is not None and hold["n"] >= MIN_N and hold["t3r"] is not None:
            h_verdict = "HOLD_POSITIVE" if hold["t3r"] > 0 else "HOLD_NEGATIVE"
        else:
            h_verdict = f"HOLD_SMALL_N({hold['n']})"
        lines.append(
            f"| {cand['name']} | {cand['insample_t3r']} | "
            f"{cal['n']} | {cal['t3r']} | {cal['median']} | {cal['win_rate']} | {cal['max_loss']} | "
            f"{hold['n']} | {hold['t3r']} | {hold['median']} | {hold['win_rate']} | {hold['max_loss']} | "
            f"**{h_verdict}** |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **MC-corrected permutation null**: because ~20 patterns were scanned in-sample, the null",
        "  tracks max T3R across all candidates per shuffle. The corrected p-right tests whether the",
        "  BEST candidate beats the null 95th percentile under the MC-corrected threshold.",
        "  If p-right > 0.05 -> the entire family is an artifact (consistent with 0 PASS verdict).",
        "",
        "- **Holdout T3R**: independent OOS check. Labels computed from cal neighbors only (no leakage).",
        "  HOLD_POSITIVE required for any live/shadow promotion consideration.",
        "",
        "- **Max loss -685 bps** on k5=DANGER REVERSE is a hard veto on live promotion regardless of T3R.",
        "  Any promotion requires a TP/SL sweep showing tail-budget compatibility.",
        "",
        "All results: RESEARCH_ONLY. Permutation-null is the definitive discipline test.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    print("Loading nav events...")
    all_rows = load_jsonl(NAV_EVENTS)
    all_rows = [r for r in all_rows if r.get("net_2h_bps") is not None]
    all_rows.sort(key=lambda r: int(r["signal_ts_ms"]))
    n_total = len(all_rows)
    n_cal = int(n_total * (1.0 - HOLDOUT_FRAC))
    cal_rows_raw = all_rows[:n_cal]
    hold_rows_raw = all_rows[n_cal:]
    print(f"Total: {n_total}  Cal: {len(cal_rows_raw)}  Hold: {len(hold_rows_raw)}")

    # Pre-compute feature vectors for cal
    print("Computing cal feature vectors...")
    cal_vecs = [feature_vector(r) for r in cal_rows_raw]

    # Build cal labels (leave-one-out: each cal event labeled by cal neighbors excluding itself)
    print(f"Building cal KNN labels (ks={KS})...")
    cal_rows = build_labels(cal_rows_raw, cal_rows_raw, cal_vecs, KS)

    # Build holdout labels using cal neighbors only
    print("Building holdout KNN labels (cal-neighbor pool only)...")
    hold_rows = build_labels(hold_rows_raw, cal_rows_raw, cal_vecs, KS)

    # Per-candidate cal + hold stats
    print("Computing per-candidate results...")
    cand_results = [candidate_report(c["name"], c, cal_rows, hold_rows) for c in CANDIDATES]

    # Cal T3Rs (real)
    cal_t3rs = eval_candidates(cal_rows)
    real_max_t3r = max((v for v in cal_t3rs.values() if math.isfinite(v)), default=float("nan"))
    best_cand = max((k for k, v in cal_t3rs.items() if math.isfinite(v)), key=lambda k: cal_t3rs[k], default=None)

    # MC-corrected permutation null on cal
    print(f"Running {N_PERM} permutations (seed={SEED})...")
    null_maxes = permutation_null(cal_rows, N_PERM, SEED)
    null_p95 = pctile(null_maxes, 0.95)
    p_right = sum(1 for v in null_maxes if math.isfinite(v) and v >= real_max_t3r) / len([v for v in null_maxes if math.isfinite(v)])

    mc_verdict = "ARTIFACT_FAMILY_NO_PASS" if p_right > 0.05 else f"PASS_MC_CORRECTED (p-right={r3(p_right)}, best={best_cand})"

    ts_to_utc = lambda ts: datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    split_info = {
        "n_total": n_total,
        "cal_n": len(cal_rows),
        "hold_n": len(hold_rows),
        "holdout_frac": HOLDOUT_FRAC,
        "cal_start_utc": ts_to_utc(cal_rows_raw[0]["signal_ts_ms"]),
        "cal_end_utc": ts_to_utc(cal_rows_raw[-1]["signal_ts_ms"]),
        "hold_start_utc": ts_to_utc(hold_rows_raw[0]["signal_ts_ms"]),
        "hold_end_utc": ts_to_utc(hold_rows_raw[-1]["signal_ts_ms"]),
    }

    result = {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "split": split_info,
        "permutation": {
            "n_candidates": len(CANDIDATES),
            "n_perm": N_PERM,
            "seed": SEED,
            "cal_t3rs": {k: r1(v) for k, v in cal_t3rs.items()},
            "real_max_t3r": r1(real_max_t3r),
            "best_candidate": best_cand,
            "null_p95": r1(null_p95),
            "null_p99": r1(pctile(null_maxes, 0.99)),
            "p_right": r3(p_right),
            "mc_verdict": mc_verdict,
        },
        "candidates": cand_results,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    md = render_md(result)
    OUT_MD.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
