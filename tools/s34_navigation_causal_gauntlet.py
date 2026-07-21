"""Causal holdout gauntlet for S34 navigation KNN patterns.

This is stricter than the descriptive/meta gauntlet:
- calibration rows are historical;
- holdout rows are classified using only calibration neighbors;
- permutation recomputes holdout KNN labels from permuted calibration outcomes and
  evaluates max-stat over the holdout cell universe.

Research-only. No live executor, order logic, size, leverage, or .env changes.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_navigation_full_followup import (  # noqa: E402
    FEE_BPS,
    NAV_EVENTS,
    classify_neighbor,
    distance,
    feature_vector,
    load_jsonl,
    r1,
    r3,
    summary,
)
from tools.s34_navigation_meta_patterns import ILLEGAL_PRE_ENTRY_TAGS  # noqa: E402

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_CAUSAL_GAUNTLET.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_CAUSAL_GAUNTLET.md"

KS = [5, 8, 10, 12, 15, 20]
PREDS = ["CLEAN", "DANGER", "MIXED"]
MIN_CELL_N = 40
PERMUTATIONS = 1000
SEED = 34030


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ts(row: dict[str, Any]) -> int:
    return int(row.get("signal_ts_ms") or 0)


def normal_value(row: dict[str, Any]) -> float:
    return float(row.get("net_2h_bps") if row.get("net_2h_bps") is not None else row.get("normal_2h_bps"))


def value_from_normal(v: float, direction: str) -> float:
    return float(v) if direction == "NORMAL" else -float(v) - 2.0 * FEE_BPS


def t3r(vals: list[float]) -> float:
    if len(vals) <= 3:
        return float(sum(vals))
    return float(sum(sorted(vals, reverse=True)[3:]))


def stat(vals: list[float]) -> float:
    if len(vals) < MIN_CELL_N:
        return float("-inf")
    return t3r(vals)


def classify_from_values(vals: list[float]) -> str:
    return classify_neighbor(summary(vals), "base")


def prepare(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted([r for r in rows if r.get("net_2h_bps") is not None], key=ts)
    split_idx = int(len(ordered) * 0.7)
    cal = ordered[:split_idx]
    hold = ordered[split_idx:]
    cal_vecs = [feature_vector(r) for r in cal]
    hold_vecs = [feature_vector(r) for r in hold]

    # Top 20 calibration neighbors for each calibration row (leave-one-out) and holdout row.
    cal_neighbors: list[list[int]] = []
    for i, v in enumerate(cal_vecs):
        ds = [(distance(v, other), j) for j, other in enumerate(cal_vecs) if j != i]
        cal_neighbors.append([j for _, j in sorted(ds, key=lambda x: x[0])[: max(KS)]])

    hold_neighbors: list[list[int]] = []
    for v in hold_vecs:
        ds = [(distance(v, other), j) for j, other in enumerate(cal_vecs)]
        hold_neighbors.append([j for _, j in sorted(ds, key=lambda x: x[0])[: max(KS)]])

    return {
        "ordered": ordered,
        "cal": cal,
        "hold": hold,
        "split_idx": split_idx,
        "split_ts_ms": ts(ordered[split_idx]),
        "cal_neighbors": cal_neighbors,
        "hold_neighbors": hold_neighbors,
    }


def causal_preds(neighbors: list[list[int]], cal_normals: list[float]) -> list[dict[str, str]]:
    preds = []
    for nns in neighbors:
        item = {}
        for k in KS:
            vals = [cal_normals[j] for j in nns[:k]]
            item[f"k{k}"] = classify_from_values(vals)
        preds.append(item)
    return preds


def attach(items: list[dict[str, Any]], preds: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for row, pred in zip(items, preds):
        r = dict(row)
        r["preds"] = pred
        r["normal_2h_bps"] = normal_value(row)
        r["reverse_2h_bps"] = value_from_normal(normal_value(row), "REVERSE")
        out.append(r)
    return out


def cell_indices(rows: list[dict[str, Any]], cell_name: str) -> tuple[list[int], str]:
    parts = cell_name.split("_")
    direction = parts[-1]
    if cell_name.startswith("k5_to_k20_"):
        left = parts[3]
        right = parts[5]
        idxs = [i for i, r in enumerate(rows) if r["preds"].get("k5") == left and r["preds"].get("k20") == right]
        return idxs, direction
    if cell_name.startswith("k20_") and "_thr" in cell_name:
        pred = parts[1]
        threshold = int(parts[2].replace("thr", ""))
        idxs = [
            i for i, r in enumerate(rows)
            if r["preds"].get("k20") == pred and int(float(r.get("threshold_usd") or 0)) == threshold
        ]
        return idxs, direction
    if cell_name.startswith("k"):
        k = parts[0]
        pred = parts[1]
        idxs = [i for i, r in enumerate(rows) if r["preds"].get(k) == pred]
        return idxs, direction
    raise ValueError(cell_name)


def vals_for_cell(rows: list[dict[str, Any]], cell_name: str) -> list[float]:
    idxs, direction = cell_indices(rows, cell_name)
    return [value_from_normal(normal_value(rows[i]), direction) for i in idxs]


def build_universe(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    for k in KS:
        for pred in PREDS:
            for direction in ("NORMAL", "REVERSE"):
                cells.append((f"k{k}_{pred}_{direction}", direction))
    for left in PREDS:
        for right in PREDS:
            for direction in ("NORMAL", "REVERSE"):
                cells.append((f"k5_to_k20_{left}_to_{right}_{direction}", direction))
    for pred in PREDS:
        for threshold in (50_000, 100_000, 200_000):
            for direction in ("NORMAL", "REVERSE"):
                cells.append((f"k20_{pred}_thr{threshold}_{direction}", direction))
    tags = sorted({str(t) for r in rows for t in (r.get("tags") or []) if str(t) not in ILLEGAL_PRE_ENTRY_TAGS})
    for size in (1, 2):
        for combo in combinations(tags, size):
            for direction in ("NORMAL", "REVERSE"):
                cells.append((f"tags_{'+'.join(combo)}_{direction}", direction))
    return cells


def vals_for_universe_cell(rows: list[dict[str, Any]], name: str) -> list[float]:
    if name.startswith("tags_"):
        body, direction = name[5:].rsplit("_", 1)
        tags = set(body.split("+"))
        vals = []
        for row in rows:
            row_tags = {str(t) for t in (row.get("tags") or []) if str(t) not in ILLEGAL_PRE_ENTRY_TAGS}
            if tags.issubset(row_tags):
                vals.append(value_from_normal(normal_value(row), direction))
        return vals
    return vals_for_cell(rows, name)


def candidate_names() -> list[str]:
    return [
        "k5_DANGER_REVERSE",
        "k5_CLEAN_NORMAL",
        "k8_DANGER_REVERSE",
        "k10_DANGER_REVERSE",
        "k5_to_k20_CLEAN_to_DANGER_NORMAL",
        "k20_DANGER_REVERSE",
        "k20_DANGER_thr100000_REVERSE",
        "k20_DANGER_thr200000_REVERSE",
        "k20_DANGER_thr50000_REVERSE",
    ]


def permutation_holdout(prep: dict[str, Any], observed_hold: list[dict[str, Any]], universe: list[tuple[str, str]]) -> dict[str, Any]:
    rng = random.Random(SEED)
    cal_normals = [normal_value(r) for r in prep["cal"]]
    hold_normals = [normal_value(r) for r in prep["hold"]]

    observed_stats = {}
    for name, _ in universe:
        vals = vals_for_universe_cell(observed_hold, name)
        if len(vals) >= MIN_CELL_N:
            observed_stats[name] = stat(vals)

    raw_counts = defaultdict(int)
    max_counts = defaultdict(int)
    candidate_set = set(candidate_names())
    max_null = []

    for _ in range(PERMUTATIONS):
        perm_cal = cal_normals[:]
        perm_hold = hold_normals[:]
        rng.shuffle(perm_cal)
        rng.shuffle(perm_hold)
        pred_hold = causal_preds(prep["hold_neighbors"], perm_cal)
        perm_rows = []
        for row, pred, value in zip(prep["hold"], pred_hold, perm_hold):
            r = dict(row)
            r["preds"] = pred
            r["net_2h_bps"] = value
            r["normal_2h_bps"] = value
            r["reverse_2h_bps"] = value_from_normal(value, "REVERSE")
            perm_rows.append(r)

        max_s = float("-inf")
        perm_stats = {}
        for name, _ in universe:
            vals = vals_for_universe_cell(perm_rows, name)
            s = stat(vals)
            perm_stats[name] = s
            if s > max_s:
                max_s = s
        max_null.append(max_s)
        for name in candidate_set:
            obs = observed_stats.get(name, float("-inf"))
            if perm_stats.get(name, float("-inf")) >= obs:
                raw_counts[name] += 1
            if max_s >= obs:
                max_counts[name] += 1

    out = {}
    for name in candidate_names():
        obs = observed_stats.get(name)
        if obs is None:
            out[name] = {"observed_hold_t3r": None, "raw_p": None, "mc_maxstat_p": None}
            continue
        out[name] = {
            "observed_hold_t3r": r1(obs),
            "raw_p": r3((raw_counts[name] + 1) / (PERMUTATIONS + 1)),
            "mc_maxstat_p": r3((max_counts[name] + 1) / (PERMUTATIONS + 1)),
        }
    return {
        "permutations": PERMUTATIONS,
        "seed": SEED,
        "holdout_cell_universe_n": len(universe),
        "max_null_p95": r1(sorted(max_null)[int(0.95 * len(max_null))]),
        "max_null_p99": r1(sorted(max_null)[int(0.99 * len(max_null))]),
        "by_candidate": out,
    }


def candidate_report(cal_rows: list[dict[str, Any]], hold_rows: list[dict[str, Any]], perm: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for name in candidate_names():
        cal_vals = vals_for_cell(cal_rows, name)
        hold_vals = vals_for_cell(hold_rows, name)
        p = perm["by_candidate"].get(name, {})
        pass_ = (
            len(cal_vals) >= MIN_CELL_N
            and len(hold_vals) >= MIN_CELL_N
            and t3r(cal_vals) > 0
            and t3r(hold_vals) > 0
            and p.get("mc_maxstat_p") is not None
            and float(p["mc_maxstat_p"]) <= 0.05
        )
        out.append(
            {
                "candidate": name,
                "status": "PASS" if pass_ else "FAIL",
                "cal": summary(cal_vals),
                "hold": summary(hold_vals),
                "permutation_holdout": p,
            }
        )
    return out


def fmt(s: dict[str, Any]) -> str:
    return (
        f"N={s.get('n')} sum={s.get('sum_bps')} med={s.get('median_bps')} "
        f"T3R={s.get('t3r_bps')} tail150={s.get('tail_lte_minus150_n')} maxLoss={s.get('max_loss_bps')}"
    )


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# S34 Navigation Causal Gauntlet",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        "Holdout rows are classified using calibration neighbors only. This is stricter than the descriptive full-sample navigation report.",
        "",
        f"Split: cal `{result['cal_n']}`, hold `{result['hold_n']}`; holdout cell universe `{result['permutation']['holdout_cell_universe_n']}`",
        f"Permutations: `{result['permutation']['permutations']}`; holdout max-null p95 T3R `{result['permutation']['max_null_p95']}`, p99 `{result['permutation']['max_null_p99']}`",
        "",
        "| Candidate | Status | Cal | Hold | Hold raw p | Hold MC p |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in result["candidates"]:
        p = row["permutation_holdout"]
        lines.append(
            f"| {row['candidate']} | {row['status']} | {fmt(row['cal'])} | {fmt(row['hold'])} | "
            f"{p.get('raw_p')} | {p.get('mc_maxstat_p')} |"
        )
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_jsonl(NAV_EVENTS)
    prep = prepare(rows)
    cal_normals = [normal_value(r) for r in prep["cal"]]
    cal_pred = causal_preds(prep["cal_neighbors"], cal_normals)
    hold_pred = causal_preds(prep["hold_neighbors"], cal_normals)
    cal_rows = attach(prep["cal"], cal_pred)
    hold_rows = attach(prep["hold"], hold_pred)
    universe = build_universe(hold_rows)
    perm = permutation_holdout(prep, hold_rows, universe)
    result = {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "cal_n": len(cal_rows),
        "hold_n": len(hold_rows),
        "split_ts_ms": prep["split_ts_ms"],
        "candidates": candidate_report(cal_rows, hold_rows, perm),
        "permutation": perm,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(result)
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
