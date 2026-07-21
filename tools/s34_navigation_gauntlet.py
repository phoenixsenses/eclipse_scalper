"""S34 navigation candidate gauntlet.

Research-only gauntlet for the newly found navigation/meta-pattern leads:
- chronological holdout;
- real-cost net outcomes already embedded in the navigation ledger;
- multiple-comparison-corrected permutation null using max statistic over the
  searched cell universe.

No live executor, order logic, size, leverage, or .env changes.
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
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_navigation_full_followup import (  # noqa: E402
    FEE_BPS,
    NAV_EVENTS,
    knn_cards,
    load_jsonl,
    r1,
    r3,
    summary,
)
from tools.s34_navigation_meta_patterns import ILLEGAL_PRE_ENTRY_TAGS, key_for  # noqa: E402

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_GAUNTLET.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_GAUNTLET.md"

KS = [5, 8, 10, 12, 15, 20]
PREDS = ["CLEAN", "DANGER", "MIXED"]
MIN_CELL_N = 40
PERMUTATIONS = 1000
SEED = 34029


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ts(row: dict[str, Any]) -> int:
    return int(row.get("signal_ts_ms") or 0)


def t3r(vals: list[float]) -> float:
    if len(vals) <= 3:
        return float(sum(vals))
    return float(sum(sorted(vals, reverse=True)[3:]))


def stat(vals: list[float]) -> float:
    """Selection statistic used for the max-stat permutation null."""
    if len(vals) < MIN_CELL_N:
        return float("-inf")
    return t3r(vals)


def outcome(row: dict[str, Any], direction: str) -> float | None:
    v = row.get("normal_2h_bps")
    if v is None:
        v = row.get("net_2h_bps")
    if v is None:
        return None
    normal = float(v)
    if not math.isfinite(normal):
        return None
    if direction == "NORMAL":
        return normal
    return -normal - 2.0 * FEE_BPS


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    maps: dict[int, dict[str, str]] = {}
    for k in KS:
        cards = knn_cards(rows, k=k, strictness="base")
        maps[k] = {key_for(c["row"]): str(c["prediction"]) for c in cards}
    out = []
    for row in rows:
        item = dict(row)
        item["preds"] = {f"k{k}": maps[k].get(key_for(row), "UNKNOWN") for k in KS}
        if item.get("net_2h_bps") is not None:
            item["normal_2h_bps"] = float(item["net_2h_bps"])
            item["reverse_2h_bps"] = -float(item["net_2h_bps"]) - 2.0 * FEE_BPS
        out.append(item)
    return out


def build_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    def add(name: str, idxs: list[int], direction: str, family: str) -> None:
        vals = [outcome(rows[i], direction) for i in idxs]
        pairs = [(i, v) for i, v in zip(idxs, vals) if v is not None]
        if len(pairs) < MIN_CELL_N:
            return
        cells.append(
            {
                "name": name,
                "family": family,
                "direction": direction,
                "idxs": [i for i, _ in pairs],
                "values": [float(v) for _, v in pairs],
            }
        )

    for k in KS:
        for pred in PREDS:
            idxs = [i for i, r in enumerate(rows) if r["preds"].get(f"k{k}") == pred]
            for direction in ("NORMAL", "REVERSE"):
                add(f"k{k}_{pred}_{direction}", idxs, direction, "knn_pred")

    for left in PREDS:
        for right in PREDS:
            idxs = [
                i for i, r in enumerate(rows)
                if r["preds"].get("k5") == left and r["preds"].get("k20") == right
            ]
            for direction in ("NORMAL", "REVERSE"):
                add(f"k5_to_k20_{left}_to_{right}_{direction}", idxs, direction, "scale_transition")

    for count in range(7):
        idxs = [
            i for i, r in enumerate(rows)
            if sum(1 for k in KS if r["preds"].get(f"k{k}") == "DANGER") == count
        ]
        for direction in ("NORMAL", "REVERSE"):
            add(f"danger_count_{count}_{direction}", idxs, direction, "danger_count")

    for pred in PREDS:
        for threshold in (50_000, 100_000, 200_000):
            idxs = [
                i for i, r in enumerate(rows)
                if r["preds"].get("k20") == pred and int(float(r.get("threshold_usd") or 0)) == threshold
            ]
            for direction in ("NORMAL", "REVERSE"):
                add(f"k20_{pred}_thr{threshold}_{direction}", idxs, direction, "k20_threshold")

    tags = sorted({str(t) for r in rows for t in (r.get("tags") or []) if str(t) not in ILLEGAL_PRE_ENTRY_TAGS})
    for size in (1, 2, 3):
        for combo in combinations(tags, size):
            tag_set = set(combo)
            idxs = [
                i for i, r in enumerate(rows)
                if tag_set.issubset({str(t) for t in (r.get("tags") or []) if str(t) not in ILLEGAL_PRE_ENTRY_TAGS})
            ]
            for direction in ("NORMAL", "REVERSE"):
                add(f"tags_{'+'.join(combo)}_{direction}", idxs, direction, "tag_combo")

    return cells


def chronological_summary(rows: list[dict[str, Any]], idxs: list[int], direction: str, split_ts: int) -> dict[str, Any]:
    cal = []
    hold = []
    for i in idxs:
        v = outcome(rows[i], direction)
        if v is None:
            continue
        if ts(rows[i]) <= split_ts:
            cal.append(float(v))
        else:
            hold.append(float(v))
    return {"cal": summary(cal), "hold": summary(hold)}


def raw_permutation_p(cell: dict[str, Any], permuted_normal: list[float], rows: list[dict[str, Any]]) -> float:
    direction = cell["direction"]
    vals = []
    for i in cell["idxs"]:
        normal = permuted_normal[i]
        vals.append(normal if direction == "NORMAL" else -normal - 2.0 * FEE_BPS)
    return stat(vals)


def permutation_test(rows: list[dict[str, Any]], cells: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    rng = random.Random(SEED)
    normal_values = [outcome(r, "NORMAL") for r in rows]
    valid_normals = [float(v) for v in normal_values if v is not None and math.isfinite(float(v))]
    # Rows in this ledger all have normal values, but keep a complete vector for index safety.
    base = [float(v) if v is not None else 0.0 for v in normal_values]

    observed_stats = {c["name"]: stat(c["values"]) for c in cells}
    candidate_names = {c["cell_name"] for c in candidates}
    raw_counts = defaultdict(int)
    max_counts = defaultdict(int)
    max_null_stats: list[float] = []

    candidate_cells = {c["name"]: c for c in cells if c["name"] in candidate_names}
    for _ in range(PERMUTATIONS):
        shuffled = valid_normals[:]
        rng.shuffle(shuffled)
        permuted = base[:]
        j = 0
        for i, v in enumerate(normal_values):
            if v is not None:
                permuted[i] = shuffled[j]
                j += 1

        perm_stats: dict[str, float] = {}
        max_stat = float("-inf")
        for cell in cells:
            s = raw_permutation_p(cell, permuted, rows)
            perm_stats[cell["name"]] = s
            if s > max_stat:
                max_stat = s
        max_null_stats.append(max_stat)
        for name, cell in candidate_cells.items():
            obs = observed_stats[name]
            if perm_stats[name] >= obs:
                raw_counts[name] += 1
            if max_stat >= obs:
                max_counts[name] += 1

    by_candidate = {}
    for cand in candidates:
        name = cand["cell_name"]
        obs = observed_stats.get(name, float("-inf"))
        by_candidate[name] = {
            "observed_stat_t3r": r1(obs),
            "raw_p": r3((raw_counts[name] + 1) / (PERMUTATIONS + 1)),
            "mc_maxstat_p": r3((max_counts[name] + 1) / (PERMUTATIONS + 1)),
        }
    return {
        "permutations": PERMUTATIONS,
        "seed": SEED,
        "cell_universe_n": len(cells),
        "max_null_p95": r1(sorted(max_null_stats)[int(0.95 * len(max_null_stats))]),
        "max_null_p99": r1(sorted(max_null_stats)[int(0.99 * len(max_null_stats))]),
        "by_candidate": by_candidate,
    }


def make_candidate(name: str, cell_name: str, rationale: str) -> dict[str, Any]:
    return {"candidate": name, "cell_name": cell_name, "rationale": rationale}


def candidate_specs() -> list[dict[str, Any]]:
    return [
        make_candidate("k5_DANGER_reverse", "k5_DANGER_REVERSE", "small-neighborhood danger as fade/reversal"),
        make_candidate("k5_CLEAN_normal", "k5_CLEAN_NORMAL", "small-neighborhood clean as continuation/permission"),
        make_candidate("k8_DANGER_reverse", "k8_DANGER_REVERSE", "danger fade survives one wider scale"),
        make_candidate("k10_DANGER_reverse", "k10_DANGER_REVERSE", "danger fade at the last strong scale before k20 decay"),
        make_candidate("k5_CLEAN_to_k20_DANGER_normal", "k5_to_k20_CLEAN_to_DANGER_NORMAL", "local clean but broad dirty scale transition"),
        make_candidate("k20_DANGER_reverse", "k20_DANGER_REVERSE", "broad danger reverse/reversal question"),
        make_candidate("k20_DANGER_thr100k_reverse", "k20_DANGER_thr100000_REVERSE", "broad danger reverse that only looked positive at 100K"),
        make_candidate("k20_DANGER_thr200k_reverse", "k20_DANGER_thr200000_REVERSE", "broad danger reverse at 200K"),
        make_candidate("k20_DANGER_thr50k_reverse", "k20_DANGER_thr50000_REVERSE", "broad danger reverse at 50K"),
    ]


def report_candidate(rows: list[dict[str, Any]], cells: list[dict[str, Any]], split_ts: int, perm: dict[str, Any]) -> list[dict[str, Any]]:
    cell_by_name = {c["name"]: c for c in cells}
    out = []
    for spec in candidate_specs():
        cell = cell_by_name.get(spec["cell_name"])
        if not cell:
            item = {**spec, "status": "MISSING_OR_N_LT_40"}
            out.append(item)
            continue
        chrono = chronological_summary(rows, cell["idxs"], cell["direction"], split_ts)
        full = summary(cell["values"])
        p = perm["by_candidate"].get(cell["name"], {})
        hold = chrono["hold"]
        cal = chrono["cal"]
        passes = (
            cal["n"] >= 40
            and hold["n"] >= 40
            and float(cal.get("t3r_bps") or -1e9) > 0
            and float(hold.get("t3r_bps") or -1e9) > 0
            and float(p.get("mc_maxstat_p") or 1.0) <= 0.05
        )
        out.append(
            {
                **spec,
                "status": "PASS" if passes else "FAIL",
                "full": full,
                "chronological": chrono,
                "permutation": p,
            }
        )
    return out


def top_cells(cells: list[dict[str, Any]], n: int = 20) -> list[dict[str, Any]]:
    rows = []
    for c in cells:
        s = summary(c["values"])
        if float(s.get("t3r_bps") or -1e9) <= 0:
            continue
        rows.append(
            {
                "cell": c["name"],
                "family": c["family"],
                "direction": c["direction"],
                "summary": s,
                "score_t3r": s["t3r_bps"],
            }
        )
    rows.sort(key=lambda r: float(r["score_t3r"] or -1e9), reverse=True)
    return rows[:n]


def fmt(s: dict[str, Any]) -> str:
    return (
        f"N={s.get('n')} sum={s.get('sum_bps')} med={s.get('median_bps')} "
        f"T3R={s.get('t3r_bps')} tail150={s.get('tail_lte_minus150_n')} maxLoss={s.get('max_loss_bps')}"
    )


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# S34 Navigation Gauntlet",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        f"Cell universe tested for max-stat correction: `{result['permutation']['cell_universe_n']}`",
        f"Permutations: `{result['permutation']['permutations']}`; max-null p95 T3R: `{result['permutation']['max_null_p95']}`; p99: `{result['permutation']['max_null_p99']}`",
        "",
        "## Candidate Gauntlet",
        "",
        "| Candidate | Status | Full | Cal | Hold | Raw p | MC p |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for c in result["candidates"]:
        if c["status"] == "MISSING_OR_N_LT_40":
            lines.append(f"| {c['candidate']} | {c['status']} | - | - | - | - | - |")
            continue
        p = c["permutation"]
        lines.append(
            f"| {c['candidate']} | {c['status']} | {fmt(c['full'])} | "
            f"{fmt(c['chronological']['cal'])} | {fmt(c['chronological']['hold'])} | "
            f"{p.get('raw_p')} | {p.get('mc_maxstat_p')} |"
        )

    lines.extend(["", "## k20 DANGER interpretation", ""])
    lines.extend(
        [
            "- Broad `k20 DANGER` is not a clean reversal: reverse 2h remains negative after costs and fails MC correction.",
            "- The only attractive-looking broad subcell is `k20 DANGER + 100K + reverse`, but it fails the chronological holdout and MC-corrected threshold.",
            "- Current interpretation: `k20 DANGER` is an avoid/risk label, not a standalone direction.",
        ]
    )

    lines.extend(["", "## Top In-Sample Cells Before Correction", ""])
    lines.append("| Cell | Family | Direction | Summary |")
    lines.append("| --- | --- | --- | --- |")
    for row in result["top_cells_pre_correction"]:
        lines.append(f"| {row['cell']} | {row['family']} | {row['direction']} | {fmt(row['summary'])} |")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raw_rows = load_jsonl(NAV_EVENTS)
    rows = enrich_rows(raw_rows)
    rows_sorted = sorted(rows, key=ts)
    split_idx = int(len(rows_sorted) * 0.7)
    split_ts = ts(rows_sorted[split_idx])
    cells = build_cells(rows)
    candidates = candidate_specs()
    perm = permutation_test(rows, cells, candidates)
    result = {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "row_count": len(rows),
        "chronological_split_ts_ms": split_ts,
        "chronological_split_index": split_idx,
        "candidates": report_candidate(rows, cells, split_ts, perm),
        "permutation": perm,
        "top_cells_pre_correction": top_cells(cells, 25),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(result)
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
