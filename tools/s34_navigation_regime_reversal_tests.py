"""S34 navigation regime-reversal tests.

Question:
Do the in-sample / calibration navigation patterns systematically invert in the
causal holdout? If yes, the next edge may be a regime-reversal/meta-reversal
rule. If no, the failures are just unstable/noisy cells.

Research-only. No live executor, order logic, size, leverage, or .env changes.
"""

from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_navigation_causal_gauntlet import (  # noqa: E402
    MIN_CELL_N,
    SEED,
    attach,
    build_universe,
    causal_preds,
    normal_value,
    prepare,
    value_from_normal,
    vals_for_universe_cell,
)
from tools.s34_navigation_full_followup import NAV_EVENTS, load_jsonl, r1, r3, summary  # noqa: E402

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_REGIME_REVERSAL_TESTS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_REGIME_REVERSAL_TESTS.md"

PERMUTATIONS = 1000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def t3r(vals: list[float]) -> float:
    if len(vals) <= 3:
        return float(sum(vals))
    return float(sum(sorted(vals, reverse=True)[3:]))


def safe_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def rankdata(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def safe_spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return safe_corr(rankdata(xs), rankdata(ys))


def opposite_cell_name(name: str) -> str:
    if name.endswith("_NORMAL"):
        return name[:-7] + "_REVERSE"
    if name.endswith("_REVERSE"):
        return name[:-8] + "_NORMAL"
    return name


def cell_stats(rows: list[dict[str, Any]], universe: list[tuple[str, str]]) -> dict[str, dict[str, Any]]:
    out = {}
    for name, _direction in universe:
        vals = vals_for_universe_cell(rows, name)
        if len(vals) < MIN_CELL_N:
            continue
        s = summary(vals)
        out[name] = {
            "name": name,
            "n": s["n"],
            "sum_bps": float(s.get("sum_bps") or 0.0),
            "median_bps": s.get("median_bps"),
            "t3r_bps": float(s.get("t3r_bps") or 0.0),
            "tail_lte_minus150_n": int(s.get("tail_lte_minus150_n") or 0),
            "max_loss_bps": s.get("max_loss_bps"),
            "summary": s,
        }
    return out


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prep = prepare(load_jsonl(NAV_EVENTS))
    cal_normals = [normal_value(r) for r in prep["cal"]]
    cal_pred = causal_preds(prep["cal_neighbors"], cal_normals)
    hold_pred = causal_preds(prep["hold_neighbors"], cal_normals)
    return attach(prep["cal"], cal_pred), attach(prep["hold"], hold_pred), prep


def correlation_block(cal_stats: dict[str, dict[str, Any]], hold_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    names = sorted(set(cal_stats) & set(hold_stats))
    xs_t3r = [cal_stats[n]["t3r_bps"] for n in names]
    ys_t3r = [hold_stats[n]["t3r_bps"] for n in names]
    xs_sum = [cal_stats[n]["sum_bps"] for n in names]
    ys_sum = [hold_stats[n]["sum_bps"] for n in names]
    return {
        "cell_n": len(names),
        "pearson_t3r": r3(safe_corr(xs_t3r, ys_t3r)),
        "spearman_t3r": r3(safe_spearman(xs_t3r, ys_t3r)),
        "pearson_sum": r3(safe_corr(xs_sum, ys_sum)),
        "spearman_sum": r3(safe_spearman(xs_sum, ys_sum)),
    }


def top_cal_analysis(cal_stats: dict[str, dict[str, Any]], hold_stats: dict[str, dict[str, Any]], top_n: int = 20) -> dict[str, Any]:
    eligible = [v for v in cal_stats.values() if v["t3r_bps"] > 0]
    top = sorted(eligible, key=lambda x: x["t3r_bps"], reverse=True)[:top_n]
    rows = []
    hold_same = []
    hold_opp = []
    for cell in top:
        name = cell["name"]
        opp = opposite_cell_name(name)
        same = hold_stats.get(name)
        opposite = hold_stats.get(opp)
        rows.append(
            {
                "cell": name,
                "cal": cell["summary"],
                "hold_same": same["summary"] if same else None,
                "hold_opposite": opposite["summary"] if opposite else None,
            }
        )
        if same:
            hold_same.append(same["t3r_bps"])
        if opposite:
            hold_opp.append(opposite["t3r_bps"])
    return {
        "top_n": top_n,
        "same_positive_n": sum(1 for x in hold_same if x > 0),
        "opposite_positive_n": sum(1 for x in hold_opp if x > 0),
        "same_median_t3r": r1(median(hold_same)) if hold_same else None,
        "opposite_median_t3r": r1(median(hold_opp)) if hold_opp else None,
        "same_sum_t3r": r1(sum(hold_same)) if hold_same else None,
        "opposite_sum_t3r": r1(sum(hold_opp)) if hold_opp else None,
        "rows": rows,
    }


def sign_flip_scan(cal_stats: dict[str, dict[str, Any]], hold_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for name, cal in cal_stats.items():
        hold = hold_stats.get(name)
        opp = hold_stats.get(opposite_cell_name(name))
        if not hold or not opp:
            continue
        if cal["t3r_bps"] > 500 and hold["t3r_bps"] < -500 and opp["t3r_bps"] > 0:
            rows.append(
                {
                    "cell": name,
                    "opposite_cell": opposite_cell_name(name),
                    "cal_t3r": r1(cal["t3r_bps"]),
                    "hold_same_t3r": r1(hold["t3r_bps"]),
                    "hold_opposite_t3r": r1(opp["t3r_bps"]),
                    "hold_opposite": opp["summary"],
                }
            )
    rows.sort(key=lambda x: float(x["hold_opposite_t3r"] or -1e9), reverse=True)
    return {"n": len(rows), "top": rows[:25]}


def specific_pairs(cal_stats: dict[str, dict[str, Any]], hold_stats: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    names = [
        "k5_DANGER_REVERSE",
        "k5_DANGER_NORMAL",
        "k5_CLEAN_NORMAL",
        "k5_CLEAN_REVERSE",
        "k8_DANGER_REVERSE",
        "k8_DANGER_NORMAL",
        "k10_DANGER_REVERSE",
        "k10_DANGER_NORMAL",
        "k20_DANGER_REVERSE",
        "k20_DANGER_NORMAL",
        "k20_DANGER_thr100000_REVERSE",
        "k20_DANGER_thr100000_NORMAL",
    ]
    out = []
    for name in names:
        out.append(
            {
                "cell": name,
                "cal": cal_stats.get(name, {}).get("summary"),
                "hold": hold_stats.get(name, {}).get("summary"),
            }
        )
    return out


def permutation_for_correlation(cal_stats: dict[str, dict[str, Any]], hold_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    names = sorted(set(cal_stats) & set(hold_stats))
    xs = [cal_stats[n]["t3r_bps"] for n in names]
    ys = [hold_stats[n]["t3r_bps"] for n in names]
    obs = safe_corr(xs, ys)
    if obs is None:
        return {"observed_pearson_t3r": None}
    rng = random.Random(SEED + 77)
    le = 0
    ge_abs = 0
    for _ in range(PERMUTATIONS):
        shuffled = ys[:]
        rng.shuffle(shuffled)
        c = safe_corr(xs, shuffled)
        if c is None:
            continue
        if c <= obs:
            le += 1
        if abs(c) >= abs(obs):
            ge_abs += 1
    return {
        "observed_pearson_t3r": r3(obs),
        "p_negative_or_lower": r3((le + 1) / (PERMUTATIONS + 1)),
        "p_two_sided_abs": r3((ge_abs + 1) / (PERMUTATIONS + 1)),
        "permutations": PERMUTATIONS,
    }


def fmt(s: dict[str, Any] | None) -> str:
    if not s:
        return "-"
    return (
        f"N={s.get('n')} sum={s.get('sum_bps')} med={s.get('median_bps')} "
        f"T3R={s.get('t3r_bps')} tail150={s.get('tail_lte_minus150_n')} maxLoss={s.get('max_loss_bps')}"
    )


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# S34 Navigation Regime-Reversal Tests",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        "## Cal -> Hold Correlation",
        "",
        f"- Cells with N>=40 in both: `{result['correlation']['cell_n']}`",
        f"- Pearson T3R: `{result['correlation']['pearson_t3r']}`; Spearman T3R: `{result['correlation']['spearman_t3r']}`",
        f"- Permutation p(negative-or-lower): `{result['correlation_permutation']['p_negative_or_lower']}`; two-sided: `{result['correlation_permutation']['p_two_sided_abs']}`",
        "",
        "## Top-Cal Meta Reversal",
        "",
        f"- Top cal cells: `{result['top_cal_analysis']['top_n']}`",
        f"- Hold same positive: `{result['top_cal_analysis']['same_positive_n']}`",
        f"- Hold opposite positive: `{result['top_cal_analysis']['opposite_positive_n']}`",
        f"- Hold same median T3R: `{result['top_cal_analysis']['same_median_t3r']}`",
        f"- Hold opposite median T3R: `{result['top_cal_analysis']['opposite_median_t3r']}`",
        "",
        "## Specific Reversal Checks",
        "",
        "| Cell | Cal | Hold |",
        "| --- | --- | --- |",
    ]
    for row in result["specific_pairs"]:
        lines.append(f"| {row['cell']} | {fmt(row['cal'])} | {fmt(row['hold'])} |")

    lines.extend(["", "## Sign-Flip Candidates", ""])
    lines.append(f"Count: `{result['sign_flip_scan']['n']}`")
    lines.append("")
    lines.append("| Cell | Opposite | Cal T3R | Hold Same T3R | Hold Opp T3R | Hold Opp Summary |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for row in result["sign_flip_scan"]["top"][:15]:
        lines.append(
            f"| {row['cell']} | {row['opposite_cell']} | {row['cal_t3r']} | "
            f"{row['hold_same_t3r']} | {row['hold_opposite_t3r']} | {fmt(row['hold_opposite'])} |"
        )

    lines.extend(["", "## Top-Cal Rows", ""])
    lines.append("| Cell | Cal | Hold Same | Hold Opposite |")
    lines.append("| --- | --- | --- | --- |")
    for row in result["top_cal_analysis"]["rows"][:15]:
        lines.append(f"| {row['cell']} | {fmt(row['cal'])} | {fmt(row['hold_same'])} | {fmt(row['hold_opposite'])} |")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cal_rows, hold_rows, prep = build_rows()
    universe = build_universe(hold_rows)
    cal_stats = cell_stats(cal_rows, universe)
    hold_stats = cell_stats(hold_rows, universe)
    result = {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "cal_n": len(cal_rows),
        "hold_n": len(hold_rows),
        "split_ts_ms": prep["split_ts_ms"],
        "correlation": correlation_block(cal_stats, hold_stats),
        "correlation_permutation": permutation_for_correlation(cal_stats, hold_stats),
        "top_cal_analysis": top_cal_analysis(cal_stats, hold_stats, 20),
        "sign_flip_scan": sign_flip_scan(cal_stats, hold_stats),
        "specific_pairs": specific_pairs(cal_stats, hold_stats),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(result)
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
