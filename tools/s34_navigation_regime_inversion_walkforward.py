"""Walk-forward tests for S34 navigation regime inversion.

Research-only:
- repeated chronological folds;
- each holdout fold is classified using prior calibration rows only;
- tests whether top calibration cells invert in the next fold;
- tests specific branches such as NEUTRAL_CONTEXT reverse and k5 CLEAN reverse.

No live executor, order logic, size, leverage, or .env changes.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_navigation_causal_gauntlet import (  # noqa: E402
    KS,
    MIN_CELL_N,
    PREDS,
    causal_preds,
    normal_value,
    value_from_normal,
)
from tools.s34_navigation_full_followup import (  # noqa: E402
    FEE_BPS,
    NAV_EVENTS,
    distance,
    feature_vector,
    load_jsonl,
    r1,
    r3,
    summary,
)
from tools.s34_navigation_meta_patterns import ILLEGAL_PRE_ENTRY_TAGS  # noqa: E402
from tools.s34_navigation_regime_reversal_tests import safe_corr, safe_spearman  # noqa: E402

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_REGIME_INVERSION_WALKFORWARD.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_REGIME_INVERSION_WALKFORWARD.md"

SEED = 34031
TOP_NS = [5, 10, 20]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ts(row: dict[str, Any]) -> int:
    return int(row.get("signal_ts_ms") or 0)


def iso_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).isoformat()


def t3r(vals: list[float]) -> float:
    if len(vals) <= 3:
        return float(sum(vals))
    return float(sum(sorted(vals, reverse=True)[3:]))


def tagset(row: dict[str, Any]) -> set[str]:
    return {str(t) for t in (row.get("tags") or []) if str(t) not in ILLEGAL_PRE_ENTRY_TAGS}


class Cell:
    def __init__(self, name: str, selector: Callable[[dict[str, Any]], bool], direction: str, family: str) -> None:
        self.name = name
        self.selector = selector
        self.direction = direction
        self.family = family

    def values(self, rows: list[dict[str, Any]], direction: str | None = None) -> list[float]:
        side = direction or self.direction
        vals = []
        for row in rows:
            if self.selector(row):
                vals.append(value_from_normal(normal_value(row), side))
        return vals

    def opposite(self) -> str:
        return "REVERSE" if self.direction == "NORMAL" else "NORMAL"


def build_cells(rows: list[dict[str, Any]]) -> list[Cell]:
    cells: list[Cell] = []
    for k in KS:
        for pred in PREDS:
            for direction in ("NORMAL", "REVERSE"):
                cells.append(
                    Cell(
                        f"k{k}_{pred}_{direction}",
                        lambda r, kk=k, pp=pred: r.get("preds", {}).get(f"k{kk}") == pp,
                        direction,
                        "knn_pred",
                    )
                )
    for left in PREDS:
        for right in PREDS:
            for direction in ("NORMAL", "REVERSE"):
                cells.append(
                    Cell(
                        f"k5_to_k20_{left}_to_{right}_{direction}",
                        lambda r, l=left, rr=right: r.get("preds", {}).get("k5") == l and r.get("preds", {}).get("k20") == rr,
                        direction,
                        "scale_transition",
                    )
                )
    for count in range(7):
        for direction in ("NORMAL", "REVERSE"):
            cells.append(
                Cell(
                    f"danger_count_{count}_{direction}",
                    lambda r, c=count: sum(1 for k in KS if r.get("preds", {}).get(f"k{k}") == "DANGER") == c,
                    direction,
                    "danger_count",
                )
            )
    for pred in PREDS:
        for threshold in (50_000, 100_000, 200_000):
            for direction in ("NORMAL", "REVERSE"):
                cells.append(
                    Cell(
                        f"k20_{pred}_thr{threshold}_{direction}",
                        lambda r, p=pred, th=threshold: r.get("preds", {}).get("k20") == p
                        and int(float(r.get("threshold_usd") or 0)) == th,
                        direction,
                        "k20_threshold",
                    )
                )
    tags = sorted({str(t) for r in rows for t in tagset(r)})
    for tag in tags:
        for direction in ("NORMAL", "REVERSE"):
            cells.append(Cell(f"tags_{tag}_{direction}", lambda r, tg=tag: tg in tagset(r), direction, "tag"))
    # A short hand-picked set from prior sign-flips.
    combos = [
        ("BID_DEPTH_THIN", "NEUTRAL_CONTEXT"),
        ("NEUTRAL_CONTEXT", "VDEPTH_DANGER_LOW"),
        ("BULL_PULLBACK", "BID_DEPTH_THIN"),
    ]
    for combo in combos:
        for direction in ("NORMAL", "REVERSE"):
            cells.append(
                Cell(
                    f"tags_{'+'.join(combo)}_{direction}",
                    lambda r, cc=set(combo): cc.issubset(tagset(r)),
                    direction,
                    "tag_combo",
                )
            )
    return cells


def attach_preds(rows: list[dict[str, Any]], preds: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for row, pred in zip(rows, preds):
        item = dict(row)
        item["preds"] = pred
        item["normal_2h_bps"] = normal_value(row)
        item["reverse_2h_bps"] = value_from_normal(normal_value(row), "REVERSE")
        out.append(item)
    return out


def neighbors(train: list[dict[str, Any]], target: list[dict[str, Any]], *, leave_one_out: bool) -> list[list[int]]:
    train_vecs = [feature_vector(r) for r in train]
    target_vecs = [feature_vector(r) for r in target]
    out = []
    for i, v in enumerate(target_vecs):
        ds = []
        for j, other in enumerate(train_vecs):
            if leave_one_out and i == j:
                continue
            ds.append((distance(v, other), j))
        out.append([j for _, j in sorted(ds, key=lambda x: x[0])[: max(KS)]])
    return out


def make_folds(rows: list[dict[str, Any]], folds: int = 5, min_train_frac: float = 0.4) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    ordered = sorted([r for r in rows if r.get("net_2h_bps") is not None], key=ts)
    start = int(len(ordered) * min_train_frac)
    remaining = len(ordered) - start
    fold_size = remaining // folds
    out = []
    for i in range(folds):
        a = start + i * fold_size
        b = start + (i + 1) * fold_size if i < folds - 1 else len(ordered)
        train = ordered[:a]
        hold = ordered[a:b]
        if len(train) >= 100 and len(hold) >= 40:
            out.append((train, hold))
    return out


def cell_stats(rows: list[dict[str, Any]], cells: list[Cell]) -> dict[str, dict[str, Any]]:
    out = {}
    for cell in cells:
        vals = cell.values(rows)
        if len(vals) < MIN_CELL_N:
            continue
        s = summary(vals)
        out[cell.name] = {
            "cell": cell,
            "summary": s,
            "t3r": float(s.get("t3r_bps") or 0.0),
            "sum": float(s.get("sum_bps") or 0.0),
        }
    return out


def summarize_cell_basket(rows: list[dict[str, Any]], selected: list[Cell], *, invert: bool) -> dict[str, Any]:
    vals = []
    event_hits: dict[str, list[float]] = defaultdict(list)
    for cell in selected:
        direction = cell.opposite() if invert else cell.direction
        for row in rows:
            if cell.selector(row):
                value = value_from_normal(normal_value(row), direction)
                vals.append(value)
                event_hits[str(row.get("event_id") or ts(row))].append(value)
    event_avg = [sum(v) / len(v) for v in event_hits.values()]
    return {
        "cell_trade": summary(vals),
        "event_avg": summary(event_avg),
        "event_n": len(event_avg),
        "cell_trade_n": len(vals),
    }


def top_cell_names(cal_stats: dict[str, dict[str, Any]], top_n: int) -> list[str]:
    eligible = [v for v in cal_stats.values() if v["t3r"] > 0]
    eligible.sort(key=lambda x: x["t3r"], reverse=True)
    return [str(v["cell"].name) for v in eligible[:top_n]]


def fold_result(train_raw: list[dict[str, Any]], hold_raw: list[dict[str, Any]], fold_idx: int) -> dict[str, Any]:
    train_normals = [normal_value(r) for r in train_raw]
    train_pred = causal_preds(neighbors(train_raw, train_raw, leave_one_out=True), train_normals)
    hold_pred = causal_preds(neighbors(train_raw, hold_raw, leave_one_out=False), train_normals)
    train = attach_preds(train_raw, train_pred)
    hold = attach_preds(hold_raw, hold_pred)
    cells = build_cells(train + hold)
    by_name = {c.name: c for c in cells}
    cal_stats = cell_stats(train, cells)
    hold_stats = cell_stats(hold, cells)

    common = sorted(set(cal_stats) & set(hold_stats))
    cal_t3r = [cal_stats[n]["t3r"] for n in common]
    hold_t3r = [hold_stats[n]["t3r"] for n in common]

    top_blocks = {}
    for n in TOP_NS:
        names = top_cell_names(cal_stats, n)
        selected = [by_name[name] for name in names if name in by_name]
        top_blocks[f"top{n}"] = {
            "selected": names,
            "same": summarize_cell_basket(hold, selected, invert=False),
            "inverted": summarize_cell_basket(hold, selected, invert=True),
        }

    fixed_names = [
        "tags_NEUTRAL_CONTEXT_REVERSE",
        "tags_NEUTRAL_CONTEXT_NORMAL",
        "k5_CLEAN_REVERSE",
        "k5_CLEAN_NORMAL",
        "k20_DANGER_REVERSE",
        "k20_DANGER_NORMAL",
        "k20_DANGER_thr100000_REVERSE",
    ]
    fixed = {}
    for name in fixed_names:
        cell = by_name.get(name)
        if cell:
            fixed[name] = {"hold": summary(cell.values(hold)), "cal": summary(cell.values(train))}

    return {
        "fold": fold_idx,
        "train_n": len(train),
        "hold_n": len(hold),
        "hold_start": iso_ms(ts(hold[0])),
        "hold_end": iso_ms(ts(hold[-1])),
        "cell_common_n": len(common),
        "pearson_t3r": r3(safe_corr(cal_t3r, hold_t3r)),
        "spearman_t3r": r3(safe_spearman(cal_t3r, hold_t3r)),
        "top_blocks": top_blocks,
        "fixed_branches": fixed,
    }


def aggregate_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for top in (f"top{n}" for n in TOP_NS):
        same_vals = []
        inv_vals = []
        same_event_vals = []
        inv_event_vals = []
        for fold in folds:
            same_vals.extend(extract_values_from_summary_proxy(fold["top_blocks"][top]["same"]["cell_trade"]))
            inv_vals.extend(extract_values_from_summary_proxy(fold["top_blocks"][top]["inverted"]["cell_trade"]))
            same_event_vals.extend(extract_values_from_summary_proxy(fold["top_blocks"][top]["same"]["event_avg"]))
            inv_event_vals.extend(extract_values_from_summary_proxy(fold["top_blocks"][top]["inverted"]["event_avg"]))
        # We cannot reconstruct distributions from summaries. Keep fold-level aggregate metrics instead.
        same_t3rs = [float(f["top_blocks"][top]["same"]["event_avg"].get("t3r_bps") or 0.0) for f in folds]
        inv_t3rs = [float(f["top_blocks"][top]["inverted"]["event_avg"].get("t3r_bps") or 0.0) for f in folds]
        out[top] = {
            "folds_same_positive": sum(1 for x in same_t3rs if x > 0),
            "folds_inverted_positive": sum(1 for x in inv_t3rs if x > 0),
            "same_sum_fold_t3r": r1(sum(same_t3rs)),
            "inverted_sum_fold_t3r": r1(sum(inv_t3rs)),
            "same_median_fold_t3r": r1(median(same_t3rs)),
            "inverted_median_fold_t3r": r1(median(inv_t3rs)),
        }
    fixed_names = sorted({name for f in folds for name in f["fixed_branches"]})
    fixed = {}
    for name in fixed_names:
        hold_t3rs = [float(f["fixed_branches"][name]["hold"].get("t3r_bps") or 0.0) for f in folds if name in f["fixed_branches"]]
        hold_sums = [float(f["fixed_branches"][name]["hold"].get("sum_bps") or 0.0) for f in folds if name in f["fixed_branches"]]
        fixed[name] = {
            "folds": len(hold_t3rs),
            "positive_t3r_folds": sum(1 for x in hold_t3rs if x > 0),
            "sum_fold_t3r": r1(sum(hold_t3rs)),
            "median_fold_t3r": r1(median(hold_t3rs)) if hold_t3rs else None,
            "sum_fold_sum_bps": r1(sum(hold_sums)),
        }
    cors = [float(f["pearson_t3r"]) for f in folds if f.get("pearson_t3r") is not None]
    return {
        "top_meta_rules": out,
        "fixed_branches": fixed,
        "fold_correlation": {
            "median_pearson_t3r": r3(median(cors)) if cors else None,
            "negative_folds": sum(1 for c in cors if c < 0),
            "folds": len(cors),
        },
    }


def extract_values_from_summary_proxy(_s: dict[str, Any]) -> list[float]:
    # Placeholder kept to make the aggregation intent explicit. Distributions are
    # intentionally not reconstructed from summaries.
    return []


def permutation_top_inversion(folds: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]], observed: dict[str, Any]) -> dict[str, Any]:
    """Permutation for whether inverted top10 beats same top10 at fold level."""
    rng = random.Random(SEED)
    obs = float(observed["top_meta_rules"]["top10"]["inverted_sum_fold_t3r"] or 0.0) - float(
        observed["top_meta_rules"]["top10"]["same_sum_fold_t3r"] or 0.0
    )
    ge = 0
    # This permutation recomputes causal KNN labels per fold, so it is
    # intentionally capped for an overnight-friendly exploratory check.
    sims = 30
    for _ in range(sims):
        perm_fold_results = []
        for idx, (train, hold) in enumerate(folds, start=1):
            shuffled_train = [dict(r) for r in train]
            shuffled_hold = [dict(r) for r in hold]
            all_values = [normal_value(r) for r in shuffled_train + shuffled_hold]
            rng.shuffle(all_values)
            for row, value in zip(shuffled_train + shuffled_hold, all_values):
                row["net_2h_bps"] = value
            perm_fold_results.append(fold_result(shuffled_train, shuffled_hold, idx))
        agg = aggregate_folds(perm_fold_results)
        diff = float(agg["top_meta_rules"]["top10"]["inverted_sum_fold_t3r"] or 0.0) - float(
            agg["top_meta_rules"]["top10"]["same_sum_fold_t3r"] or 0.0
        )
        if diff >= obs:
            ge += 1
    return {"observed_top10_inverted_minus_same_fold_t3r": r1(obs), "permutations": sims, "p_ge": r3((ge + 1) / (sims + 1))}


def fmt(s: dict[str, Any]) -> str:
    return (
        f"N={s.get('n')} sum={s.get('sum_bps')} med={s.get('median_bps')} "
        f"T3R={s.get('t3r_bps')} tail150={s.get('tail_lte_minus150_n')} maxLoss={s.get('max_loss_bps')}"
    )


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# S34 Navigation Regime-Inversion Walk-Forward",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        "## Aggregate",
        "",
        f"- Fold correlation: `{result['aggregate']['fold_correlation']}`",
        f"- Top10 inversion permutation: `{result['permutation_top10']}`",
        "",
        "### Top-Cell Meta Rules",
        "",
        "| Rule | Same positive folds | Inverted positive folds | Same fold T3R sum | Inverted fold T3R sum | Same median | Inverted median |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, row in result["aggregate"]["top_meta_rules"].items():
        lines.append(
            f"| {key} | {row['folds_same_positive']} | {row['folds_inverted_positive']} | "
            f"{row['same_sum_fold_t3r']} | {row['inverted_sum_fold_t3r']} | "
            f"{row['same_median_fold_t3r']} | {row['inverted_median_fold_t3r']} |"
        )

    lines.extend(["", "### Fixed Branches", ""])
    lines.append("| Branch | Positive folds | Fold T3R sum | Median fold T3R | Fold sum bps |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name, row in sorted(result["aggregate"]["fixed_branches"].items(), key=lambda kv: float(kv[1].get("sum_fold_t3r") or -1e9), reverse=True):
        lines.append(
            f"| {name} | {row['positive_t3r_folds']}/{row['folds']} | {row['sum_fold_t3r']} | "
            f"{row['median_fold_t3r']} | {row['sum_fold_sum_bps']} |"
        )

    lines.extend(["", "## Folds", ""])
    for fold in result["folds"]:
        lines.extend(
            [
                f"### Fold {fold['fold']}",
                "",
                f"- Train N `{fold['train_n']}`, hold N `{fold['hold_n']}`; hold `{fold['hold_start']}` -> `{fold['hold_end']}`",
                f"- Pearson T3R `{fold['pearson_t3r']}`, Spearman T3R `{fold['spearman_t3r']}`",
                "",
                "| Top rule | Same event avg | Inverted event avg |",
                "| --- | --- | --- |",
            ]
        )
        for key, block in fold["top_blocks"].items():
            lines.append(f"| {key} | {fmt(block['same']['event_avg'])} | {fmt(block['inverted']['event_avg'])} |")
        lines.extend(["", "| Fixed branch | Cal | Hold |", "| --- | --- | --- |"])
        for name, block in sorted(fold["fixed_branches"].items()):
            lines.append(f"| {name} | {fmt(block['cal'])} | {fmt(block['hold'])} |")
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_jsonl(NAV_EVENTS)
    raw_folds = make_folds(rows, folds=5, min_train_frac=0.4)
    fold_results = [fold_result(train, hold, idx) for idx, (train, hold) in enumerate(raw_folds, start=1)]
    aggregate = aggregate_folds(fold_results)
    perm = permutation_top_inversion(raw_folds, aggregate)
    result = {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "fold_count": len(fold_results),
        "aggregate": aggregate,
        "permutation_top10": perm,
        "folds": fold_results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(result)
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
