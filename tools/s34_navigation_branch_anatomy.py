"""S34 branch sign-flip anatomy.

Research-only anatomy for the + / - branch reversals:
- same/opposite pair matrix;
- fold-state map;
- branch cohort drift;
- tail attribution;
- event-sequence density;
- sign-flip trigger;
- lead-lag between branches;
- directional entropy;
- anti-crowding quantiles;
- navigation action simulation.

No live executor, order logic, size, leverage, or .env changes.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_navigation_causal_gauntlet import normal_value, value_from_normal  # noqa: E402
from tools.s34_navigation_full_followup import NAV_EVENTS, load_jsonl, r1, r3, summary  # noqa: E402
from tools.s34_navigation_regime_inversion_walkforward import (  # noqa: E402
    Cell,
    attach_preds,
    build_cells,
    cell_stats,
    make_folds,
    neighbors,
    safe_corr,
    safe_spearman,
    tagset,
    t3r,
)
from tools.s34_navigation_causal_gauntlet import causal_preds  # noqa: E402

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_BRANCH_ANATOMY.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_BRANCH_ANATOMY.md"

BRANCH_PAIRS = [
    ("k5_CLEAN", "k5_CLEAN_NORMAL", "k5_CLEAN_REVERSE"),
    ("k5_DANGER", "k5_DANGER_NORMAL", "k5_DANGER_REVERSE"),
    ("k8_DANGER", "k8_DANGER_NORMAL", "k8_DANGER_REVERSE"),
    ("k10_DANGER", "k10_DANGER_NORMAL", "k10_DANGER_REVERSE"),
    ("k20_DANGER", "k20_DANGER_NORMAL", "k20_DANGER_REVERSE"),
    ("k20_DANGER_100K", "k20_DANGER_thr100000_NORMAL", "k20_DANGER_thr100000_REVERSE"),
    ("NEUTRAL_CONTEXT", "tags_NEUTRAL_CONTEXT_NORMAL", "tags_NEUTRAL_CONTEXT_REVERSE"),
    ("BID_THIN_NEUTRAL", "tags_BID_DEPTH_THIN+NEUTRAL_CONTEXT_NORMAL", "tags_BID_DEPTH_THIN+NEUTRAL_CONTEXT_REVERSE"),
]

FEATURES = ["threshold_usd", "vdepth_bps", "bid_depth_usd", "prior4h_bps", "eth1h_bps", "btc4h_bps", "book_imbalance"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ts(row: dict[str, Any]) -> int:
    return int(row.get("signal_ts_ms") or 0)


def iso_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).isoformat()


def avg(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    if not vals:
        return None
    return r1(sum(vals) / len(vals))


def med(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    if not vals:
        return None
    return r1(median(vals))


def fold_state(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [normal_value(r) for r in rows]
    times = sorted(ts(r) for r in rows)
    duration_hours = (times[-1] - times[0]) / 3_600_000.0 if len(times) > 1 else 0.0
    tail_n = sum(1 for v in vals if v <= -150.0)
    return {
        "n": len(rows),
        "start": iso_ms(times[0]) if times else None,
        "end": iso_ms(times[-1]) if times else None,
        "duration_hours": r1(duration_hours),
        "event_density_per_day": r1(len(rows) / max(duration_hours / 24.0, 1e-9)) if duration_hours else None,
        "normal_2h": summary(vals),
        "tail150_rate": r3(tail_n / len(vals)) if vals else None,
        "avg": {k: avg(rows, k) for k in FEATURES},
        "median": {k: med(rows, k) for k in FEATURES},
        "tag_mix": top_counts([t for r in rows for t in tagset(r)], 8),
        "threshold_mix": top_counts([f"thr{int(float(r.get('threshold_usd') or 0))}" for r in rows], 5),
    }


def top_counts(items: list[str], n: int) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[item] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n])


def prepare_fold(train_raw: list[dict[str, Any]], hold_raw: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Cell]]:
    train_normals = [normal_value(r) for r in train_raw]
    train_pred = causal_preds(neighbors(train_raw, train_raw, leave_one_out=True), train_normals)
    hold_pred = causal_preds(neighbors(train_raw, hold_raw, leave_one_out=False), train_normals)
    train = attach_preds(train_raw, train_pred)
    hold = attach_preds(hold_raw, hold_pred)
    cells = build_cells(train + hold)
    return train, hold, {c.name: c for c in cells}


def cell_summary(rows: list[dict[str, Any]], cell: Cell | None) -> dict[str, Any]:
    if not cell:
        return summary([])
    return summary(cell.values(rows))


def selected_rows(rows: list[dict[str, Any]], cell: Cell | None) -> list[dict[str, Any]]:
    if not cell:
        return []
    return [r for r in rows if cell.selector(r)]


def pair_matrix(train: list[dict[str, Any]], hold: list[dict[str, Any]], by_name: dict[str, Cell]) -> list[dict[str, Any]]:
    out = []
    for label, normal_name, reverse_name in BRANCH_PAIRS:
        n_cell = by_name.get(normal_name)
        r_cell = by_name.get(reverse_name)
        out.append(
            {
                "branch": label,
                "cal_normal": cell_summary(train, n_cell),
                "cal_reverse": cell_summary(train, r_cell),
                "hold_normal": cell_summary(hold, n_cell),
                "hold_reverse": cell_summary(hold, r_cell),
            }
        )
    return out


def cohort_drift(train: list[dict[str, Any]], hold: list[dict[str, Any]], by_name: dict[str, Cell]) -> list[dict[str, Any]]:
    out = []
    for label, normal_name, reverse_name in BRANCH_PAIRS:
        cell = by_name.get(normal_name) or by_name.get(reverse_name)
        cal_rows = selected_rows(train, cell)
        hold_rows = selected_rows(hold, cell)
        drift = {}
        for key in FEATURES:
            a = med(cal_rows, key)
            b = med(hold_rows, key)
            drift[key] = {"cal_median": a, "hold_median": b, "delta": r1((b or 0.0) - (a or 0.0)) if a is not None and b is not None else None}
        out.append({"branch": label, "cal_n": len(cal_rows), "hold_n": len(hold_rows), "drift": drift})
    return out


def tail_attribution(hold: list[dict[str, Any]], by_name: dict[str, Cell]) -> list[dict[str, Any]]:
    out = []
    for label, normal_name, reverse_name in BRANCH_PAIRS:
        for side, name in (("normal", normal_name), ("reverse", reverse_name)):
            cell = by_name.get(name)
            rows = selected_rows(hold, cell)
            scored = []
            for r in rows:
                value = value_from_normal(normal_value(r), cell.direction if cell else "NORMAL")
                scored.append((value, r))
            scored.sort(key=lambda x: x[0])
            vals = [v for v, _ in scored]
            out.append(
                {
                    "branch": label,
                    "side": side,
                    "summary": summary(vals),
                    "top3_removed": r1(t3r(vals)) if vals else 0.0,
                    "worst3": event_cards(scored[:3]),
                    "best3": event_cards(list(reversed(scored[-3:]))),
                }
            )
    return out


def event_cards(items: list[tuple[float, dict[str, Any]]]) -> list[dict[str, Any]]:
    cards = []
    for value, row in items:
        cards.append(
            {
                "event_id": row.get("event_id"),
                "signal_utc": row.get("signal_utc"),
                "value_bps": r1(value),
                "threshold": row.get("threshold_usd"),
                "vdepth": row.get("vdepth_bps"),
                "prior4h": row.get("prior4h_bps"),
                "eth1h": row.get("eth1h_bps"),
                "btc4h": row.get("btc4h_bps"),
                "tags": row.get("tags"),
            }
        )
    return cards


def sequence_features(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    ordered = sorted(rows, key=ts)
    out = {}
    for i, row in enumerate(ordered):
        t = ts(row)
        prior_1h = [r for r in ordered if 0 < t - ts(r) <= 3_600_000]
        next_1h = [r for r in ordered if 0 < ts(r) - t <= 3_600_000]
        near_15m = [r for r in ordered if 0 <= abs(ts(r) - t) <= 900_000]
        out[str(row.get("event_id"))] = {
            "prior_1h_n": len(prior_1h),
            "next_1h_n": len(next_1h),
            "near_15m_n": len(near_15m),
            "near_15m_threshold_count": len({int(float(r.get("threshold_usd") or 0)) for r in near_15m}),
            "prior_1h_tail150_n": sum(1 for r in prior_1h if normal_value(r) <= -150.0),
        }
    return out


def sequence_anatomy(all_rows: list[dict[str, Any]], hold: list[dict[str, Any]], by_name: dict[str, Cell]) -> list[dict[str, Any]]:
    seq = sequence_features(all_rows)
    out = []
    for label, normal_name, reverse_name in BRANCH_PAIRS:
        cell = by_name.get(normal_name) or by_name.get(reverse_name)
        rows = selected_rows(hold, cell)
        winners = [r for r in rows if normal_value(r) > 0]
        losers = [r for r in rows if normal_value(r) <= -100.0]
        def avg_seq(items: list[dict[str, Any]], key: str) -> float | None:
            vals = [float(seq.get(str(r.get("event_id")), {}).get(key) or 0.0) for r in items]
            return r1(sum(vals) / len(vals)) if vals else None
        out.append(
            {
                "branch": label,
                "hold_n": len(rows),
                "winner_seq": {k: avg_seq(winners, k) for k in ("prior_1h_n", "next_1h_n", "near_15m_n", "near_15m_threshold_count", "prior_1h_tail150_n")},
                "loser_seq": {k: avg_seq(losers, k) for k in ("prior_1h_n", "next_1h_n", "near_15m_n", "near_15m_threshold_count", "prior_1h_tail150_n")},
            }
        )
    return out


def directional_entropy(s_normal: dict[str, Any], s_reverse: dict[str, Any]) -> dict[str, Any]:
    # Since reverse is mechanically related to normal, use sign balance and
    # direction dominance as a chop/clarity proxy.
    n_sum = float(s_normal.get("sum_bps") or 0.0)
    r_sum = float(s_reverse.get("sum_bps") or 0.0)
    total = abs(n_sum) + abs(r_sum)
    dominance = abs(n_sum - r_sum) / total if total > 0 else None
    p = abs(n_sum) / total if total > 0 else None
    entropy = None
    if p is not None and 0 < p < 1:
        entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    return {"dominance": r3(dominance), "entropy": r3(entropy), "clearer_side": "normal" if n_sum > r_sum else "reverse"}


def entropy_report(matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in matrix:
        out.append(
            {
                "branch": row["branch"],
                "cal": directional_entropy(row["cal_normal"], row["cal_reverse"]),
                "hold": directional_entropy(row["hold_normal"], row["hold_reverse"]),
            }
        )
    return out


def fold_lead_lag(fold_matrices: list[dict[str, Any]]) -> dict[str, Any]:
    branches = [b[0] for b in BRANCH_PAIRS]
    cal_by_branch: dict[str, list[float]] = defaultdict(list)
    hold_by_branch: dict[str, list[float]] = defaultdict(list)
    for fold in fold_matrices:
        for row in fold["pair_matrix"]:
            branch = row["branch"]
            cal_by_branch[branch].append(float(row["cal_normal"].get("t3r_bps") or 0.0))
            hold_by_branch[branch].append(float(row["hold_normal"].get("t3r_bps") or 0.0))
    pairs = []
    for a in branches:
        for b in branches:
            c = safe_corr(cal_by_branch[a], hold_by_branch[b])
            if c is not None:
                pairs.append({"cal_branch": a, "hold_branch": b, "pearson": r3(c)})
    pairs.sort(key=lambda x: abs(float(x["pearson"] or 0.0)), reverse=True)
    return {"top_abs_correlations": pairs[:20]}


def anti_crowding(fold_matrices: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for fold in fold_matrices:
        for row in fold["pair_matrix"]:
            rows.append(
                {
                    "fold": fold["fold"],
                    "branch": row["branch"],
                    "cal_normal_t3r": float(row["cal_normal"].get("t3r_bps") or 0.0),
                    "hold_normal_t3r": float(row["hold_normal"].get("t3r_bps") or 0.0),
                    "hold_reverse_t3r": float(row["hold_reverse"].get("t3r_bps") or 0.0),
                }
            )
    vals = sorted(r["cal_normal_t3r"] for r in rows)
    q75 = vals[int(0.75 * len(vals))] if vals else 0.0
    q90 = vals[int(0.90 * len(vals))] if vals else 0.0
    buckets = {
        "all": rows,
        "cal_top25pct": [r for r in rows if r["cal_normal_t3r"] >= q75],
        "cal_top10pct": [r for r in rows if r["cal_normal_t3r"] >= q90],
    }
    out = {}
    for name, items in buckets.items():
        out[name] = {
            "n": len(items),
            "hold_normal_t3r_sum": r1(sum(r["hold_normal_t3r"] for r in items)),
            "hold_reverse_t3r_sum": r1(sum(r["hold_reverse_t3r"] for r in items)),
            "normal_positive_n": sum(1 for r in items if r["hold_normal_t3r"] > 0),
            "reverse_positive_n": sum(1 for r in items if r["hold_reverse_t3r"] > 0),
        }
    return out


def action_simulation(hold: list[dict[str, Any]], top_cells: list[Cell]) -> dict[str, Any]:
    baseline = [normal_value(r) for r in hold]
    no_trade_vals = []
    invert_vals = []
    reduce_vals = []
    flagged_n = 0
    for row in hold:
        hit = any(c.selector(row) for c in top_cells)
        if hit:
            flagged_n += 1
            invert_vals.append(value_from_normal(normal_value(row), "REVERSE"))
            reduce_vals.append(normal_value(row) * 0.5)
        else:
            no_trade_vals.append(normal_value(row))
            invert_vals.append(normal_value(row))
            reduce_vals.append(normal_value(row))
    return {
        "flagged_n": flagged_n,
        "baseline_all_normal": summary(baseline),
        "no_trade_flagged": summary(no_trade_vals),
        "invert_flagged": summary(invert_vals),
        "half_size_flagged": summary(reduce_vals),
    }


def sign_flip_trigger(train: list[dict[str, Any]], hold: list[dict[str, Any]], by_name: dict[str, Cell]) -> dict[str, Any]:
    cal_stats = cell_stats(train, list(by_name.values()))
    top_names = [x["cell"].name for x in sorted(cal_stats.values(), key=lambda v: float(v["t3r"]), reverse=True)[:5]]
    top_cells = [by_name[n] for n in top_names if n in by_name]
    return {"top_cells": top_names, "action_sim": action_simulation(hold, top_cells)}


def run() -> dict[str, Any]:
    raw = load_jsonl(NAV_EVENTS)
    folds_raw = make_folds(raw, folds=5, min_train_frac=0.4)
    fold_reports = []
    for idx, (train_raw, hold_raw) in enumerate(folds_raw, start=1):
        train, hold, by_name = prepare_fold(train_raw, hold_raw)
        matrix = pair_matrix(train, hold, by_name)
        fold_reports.append(
            {
                "fold": idx,
                "train_n": len(train),
                "hold_n": len(hold),
                "hold_start": iso_ms(ts(hold[0])),
                "hold_end": iso_ms(ts(hold[-1])),
                "fold_state": fold_state(hold),
                "pair_matrix": matrix,
                "cohort_drift": cohort_drift(train, hold, by_name),
                "tail_attribution": tail_attribution(hold, by_name),
                "sequence_anatomy": sequence_anatomy(train + hold, hold, by_name),
                "directional_entropy": entropy_report(matrix),
                "sign_flip_trigger": sign_flip_trigger(train, hold, by_name),
            }
        )
    return {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "fold_count": len(fold_reports),
        "folds": fold_reports,
        "lead_lag": fold_lead_lag(fold_reports),
        "anti_crowding": anti_crowding(fold_reports),
    }


def fmt(s: dict[str, Any]) -> str:
    return (
        f"N={s.get('n')} sum={s.get('sum_bps')} med={s.get('median_bps')} "
        f"T3R={s.get('t3r_bps')} tail150={s.get('tail_lte_minus150_n')} maxLoss={s.get('max_loss_bps')}"
    )


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# S34 Navigation Branch Anatomy",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        "## Anti-Crowding",
        "",
        "| Bucket | N | Hold normal T3R sum | Hold reverse T3R sum | Normal positive | Reverse positive |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in result["anti_crowding"].items():
        lines.append(
            f"| {name} | {row['n']} | {row['hold_normal_t3r_sum']} | {row['hold_reverse_t3r_sum']} | "
            f"{row['normal_positive_n']} | {row['reverse_positive_n']} |"
        )

    lines.extend(["", "## Lead-Lag Top Correlations", ""])
    lines.append("| Cal branch | Hold branch | Pearson |")
    lines.append("| --- | --- | ---: |")
    for row in result["lead_lag"]["top_abs_correlations"][:12]:
        lines.append(f"| {row['cal_branch']} | {row['hold_branch']} | {row['pearson']} |")

    lines.extend(["", "## Fold Details", ""])
    for fold in result["folds"]:
        lines.extend(
            [
                f"### Fold {fold['fold']}",
                "",
                f"- Hold `{fold['hold_start']}` -> `{fold['hold_end']}`, N `{fold['hold_n']}`",
                f"- State: density/day `{fold['fold_state']['event_density_per_day']}`, tail150 rate `{fold['fold_state']['tail150_rate']}`, avg `{fold['fold_state']['avg']}`",
                "",
                "#### Pair Matrix",
                "",
                "| Branch | Cal normal | Cal reverse | Hold normal | Hold reverse |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in fold["pair_matrix"]:
            lines.append(
                f"| {row['branch']} | {fmt(row['cal_normal'])} | {fmt(row['cal_reverse'])} | "
                f"{fmt(row['hold_normal'])} | {fmt(row['hold_reverse'])} |"
            )
        lines.extend(["", "#### Action Simulation", ""])
        sim = fold["sign_flip_trigger"]["action_sim"]
        lines.append(f"- Top cells: `{fold['sign_flip_trigger']['top_cells']}`; flagged N `{sim['flagged_n']}`")
        lines.append(f"- Baseline: {fmt(sim['baseline_all_normal'])}")
        lines.append(f"- No-trade flagged: {fmt(sim['no_trade_flagged'])}")
        lines.append(f"- Invert flagged: {fmt(sim['invert_flagged'])}")
        lines.append(f"- Half-size flagged: {fmt(sim['half_size_flagged'])}")
        lines.extend(["", "#### Cohort Drift Highlights", ""])
        lines.append("| Branch | Cal N | Hold N | vdepth delta | prior4h delta | btc4h delta | threshold delta |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in fold["cohort_drift"]:
            d = row["drift"]
            lines.append(
                f"| {row['branch']} | {row['cal_n']} | {row['hold_n']} | "
                f"{d['vdepth_bps']['delta']} | {d['prior4h_bps']['delta']} | {d['btc4h_bps']['delta']} | {d['threshold_usd']['delta']} |"
            )
        lines.extend(["", "#### Sequence Anatomy", ""])
        lines.append("| Branch | Winner seq | Loser seq |")
        lines.append("| --- | --- | --- |")
        for row in fold["sequence_anatomy"]:
            lines.append(f"| {row['branch']} | `{row['winner_seq']}` | `{row['loser_seq']}` |")
        lines.extend(["", "#### Tail Attribution", ""])
        lines.append("| Branch | Side | Summary | Worst event | Best event |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in fold["tail_attribution"]:
            worst = row["worst3"][0] if row["worst3"] else {}
            best = row["best3"][0] if row["best3"] else {}
            lines.append(f"| {row['branch']} | {row['side']} | {fmt(row['summary'])} | `{worst}` | `{best}` |")
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = run()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(result)
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
