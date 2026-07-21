"""S34 navigation stress-action and scalping tests.

Research-only:
- Compare REGIME_STRESS actions: baseline vs no-trade vs half-size vs invert.
- Test whether short-horizon scalping exists in stress/sign-flip zones.
- Run both full navigation universe and v0.2/v0.3-like route subset.

No live executor, order logic, size, leverage, or .env changes.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_navigation_branch_anatomy import fold_state  # noqa: E402
from tools.s34_navigation_causal_gauntlet import normal_value, value_from_normal  # noqa: E402
from tools.s34_navigation_full_followup import DEFAULT_DB, NAV_EVENTS, load_jsonl, mark_at_or_after, r1, r3, summary  # noqa: E402
from tools.s34_navigation_regime_inversion_walkforward import (  # noqa: E402
    attach_preds,
    build_cells,
    cell_stats,
    make_folds,
    neighbors,
    t3r,
)
from tools.s34_navigation_causal_gauntlet import causal_preds  # noqa: E402

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_SCALP_AND_STRESS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_SCALP_AND_STRESS.md"

FEE_BPS_ROUND = 5.0
HORIZONS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ts(row: dict[str, Any]) -> int:
    return int(row.get("signal_ts_ms") or 0)


def horizon_net(conn: sqlite3.Connection, row: dict[str, Any], sec: int) -> tuple[float | None, float | None]:
    t = ts(row)
    entry = mark_at_or_after(conn, "ETHUSDT", t)
    exit_ = mark_at_or_after(conn, "ETHUSDT", t + sec * 1000)
    if not entry or not exit_ or entry[1] <= 0:
        return None, None
    raw = (exit_[1] - entry[1]) / entry[1] * 10_000.0
    return raw - FEE_BPS_ROUND, -raw - FEE_BPS_ROUND


def prepare_fold(train_raw: list[dict[str, Any]], hold_raw: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_normals = [normal_value(r) for r in train_raw]
    train_pred = causal_preds(neighbors(train_raw, train_raw, leave_one_out=True), train_normals)
    hold_pred = causal_preds(neighbors(train_raw, hold_raw, leave_one_out=False), train_normals)
    train = attach_preds(train_raw, train_pred)
    hold = attach_preds(hold_raw, hold_pred)
    cells = build_cells(train + hold)
    by_name = {c.name: c for c in cells}
    return train, hold, by_name


def route_v02(row: dict[str, Any]) -> bool:
    return (
        str(row.get("symbol")) == "ETHUSDT"
        and str(row.get("liq_side")) == "SELL"
        and int(float(row.get("threshold_usd") or 0)) == 200_000
        and 28.0 <= float(row.get("vdepth_bps") or 0.0) < 40.0
        and float(row.get("prior4h_bps") or 0.0) < -50.0
    )


def route_v03_like(row: dict[str, Any]) -> bool:
    # Same entry lane; v0.3 was primarily an exit/sizing shadow.
    return route_v02(row)


def top_cells(train: list[dict[str, Any]], by_name: dict[str, Any], n: int = 5) -> list[Any]:
    stats = cell_stats(train, list(by_name.values()))
    eligible = [v for v in stats.values() if float(v.get("t3r") or 0.0) > 0]
    eligible.sort(key=lambda v: float(v.get("t3r") or 0.0), reverse=True)
    return [by_name[v["cell"].name] for v in eligible[:n] if v["cell"].name in by_name]


def stress_score(fold: dict[str, Any], row: dict[str, Any], top_hit: bool) -> int:
    score = 0
    if float(fold["state"].get("event_density_per_day") or 0.0) >= 20.0:
        score += 1
    if float(fold["state"].get("tail150_rate") or 0.0) >= 0.06:
        score += 1
    if top_hit:
        score += 1
    if float(row.get("btc4h_bps") or 0.0) < -75.0:
        score += 1
    return score


def action_values(rows: list[dict[str, Any]], flags: dict[str, int], *, min_score: int, route_filter=None) -> dict[str, Any]:
    baseline = []
    no_trade = []
    half = []
    invert = []
    flagged = 0
    for row in rows:
        if route_filter and not route_filter(row):
            continue
        v = normal_value(row)
        flag = flags.get(str(row.get("event_id")), 0) >= min_score
        baseline.append(v)
        if flag:
            flagged += 1
            half.append(v * 0.5)
            invert.append(value_from_normal(v, "REVERSE"))
        else:
            no_trade.append(v)
            half.append(v)
            invert.append(v)
    return {
        "min_score": min_score,
        "n": len(baseline),
        "flagged_n": flagged,
        "baseline": summary(baseline),
        "no_trade_flagged": summary(no_trade),
        "half_size_flagged": summary(half),
        "invert_flagged": summary(invert),
    }


def scalp_by_group(conn: sqlite3.Connection, rows: list[dict[str, Any]], flags: dict[str, int], *, min_score: int, route_filter=None) -> dict[str, Any]:
    groups = {"all": [], "stress": [], "nonstress": []}
    for row in rows:
        if route_filter and not route_filter(row):
            continue
        groups["all"].append(row)
        groups["stress" if flags.get(str(row.get("event_id")), 0) >= min_score else "nonstress"].append(row)
    out = {}
    for gname, items in groups.items():
        cells = {}
        for h, sec in HORIZONS.items():
            normal_vals = []
            reverse_vals = []
            for row in items:
                normal, reverse = horizon_net(conn, row, sec)
                if normal is not None:
                    normal_vals.append(normal)
                if reverse is not None:
                    reverse_vals.append(reverse)
            cells[h] = {"normal": summary(normal_vals), "reverse": summary(reverse_vals)}
        out[gname] = {"n": len(items), "horizons": cells}
    return out


def fold_run(conn: sqlite3.Connection, train_raw: list[dict[str, Any]], hold_raw: list[dict[str, Any]], idx: int) -> dict[str, Any]:
    train, hold, by_name = prepare_fold(train_raw, hold_raw)
    selected = top_cells(train, by_name, 5)
    state = fold_state(hold)
    fold_meta = {"state": state}
    flags = {}
    for row in hold:
        hit = any(c.selector(row) for c in selected)
        flags[str(row.get("event_id"))] = stress_score(fold_meta, row, hit)
    return {
        "fold": idx,
        "hold_n": len(hold),
        "hold_start": hold[0].get("signal_utc"),
        "hold_end": hold[-1].get("signal_utc"),
        "state": state,
        "top_cells": [c.name for c in selected],
        "stress_score_mix": dict(sorted({i: sum(1 for v in flags.values() if v == i) for i in range(5)}.items())),
        "actions_full": {f"score_ge_{s}": action_values(hold, flags, min_score=s) for s in (1, 2, 3)},
        "actions_v02": {f"score_ge_{s}": action_values(hold, flags, min_score=s, route_filter=route_v02) for s in (1, 2, 3)},
        "scalp_full": {f"score_ge_{s}": scalp_by_group(conn, hold, flags, min_score=s) for s in (1, 2, 3)},
        "scalp_v02": {f"score_ge_{s}": scalp_by_group(conn, hold, flags, min_score=s, route_filter=route_v02) for s in (1, 2, 3)},
    }


def aggregate_actions(folds: list[dict[str, Any]], key: str) -> dict[str, Any]:
    out = {}
    for scope in ("actions_full", "actions_v02"):
        rows = []
        for fold in folds:
            cell = fold[scope][key]
            rows.append(cell)
        out[scope] = {
            "folds": len(rows),
            "baseline_t3r_sum": r1(sum(float(r["baseline"].get("t3r_bps") or 0.0) for r in rows)),
            "no_trade_t3r_sum": r1(sum(float(r["no_trade_flagged"].get("t3r_bps") or 0.0) for r in rows)),
            "half_t3r_sum": r1(sum(float(r["half_size_flagged"].get("t3r_bps") or 0.0) for r in rows)),
            "invert_t3r_sum": r1(sum(float(r["invert_flagged"].get("t3r_bps") or 0.0) for r in rows)),
            "flagged_n_sum": sum(int(r["flagged_n"]) for r in rows),
            "n_sum": sum(int(r["n"]) for r in rows),
        }
    return out


def aggregate_scalp(folds: list[dict[str, Any]], score_key: str, scope: str) -> dict[str, Any]:
    out = {}
    for group in ("all", "stress", "nonstress"):
        out[group] = {}
        for h in HORIZONS:
            normal_t3r = []
            reverse_t3r = []
            normal_sum = []
            reverse_sum = []
            n_sum = 0
            for fold in folds:
                cell = fold[scope][score_key][group]
                n_sum += int(cell["n"])
                hs = cell["horizons"][h]
                normal_t3r.append(float(hs["normal"].get("t3r_bps") or 0.0))
                reverse_t3r.append(float(hs["reverse"].get("t3r_bps") or 0.0))
                normal_sum.append(float(hs["normal"].get("sum_bps") or 0.0))
                reverse_sum.append(float(hs["reverse"].get("sum_bps") or 0.0))
            out[group][h] = {
                "n_sum": n_sum,
                "normal_fold_t3r_sum": r1(sum(normal_t3r)),
                "reverse_fold_t3r_sum": r1(sum(reverse_t3r)),
                "normal_fold_sum_sum": r1(sum(normal_sum)),
                "reverse_fold_sum_sum": r1(sum(reverse_sum)),
            }
    return out


def run() -> dict[str, Any]:
    rows = load_jsonl(NAV_EVENTS)
    raw_folds = make_folds(rows, folds=5, min_train_frac=0.4)
    with sqlite3.connect(DEFAULT_DB) as conn:
        folds = [fold_run(conn, train, hold, i) for i, (train, hold) in enumerate(raw_folds, start=1)]
    return {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "fold_count": len(folds),
        "folds": folds,
        "aggregate_actions": {f"score_ge_{s}": aggregate_actions(folds, f"score_ge_{s}") for s in (1, 2, 3)},
        "aggregate_scalp_full": {f"score_ge_{s}": aggregate_scalp(folds, f"score_ge_{s}", "scalp_full") for s in (1, 2, 3)},
        "aggregate_scalp_v02": {f"score_ge_{s}": aggregate_scalp(folds, f"score_ge_{s}", "scalp_v02") for s in (1, 2, 3)},
    }


def fmt(s: dict[str, Any]) -> str:
    return (
        f"N={s.get('n')} sum={s.get('sum_bps')} med={s.get('median_bps')} "
        f"T3R={s.get('t3r_bps')} tail150={s.get('tail_lte_minus150_n')} maxLoss={s.get('max_loss_bps')}"
    )


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# S34 Navigation Scalp And Stress",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        "## Stress Action Comparison",
        "",
        "| Score | Scope | N | Flagged | Baseline T3R | No-trade T3R | Half-size T3R | Invert T3R |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for score, cell in result["aggregate_actions"].items():
        for scope, row in cell.items():
            lines.append(
                f"| {score} | {scope} | {row['n_sum']} | {row['flagged_n_sum']} | "
                f"{row['baseline_t3r_sum']} | {row['no_trade_t3r_sum']} | {row['half_t3r_sum']} | {row['invert_t3r_sum']} |"
            )

    lines.extend(["", "## Scalping Aggregate - Full Universe", ""])
    write_scalp_table(lines, result["aggregate_scalp_full"])
    lines.extend(["", "## Scalping Aggregate - v0.2/v0.3 Route", ""])
    write_scalp_table(lines, result["aggregate_scalp_v02"])

    lines.extend(["", "## Fold Details", ""])
    for fold in result["folds"]:
        lines.extend(
            [
                f"### Fold {fold['fold']}",
                "",
                f"- Hold `{fold['hold_start']}` -> `{fold['hold_end']}`, N `{fold['hold_n']}`",
                f"- State density/day `{fold['state']['event_density_per_day']}`, tail150 rate `{fold['state']['tail150_rate']}`",
                f"- Top cells: `{fold['top_cells']}`",
                f"- Stress score mix: `{fold['stress_score_mix']}`",
                "",
            ]
        )
        for score in ("score_ge_1", "score_ge_2", "score_ge_3"):
            a = fold["actions_full"][score]
            lines.append(
                f"- Full {score}: flagged `{a['flagged_n']}/{a['n']}`, baseline {fmt(a['baseline'])}, "
                f"no-trade {fmt(a['no_trade_flagged'])}, half {fmt(a['half_size_flagged'])}, invert {fmt(a['invert_flagged'])}"
            )
            b = fold["actions_v02"][score]
            lines.append(
                f"- v0.2 {score}: flagged `{b['flagged_n']}/{b['n']}`, baseline {fmt(b['baseline'])}, "
                f"no-trade {fmt(b['no_trade_flagged'])}, half {fmt(b['half_size_flagged'])}, invert {fmt(b['invert_flagged'])}"
            )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scalp_table(lines: list[str], block: dict[str, Any]) -> None:
    lines.append("| Score | Group | Horizon | N | Normal T3R | Reverse T3R | Normal Sum | Reverse Sum |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for score, score_block in block.items():
        for group, horizons in score_block.items():
            for h, row in horizons.items():
                lines.append(
                    f"| {score} | {group} | {h} | {row['n_sum']} | {row['normal_fold_t3r_sum']} | "
                    f"{row['reverse_fold_t3r_sum']} | {row['normal_fold_sum_sum']} | {row['reverse_fold_sum_sum']} |"
                )


def main() -> None:
    result = run()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(result)
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
