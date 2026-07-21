"""S34 stress-scalp live-readiness tests.

Research-only follow-up before any live/paper wiring:
- de-lookahead stress score (train-state only);
- causal chain count (past/current only) vs near-window chain;
- entry delay decay;
- exit robustness under causal variants;
- short execution realism;
- duplicate/overlap guard simulation;
- stress bucket kill simulation;
- v0.2 conflict matrix.

No live executor, order logic, size, leverage, config, or env changes.
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
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_navigation_branch_anatomy import fold_state  # noqa: E402
from tools.s34_navigation_causal_gauntlet import causal_preds, normal_value  # noqa: E402
from tools.s34_navigation_full_followup import DEFAULT_DB, NAV_EVENTS, load_jsonl, mark_at_or_after, r1, r3, summary  # noqa: E402
from tools.s34_navigation_regime_inversion_walkforward import attach_preds, build_cells, cell_stats, make_folds, neighbors  # noqa: E402
from tools.s34_navigation_scalp_and_stress import route_v02  # noqa: E402
from tools.s34_stress_reaction_deep_tests import BASE_FEE_BPS, bracket_outcome, mark_series  # noqa: E402

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_STRESS_SCALP_LIVE_READINESS_TESTS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_STRESS_SCALP_LIVE_READINESS_TESTS.md"

TP = 200.0
SL = 40.0
HORIZON = 1200


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ts(row: dict[str, Any]) -> int:
    return int(row.get("signal_ts_ms") or 0)


def t3r(vals: list[float]) -> float:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    return float(sum(sorted(vals, reverse=True)[3:])) if len(vals) > 3 else float(sum(vals))


def top_cells(train: list[dict[str, Any]], by_name: dict[str, Any], n: int = 5) -> list[Any]:
    stats = cell_stats(train, list(by_name.values()))
    eligible = [v for v in stats.values() if float(v.get("t3r") or 0.0) > 0]
    eligible.sort(key=lambda v: float(v.get("t3r") or 0.0), reverse=True)
    return [by_name[v["cell"].name] for v in eligible[:n] if v["cell"].name in by_name]


def build_live_like_rows() -> list[dict[str, Any]]:
    raw = load_jsonl(NAV_EVENTS)
    folds = make_folds(raw, folds=5, min_train_frac=0.4)
    out: list[dict[str, Any]] = []
    for fold_idx, (train_raw, hold_raw) in enumerate(folds, start=1):
        train_normals = [normal_value(r) for r in train_raw]
        train_pred = causal_preds(neighbors(train_raw, train_raw, leave_one_out=True), train_normals)
        hold_pred = causal_preds(neighbors(train_raw, hold_raw, leave_one_out=False), train_normals)
        train = attach_preds(train_raw, train_pred)
        hold = attach_preds(hold_raw, hold_pred)
        by_name = {c.name: c for c in build_cells(train + hold)}
        selected = top_cells(train, by_name, 5)
        train_state = fold_state(train)
        hold_state = fold_state(hold)
        for row in hold:
            top_hit = any(c.selector(row) for c in selected)
            live_score = 0
            hold_score = 0
            if float(train_state.get("event_density_per_day") or 0.0) >= 20.0:
                live_score += 1
            if float(train_state.get("tail150_rate") or 0.0) >= 0.06:
                live_score += 1
            if float(hold_state.get("event_density_per_day") or 0.0) >= 20.0:
                hold_score += 1
            if float(hold_state.get("tail150_rate") or 0.0) >= 0.06:
                hold_score += 1
            if top_hit:
                live_score += 1
                hold_score += 1
            if float(row.get("btc4h_bps") or 0.0) < -75.0:
                live_score += 1
                hold_score += 1
            item = dict(row)
            item["fold"] = fold_idx
            item["top_hit"] = top_hit
            item["stress_score_live_like"] = live_score
            item["stress_score_original_holdstate"] = hold_score
            item["train_state"] = {
                "event_density_per_day": train_state.get("event_density_per_day"),
                "tail150_rate": train_state.get("tail150_rate"),
            }
            item["hold_state"] = {
                "event_density_per_day": hold_state.get("event_density_per_day"),
                "tail150_rate": hold_state.get("tail150_rate"),
            }
            out.append(item)
    return enrich_chain_counts(out)


def enrich_chain_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=ts)
    out = []
    for row in ordered:
        t = ts(row)
        near = [r for r in ordered if 0 <= abs(ts(r) - t) <= 900_000]
        causal = [r for r in ordered if 0 <= t - ts(r) <= 900_000]
        prior = [r for r in ordered if 0 < t - ts(r) <= 900_000]
        item = dict(row)
        item["chain_near_15m_thresholds"] = len({int(float(r.get("threshold_usd") or 0)) for r in near})
        item["chain_causal_15m_thresholds"] = len({int(float(r.get("threshold_usd") or 0)) for r in causal})
        item["chain_prior_15m_thresholds"] = len({int(float(r.get("threshold_usd") or 0)) for r in prior})
        item["chain_causal_15m_n"] = len(causal)
        item["chain_prior_15m_n"] = len(prior)
        out.append(item)
    return out


def selector_factory(score_key: str, chain_key: str, chain_min: int = 3) -> Callable[[dict[str, Any]], bool]:
    return lambda r: (
        int(r.get(score_key) or 0) >= 3
        and float(r.get("btc4h_bps") or 0.0) < -75.0
        and float(r.get("vdepth_bps") or 0.0) < 50.0
        and int(r.get(chain_key) or 0) >= chain_min
    )


SELECTORS = {
    "original_holdstate_near3": selector_factory("stress_score_original_holdstate", "chain_near_15m_thresholds", 3),
    "live_like_near3": selector_factory("stress_score_live_like", "chain_near_15m_thresholds", 3),
    "live_like_causal3": selector_factory("stress_score_live_like", "chain_causal_15m_thresholds", 3),
    "live_like_prior3": selector_factory("stress_score_live_like", "chain_prior_15m_thresholds", 3),
    "live_like_causal2": selector_factory("stress_score_live_like", "chain_causal_15m_thresholds", 2),
}


def bracket_from_entry(
    conn: sqlite3.Connection,
    *,
    entry_ts_ms: int,
    entry_px: float,
    horizon_sec: int,
    direction: str,
    tp: float,
    sl: float,
    fee_bps: float = BASE_FEE_BPS,
) -> tuple[float | None, str, int | None]:
    series = mark_series(conn, entry_ts_ms, entry_ts_ms + horizon_sec * 1000)
    if not series or entry_px <= 0:
        return None, "NO_SERIES", None
    for t, px in series:
        raw = (float(px) - entry_px) / entry_px * 10_000.0
        pnl = raw if direction == "NORMAL" else -raw
        if pnl >= tp:
            return tp - fee_bps, "TP", int((int(t) - entry_ts_ms) / 1000)
        if pnl <= -sl:
            return -sl - fee_bps, "SL", int((int(t) - entry_ts_ms) / 1000)
    end_ts, end_px = series[-1]
    raw = (float(end_px) - entry_px) / entry_px * 10_000.0
    pnl = raw if direction == "NORMAL" else -raw
    return pnl - fee_bps, "TIME", int((int(end_ts) - entry_ts_ms) / 1000)


def outcome(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    *,
    delay_sec: int = 0,
    tp: float = TP,
    sl: float = SL,
    horizon_sec: int = HORIZON,
    fee_bps: float = BASE_FEE_BPS,
) -> tuple[float | None, str, int | None]:
    entry = mark_at_or_after(conn, "ETHUSDT", ts(row) + delay_sec * 1000)
    if not entry:
        return None, "NO_ENTRY", None
    return bracket_from_entry(
        conn,
        entry_ts_ms=int(entry[0]),
        entry_px=float(entry[1]),
        horizon_sec=horizon_sec,
        direction="REVERSE",
        tp=tp,
        sl=sl,
        fee_bps=fee_bps,
    )


def eval_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]], selector: Callable[[dict[str, Any]], bool], **kwargs: Any) -> dict[str, Any]:
    vals = []
    exits: dict[str, int] = defaultdict(int)
    for row in rows:
        if not selector(row):
            continue
        val, exit_, _ = outcome(conn, row, **kwargs)
        if val is not None:
            vals.append(float(val))
            exits[str(exit_)] += 1
    return {"matched_n": len([r for r in rows if selector(r)]), "summary": summary(vals), "exits": dict(exits)}


def fold_eval(conn: sqlite3.Connection, rows: list[dict[str, Any]], selector: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    out = {}
    for fold in sorted({int(r.get("fold") or 0) for r in rows}):
        frows = [r for r in rows if int(r.get("fold") or 0) == fold]
        out[f"fold_{fold}"] = eval_rows(conn, frows, selector)
    return {
        "folds": out,
        "positive_t3r_folds": sum(1 for r in out.values() if float(r["summary"].get("t3r_bps") or 0.0) > 0),
        "positive_sum_folds": sum(1 for r in out.values() if float(r["summary"].get("sum_bps") or 0.0) > 0),
        "fold_t3r_total": r1(sum(float(r["summary"].get("t3r_bps") or 0.0) for r in out.values())),
    }


def non_overlap_rows(rows: list[dict[str, Any]], window_sec: int) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    end = -1
    for row in sorted(rows, key=ts):
        t = ts(row)
        if not cur or t <= end:
            cur.append(row)
            end = max(end, t + window_sec * 1000)
        else:
            groups.append(cur)
            cur = [row]
            end = t + window_sec * 1000
    if cur:
        groups.append(cur)
    return [min(g, key=ts) for g in groups]


def non_overlap_eval(conn: sqlite3.Connection, rows: list[dict[str, Any]], selector: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    target = [r for r in rows if selector(r)]
    out = {"overlap": eval_rows(conn, rows, selector)}
    for window in (900, 1800, 3600):
        selected = non_overlap_rows(target, window)
        vals = []
        exits: dict[str, int] = defaultdict(int)
        for row in selected:
            val, exit_, _ = outcome(conn, row)
            if val is not None:
                vals.append(float(val))
                exits[str(exit_)] += 1
        out[f"nonoverlap_{int(window/60)}m_first"] = {"matched_n": len(selected), "summary": summary(vals), "exits": dict(exits)}
    return out


def entry_delay(conn: sqlite3.Connection, rows: list[dict[str, Any]], selector: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    return {
        f"delay_{d}s": eval_rows(conn, rows, selector, delay_sec=d)
        for d in (0, 5, 15, 30, 60)
    }


def exit_robustness(conn: sqlite3.Connection, rows: list[dict[str, Any]], selector: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    variants = [
        ("TP150_SL30_15M", 150.0, 30.0, 900),
        ("TP200_SL40_20M", 200.0, 40.0, 1200),
        ("TP250_SL50_30M", 250.0, 50.0, 1800),
        ("TP200_SL50_20M", 200.0, 50.0, 1200),
        ("TRAIL_PROXY_TP150_SL40_20M", 150.0, 40.0, 1200),
    ]
    return {name: eval_rows(conn, rows, selector, tp=tp, sl=sl, horizon_sec=sec) for name, tp, sl, sec in variants}


def passive_short_entry(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    *,
    offset_bps: float,
    wait_sec: int,
    fallback: bool,
) -> tuple[int | None, float | None, str]:
    entry = mark_at_or_after(conn, "ETHUSDT", ts(row))
    if not entry:
        return None, None, "NO_ENTRY"
    entry_ts, entry_px = int(entry[0]), float(entry[1])
    limit_px = entry_px * (1.0 + offset_bps / 10_000.0)
    series = mark_series(conn, entry_ts, entry_ts + wait_sec * 1000)
    for t, px in series:
        if float(px) >= limit_px:
            return int(t), float(limit_px), "PASSIVE_FILL"
    if fallback:
        fb = mark_at_or_after(conn, "ETHUSDT", entry_ts + wait_sec * 1000)
        if fb:
            return int(fb[0]), float(fb[1]), "FALLBACK_TAKER"
    return None, None, "NO_FILL"


def execution_realism(conn: sqlite3.Connection, rows: list[dict[str, Any]], selector: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    target = [r for r in rows if selector(r)]
    out: dict[str, Any] = {"taker_immediate": eval_rows(conn, rows, selector)}
    for offset in (5.0, 10.0, 20.0):
        for wait in (15, 30, 60):
            for fallback in (False, True):
                vals = []
                exits: dict[str, int] = defaultdict(int)
                kinds: dict[str, int] = defaultdict(int)
                nofill = []
                for row in target:
                    fill_ts, fill_px, kind = passive_short_entry(conn, row, offset_bps=offset, wait_sec=wait, fallback=fallback)
                    kinds[kind] += 1
                    if fill_ts is None or fill_px is None:
                        val, _, _ = outcome(conn, row)
                        if val is not None:
                            nofill.append(float(val))
                        continue
                    val, exit_, _ = bracket_from_entry(
                        conn,
                        entry_ts_ms=fill_ts,
                        entry_px=fill_px,
                        horizon_sec=HORIZON,
                        direction="REVERSE",
                        tp=TP,
                        sl=SL,
                    )
                    if val is not None:
                        vals.append(float(val))
                        exits[str(exit_)] += 1
                name = f"{'passive_then_taker' if fallback else 'passive_only'}_off{offset:g}_wait{wait}s"
                out[name] = {
                    "fill_rate": r3(len(vals) / len(target)) if target else None,
                    "fill_kinds": dict(kinds),
                    "summary": summary(vals),
                    "exits": dict(exits),
                    "no_fill_counterfactual": summary(nofill),
                }
    return out


def duplicate_guard(conn: sqlite3.Connection, rows: list[dict[str, Any]], selector: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    target = [r for r in rows if selector(r)]
    out = {"raw": eval_rows(conn, rows, selector)}
    for window in (60, 300, 900, 1800):
        selected = non_overlap_rows(target, window)
        vals = []
        exits: dict[str, int] = defaultdict(int)
        for row in selected:
            val, exit_, _ = outcome(conn, row)
            if val is not None:
                vals.append(float(val))
                exits[str(exit_)] += 1
        out[f"dedup_{window}s_first"] = {"matched_n": len(selected), "summary": summary(vals), "exits": dict(exits)}
    return out


def kill_rule_sim(conn: sqlite3.Connection, rows: list[dict[str, Any]], selector: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    scored = []
    for row in sorted([r for r in rows if selector(r)], key=ts):
        val, exit_, _ = outcome(conn, row)
        if val is not None:
            scored.append((ts(row), float(val), str(exit_), row))
    rules: dict[str, Callable[[list[float], float], bool]] = {
        "pause_after_1_sl": lambda hist, v: v <= -40.0,
        "pause_after_2_sl": lambda hist, v: len([x for x in hist[-2:] if x <= -40.0]) >= 2,
        "rolling_3_sum_lt_-90": lambda hist, v: sum(hist[-3:]) < -90.0,
        "rolling_5_sum_lt_-120": lambda hist, v: sum(hist[-5:]) < -120.0,
        "daily_loss_lt_-90": lambda hist, v: False,
    }
    out = {}
    for name, rule in rules.items():
        active_vals = []
        hist = []
        paused_until = -1
        pauses = 0
        for t, v, _, _ in scored:
            if t < paused_until:
                continue
            active_vals.append(v)
            hist.append(v)
            if rule(hist, v):
                pauses += 1
                paused_until = t + 24 * 3600 * 1000
                hist = []
        out[name] = {"traded_n": len(active_vals), "pauses": pauses, "summary": summary(active_vals)}
    return out


def conflict_matrix(conn: sqlite3.Connection, rows: list[dict[str, Any]], selector: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    stress = [r for r in rows if selector(r)]
    v02 = [r for r in rows if route_v02(r)]
    conflicts = []
    for s in stress:
        close = [v for v in v02 if abs(ts(v) - ts(s)) <= 900_000]
        if close:
            conflicts.append(s)
    vals = []
    exits: dict[str, int] = defaultdict(int)
    for row in conflicts:
        val, exit_, _ = outcome(conn, row)
        if val is not None:
            vals.append(float(val))
            exits[str(exit_)] += 1
    return {
        "stress_n": len(stress),
        "v02_n": len(v02),
        "conflict_15m_n": len(conflicts),
        "conflict_stress_summary": summary(vals),
        "conflict_exits": dict(exits),
    }


def feature_availability() -> dict[str, Any]:
    return {
        "btc4h_bps": {"knowable": True, "reason": "prior return ending at signal timestamp"},
        "vdepth_bps": {"knowable": True, "reason": "anchor-local V depth as computed at anchor in current research objects; must be recomputed from past/current marks in live"},
        "top_hit": {"knowable": True, "reason": "selected from prior calibration cells; live must use frozen cells only"},
        "train_state_density_tail": {"knowable": True, "reason": "computed from prior calibration/train rows only"},
        "hold_state_density_tail": {"knowable": False, "reason": "uses complete future holdout fold; research-only contamination"},
        "near_15m_thresholds": {"knowable": False, "reason": "includes future events after current timestamp"},
        "causal_15m_thresholds": {"knowable": True, "reason": "uses events with ts<=current only"},
        "prior_15m_thresholds": {"knowable": True, "reason": "uses events before current timestamp only"},
    }


def run() -> dict[str, Any]:
    rows = build_live_like_rows()
    final_hold = [r for r in rows if int(r.get("fold") or 0) >= 4]
    with sqlite3.connect(DEFAULT_DB) as conn:
        selector_results = {
            name: {
                "all": eval_rows(conn, rows, sel),
                "final_hold": eval_rows(conn, final_hold, sel),
                "walkforward": fold_eval(conn, rows, sel),
                "non_overlap": non_overlap_eval(conn, rows, sel),
            }
            for name, sel in SELECTORS.items()
        }
        primary = SELECTORS["live_like_causal3"]
        result = {
            "generated_at_utc": utc_now(),
            "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
            "feature_availability": feature_availability(),
            "selector_comparison": selector_results,
            "entry_delay_live_like_causal3": entry_delay(conn, rows, primary),
            "exit_robustness_live_like_causal3": exit_robustness(conn, rows, primary),
            "execution_realism_live_like_causal3": execution_realism(conn, rows, primary),
            "duplicate_guard_live_like_causal3": duplicate_guard(conn, rows, primary),
            "kill_rule_sim_live_like_causal3": kill_rule_sim(conn, rows, primary),
            "conflict_matrix_live_like_causal3": conflict_matrix(conn, rows, primary),
        }
    return result


def fmt(s: dict[str, Any]) -> str:
    return (
        f"N={s.get('n')} sum={s.get('sum_bps')} med={s.get('median_bps')} "
        f"T3R={s.get('t3r_bps')} tail150={s.get('tail_lte_minus150_n')} maxLoss={s.get('max_loss_bps')}"
    )


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# S34 Stress Scalp Live Readiness Tests",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        "## 1. Feature Availability",
        "",
    ]
    for name, row in result["feature_availability"].items():
        lines.append(f"- `{name}`: knowable=`{row['knowable']}`; {row['reason']}")

    lines.extend(["", "## 2. Causal Chain / Stress Score Rebuild", ""])
    lines.append("| Selector | All | Final hold | Positive T3R folds | 15m non-overlap first |")
    lines.append("| --- | --- | --- | ---: | --- |")
    for name, row in result["selector_comparison"].items():
        lines.append(
            f"| `{name}` | {fmt(row['all']['summary'])} | {fmt(row['final_hold']['summary'])} | "
            f"{row['walkforward']['positive_t3r_folds']}/5 | {fmt(row['non_overlap']['nonoverlap_15m_first']['summary'])} |"
        )

    lines.extend(["", "## 3. Entry Delay (live_like_causal3)", ""])
    lines.append("| Delay | Summary | Exits |")
    lines.append("| --- | --- | --- |")
    for name, row in result["entry_delay_live_like_causal3"].items():
        lines.append(f"| `{name}` | {fmt(row['summary'])} | `{row['exits']}` |")

    lines.extend(["", "## 4. Exit Robustness (live_like_causal3)", ""])
    lines.append("| Exit | Summary | Exits |")
    lines.append("| --- | --- | --- |")
    for name, row in result["exit_robustness_live_like_causal3"].items():
        lines.append(f"| `{name}` | {fmt(row['summary'])} | `{row['exits']}` |")

    lines.extend(["", "## 5. SHORT Execution Realism (live_like_causal3)", ""])
    lines.append("| Model | Fill rate | Summary | Fill kinds | No-fill counterfactual |")
    lines.append("| --- | ---: | --- | --- | --- |")
    for name, row in result["execution_realism_live_like_causal3"].items():
        lines.append(
            f"| `{name}` | {row.get('fill_rate')} | {fmt(row['summary'])} | "
            f"`{row.get('fill_kinds', {})}` | {fmt(row.get('no_fill_counterfactual', {}))} |"
        )

    lines.extend(["", "## 6. Duplicate / Overlap Guard (live_like_causal3)", ""])
    lines.append("| Policy | Summary | Exits |")
    lines.append("| --- | --- | --- |")
    for name, row in result["duplicate_guard_live_like_causal3"].items():
        lines.append(f"| `{name}` | {fmt(row['summary'])} | `{row['exits']}` |")

    lines.extend(["", "## 7. Stress Bucket Kill Rule Simulation (live_like_causal3)", ""])
    lines.append("| Rule | Traded N | Pauses | Summary |")
    lines.append("| --- | ---: | ---: | --- |")
    for name, row in result["kill_rule_sim_live_like_causal3"].items():
        lines.append(f"| `{name}` | {row['traded_n']} | {row['pauses']} | {fmt(row['summary'])} |")

    lines.extend(["", "## 8. v0.2 Conflict Matrix (live_like_causal3)", ""])
    cm = result["conflict_matrix_live_like_causal3"]
    lines.append(f"- stress N: `{cm['stress_n']}`")
    lines.append(f"- v0.2 N: `{cm['v02_n']}`")
    lines.append(f"- conflict within 15m N: `{cm['conflict_15m_n']}`")
    lines.append(f"- conflict stress summary: {fmt(cm['conflict_stress_summary'])}; exits `{cm['conflict_exits']}`")

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
