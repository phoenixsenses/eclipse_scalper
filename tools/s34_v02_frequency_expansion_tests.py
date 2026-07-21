"""S34 V02 frequency expansion tests.

Research-only. Maps ways to increase trade frequency around the live V02
SELL-maker-long lane without touching live/paper/executor state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import load_mark_index, pctile, r1, r3, signed_return_bps  # noqa: E402
from tools.s34_v02_event_chain_puzzle_tests import ASSET_THRESHOLDS, build_events, metrics, neighbor_events  # noqa: E402
from tools.s34_v02_next_navigation_tests import add_states  # noqa: E402
from tools.s34_v02_propagation_puzzle_suite import enrich_events  # noqa: E402


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V02_FREQUENCY_EXPANSION_TESTS.json"
OUT_MD = OUT_DIR / "S34_V02_FREQUENCY_EXPANSION_TESTS.md"
HANDOFF_MD = OUT_DIR / "S34_CLAUDE_COMPREHENSIVE_HANDOFF.md"
H4_LEDGER_CSV = OUT_DIR / "S34_V02_H4_FORWARD_SHADOW_LEDGER.csv"

SYMBOL = "ETHUSDT"
THRESHOLDS = (50_000.0, 100_000.0, 150_000.0, 200_000.0, 300_000.0)
TAUS_SEC = (30, 60, 120, 300, 600, 900, 1800)
HORIZONS = {"M15": 900, "H1": 3600, "H2": 7200, "H4": 14400}
MAKER_OFFSETS_BPS = (0.0, 5.0, 10.0, 20.0)
LIVE_V02_RULE = "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID"
TAKER_COST_BPS = 8.0
MAKER_PROXY_COST_BPS = 4.0
MAKER_PROXY_FILL_WINDOW_SEC = 900
MAKER_PROXY_CROSS_BPS = 1.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def month_of(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).strftime("%Y-%m")


def split_months(rows: list[dict[str, Any]]) -> tuple[set[str], dict[str, Any]]:
    months = sorted({str(r.get("month")) for r in rows if r.get("month")})
    hold_n = max(1, round(len(months) * 0.35)) if months else 0
    hold = set(months[-hold_n:]) if hold_n else set()
    return hold, {"method": "chronological_month_tail_35pct", "months": months, "holdout_months": sorted(hold)}


def pass_read(m: dict[str, Any], *, min_n: int = 40) -> bool:
    return (
        int(m.get("n") or m.get("filled_n") or 0) >= min_n
        and float(m.get("sum_bps") or 0.0) > 0.0
        and float(m.get("t3r_bps") or 0.0) > 0.0
        and (m.get("median_bps") is None or float(m.get("median_bps") or 0.0) > 0.0)
    )


def mark_at_or_after(marks: Any, ts_ms: int) -> tuple[int, float] | None:
    row = marks.at_or_after(int(ts_ms))
    if not row:
        return None
    return int(row[0]), float(row[1])


def load_live_signal_times() -> set[int]:
    if not H4_LEDGER_CSV.exists():
        return set()
    out: set[int] = set()
    with H4_LEDGER_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("protocol_id") == LIVE_V02_RULE and row.get("observation_status") == "CLOSED":
                try:
                    out.add(int(row["signal_ts_ms"]))
                except (KeyError, TypeError, ValueError):
                    pass
    return out


def near_any(ts_ms: int, times: set[int], tolerance_ms: int = 2_000) -> bool:
    if not times:
        return False
    ordered = sorted(times)
    pos = bisect_left(ordered, int(ts_ms))
    for j in (pos - 1, pos, pos + 1):
        if 0 <= j < len(ordered) and abs(ordered[j] - int(ts_ms)) <= tolerance_ms:
            return True
    return False


def build_asset_events(conn: sqlite3.Connection, threshold: float = 200_000.0) -> dict[str, list[dict[str, Any]]]:
    return {sym: build_events(conn, symbol=sym, threshold=(threshold if sym == SYMBOL else thr)) for sym, thr in ASSET_THRESHOLDS.items()}


def build_rows_for_threshold(conn: sqlite3.Connection, threshold: float, *, enrich: bool = False) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    asset_events = build_asset_events(conn, threshold)
    rows = enrich_events(conn, asset_events[SYMBOL], asset_events) if enrich else asset_events[SYMBOL]
    add_states(rows, asset_events)
    return rows, asset_events


def taker_rows(
    marks: Any,
    rows: list[dict[str, Any]],
    *,
    direction: str,
    entry_key: str,
    horizon_sec: int,
) -> list[dict[str, Any]]:
    sims = []
    for row in rows:
        entry_ts = row.get(entry_key)
        if entry_ts is None:
            continue
        entry = mark_at_or_after(marks, int(entry_ts))
        exit_ = mark_at_or_after(marks, int(entry_ts) + int(horizon_sec) * 1000)
        if not entry or not exit_:
            sim = {"status": "NO_MARK", "net_bps": None}
        else:
            gross = signed_return_bps(direction, float(entry[1]), float(exit_[1]))
            sim = {
                "status": "FILLED",
                "exec_model": "MARK_TAKER_PROXY",
                "direction": direction,
                "entry_ts_ms": int(entry[0]),
                "exit_ts_ms": int(exit_[0]),
                "gross_bps": r1(gross),
                "fee_bps": TAKER_COST_BPS,
                "net_bps": r1(gross - TAKER_COST_BPS),
            }
        sim.update({"month": row.get("month"), "anchor_ts_ms": row.get("anchor_ts_ms"), "side": row.get("side"), "threshold_usd": row.get("threshold_usd")})
        sims.append(sim)
    return sims


def exec_summary(sim_rows: list[dict[str, Any]], hold_months: set[str]) -> dict[str, Any]:
    filled = [r for r in sim_rows if r.get("status") == "FILLED" and finite(r.get("net_bps")) is not None]
    base = {
        **metrics([r.get("net_bps") for r in filled]),
        "filled_n": len(filled),
        "attempt_n": len(sim_rows),
        "fill_rate": r3(len(filled) / len(sim_rows)) if sim_rows else None,
    }
    cal_rows = [r for r in sim_rows if r.get("month") not in hold_months]
    hold_rows = [r for r in sim_rows if r.get("month") in hold_months]
    def one(xs: list[dict[str, Any]]) -> dict[str, Any]:
        f = [r for r in xs if r.get("status") == "FILLED" and finite(r.get("net_bps")) is not None]
        return {
            **metrics([r.get("net_bps") for r in f]),
            "filled_n": len(f),
            "attempt_n": len(xs),
            "fill_rate": r3(len(f) / len(xs)) if xs else None,
        }
    return {
        "all": base,
        "cal": one(cal_rows),
        "hold": one(hold_rows),
    }


def simulate_maker_proxy(marks: Any, row: dict[str, Any], *, direction: str, detect_ts_ms: int, horizon_sec: int, offset_bps: float) -> dict[str, Any]:
    detect = mark_at_or_after(marks, int(detect_ts_ms))
    if not detect:
        return {"status": "NO_MARK", "net_bps": None, "offset_bps": float(offset_bps)}
    detect_ts, detect_px = detect
    if direction == "LONG":
        limit_px = float(detect_px) * (1.0 - float(offset_bps) / 10_000.0)
        required = float(limit_px) * (1.0 - MAKER_PROXY_CROSS_BPS / 10_000.0)
        fill = next(((int(ts), float(limit_px)) for ts, px in marks.slice_range(int(detect_ts), int(detect_ts) + MAKER_PROXY_FILL_WINDOW_SEC * 1000) if float(px) <= required), None)
    else:
        limit_px = float(detect_px) * (1.0 + float(offset_bps) / 10_000.0)
        required = float(limit_px) * (1.0 + MAKER_PROXY_CROSS_BPS / 10_000.0)
        fill = next(((int(ts), float(limit_px)) for ts, px in marks.slice_range(int(detect_ts), int(detect_ts) + MAKER_PROXY_FILL_WINDOW_SEC * 1000) if float(px) >= required), None)
    if not fill:
        return {"status": "NO_MAKER_FILL", "net_bps": None, "offset_bps": float(offset_bps)}
    fill_ts, entry_px = fill
    exit_ = mark_at_or_after(marks, int(fill_ts) + int(horizon_sec) * 1000)
    if not exit_:
        return {"status": "NO_EXIT_MARK", "net_bps": None, "offset_bps": float(offset_bps)}
    gross = signed_return_bps(direction, float(entry_px), float(exit_[1]))
    return {
        "status": "FILLED",
        "exec_model": "MARK_MAKER_PULLBACK_PROXY",
        "direction": direction,
        "offset_bps": float(offset_bps),
        "detect_ts_ms": int(detect_ts_ms),
        "entry_ts_ms": int(fill_ts),
        "fill_delay_sec": r1((int(fill_ts) - int(detect_ts)) / 1000.0),
        "gross_bps": r1(gross),
        "fee_bps": MAKER_PROXY_COST_BPS,
        "net_bps": r1(gross - MAKER_PROXY_COST_BPS),
        "month": row.get("month"),
        "anchor_ts_ms": row.get("anchor_ts_ms"),
        "side": row.get("side"),
    }


def add_detect_keys(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for tau in TAUS_SEC:
            row[f"detect_{tau}_ts_ms"] = int(row["anchor_ts_ms"]) + int(tau) * 1000


def state_at(row: dict[str, Any], tau: int) -> str:
    return str(row[f"state_{tau}"]["state"])


def sell_silence_lane_expansion(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    marks = load_mark_index(conn, SYMBOL)
    live_times = load_live_signal_times()
    hold_months, split = split_months(rows)
    sell = [r for r in rows if r["side"] == "SELL"]
    out: dict[str, Any] = {"split": split, "live_v02_signal_count": len(live_times), "lanes": {}}
    for tau in (30, 60, 120, 300, 600, 900):
        selected = [r for r in sell if state_at(r, tau) == "SILENCE_RECLAIM"]
        for lane_name, lane_rows in {
            "all_silence": selected,
            "inside_current_v02_shadow_times": [r for r in selected if near_any(int(r["anchor_ts_ms"]), live_times)],
            "outside_current_v02_shadow_times": [r for r in selected if not near_any(int(r["anchor_ts_ms"]), live_times)],
        }.items():
            sims = taker_rows(marks, lane_rows, direction="LONG", entry_key=f"detect_{tau}_ts_ms", horizon_sec=14_400)
            key = f"tau{tau}_{lane_name}"
            out["lanes"][key] = {**exec_summary(sims, hold_months), "attempt_n": len(lane_rows)}
    return out


def threshold_expansion(conn: sqlite3.Connection) -> dict[str, Any]:
    marks = load_mark_index(conn, SYMBOL)
    out: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        rows, _ = build_rows_for_threshold(conn, threshold, enrich=False)
        add_detect_keys(rows)
        hold_months, split = split_months(rows)
        sell = [r for r in rows if r["side"] == "SELL"]
        cells = {}
        for tau in (30, 60, 120, 300, 600, 900):
            selected = [r for r in sell if state_at(r, tau) == "SILENCE_RECLAIM"]
            for horizon_name, horizon_sec in (("H2", 7200), ("H4", 14400)):
                sims = taker_rows(marks, selected, direction="LONG", entry_key=f"detect_{tau}_ts_ms", horizon_sec=horizon_sec)
                summ = exec_summary(sims, hold_months)
                cells[f"tau{tau}_{horizon_name}"] = {
                    **summ,
                    "attempt_n": len(selected),
                    "pass_all": pass_read(summ["all"]),
                    "pass_hold": pass_read(summ["hold"], min_n=12),
                }
        out[str(int(threshold))] = {"event_n": len(rows), "split": split, "cells": cells}
    return out


def bucket_by_quantile(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    vals = sorted(float(v) for v in (finite(r.get(key)) for r in rows) if v is not None)
    if len(vals) < 10:
        return {"ALL": rows}
    q33 = pctile(vals, 0.33)
    q66 = pctile(vals, 0.66)
    return {
        f"LOW_<={r1(q33)}": [r for r in rows if finite(r.get(key)) is not None and float(r[key]) <= q33],
        f"MID_{r1(q33)}_{r1(q66)}": [r for r in rows if finite(r.get(key)) is not None and q33 < float(r[key]) <= q66],
        f"HIGH_>{r1(q66)}": [r for r in rows if finite(r.get(key)) is not None and float(r[key]) > q66],
    }


def deepbid_ablation(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    marks = load_mark_index(conn, SYMBOL)
    hold_months, split = split_months(rows)
    base = [r for r in rows if r["side"] == "SELL" and state_at(r, 60) == "SILENCE_RECLAIM"]
    out = {"split": split, "base_attempt_n": len(base), "buckets": {}}
    for feature in ("bid_depth_usd", "book_imbalance", "spread_bps", "running_accel", "running_rate"):
        out["buckets"][feature] = {}
        for bucket, bucket_rows in bucket_by_quantile(base, feature).items():
            sims = taker_rows(marks, bucket_rows, direction="LONG", entry_key="detect_60_ts_ms", horizon_sec=14_400)
            out["buckets"][feature][bucket] = {**exec_summary(sims, hold_months), "attempt_n": len(bucket_rows)}
    return out


def event_end_vs_maker(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    marks = load_mark_index(conn, SYMBOL)
    hold_months, split = split_months(rows)
    sell = [r for r in rows if r["side"] == "SELL" and state_at(r, 1800) == "SILENCE_RECLAIM"]
    out: dict[str, Any] = {"split": split, "attempt_n": len(sell), "taker": {}, "maker": {}}
    for entry_key, label in (("event_end_ts_ms", "event_end"), ("reclaim_ts_ms", "reclaim")):
        for horizon_name, horizon_sec in HORIZONS.items():
            sims = taker_rows(marks, sell, direction="LONG", entry_key=entry_key, horizon_sec=horizon_sec)
            out["taker"][f"{label}_{horizon_name}"] = exec_summary(sims, hold_months)
    for tau in (60, 300, 900, 1800):
        selected = [r for r in rows if r["side"] == "SELL" and state_at(r, tau) == "SILENCE_RECLAIM"]
        for offset in MAKER_OFFSETS_BPS:
            sims = []
            for row in selected:
                sim = simulate_maker_proxy(marks, row, direction="LONG", detect_ts_ms=int(row[f"detect_{tau}_ts_ms"]), horizon_sec=14_400, offset_bps=offset)
                sims.append(sim)
            out["maker"][f"tau{tau}_O{r1(offset)}_H4"] = {**exec_summary(sims, hold_months), "attempt_n": len(selected)}
    return out


def cross_asset_lead(conn: sqlite3.Connection, rows: list[dict[str, Any]], asset_events: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    marks = load_mark_index(conn, SYMBOL)
    hold_months, split = split_months(rows)
    asset_idx = {sym: [int(e["anchor_ts_ms"]) for e in evts] for sym, evts in asset_events.items()}
    out: dict[str, Any] = {"split": split, "cells": {}}
    for side in ("SELL", "BUY"):
        direction = "LONG" if side == "SELL" else "SHORT"
        for lead_sym in ("BTCUSDT", "SOLUSDT"):
            for win in (60, 300, 900, 1800):
                selected = []
                for row in rows:
                    if row["side"] != side:
                        continue
                    ts = int(row["anchor_ts_ms"])
                    prev = [
                        x for x in neighbor_events(asset_idx[lead_sym], asset_events[lead_sym], ts_ms=ts, before_sec=win, after_sec=0, side=side)
                        if int(x["anchor_ts_ms"]) < ts
                    ]
                    if prev:
                        selected.append(row)
                sims = taker_rows(marks, selected, direction=direction, entry_key="anchor_ts_ms", horizon_sec=14_400)
                out["cells"][f"{lead_sym}_{side}_prev{win}s_eth_fade_H4"] = {**exec_summary(sims, hold_months), "attempt_n": len(selected)}
    return out


def propagation_precursor(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    hold_months, split = split_months(rows)
    labels = []
    for row in rows:
        same_60m = bool(row.get("has_next_same_60m"))
        cross_30m = int(row.get("cross_next_same_1800s") or 0) > 0
        labels.append({**row, "propagates": bool(same_60m or cross_30m)})
    out: dict[str, Any] = {"split": split, "overall": {}, "features": {}}
    out["overall"] = {
        side: {
            "n": sum(1 for r in labels if r["side"] == side),
            "propagation_rate": r3(sum(1 for r in labels if r["side"] == side and r["propagates"]) / max(1, sum(1 for r in labels if r["side"] == side))),
        }
        for side in ("SELL", "BUY")
    }
    for feature in ("running_accel", "running_rate", "event_duration_sec", "post_anchor_liq_notional", "single_dominance_pct", "bid_depth_usd", "book_imbalance", "spread_bps"):
        out["features"][feature] = {}
        for side in ("SELL", "BUY"):
            side_rows = [r for r in labels if r["side"] == side]
            out["features"][feature][side] = {}
            for bucket, bucket_rows in bucket_by_quantile(side_rows, feature).items():
                n = len(bucket_rows)
                out["features"][feature][side][bucket] = {
                    "n": n,
                    "propagation_rate": r3(sum(1 for r in bucket_rows if r["propagates"]) / n) if n else None,
                    "fade_h4": metrics([r.get("fade_h4_bps") for r in bucket_rows]),
                }
    return out


def run(db: Path) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    try:
        rows, asset_events = build_rows_for_threshold(conn, 200_000.0, enrich=True)
        add_detect_keys(rows)
        report = {
            "generated_at_utc": utc_now(),
            "research_only": True,
            "live_executor_touched": False,
            "live_rule": LIVE_V02_RULE,
            "event_counts": {sym: len(evts) for sym, evts in asset_events.items()},
            "tests": {
                "sell_silence_lane_expansion": sell_silence_lane_expansion(conn, rows),
                "threshold_expansion": threshold_expansion(conn),
                "deepbid_ablation": deepbid_ablation(conn, rows),
                "event_end_vs_maker": event_end_vs_maker(conn, rows),
                "cross_asset_lead": cross_asset_lead(conn, rows, asset_events),
                "propagation_precursor": propagation_precursor(conn, rows),
            },
            "verdict": "FREQUENCY_MAP_BUILT_NO_LIVE_PROMOTION",
            "read": [
                "The strongest executable expansion remains SELL silence/reclaim fade; broad state is stronger than executable entry after book staleness and holdout.",
                "BUY-side mirror still does not become a deployable short/fade lane.",
                "Propagation tags are useful as navigation/danger labels but not yet as live order logic.",
            ],
        }
    finally:
        conn.close()
    return report


def top_cells(report: dict[str, Any]) -> list[dict[str, Any]]:
    cells = []
    def add(path: str, obj: Any) -> None:
        if isinstance(obj, dict):
            if "all" in obj and isinstance(obj["all"], dict) and ("sum_bps" in obj["all"] or "filled_n" in obj["all"]):
                allm = obj["all"]
                cells.append({
                    "path": path,
                    "n": allm.get("n") or allm.get("filled_n"),
                    "sum_bps": allm.get("sum_bps"),
                    "median_bps": allm.get("median_bps"),
                    "t3r_bps": allm.get("t3r_bps"),
                    "hold_t3r_bps": obj.get("hold", {}).get("t3r_bps") if isinstance(obj.get("hold"), dict) else None,
                })
            for k, v in obj.items():
                add(f"{path}.{k}" if path else str(k), v)
    add("tests", report.get("tests", {}))
    return sorted([c for c in cells if finite(c.get("sum_bps")) is not None], key=lambda c: float(c["sum_bps"]), reverse=True)[:20]


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V02 Frequency Expansion Tests",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Research-only. No live executor, paper buckets, config, order logic, or sizing was changed.",
        "",
        "## Verdict",
        "",
        f"- `{report['verdict']}`",
        "",
        "## Top Cells By All-Sample Sum",
        "",
        "| rank | cell | N | sum | median | T3R | hold T3R |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, c in enumerate(top_cells(report), 1):
        lines.append(f"| {i} | `{c['path']}` | {c.get('n')} | {c.get('sum_bps')} | {c.get('median_bps')} | {c.get('t3r_bps')} | {c.get('hold_t3r_bps')} |")
    lines.extend([
        "",
        "## Full JSON",
        "",
        "```json",
        json.dumps(report["tests"], indent=2, sort_keys=True),
        "```",
        "",
        "## Read",
        "",
    ])
    lines.extend(f"- {x}" for x in report["read"])
    lines.append("")
    return "\n".join(lines)


def build_handoff(report: dict[str, Any]) -> str:
    existing_reports = [
        "S34_V02_MANAGEMENT_NAVIGATION_SUITE.md",
        "S34_V02_FOUR_ARM_SYMMETRY_TESTS.md",
        "S34_V02_EVENT_CHAIN_PUZZLE_TESTS.md",
        "S34_V02_PROPAGATION_PUZZLE_SUITE.md",
        "S34_V02_PROPAGATION_CANDIDATE_GAUNTLET.md",
        "S34_V02_CANDIDATE_EXECUTION_GAUNTLET.md",
        "S34_V02_NEXT_NAVIGATION_TESTS.md",
        "S34_V02_FREQUENCY_EXPANSION_TESTS.md",
    ]
    lines = [
        "# S34 Claude Comprehensive Handoff",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "This is a research-only synthesis for Claude. Live executor/config/order logic were not changed.",
        "",
        "## Executive Conclusion",
        "",
        "- Current live alpha remains `S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID`.",
        "- The best durable concept is not a new live rule yet; it is a navigation map: SELL silence/reclaim is fade-friendly, SELL propagation is fade-danger/momentum-watch, BUY-side mirror is not deployable.",
        "- Frequency expansion should come from separate shadow lanes after execution gauntlet, not from loosening the live lane blindly.",
        "- The latest frequency suite found map value but no immediate live promotion.",
        "",
        "## Latest Frequency Expansion Top Cells",
        "",
        "| rank | cell | N | sum | median | T3R | hold T3R |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, c in enumerate(top_cells(report), 1):
        lines.append(f"| {i} | `{c['path']}` | {c.get('n')} | {c.get('sum_bps')} | {c.get('median_bps')} | {c.get('t3r_bps')} | {c.get('hold_t3r_bps')} |")
    lines.extend([
        "",
        "## Settled Results From This Research Block",
        "",
        "1. Four-arm symmetry: SELL->LONG is the only materially positive V02-family arm; BUY->SHORT mirror is weak and T3R-negative.",
        "2. Event-chain puzzle: same-side continuation cells are strongly negative for fade; cross-asset next SELL propagation flips SELL fade from positive to dangerous.",
        "3. Propagation suite: silence after SELL shock is fade-friendly; propagation is momentum-watch but not yet causally/execution validated.",
        "4. Candidate gauntlet: broad in-sample candidates failed multiple-comparison correction; strongest raw cells are navigation hypotheses.",
        "5. Execution gauntlet: with book staleness guard, only SELL silence fade remains marginal; propagation momentum collapses under causal executable entry.",
        "6. Next-navigation tests: SELL silence/reclaim at 30-60s is the cleanest navigation state; BUY silence fade remains bad.",
        "7. Frequency expansion: threshold/deepbid/event_end/maker mapping expanded the surface; no new lane is live-ready.",
        "",
        "## Current Working Model",
        "",
        "Cascade is no longer treated as a single event. The better model is a state sequence:",
        "",
        "```text",
        "SELL shock -> silence/reclaim -> fade-friendly recovery",
        "SELL shock -> same-side/cross-asset propagation -> fade danger / momentum watch",
        "BUY shock -> mirror does not symmetrically validate",
        "```",
        "",
        "## Open Questions For Next Round",
        "",
        "1. Can SELL silence/reclaim be made executable with maker pullback/reclaim entry without losing holdout T3R?",
        "2. Can propagation pressure be detected before 900-1800s using tick/book features rather than future event labels?",
        "3. Is deep_bid an independent causal condition, or just a proxy for the current V02 sample?",
        "4. Can cross-asset lead be converted into a permission/avoidance tag rather than a directional entry?",
        "5. Should H4 shadow be promoted only as management/navigation for current V02, not a separate live entry?",
        "",
        "## Source Reports",
        "",
    ])
    for name in existing_reports:
        path = OUT_DIR / name
        lines.append(f"- `{path}` ({'exists' if path.exists() else 'missing'})")
    lines.extend([
        "",
        "## Guardrail",
        "",
        "No candidate here should be promoted live without: causal entry, executable book fill, chronological holdout positive, T3R positive after top winners removed, and preferably forward shadow.",
        "",
    ])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run S34 V02 frequency expansion tests.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    p.add_argument("--handoff-md", type=Path, default=HANDOFF_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args.db)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render_md(report), encoding="utf-8")
    args.handoff_md.write_text(build_handoff(report), encoding="utf-8")
    print(render_md(report))
    print(f"\nHandoff: {args.handoff_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
