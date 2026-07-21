"""S34 V02 four-arm symmetry and mirror test suite.

Research-only. Tests whether the current ETH SELL-liq maker LONG route has a
tradeable mirror on the BUY-liq maker SHORT side, and whether same-event
negative controls expose a fill/model artefact.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (  # noqa: E402
    file_fingerprint,
    iso_ms,
    load_mark_index,
    pctile,
    r1,
    r3,
    signed_return_bps,
)
from tools.research_s34_maker_fade import collect_events, maker_limit_price  # noqa: E402
from tools.research_s34_wave_absorption import book_features_at  # noqa: E402
from tools.s34_v_engine_cancel_replace import find_fill_between  # noqa: E402
from tools.s34_v_engine_execution_frontier import prior_return_bps  # noqa: E402
from tools.s34_v_engine_shadow_observer import ACCEL_WINDOW_SEC, BUCKET_SEC, MIN_GAP_SEC  # noqa: E402


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V02_FOUR_ARM_SYMMETRY_TESTS.json"
OUT_MD = OUT_DIR / "S34_V02_FOUR_ARM_SYMMETRY_TESTS.md"

SYMBOL = "ETHUSDT"
THRESHOLD_USD = 200_000.0
VDEPTH_MIN_BPS = 28.0
VDEPTH_MAX_BPS = 40.0
PRIOR_ABS_BPS = 50.0
MIN_DEPTH_USD = 135_423.8
INITIAL_OFFSET_BPS = 20.0
REPLACE_OFFSET_BPS = 5.0
WAIT_SEC = 300
CROSS_MARGIN_BPS = 1.0
FILL_SEARCH_SEC = 2 * 3600
HORIZONS_SEC = {"H2": 2 * 3600, "H3": 3 * 3600, "H4": 4 * 3600}
FEE_BPS = 8.0
MAX_BOOK_STALENESS_SEC = 10
STOP_LEVELS_BPS = (100.0, 125.0, 150.0, 175.0, 200.0)


ARM_DEFS = {
    "SELL_LONG_BASELINE": {
        "liq_side": "SELL",
        "direction": "LONG",
        "role": "current_live_family",
        "depth_key": "bid_depth_usd",
        "prior_rule": "prior_4h_bps < -50",
    },
    "SELL_SHORT_NEG_CONTROL": {
        "liq_side": "SELL",
        "direction": "SHORT",
        "role": "same_event_opposite_direction_negative_control",
        "depth_key": "bid_depth_usd",
        "prior_rule": "prior_4h_bps < -50",
    },
    "BUY_SHORT_MIRROR": {
        "liq_side": "BUY",
        "direction": "SHORT",
        "role": "true_mirror_candidate",
        "depth_key": "ask_depth_usd",
        "prior_rule": "prior_4h_bps > +50",
    },
    "BUY_LONG_NEG_CONTROL": {
        "liq_side": "BUY",
        "direction": "LONG",
        "role": "mirror_event_opposite_direction_negative_control",
        "depth_key": "ask_depth_usd",
        "prior_rule": "prior_4h_bps > +50",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def metrics(vals: list[Any]) -> dict[str, Any]:
    xs = [float(x) for x in (finite(v) for v in vals) if x is not None]
    if not xs:
        return {
            "n": 0,
            "sum_bps": 0.0,
            "mean_bps": None,
            "median_bps": None,
            "win_rate": None,
            "t3r_bps": 0.0,
            "top1_removed_bps": 0.0,
            "min_bps": None,
            "max_bps": None,
            "tail_lt_-100_n": 0,
            "tail_lt_-200_n": 0,
        }
    ordered = sorted(xs, reverse=True)
    return {
        "n": len(xs),
        "sum_bps": r1(sum(xs)),
        "mean_bps": r1(mean(xs)),
        "median_bps": r1(pctile(xs, 0.5)),
        "win_rate": r3(sum(1 for x in xs if x > 0.0) / len(xs)),
        "t3r_bps": r1(sum(ordered[3:]) if len(ordered) > 3 else sum(ordered)),
        "top1_removed_bps": r1(sum(ordered[1:]) if len(ordered) > 1 else sum(ordered)),
        "min_bps": r1(min(xs)),
        "max_bps": r1(max(xs)),
        "tail_lt_-100_n": sum(1 for x in xs if x < -100.0),
        "tail_lt_-200_n": sum(1 for x in xs if x < -200.0),
    }


def month_of(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).strftime("%Y-%m")


def mark_at_or_after(marks: Any, ts_ms: int) -> tuple[int, float] | None:
    row = marks.at_or_after(int(ts_ms))
    return (int(row[0]), float(row[1])) if row else None


def mark_ret_bps(marks: Any, start_ms: int, end_ms: int) -> float | None:
    a = mark_at_or_after(marks, start_ms)
    b = mark_at_or_after(marks, end_ms)
    if not a or not b or float(a[1]) <= 0.0:
        return None
    return r1((float(b[1]) - float(a[1])) / float(a[1]) * 10_000.0)


def exit_price_at(
    conn: sqlite3.Connection,
    marks: Any,
    *,
    direction: str,
    ts_ms: int,
) -> tuple[int, float, str] | None:
    book = book_features_at(conn, SYMBOL, int(ts_ms), MAX_BOOK_STALENESS_SEC)
    if book:
        return int(book["book_ts_ms"]), float(book["bid"] if direction == "LONG" else book["ask"]), "book"
    mark = mark_at_or_after(marks, ts_ms)
    if mark:
        return int(mark[0]), float(mark[1]), "mark_fallback"
    return None


def path_stats(event: Any, *, entry_price: float, fill_ts_ms: int, direction: str, horizon_sec: int) -> dict[str, Any]:
    end_ms = int(fill_ts_ms) + int(horizon_sec) * 1000
    path = [
        (int(ts), signed_return_bps(direction, float(entry_price), float(px)))
        for ts, px in event.path
        if int(fill_ts_ms) <= int(ts) <= end_ms
    ]
    if not path:
        return {}
    mfe_ts, mfe = max(path, key=lambda x: x[1])
    mae_ts, mae = min(path, key=lambda x: x[1])

    def first_le(level: float) -> float | None:
        for ts, ret in path:
            if ret <= level:
                return r1((int(ts) - int(fill_ts_ms)) / 1000.0)
        return None

    return {
        "mfe_bps": r1(mfe),
        "mae_bps": r1(mae),
        "mfe_sec": r1((int(mfe_ts) - int(fill_ts_ms)) / 1000.0),
        "mae_sec": r1((int(mae_ts) - int(fill_ts_ms)) / 1000.0),
        **{f"sl{int(s)}_touch_sec": first_le(-float(s)) for s in STOP_LEVELS_BPS},
    }


def simulate_v02_lifecycle(
    conn: sqlite3.Connection,
    marks: Any,
    event: Any,
    *,
    direction: str,
) -> dict[str, Any]:
    event = replace(event, fade_direction=direction)
    anchor_ts = int(event.anchor_mark_ts_ms)
    cancel_ts = anchor_ts + WAIT_SEC * 1000
    initial_limit = maker_limit_price(event.anchor_mark_price, direction, INITIAL_OFFSET_BPS)
    fill = find_fill_between(
        event,
        limit_px=initial_limit,
        cross_margin_bps=CROSS_MARGIN_BPS,
        start_ts_ms=anchor_ts,
        end_ts_ms=cancel_ts,
    )
    fill_leg = "initial"
    limit_px = initial_limit
    if fill is None:
        replace_limit = maker_limit_price(event.anchor_mark_price, direction, REPLACE_OFFSET_BPS)
        fill = find_fill_between(
            event,
            limit_px=replace_limit,
            cross_margin_bps=CROSS_MARGIN_BPS,
            start_ts_ms=cancel_ts,
            end_ts_ms=anchor_ts + FILL_SEARCH_SEC * 1000,
        )
        fill_leg = "replacement"
        limit_px = replace_limit
    else:
        replace_limit = None

    base: dict[str, Any] = {
        "anchor_ts_ms": int(event.anchor.anchor_ts_ms),
        "anchor_mark_ts_ms": anchor_ts,
        "anchor_utc": iso_ms(anchor_ts),
        "anchor_mark_price": r1(event.anchor_mark_price),
        "initial_limit_price": float(initial_limit),
        "replace_limit_price": None if replace_limit is None else float(replace_limit),
        "direction": direction,
    }
    if fill is None:
        return {**base, "status": "NO_MAKER_FILL", "net_bps_by_horizon": {}}

    fill_ts, entry_px = fill
    out = {
        **base,
        "status": "FILLED",
        "fill_leg": fill_leg,
        "maker_fill_ts_ms": int(fill_ts),
        "maker_fill_utc": iso_ms(fill_ts),
        "fill_delay_sec": r1((int(fill_ts) - anchor_ts) / 1000.0),
        "entry_price": float(entry_px),
        "limit_price": float(limit_px),
        "net_bps_by_horizon": {},
        "exit_source_by_horizon": {},
        "path_by_horizon": {},
    }
    for label, horizon in HORIZONS_SEC.items():
        exit_row = exit_price_at(conn, marks, direction=direction, ts_ms=int(fill_ts) + int(horizon) * 1000)
        if not exit_row:
            out["net_bps_by_horizon"][label] = None
            continue
        exit_ts, exit_px, source = exit_row
        gross = signed_return_bps(direction, float(entry_px), float(exit_px))
        out["net_bps_by_horizon"][label] = r1(gross - FEE_BPS)
        out["exit_source_by_horizon"][label] = source
        out["path_by_horizon"][label] = path_stats(
            event,
            entry_price=float(entry_px),
            fill_ts_ms=int(fill_ts),
            direction=direction,
            horizon_sec=int(horizon),
        )
        if label == "H4":
            out["h4_exit_ts_ms"] = int(exit_ts)
            out["h4_exit_utc"] = iso_ms(exit_ts)
            out["h4_exit_price"] = float(exit_px)
    return out


def collect_side_events(conn: sqlite3.Connection, *, side: str) -> list[Any]:
    return collect_events(
        conn,
        symbol=SYMBOL,
        threshold=THRESHOLD_USD,
        sides=(side,),
        min_vdepth_bps=VDEPTH_MIN_BPS,
        bucket_sec=BUCKET_SEC,
        min_gap_sec=MIN_GAP_SEC,
        accel_window_sec=ACCEL_WINDOW_SEC,
        max_horizon_sec=FILL_SEARCH_SEC,
    )


def eligible_events(conn: sqlite3.Connection, *, side: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    marks = load_mark_index(conn, SYMBOL)
    raw_events = collect_side_events(conn, side=side)
    rows = []
    rejects = {"raw": len(raw_events), "vdepth": 0, "prior": 0, "book": 0, "depth": 0}
    for event in raw_events:
        if not (VDEPTH_MIN_BPS <= float(event.vdepth_bps) < VDEPTH_MAX_BPS):
            rejects["vdepth"] += 1
            continue
        prior4h = prior_return_bps(marks, int(event.anchor.anchor_ts_ms), 4 * 3600)
        if prior4h is None or not math.isfinite(float(prior4h)):
            rejects["prior"] += 1
            continue
        if side == "SELL" and not (float(prior4h) < -PRIOR_ABS_BPS):
            rejects["prior"] += 1
            continue
        if side == "BUY" and not (float(prior4h) > PRIOR_ABS_BPS):
            rejects["prior"] += 1
            continue
        book = book_features_at(conn, SYMBOL, int(event.anchor.anchor_ts_ms), MAX_BOOK_STALENESS_SEC)
        if not book:
            rejects["book"] += 1
            continue
        depth_key = "bid_depth_usd" if side == "SELL" else "ask_depth_usd"
        if float(book.get(depth_key) or 0.0) < MIN_DEPTH_USD:
            rejects["depth"] += 1
            continue
        rows.append({"event": event, "prior4h_bps": r1(prior4h), "book": book})
    return rows, rejects


def run_arm(conn: sqlite3.Connection, *, arm_id: str, side_rows: list[dict[str, Any]]) -> dict[str, Any]:
    arm = ARM_DEFS[arm_id]
    marks = load_mark_index(conn, SYMBOL)
    btc_marks = load_mark_index(conn, "BTCUSDT")
    sol_marks = load_mark_index(conn, "SOLUSDT")
    rows = []
    for item in side_rows:
        event = item["event"]
        sim = simulate_v02_lifecycle(conn, marks, event, direction=arm["direction"])
        h4 = sim.get("net_bps_by_horizon", {}).get("H4")
        row = {
            "arm_id": arm_id,
            "role": arm["role"],
            "symbol": SYMBOL,
            "liq_side": arm["liq_side"],
            "direction": arm["direction"],
            "signal_ts_ms": int(event.anchor.anchor_ts_ms),
            "signal_utc": iso_ms(event.anchor.anchor_ts_ms),
            "month": month_of(event.anchor.anchor_ts_ms),
            "bucket": int(event.anchor.bucket),
            "vdepth_bps": r1(event.vdepth_bps),
            "prior4h_bps": item["prior4h_bps"],
            "running_notional": r1(event.anchor.running_notional),
            "running_liq_count": int(event.anchor.running_liq_count),
            "running_accel_usd_per_sec": r1(event.anchor.running_accel),
            "single_liq_dominance_pct": r3(event.anchor.running_single_liq_dominance),
            "book_imbalance": r3(item["book"]["book_imbalance"]),
            "bid_depth_usd": r1(item["book"]["bid_depth_usd"]),
            "ask_depth_usd": r1(item["book"]["ask_depth_usd"]),
            "spread_bps": r1(item["book"]["spread_bps"]),
            "btc30_bps": mark_ret_bps(btc_marks, int(event.anchor.anchor_ts_ms), int(event.anchor.anchor_ts_ms) + 30 * 60 * 1000),
            "sol30_bps": mark_ret_bps(sol_marks, int(event.anchor.anchor_ts_ms), int(event.anchor.anchor_ts_ms) + 30 * 60 * 1000),
            **sim,
        }
        row["h2_net_bps"] = sim.get("net_bps_by_horizon", {}).get("H2")
        row["h3_net_bps"] = sim.get("net_bps_by_horizon", {}).get("H3")
        row["h4_net_bps"] = h4
        row["h4_minus_h2_bps"] = (
            r1(float(row["h4_net_bps"]) - float(row["h2_net_bps"]))
            if row.get("h4_net_bps") is not None and row.get("h2_net_bps") is not None
            else None
        )
        if arm["direction"] == "LONG":
            row["cross_support_ok"] = bool((row["btc30_bps"] is None or row["btc30_bps"] > -50.0) and (row["sol30_bps"] is None or row["sol30_bps"] > -50.0))
        else:
            row["cross_support_ok"] = bool((row["btc30_bps"] is None or row["btc30_bps"] < 50.0) and (row["sol30_bps"] is None or row["sol30_bps"] < 50.0))
        path = sim.get("path_by_horizon", {}).get("H4", {})
        row["mae_bps"] = path.get("mae_bps")
        row["mfe_bps"] = path.get("mfe_bps")
        rows.append(row)

    filled = [r for r in rows if r.get("status") == "FILLED" and r.get("h4_net_bps") is not None]
    return {
        "arm_id": arm_id,
        "definition": arm,
        "eligible_n": len(rows),
        "filled_n": len(filled),
        "fill_rate": r3(len(filled) / len(rows)) if rows else None,
        "h2": metrics([r.get("h2_net_bps") for r in filled]),
        "h3": metrics([r.get("h3_net_bps") for r in filled]),
        "h4": metrics([r.get("h4_net_bps") for r in filled]),
        "h4_minus_h2": metrics([r.get("h4_minus_h2_bps") for r in filled]),
        "by_cross_support": {
            str(k): metrics([r.get("h4_net_bps") for r in filled if bool(r.get("cross_support_ok")) is k])
            for k in (True, False)
        },
        "by_month_h4": {
            m: metrics([r.get("h4_net_bps") for r in filled if r.get("month") == m])
            for m in sorted({str(r.get("month")) for r in filled})
        },
        "stop_touch_h4": {
            f"SL{int(stop)}": sum(
                1
                for r in filled
                if (r.get("path_by_horizon", {}).get("H4", {}).get(f"sl{int(stop)}_touch_sec") is not None)
            )
            for stop in STOP_LEVELS_BPS
        },
        "sample_rows": filled[:20],
        "rows": rows,
    }


def chronological_split(rows: list[dict[str, Any]]) -> tuple[set[str], dict[str, Any]]:
    months = sorted({str(r.get("month")) for r in rows if r.get("month")})
    hold_n = max(1, round(len(months) * 0.35)) if months else 0
    hold_months = set(months[-hold_n:]) if hold_n else set()
    return hold_months, {"method": "chronological_month_tail_35pct", "months": months, "holdout_months": sorted(hold_months)}


def split_report(arm_reports: dict[str, Any]) -> dict[str, Any]:
    all_rows = []
    for arm in arm_reports.values():
        all_rows.extend([r for r in arm["rows"] if r.get("status") == "FILLED" and r.get("h4_net_bps") is not None])
    hold_months, meta = chronological_split(all_rows)
    out = {"split": meta, "by_arm": {}}
    for arm_id, arm in arm_reports.items():
        filled = [r for r in arm["rows"] if r.get("status") == "FILLED" and r.get("h4_net_bps") is not None]
        out["by_arm"][arm_id] = {
            "cal": metrics([r["h4_net_bps"] for r in filled if r.get("month") not in hold_months]),
            "hold": metrics([r["h4_net_bps"] for r in filled if r.get("month") in hold_months]),
        }
    return out


def permutation_maxstat(arm_reports: dict[str, Any], *, iterations: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    arm_ids = list(arm_reports)
    rows_by_arm = {
        arm_id: [r for r in arm_reports[arm_id]["rows"] if r.get("status") == "FILLED" and r.get("h4_net_bps") is not None]
        for arm_id in arm_ids
    }
    sizes = {arm_id: len(rows) for arm_id, rows in rows_by_arm.items()}
    all_vals = [float(r["h4_net_bps"]) for rows in rows_by_arm.values() for r in rows]
    observed = {arm_id: metrics([r["h4_net_bps"] for r in rows])["t3r_bps"] for arm_id, rows in rows_by_arm.items()}
    observed_max = max([float(v or 0.0) for v in observed.values()] or [0.0])
    if not all_vals or sum(1 for n in sizes.values() if n) < 2:
        return {"status": "INSUFFICIENT_DATA", "observed_t3r": observed}
    max_stats = []
    for _ in range(int(iterations)):
        vals = all_vals[:]
        rng.shuffle(vals)
        idx = 0
        t3rs = []
        for arm_id in arm_ids:
            n = sizes[arm_id]
            sample = vals[idx:idx + n]
            idx += n
            t3rs.append(float(metrics(sample)["t3r_bps"] or 0.0))
        max_stats.append(max(t3rs))
    p_right = (1 + sum(1 for x in max_stats if x >= observed_max)) / (len(max_stats) + 1)
    return {
        "status": "OK",
        "iterations": int(iterations),
        "seed": int(seed),
        "observed_t3r": observed,
        "observed_max_t3r": r1(observed_max),
        "null_p95_max_t3r": r1(pctile(max_stats, 0.95)),
        "mc_corrected_p_right": r3(p_right),
        "read": "Pass requires observed max T3R > null p95 and low corrected p; this controls the four-arm search at a coarse level.",
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V02 Four-Arm Symmetry Tests",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Research-only. No live executor/config/order logic is touched.",
        "",
        "## Verdict",
        "",
        f"- Overall: `{report['verdict']}`",
        f"- Mirror decision: `{report['mirror_decision']}`",
        "",
        "## Arm Summary",
        "",
        "| Arm | Role | Eligible | Filled | Fill% | H2 | H4 | H4-H2 | Cross-support H4 |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for arm_id, arm in report["arms"].items():
        fill_pct = None if arm["fill_rate"] is None else r1(float(arm["fill_rate"]) * 100.0)
        cross = arm["by_cross_support"].get("True", {})
        lines.append(
            f"| `{arm_id}` | {arm['definition']['role']} | {arm['eligible_n']} | {arm['filled_n']} | {fill_pct} | "
            f"N={arm['h2']['n']} sum={arm['h2']['sum_bps']} med={arm['h2']['median_bps']} T3R={arm['h2']['t3r_bps']} | "
            f"N={arm['h4']['n']} sum={arm['h4']['sum_bps']} med={arm['h4']['median_bps']} T3R={arm['h4']['t3r_bps']} | "
            f"N={arm['h4_minus_h2']['n']} sum={arm['h4_minus_h2']['sum_bps']} med={arm['h4_minus_h2']['median_bps']} | "
            f"N={cross.get('n')} sum={cross.get('sum_bps')} T3R={cross.get('t3r_bps')} |"
        )
    lines += [
        "",
        "## Chronological Holdout",
        "",
        f"Split: `{report['chronological']['split']}`",
        "",
        "| Arm | Cal H4 | Hold H4 |",
        "| --- | --- | --- |",
    ]
    for arm_id, row in report["chronological"]["by_arm"].items():
        lines.append(
            f"| `{arm_id}` | N={row['cal']['n']} sum={row['cal']['sum_bps']} T3R={row['cal']['t3r_bps']} | "
            f"N={row['hold']['n']} sum={row['hold']['sum_bps']} T3R={row['hold']['t3r_bps']} |"
        )
    lines += [
        "",
        "## Permutation Max-Stat",
        "",
        "```json",
        json.dumps(report["permutation"], indent=2, sort_keys=True),
        "```",
        "",
        "## Rejection Counts",
        "",
        "```json",
        json.dumps(report["rejections"], indent=2, sort_keys=True),
        "```",
        "",
        "## Read",
        "",
    ]
    lines.extend([f"- {x}" for x in report["read"]])
    lines.append("")
    return "\n".join(lines)


def run(db: Path, *, permutations: int) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    side_rows = {}
    rejects = {}
    try:
        for side in ("SELL", "BUY"):
            side_rows[side], rejects[side] = eligible_events(conn, side=side)
        arms = {
            arm_id: run_arm(conn, arm_id=arm_id, side_rows=side_rows[ARM_DEFS[arm_id]["liq_side"]])
            for arm_id in ARM_DEFS
        }
    finally:
        conn.close()
    chronological = split_report(arms)
    perm = permutation_maxstat(arms, iterations=permutations, seed=3402)

    baseline = arms["SELL_LONG_BASELINE"]["h4"]
    sell_neg = arms["SELL_SHORT_NEG_CONTROL"]["h4"]
    mirror = arms["BUY_SHORT_MIRROR"]["h4"]
    buy_neg = arms["BUY_LONG_NEG_CONTROL"]["h4"]
    mirror_pass = (
        mirror["n"] >= 10
        and mirror["sum_bps"] > 0
        and mirror["t3r_bps"] > 0
        and chronological["by_arm"]["BUY_SHORT_MIRROR"]["hold"]["sum_bps"] > 0
        and chronological["by_arm"]["BUY_SHORT_MIRROR"]["hold"]["t3r_bps"] > 0
        and mirror["sum_bps"] > buy_neg["sum_bps"]
    )
    neg_control_warning = bool(sell_neg["sum_bps"] > 0 or buy_neg["sum_bps"] > 0)
    verdict = "NO_DEPLOYABLE_MIRROR_SHORT"
    if mirror_pass and not neg_control_warning:
        verdict = "MIRROR_SHORT_SHADOW_CANDIDATE"
    elif mirror_pass:
        verdict = "MIRROR_POSITIVE_BUT_NEGATIVE_CONTROL_WARNING"

    read = [
        "SELL->LONG baseline remains the reference arm; BUY->SHORT must beat BUY->LONG and survive holdout before it is even shadow-candidate.",
        "Same-event opposite-direction arms are negative controls. If they are positive, the result is likely regime/fill bias rather than clean direction.",
        "This suite uses the V02 O20/W300/O5/C1 maker lifecycle and H2/H3/H4 exits, but still uses top-of-book/proxy queue, not full tick queue replay.",
        "No live or paper bucket is changed by this script.",
    ]
    if neg_control_warning:
        read.append("At least one negative-control arm is positive on H4 sum; treat any mirror-positive result as contaminated until stricter queue and forward data confirm.")
    if mirror["n"] < 10:
        read.append("BUY->SHORT mirror has very small filled N; absence of evidence is not proof, but it is not promotable.")

    return {
        "generated_at_utc": utc_now(),
        "research_only": True,
        "live_executor_touched": False,
        "config": {
            "symbol": SYMBOL,
            "threshold_usd": THRESHOLD_USD,
            "vdepth_bps": [VDEPTH_MIN_BPS, VDEPTH_MAX_BPS],
            "prior_abs_bps": PRIOR_ABS_BPS,
            "min_depth_usd": MIN_DEPTH_USD,
            "lifecycle": f"O{INITIAL_OFFSET_BPS:g}_W{WAIT_SEC}_O{REPLACE_OFFSET_BPS:g}_C{CROSS_MARGIN_BPS:g}",
            "horizons_sec": HORIZONS_SEC,
            "fee_bps": FEE_BPS,
            "db_fingerprint": file_fingerprint(db),
        },
        "rejections": rejects,
        "arms": arms,
        "chronological": chronological,
        "permutation": perm,
        "verdict": verdict,
        "mirror_decision": "DO_NOT_ADD_TO_LIVE_OR_PAPER; research/shadow-observe only" if verdict != "MIRROR_SHORT_SHADOW_CANDIDATE" else "eligible_for_separate_shadow_review_only",
        "read": read,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run S34 V02 four-arm symmetry tests.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--permutations", type=int, default=500)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args.db, permutations=int(args.permutations))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
