"""S34 V02 candidate execution gauntlet.

Research-only. Re-tests propagation candidates at the first causal time their
state is knowable (anchor+tau), then applies taker/maker execution, delay, stop,
holdout, and negative-control anatomy. No live/paper state is changed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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

from tools.research_s34_knowable_anchor_continuation import iso_ms, load_mark_index, pctile, r1, r3, signed_return_bps  # noqa: E402
from tools.research_s34_wave_absorption import book_features_at  # noqa: E402
from tools.s34_v02_event_chain_puzzle_tests import ASSET_THRESHOLDS, build_events, metrics, neighbor_events  # noqa: E402
from tools.s34_v02_propagation_candidate_gauntlet import CANDIDATES, causal_counts  # noqa: E402
from tools.s34_v02_propagation_puzzle_suite import enrich_events  # noqa: E402


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V02_CANDIDATE_EXECUTION_GAUNTLET.json"
OUT_MD = OUT_DIR / "S34_V02_CANDIDATE_EXECUTION_GAUNTLET.md"

SYMBOL = "ETHUSDT"
TAKER_FEE_BPS = 8.0
MAKER_TAKER_FEE_BPS = 4.0
MAX_BOOK_STALENESS_SEC = 10
TAKER_DELAYS_SEC = (0, 30, 60, 300)
MAKER_OFFSETS_BPS = (0.0, 5.0, 10.0)
MAKER_FILL_WINDOW_SEC = 900
MAKER_CROSS_MARGIN_BPS = 1.0
STOP_LEVELS_BPS = (50.0, 100.0, 150.0)
PERMUTATIONS = 500

PRIMARY_CANDIDATES = {
    "SELL_SILENCE_FADE_LONG_H4": 1800,
    "SELL_PROPAGATION_MOMENTUM_SHORT_H1": 3600,
    "BUY_PROPAGATION_MOMENTUM_LONG_H1": 3600,
    "BUY_SILENCE_FADE_SHORT_H4": 3600,
}


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


def event_index(events: list[dict[str, Any]]) -> list[int]:
    return [int(e["anchor_ts_ms"]) for e in events]


def mark_at_or_after(marks: Any, ts_ms: int) -> tuple[int, float] | None:
    row = marks.at_or_after(int(ts_ms))
    return (int(row[0]), float(row[1])) if row else None


def book_at_or_after(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT ts_ms, bid_price, bid_qty, ask_price, ask_qty, mid_price, spread_pct, book_imbalance, bid_depth_usd
        FROM book_ticker
        WHERE symbol=? AND ts_ms>=?
        ORDER BY ts_ms ASC LIMIT 1
        """,
        (symbol, int(ts_ms)),
    ).fetchone()
    if not row:
        return None
    staleness_ms = int(row[0]) - int(ts_ms)
    if staleness_ms < 0 or staleness_ms > MAX_BOOK_STALENESS_SEC * 1000:
        return None
    bid = float(row[1])
    bid_qty = float(row[2])
    ask = float(row[3])
    ask_qty = float(row[4])
    bid_depth = float(row[8]) if row[8] is not None else bid * bid_qty
    ask_depth = ask * ask_qty
    return {
        "book_ts_ms": int(row[0]),
        "book_forward_staleness_ms": staleness_ms,
        "bid": bid,
        "ask": ask,
        "mid": float(row[5]),
        "spread_bps": float(row[6]) * 10_000.0,
        "book_imbalance": float(row[7]),
        "bid_depth_usd": bid_depth,
        "ask_depth_usd": ask_depth,
    }


def exit_book_price(conn: sqlite3.Connection, direction: str, ts_ms: int) -> tuple[int, float, str] | None:
    book = book_at_or_after(conn, SYMBOL, int(ts_ms))
    if book:
        return int(book["book_ts_ms"]), float(book["bid"] if direction == "LONG" else book["ask"]), "book"
    return None


def taker_entry_price(conn: sqlite3.Connection, direction: str, ts_ms: int) -> tuple[int, float, dict[str, Any]] | None:
    book = book_at_or_after(conn, SYMBOL, int(ts_ms))
    if not book:
        return None
    px = float(book["ask"] if direction == "LONG" else book["bid"])
    return int(book["book_ts_ms"]), px, book


def maker_limit(book: dict[str, Any], direction: str, offset_bps: float) -> float:
    if direction == "LONG":
        return float(book["bid"]) * (1.0 - float(offset_bps) / 10_000.0)
    return float(book["ask"]) * (1.0 + float(offset_bps) / 10_000.0)


def find_maker_fill(marks: Any, direction: str, detect_ts_ms: int, limit_px: float, offset_window_sec: int) -> tuple[int, float] | None:
    end = int(detect_ts_ms) + int(offset_window_sec) * 1000
    if direction == "LONG":
        required = float(limit_px) * (1.0 - MAKER_CROSS_MARGIN_BPS / 10_000.0)
        for ts, px in marks.slice_range(int(detect_ts_ms), end):
            if float(px) <= required:
                return int(ts), float(limit_px)
    else:
        required = float(limit_px) * (1.0 + MAKER_CROSS_MARGIN_BPS / 10_000.0)
        for ts, px in marks.slice_range(int(detect_ts_ms), end):
            if float(px) >= required:
                return int(ts), float(limit_px)
    return None


def path_with_stop(
    marks: Any,
    *,
    direction: str,
    entry_px: float,
    entry_ts_ms: int,
    horizon_sec: int,
    stop_bps: float | None,
) -> tuple[int | None, float | None, str, dict[str, Any]]:
    end = int(entry_ts_ms) + int(horizon_sec) * 1000
    path = [
        (int(ts), float(px), signed_return_bps(direction, float(entry_px), float(px)))
        for ts, px in marks.slice_range(int(entry_ts_ms), end)
        if int(ts) >= int(entry_ts_ms)
    ]
    if not path:
        return None, None, "NO_PATH", {}
    mfe_ts, _, mfe = max(path, key=lambda x: x[2])
    mae_ts, _, mae = min(path, key=lambda x: x[2])
    stats = {
        "mfe_bps": r1(mfe),
        "mae_bps": r1(mae),
        "mfe_sec": r1((mfe_ts - int(entry_ts_ms)) / 1000.0),
        "mae_sec": r1((mae_ts - int(entry_ts_ms)) / 1000.0),
    }
    if stop_bps is not None:
        for ts, px, ret in path:
            if ret <= -float(stop_bps):
                stats["stop_touch_sec"] = r1((ts - int(entry_ts_ms)) / 1000.0)
                return ts, px, f"SL{int(stop_bps)}", stats
    ts, px, _ = path[-1]
    return ts, px, "TIME_MARK", stats


def simulate_taker(
    conn: sqlite3.Connection,
    marks: Any,
    *,
    row: dict[str, Any],
    direction: str,
    detect_ts_ms: int,
    horizon_sec: int,
    delay_sec: int,
    stop_bps: float | None = None,
) -> dict[str, Any]:
    entry = taker_entry_price(conn, direction, int(detect_ts_ms) + int(delay_sec) * 1000)
    if not entry:
        return {"status": "NO_ENTRY_BOOK", "net_bps": None}
    entry_ts, entry_px, entry_book = entry
    if stop_bps is None:
        exit_row = exit_book_price(conn, direction, int(entry_ts) + int(horizon_sec) * 1000)
        if not exit_row:
            return {"status": "NO_EXIT_BOOK", "net_bps": None}
        exit_ts, exit_px, source = exit_row
        stats = {}
        gross = signed_return_bps(direction, float(entry_px), float(exit_px))
    else:
        exit_ts, mark_exit_px, reason, stats = path_with_stop(
            marks,
            direction=direction,
            entry_px=float(entry_px),
            entry_ts_ms=int(entry_ts),
            horizon_sec=int(horizon_sec),
            stop_bps=float(stop_bps),
        )
        if exit_ts is None or mark_exit_px is None:
            return {"status": "NO_EXIT_PATH", "net_bps": None}
        if reason.startswith("SL"):
            # Stop exits as taker at available book side after the mark stop touch.
            exit_row = exit_book_price(conn, direction, int(exit_ts))
            if exit_row:
                exit_ts, exit_px, source = exit_row
            else:
                exit_px = float(mark_exit_px)
                source = "mark_stop_fallback"
        else:
            exit_row = exit_book_price(conn, direction, int(exit_ts))
            if exit_row:
                exit_ts, exit_px, source = exit_row
            else:
                exit_px = float(mark_exit_px)
                source = "mark_time_fallback"
        gross = signed_return_bps(direction, float(entry_px), float(exit_px))
    return {
        "status": "FILLED",
        "exec_model": "TAKER",
        "direction": direction,
        "detect_ts_ms": int(detect_ts_ms),
        "entry_ts_ms": int(entry_ts),
        "entry_utc": iso_ms(entry_ts),
        "entry_price": float(entry_px),
        "exit_ts_ms": int(exit_ts),
        "exit_utc": iso_ms(exit_ts),
        "exit_price": float(exit_px),
        "exit_source": source,
        "entry_spread_bps": r1(entry_book["spread_bps"]),
        "entry_bid_depth_usd": r1(entry_book["bid_depth_usd"]),
        "entry_ask_depth_usd": r1(entry_book["ask_depth_usd"]),
        "gross_bps": r1(gross),
        "fee_bps": TAKER_FEE_BPS,
        "net_bps": r1(gross - TAKER_FEE_BPS),
        **stats,
    }


def simulate_maker(
    conn: sqlite3.Connection,
    marks: Any,
    *,
    direction: str,
    detect_ts_ms: int,
    horizon_sec: int,
    offset_bps: float,
) -> dict[str, Any]:
    book = book_at_or_after(conn, SYMBOL, detect_ts_ms)
    if not book:
        return {"status": "NO_ENTRY_BOOK", "net_bps": None}
    limit = maker_limit(book, direction, float(offset_bps))
    fill = find_maker_fill(marks, direction, int(book["book_ts_ms"]), limit, MAKER_FILL_WINDOW_SEC)
    if not fill:
        return {"status": "NO_MAKER_FILL", "net_bps": None, "offset_bps": float(offset_bps)}
    fill_ts, entry_px = fill
    exit_row = exit_book_price(conn, direction, int(fill_ts) + int(horizon_sec) * 1000)
    if not exit_row:
        return {"status": "NO_EXIT_BOOK", "net_bps": None, "offset_bps": float(offset_bps)}
    exit_ts, exit_px, source = exit_row
    gross = signed_return_bps(direction, float(entry_px), float(exit_px))
    return {
        "status": "FILLED",
        "exec_model": "MAKER_PROXY",
        "direction": direction,
        "offset_bps": float(offset_bps),
        "detect_ts_ms": int(detect_ts_ms),
        "entry_ts_ms": int(fill_ts),
        "fill_delay_sec": r1((int(fill_ts) - int(book["book_ts_ms"])) / 1000.0),
        "entry_price": float(entry_px),
        "exit_ts_ms": int(exit_ts),
        "exit_price": float(exit_px),
        "exit_source": source,
        "gross_bps": r1(gross),
        "fee_bps": MAKER_TAKER_FEE_BPS,
        "net_bps": r1(gross - MAKER_TAKER_FEE_BPS),
    }


def candidate_direction_and_horizon(candidate_id: str) -> tuple[str, int]:
    if candidate_id == "SELL_SILENCE_FADE_LONG_H4":
        return "LONG", 14_400
    if candidate_id == "SELL_PROPAGATION_MOMENTUM_SHORT_H1":
        return "SHORT", 3_600
    if candidate_id == "BUY_PROPAGATION_MOMENTUM_LONG_H1":
        return "LONG", 3_600
    if candidate_id == "BUY_SILENCE_FADE_SHORT_H4":
        return "SHORT", 14_400
    raise KeyError(candidate_id)


def selected_for_candidate(row: dict[str, Any], candidate_id: str, tau: int) -> bool:
    cand = CANDIDATES[candidate_id]
    if row.get("side") != cand["side"]:
        return False
    state = row.get(f"tau_{tau}", {})
    if cand["mode"] == "pressure":
        return bool(state.get("pressure_high"))
    return bool(state.get("silence_after_shock"))


def build_rows(conn: sqlite3.Connection) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    asset_events = {sym: build_events(conn, symbol=sym, threshold=thr) for sym, thr in ASSET_THRESHOLDS.items()}
    rows = enrich_events(conn, asset_events[SYMBOL], asset_events)
    eth_idx = event_index(asset_events[SYMBOL])
    asset_idx = {sym: event_index(evts) for sym, evts in asset_events.items()}
    for row in rows:
        for tau in sorted(set(PRIMARY_CANDIDATES.values())):
            row[f"tau_{tau}"] = __import__(
                "tools.s34_v02_propagation_candidate_gauntlet",
                fromlist=["causal_counts"],
            ).causal_counts(
                row,
                eth_events=asset_events[SYMBOL],
                eth_idx=eth_idx,
                asset_events=asset_events,
                asset_idx=asset_idx,
                tau_sec=tau,
            )
    return rows, asset_events


def split_months(rows: list[dict[str, Any]]) -> tuple[set[str], dict[str, Any]]:
    months = sorted({str(r.get("month")) for r in rows if r.get("month")})
    hold_n = max(1, round(len(months) * 0.35)) if months else 0
    hold = set(months[-hold_n:]) if hold_n else set()
    return hold, {"method": "chronological_month_tail_35pct", "months": months, "holdout_months": sorted(hold)}


def summarize_exec(rows: list[dict[str, Any]], value_key: str = "net_bps") -> dict[str, Any]:
    filled = [r for r in rows if r.get("status") == "FILLED" and finite(r.get(value_key)) is not None]
    return {
        **metrics([r.get(value_key) for r in filled]),
        "filled_n": len(filled),
        "attempt_n": len(rows),
        "fill_rate": r3(len(filled) / len(rows)) if rows else None,
    }


def run_candidate_exec(conn: sqlite3.Connection, rows: list[dict[str, Any]], candidate_id: str, tau: int, hold_months: set[str]) -> dict[str, Any]:
    marks = load_mark_index(conn, SYMBOL)
    direction, horizon = candidate_direction_and_horizon(candidate_id)
    selected = [r for r in rows if selected_for_candidate(r, candidate_id, tau)]
    taker_by_delay: dict[str, Any] = {}
    taker_rows_by_delay: dict[int, list[dict[str, Any]]] = {}
    for delay in TAKER_DELAYS_SEC:
        sim_rows = []
        for row in selected:
            detect_ts = int(row["anchor_ts_ms"]) + int(tau) * 1000
            sim = simulate_taker(conn, marks, row=row, direction=direction, detect_ts_ms=detect_ts, horizon_sec=horizon, delay_sec=delay)
            sim.update({"candidate_id": candidate_id, "tau_sec": tau, "month": row.get("month"), "anchor_utc": row.get("anchor_utc"), "side": row.get("side")})
            sim_rows.append(sim)
        taker_rows_by_delay[delay] = sim_rows
        filled = [r for r in sim_rows if r.get("status") == "FILLED"]
        taker_by_delay[str(delay)] = {
            "all": summarize_exec(sim_rows),
            "cal": summarize_exec([r for r in sim_rows if r.get("month") not in hold_months]),
            "hold": summarize_exec([r for r in sim_rows if r.get("month") in hold_months]),
            "spread_bps": metrics([r.get("entry_spread_bps") for r in filled]),
        }
    maker_by_offset: dict[str, Any] = {}
    for offset in MAKER_OFFSETS_BPS:
        sim_rows = []
        for row in selected:
            detect_ts = int(row["anchor_ts_ms"]) + int(tau) * 1000
            sim = simulate_maker(conn, marks, direction=direction, detect_ts_ms=detect_ts, horizon_sec=horizon, offset_bps=offset)
            sim.update({"candidate_id": candidate_id, "tau_sec": tau, "month": row.get("month"), "anchor_utc": row.get("anchor_utc"), "side": row.get("side")})
            sim_rows.append(sim)
        maker_by_offset[str(offset)] = {
            "all": summarize_exec(sim_rows),
            "cal": summarize_exec([r for r in sim_rows if r.get("month") not in hold_months]),
            "hold": summarize_exec([r for r in sim_rows if r.get("month") in hold_months]),
        }
    stop_by_level = {}
    for stop in STOP_LEVELS_BPS:
        sim_rows = []
        for row in selected:
            detect_ts = int(row["anchor_ts_ms"]) + int(tau) * 1000
            sim = simulate_taker(
                conn,
                marks,
                row=row,
                direction=direction,
                detect_ts_ms=detect_ts,
                horizon_sec=horizon,
                delay_sec=0,
                stop_bps=stop,
            )
            sim.update({"candidate_id": candidate_id, "tau_sec": tau, "month": row.get("month"), "anchor_utc": row.get("anchor_utc"), "side": row.get("side")})
            sim_rows.append(sim)
        stop_by_level[str(stop)] = {
            "all": summarize_exec(sim_rows),
            "hold": summarize_exec([r for r in sim_rows if r.get("month") in hold_months]),
            "stop_touch_n": sum(1 for r in sim_rows if str(r.get("exit_source", "")).startswith("SL") or r.get("stop_touch_sec") is not None),
        }
    negative_direction = "SHORT" if direction == "LONG" else "LONG"
    negative_rows = []
    for row in selected:
        detect_ts = int(row["anchor_ts_ms"]) + int(tau) * 1000
        sim = simulate_taker(conn, marks, row=row, direction=negative_direction, detect_ts_ms=detect_ts, horizon_sec=horizon, delay_sec=0)
        sim.update({"candidate_id": candidate_id, "tau_sec": tau, "month": row.get("month"), "anchor_utc": row.get("anchor_utc"), "side": row.get("side")})
        negative_rows.append(sim)
    return {
        "candidate_id": candidate_id,
        "tau_sec": int(tau),
        "direction": direction,
        "horizon_sec": int(horizon),
        "selected_n": len(selected),
        "taker_by_delay": taker_by_delay,
        "maker_by_offset": maker_by_offset,
        "stop_by_level": stop_by_level,
        "negative_control_taker0": {
            "all": summarize_exec(negative_rows),
            "hold": summarize_exec([r for r in negative_rows if r.get("month") in hold_months]),
            "direction": negative_direction,
        },
        "negative_control_anatomy": negative_anatomy(taker_rows_by_delay[0], negative_rows),
        "sample_taker0": [r for r in taker_rows_by_delay[0] if r.get("status") == "FILLED"][:10],
    }


def negative_anatomy(main_rows: list[dict[str, Any]], negative_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = []
    for main, neg in zip(main_rows, negative_rows):
        if main.get("status") != "FILLED" or neg.get("status") != "FILLED":
            continue
        pairs.append({
            "month": main.get("month"),
            "main": finite(main.get("net_bps")),
            "neg": finite(neg.get("net_bps")),
            "anchor_utc": main.get("anchor_utc"),
        })
    return {
        "negative_all": metrics([p["neg"] for p in pairs]),
        "negative_by_month": {
            m: metrics([p["neg"] for p in pairs if p["month"] == m])
            for m in sorted({str(p["month"]) for p in pairs})
        },
        "main_minus_negative": metrics([
            (float(p["main"]) - float(p["neg"]))
            for p in pairs
            if p["main"] is not None and p["neg"] is not None
        ]),
        "negative_tail_lt_-100_examples": [p for p in pairs if p["neg"] is not None and p["neg"] < -100.0][:10],
        "read": "If negative control is strongly negative at causal entry, the state is directional; if it is tail-only or month-only, treat as regime artefact.",
    }


def permutation_exec(results: list[dict[str, Any]], *, iterations: int, seed: int) -> dict[str, Any]:
    cells = []
    for res in results:
        # Use taker0 all summary from materialized sample rows only for max-stat
        vals = [finite(r.get("net_bps")) for r in res.get("sample_taker0", [])]
        vals = [float(v) for v in vals if v is not None]
        if vals:
            cells.append(vals)
    # sample_taker0 is truncated; therefore do not pretend this is a full stat.
    return {
        "status": "SKIPPED_FULL_ROWS_NOT_STORED",
        "read": "Execution suite reports holdout/T3R. Use the previous broad gauntlet max-stat plus future full-row storage for final MC correction.",
        "iterations_requested": iterations,
        "seed": seed,
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V02 Candidate Execution Gauntlet",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Research-only. No live executor/config/order logic is touched.",
        "",
        "## Verdict",
        "",
        f"- `{report['verdict']}`",
        "",
        "## Taker Causal Entry Leaderboard",
        "",
        "| Candidate | Direction | Tau | Horizon | Taker0 All | Taker0 Hold | Negative0 | Maker best | Stop best |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in report["candidate_results"]:
        taker0 = row["taker_by_delay"]["0"]
        neg = row["negative_control_taker0"]
        maker_best = max(row["maker_by_offset"].items(), key=lambda kv: float(kv[1]["all"].get("t3r_bps") or -1e18))
        stop_best = max(row["stop_by_level"].items(), key=lambda kv: float(kv[1]["all"].get("t3r_bps") or -1e18))
        lines.append(
            f"| `{row['candidate_id']}` | {row['direction']} | {row['tau_sec']} | {row['horizon_sec']} | "
            f"N={taker0['all']['n']} sum={taker0['all']['sum_bps']} med={taker0['all']['median_bps']} T3R={taker0['all']['t3r_bps']} | "
            f"N={taker0['hold']['n']} sum={taker0['hold']['sum_bps']} T3R={taker0['hold']['t3r_bps']} | "
            f"N={neg['all']['n']} sum={neg['all']['sum_bps']} T3R={neg['all']['t3r_bps']} | "
            f"O{maker_best[0]} N={maker_best[1]['all']['n']} sum={maker_best[1]['all']['sum_bps']} T3R={maker_best[1]['all']['t3r_bps']} fill={maker_best[1]['all']['fill_rate']} | "
            f"SL{stop_best[0]} sum={stop_best[1]['all']['sum_bps']} T3R={stop_best[1]['all']['t3r_bps']} |"
        )
    lines += [
        "",
        "## Delay Sensitivity",
        "",
        "```json",
        json.dumps({
            r["candidate_id"]: {delay: cell["all"] for delay, cell in r["taker_by_delay"].items()}
            for r in report["candidate_results"]
        }, indent=2, sort_keys=True),
        "```",
        "",
        "## Negative-Control Anatomy",
        "",
        "```json",
        json.dumps({
            r["candidate_id"]: r["negative_control_anatomy"]
            for r in report["candidate_results"]
        }, indent=2, sort_keys=True),
        "```",
        "",
        "## Read",
        "",
    ]
    lines.extend(f"- {x}" for x in report["read"])
    lines.append("")
    return "\n".join(lines)


def run(db: Path, *, permutations: int) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    try:
        rows, _ = build_rows(conn)
        hold_months, split = split_months(rows)
        results = [
            run_candidate_exec(conn, rows, cand_id, tau, hold_months)
            for cand_id, tau in PRIMARY_CANDIDATES.items()
        ]
    finally:
        conn.close()
    read = [
        "All entries are re-anchored to the first causal detection time: anchor + tau. This is stricter than the broad anchor-mark gauntlet.",
        "Taker0 is the cleanest executable proxy; maker rows are pullback-fill proxies and still need real queue replay.",
        "If a candidate dies after causal re-anchoring, the broad result was mostly an early-label/entry-price effect.",
        "Negative-control anatomy checks whether the large negative control is broad directional evidence or just regime/tail artefact.",
    ]
    verdict = "NO_EXECUTION_READY_CANDIDATE"
    survivors = []
    for res in results:
        t0 = res["taker_by_delay"]["0"]
        if (
            t0["all"]["n"] >= 40
            and float(t0["all"].get("sum_bps") or 0.0) > 0
            and float(t0["all"].get("t3r_bps") or 0.0) > 0
            and float(t0["hold"].get("sum_bps") or 0.0) > 0
            and float(t0["hold"].get("t3r_bps") or 0.0) > 0
            and float(t0["all"].get("t3r_bps") or -1e18) > float(res["negative_control_taker0"]["all"].get("t3r_bps") or -1e18)
        ):
            survivors.append(res["candidate_id"])
    if survivors:
        verdict = "EXECUTION_PROXY_SURVIVORS_SHADOW_ONLY"
        read.append(f"Execution proxy survivors: {', '.join(survivors)}. They still need full-row max-stat permutation, maker/taker queue realism, and forward shadow.")
    else:
        read.append("No candidate clears causal execution gates. Keep as navigation until stronger evidence appears.")
    return {
        "generated_at_utc": utc_now(),
        "research_only": True,
        "live_executor_touched": False,
        "split": split,
        "candidate_results": results,
        "permutation": permutation_exec(results, iterations=permutations, seed=3404),
        "verdict": verdict,
        "read": read,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run S34 V02 candidate execution gauntlet.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--permutations", type=int, default=PERMUTATIONS)
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
