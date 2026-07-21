"""Confirmation delay sweep for the current S34 V Engine execution.

Tests whether earlier confirmation windows (5/10/15/30m) preserve the useful
state signal while reducing delayed-entry price deterioration.

Research-only. No live/paper state changes.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import book_at, file_fingerprint, load_mark_index, r1, r3, signed_return_bps
from tools.research_s34_maker_fade import summarize
from tools.s34_v_engine_confirmation_cost_current import CONFIG_ID, load_json
from tools.s34_v_engine_failure_anatomy import candle_features, finite_float, ohlc_after
from tools.s34_v_engine_shadow_observer import HORIZON_SEC, PROTOCOL_ID, SYMBOL, utc_now


DEFAULT_DB = ROOT / "data" / "microstructure.db"
DEFAULT_CANCEL_REPLACE_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_CANCEL_REPLACE.json"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V_ENGINE_CONFIRMATION_DELAY_SWEEP.json"
OUT_MD = OUT_DIR / "S34_V_ENGINE_CONFIRMATION_DELAY_SWEEP.md"

FADE_DIRECTION = "LONG"


def path_window(marks: Any, *, entry_px: float, start_ms: int, minutes: int) -> dict[str, Any]:
    rows = [(int(ts), float(px)) for ts, px in marks.slice_range(int(start_ms), int(start_ms) + int(minutes) * 60_000)]
    if not rows:
        return {"ret_bps": None, "max_price": None, "min_price": None}
    end = marks.at_or_after(int(start_ms) + int(minutes) * 60_000)
    ret = None if not end else signed_return_bps(FADE_DIRECTION, float(entry_px), float(end[1]))
    return {"ret_bps": r1(ret), "max_price": max(px for _, px in rows), "min_price": min(px for _, px in rows)}


def btc_context(prior4h: float | None, after: float | None) -> str:
    if prior4h is None or after is None:
        return "btc_na"
    if prior4h < -50.0 and after >= 0.0:
        return "btc_down_then_stable"
    if prior4h < -50.0 and after < 0.0:
        return "btc_down_continues"
    if prior4h >= -50.0 and after >= 0.0:
        return "btc_supportive"
    return "btc_softening"


def delayed_net(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    *,
    delay_min: int,
    eth_marks: Any,
    max_book_staleness_sec: int,
) -> tuple[float | None, float | None, str]:
    entry0 = finite_float(row.get("entry_price"))
    fill_ts = row.get("maker_fill_ts_ms")
    if entry0 is None or fill_ts is None:
        return None, None, "missing_entry"
    entry_ts = int(fill_ts) + int(delay_min) * 60_000
    exit_ts = int(fill_ts) + HORIZON_SEC * 1000
    entry_book = book_at(conn, SYMBOL, entry_ts, max_book_staleness_sec)
    exit_book = book_at(conn, SYMBOL, exit_ts, max_book_staleness_sec)
    if entry_book and exit_book:
        entry_px = float(entry_book.ask)
        gross = signed_return_bps(FADE_DIRECTION, entry_px, float(exit_book.bid))
        return gross - 6.1, r1((entry_px - entry0) / entry0 * 10_000.0), "book_ticker"
    entry = eth_marks.at_or_after(entry_ts)
    exit_ = eth_marks.at_or_after(exit_ts)
    if entry and exit_:
        entry_px = float(entry[1])
        gross = signed_return_bps(FADE_DIRECTION, entry_px, float(exit_[1]))
        return gross - 6.1, r1((entry_px - entry0) / entry0 * 10_000.0), "mark_fallback"
    return None, None, "no_book"


def kill_net(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    *,
    delay_min: int,
    eth_marks: Any,
    max_book_staleness_sec: int,
) -> tuple[float | None, str]:
    entry0 = finite_float(row.get("entry_price"))
    fill_ts = row.get("maker_fill_ts_ms")
    fee = finite_float(row.get("fee_bps")) or 5.05
    if entry0 is None or fill_ts is None:
        return None, "missing_entry"
    exit_ts = int(fill_ts) + int(delay_min) * 60_000
    book = book_at(conn, SYMBOL, exit_ts, max_book_staleness_sec)
    if book:
        return signed_return_bps(FADE_DIRECTION, entry0, float(book.bid)) - float(fee), "book_ticker"
    mark = eth_marks.at_or_after(exit_ts)
    if mark:
        return signed_return_bps(FADE_DIRECTION, entry0, float(mark[1])) - float(fee), "mark_fallback"
    return None, "no_book"


def annotate(row: dict[str, Any], *, delay_min: int, eth_marks: Any, btc_marks: Any) -> dict[str, Any]:
    entry = finite_float(row.get("entry_price"))
    fill_ts = row.get("maker_fill_ts_ms")
    signal_ts = row.get("anchor_ts_ms")
    anchor_mark = finite_float(row.get("anchor_mark_price")) or entry
    if entry is None or fill_ts is None or signal_ts is None:
        return dict(row)
    fill_ts = int(fill_ts)
    signal_ts = int(signal_ts)
    path = path_window(eth_marks, entry_px=entry, start_ms=fill_ts, minutes=delay_min)
    btc_prior = btc_marks.ret_bps(signal_ts - 4 * 3600 * 1000, signal_ts)
    btc_after = btc_marks.ret_bps(fill_ts, fill_ts + int(delay_min) * 60_000)
    candle = candle_features(ohlc_after(eth_marks, fill_ts, delay_min), ref_price=entry)
    out = dict(row)
    out.update(
        {
            "delay_min": int(delay_min),
            "ret_delay_bps": path["ret_bps"],
            "anchor_reclaimed": bool(path["max_price"] is not None and float(path["max_price"]) >= float(anchor_mark)),
            "btc_context_bucket": btc_context(btc_prior, btc_after),
            "candle_pattern": candle["pattern"],
            "candle_close_ret_bps": candle["close_ret_bps"],
        }
    )
    return out


def passes(row: dict[str, Any], condition: str) -> bool:
    if condition == "anchor_reclaimed":
        return bool(row.get("anchor_reclaimed"))
    if condition == "btc_not_down_continues":
        return row.get("btc_context_bucket") != "btc_down_continues"
    if condition == "anchor_and_btc":
        return bool(row.get("anchor_reclaimed")) and row.get("btc_context_bucket") != "btc_down_continues"
    if condition == "candle_bull_reclaim":
        return row.get("candle_pattern") == "bull_reclaim"
    if condition == "all3":
        return bool(row.get("anchor_reclaimed")) and row.get("btc_context_bucket") != "btc_down_continues" and row.get("candle_pattern") == "bull_reclaim"
    raise ValueError(condition)


def summarize_key(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return summarize([float(v) for r in rows if (v := finite_float(r.get(key))) is not None])


def run_cell(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
    *,
    delay_min: int,
    condition: str,
    eth_marks: Any,
    btc_marks: Any,
    max_book_staleness_sec: int,
) -> dict[str, Any]:
    eval_rows = []
    for raw in rows:
        row = annotate(raw, delay_min=delay_min, eth_marks=eth_marks, btc_marks=btc_marks)
        ok = passes(row, condition)
        dnet, deterioration, dsrc = delayed_net(conn, row, delay_min=delay_min, eth_marks=eth_marks, max_book_staleness_sec=max_book_staleness_sec)
        knet, ksrc = kill_net(conn, row, delay_min=delay_min, eth_marks=eth_marks, max_book_staleness_sec=max_book_staleness_sec)
        original = finite_float(row.get("net_bps"))
        eval_rows.append(
            {
                **row,
                "confirmation_pass": ok,
                "delayed_net_bps": r1(dnet) if ok else None,
                "price_deterioration_bps": deterioration if ok else None,
                "delayed_source": dsrc if ok else "not_confirmed",
                "kill_net_bps": r1(original if ok else knet),
                "kill_source": "original_hold" if ok else ksrc,
            }
        )
    pass_rows = [r for r in eval_rows if r["confirmation_pass"]]
    fail_rows = [r for r in eval_rows if not r["confirmation_pass"]]
    return {
        "delay_min": int(delay_min),
        "condition": condition,
        "input_n": len(eval_rows),
        "pass_n": len(pass_rows),
        "pass_rate": r3(len(pass_rows) / len(eval_rows)) if eval_rows else None,
        "filter_original": summarize_key(pass_rows, "net_bps"),
        "failed_original": summarize_key(fail_rows, "net_bps"),
        "delayed_entry": summarize_key(pass_rows, "delayed_net_bps"),
        "kill_hold": summarize_key(eval_rows, "kill_net_bps"),
        "price_deterioration": summarize_key(pass_rows, "price_deterioration_bps"),
    }


def build_report(conn: sqlite3.Connection, *, cancel_replace_path: Path, db_path: Path, max_book_staleness_sec: int) -> dict[str, Any]:
    payload = load_json(cancel_replace_path)
    rows = [
        r
        for r in payload.get("rows", [])
        if r.get("config_id") == CONFIG_ID and r.get("status") == "FILLED" and finite_float(r.get("net_bps")) is not None
    ]
    eth_marks = load_mark_index(conn, SYMBOL)
    btc_marks = load_mark_index(conn, "BTCUSDT")
    cells = [
        run_cell(
            conn,
            rows,
            delay_min=delay,
            condition=condition,
            eth_marks=eth_marks,
            btc_marks=btc_marks,
            max_book_staleness_sec=max_book_staleness_sec,
        )
        for delay in (5, 10, 15, 30)
        for condition in ("anchor_reclaimed", "btc_not_down_continues", "anchor_and_btc", "candle_bull_reclaim", "all3")
    ]
    cells.sort(
        key=lambda r: (
            float(r["delayed_entry"].get("top3_winner_removed_sum_bps") or -1e18),
            float(r["kill_hold"].get("top3_winner_removed_sum_bps") or -1e18),
            float(r["delayed_entry"].get("sum_bps") or -1e18),
        ),
        reverse=True,
    )
    return {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(db_path),
        "protocol_id": PROTOCOL_ID,
        "config_id": CONFIG_ID,
        "baseline": summarize_key(rows, "net_bps"),
        "cells": cells,
    }


def cell(summary: dict[str, Any]) -> str:
    return f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} T3R={summary['top3_winner_removed_sum_bps']}"


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine Confirmation Delay Sweep",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Config: `{report['config_id']}`",
        "",
        f"Baseline: {cell(report['baseline'])}",
        "",
        "| Rank | Delay | Condition | Pass | Delayed entry | Kill/hold | Filter original | Deterioration |",
        "| ---: | ---: | --- | ---: | --- | --- | --- | --- |",
    ]
    for i, row in enumerate(report["cells"], start=1):
        lines.append(
            f"| {i} | {row['delay_min']}m | `{row['condition']}` | {row['pass_n']}/{row['input_n']} | "
            f"{cell(row['delayed_entry'])} | {cell(row['kill_hold'])} | {cell(row['filter_original'])} | {cell(row['price_deterioration'])} |"
        )
    lines.extend(["", "## Read", ""])
    best = report["cells"][0] if report["cells"] else None
    if best:
        lines.append(
            f"- Best delayed-entry T3R cell: {best['delay_min']}m `{best['condition']}` -> delayed {cell(best['delayed_entry'])}, deterioration {cell(best['price_deterioration'])}."
        )
    lines.append("- A confirmation layer must beat the baseline after deterioration; otherwise it is only a dashboard state label.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep confirmation delays for current V Engine execution.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--cancel-replace-json", type=Path, default=DEFAULT_CANCEL_REPLACE_JSON)
    parser.add_argument("--max-book-staleness-sec", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        report = build_report(conn, cancel_replace_path=args.cancel_replace_json, db_path=args.db, max_book_staleness_sec=int(args.max_book_staleness_sec))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
