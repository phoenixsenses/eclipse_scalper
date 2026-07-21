"""Confirmation cost test for the current S34 V Engine live execution model.

The old confirmation report used the fixed O20 ledger. The current live route is
O20 -> wait 300s -> O5 replacement with C1 fill logic, so confirmation must be
re-tested on that execution model before it can become a permission layer.

Research-only: reads DB and cancel/replace report rows, writes reports only.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import book_at, file_fingerprint, load_mark_index, r1, r3, signed_return_bps
from tools.research_s34_maker_fade import summarize
from tools.s34_v_engine_failure_anatomy import candle_features, finite_float, ohlc_after
from tools.s34_v_engine_shadow_observer import HORIZON_SEC, PROTOCOL_ID, SYMBOL, utc_now


DEFAULT_DB = ROOT / "data" / "microstructure.db"
DEFAULT_CANCEL_REPLACE_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_CANCEL_REPLACE.json"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V_ENGINE_CONFIRMATION_COST_CURRENT.json"
OUT_MD = OUT_DIR / "S34_V_ENGINE_CONFIRMATION_COST_CURRENT.md"

CONFIG_ID = "O20_W300_O5_C1"
FADE_DIRECTION = "LONG"


ConditionFn = Callable[[dict[str, Any]], bool]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ret_at(marks: Any, direction: str, entry_px: float, ts_ms: int) -> float | None:
    row = marks.at_or_after(int(ts_ms))
    if not row:
        return None
    return signed_return_bps(direction, float(entry_px), float(row[1]))


def path_stats(marks: Any, *, entry_px: float, start_ms: int, horizon_min: int) -> dict[str, Any]:
    rows = [(int(ts), float(px)) for ts, px in marks.slice_range(int(start_ms), int(start_ms) + int(horizon_min) * 60_000)]
    vals = [signed_return_bps(FADE_DIRECTION, float(entry_px), px) for _, px in rows]
    if not rows or not vals:
        return {"ret_bps": None, "mfe_bps": None, "mae_bps": None, "min_price": None, "max_price": None}
    return {
        "ret_bps": r1(ret_at(marks, FADE_DIRECTION, entry_px, int(start_ms) + int(horizon_min) * 60_000)),
        "mfe_bps": r1(max(vals)),
        "mae_bps": r1(min(vals)),
        "min_price": min(px for _, px in rows),
        "max_price": max(px for _, px in rows),
    }


def btc_context(prior4h: float | None, after15: float | None) -> str:
    if prior4h is None or after15 is None:
        return "btc_na"
    if prior4h < -50.0 and after15 >= 0.0:
        return "btc_down_then_stable"
    if prior4h < -50.0 and after15 < 0.0:
        return "btc_down_continues"
    if prior4h >= -50.0 and after15 >= 0.0:
        return "btc_supportive"
    return "btc_softening"


def annotate_row(row: dict[str, Any], *, eth_marks: Any, btc_marks: Any) -> dict[str, Any]:
    entry = finite_float(row.get("entry_price"))
    fill_ts = row.get("maker_fill_ts_ms")
    signal_ts = row.get("anchor_ts_ms")
    anchor_mark = finite_float(row.get("anchor_mark_price"))
    if entry is None or fill_ts is None or signal_ts is None:
        return dict(row)
    fill_ts = int(fill_ts)
    signal_ts = int(signal_ts)
    stats15 = path_stats(eth_marks, entry_px=entry, start_ms=fill_ts, horizon_min=15)
    stats30 = path_stats(eth_marks, entry_px=entry, start_ms=fill_ts, horizon_min=30)
    anchor_reclaim_px = anchor_mark if anchor_mark is not None else entry
    btc_prior = btc_marks.ret_bps(signal_ts - 4 * 3600 * 1000, signal_ts)
    btc_after15 = btc_marks.ret_bps(fill_ts, fill_ts + 15 * 60_000)
    candle15 = candle_features(ohlc_after(eth_marks, fill_ts, 15), ref_price=entry)
    out = dict(row)
    out.update(
        {
            "ret_15m_bps": stats15["ret_bps"],
            "mfe_15m_bps": stats15["mfe_bps"],
            "mae_15m_bps": stats15["mae_bps"],
            "ret_30m_bps": stats30["ret_bps"],
            "mfe_30m_bps": stats30["mfe_bps"],
            "mae_30m_bps": stats30["mae_bps"],
            "anchor_reclaimed_15m": bool(stats15["max_price"] is not None and float(stats15["max_price"]) >= float(anchor_reclaim_px)),
            "btc_prior_4h_bps": r1(btc_prior),
            "btc_after_15m_bps": r1(btc_after15),
            "btc_context_bucket": btc_context(btc_prior, btc_after15),
            "candle15_pattern": candle15["pattern"],
            "candle15_close_ret_bps": candle15["close_ret_bps"],
        }
    )
    return out


def conditions() -> dict[str, ConditionFn]:
    return {
        "anchor_reclaimed_15m": lambda r: bool(r.get("anchor_reclaimed_15m")),
        "btc_not_down_continues": lambda r: r.get("btc_context_bucket") != "btc_down_continues",
        "candle15_bull_reclaim": lambda r: r.get("candle15_pattern") == "bull_reclaim",
        "anchor_and_btc": lambda r: bool(r.get("anchor_reclaimed_15m")) and r.get("btc_context_bucket") != "btc_down_continues",
        "all3": lambda r: bool(r.get("anchor_reclaimed_15m"))
        and r.get("btc_context_bucket") != "btc_down_continues"
        and r.get("candle15_pattern") == "bull_reclaim",
    }


def exit_at_15(conn: sqlite3.Connection, row: dict[str, Any], *, max_book_staleness_sec: int, eth_marks: Any) -> tuple[float | None, str]:
    entry = finite_float(row.get("entry_price"))
    fill_ts = row.get("maker_fill_ts_ms")
    fee = finite_float(row.get("fee_bps")) or 5.05
    if entry is None or fill_ts is None:
        return None, "missing_entry"
    exit_ts = int(fill_ts) + 15 * 60_000
    book = book_at(conn, SYMBOL, exit_ts, max_book_staleness_sec)
    if book:
        return signed_return_bps(FADE_DIRECTION, entry, float(book.bid)) - float(fee), "book_ticker"
    mark = eth_marks.at_or_after(exit_ts)
    if mark:
        return signed_return_bps(FADE_DIRECTION, entry, float(mark[1])) - float(fee), "mark_fallback"
    return None, "no_exit"


def delayed_entry(conn: sqlite3.Connection, row: dict[str, Any], *, max_book_staleness_sec: int, eth_marks: Any) -> tuple[float | None, float | None, str]:
    original_entry = finite_float(row.get("entry_price"))
    fill_ts = row.get("maker_fill_ts_ms")
    if original_entry is None or fill_ts is None:
        return None, None, "missing_entry"
    entry_ts = int(fill_ts) + 15 * 60_000
    exit_ts = int(fill_ts) + HORIZON_SEC * 1000
    entry_book = book_at(conn, SYMBOL, entry_ts, max_book_staleness_sec)
    exit_book = book_at(conn, SYMBOL, exit_ts, max_book_staleness_sec)
    if entry_book and exit_book:
        entry_px = float(entry_book.ask)
        gross = signed_return_bps(FADE_DIRECTION, entry_px, float(exit_book.bid))
        deterioration = (entry_px - float(original_entry)) / float(original_entry) * 10_000.0
        return gross - 2.0 * 3.05, r1(deterioration), "book_ticker"
    entry_mark = eth_marks.at_or_after(entry_ts)
    exit_mark = eth_marks.at_or_after(exit_ts)
    if entry_mark and exit_mark:
        entry_px = float(entry_mark[1])
        gross = signed_return_bps(FADE_DIRECTION, entry_px, float(exit_mark[1]))
        deterioration = (entry_px - float(original_entry)) / float(original_entry) * 10_000.0
        return gross - 2.0 * 3.05, r1(deterioration), "mark_fallback"
    return None, None, "no_books"


def summarize_values(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return summarize([float(v) for r in rows if (v := finite_float(r.get(key))) is not None])


def run_condition(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
    *,
    label: str,
    fn: ConditionFn,
    max_book_staleness_sec: int,
    eth_marks: Any,
) -> dict[str, Any]:
    eval_rows = []
    for row in rows:
        passed = bool(fn(row))
        exit15, exit15_source = exit_at_15(conn, row, max_book_staleness_sec=max_book_staleness_sec, eth_marks=eth_marks)
        delayed, deterioration, delayed_source = delayed_entry(conn, row, max_book_staleness_sec=max_book_staleness_sec, eth_marks=eth_marks)
        original = finite_float(row.get("net_bps"))
        eval_rows.append(
            {
                **row,
                "confirmation_pass": passed,
                "exit15_net_bps": r1(exit15),
                "exit15_source": exit15_source,
                "kill15_net_bps": r1(original if passed else exit15),
                "delayed15_net_bps": r1(delayed) if passed else None,
                "delayed15_source": delayed_source if passed else "not_confirmed",
                "price_deterioration_bps": deterioration if passed else None,
            }
        )
    passed = [r for r in eval_rows if r["confirmation_pass"]]
    failed = [r for r in eval_rows if not r["confirmation_pass"]]
    return {
        "condition": label,
        "input_n": len(eval_rows),
        "pass_n": len(passed),
        "fail_n": len(failed),
        "pass_rate": r3(len(passed) / len(eval_rows)) if eval_rows else None,
        "filter_original": summarize_values(passed, "net_bps"),
        "failed_original": summarize_values(failed, "net_bps"),
        "kill15_hold": summarize_values(eval_rows, "kill15_net_bps"),
        "delayed15_entry": summarize_values(passed, "delayed15_net_bps"),
        "price_deterioration": summarize_values(passed, "price_deterioration_bps"),
        "rows": eval_rows,
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
    annotated = [annotate_row(r, eth_marks=eth_marks, btc_marks=btc_marks) for r in rows]
    variants = [
        run_condition(
            conn,
            annotated,
            label=label,
            fn=fn,
            max_book_staleness_sec=max_book_staleness_sec,
            eth_marks=eth_marks,
        )
        for label, fn in conditions().items()
    ]
    return {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(db_path),
        "protocol_id": PROTOCOL_ID,
        "config_id": CONFIG_ID,
        "scope": "current live execution model O20 wait300 O5 C1 filled rows",
        "baseline": summarize_values(annotated, "net_bps"),
        "variants": variants,
    }


def cell(summary: dict[str, Any]) -> str:
    return f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} T3R={summary['top3_winner_removed_sum_bps']}"


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine Confirmation Cost - Current Execution",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Config: `{report['config_id']}`",
        "",
        "Research-only. Tests whether waiting for confirmation still pays after price deterioration on the current cancel/replace execution model.",
        "",
        f"Baseline: {cell(report['baseline'])}",
        "",
        "| Condition | Pass | Filter original | Failed original | Kill@15 hold | Delayed@15 entry | Price deterioration |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in report["variants"]:
        lines.append(
            f"| `{row['condition']}` | {row['pass_n']}/{row['input_n']} | {cell(row['filter_original'])} | "
            f"{cell(row['failed_original'])} | {cell(row['kill15_hold'])} | {cell(row['delayed15_entry'])} | {cell(row['price_deterioration'])} |"
        )
    lines.extend(["", "## Read", ""])
    ranked = sorted(
        report["variants"],
        key=lambda r: (
            float(r["delayed15_entry"].get("top3_winner_removed_sum_bps") or -1e18),
            float(r["kill15_hold"].get("top3_winner_removed_sum_bps") or -1e18),
        ),
        reverse=True,
    )
    for row in ranked:
        lines.append(
            f"- `{row['condition']}` delayed {cell(row['delayed15_entry'])}; kill@15 {cell(row['kill15_hold'])}; deterioration {cell(row['price_deterioration'])}."
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test confirmation cost on current V Engine execution model.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--cancel-replace-json", type=Path, default=DEFAULT_CANCEL_REPLACE_JSON)
    parser.add_argument("--max-book-staleness-sec", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        report = build_report(
            conn,
            cancel_replace_path=args.cancel_replace_json,
            db_path=args.db,
            max_book_staleness_sec=int(args.max_book_staleness_sec),
        )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
