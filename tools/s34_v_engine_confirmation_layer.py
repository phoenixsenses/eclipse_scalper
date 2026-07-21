"""S34 V Engine v0.1 confirmation-layer test.

Tests the 15-minute confirmation ideas from failure anatomy:
- anchor reclaimed within 15m
- BTC does not keep dumping
- 15m candle is a bull reclaim

Reports three interpretations:
1. filter_original: keep only observations matching the confirmation.
2. kill15_hold: original maker entry, but exit at 15m when confirmation fails.
3. delayed15_entry: wait 15m and enter only confirmed observations.

This is RESEARCH_ONLY and does not change v0.1 or create orders.
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

from tools.research_s34_knowable_anchor_continuation import (
    book_at,
    file_fingerprint,
    iso_ms,
    load_mark_index,
    r1,
    r3,
    signed_return_bps,
)
from tools.research_s34_maker_fade import summarize
from tools.s34_v_engine_failure_anatomy import (
    build_anatomy_rows,
    finite_float,
    load_ledger,
)
from tools.s34_v_engine_shadow_observer import (
    DEFAULT_LEDGER_JSONL,
    FADE_DIRECTION,
    HORIZON_SEC,
    PROTOCOL_ID,
    SYMBOL,
    utc_now,
)


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V_ENGINE_CONFIRMATION_LAYER.json"
OUT_MD = OUT_DIR / "S34_V_ENGINE_CONFIRMATION_LAYER.md"


ConditionFn = Callable[[dict[str, Any]], bool]


def condition_set() -> dict[str, ConditionFn]:
    return {
        "anchor_reclaimed_15m": lambda r: bool(r.get("anchor_reclaimed_15m")),
        "btc_not_down_continues": lambda r: r.get("btc_context_bucket") != "btc_down_continues",
        "candle15_bull_reclaim": lambda r: r.get("candle15_pattern") == "bull_reclaim",
        "anchor_and_btc": lambda r: bool(r.get("anchor_reclaimed_15m")) and r.get("btc_context_bucket") != "btc_down_continues",
        "anchor_and_candle15": lambda r: bool(r.get("anchor_reclaimed_15m")) and r.get("candle15_pattern") == "bull_reclaim",
        "btc_and_candle15": lambda r: r.get("btc_context_bucket") != "btc_down_continues" and r.get("candle15_pattern") == "bull_reclaim",
        "all3": lambda r: bool(r.get("anchor_reclaimed_15m"))
        and r.get("btc_context_bucket") != "btc_down_continues"
        and r.get("candle15_pattern") == "bull_reclaim",
    }


def ledger_by_id(ledger: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r.get("observation_id")): r for r in ledger if r.get("observation_id")}


def exit15_net_bps(
    conn: sqlite3.Connection,
    ledger_row: dict[str, Any],
    *,
    max_book_staleness_sec: int,
    fallback_to_mark: bool,
    marks: Any,
) -> tuple[float | None, str]:
    entry_px = finite_float(ledger_row.get("entry_price"))
    fill_ts = ledger_row.get("maker_fill_ts_ms")
    fee_bps = finite_float(ledger_row.get("fee_bps"))
    if entry_px is None or fill_ts is None:
        return None, "missing_entry"
    exit_ts = int(fill_ts) + 15 * 60_000
    book = book_at(conn, SYMBOL, exit_ts, max_book_staleness_sec)
    if book:
        exit_px = float(book.bid)
        fee = fee_bps if fee_bps is not None else 5.05
        return signed_return_bps(FADE_DIRECTION, entry_px, exit_px) - float(fee), "book_ticker"
    if fallback_to_mark:
        mark = marks.at_or_after(exit_ts)
        if mark:
            fee = fee_bps if fee_bps is not None else 5.05
            return signed_return_bps(FADE_DIRECTION, entry_px, float(mark[1])) - float(fee), "mark_fallback"
    return None, "no_exit15"


def delayed15_entry_net_bps(
    conn: sqlite3.Connection,
    ledger_row: dict[str, Any],
    *,
    max_book_staleness_sec: int,
    fallback_to_mark: bool,
    marks: Any,
    taker_fee_bps: float,
) -> tuple[float | None, str]:
    fill_ts = ledger_row.get("maker_fill_ts_ms")
    if fill_ts is None:
        return None, "missing_fill"
    entry_ts = int(fill_ts) + 15 * 60_000
    exit_ts = int(fill_ts) + HORIZON_SEC * 1000
    entry_book = book_at(conn, SYMBOL, entry_ts, max_book_staleness_sec)
    exit_book = book_at(conn, SYMBOL, exit_ts, max_book_staleness_sec)
    if entry_book and exit_book:
        gross = signed_return_bps(FADE_DIRECTION, float(entry_book.ask), float(exit_book.bid))
        return gross - 2.0 * float(taker_fee_bps), "book_ticker"
    if fallback_to_mark:
        entry_mark = marks.at_or_after(entry_ts)
        exit_mark = marks.at_or_after(exit_ts)
        if entry_mark and exit_mark:
            gross = signed_return_bps(FADE_DIRECTION, float(entry_mark[1]), float(exit_mark[1]))
            return gross - 2.0 * float(taker_fee_bps), "mark_fallback"
    return None, "no_delayed_books"


def row_card(row: dict[str, Any], *, net_key: str = "net_bps") -> dict[str, Any]:
    return {
        "signal_utc": row.get("signal_utc"),
        "net_bps": r1(row.get(net_key)),
        "original_net_bps": row.get("net_bps"),
        "confirmation_pass": row.get("confirmation_pass"),
        "anchor_reclaimed_15m": row.get("anchor_reclaimed_15m"),
        "btc_context_bucket": row.get("btc_context_bucket"),
        "candle15_pattern": row.get("candle15_pattern"),
        "ret_15m_bps": row.get("ret_15m_bps"),
        "fill_delay_sec": row.get("fill_delay_sec"),
        "vdepth_bps": row.get("vdepth_bps"),
        "prior_4h_bps": row.get("prior_4h_bps"),
    }


def summarize_variant(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [float(v) for r in rows if (v := finite_float(r.get(key))) is not None]
    return summarize(vals)


def run_variant(
    conn: sqlite3.Connection,
    anatomy_rows: list[dict[str, Any]],
    ledger_index: dict[str, dict[str, Any]],
    *,
    condition_label: str,
    condition_fn: ConditionFn,
    max_book_staleness_sec: int,
    fallback_to_mark: bool,
    marks: Any,
    taker_fee_bps: float,
) -> dict[str, Any]:
    eval_rows = []
    for row in anatomy_rows:
        lr = ledger_index.get(str(row.get("observation_id")))
        if not lr:
            continue
        passed = bool(condition_fn(row))
        exit15, exit15_source = exit15_net_bps(
            conn,
            lr,
            max_book_staleness_sec=max_book_staleness_sec,
            fallback_to_mark=fallback_to_mark,
            marks=marks,
        )
        delayed, delayed_source = delayed15_entry_net_bps(
            conn,
            lr,
            max_book_staleness_sec=max_book_staleness_sec,
            fallback_to_mark=fallback_to_mark,
            marks=marks,
            taker_fee_bps=taker_fee_bps,
        )
        original = finite_float(row.get("net_bps"))
        kill15 = original if passed else exit15
        eval_rows.append(
            {
                **row,
                "confirmation_pass": passed,
                "exit15_net_bps": r1(exit15),
                "exit15_source": exit15_source,
                "kill15_hold_net_bps": r1(kill15),
                "delayed15_entry_net_bps": r1(delayed) if passed else None,
                "delayed15_source": delayed_source if passed else "not_confirmed",
            }
        )

    passed_rows = [r for r in eval_rows if r["confirmation_pass"]]
    failed_rows = [r for r in eval_rows if not r["confirmation_pass"]]
    kill_rows = [r for r in eval_rows if finite_float(r.get("kill15_hold_net_bps")) is not None]
    delayed_rows = [r for r in passed_rows if finite_float(r.get("delayed15_entry_net_bps")) is not None]
    return {
        "condition": condition_label,
        "input_n": len(eval_rows),
        "pass_n": len(passed_rows),
        "fail_n": len(failed_rows),
        "pass_rate": r3(len(passed_rows) / len(eval_rows)) if eval_rows else None,
        "filter_original": summarize_variant(passed_rows, "net_bps"),
        "failed_original": summarize_variant(failed_rows, "net_bps"),
        "kill15_hold": summarize_variant(kill_rows, "kill15_hold_net_bps"),
        "delayed15_entry": summarize_variant(delayed_rows, "delayed15_entry_net_bps"),
        "exit15_source_counts": source_counts(eval_rows, "exit15_source"),
        "delayed15_source_counts": source_counts(eval_rows, "delayed15_source"),
        "top_passed": [row_card(r) for r in sorted(passed_rows, key=lambda r: float(r["net_bps"]), reverse=True)[:8]],
        "top_failed": [row_card(r) for r in sorted(failed_rows, key=lambda r: float(r["net_bps"]))[:8]],
        "rows": eval_rows,
    }


def source_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "none")
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def build_report(
    conn: sqlite3.Connection,
    *,
    ledger: list[dict[str, Any]],
    anatomy_rows: list[dict[str, Any]],
    db_path: Path,
    max_book_staleness_sec: int,
    fallback_to_mark: bool,
    taker_fee_bps: float,
) -> dict[str, Any]:
    marks = load_mark_index(conn, SYMBOL)
    idx = ledger_by_id(ledger)
    variants = [
        run_variant(
            conn,
            anatomy_rows,
            idx,
            condition_label=label,
            condition_fn=fn,
            max_book_staleness_sec=max_book_staleness_sec,
            fallback_to_mark=fallback_to_mark,
            marks=marks,
            taker_fee_bps=taker_fee_bps,
        )
        for label, fn in condition_set().items()
    ]
    return {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(db_path),
        "protocol_id": PROTOCOL_ID,
        "scope": "closed FILLED v0.1 observations; confirmation known 15m after maker fill",
        "config": {
            "symbol": SYMBOL,
            "direction": FADE_DIRECTION,
            "max_book_staleness_sec": int(max_book_staleness_sec),
            "fallback_to_mark": bool(fallback_to_mark),
            "taker_fee_bps": float(taker_fee_bps),
        },
        "baseline": summarize_variant(anatomy_rows, "net_bps"),
        "variants": variants,
    }


def cell(summary: dict[str, Any]) -> str:
    return f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} T3R={summary['top3_winner_removed_sum_bps']}"


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine Confirmation Layer",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Protocol: `{report['protocol_id']}`",
        "",
        "Research-only test. Confirmation is known 15 minutes after maker fill, so this is not an entry-time filter.",
        "",
        f"Baseline closed-fill original: {cell(report['baseline'])}",
        "",
        "## Variant Table",
        "",
        "| Condition | Pass | Filter original | Failed original | Kill@15 hold | Delayed@15 entry |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for row in report["variants"]:
        lines.append(
            f"| `{row['condition']}` | {row['pass_n']}/{row['input_n']} | {cell(row['filter_original'])} | "
            f"{cell(row['failed_original'])} | {cell(row['kill15_hold'])} | {cell(row['delayed15_entry'])} |"
        )
    lines.extend(["", "## Best Practical Read", ""])
    ranked = sorted(
        report["variants"],
        key=lambda r: (
            float(r["kill15_hold"].get("top3_winner_removed_sum_bps") or -1e18),
            float(r["kill15_hold"].get("sum_bps") or -1e18),
        ),
        reverse=True,
    )
    for row in ranked[:5]:
        lines.append(
            f"- `{row['condition']}` kill@15: {cell(row['kill15_hold'])}; filter-original: {cell(row['filter_original'])}; delayed@15: {cell(row['delayed15_entry'])}"
        )
    lines.extend(["", "## Fill Source Check", ""])
    for row in report["variants"]:
        lines.append(
            f"- `{row['condition']}` exit15 sources `{row['exit15_source_counts']}`, delayed15 sources `{row['delayed15_source_counts']}`"
        )
    all3 = next((r for r in report["variants"] if r["condition"] == "all3"), None)
    if all3:
        lines.extend(["", "## all3 Cards", ""])
        lines.append("Passed observations:")
        for card in all3["top_passed"]:
            lines.append(
                f"- {card['signal_utc']} net={card['original_net_bps']} ret15={card['ret_15m_bps']} btc={card['btc_context_bucket']} candle15={card['candle15_pattern']}"
            )
        lines.append("")
        lines.append("Failed observations:")
        for card in all3["top_failed"]:
            lines.append(
                f"- {card['signal_utc']} net={card['original_net_bps']} ret15={card['ret_15m_bps']} btc={card['btc_context_bucket']} candle15={card['candle15_pattern']}"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test 15m confirmation layers for S34 V Engine v0.1.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--ledger-jsonl", type=Path, default=DEFAULT_LEDGER_JSONL)
    p.add_argument("--max-book-staleness-sec", type=int, default=10)
    p.add_argument("--fallback-to-mark", action="store_true", default=True)
    p.add_argument("--no-fallback-to-mark", action="store_false", dest="fallback_to_mark")
    p.add_argument("--taker-fee-bps", type=float, default=3.05)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ledger = load_ledger(args.ledger_jsonl)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        eth_marks = load_mark_index(conn, SYMBOL)
        btc_marks = load_mark_index(conn, "BTCUSDT")
        anatomy_rows = build_anatomy_rows(ledger, eth_marks=eth_marks, btc_marks=btc_marks, rebreak_bps=10.0)
        report = build_report(
            conn,
            ledger=ledger,
            anatomy_rows=anatomy_rows,
            db_path=args.db,
            max_book_staleness_sec=int(args.max_book_staleness_sec),
            fallback_to_mark=bool(args.fallback_to_mark),
            taker_fee_bps=float(args.taker_fee_bps),
        )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
