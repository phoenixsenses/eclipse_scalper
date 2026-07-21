"""Position-management overlays for the current S34 V Engine route.

Tests confirmation/failure states as trade management, not as delayed entry:

- keep baseline hold when recovery state appears,
- partial reduce when danger state appears,
- tighten stop after danger state appears.

Research-only. Reads historical DB + current cancel/replace report. Does not
touch live/paper/executor state.
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
from tools.s34_v_engine_confirmation_cost_current import CONFIG_ID, load_json
from tools.s34_v_engine_failure_anatomy import candle_features, finite_float, ohlc_after
from tools.s34_v_engine_shadow_observer import HORIZON_SEC, PROTOCOL_ID, SYMBOL, utc_now


DEFAULT_DB = ROOT / "data" / "microstructure.db"
DEFAULT_CANCEL_REPLACE_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_CANCEL_REPLACE.json"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V_ENGINE_POSITION_MANAGEMENT.json"
OUT_MD = OUT_DIR / "S34_V_ENGINE_POSITION_MANAGEMENT.md"

FADE_DIRECTION = "LONG"


ConditionFn = Callable[[dict[str, Any]], bool]


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


def condition_map() -> dict[str, ConditionFn]:
    return {
        "btc_down_continues": lambda r: r.get("btc_context_bucket") == "btc_down_continues",
        "failed_v": lambda r: (finite_float(r.get("ret_delay_bps")) is not None and float(r["ret_delay_bps"]) <= 0.0)
        and not bool(r.get("anchor_reclaimed")),
        "weak_first": lambda r: finite_float(r.get("ret_delay_bps")) is not None and float(r["ret_delay_bps"]) <= 0.0,
        "no_reclaim": lambda r: not bool(r.get("anchor_reclaimed")),
        "no_reclaim_btc_down": lambda r: (not bool(r.get("anchor_reclaimed"))) and r.get("btc_context_bucket") == "btc_down_continues",
    }


def exit_net_at(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    *,
    exit_ts_ms: int,
    max_book_staleness_sec: int,
    eth_marks: Any,
) -> tuple[float | None, str]:
    entry = finite_float(row.get("entry_price"))
    fee = finite_float(row.get("fee_bps")) or 5.05
    if entry is None:
        return None, "missing_entry"
    book = book_at(conn, SYMBOL, int(exit_ts_ms), max_book_staleness_sec)
    if book:
        return signed_return_bps(FADE_DIRECTION, float(entry), float(book.bid)) - float(fee), "book_ticker"
    mark = eth_marks.at_or_after(int(exit_ts_ms))
    if mark:
        return signed_return_bps(FADE_DIRECTION, float(entry), float(mark[1])) - float(fee), "mark_fallback"
    return None, "no_book"


def first_stop_after(
    row: dict[str, Any],
    *,
    trigger_ts_ms: int,
    stop_px: float,
    eth_marks: Any,
) -> int | None:
    deadline = int(row["maker_fill_ts_ms"]) + HORIZON_SEC * 1000
    for ts_ms, px in eth_marks.slice_range(int(trigger_ts_ms), deadline):
        if int(ts_ms) <= int(trigger_ts_ms):
            continue
        if float(px) <= float(stop_px):
            return int(ts_ms)
    return None


def partial_reduce(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    *,
    delay_min: int,
    fraction: float,
    condition: str,
    fn: ConditionFn,
    eth_marks: Any,
    btc_marks: Any,
    max_book_staleness_sec: int,
) -> dict[str, Any]:
    ann = annotate(row, delay_min=delay_min, eth_marks=eth_marks, btc_marks=btc_marks)
    triggered = bool(fn(ann))
    original = finite_float(row.get("net_bps"))
    trigger_ts = int(row["maker_fill_ts_ms"]) + int(delay_min) * 60_000
    if not triggered:
        net = original
        action_net = None
        source = "baseline_time"
    else:
        action_net, source = exit_net_at(conn, ann, exit_ts_ms=trigger_ts, max_book_staleness_sec=max_book_staleness_sec, eth_marks=eth_marks)
        if action_net is None or original is None:
            net = None
        else:
            net = float(fraction) * float(action_net) + (1.0 - float(fraction)) * float(original)
    return {
        **ann,
        "variant": f"partial{int(float(fraction)*100)}_{delay_min}m_{condition}",
        "triggered": triggered,
        "action_net_bps": r1(action_net),
        "managed_net_bps": r1(net),
        "source": source,
    }


def tighten_stop(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    *,
    delay_min: int,
    stop_bps: float,
    condition: str,
    fn: ConditionFn,
    stop_reference: str,
    eth_marks: Any,
    btc_marks: Any,
    max_book_staleness_sec: int,
) -> dict[str, Any]:
    ann = annotate(row, delay_min=delay_min, eth_marks=eth_marks, btc_marks=btc_marks)
    triggered = bool(fn(ann))
    original = finite_float(row.get("net_bps"))
    trigger_ts = int(row["maker_fill_ts_ms"]) + int(delay_min) * 60_000
    action_net = None
    source = "baseline_time"
    net = original
    if triggered:
        entry = finite_float(row.get("entry_price"))
        trigger_mark = eth_marks.at_or_after(trigger_ts)
        if entry is None or not trigger_mark:
            net = None
            source = "missing_ref"
        else:
            if stop_reference == "entry":
                stop_px = float(entry) * (1.0 - float(stop_bps) / 10_000.0)
            elif stop_reference == "trigger":
                stop_px = float(trigger_mark[1]) * (1.0 - float(stop_bps) / 10_000.0)
            else:
                raise ValueError(stop_reference)
            if float(trigger_mark[1]) <= stop_px:
                stop_ts = trigger_ts
            else:
                stop_ts = first_stop_after(ann, trigger_ts_ms=trigger_ts, stop_px=stop_px, eth_marks=eth_marks)
            if stop_ts is not None:
                action_net, source = exit_net_at(conn, ann, exit_ts_ms=stop_ts, max_book_staleness_sec=max_book_staleness_sec, eth_marks=eth_marks)
                net = action_net
    return {
        **ann,
        "variant": f"tight_{stop_reference}_sl{int(stop_bps)}_{delay_min}m_{condition}",
        "triggered": triggered,
        "action_net_bps": r1(action_net),
        "managed_net_bps": r1(net),
        "source": source,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    triggered = [r for r in rows if r.get("triggered")]
    return {
        "n": len(rows),
        "trigger_n": len(triggered),
        "trigger_rate": r3(len(triggered) / len(rows)) if rows else None,
        "summary": summarize([float(r["managed_net_bps"]) for r in rows if finite_float(r.get("managed_net_bps")) is not None]),
        "triggered_original": summarize([float(r["net_bps"]) for r in triggered if finite_float(r.get("net_bps")) is not None]),
        "triggered_managed": summarize([float(r["managed_net_bps"]) for r in triggered if finite_float(r.get("managed_net_bps")) is not None]),
    }


def build_report(conn: sqlite3.Connection, *, cancel_replace_path: Path, db_path: Path, max_book_staleness_sec: int) -> dict[str, Any]:
    payload = load_json(cancel_replace_path)
    base_rows = [
        r
        for r in payload.get("rows", [])
        if r.get("config_id") == CONFIG_ID and r.get("status") == "FILLED" and finite_float(r.get("net_bps")) is not None
    ]
    eth_marks = load_mark_index(conn, SYMBOL)
    btc_marks = load_mark_index(conn, "BTCUSDT")
    variants = []
    conds = condition_map()
    for delay in (5, 10, 15, 30):
        for condition in ("btc_down_continues", "failed_v", "weak_first", "no_reclaim_btc_down"):
            fn = conds[condition]
            for frac in (0.25, 0.50, 0.75):
                rows = [
                    partial_reduce(
                        conn,
                        row,
                        delay_min=delay,
                        fraction=frac,
                        condition=condition,
                        fn=fn,
                        eth_marks=eth_marks,
                        btc_marks=btc_marks,
                        max_book_staleness_sec=max_book_staleness_sec,
                    )
                    for row in base_rows
                ]
                variants.append({"variant": rows[0]["variant"] if rows else "", "type": "partial_reduce", **summarize_rows(rows)})
            for stop_ref in ("entry", "trigger"):
                for stop_bps in (40.0, 60.0, 80.0, 100.0):
                    rows = [
                        tighten_stop(
                            conn,
                            row,
                            delay_min=delay,
                            stop_bps=stop_bps,
                            condition=condition,
                            fn=fn,
                            stop_reference=stop_ref,
                            eth_marks=eth_marks,
                            btc_marks=btc_marks,
                            max_book_staleness_sec=max_book_staleness_sec,
                        )
                        for row in base_rows
                    ]
                    variants.append({"variant": rows[0]["variant"] if rows else "", "type": "tighten_stop", **summarize_rows(rows)})
    baseline = summarize([float(r["net_bps"]) for r in base_rows])
    variants.sort(
        key=lambda r: (
            float(r["summary"]["top3_winner_removed_sum_bps"] or -1e18),
            float(r["summary"]["sum_bps"] or -1e18),
            -float(r["trigger_rate"] or 0.0),
        ),
        reverse=True,
    )
    return {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(db_path),
        "protocol_id": PROTOCOL_ID,
        "config_id": CONFIG_ID,
        "baseline": baseline,
        "variants": variants,
    }


def cell(summary: dict[str, Any]) -> str:
    return f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} T3R={summary['top3_winner_removed_sum_bps']} max_loss={summary['max_loss_bps']}"


def render_md(report: dict[str, Any]) -> str:
    base = report["baseline"]
    lines = [
        "# S34 V Engine Position Management",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Config: `{report['config_id']}`",
        "",
        "Research-only. Tests post-fill management overlays on current live route.",
        "",
        f"Baseline: {cell(base)}",
        "",
        "| Rank | Variant | Type | Trigger | Managed | Triggered original | Triggered managed | Delta sum | Delta T3R |",
        "| ---: | --- | --- | ---: | --- | --- | --- | ---: | ---: |",
    ]
    base_sum = float(base.get("sum_bps") or 0.0)
    base_t3r = float(base.get("top3_winner_removed_sum_bps") or 0.0)
    for idx, row in enumerate(report["variants"][:40], start=1):
        summary = row["summary"]
        lines.append(
            f"| {idx} | `{row['variant']}` | `{row['type']}` | {row['trigger_n']}/{row['n']} | "
            f"{cell(summary)} | {cell(row['triggered_original'])} | {cell(row['triggered_managed'])} | "
            f"{r1(float(summary.get('sum_bps') or 0.0) - base_sum)} | {r1(float(summary.get('top3_winner_removed_sum_bps') or 0.0) - base_t3r)} |"
        )
    lines.extend(["", "## Read", ""])
    best = report["variants"][0] if report["variants"] else None
    if best:
        lines.append(
            f"- Best overlay by managed T3R: `{best['variant']}` -> {cell(best['summary'])}; delta T3R `{r1(float(best['summary'].get('top3_winner_removed_sum_bps') or 0.0) - base_t3r)}`."
        )
    lines.append("- Overlay is only useful if it improves T3R or materially reduces tail loss without consuming most expectancy.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test S34 V Engine position-management overlays.")
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
