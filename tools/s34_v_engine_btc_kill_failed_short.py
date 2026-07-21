"""S34 V Engine BTC kill switch and failed-V SHORT shadow.

Tests two practical branches from the V Engine state-machine work:

1. Keep the frozen maker LONG entry, but exit early when BTC keeps dropping.
2. In the same failed-V state, evaluate a separate SHORT continuation shadow.

Research-only. Uses book_ticker for early exits and shadow SHORT fills; no mark
fallback, no live/paper state changes.
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
    load_mark_index,
    r1,
    r3,
    signed_return_bps,
)
from tools.research_s34_maker_fade import summarize
from tools.s34_v_engine_failure_anatomy import (
    build_anatomy_rows,
    candle_features,
    finite_float,
    load_ledger,
    ohlc_after,
)
from tools.s34_v_engine_shadow_observer import (
    DEFAULT_LEDGER_JSONL,
    FADE_DIRECTION,
    PROTOCOL_ID,
    SYMBOL,
    utc_now,
)


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V_ENGINE_BTC_KILL_FAILED_SHORT.json"
OUT_MD = OUT_DIR / "S34_V_ENGINE_BTC_KILL_FAILED_SHORT.md"


ConditionFn = Callable[[dict[str, Any]], bool]


def parse_int_tuple(text: str) -> tuple[int, ...]:
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    if not vals:
        raise ValueError("empty int tuple")
    return tuple(vals)


def parse_float_tuple(text: str) -> tuple[float, ...]:
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    if not vals:
        raise ValueError("empty float tuple")
    return tuple(vals)


def source_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "none")
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def ledger_index(ledger: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("observation_id")): row for row in ledger if row.get("observation_id")}


def max_mark_between(marks: Any, start_ms: int, end_ms: int) -> float | None:
    vals = [float(px) for _, px in marks.slice_range(int(start_ms), int(end_ms))]
    return max(vals) if vals else None


def btc_ret_after_fill(row: dict[str, Any], btc_marks: Any, check_min: int) -> float | None:
    fill_ts = row.get("maker_fill_ts_ms")
    if fill_ts is None:
        return None
    return btc_marks.ret_bps(int(fill_ts), int(fill_ts) + int(check_min) * 60_000)


def anchor_reclaimed_by_check(row: dict[str, Any], eth_marks: Any, check_min: int) -> bool:
    fill_ts = row.get("maker_fill_ts_ms")
    anchor_px = finite_float(row.get("anchor_mark_price"))
    if fill_ts is None or anchor_px is None:
        return False
    hi = max_mark_between(eth_marks, int(fill_ts), int(fill_ts) + int(check_min) * 60_000)
    return bool(hi is not None and float(hi) >= float(anchor_px))


def candle_bear_by_check(row: dict[str, Any], eth_marks: Any, check_min: int) -> bool:
    fill_ts = row.get("maker_fill_ts_ms")
    entry_px = finite_float(row.get("entry_price"))
    if fill_ts is None or entry_px is None:
        return False
    candle = candle_features(ohlc_after(eth_marks, int(fill_ts), int(check_min)), ref_price=float(entry_px))
    return candle.get("pattern") == "bear_followthrough"


def early_long_exit_net(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    *,
    check_min: int,
    max_book_staleness_sec: int,
) -> tuple[float | None, str]:
    entry_px = finite_float(row.get("entry_price"))
    fill_ts = row.get("maker_fill_ts_ms")
    fee_bps = finite_float(row.get("fee_bps"))
    if entry_px is None or fill_ts is None:
        return None, "missing_entry"
    exit_ts = int(fill_ts) + int(check_min) * 60_000
    quote = book_at(conn, SYMBOL, exit_ts, int(max_book_staleness_sec))
    if not quote:
        return None, "no_exit_book"
    fee = fee_bps if fee_bps is not None else 5.05
    gross = signed_return_bps(FADE_DIRECTION, entry_px, float(quote.bid))
    return gross - float(fee), "book_ticker"


def short_shadow_net(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    *,
    check_min: int,
    max_book_staleness_sec: int,
    taker_fee_bps: float,
) -> tuple[float | None, str]:
    fill_ts = row.get("maker_fill_ts_ms")
    exit_ts = row.get("exit_ts_ms")
    if fill_ts is None or exit_ts is None:
        return None, "missing_ts"
    entry_ts = int(fill_ts) + int(check_min) * 60_000
    if entry_ts >= int(exit_ts):
        return None, "entry_after_exit"
    entry_quote = book_at(conn, SYMBOL, entry_ts, int(max_book_staleness_sec))
    exit_quote = book_at(conn, SYMBOL, int(exit_ts), int(max_book_staleness_sec))
    if not entry_quote or not exit_quote:
        return None, "no_book"
    gross = signed_return_bps("SHORT", float(entry_quote.bid), float(exit_quote.ask))
    return gross - 2.0 * float(taker_fee_bps), "book_ticker"


def condition_rows(
    anatomy_rows: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    *,
    eth_marks: Any,
    btc_marks: Any,
    check_min: int,
    btc_threshold_bps: float,
) -> list[dict[str, Any]]:
    idx = ledger_index(ledger)
    rows = []
    for ar in anatomy_rows:
        lr = idx.get(str(ar.get("observation_id")))
        if not lr:
            continue
        if lr.get("observation_status") != "CLOSED" or lr.get("sim_status") != "FILLED":
            continue
        btc_ret = btc_ret_after_fill(lr, btc_marks, int(check_min))
        anchor_reclaimed = anchor_reclaimed_by_check(lr, eth_marks, int(check_min))
        candle_bear = candle_bear_by_check(lr, eth_marks, int(check_min))
        rows.append(
            {
                **lr,
                **{
                    "btc_ret_check_bps": r1(btc_ret),
                    "btc_down_condition": btc_ret is not None and float(btc_ret) <= float(btc_threshold_bps),
                    "anchor_not_reclaimed_condition": not anchor_reclaimed,
                    "candle_bear_condition": bool(candle_bear),
                    "btc_and_anchor_fail": (btc_ret is not None and float(btc_ret) <= float(btc_threshold_bps)) and not anchor_reclaimed,
                    "btc_and_candle_fail": (btc_ret is not None and float(btc_ret) <= float(btc_threshold_bps)) and bool(candle_bear),
                    "any_failure": (btc_ret is not None and float(btc_ret) <= float(btc_threshold_bps)) or (not anchor_reclaimed) or bool(candle_bear),
                    "anatomy_net_bps": ar.get("net_bps"),
                    "anatomy_ret_15m_bps": ar.get("ret_15m_bps"),
                    "anatomy_btc_context": ar.get("btc_context_bucket"),
                    "anatomy_candle15": ar.get("candle15_pattern"),
                },
            }
        )
    return rows


def apply_variant(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
    *,
    condition_key: str,
    check_min: int,
    max_book_staleness_sec: int,
    taker_fee_bps: float,
) -> dict[str, Any]:
    eval_rows = []
    for row in rows:
        condition = bool(row.get(condition_key))
        early_net, early_source = early_long_exit_net(
            conn,
            row,
            check_min=int(check_min),
            max_book_staleness_sec=int(max_book_staleness_sec),
        )
        original = finite_float(row.get("net_bps"))
        kill_net = early_net if condition else original
        short_net, short_source = short_shadow_net(
            conn,
            row,
            check_min=int(check_min),
            max_book_staleness_sec=int(max_book_staleness_sec),
            taker_fee_bps=float(taker_fee_bps),
        )
        eval_rows.append(
            {
                "observation_id": row.get("observation_id"),
                "signal_utc": row.get("signal_utc"),
                "condition": condition,
                "condition_key": condition_key,
                "check_min": int(check_min),
                "btc_ret_check_bps": row.get("btc_ret_check_bps"),
                "original_net_bps": r1(original),
                "early_long_exit_net_bps": r1(early_net),
                "early_long_exit_source": early_source,
                "kill_switch_net_bps": r1(kill_net),
                "short_shadow_net_bps": r1(short_net) if condition else None,
                "short_shadow_source": short_source if condition else "not_triggered",
                "fill_delay_sec": row.get("fill_delay_sec"),
                "vdepth_bps": row.get("vdepth_bps"),
                "prior_4h_bps": row.get("prior_4h_bps"),
                "anatomy_ret_15m_bps": row.get("anatomy_ret_15m_bps"),
                "anatomy_btc_context": row.get("anatomy_btc_context"),
                "anatomy_candle15": row.get("anatomy_candle15"),
            }
        )
    triggered = [r for r in eval_rows if r["condition"]]
    not_triggered = [r for r in eval_rows if not r["condition"]]
    return {
        "condition_key": condition_key,
        "check_min": int(check_min),
        "input_n": len(eval_rows),
        "trigger_n": len(triggered),
        "trigger_rate": r3(len(triggered) / len(eval_rows)) if eval_rows else None,
        "triggered_original": summarize_values(triggered, "original_net_bps"),
        "not_triggered_original": summarize_values(not_triggered, "original_net_bps"),
        "kill_switch": summarize_values(eval_rows, "kill_switch_net_bps"),
        "short_shadow": summarize_values(triggered, "short_shadow_net_bps"),
        "early_exit_source_counts": source_counts(eval_rows, "early_long_exit_source"),
        "short_source_counts": source_counts(eval_rows, "short_shadow_source"),
        "trigger_cards": cards(triggered, "kill_switch_net_bps"),
        "rows": eval_rows,
    }


def summarize_values(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [float(v) for r in rows if (v := finite_float(r.get(key))) is not None]
    return summarize(vals)


def cards(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    sortable = [r for r in rows if finite_float(r.get(key)) is not None]
    sortable.sort(key=lambda r: float(r[key]))
    return [
        {
            "signal_utc": r.get("signal_utc"),
            "original_net_bps": r.get("original_net_bps"),
            "kill_switch_net_bps": r.get("kill_switch_net_bps"),
            "short_shadow_net_bps": r.get("short_shadow_net_bps"),
            "btc_ret_check_bps": r.get("btc_ret_check_bps"),
            "ret15_bps": r.get("anatomy_ret_15m_bps"),
            "btc_context": r.get("anatomy_btc_context"),
            "candle15": r.get("anatomy_candle15"),
        }
        for r in sortable[:8]
    ]


def run_sweep(
    conn: sqlite3.Connection,
    *,
    ledger: list[dict[str, Any]],
    anatomy_rows: list[dict[str, Any]],
    check_minutes: tuple[int, ...],
    btc_thresholds: tuple[float, ...],
    max_book_staleness_sec: int,
    taker_fee_bps: float,
) -> list[dict[str, Any]]:
    eth_marks = load_mark_index(conn, SYMBOL)
    btc_marks = load_mark_index(conn, "BTCUSDT")
    variants = []
    condition_keys = (
        "btc_down_condition",
        "anchor_not_reclaimed_condition",
        "candle_bear_condition",
        "btc_and_anchor_fail",
        "btc_and_candle_fail",
        "any_failure",
    )
    for check_min in check_minutes:
        for btc_threshold in btc_thresholds:
            base_rows = condition_rows(
                anatomy_rows,
                ledger,
                eth_marks=eth_marks,
                btc_marks=btc_marks,
                check_min=int(check_min),
                btc_threshold_bps=float(btc_threshold),
            )
            for key in condition_keys:
                variant = apply_variant(
                    conn,
                    base_rows,
                    condition_key=key,
                    check_min=int(check_min),
                    max_book_staleness_sec=int(max_book_staleness_sec),
                    taker_fee_bps=float(taker_fee_bps),
                )
                variant["btc_threshold_bps"] = float(btc_threshold)
                variant["label"] = f"{key}_T{float(btc_threshold):g}_M{int(check_min)}"
                variants.append(variant)
    variants.sort(
        key=lambda v: (
            float(v["kill_switch"].get("top3_winner_removed_sum_bps") or -1e18),
            float(v["kill_switch"].get("sum_bps") or -1e18),
            float(v["short_shadow"].get("top3_winner_removed_sum_bps") or -1e18),
        ),
        reverse=True,
    )
    return variants


def cell(summary: dict[str, Any]) -> str:
    return f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} T3R={summary['top3_winner_removed_sum_bps']}"


def render_md(report: dict[str, Any]) -> str:
    baseline_t3r = float(report["baseline"].get("top3_winner_removed_sum_bps") or 0.0)
    baseline_sum = float(report["baseline"].get("sum_bps") or 0.0)
    best_kill = report["variants"][0] if report["variants"] else None
    short_ranked = sorted(
        report["variants"],
        key=lambda v: (
            float(v["short_shadow"].get("top3_winner_removed_sum_bps") or -1e18),
            float(v["short_shadow"].get("sum_bps") or -1e18),
        ),
        reverse=True,
    )
    best_short = short_ranked[0] if short_ranked else None
    kill_pass = bool(
        best_kill
        and float(best_kill["kill_switch"].get("top3_winner_removed_sum_bps") or -1e18) > baseline_t3r
        and float(best_kill["kill_switch"].get("sum_bps") or -1e18) > baseline_sum
    )
    short_pass = bool(best_short and float(best_short["short_shadow"].get("sum_bps") or -1e18) > 0.0)
    lines = [
        "# S34 V Engine BTC Kill + Failed SHORT",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Protocol: `{report['protocol_id']}`",
        "",
        "Research-only. Early exits and SHORT shadows use book_ticker only; no mark fallback.",
        "",
        f"Baseline original LONG: {cell(report['baseline'])}",
        "",
        "## Verdict",
        "",
        f"- BTC kill switch improves baseline: `{'YES' if kill_pass else 'NO'}`",
        f"- Failed-V SHORT has positive shadow expectancy: `{'YES' if short_pass else 'NO'}`",
        f"- best kill variant: `{best_kill['label'] if best_kill else None}` -> {cell(best_kill['kill_switch']) if best_kill else None}",
        f"- best SHORT shadow: `{best_short['label'] if best_short else None}` -> {cell(best_short['short_shadow']) if best_short else None}",
        "",
        "## Best Kill Switch Variants",
        "",
        "| Rank | Label | Trig | Triggered original | Not-triggered original | Kill switch | SHORT shadow |",
        "| ---: | --- | ---: | --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(report["variants"][:30], start=1):
        trig_pct = None if row["trigger_rate"] is None else r1(row["trigger_rate"] * 100.0)
        lines.append(
            f"| {idx} | `{row['label']}` | {row['trigger_n']}/{row['input_n']} ({trig_pct}%) | "
            f"{cell(row['triggered_original'])} | {cell(row['not_triggered_original'])} | "
            f"{cell(row['kill_switch'])} | {cell(row['short_shadow'])} |"
        )

    lines.extend(["", "## Best Failed-V SHORT Shadows", ""])
    lines.append("| Rank | Label | Trig | SHORT shadow | Triggered original |")
    lines.append("| ---: | --- | ---: | --- | --- |")
    for idx, row in enumerate(short_ranked[:20], start=1):
        trig_pct = None if row["trigger_rate"] is None else r1(row["trigger_rate"] * 100.0)
        lines.append(
            f"| {idx} | `{row['label']}` | {row['trigger_n']}/{row['input_n']} ({trig_pct}%) | "
            f"{cell(row['short_shadow'])} | {cell(row['triggered_original'])} |"
        )

    best = report["variants"][0] if report["variants"] else None
    if best:
        lines.extend(["", "## Best Kill Trigger Cards", ""])
        lines.append(f"Best label: `{best['label']}`")
        lines.append("")
        lines.append("| UTC | Original | Kill | SHORT | BTC ret | Ret15 | BTC context | Candle15 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
        for card in best["trigger_cards"]:
            lines.append(
                f"| {card['signal_utc']} | {card['original_net_bps']} | {card['kill_switch_net_bps']} | "
                f"{card['short_shadow_net_bps']} | {card['btc_ret_check_bps']} | {card['ret15_bps']} | "
                f"{card['btc_context']} | {card['candle15']} |"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test BTC kill switch and failed-V SHORT shadow for S34 V Engine.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--ledger-jsonl", type=Path, default=DEFAULT_LEDGER_JSONL)
    p.add_argument("--check-minutes", default="5,10,15,30")
    p.add_argument("--btc-thresholds-bps", default="0,-10,-20")
    p.add_argument("--max-book-staleness-sec", type=int, default=10)
    p.add_argument("--taker-fee-bps", type=float, default=3.05)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checks = parse_int_tuple(args.check_minutes)
    thresholds = parse_float_tuple(args.btc_thresholds_bps)
    ledger = load_ledger(args.ledger_jsonl)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        eth_marks = load_mark_index(conn, SYMBOL)
        btc_marks = load_mark_index(conn, "BTCUSDT")
        anatomy_rows = build_anatomy_rows(ledger, eth_marks=eth_marks, btc_marks=btc_marks, rebreak_bps=10.0)
        variants = run_sweep(
            conn,
            ledger=ledger,
            anatomy_rows=anatomy_rows,
            check_minutes=checks,
            btc_thresholds=thresholds,
            max_book_staleness_sec=int(args.max_book_staleness_sec),
            taker_fee_bps=float(args.taker_fee_bps),
        )
    baseline = summarize_values(anatomy_rows, "net_bps")
    report = {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(args.db),
        "protocol_id": PROTOCOL_ID,
        "config": {
            "symbol": SYMBOL,
            "check_minutes": list(checks),
            "btc_thresholds_bps": list(thresholds),
            "max_book_staleness_sec": int(args.max_book_staleness_sec),
            "taker_fee_bps": float(args.taker_fee_bps),
        },
        "baseline": baseline,
        "variants": variants,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
