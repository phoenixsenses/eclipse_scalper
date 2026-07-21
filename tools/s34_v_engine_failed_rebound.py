"""S34 V Engine failed-rebound and trap intervention research.

Uses the v0.1 shadow ledger plus failure anatomy features to test whether
observable post-fill failure states can improve handling:

- exit the maker LONG early when rebound fails,
- or pivot to a SHORT after the failed-rebound state appears.

Research-only; no live/paper state is changed.
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
from tools.s34_v_engine_failure_anatomy import build_anatomy_rows, finite_float, load_ledger
from tools.s34_v_engine_shadow_observer import DEFAULT_LEDGER_JSONL, FADE_DIRECTION, PROTOCOL_ID, SYMBOL, utc_now


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V_ENGINE_FAILED_REBOUND.json"
OUT_MD = OUT_DIR / "S34_V_ENGINE_FAILED_REBOUND.md"

ConditionFn = Callable[[dict[str, Any]], bool]


def ledger_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r.get("observation_id")): r for r in rows if r.get("observation_id")}


def failure_conditions(*, min_mfe_bps: float) -> dict[str, tuple[int, ConditionFn]]:
    return {
        "weak_first_15m": (15, lambda r: finite_float(r.get("ret_15m_bps")) is not None and float(r["ret_15m_bps"]) <= 0.0),
        "no_rebound_mfe15": (15, lambda r: finite_float(r.get("mfe_15m_bps")) is not None and float(r["mfe_15m_bps"]) < float(min_mfe_bps)),
        "no_anchor_reclaim_15m": (15, lambda r: not bool(r.get("anchor_reclaimed_15m"))),
        "low_rebreak_15m": (15, lambda r: bool(r.get("low_rebreak_15m"))),
        "btc_down_continues_15m": (15, lambda r: r.get("btc_context_bucket") == "btc_down_continues"),
        "failed_v_15m": (
            15,
            lambda r: (finite_float(r.get("ret_15m_bps")) is not None and float(r["ret_15m_bps"]) <= 0.0)
            and not bool(r.get("anchor_reclaimed_15m")),
        ),
        "trap_composite_15m": (
            15,
            lambda r: bool(r.get("low_rebreak_15m"))
            or (
                not bool(r.get("anchor_reclaimed_15m"))
                and r.get("btc_context_bucket") == "btc_down_continues"
            ),
        ),
        "weak_first_30m": (30, lambda r: finite_float(r.get("ret_30m_bps")) is not None and float(r["ret_30m_bps"]) <= 0.0),
        "no_anchor_reclaim_30m": (30, lambda r: not bool(r.get("anchor_reclaimed_30m"))),
        "low_rebreak_30m": (30, lambda r: bool(r.get("low_rebreak_30m"))),
        "failed_v_30m": (
            30,
            lambda r: (finite_float(r.get("ret_30m_bps")) is not None and float(r["ret_30m_bps"]) <= 0.0)
            and not bool(r.get("anchor_reclaimed_30m")),
        ),
    }


def exit_long_at(
    conn: sqlite3.Connection,
    ledger_row: dict[str, Any],
    *,
    trigger_ts_ms: int,
    max_book_staleness_sec: int,
    fallback_to_mark: bool,
    marks: Any,
) -> tuple[float | None, str]:
    entry = finite_float(ledger_row.get("entry_price"))
    fee = finite_float(ledger_row.get("fee_bps")) or 5.05
    if entry is None:
        return None, "missing_entry"
    quote = book_at(conn, SYMBOL, int(trigger_ts_ms), int(max_book_staleness_sec))
    if quote:
        return signed_return_bps(FADE_DIRECTION, entry, float(quote.bid)) - float(fee), "book_ticker"
    if fallback_to_mark:
        mark = marks.at_or_after(int(trigger_ts_ms))
        if mark:
            return signed_return_bps(FADE_DIRECTION, entry, float(mark[1])) - float(fee), "mark_fallback"
    return None, "no_exit_book"


def short_pivot(
    conn: sqlite3.Connection,
    *,
    entry_ts_ms: int,
    exit_ts_ms: int,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
    fallback_to_mark: bool,
    marks: Any,
) -> tuple[float | None, str]:
    entry_quote = book_at(conn, SYMBOL, int(entry_ts_ms), int(max_book_staleness_sec))
    exit_quote = book_at(conn, SYMBOL, int(exit_ts_ms), int(max_book_staleness_sec))
    if entry_quote and exit_quote:
        gross = signed_return_bps("SHORT", float(entry_quote.bid), float(exit_quote.ask))
        return gross - 2.0 * float(taker_fee_bps), "book_ticker"
    if fallback_to_mark:
        entry_mark = marks.at_or_after(int(entry_ts_ms))
        exit_mark = marks.at_or_after(int(exit_ts_ms))
        if entry_mark and exit_mark:
            gross = signed_return_bps("SHORT", float(entry_mark[1]), float(exit_mark[1]))
            return gross - 2.0 * float(taker_fee_bps), "mark_fallback"
    return None, "no_book"


def source_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "none")
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def eval_condition(
    conn: sqlite3.Connection,
    anatomy_rows: list[dict[str, Any]],
    ledger_idx: dict[str, dict[str, Any]],
    *,
    label: str,
    trigger_min: int,
    condition: ConditionFn,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
    fallback_to_mark: bool,
    marks: Any,
) -> dict[str, Any]:
    rows = []
    for row in anatomy_rows:
        oid = str(row.get("observation_id"))
        ledger_row = ledger_idx.get(oid)
        if not ledger_row:
            continue
        fill_ts = ledger_row.get("maker_fill_ts_ms")
        original_exit_ts = ledger_row.get("exit_ts_ms")
        if fill_ts is None or original_exit_ts is None:
            continue
        trigger_ts = int(fill_ts) + int(trigger_min) * 60_000
        original = finite_float(row.get("net_bps"))
        triggered = bool(condition(row))
        kill_net = original
        kill_source = "original_hold"
        short_to_original = None
        short_to_original_source = "not_triggered"
        short_60m = None
        short_60m_source = "not_triggered"
        if triggered:
            kill_net, kill_source = exit_long_at(
                conn,
                ledger_row,
                trigger_ts_ms=trigger_ts,
                max_book_staleness_sec=int(max_book_staleness_sec),
                fallback_to_mark=bool(fallback_to_mark),
                marks=marks,
            )
            short_to_original, short_to_original_source = short_pivot(
                conn,
                entry_ts_ms=trigger_ts,
                exit_ts_ms=int(original_exit_ts),
                taker_fee_bps=float(taker_fee_bps),
                max_book_staleness_sec=int(max_book_staleness_sec),
                fallback_to_mark=bool(fallback_to_mark),
                marks=marks,
            )
            short_60m, short_60m_source = short_pivot(
                conn,
                entry_ts_ms=trigger_ts,
                exit_ts_ms=trigger_ts + 60 * 60_000,
                taker_fee_bps=float(taker_fee_bps),
                max_book_staleness_sec=int(max_book_staleness_sec),
                fallback_to_mark=bool(fallback_to_mark),
                marks=marks,
            )
        rows.append(
            {
                **row,
                "condition": label,
                "trigger_min": int(trigger_min),
                "triggered": triggered,
                "trigger_ts_ms": trigger_ts,
                "original_net_bps": r1(original),
                "kill_net_bps": r1(kill_net),
                "kill_source": kill_source,
                "short_to_original_exit_bps": r1(short_to_original),
                "short_to_original_source": short_to_original_source,
                "short_60m_bps": r1(short_60m),
                "short_60m_source": short_60m_source,
            }
        )
    triggered_rows = [r for r in rows if r["triggered"]]
    nontriggered_rows = [r for r in rows if not r["triggered"]]
    return {
        "condition": label,
        "trigger_min": int(trigger_min),
        "input_n": len(rows),
        "trigger_n": len(triggered_rows),
        "trigger_rate": r3(len(triggered_rows) / len(rows)) if rows else None,
        "triggered_loser_rate": r3(sum(1 for r in triggered_rows if r.get("outcome_class") == "loser") / len(triggered_rows)) if triggered_rows else None,
        "original_triggered": summarize([float(r["original_net_bps"]) for r in triggered_rows if finite_float(r.get("original_net_bps")) is not None]),
        "original_not_triggered": summarize([float(r["original_net_bps"]) for r in nontriggered_rows if finite_float(r.get("original_net_bps")) is not None]),
        "kill_hold_combined": summarize([float(r["kill_net_bps"]) for r in rows if finite_float(r.get("kill_net_bps")) is not None]),
        "short_to_original_exit": summarize([float(r["short_to_original_exit_bps"]) for r in triggered_rows if finite_float(r.get("short_to_original_exit_bps")) is not None]),
        "short_60m": summarize([float(r["short_60m_bps"]) for r in triggered_rows if finite_float(r.get("short_60m_bps")) is not None]),
        "kill_source_counts": source_counts(rows, "kill_source"),
        "short_to_original_source_counts": source_counts(rows, "short_to_original_source"),
        "short_60m_source_counts": source_counts(rows, "short_60m_source"),
        "triggered_cards": cards(triggered_rows),
        "rows": rows,
    }


def cards(rows: list[dict[str, Any]], n: int = 8) -> list[dict[str, Any]]:
    out = []
    for row in sorted(rows, key=lambda r: float(r.get("original_net_bps") or 0.0))[:n]:
        out.append(
            {
                "signal_utc": row.get("signal_utc"),
                "original_net_bps": row.get("original_net_bps"),
                "kill_net_bps": row.get("kill_net_bps"),
                "short_to_original_exit_bps": row.get("short_to_original_exit_bps"),
                "short_60m_bps": row.get("short_60m_bps"),
                "ret_15m_bps": row.get("ret_15m_bps"),
                "mfe_15m_bps": row.get("mfe_15m_bps"),
                "anchor_reclaimed_15m": row.get("anchor_reclaimed_15m"),
                "low_rebreak_15m": row.get("low_rebreak_15m"),
                "btc_context_bucket": row.get("btc_context_bucket"),
                "trap_tags": row.get("trap_tags"),
            }
        )
    return out


def build_report(
    conn: sqlite3.Connection,
    *,
    ledger: list[dict[str, Any]],
    anatomy_rows: list[dict[str, Any]],
    db_path: Path,
    min_mfe_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
    fallback_to_mark: bool,
) -> dict[str, Any]:
    marks = load_mark_index(conn, SYMBOL)
    idx = ledger_by_id(ledger)
    variants = []
    for label, (trigger_min, fn) in failure_conditions(min_mfe_bps=float(min_mfe_bps)).items():
        variants.append(
            eval_condition(
                conn,
                anatomy_rows,
                idx,
                label=label,
                trigger_min=trigger_min,
                condition=fn,
                taker_fee_bps=float(taker_fee_bps),
                max_book_staleness_sec=int(max_book_staleness_sec),
                fallback_to_mark=bool(fallback_to_mark),
                marks=marks,
            )
        )
    variants.sort(
        key=lambda r: (
            float(r["kill_hold_combined"].get("top3_winner_removed_sum_bps") or -1e18),
            float(r["kill_hold_combined"].get("sum_bps") or -1e18),
            float(r["original_not_triggered"].get("sum_bps") or -1e18),
        ),
        reverse=True,
    )
    return {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(db_path),
        "protocol_id": PROTOCOL_ID,
        "scope": "closed FILLED v0.1 observations; failed-rebound labels are measured after fill",
        "config": {
            "symbol": SYMBOL,
            "direction": FADE_DIRECTION,
            "min_mfe_bps": float(min_mfe_bps),
            "taker_fee_bps": float(taker_fee_bps),
            "max_book_staleness_sec": int(max_book_staleness_sec),
            "fallback_to_mark": bool(fallback_to_mark),
        },
        "counts": {
            "ledger_rows": len(ledger),
            "anatomy_rows": len(anatomy_rows),
        },
        "baseline_original": summarize([float(r["net_bps"]) for r in anatomy_rows if finite_float(r.get("net_bps")) is not None]),
        "variants": variants,
    }


def cell(summary: dict[str, Any]) -> str:
    return f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} T3R={summary['top3_winner_removed_sum_bps']}"


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine Failed Rebound",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Protocol: `{report['protocol_id']}`",
        "",
        "Research-only. Tests whether post-fill failed-rebound states can improve exits or justify a SHORT pivot.",
        "",
        "## Baseline",
        "",
        f"- ledger rows: `{report['counts']['ledger_rows']}`",
        f"- closed filled anatomy rows: `{report['counts']['anatomy_rows']}`",
        f"- original v0.1 filled: {cell(report['baseline_original'])}",
        "",
        "## Failure Conditions",
        "",
        "| Rank | Condition | Trigger | Trigger N | Loser% | Triggered original | Not-triggered original | Kill/hold combined | SHORT to original exit | SHORT 60m |",
        "| ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(report["variants"], start=1):
        loser_pct = None if row["triggered_loser_rate"] is None else r1(row["triggered_loser_rate"] * 100.0)
        lines.append(
            f"| {idx} | `{row['condition']}` | {row['trigger_min']}m | {row['trigger_n']} | {loser_pct} | "
            f"{cell(row['original_triggered'])} | {cell(row['original_not_triggered'])} | "
            f"{cell(row['kill_hold_combined'])} | {cell(row['short_to_original_exit'])} | {cell(row['short_60m'])} |"
        )
    lines.extend(["", "## Read", ""])
    best = report["variants"][0] if report["variants"] else None
    if best:
        base_t3r = float(report["baseline_original"].get("top3_winner_removed_sum_bps") or 0.0)
        best_t3r = float(best["kill_hold_combined"].get("top3_winner_removed_sum_bps") or 0.0)
        lines.append(
            f"- Best kill/hold condition by T3R: `{best['condition']}` -> {cell(best['kill_hold_combined'])}; delta vs baseline T3R `{r1(best_t3r - base_t3r)}` bps."
        )
        lines.append(
            "- A failed-rebound label is useful only if it both isolates losing originals and improves kill/hold or SHORT outcomes after fees."
        )
    lines.extend(["", "## Triggered Worst Cards", ""])
    for row in report["variants"][:5]:
        lines.append(f"### `{row['condition']}`")
        lines.append("")
        lines.append("| UTC | Orig | Kill | Short orig-exit | Short 60m | Ret15 | MFE15 | Reclaim15 | Rebreak15 | BTC | Tags |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |")
        for card in row["triggered_cards"]:
            lines.append(
                f"| {card['signal_utc']} | {card['original_net_bps']} | {card['kill_net_bps']} | "
                f"{card['short_to_original_exit_bps']} | {card['short_60m_bps']} | {card['ret_15m_bps']} | "
                f"{card['mfe_15m_bps']} | {card['anchor_reclaimed_15m']} | {card['low_rebreak_15m']} | "
                f"{card['btc_context_bucket']} | `{card['trap_tags']}` |"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test failed-rebound labels and interventions for S34 V Engine v0.1.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--ledger-jsonl", type=Path, default=DEFAULT_LEDGER_JSONL)
    p.add_argument("--low-rebreak-bps", type=float, default=10.0)
    p.add_argument("--min-mfe-bps", type=float, default=20.0)
    p.add_argument("--taker-fee-bps", type=float, default=3.05)
    p.add_argument("--max-book-staleness-sec", type=int, default=10)
    p.add_argument("--fallback-to-mark", action="store_true")
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ledger = load_ledger(args.ledger_jsonl)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        eth_marks = load_mark_index(conn, SYMBOL)
        btc_marks = load_mark_index(conn, "BTCUSDT")
        anatomy_rows = build_anatomy_rows(
            ledger,
            eth_marks=eth_marks,
            btc_marks=btc_marks,
            rebreak_bps=float(args.low_rebreak_bps),
        )
        report = build_report(
            conn,
            ledger=ledger,
            anatomy_rows=anatomy_rows,
            db_path=args.db,
            min_mfe_bps=float(args.min_mfe_bps),
            taker_fee_bps=float(args.taker_fee_bps),
            max_book_staleness_sec=int(args.max_book_staleness_sec),
            fallback_to_mark=bool(args.fallback_to_mark),
        )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
