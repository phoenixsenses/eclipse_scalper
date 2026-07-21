"""S34 V Engine state-machine observer.

Models the frozen V Engine candidate as a transition:

    Cascade -> Capitulation -> Reclaim -> Acceptance

This is observation-only. It measures whether/when recovery confirmation appears,
how much entry price deteriorates while waiting, and whether delayed entry or
15m kill logic improves skew. It does not alter the frozen rule.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

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
    candle_features,
    finite_float,
    load_ledger,
    ohlc_after,
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
OUT_JSON = OUT_DIR / "S34_V_ENGINE_STATE_MACHINE_OBSERVER.json"
OUT_MD = OUT_DIR / "S34_V_ENGINE_STATE_MACHINE_OBSERVER.md"

MAX_CONFIRMATION_MIN = 60


def bps_between(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or float(a) <= 0.0:
        return None
    return (float(b) - float(a)) / float(a) * 10_000.0


def time_bucket(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "no_confirm"
    if seconds <= 5 * 60:
        return "confirm_0_5m"
    if seconds <= 15 * 60:
        return "confirm_5_15m"
    if seconds <= 30 * 60:
        return "confirm_15_30m"
    if seconds <= 60 * 60:
        return "confirm_30_60m"
    return "confirm_60m_plus"


def first_anchor_reclaim_ts(marks: Any, *, start_ms: int, anchor_mark_price: float, max_minutes: int) -> int | None:
    end_ms = int(start_ms) + int(max_minutes) * 60_000
    for ts, px in marks.slice_range(int(start_ms), end_ms):
        if float(px) >= float(anchor_mark_price):
            return int(ts)
    return None


def first_btc_stable_ts(btc_marks: Any, *, start_ms: int, max_minutes: int, min_wait_min: int = 5) -> int | None:
    start = btc_marks.at_or_before(int(start_ms))
    if not start or float(start[1]) <= 0.0:
        return None
    for minute in range(int(min_wait_min), int(max_minutes) + 1):
        ts = int(start_ms) + minute * 60_000
        cur = btc_marks.at_or_before(ts)
        if cur and bps_between(float(start[1]), float(cur[1])) is not None and float(bps_between(float(start[1]), float(cur[1])) or 0.0) >= 0.0:
            return ts
    return None


def first_candle_reclaim_ts(marks: Any, *, start_ms: int, ref_price: float, max_minutes: int, candle_min: int = 15) -> int | None:
    for offset_min in range(0, int(max_minutes), int(candle_min)):
        open_ts = int(start_ms) + offset_min * 60_000
        candle = candle_features(ohlc_after(marks, open_ts, int(candle_min)), ref_price=float(ref_price))
        if candle["pattern"] == "bull_reclaim":
            return open_ts + int(candle_min) * 60_000
    return None


def book_long_entry_price(conn: sqlite3.Connection, ts_ms: int, *, max_book_staleness_sec: int) -> tuple[float | None, str]:
    quote = book_at(conn, SYMBOL, int(ts_ms), int(max_book_staleness_sec))
    if quote:
        return float(quote.ask), "book_ticker"
    return None, "no_book"


def book_long_exit_price(conn: sqlite3.Connection, ts_ms: int, *, max_book_staleness_sec: int) -> tuple[float | None, str]:
    quote = book_at(conn, SYMBOL, int(ts_ms), int(max_book_staleness_sec))
    if quote:
        return float(quote.bid), "book_ticker"
    return None, "no_book"


def delayed_outcome(
    conn: sqlite3.Connection,
    *,
    entry_ts_ms: int | None,
    original_exit_ts_ms: int | None,
    max_book_staleness_sec: int,
    taker_fee_bps: float,
) -> dict[str, Any]:
    if entry_ts_ms is None or original_exit_ts_ms is None or int(entry_ts_ms) >= int(original_exit_ts_ms):
        return {"net_to_original_exit_bps": None, "source": "not_available"}
    entry_px, entry_source = book_long_entry_price(conn, int(entry_ts_ms), max_book_staleness_sec=max_book_staleness_sec)
    exit_px, exit_source = book_long_exit_price(conn, int(original_exit_ts_ms), max_book_staleness_sec=max_book_staleness_sec)
    if entry_px is None or exit_px is None:
        return {"net_to_original_exit_bps": None, "source": f"{entry_source}/{exit_source}"}
    gross = signed_return_bps(FADE_DIRECTION, entry_px, exit_px)
    return {
        "net_to_original_exit_bps": r1(gross - 2.0 * float(taker_fee_bps)),
        "entry_price": entry_px,
        "exit_price": exit_px,
        "source": "book_ticker",
    }


def build_state_rows(
    conn: sqlite3.Connection,
    ledger: list[dict[str, Any]],
    *,
    max_book_staleness_sec: int,
    taker_fee_bps: float,
) -> list[dict[str, Any]]:
    eth_marks = load_mark_index(conn, SYMBOL)
    btc_marks = load_mark_index(conn, "BTCUSDT")
    rows = []
    for row in ledger:
        if row.get("observation_status") != "CLOSED":
            continue
        signal_ts = row.get("signal_ts_ms")
        anchor_px = finite_float(row.get("anchor_mark_price"))
        limit_px = finite_float(row.get("limit_price"))
        if signal_ts is None or anchor_px is None:
            continue
        signal_ts = int(signal_ts)
        anchor_reclaim = first_anchor_reclaim_ts(
            eth_marks,
            start_ms=signal_ts,
            anchor_mark_price=anchor_px,
            max_minutes=MAX_CONFIRMATION_MIN,
        )
        btc_stable = first_btc_stable_ts(
            btc_marks,
            start_ms=signal_ts,
            max_minutes=MAX_CONFIRMATION_MIN,
        )
        candle_reclaim = first_candle_reclaim_ts(
            eth_marks,
            start_ms=signal_ts,
            ref_price=anchor_px,
            max_minutes=MAX_CONFIRMATION_MIN,
        )
        composite = None
        if anchor_reclaim is not None and btc_stable is not None and candle_reclaim is not None:
            composite = max(anchor_reclaim, btc_stable, candle_reclaim)
        original_exit_ts = row.get("exit_ts_ms")
        original_net = finite_float(row.get("net_bps"))
        delayed = delayed_outcome(
            conn,
            entry_ts_ms=composite,
            original_exit_ts_ms=int(original_exit_ts) if original_exit_ts is not None else None,
            max_book_staleness_sec=max_book_staleness_sec,
            taker_fee_bps=taker_fee_bps,
        )
        confirm_entry_px = finite_float(delayed.get("entry_price"))
        wait_cost_vs_limit = bps_between(limit_px, confirm_entry_px) if confirm_entry_px is not None else None
        wait_cost_vs_anchor = bps_between(anchor_px, confirm_entry_px) if confirm_entry_px is not None else None
        delayed_net = finite_float(delayed.get("net_to_original_exit_bps"))
        opportunity_cost = original_net - delayed_net if original_net is not None and delayed_net is not None else None
        state = "cascade"
        if anchor_reclaim is not None:
            state = "reclaim"
        if composite is not None:
            state = "acceptance"
        rows.append(
            {
                "observation_id": row.get("observation_id"),
                "signal_utc": row.get("signal_utc"),
                "sim_status": row.get("sim_status"),
                "original_net_bps": r1(original_net),
                "counterfactual_anchor_mark_net_bps": row.get("counterfactual_anchor_mark_net_bps"),
                "state_reached": state,
                "anchor_reclaim_ts_ms": anchor_reclaim,
                "anchor_reclaim_utc": iso_ms(anchor_reclaim),
                "anchor_reclaim_delay_sec": r1((anchor_reclaim - signal_ts) / 1000.0) if anchor_reclaim else None,
                "btc_stable_ts_ms": btc_stable,
                "btc_stable_utc": iso_ms(btc_stable),
                "btc_stable_delay_sec": r1((btc_stable - signal_ts) / 1000.0) if btc_stable else None,
                "candle_reclaim_ts_ms": candle_reclaim,
                "candle_reclaim_utc": iso_ms(candle_reclaim),
                "candle_reclaim_delay_sec": r1((candle_reclaim - signal_ts) / 1000.0) if candle_reclaim else None,
                "acceptance_ts_ms": composite,
                "acceptance_utc": iso_ms(composite),
                "acceptance_delay_sec": r1((composite - signal_ts) / 1000.0) if composite else None,
                "acceptance_bucket": time_bucket((composite - signal_ts) / 1000.0 if composite else None),
                "anchor_mark_price": anchor_px,
                "limit_price": limit_px,
                "acceptance_entry_price": r1(confirm_entry_px),
                "wait_cost_vs_limit_bps": r1(wait_cost_vs_limit),
                "wait_cost_vs_anchor_bps": r1(wait_cost_vs_anchor),
                "delayed_to_original_exit_bps": r1(delayed_net),
                "delayed_source": delayed.get("source"),
                "opportunity_cost_bps": r1(opportunity_cost),
                "vdepth_bps": row.get("vdepth_bps"),
                "prior_4h_bps": row.get("prior_4h_bps"),
                "running_accel_usd_per_sec": row.get("running_accel_usd_per_sec"),
                "single_liq_dominance_pct": row.get("single_liq_dominance_pct"),
            }
        )
    rows.sort(key=lambda r: str(r.get("signal_utc") or ""))
    return rows


def vals(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(v) for r in rows if (v := finite_float(r.get(key))) is not None]


def group_summary(rows: list[dict[str, Any]], key: str, value_key: str) -> list[dict[str, Any]]:
    out = []
    for value in sorted({str(r.get(key)) for r in rows}):
        subset = [r for r in rows if str(r.get(key)) == value]
        out.append(
            {
                "key": key,
                "value": value,
                "n": len(subset),
                "summary": summarize(vals(subset, value_key)),
                "wait_cost_summary": summarize(vals(subset, "wait_cost_vs_limit_bps")),
                "opportunity_cost_summary": summarize(vals(subset, "opportunity_cost_bps")),
            }
        )
    return out


def build_report(rows: list[dict[str, Any]], *, db_path: Path, ledger: list[dict[str, Any]], max_book_staleness_sec: int) -> dict[str, Any]:
    accepted = [r for r in rows if r.get("state_reached") == "acceptance"]
    closed_filled = [r for r in rows if r.get("sim_status") == "FILLED"]
    no_fill = [r for r in rows if r.get("sim_status") == "NO_MAKER_FILL"]
    return {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(db_path),
        "protocol_id": PROTOCOL_ID,
        "scope": "closed V Engine observations; state transitions measured from signal_ts",
        "config": {
            "symbol": SYMBOL,
            "direction": FADE_DIRECTION,
            "max_confirmation_min": MAX_CONFIRMATION_MIN,
            "max_book_staleness_sec": int(max_book_staleness_sec),
        },
        "counts": {
            "ledger_rows": len(ledger),
            "closed_state_rows": len(rows),
            "closed_filled_rows": len(closed_filled),
            "closed_no_fill_rows": len(no_fill),
            "acceptance_rows": len(accepted),
            "acceptance_rate": r3(len(accepted) / len(rows)) if rows else None,
        },
        "summaries": {
            "original_filled": summarize(vals(closed_filled, "original_net_bps")),
            "acceptance_delayed_to_original_exit": summarize(vals(accepted, "delayed_to_original_exit_bps")),
            "acceptance_wait_cost_vs_limit": summarize(vals(accepted, "wait_cost_vs_limit_bps")),
            "acceptance_opportunity_cost": summarize(vals(accepted, "opportunity_cost_bps")),
            "no_fill_counterfactual": summarize(vals(no_fill, "counterfactual_anchor_mark_net_bps")),
        },
        "by_acceptance_bucket": group_summary(rows, "acceptance_bucket", "delayed_to_original_exit_bps"),
        "by_state": group_summary(rows, "state_reached", "delayed_to_original_exit_bps"),
        "latest_rows": rows[-12:],
        "rows": rows,
    }


def cell(summary: dict[str, Any]) -> str:
    return f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} T3R={summary['top3_winner_removed_sum_bps']}"


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine State Machine Observer",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Protocol: `{report['protocol_id']}`",
        "",
        "Observation only. This treats the V Engine as Cascade -> Capitulation -> Reclaim -> Acceptance.",
        "",
        "## Counts",
        "",
    ]
    for key, value in report["counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Summaries",
            "",
            f"- original filled: {cell(report['summaries']['original_filled'])}",
            f"- delayed entry after acceptance to original exit: {cell(report['summaries']['acceptance_delayed_to_original_exit'])}",
            f"- wait cost vs original maker limit: {cell(report['summaries']['acceptance_wait_cost_vs_limit'])}",
            f"- opportunity cost vs original filled trade: {cell(report['summaries']['acceptance_opportunity_cost'])}",
            f"- no-fill counterfactual: {cell(report['summaries']['no_fill_counterfactual'])}",
            "",
            "## Acceptance Timing",
            "",
            "| Bucket | N | Delayed outcome | Wait cost | Opportunity cost |",
            "| --- | ---: | --- | --- | --- |",
        ]
    )
    for row in report["by_acceptance_bucket"]:
        lines.append(
            f"| `{row['value']}` | {row['n']} | {cell(row['summary'])} | "
            f"{cell(row['wait_cost_summary'])} | {cell(row['opportunity_cost_summary'])} |"
        )
    lines.extend(["", "## State", "", "| State | N | Delayed outcome | Wait cost | Opportunity cost |", "| --- | ---: | --- | --- | --- |"])
    for row in report["by_state"]:
        lines.append(
            f"| `{row['value']}` | {row['n']} | {cell(row['summary'])} | "
            f"{cell(row['wait_cost_summary'])} | {cell(row['opportunity_cost_summary'])} |"
        )
    lines.extend(["", "## Latest Rows", ""])
    lines.append("| UTC | Sim | State | Acceptance delay | Wait cost | Original | Delayed | Opp cost |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in report["latest_rows"]:
        lines.append(
            f"| {row.get('signal_utc')} | {row.get('sim_status')} | {row.get('state_reached')} | "
            f"{row.get('acceptance_delay_sec')} | {row.get('wait_cost_vs_limit_bps')} | "
            f"{row.get('original_net_bps')} | {row.get('delayed_to_original_exit_bps')} | {row.get('opportunity_cost_bps')} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Observe S34 V Engine state-machine transitions.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--ledger-jsonl", type=Path, default=DEFAULT_LEDGER_JSONL)
    p.add_argument("--max-book-staleness-sec", type=int, default=10)
    p.add_argument("--taker-fee-bps", type=float, default=3.05)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ledger = load_ledger(args.ledger_jsonl)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        rows = build_state_rows(
            conn,
            ledger,
            max_book_staleness_sec=int(args.max_book_staleness_sec),
            taker_fee_bps=float(args.taker_fee_bps),
        )
    report = build_report(rows, db_path=args.db, ledger=ledger, max_book_staleness_sec=int(args.max_book_staleness_sec))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
