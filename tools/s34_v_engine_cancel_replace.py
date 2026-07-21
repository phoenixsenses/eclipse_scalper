"""S34 V Engine cancel/replace execution research.

Tests whether the frozen V Engine v0.1 maker entry should keep a passive O20
limit, cancel if unfilled, or replace with a more aggressive limit after a
predefined wait. Research-only; does not touch live/paper state.
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

from tools.research_s34_knowable_anchor_continuation import book_at, file_fingerprint, iso_ms, load_mark_index, r1, r3, signed_return_bps
from tools.research_s34_maker_fade import maker_limit_price, summarize
from tools.s34_v_engine_execution_frontier import anchor_mark_counterfactual, collect_v01_events, parse_float_tuple, prior_return_bps
from tools.s34_v_engine_shadow_observer import HORIZON_SEC, PROTOCOL_ID, SYMBOL, utc_now


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V_ENGINE_CANCEL_REPLACE.json"
OUT_MD = OUT_DIR / "S34_V_ENGINE_CANCEL_REPLACE.md"

NO_REPLACE = "cancel"


def parse_int_tuple(text: str) -> tuple[int, ...]:
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    if not vals:
        raise ValueError("empty int tuple")
    return tuple(vals)


def parse_replace_offsets(text: str) -> tuple[float | None, ...]:
    vals: list[float | None] = []
    for part in str(text).split(","):
        part = part.strip().lower()
        if not part:
            continue
        vals.append(None if part in {NO_REPLACE, "none"} else float(part))
    if not vals:
        raise ValueError("empty replace offset tuple")
    return tuple(vals)


def replace_label(offset: float | None) -> str:
    return "CANCEL" if offset is None else f"O{float(offset):g}"


def config_id(*, initial_offset_bps: float, replace_offset_bps: float | None, wait_sec: int, cross_margin_bps: float) -> str:
    return f"O{float(initial_offset_bps):g}_W{int(wait_sec)}_{replace_label(replace_offset_bps)}_C{float(cross_margin_bps):g}"


def find_fill_between(
    event: Any,
    *,
    limit_px: float,
    cross_margin_bps: float,
    start_ts_ms: int,
    end_ts_ms: int | None,
) -> tuple[int, float] | None:
    if event.fade_direction == "LONG":
        required = float(limit_px) * (1.0 - float(cross_margin_bps) / 10_000.0)
        for ts_ms, px in event.path:
            if int(ts_ms) <= int(start_ts_ms):
                continue
            if end_ts_ms is not None and int(ts_ms) > int(end_ts_ms):
                break
            if float(px) <= required:
                return int(ts_ms), float(limit_px)
    else:
        required = float(limit_px) * (1.0 + float(cross_margin_bps) / 10_000.0)
        for ts_ms, px in event.path:
            if int(ts_ms) <= int(start_ts_ms):
                continue
            if end_ts_ms is not None and int(ts_ms) > int(end_ts_ms):
                break
            if float(px) >= required:
                return int(ts_ms), float(limit_px)
    return None


def fill_exit(
    conn: sqlite3.Connection,
    event: Any,
    *,
    fill_ts_ms: int,
    entry_px: float,
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
) -> dict[str, Any]:
    exit_ts_ms = int(fill_ts_ms) + HORIZON_SEC * 1000
    exit_book = book_at(conn, event.symbol, exit_ts_ms, int(max_book_staleness_sec))
    if not exit_book:
        return {
            "status": "NO_EXIT_BOOK",
            "maker_fill_ts_ms": int(fill_ts_ms),
            "maker_fill_utc": iso_ms(fill_ts_ms),
            "entry_price": float(entry_px),
            "net_bps": None,
        }
    exit_px = float(exit_book.bid if event.fade_direction == "LONG" else exit_book.ask)
    gross = signed_return_bps(event.fade_direction, float(entry_px), exit_px)
    net = gross - float(maker_fee_bps) - float(taker_fee_bps)
    return {
        "status": "FILLED",
        "maker_fill_ts_ms": int(fill_ts_ms),
        "maker_fill_utc": iso_ms(fill_ts_ms),
        "fill_delay_sec": (int(fill_ts_ms) - int(event.anchor_mark_ts_ms)) / 1000.0,
        "entry_price": float(entry_px),
        "exit_ts_ms": int(exit_ts_ms),
        "exit_utc": iso_ms(exit_ts_ms),
        "exit_reason": "TIME",
        "exit_book_ts_ms": int(exit_book.ts_ms),
        "exit_staleness_ms": int(exit_book.staleness_ms),
        "exit_price": exit_px,
        "gross_bps": r1(gross),
        "fee_bps": r1(float(maker_fee_bps) + float(taker_fee_bps)),
        "net_bps": float(net),
    }


def simulate_cancel_replace(
    conn: sqlite3.Connection,
    event: Any,
    *,
    initial_offset_bps: float,
    replace_offset_bps: float | None,
    wait_sec: int,
    cross_margin_bps: float,
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
) -> dict[str, Any]:
    anchor_ts = int(event.anchor_mark_ts_ms)
    cancel_ts = anchor_ts + int(wait_sec) * 1000
    initial_limit = maker_limit_price(event.anchor_mark_price, event.fade_direction, initial_offset_bps)
    initial_fill = find_fill_between(
        event,
        limit_px=initial_limit,
        cross_margin_bps=float(cross_margin_bps),
        start_ts_ms=anchor_ts,
        end_ts_ms=cancel_ts,
    )
    base: dict[str, Any] = {
        "symbol": event.symbol,
        "side": event.side,
        "fade_direction": event.fade_direction,
        "bucket": int(event.anchor.bucket),
        "anchor_ts_ms": int(event.anchor.anchor_ts_ms),
        "anchor_utc": iso_ms(event.anchor.anchor_ts_ms),
        "anchor_mark_ts_ms": anchor_ts,
        "anchor_mark_price": float(event.anchor_mark_price),
        "vdepth_bps": float(event.vdepth_bps),
        "elapsed_since_first_sec": float(event.anchor.elapsed_since_first_sec),
        "running_notional": float(event.anchor.running_notional),
        "running_liq_count": int(event.anchor.running_liq_count),
        "running_accel_usd_per_sec": float(event.anchor.running_accel),
        "running_single_liq_dominance": float(event.anchor.running_single_liq_dominance),
        "initial_offset_bps": float(initial_offset_bps),
        "replace_offset_bps": replace_offset_bps,
        "wait_sec": int(wait_sec),
        "cross_margin_bps": float(cross_margin_bps),
        "initial_limit_price": float(initial_limit),
        "replace_limit_price": None,
        "fill_leg": None,
    }
    if initial_fill is not None:
        fill_ts, entry_px = initial_fill
        base["fill_leg"] = "initial"
        base.update(fill_exit(
            conn,
            event,
            fill_ts_ms=fill_ts,
            entry_px=entry_px,
            maker_fee_bps=float(maker_fee_bps),
            taker_fee_bps=float(taker_fee_bps),
            max_book_staleness_sec=int(max_book_staleness_sec),
        ))
        return base

    base["initial_cancel_ts_ms"] = int(cancel_ts)
    base["initial_cancel_utc"] = iso_ms(cancel_ts)
    if replace_offset_bps is None:
        base.update({"status": "NO_MAKER_FILL", "net_bps": None})
        return base

    replace_limit = maker_limit_price(event.anchor_mark_price, event.fade_direction, float(replace_offset_bps))
    base["replace_limit_price"] = float(replace_limit)
    replacement_fill = find_fill_between(
        event,
        limit_px=replace_limit,
        cross_margin_bps=float(cross_margin_bps),
        start_ts_ms=cancel_ts,
        end_ts_ms=None,
    )
    if replacement_fill is None:
        base.update({"status": "NO_MAKER_FILL", "net_bps": None})
        return base
    fill_ts, entry_px = replacement_fill
    base["fill_leg"] = "replacement"
    base.update(fill_exit(
        conn,
        event,
        fill_ts_ms=fill_ts,
        entry_px=entry_px,
        maker_fee_bps=float(maker_fee_bps),
        taker_fee_bps=float(taker_fee_bps),
        max_book_staleness_sec=int(max_book_staleness_sec),
    ))
    return base


def run_cancel_replace(
    conn: sqlite3.Connection,
    *,
    initial_offset_bps: float,
    replace_offsets_bps: tuple[float | None, ...],
    wait_seconds: tuple[int, ...],
    cross_margins_bps: tuple[float, ...],
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
) -> dict[str, Any]:
    events = collect_v01_events(conn)
    eth_marks = load_mark_index(conn, SYMBOL)
    fee_bps = float(maker_fee_bps) + float(taker_fee_bps)
    results = []
    all_rows = []
    for cross in cross_margins_bps:
        for wait_sec in wait_seconds:
            for replace_offset in replace_offsets_bps:
                rows = []
                no_fill_cf = []
                for event in events:
                    row = simulate_cancel_replace(
                        conn,
                        event,
                        initial_offset_bps=float(initial_offset_bps),
                        replace_offset_bps=replace_offset,
                        wait_sec=int(wait_sec),
                        cross_margin_bps=float(cross),
                        maker_fee_bps=float(maker_fee_bps),
                        taker_fee_bps=float(taker_fee_bps),
                        max_book_staleness_sec=int(max_book_staleness_sec),
                    )
                    row["config_id"] = config_id(
                        initial_offset_bps=float(initial_offset_bps),
                        replace_offset_bps=replace_offset,
                        wait_sec=int(wait_sec),
                        cross_margin_bps=float(cross),
                    )
                    row["prior_4h_bps"] = r1(prior_return_bps(eth_marks, int(event.anchor.anchor_ts_ms), 4 * 3600))
                    cf = anchor_mark_counterfactual(eth_marks, int(event.anchor.anchor_ts_ms), fee_bps=fee_bps)
                    row["anchor_cf_net_bps"] = r1(cf)
                    if row.get("status") == "NO_MAKER_FILL" and cf is not None and math.isfinite(float(cf)):
                        no_fill_cf.append(float(cf))
                    rows.append(row)
                    all_rows.append(row)
                filled = [r for r in rows if r.get("status") == "FILLED" and r.get("net_bps") is not None]
                replacement_filled = [r for r in filled if r.get("fill_leg") == "replacement"]
                initial_filled = [r for r in filled if r.get("fill_leg") == "initial"]
                nets = [float(r["net_bps"]) for r in filled]
                results.append(
                    {
                        "config_id": rows[0]["config_id"] if rows else "",
                        "initial_offset_bps": float(initial_offset_bps),
                        "replace_offset_bps": replace_offset,
                        "wait_sec": int(wait_sec),
                        "cross_margin_bps": float(cross),
                        "eligible_n": len(rows),
                        "filled_n": len(filled),
                        "initial_filled_n": len(initial_filled),
                        "replacement_filled_n": len(replacement_filled),
                        "no_fill_n": len(rows) - len(filled),
                        "fill_rate": r3(len(filled) / len(rows)) if rows else None,
                        "replacement_fill_rate": r3(len(replacement_filled) / len(rows)) if rows else None,
                        "filled_summary": summarize(nets),
                        "replacement_summary": summarize([float(r["net_bps"]) for r in replacement_filled]),
                        "no_fill_anchor_cf_summary": summarize(no_fill_cf),
                        "missed_cf_sum_bps": r1(sum(no_fill_cf)),
                        "rows": rows,
                    }
                )
    results.sort(
        key=lambda r: (
            float(r["filled_summary"].get("top3_winner_removed_sum_bps") or -1e18),
            float(r["filled_summary"].get("sum_bps") or -1e18),
            float(r["fill_rate"] or 0.0),
        ),
        reverse=True,
    )
    return {"event_n": len(events), "results": results, "rows": all_rows}


def cell(summary: dict[str, Any]) -> str:
    return f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} T3R={summary['top3_winner_removed_sum_bps']}"


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine Cancel/Replace",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Protocol: `{report['protocol_id']}`",
        "",
        "Research-only. Tests whether unfilled O20 maker entries should be cancelled or replaced with a more aggressive limit after a fixed wait.",
        "",
        f"Events: `{report['event_n']}`",
        "",
        "## Ranked Configs",
        "",
        "| Rank | Config | Fill% | Initial | Replacement | Filled | Replacement only | No-fill CF | Missed CF |",
        "| ---: | --- | ---: | ---: | ---: | --- | --- | --- | ---: |",
    ]
    for idx, row in enumerate(report["results"], start=1):
        fill_pct = None if row["fill_rate"] is None else r1(row["fill_rate"] * 100.0)
        lines.append(
            f"| {idx} | `{row['config_id']}` | {fill_pct} | {row['initial_filled_n']} | {row['replacement_filled_n']} | "
            f"{cell(row['filled_summary'])} | {cell(row['replacement_summary'])} | "
            f"{cell(row['no_fill_anchor_cf_summary'])} | {row['missed_cf_sum_bps']} |"
        )
    best = report["results"][0] if report["results"] else None
    controls = [
        r
        for r in report["results"]
        if r["replace_offset_bps"] is None and int(r["wait_sec"]) == int(report["config"]["control_wait_sec"])
    ]
    lines.extend(["", "## Read", ""])
    if best:
        lines.append(f"- Best T3R-ranked cancel/replace config: `{best['config_id']}` -> {cell(best['filled_summary'])}.")
    for control in controls:
        lines.append(f"- Cancel-only control `{control['config_id']}` -> {cell(control['filled_summary'])}.")
    if best:
        lines.append(
            "- A positive result here must beat fixed O20/O20 shadow after skew removal, not just increase fill count."
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test cancel/replace execution for S34 V Engine v0.1.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--initial-offset-bps", type=float, default=20.0)
    p.add_argument("--replace-offsets-bps", default="cancel,15,10,5")
    p.add_argument("--wait-seconds", default="30,60,120,300,600,1200")
    p.add_argument("--cross-margins-bps", default="1,2")
    p.add_argument("--maker-fee-bps", type=float, default=2.0)
    p.add_argument("--taker-fee-bps", type=float, default=3.05)
    p.add_argument("--max-book-staleness-sec", type=int, default=10)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    waits = parse_int_tuple(args.wait_seconds)
    replace_offsets = parse_replace_offsets(args.replace_offsets_bps)
    crosses = parse_float_tuple(args.cross_margins_bps)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        result = run_cancel_replace(
            conn,
            initial_offset_bps=float(args.initial_offset_bps),
            replace_offsets_bps=replace_offsets,
            wait_seconds=waits,
            cross_margins_bps=crosses,
            maker_fee_bps=float(args.maker_fee_bps),
            taker_fee_bps=float(args.taker_fee_bps),
            max_book_staleness_sec=int(args.max_book_staleness_sec),
        )
    report = {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(args.db),
        "protocol_id": PROTOCOL_ID,
        "config": {
            "initial_offset_bps": float(args.initial_offset_bps),
            "replace_offsets_bps": [NO_REPLACE if v is None else v for v in replace_offsets],
            "wait_seconds": list(waits),
            "cross_margins_bps": list(crosses),
            "maker_fee_bps": float(args.maker_fee_bps),
            "taker_fee_bps": float(args.taker_fee_bps),
            "control_wait_sec": int(waits[0]),
        },
        **result,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
