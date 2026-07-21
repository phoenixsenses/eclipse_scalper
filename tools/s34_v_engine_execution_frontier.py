"""S34 V Engine v0.1 maker execution frontier.

Sweeps maker offsets for the frozen V Engine state to measure the frontier
between fill probability, expectancy, skew, and missed no-fill opportunity.
Research-only; no live/paper state is changed.
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
    file_fingerprint,
    load_mark_index,
    r1,
    r3,
    signed_return_bps,
)
from tools.research_s34_maker_fade import (
    NO_TP_OR_SL,
    collect_events,
    simulate_event,
    summarize,
)
from tools.s34_v_engine_shadow_observer import (
    ACCEL_WINDOW_SEC,
    BUCKET_SEC,
    FADE_DIRECTION,
    HORIZON_SEC,
    LIQ_SIDE,
    MIN_GAP_SEC,
    PRIOR4H_LT_BPS,
    PROTOCOL_ID,
    SYMBOL,
    THRESHOLD_USD,
    VDEPTH_MAX_BPS,
    VDEPTH_MIN_BPS,
    utc_now,
)


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V_ENGINE_EXECUTION_FRONTIER.json"
OUT_MD = OUT_DIR / "S34_V_ENGINE_EXECUTION_FRONTIER.md"


def parse_float_tuple(text: str) -> tuple[float, ...]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise ValueError("empty float tuple")
    return tuple(out)


def prior_return_bps(marks: Any, ts_ms: int, window_sec: int) -> float | None:
    return marks.ret_bps(int(ts_ms) - int(window_sec) * 1000, int(ts_ms))


def anchor_mark_counterfactual(marks: Any, anchor_ts_ms: int, *, fee_bps: float) -> float | None:
    entry = marks.at_or_after(int(anchor_ts_ms))
    exit_ = marks.at_or_after(int(anchor_ts_ms) + HORIZON_SEC * 1000)
    if not entry or not exit_:
        return None
    return signed_return_bps(FADE_DIRECTION, float(entry[1]), float(exit_[1])) - float(fee_bps)


def collect_v01_events(conn: sqlite3.Connection) -> list[Any]:
    marks = load_mark_index(conn, SYMBOL)
    events = collect_events(
        conn,
        symbol=SYMBOL,
        threshold=THRESHOLD_USD,
        sides=(LIQ_SIDE,),
        min_vdepth_bps=VDEPTH_MIN_BPS,
        bucket_sec=BUCKET_SEC,
        min_gap_sec=MIN_GAP_SEC,
        accel_window_sec=ACCEL_WINDOW_SEC,
        max_horizon_sec=HORIZON_SEC,
    )
    out = []
    for event in events:
        if not (VDEPTH_MIN_BPS <= float(event.vdepth_bps) < VDEPTH_MAX_BPS):
            continue
        prior4h = prior_return_bps(marks, int(event.anchor.anchor_ts_ms), 4 * 3600)
        if prior4h is None or not math.isfinite(float(prior4h)) or not (float(prior4h) < PRIOR4H_LT_BPS):
            continue
        out.append(event)
    return out


def run_frontier(
    conn: sqlite3.Connection,
    *,
    offsets: tuple[float, ...],
    cross_margins: tuple[float, ...],
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
) -> dict[str, Any]:
    marks = load_mark_index(conn, SYMBOL)
    events = collect_v01_events(conn)
    fee_bps = float(maker_fee_bps) + float(taker_fee_bps)
    rows = []
    ranked = []
    for cross in cross_margins:
        for offset in offsets:
            cfg_rows = []
            no_fill_cf = []
            for event in events:
                row = simulate_event(
                    conn,
                    event,
                    offset_bps=float(offset),
                    cross_margin_bps=float(cross),
                    horizon_sec=HORIZON_SEC,
                    maker_fee_bps=float(maker_fee_bps),
                    taker_fee_bps=float(taker_fee_bps),
                    max_book_staleness_sec=int(max_book_staleness_sec),
                    horizon_from="fill",
                    tp_bps=NO_TP_OR_SL,
                    sl_bps=NO_TP_OR_SL,
                )
                row["config_id"] = f"O{float(offset):g}_C{float(cross):g}"
                row["anchor_cf_net_bps"] = r1(anchor_mark_counterfactual(marks, int(event.anchor.anchor_ts_ms), fee_bps=fee_bps))
                row["prior_4h_bps"] = r1(prior_return_bps(marks, int(event.anchor.anchor_ts_ms), 4 * 3600))
                if row.get("status") == "NO_MAKER_FILL" and row.get("anchor_cf_net_bps") is not None:
                    no_fill_cf.append(float(row["anchor_cf_net_bps"]))
                cfg_rows.append(row)
                rows.append(row)
            filled = [r for r in cfg_rows if r.get("status") == "FILLED" and r.get("net_bps") is not None]
            nets = [float(r["net_bps"]) for r in filled]
            ranked.append(
                {
                    "config_id": f"O{float(offset):g}_C{float(cross):g}",
                    "offset_bps": float(offset),
                    "cross_margin_bps": float(cross),
                    "eligible_n": len(cfg_rows),
                    "filled_n": len(filled),
                    "no_fill_n": len(cfg_rows) - len(filled),
                    "fill_rate": r3(len(filled) / len(cfg_rows)) if cfg_rows else None,
                    "filled_summary": summarize(nets),
                    "no_fill_anchor_cf_summary": summarize(no_fill_cf),
                    "total_with_no_fill_as_zero_bps": r1(sum(nets)),
                    "missed_cf_sum_bps": r1(sum(no_fill_cf)),
                }
            )
    ranked.sort(
        key=lambda r: (
            float(r["filled_summary"].get("top3_winner_removed_sum_bps") or -1e18),
            float(r["filled_summary"].get("sum_bps") or -1e18),
            float(r["fill_rate"] or 0.0),
        ),
        reverse=True,
    )
    return {
        "event_n": len(events),
        "ranked": ranked,
        "rows": rows,
    }


def cell(summary: dict[str, Any]) -> str:
    return f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} T3R={summary['top3_winner_removed_sum_bps']}"


def render_md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    lines = [
        "# S34 V Engine Execution Frontier",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Protocol: `{report['protocol_id']}`",
        "",
        "Research-only maker offset frontier for the frozen v0.1 state.",
        "",
        f"Events: `{report['event_n']}`. Offsets: `{cfg['offsets_bps']}`. Cross margins: `{cfg['cross_margins_bps']}`.",
        "",
        "## Frontier",
        "",
        "| Rank | Config | Fill% | Filled | No-fill CF | Missed CF sum |",
        "| ---: | --- | ---: | --- | --- | ---: |",
    ]
    for idx, row in enumerate(report["ranked"], start=1):
        fill_pct = None if row["fill_rate"] is None else r1(row["fill_rate"] * 100.0)
        lines.append(
            f"| {idx} | `{row['config_id']}` | {fill_pct} | {cell(row['filled_summary'])} | "
            f"{cell(row['no_fill_anchor_cf_summary'])} | {row['missed_cf_sum_bps']} |"
        )
    lines.extend(["", "## Read", ""])
    best = report["ranked"][0] if report["ranked"] else None
    if best:
        lines.append(
            f"- Best T3R-ranked config: `{best['config_id']}` fill `{None if best['fill_rate'] is None else r1(best['fill_rate'] * 100.0)}%`, filled {cell(best['filled_summary'])}."
        )
        lines.append(
            f"- Missed no-fill anchor counterfactual for that config: {cell(best['no_fill_anchor_cf_summary'])}."
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep maker offsets for S34 V Engine v0.1.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--offsets-bps", default="0,5,10,15,20,25,30,40")
    p.add_argument("--cross-margins-bps", default="1,2,5")
    p.add_argument("--maker-fee-bps", type=float, default=2.0)
    p.add_argument("--taker-fee-bps", type=float, default=3.05)
    p.add_argument("--max-book-staleness-sec", type=int, default=10)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    offsets = parse_float_tuple(args.offsets_bps)
    crosses = parse_float_tuple(args.cross_margins_bps)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        result = run_frontier(
            conn,
            offsets=offsets,
            cross_margins=crosses,
            maker_fee_bps=float(args.maker_fee_bps),
            taker_fee_bps=float(args.taker_fee_bps),
            max_book_staleness_sec=int(args.max_book_staleness_sec),
        )
    report = {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(args.db),
        "protocol_id": PROTOCOL_ID,
        "config": {
            "symbol": SYMBOL,
            "threshold_usd": THRESHOLD_USD,
            "vdepth_min_bps": VDEPTH_MIN_BPS,
            "vdepth_max_bps": VDEPTH_MAX_BPS,
            "prior4h_lt_bps": PRIOR4H_LT_BPS,
            "offsets_bps": list(offsets),
            "cross_margins_bps": list(crosses),
            "maker_fee_bps": float(args.maker_fee_bps),
            "taker_fee_bps": float(args.taker_fee_bps),
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
