"""S34 V Engine protective-stop research.

Evaluates stop/exit overlays on the current live V Engine lifecycle:

    O20 initial maker limit
    wait 300s
    replace to O5
    cross margin 1 bps
    baseline exit at 2h from fill

Research-only; writes reports and does not alter live state.
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
from tools.research_s34_maker_fade import summarize
from tools.s34_v_engine_cancel_replace import simulate_cancel_replace
from tools.s34_v_engine_execution_frontier import collect_v01_events, parse_float_tuple
from tools.s34_v_engine_shadow_observer import HORIZON_SEC, PROTOCOL_ID, SYMBOL, utc_now


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V_ENGINE_PROTECTIVE_STOP.json"
OUT_MD = OUT_DIR / "S34_V_ENGINE_PROTECTIVE_STOP.md"

INITIAL_OFFSET_BPS = 20.0
REPLACE_OFFSET_BPS = 5.0
WAIT_SEC = 300
CROSS_MARGIN_BPS = 1.0
MAKER_FEE_BPS = 2.0
TAKER_FEE_BPS = 3.05


def first_stop_ts(event: Any, *, fill_ts_ms: int, entry_px: float, sl_bps: float, deadline_ms: int) -> int | None:
    for ts_ms, px in event.path:
        ts = int(ts_ms)
        if ts <= int(fill_ts_ms):
            continue
        if ts > int(deadline_ms):
            break
        ret = signed_return_bps(event.fade_direction, float(entry_px), float(px))
        if ret <= -float(sl_bps):
            return ts
    return None


def mark_ret_at(marks: Any, *, direction: str, entry_px: float, ts_ms: int) -> float | None:
    row = marks.at_or_after(int(ts_ms))
    if not row:
        return None
    return signed_return_bps(direction, float(entry_px), float(row[1]))


def anchor_reclaimed(event: Any, *, fill_ts_ms: int, horizon_min: int) -> bool:
    end = int(fill_ts_ms) + int(horizon_min) * 60_000
    for ts_ms, px in event.path:
        if int(ts_ms) <= int(fill_ts_ms):
            continue
        if int(ts_ms) > end:
            break
        if float(px) >= float(event.anchor_mark_price):
            return True
    return False


def exit_net_at_book(
    conn: sqlite3.Connection,
    event: Any,
    *,
    entry_px: float,
    exit_ts_ms: int,
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
) -> tuple[float | None, str, float | None]:
    quote = book_at(conn, event.symbol, int(exit_ts_ms), int(max_book_staleness_sec))
    if not quote:
        return None, "no_book", None
    exit_px = float(quote.bid if event.fade_direction == "LONG" else quote.ask)
    gross = signed_return_bps(event.fade_direction, float(entry_px), exit_px)
    net = gross - float(maker_fee_bps) - float(taker_fee_bps)
    return float(net), "book_ticker", exit_px


def apply_fixed_sl(
    conn: sqlite3.Connection,
    event: Any,
    base: dict[str, Any],
    *,
    sl_bps: float,
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
) -> dict[str, Any]:
    fill_ts = int(base["maker_fill_ts_ms"])
    entry_px = float(base["entry_price"])
    deadline = fill_ts + HORIZON_SEC * 1000
    stop_ts = first_stop_ts(event, fill_ts_ms=fill_ts, entry_px=entry_px, sl_bps=float(sl_bps), deadline_ms=deadline)
    if stop_ts is None:
        return {
            "variant": f"fixed_sl_{float(sl_bps):g}",
            "net_bps": r1(base.get("net_bps")),
            "exit_reason": "TIME",
            "exit_ts_ms": base.get("exit_ts_ms"),
            "source": "baseline_time",
        }
    net, source, exit_px = exit_net_at_book(
        conn,
        event,
        entry_px=entry_px,
        exit_ts_ms=stop_ts,
        maker_fee_bps=maker_fee_bps,
        taker_fee_bps=taker_fee_bps,
        max_book_staleness_sec=max_book_staleness_sec,
    )
    return {
        "variant": f"fixed_sl_{float(sl_bps):g}",
        "net_bps": r1(net),
        "exit_reason": "SL",
        "exit_ts_ms": stop_ts,
        "exit_utc": iso_ms(stop_ts),
        "exit_price": exit_px,
        "source": source,
    }


def apply_danger_exit(
    conn: sqlite3.Connection,
    event: Any,
    base: dict[str, Any],
    marks: Any,
    *,
    minute: int,
    ret_lte_bps: float | None,
    require_no_anchor_reclaim: bool,
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
) -> dict[str, Any]:
    fill_ts = int(base["maker_fill_ts_ms"])
    entry_px = float(base["entry_price"])
    trigger_ts = fill_ts + int(minute) * 60_000
    ret = mark_ret_at(marks, direction=event.fade_direction, entry_px=entry_px, ts_ms=trigger_ts)
    no_reclaim = not anchor_reclaimed(event, fill_ts_ms=fill_ts, horizon_min=int(minute))
    ret_condition = True if ret_lte_bps is None else (ret is not None and float(ret) <= float(ret_lte_bps))
    reclaim_condition = no_reclaim if require_no_anchor_reclaim else True
    triggered = bool(ret_condition and reclaim_condition)
    label = f"danger_{minute}m"
    if ret_lte_bps is not None:
        label += f"_retlte{float(ret_lte_bps):g}"
    if require_no_anchor_reclaim:
        label += "_noreclaim"
    if not triggered:
        return {
            "variant": label,
            "net_bps": r1(base.get("net_bps")),
            "exit_reason": "TIME",
            "exit_ts_ms": base.get("exit_ts_ms"),
            "source": "baseline_time",
            "triggered": False,
            "trigger_ret_bps": r1(ret),
            "no_anchor_reclaim": no_reclaim,
        }
    net, source, exit_px = exit_net_at_book(
        conn,
        event,
        entry_px=entry_px,
        exit_ts_ms=trigger_ts,
        maker_fee_bps=maker_fee_bps,
        taker_fee_bps=taker_fee_bps,
        max_book_staleness_sec=max_book_staleness_sec,
    )
    return {
        "variant": label,
        "net_bps": r1(net),
        "exit_reason": "DANGER",
        "exit_ts_ms": trigger_ts,
        "exit_utc": iso_ms(trigger_ts),
        "exit_price": exit_px,
        "source": source,
        "triggered": True,
        "trigger_ret_bps": r1(ret),
        "no_anchor_reclaim": no_reclaim,
    }


def apply_hybrid(
    conn: sqlite3.Connection,
    event: Any,
    base: dict[str, Any],
    marks: Any,
    *,
    sl_bps: float,
    minute: int,
    ret_lte_bps: float,
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
) -> dict[str, Any]:
    fixed = apply_fixed_sl(
        conn,
        event,
        base,
        sl_bps=sl_bps,
        maker_fee_bps=maker_fee_bps,
        taker_fee_bps=taker_fee_bps,
        max_book_staleness_sec=max_book_staleness_sec,
    )
    danger = apply_danger_exit(
        conn,
        event,
        base,
        marks,
        minute=minute,
        ret_lte_bps=ret_lte_bps,
        require_no_anchor_reclaim=True,
        maker_fee_bps=maker_fee_bps,
        taker_fee_bps=taker_fee_bps,
        max_book_staleness_sec=max_book_staleness_sec,
    )
    label = f"hybrid_sl{float(sl_bps):g}_danger{minute}m_ret{float(ret_lte_bps):g}_noreclaim"
    candidates = [fixed, danger]
    candidates = [r for r in candidates if r.get("source") != "baseline_time"]
    if not candidates:
        return {
            "variant": label,
            "net_bps": r1(base.get("net_bps")),
            "exit_reason": "TIME",
            "exit_ts_ms": base.get("exit_ts_ms"),
            "source": "baseline_time",
        }
    chosen = min(candidates, key=lambda r: int(r.get("exit_ts_ms") or 10**18))
    return {**chosen, "variant": label, "exit_reason": f"HYBRID_{chosen.get('exit_reason')}"}


def run_stop_research(
    conn: sqlite3.Connection,
    *,
    fixed_sls: tuple[float, ...],
    danger_rets: tuple[float, ...],
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
) -> dict[str, Any]:
    events = collect_v01_events(conn)
    marks = load_mark_index(conn, SYMBOL)
    base_rows = []
    variant_rows = []
    for event in events:
        base = simulate_cancel_replace(
            conn,
            event,
            initial_offset_bps=INITIAL_OFFSET_BPS,
            replace_offset_bps=REPLACE_OFFSET_BPS,
            wait_sec=WAIT_SEC,
            cross_margin_bps=CROSS_MARGIN_BPS,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            max_book_staleness_sec=max_book_staleness_sec,
        )
        if base.get("status") != "FILLED" or base.get("net_bps") is None:
            continue
        base_card = {
            "event_id": f"{int(event.anchor.bucket)}:{int(event.anchor.anchor_ts_ms)}",
            "signal_utc": base.get("anchor_utc"),
            "fill_leg": base.get("fill_leg"),
            "fill_delay_sec": r1(base.get("fill_delay_sec")),
            "entry_price": base.get("entry_price"),
            "baseline_net_bps": r1(base.get("net_bps")),
            "vdepth_bps": r1(event.vdepth_bps),
            "prior_4h_bps": r1(marks.ret_bps(int(event.anchor.anchor_ts_ms) - 4 * 3600 * 1000, int(event.anchor.anchor_ts_ms))),
        }
        base_rows.append(base_card)
        variants = []
        for sl in fixed_sls:
            variants.append(apply_fixed_sl(
                conn,
                event,
                base,
                sl_bps=float(sl),
                maker_fee_bps=maker_fee_bps,
                taker_fee_bps=taker_fee_bps,
                max_book_staleness_sec=max_book_staleness_sec,
            ))
        for ret in danger_rets:
            variants.append(apply_danger_exit(
                conn,
                event,
                base,
                marks,
                minute=15,
                ret_lte_bps=float(ret),
                require_no_anchor_reclaim=True,
                maker_fee_bps=maker_fee_bps,
                taker_fee_bps=taker_fee_bps,
                max_book_staleness_sec=max_book_staleness_sec,
            ))
        for sl in (60.0, 80.0, 100.0):
            variants.append(apply_hybrid(
                conn,
                event,
                base,
                marks,
                sl_bps=sl,
                minute=15,
                ret_lte_bps=-25.0,
                maker_fee_bps=maker_fee_bps,
                taker_fee_bps=taker_fee_bps,
                max_book_staleness_sec=max_book_staleness_sec,
            ))
        for variant in variants:
            variant_rows.append({**base_card, **variant})
    summaries = []
    labels = sorted({str(r["variant"]) for r in variant_rows})
    baseline = summarize([float(r["baseline_net_bps"]) for r in base_rows if r.get("baseline_net_bps") is not None])
    for label in labels:
        rows = [r for r in variant_rows if r["variant"] == label]
        vals = [float(r["net_bps"]) for r in rows if r.get("net_bps") is not None and math.isfinite(float(r["net_bps"]))]
        stop_rows = [r for r in rows if str(r.get("exit_reason") or "") != "TIME"]
        summaries.append(
            {
                "variant": label,
                "n": len(rows),
                "exit_count": len(stop_rows),
                "exit_rate": r3(len(stop_rows) / len(rows)) if rows else None,
                "summary": summarize(vals),
                "delta_sum_vs_baseline": r1(sum(vals) - float(baseline.get("sum_bps") or 0.0)),
                "delta_t3r_vs_baseline": r1(float(summarize(vals).get("top3_winner_removed_sum_bps") or 0.0) - float(baseline.get("top3_winner_removed_sum_bps") or 0.0)),
                "worst_cards": sorted(rows, key=lambda r: float(r.get("net_bps") or 0.0))[:6],
            }
        )
    summaries.sort(
        key=lambda r: (
            float(r["summary"].get("top3_winner_removed_sum_bps") or -1e18),
            float(r["summary"].get("sum_bps") or -1e18),
        ),
        reverse=True,
    )
    return {
        "event_n": len(events),
        "filled_n": len(base_rows),
        "baseline": baseline,
        "summaries": summaries,
        "rows": variant_rows,
    }


def cell(summary: dict[str, Any]) -> str:
    return f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} T3R={summary['top3_winner_removed_sum_bps']} max_loss={summary['max_loss_bps']}"


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine Protective Stop",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Protocol: `{report['protocol_id']}`",
        "",
        "Research-only stop overlays on the current live lifecycle `O20_W300_O5_C1`.",
        "",
        "## Baseline",
        "",
        f"- eligible events: `{report['event_n']}`",
        f"- filled rows: `{report['filled_n']}`",
        f"- baseline: {cell(report['baseline'])}",
        "",
        "## Stop Variants",
        "",
        "| Rank | Variant | Exit N | Exit% | Summary | Delta sum | Delta T3R |",
        "| ---: | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for idx, row in enumerate(report["summaries"], start=1):
        exit_pct = None if row["exit_rate"] is None else r1(row["exit_rate"] * 100.0)
        lines.append(
            f"| {idx} | `{row['variant']}` | {row['exit_count']} | {exit_pct} | {cell(row['summary'])} | "
            f"{row['delta_sum_vs_baseline']} | {row['delta_t3r_vs_baseline']} |"
        )
    lines.extend(["", "## Read", ""])
    best = report["summaries"][0] if report["summaries"] else None
    if best:
        lines.append(f"- Best T3R-ranked stop overlay: `{best['variant']}` -> {cell(best['summary'])}.")
        lines.append("- For live safety, prefer an exchange-native hard SL even if it is not the top research overlay; process-only danger exits do not protect against outages.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate protective stops for S34 V Engine live lifecycle.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--fixed-sls-bps", default="40,60,80,100,120,150")
    p.add_argument("--danger-ret-lte-bps", default="-10,-25,-50")
    p.add_argument("--maker-fee-bps", type=float, default=MAKER_FEE_BPS)
    p.add_argument("--taker-fee-bps", type=float, default=TAKER_FEE_BPS)
    p.add_argument("--max-book-staleness-sec", type=int, default=10)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        result = run_stop_research(
            conn,
            fixed_sls=parse_float_tuple(args.fixed_sls_bps),
            danger_rets=parse_float_tuple(args.danger_ret_lte_bps),
            maker_fee_bps=float(args.maker_fee_bps),
            taker_fee_bps=float(args.taker_fee_bps),
            max_book_staleness_sec=int(args.max_book_staleness_sec),
        )
    report = {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(args.db),
        "protocol_id": PROTOCOL_ID,
        "live_lifecycle": {
            "initial_offset_bps": INITIAL_OFFSET_BPS,
            "replace_offset_bps": REPLACE_OFFSET_BPS,
            "wait_sec": WAIT_SEC,
            "cross_margin_bps": CROSS_MARGIN_BPS,
            "horizon_sec": HORIZON_SEC,
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
