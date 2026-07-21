"""S34 Maker Fade Research.

Tests whether the real S34 phenomenon (deep V-shape liquidation-cascade
reversal) becomes harvestable when entry is passive instead of taker.

Entry is knowable: after a running-notional threshold cross, place a maker limit
at/beyond the cascade extreme. Because book_ticker has top-of-book only and no
queue position, fills are conservative: a touch is not enough; the future mark
path must cross beyond the limit by `cross_margin_bps`.

This is RESEARCH_ONLY. It writes reports only and does not touch live rules.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    AnchorSnapshot,
    MarkIndex,
    book_at,
    file_fingerprint,
    iso_ms,
    load_liquidations,
    load_mark_index,
    mean,
    pctile,
    r1,
    r3,
    reconstruct_anchors,
    sha256_text,
    signed_return_bps,
)


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_MAKER_FADE.json"
OUT_MD = OUT_DIR / "S34_MAKER_FADE.md"

DEFAULT_OFFSETS = (0.0, 5.0, 10.0, 20.0, 40.0)
DEFAULT_CROSS_MARGINS = (1.0, 2.0, 5.0)
NO_TP_OR_SL = 100_000.0


@dataclass(frozen=True)
class FadeEvent:
    symbol: str
    side: str
    fade_direction: str
    anchor: AnchorSnapshot
    anchor_mark_ts_ms: int
    anchor_mark_price: float
    vdepth_bps: float
    path: tuple[tuple[int, float], ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_float_tuple(text: str) -> tuple[float, ...]:
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    if not vals:
        raise ValueError("empty float tuple")
    return tuple(vals)


def summarize(vals: list[float]) -> dict[str, Any]:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if not xs:
        return {
            "n": 0,
            "sum_bps": 0.0,
            "mean_bps": None,
            "median_bps": None,
            "p10_bps": None,
            "p90_bps": None,
            "win_rate": None,
            "profit_factor": None,
            "max_win_bps": None,
            "max_loss_bps": None,
            "top3_winner_removed_sum_bps": 0.0,
            "bottom3_loser_removed_sum_bps": 0.0,
        }
    wins = [v for v in xs if v > 0.0]
    losses = [-v for v in xs if v < 0.0]
    return {
        "n": len(xs),
        "sum_bps": r1(sum(xs)),
        "mean_bps": r1(mean(xs)),
        "median_bps": r1(pctile(xs, 0.5)),
        "p10_bps": r1(pctile(xs, 0.1)),
        "p90_bps": r1(pctile(xs, 0.9)),
        "win_rate": r3(len(wins) / len(xs)),
        "profit_factor": r3(sum(wins) / sum(losses)) if losses and sum(losses) > 0 else None,
        "max_win_bps": r1(max(xs)),
        "max_loss_bps": r1(min(xs)),
        "top3_winner_removed_sum_bps": r1(sum(sorted(xs, reverse=True)[3:]) if len(xs) > 3 else sum(xs)),
        "bottom3_loser_removed_sum_bps": r1(sum(sorted(xs)[3:]) if len(xs) > 3 else sum(xs)),
    }


def parse_optional_bps_tuple(text: str, *, none_value: float = NO_TP_OR_SL) -> tuple[float, ...]:
    vals = []
    for part in str(text).split(","):
        part = part.strip().lower()
        if not part:
            continue
        vals.append(float(none_value) if part == "none" else float(part))
    if not vals:
        raise ValueError("empty bps tuple")
    return tuple(vals)


def anchor_vdepth_bps(marks: MarkIndex, anchor: AnchorSnapshot, side: str) -> float | None:
    start = marks.at_or_after(int(anchor.first_ts_ms))
    anc = marks.at_or_after(int(anchor.anchor_ts_ms))
    if not start or not anc or float(start[1]) <= 0.0:
        return None
    if side.upper() == "SELL":
        return (float(start[1]) - float(anc[1])) / float(start[1]) * 10_000.0
    return (float(anc[1]) - float(start[1])) / float(start[1]) * 10_000.0


def collect_events(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    threshold: float,
    sides: tuple[str, ...],
    min_vdepth_bps: float,
    bucket_sec: int,
    min_gap_sec: int,
    accel_window_sec: int,
    max_horizon_sec: int,
) -> list[FadeEvent]:
    marks = load_mark_index(conn, symbol)
    events: list[FadeEvent] = []
    side_to_fade = {"BUY": "SHORT", "SELL": "LONG"}
    for side in sides:
        fade_dir = side_to_fade[side]
        liqs = load_liquidations(conn, symbol, side, None, None)
        anchors = reconstruct_anchors(
            liqs,
            bucket_sec=int(bucket_sec),
            min_gap_sec=int(min_gap_sec),
            thresholds=(float(threshold),),
            accel_window_sec=int(accel_window_sec),
        )
        for anchor in anchors:
            depth = anchor_vdepth_bps(marks, anchor, side)
            if depth is None or float(depth) < float(min_vdepth_bps):
                continue
            mark = marks.at_or_after(int(anchor.anchor_ts_ms))
            if not mark:
                continue
            path = tuple(
                (int(ts), float(px))
                for ts, px in marks.slice_range(int(mark[0]), int(mark[0]) + int(max_horizon_sec) * 1000)
                if int(ts) > int(mark[0])
            )
            if not path:
                continue
            events.append(
                FadeEvent(
                    symbol=symbol,
                    side=side,
                    fade_direction=fade_dir,
                    anchor=anchor,
                    anchor_mark_ts_ms=int(mark[0]),
                    anchor_mark_price=float(mark[1]),
                    vdepth_bps=float(depth),
                    path=path,
                )
            )
    events.sort(key=lambda ev: int(ev.anchor.anchor_ts_ms))
    return events


def maker_limit_price(anchor_price: float, direction: str, offset_bps: float) -> float:
    if direction.upper() == "LONG":
        return float(anchor_price) * (1.0 - float(offset_bps) / 10_000.0)
    return float(anchor_price) * (1.0 + float(offset_bps) / 10_000.0)


def find_maker_fill(event: FadeEvent, limit_px: float, cross_margin_bps: float) -> tuple[int, float] | None:
    """Conservative passive fill: future mark must move through the limit."""
    if event.fade_direction == "LONG":
        required = float(limit_px) * (1.0 - float(cross_margin_bps) / 10_000.0)
        for ts_ms, px in event.path:
            if float(px) <= required:
                return int(ts_ms), float(limit_px)
    else:
        required = float(limit_px) * (1.0 + float(cross_margin_bps) / 10_000.0)
        for ts_ms, px in event.path:
            if float(px) >= required:
                return int(ts_ms), float(limit_px)
    return None


def simulate_event(
    conn: sqlite3.Connection,
    event: FadeEvent,
    *,
    offset_bps: float,
    cross_margin_bps: float,
    horizon_sec: int,
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
    horizon_from: str,
    tp_bps: float,
    sl_bps: float,
) -> dict[str, Any]:
    limit_px = maker_limit_price(event.anchor_mark_price, event.fade_direction, offset_bps)
    fill = find_maker_fill(event, limit_px, cross_margin_bps)
    base: dict[str, Any] = {
        "symbol": event.symbol,
        "side": event.side,
        "fade_direction": event.fade_direction,
        "bucket": int(event.anchor.bucket),
        "anchor_ts_ms": int(event.anchor.anchor_ts_ms),
        "anchor_utc": iso_ms(event.anchor.anchor_ts_ms),
        "anchor_mark_price": float(event.anchor_mark_price),
        "vdepth_bps": float(event.vdepth_bps),
        "elapsed_since_first_sec": float(event.anchor.elapsed_since_first_sec),
        "running_notional": float(event.anchor.running_notional),
        "running_liq_count": int(event.anchor.running_liq_count),
        "running_single_liq_dominance": float(event.anchor.running_single_liq_dominance),
        "offset_bps": float(offset_bps),
        "cross_margin_bps": float(cross_margin_bps),
        "limit_price": float(limit_px),
    }
    if fill is None:
        base.update({"status": "NO_MAKER_FILL", "net_bps": None})
        return base

    fill_ts_ms, entry_px = fill
    exit_deadline_ms = (fill_ts_ms if horizon_from == "fill" else int(event.anchor_mark_ts_ms)) + int(horizon_sec) * 1000
    exit_ts_ms = exit_deadline_ms
    exit_reason = "TIME"
    if float(tp_bps) < NO_TP_OR_SL or float(sl_bps) < NO_TP_OR_SL:
        for ts_ms, px in event.path:
            if int(ts_ms) <= int(fill_ts_ms):
                continue
            if int(ts_ms) > int(exit_deadline_ms):
                break
            ret = signed_return_bps(event.fade_direction, float(entry_px), float(px))
            if float(tp_bps) < NO_TP_OR_SL and ret >= float(tp_bps):
                exit_ts_ms = int(ts_ms)
                exit_reason = "TP"
                break
            if float(sl_bps) < NO_TP_OR_SL and ret <= -float(sl_bps):
                exit_ts_ms = int(ts_ms)
                exit_reason = "SL"
                break
    exit_book = book_at(conn, event.symbol, exit_ts_ms, int(max_book_staleness_sec))
    if not exit_book:
        base.update({"status": "NO_EXIT_BOOK", "maker_fill_ts_ms": fill_ts_ms, "entry_price": entry_px, "net_bps": None})
        return base
    exit_px = float(exit_book.bid if event.fade_direction == "LONG" else exit_book.ask)
    gross = signed_return_bps(event.fade_direction, float(entry_px), float(exit_px))
    net = gross - float(maker_fee_bps) - float(taker_fee_bps)
    base.update(
        {
            "status": "FILLED",
            "maker_fill_ts_ms": int(fill_ts_ms),
            "maker_fill_utc": iso_ms(fill_ts_ms),
            "fill_delay_sec": (int(fill_ts_ms) - int(event.anchor_mark_ts_ms)) / 1000.0,
            "entry_price": float(entry_px),
            "exit_ts_ms": int(exit_ts_ms),
            "exit_utc": iso_ms(exit_ts_ms),
            "exit_reason": exit_reason,
            "exit_book_ts_ms": int(exit_book.ts_ms),
            "exit_staleness_ms": int(exit_book.staleness_ms),
            "exit_price": float(exit_px),
            "gross_bps": r1(gross),
            "fee_bps": r1(float(maker_fee_bps) + float(taker_fee_bps)),
            "net_bps": float(net),
        }
    )
    return base


def split_rows(rows: list[dict[str, Any]], holdout_frac: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ids = sorted({int(r["bucket"]) for r in rows})
    holdout_n = max(1, int(round(len(ids) * float(holdout_frac)))) if ids else 0
    holdout_ids = set(ids[-holdout_n:]) if holdout_n else set()
    for row in rows:
        row["split"] = "holdout" if int(row["bucket"]) in holdout_ids else "calibration"
    cal = [r for r in rows if int(r["bucket"]) not in holdout_ids]
    hold = [r for r in rows if int(r["bucket"]) in holdout_ids]
    split = {
        "method": "chronological_bucket_tail_holdout",
        "holdout_frac": float(holdout_frac),
        "calibration_bucket_n": len(ids) - len(holdout_ids),
        "holdout_bucket_n": len(holdout_ids),
        "holdout_bucket_ids": [str(x) for x in sorted(holdout_ids)],
        "holdout_bucket_ids_sha256": sha256_text("\n".join(str(x) for x in sorted(holdout_ids))),
    }
    return cal, hold, split


def side_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("calibration", "holdout", "all"):
        split_rows = rows if split == "all" else [r for r in rows if r.get("split") == split]
        out[split] = {}
        for side in ("BUY", "SELL"):
            vals = [
                float(r["net_bps"])
                for r in split_rows
                if r.get("side") == side and r.get("status") == "FILLED" and r.get("net_bps") is not None
            ]
            out[split][side] = summarize(vals)
    return out


def run_config(
    conn: sqlite3.Connection,
    events: list[FadeEvent],
    *,
    offset_bps: float,
    cross_margin_bps: float,
    horizon_sec: int,
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
    holdout_frac: float,
    horizon_from: str,
    tp_bps: float,
    sl_bps: float,
) -> dict[str, Any]:
    rows = [
        simulate_event(
            conn,
            ev,
            offset_bps=offset_bps,
            cross_margin_bps=cross_margin_bps,
            horizon_sec=horizon_sec,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            max_book_staleness_sec=max_book_staleness_sec,
            horizon_from=horizon_from,
            tp_bps=float(tp_bps),
            sl_bps=float(sl_bps),
        )
        for ev in events
    ]
    cal, hold, split = split_rows(rows, holdout_frac)
    cal_filled = [r for r in cal if r.get("status") == "FILLED" and r.get("net_bps") is not None]
    hold_filled = [r for r in hold if r.get("status") == "FILLED" and r.get("net_bps") is not None]
    all_filled = [r for r in rows if r.get("status") == "FILLED" and r.get("net_bps") is not None]
    cal_sum = summarize([float(r["net_bps"]) for r in cal_filled])
    hold_sum = summarize([float(r["net_bps"]) for r in hold_filled])
    lead = (
        cal_sum["n"] >= 20
        and hold_sum["n"] >= 10
        and float(cal_sum["sum_bps"] or 0.0) > 0.0
        and float(hold_sum["sum_bps"] or 0.0) > 0.0
        and float(cal_sum["top3_winner_removed_sum_bps"] or 0.0) > 0.0
        and float(hold_sum["top3_winner_removed_sum_bps"] or 0.0) > 0.0
    )
    return {
        "offset_bps": float(offset_bps),
        "cross_margin_bps": float(cross_margin_bps),
        "tp_bps": None if float(tp_bps) >= NO_TP_OR_SL else float(tp_bps),
        "sl_bps": None if float(sl_bps) >= NO_TP_OR_SL else float(sl_bps),
        "total_events": len(rows),
        "filled_n": len(all_filled),
        "fill_rate": r3(len(all_filled) / len(rows)) if rows else None,
        "calibration": {
            "total_n": len(cal),
            "filled_n": len(cal_filled),
            "fill_rate": r3(len(cal_filled) / len(cal)) if cal else None,
            "summary": cal_sum,
        },
        "holdout": {
            "total_n": len(hold),
            "filled_n": len(hold_filled),
            "fill_rate": r3(len(hold_filled) / len(hold)) if hold else None,
            "summary": hold_sum,
        },
        "overall": summarize([float(r["net_bps"]) for r in all_filled]),
        "by_side": side_summaries(rows),
        "split": split,
        "lead": lead,
        "rows": rows,
    }


def render_md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    lines = [
        "# S34 Maker Fade Research",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  {cfg['symbol']} {int(cfg['threshold']/1000)}K "
        f"deep-V >= {cfg['min_vdepth_bps']}bps, horizon {cfg['horizon_hr']}h from {cfg['horizon_from']}",
        "",
        "RESEARCH_ONLY. Passive fade after a knowable threshold cross. Fill is conservative: future mark must cross beyond the limit by the margin; touches do not count.",
        "",
        f"Events: total={report['event_n']} BUY/up-spike={report['event_side_counts'].get('BUY', 0)} SELL/down-spike={report['event_side_counts'].get('SELL', 0)}",
        "",
        "## Grid",
        "",
        "| Offset | Cross | TP | SL | Fill% | Cal N | Cal Sum | Cal Med | Cal PF | Cal T3R | Hold N | Hold Sum | Hold Med | Hold PF | Hold T3R | Max Loss | Lead |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["configs"]:
        cal = row["calibration"]["summary"]
        hold = row["holdout"]["summary"]
        lines.append(
            f"| {row['offset_bps']} | {row['cross_margin_bps']} "
            f"| {'none' if row['tp_bps'] is None else row['tp_bps']} | {'none' if row['sl_bps'] is None else row['sl_bps']} "
            f"| {None if row['fill_rate'] is None else r1(row['fill_rate'] * 100.0)} "
            f"| {cal['n']} | {cal['sum_bps']} | {cal['median_bps']} | {cal['profit_factor']} | {cal['top3_winner_removed_sum_bps']} "
            f"| {hold['n']} | {hold['sum_bps']} | {hold['median_bps']} | {hold['profit_factor']} | {hold['top3_winner_removed_sum_bps']} "
            f"| {hold['max_loss_bps']} | {'**' if row['lead'] else ''} |"
        )
    leads = [r for r in report["configs"] if r["lead"]]
    lines.extend(["", f"## Leads: {len(leads)}", ""])
    if leads:
        for row in leads:
            lines.append(
                f"- offset={row['offset_bps']} cross_margin={row['cross_margin_bps']} "
                f"tp={'none' if row['tp_bps'] is None else row['tp_bps']} sl={'none' if row['sl_bps'] is None else row['sl_bps']} "
                f"cal_sum={row['calibration']['summary']['sum_bps']} hold_sum={row['holdout']['summary']['sum_bps']} "
                f"hold_pf={row['holdout']['summary']['profit_factor']}"
            )
    else:
        lines.append("- none")
    side_leads: list[tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]] = []
    for row in report["configs"]:
        for side in ("BUY", "SELL"):
            cal = row["by_side"]["calibration"][side]
            hold = row["by_side"]["holdout"][side]
            if (
                cal["n"] >= 10
                and hold["n"] >= 10
                and float(cal.get("sum_bps") or 0.0) > 0.0
                and float(hold.get("sum_bps") or 0.0) > 0.0
                and float(cal.get("top3_winner_removed_sum_bps") or 0.0) > 0.0
                and float(hold.get("top3_winner_removed_sum_bps") or 0.0) > 0.0
            ):
                side_leads.append((row, side, cal, hold))
    lines.extend(["", f"## Side-Specific Leads: {len(side_leads)}", ""])
    if side_leads:
        lines.append("| Side | Offset | Cross | TP | SL | Cal N | Cal Sum | Cal Med | Cal T3R | Hold N | Hold Sum | Hold Med | Hold T3R |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row, side, cal, hold in side_leads:
            lines.append(
                f"| {side} | {row['offset_bps']} | {row['cross_margin_bps']} | {'none' if row['tp_bps'] is None else row['tp_bps']} "
                f"| {'none' if row['sl_bps'] is None else row['sl_bps']} | {cal['n']} | {cal['sum_bps']} | "
                f"{cal['median_bps']} | {cal['top3_winner_removed_sum_bps']} | {hold['n']} | {hold['sum_bps']} | "
                f"{hold['median_bps']} | {hold['top3_winner_removed_sum_bps']} |"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Best Holdout Sums: Side Breakdown", ""])
    lines.append("| Offset | Cross | TP | SL | Hold Sum | Cal Sum | Cal BUY/Sell | Hold BUY/Sell |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    ranked = sorted(
        report["configs"],
        key=lambda r: float(r["holdout"]["summary"].get("sum_bps") or -1e18),
        reverse=True,
    )[:5]
    for row in ranked:
        cal = row["by_side"]["calibration"]
        hold = row["by_side"]["holdout"]
        lines.append(
            f"| {row['offset_bps']} | {row['cross_margin_bps']} | {'none' if row['tp_bps'] is None else row['tp_bps']} "
            f"| {'none' if row['sl_bps'] is None else row['sl_bps']} | {row['holdout']['summary']['sum_bps']} | {row['calibration']['summary']['sum_bps']} "
            f"| BUY {cal['BUY']['sum_bps']} / SELL {cal['SELL']['sum_bps']} "
            f"| BUY {hold['BUY']['sum_bps']} / SELL {hold['SELL']['sum_bps']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research passive maker entry for deep-V liquidation fade.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--threshold", type=float, default=200_000.0)
    parser.add_argument("--sides", default="BUY,SELL", help="Comma-separated liquidation sides: BUY,SELL.")
    parser.add_argument("--min-vdepth-bps", type=float, default=28.0)
    parser.add_argument("--horizon-hr", type=float, default=4.0)
    parser.add_argument("--horizon-from", choices=("fill", "anchor"), default="fill")
    parser.add_argument("--bucket-sec", type=int, default=300)
    parser.add_argument("--min-gap-sec", type=int, default=900)
    parser.add_argument("--accel-window-sec", type=int, default=30)
    parser.add_argument("--offset-bps", default=",".join(str(x) for x in DEFAULT_OFFSETS))
    parser.add_argument("--cross-margin-bps", default=",".join(str(x) for x in DEFAULT_CROSS_MARGINS))
    parser.add_argument("--tp-bps", default="none", help="Comma-separated TP bps values, or none.")
    parser.add_argument("--sl-bps", default="none", help="Comma-separated SL bps values, or none.")
    parser.add_argument("--maker-fee-bps", type=float, default=2.0)
    parser.add_argument("--taker-fee-bps", type=float, default=3.05)
    parser.add_argument("--max-book-staleness-sec", type=int, default=10)
    parser.add_argument("--holdout-frac", type=float, default=0.30)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    offsets = parse_float_tuple(args.offset_bps)
    margins = parse_float_tuple(args.cross_margin_bps)
    tps = parse_optional_bps_tuple(args.tp_bps)
    sls = parse_optional_bps_tuple(args.sl_bps)
    sides = tuple(s.strip().upper() for s in str(args.sides).split(",") if s.strip())
    bad = [s for s in sides if s not in {"BUY", "SELL"}]
    if bad:
        raise ValueError(f"unsupported sides: {bad}")
    horizon_sec = int(float(args.horizon_hr) * 3600)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        events = collect_events(
            conn,
            symbol=args.symbol,
            threshold=float(args.threshold),
            sides=sides,
            min_vdepth_bps=float(args.min_vdepth_bps),
            bucket_sec=int(args.bucket_sec),
            min_gap_sec=int(args.min_gap_sec),
            accel_window_sec=int(args.accel_window_sec),
            max_horizon_sec=horizon_sec,
        )
        configs = [
            run_config(
                conn,
                events,
                offset_bps=offset,
                cross_margin_bps=margin,
                horizon_sec=horizon_sec,
                maker_fee_bps=float(args.maker_fee_bps),
                taker_fee_bps=float(args.taker_fee_bps),
                max_book_staleness_sec=int(args.max_book_staleness_sec),
                holdout_frac=float(args.holdout_frac),
                horizon_from=str(args.horizon_from),
                tp_bps=tp,
                sl_bps=sl,
            )
            for offset in offsets
            for margin in margins
            for tp in tps
            for sl in sls
        ]
    side_counts: dict[str, int] = {}
    for event in events:
        side_counts[event.side] = side_counts.get(event.side, 0) + 1
    payload = {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(args.db),
        "config": {
            "symbol": args.symbol,
            "threshold": float(args.threshold),
            "sides": list(sides),
            "min_vdepth_bps": float(args.min_vdepth_bps),
            "horizon_hr": float(args.horizon_hr),
            "horizon_from": str(args.horizon_from),
            "bucket_sec": int(args.bucket_sec),
            "min_gap_sec": int(args.min_gap_sec),
            "accel_window_sec": int(args.accel_window_sec),
            "offset_bps": list(offsets),
            "cross_margin_bps": list(margins),
            "tp_bps": [None if x >= NO_TP_OR_SL else x for x in tps],
            "sl_bps": [None if x >= NO_TP_OR_SL else x for x in sls],
            "maker_fee_bps": float(args.maker_fee_bps),
            "taker_fee_bps": float(args.taker_fee_bps),
            "max_book_staleness_sec": int(args.max_book_staleness_sec),
            "holdout_frac": float(args.holdout_frac),
        },
        "event_n": len(events),
        "event_side_counts": side_counts,
        "events_sha256": sha256_text("\n".join(f"{ev.side}:{ev.anchor.anchor_ts_ms}:{ev.vdepth_bps:.6f}" for ev in events)),
        "configs": configs,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(payload), encoding="utf-8")
    print(render_md(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
