"""S34 Reversal Backtest (fixed-horizon fade of large liquidation cascades).

Promotes the swing-event screen to a real per-trade backtest. Entry: fade the
cascade -- SHORT after a BUY-side (up-spike) cascade, LONG after a SELL-side
(down-spike) cascade, at the knowable threshold cross. Exit: fixed horizon.

Real fills: entry and exit cross the spread via book_ticker (SHORT opens at bid /
closes at ask; LONG opens at ask / closes at bid), so spread cost is paid
explicitly; taker fees added per side. Two P&L views:
  - all: every cascade event independently (statistical view).
  - sequential: a single unit of capital, skip a new entry while a position is
    still open (realistic, removes overlap double-counting).
Chronological 70/30 calibration/holdout. Default cell: ETHUSDT 1M.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    book_at,
    iso_ms,
    load_liquidations,
    load_mark_index,
    mean,
    pctile,
    r1,
    r3,
    reconstruct_anchors,
    signed_return_bps,
)

DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_REVERSAL_BACKTEST.json"
OUT_MD = OUT_DIR / "S34_REVERSAL_BACKTEST.md"

HORIZONS_HR = (4, 8, 24)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summ(vals: list[float]) -> dict[str, Any]:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {"n": 0, "median": None, "mean": None, "win_rate": None, "sum": 0.0}
    return {
        "n": len(vals),
        "median": r1(pctile(vals, 0.5)),
        "mean": r1(mean(vals)),
        "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)),
        "sum": r1(sum(vals)),
    }


def collect_events(conn, symbol, threshold, *, bucket_sec, min_gap_sec, accel_window_sec, min_vdepth_bps=0.0) -> list[dict[str, Any]]:
    """Both sides; fade direction = opposite of the liquidation-implied push. Optional deep-V overshoot filter."""
    marks = load_mark_index(conn, symbol) if float(min_vdepth_bps) > 0 else None
    events = []
    for side, fade_dir in (("BUY", "SHORT"), ("SELL", "LONG")):
        liqs = load_liquidations(conn, symbol, side, None, None)
        anchors = reconstruct_anchors(
            liqs, bucket_sec=bucket_sec, min_gap_sec=min_gap_sec,
            thresholds=(float(threshold),), accel_window_sec=accel_window_sec,
        )
        for a in anchors:
            if float(a.threshold_usd) != float(threshold):
                continue
            if marks is not None:
                start = marks.at_or_after(int(a.first_ts_ms))
                anc = marks.at_or_after(int(a.anchor_ts_ms))
                if not start or not anc or float(start[1]) <= 0:
                    continue
                depth = (float(start[1]) - float(anc[1])) / float(start[1]) * 10_000.0 if side == "SELL" else (float(anc[1]) - float(start[1])) / float(start[1]) * 10_000.0
                if depth < float(min_vdepth_bps):
                    continue
            events.append({"ts_ms": int(a.anchor_ts_ms), "side": side, "fade_dir": fade_dir})
    events.sort(key=lambda e: e["ts_ms"])
    return events


def trade_pnl(conn, symbol, ev, horizon_hr, *, fee_bps_side, max_book_staleness_sec) -> dict[str, Any] | None:
    entry_ts = int(ev["ts_ms"])
    exit_ts = entry_ts + int(horizon_hr) * 3600 * 1000
    eb = book_at(conn, symbol, entry_ts, max_book_staleness_sec)
    xb = book_at(conn, symbol, exit_ts, max_book_staleness_sec)
    if not eb or not xb:
        return None
    fade = ev["fade_dir"]
    if fade == "SHORT":
        entry_px, exit_px = eb.bid, xb.ask
    else:
        entry_px, exit_px = eb.ask, xb.bid
    gross = signed_return_bps(fade, float(entry_px), float(exit_px))
    net = gross - 2.0 * float(fee_bps_side)
    return {"entry_ts_ms": entry_ts, "exit_ts_ms": exit_ts, "fade_dir": fade, "gross_bps": r1(gross), "net_bps": net}


def run_horizon(conn, symbol, events, horizon_hr, *, fee_bps_side, max_book_staleness_sec, holdout_frac) -> dict[str, Any]:
    trades = []
    no_fill = 0
    for ev in events:
        t = trade_pnl(conn, symbol, ev, horizon_hr, fee_bps_side=fee_bps_side, max_book_staleness_sec=max_book_staleness_sec)
        if t is None:
            no_fill += 1
            continue
        trades.append(t)
    if not trades:
        return {"horizon_hr": horizon_hr, "filled_n": 0, "no_fill_n": no_fill}
    cut = trades[int(len(trades) * (1.0 - holdout_frac))]["entry_ts_ms"] if len(trades) > 1 else trades[-1]["entry_ts_ms"] + 1

    # sequential single-unit-capital pass (no overlapping positions)
    seq, open_until = [], -1
    for t in trades:
        if t["entry_ts_ms"] >= open_until:
            seq.append(t)
            open_until = t["exit_ts_ms"]

    def split(rows):
        return [r["net_bps"] for r in rows if r["entry_ts_ms"] < cut], [r["net_bps"] for r in rows if r["entry_ts_ms"] >= cut]

    all_cal, all_hold = split(trades)
    seq_cal, seq_hold = split(seq)
    return {
        "horizon_hr": horizon_hr,
        "filled_n": len(trades),
        "no_fill_n": no_fill,
        "all": {"cal": summ(all_cal), "hold": summ(all_hold), "overall": summ([t["net_bps"] for t in trades])},
        "sequential": {"n": len(seq), "cal": summ(seq_cal), "hold": summ(seq_hold), "overall": summ([t["net_bps"] for t in seq])},
    }


def render_md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    lines = [
        "# S34 Reversal Backtest (fade large liquidation cascades, fixed horizon)",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  {cfg['symbol']} {int(cfg['threshold']/1000)}K, "
        f"fee {cfg['fee_bps_side']}bps/side, book staleness {cfg['max_book_staleness_sec']}s, holdout {cfg['holdout_frac']}",
        "",
        "Entry fades the cascade at the threshold cross (SHORT after BUY-liq, LONG after SELL-liq); exit at fixed horizon. "
        "Spread paid via bid/ask fills; net = gross - 2*fee. `all` = every event; `sequential` = single-unit capital, no overlap.",
        "",
        f"Total fade events: {report['event_n']}",
        "",
        "| Horizon | Filled | all cal med | all cal win | all hold med | all hold win | all hold sum | seq N | seq hold med | seq hold sum |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for h in report["horizons"]:
        if h.get("filled_n", 0) == 0:
            lines.append(f"| {h['horizon_hr']}h | 0 | | | | | | | | |")
            continue
        a, s = h["all"], h["sequential"]
        wr = lambda x: None if x["win_rate"] is None else r1(x["win_rate"] * 100.0)
        lines.append(
            f"| {h['horizon_hr']}h | {h['filled_n']} | {a['cal']['median']} | {wr(a['cal'])} | "
            f"{a['hold']['median']} | {wr(a['hold'])} | {a['hold']['sum']} | {s['n']} | {s['hold']['median']} | {s['hold']['sum']} |"
        )
    lines.append("")
    lines.append("Read: a credible edge wants positive median on BOTH cal and hold in the `all` view, and a positive "
                 "`sequential` holdout sum (realistic single-capital P&L).")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fixed-horizon reversal backtest fading large liquidation cascades.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--threshold", type=float, default=1_000_000.0)
    p.add_argument("--min-vdepth-bps", type=float, default=0.0, help="Deep-V overshoot filter (bps).")
    p.add_argument("--bucket-sec", type=int, default=300)
    p.add_argument("--min-gap-sec", type=int, default=900)
    p.add_argument("--accel-window-sec", type=int, default=30)
    p.add_argument("--fee-bps-side", type=float, default=3.05)
    p.add_argument("--max-book-staleness-sec", type=int, default=10)
    p.add_argument("--holdout-frac", type=float, default=0.30)
    p.add_argument("--horizons-hr", default=",".join(str(h) for h in HORIZONS_HR))
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    horizons = [int(x) for x in str(args.horizons_hr).split(",") if x.strip()]
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        events = collect_events(conn, args.symbol, float(args.threshold),
                                bucket_sec=int(args.bucket_sec), min_gap_sec=int(args.min_gap_sec),
                                accel_window_sec=int(args.accel_window_sec), min_vdepth_bps=float(args.min_vdepth_bps))
        results = [
            run_horizon(conn, args.symbol, events, h, fee_bps_side=float(args.fee_bps_side),
                        max_book_staleness_sec=int(args.max_book_staleness_sec), holdout_frac=float(args.holdout_frac))
            for h in horizons
        ]
    report = {
        "generated_at_utc": utc_now(),
        "config": {"symbol": args.symbol, "threshold": float(args.threshold), "fee_bps_side": float(args.fee_bps_side),
                   "max_book_staleness_sec": int(args.max_book_staleness_sec), "holdout_frac": float(args.holdout_frac)},
        "event_n": len(events),
        "horizons": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
