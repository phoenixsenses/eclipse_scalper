"""S34 Reversal Stop Backtest.

The fixed-horizon ETH-1M fade had a positive median but a negative total at 4h:
the minority of cascades that do NOT reverse run away and, with no stop, wipe the
P&L. This adds a stop (and optional TP / BE) to cap that tail and tests whether
the fade total turns positive on BOTH calibration and holdout.

Reuses the validated `simulate_route` (TP/SL/BE path walk + bid/ask book fills)
by constructing FADE route specs: BUY-side (up-spike) cascades -> SHORT, SELL-side
(down-spike) cascades -> LONG, entered at the threshold cross. Sweeps SL / TP / BE
at a fixed horizon. A combo is a lead only if cal sum > 0 AND hold sum > 0.
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
    load_liquidations,
    load_mark_index,
    mean,
    pctile,
    r1,
    r3,
    reconstruct_anchors,
)
from tools.research_s34_knowable_anchor_route_recheck import RouteSpec, simulate_route

DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_REVERSAL_STOP_BACKTEST.json"
OUT_MD = OUT_DIR / "S34_REVERSAL_STOP_BACKTEST.md"

NO_TP = 100_000.0  # effectively disables TP (stop + horizon only)
NO_BE = 100_000.0  # effectively disables break-even (be_bps=0 would stop at entry instantly)
SWEEP_SL = (30.0, 50.0, 80.0, 120.0)
SWEEP_TP = (60.0, 120.0, NO_TP)
SWEEP_BE = (NO_BE, 40.0)  # pure stop, and a break-even-managed variant


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summ(vals: list[float]) -> dict[str, Any]:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {"n": 0, "median": None, "mean": None, "win_rate": None, "sum": 0.0}
    return {"n": len(vals), "median": r1(pctile(vals, 0.5)), "mean": r1(mean(vals)),
            "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)), "sum": r1(sum(vals))}


def fade_spec(side, sl, tp, be, horizon_sec, symbol, threshold):
    direction = "SHORT" if side == "BUY" else "LONG"
    return RouteSpec(
        family="ETH_FADE", rule_name=f"FADE_{direction}_SL{int(sl)}_TP{int(tp)}_BE{int(be)}",
        symbol=symbol, liq_side=side, direction=direction, threshold_usd=float(threshold),
        tp_bps=float(tp), sl_bps=float(sl), be_bps=float(be), entry_delay_sec=0, max_horizon_sec=int(horizon_sec),
    )


def anchor_vdepth_bps(marks, anchor, side) -> float | None:
    """Knowable overshoot from cascade start to cross, in the cascade direction."""
    start = marks.at_or_after(int(anchor.first_ts_ms))
    anc = marks.at_or_after(int(anchor.anchor_ts_ms))
    if not start or not anc or float(start[1]) <= 0:
        return None
    if side == "SELL":
        return (float(start[1]) - float(anc[1])) / float(start[1]) * 10_000.0
    return (float(anc[1]) - float(start[1])) / float(start[1]) * 10_000.0


def collect_anchors(conn, marks, symbol, threshold, *, bucket_sec, min_gap_sec, accel_window_sec, min_vdepth_bps):
    out = {}
    for side in ("BUY", "SELL"):
        liqs = load_liquidations(conn, symbol, side, None, None)
        anchors = reconstruct_anchors(liqs, bucket_sec=bucket_sec, min_gap_sec=min_gap_sec,
                                      thresholds=(float(threshold),), accel_window_sec=accel_window_sec)
        kept = []
        for a in anchors:
            if float(a.threshold_usd) != float(threshold):
                continue
            if min_vdepth_bps > 0:
                d = anchor_vdepth_bps(marks, a, side)
                if d is None or d < float(min_vdepth_bps):
                    continue
            kept.append(a)
        out[side] = kept
    return out


def run_combo(conn, marks, anchors_by_side, sl, tp, be, horizon_sec, *, symbol, threshold, fee_bps_side, max_book_staleness_sec, holdout_frac):
    trades = []  # (entry_ts_ms, net_bps, exit_reason)
    for side, anchors in anchors_by_side.items():
        spec = fade_spec(side, sl, tp, be, horizon_sec, symbol, threshold)
        for a in anchors:
            sim = simulate_route(conn, marks, spec, a, fee_bps_side=fee_bps_side, max_book_staleness_sec=max_book_staleness_sec)
            if sim.get("status") != "FILLED" or sim.get("net_bps") is None:
                continue
            trades.append((int(a.anchor_ts_ms), float(sim["net_bps"]), sim.get("exit_reason")))
    if not trades:
        return None
    trades.sort(key=lambda t: t[0])
    cut = trades[int(len(trades) * (1.0 - holdout_frac))][0]
    cal = [n for ts, n, _ in trades if ts < cut]
    hold = [n for ts, n, _ in trades if ts >= cut]
    exits: dict[str, int] = {}
    for _, _, rsn in trades:
        exits[str(rsn)] = exits.get(str(rsn), 0) + 1
    return {
        "sl_bps": sl, "tp_bps": (None if tp >= NO_TP else tp), "be_bps": be,
        "filled_n": len(trades), "exit_reasons": exits,
        "cal": summ(cal), "hold": summ(hold), "overall": summ([n for _, n, _ in trades]),
        "lead": (summ(cal)["sum"] > 0 and summ(hold)["sum"] > 0 and len(cal) >= 20 and len(hold) >= 10),
    }


def render_md(report):
    cfg = report["config"]
    lines = [
        "# S34 Reversal Stop Backtest (ETH 1M fade + stop)",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  horizon {cfg['horizon_hr']}h, fee {cfg['fee_bps_side']}bps/side, "
        f"holdout {cfg['holdout_frac']}  |  events: BUY={report['anchor_buy_n']} SELL={report['anchor_sell_n']}",
        "",
        "Fade entry (SHORT after BUY-liq, LONG after SELL-liq) with stop. `**` = cal sum>0 AND hold sum>0. "
        "TP=`none` means stop+horizon only.",
        "",
        "| SL | TP | BE | Filled | cal sum | cal med | cal win | hold sum | hold med | hold win | exits | |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for c in report["combos"]:
        wr = lambda x: None if x["win_rate"] is None else r1(x["win_rate"] * 100.0)
        ex = ",".join(f"{k}:{v}" for k, v in sorted(c["exit_reasons"].items()))
        flag = "**" if c["lead"] else ""
        be_lbl = "off" if c["be_bps"] >= NO_BE else int(c["be_bps"])
        lines.append(
            f"| {int(c['sl_bps'])} | {'none' if c['tp_bps'] is None else int(c['tp_bps'])} | {be_lbl} | {c['filled_n']} | "
            f"{c['cal']['sum']} | {c['cal']['median']} | {wr(c['cal'])} | {c['hold']['sum']} | {c['hold']['median']} | "
            f"{wr(c['hold'])} | {ex} | {flag} |"
        )
    lines.append("")
    leads = [c for c in report["combos"] if c["lead"]]
    lines.append(f"## Leads (cal sum>0 AND hold sum>0): {len(leads)}")
    for c in leads:
        lines.append(f"- SL={int(c['sl_bps'])} TP={'none' if c['tp_bps'] is None else int(c['tp_bps'])}: "
                     f"cal sum={c['cal']['sum']} ({c['cal']['n']}), hold sum={c['hold']['sum']} ({c['hold']['n']})")
    if not leads:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ETH 1M liquidation-fade backtest with TP/SL/BE stop sweep.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--threshold", type=float, default=1_000_000.0)
    p.add_argument("--min-vdepth-bps", type=float, default=0.0, help="Keep only cascades whose knowable overshoot >= this (V-shape filter).")
    p.add_argument("--horizon-hr", type=int, default=4)
    p.add_argument("--bucket-sec", type=int, default=300)
    p.add_argument("--min-gap-sec", type=int, default=900)
    p.add_argument("--accel-window-sec", type=int, default=30)
    p.add_argument("--fee-bps-side", type=float, default=3.05)
    p.add_argument("--max-book-staleness-sec", type=int, default=10)
    p.add_argument("--holdout-frac", type=float, default=0.30)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    horizon_sec = int(args.horizon_hr) * 3600
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        marks = load_mark_index(conn, args.symbol)
        anchors = collect_anchors(conn, marks, args.symbol, float(args.threshold), bucket_sec=int(args.bucket_sec),
                                  min_gap_sec=int(args.min_gap_sec), accel_window_sec=int(args.accel_window_sec),
                                  min_vdepth_bps=float(args.min_vdepth_bps))
        combos = []
        for sl in SWEEP_SL:
            for tp in SWEEP_TP:
                for be in SWEEP_BE:
                    c = run_combo(conn, marks, anchors, sl, tp, be, horizon_sec,
                                  symbol=args.symbol, threshold=float(args.threshold),
                                  fee_bps_side=float(args.fee_bps_side), max_book_staleness_sec=int(args.max_book_staleness_sec),
                                  holdout_frac=float(args.holdout_frac))
                    if c:
                        combos.append(c)
    report = {
        "generated_at_utc": utc_now(),
        "config": {"symbol": args.symbol, "threshold": float(args.threshold), "min_vdepth_bps": float(args.min_vdepth_bps),
                   "horizon_hr": int(args.horizon_hr), "fee_bps_side": float(args.fee_bps_side), "holdout_frac": float(args.holdout_frac)},
        "anchor_buy_n": len(anchors["BUY"]), "anchor_sell_n": len(anchors["SELL"]),
        "combos": combos,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
