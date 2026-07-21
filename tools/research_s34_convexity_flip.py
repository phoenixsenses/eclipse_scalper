"""S34 Convexity Flip.

The butterfly-seed test showed the revert-vs-runaway outcome is NOT knowable at
entry -- the -300 bps runaways look identical to the winners at the cross. So you
cannot filter the tail by selection. The structural consequence: instead of
FADING (short the unpredictable tail -> negative skew: win often, lose huge),
flip to being LONG the tail.

Continuation with a tight stop: trade the cascade DIRECTION (SHORT a SELL/down
cascade, LONG a BUY/up cascade) with a tight stop and no/far TP. Most cascades
revert -> many small stop-outs; the rare runaway -> one large win. Positive skew.
The question is whether bounded small losses + rare big wins net positive on
calibration AND holdout. (Note: T3R is the WRONG metric here -- the winners ARE
the edge -- so we judge on sum, win-rate, max_win, max_loss.)

Reuses simulate_route (TP/SL path walk + bid/ask fills). Default ETH SELL deep-V.
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
OUT_JSON = OUT_DIR / "S34_CONVEXITY_FLIP.json"
OUT_MD = OUT_DIR / "S34_CONVEXITY_FLIP.md"

NO_TP = 100_000.0
NO_BE = 100_000.0
SWEEP_SL = (10.0, 15.0, 20.0, 30.0, 50.0)
SWEEP_TP = (NO_TP, 200.0, 400.0)  # let it run, or far targets


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def metrics(vals: list[float]) -> dict[str, Any]:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {"n": 0, "sum": 0.0, "median": None, "mean": None, "win_rate": None, "max_win": None, "max_loss": None}
    return {"n": len(vals), "sum": r1(sum(vals)), "median": r1(pctile(vals, 0.5)), "mean": r1(mean(vals)),
            "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)), "max_win": r1(max(vals)), "max_loss": r1(min(vals))}


def cont_spec(side, sl, tp, horizon_sec, symbol, threshold):
    # continuation = trade WITH the cascade push: SELL/down-spike -> SHORT, BUY/up-spike -> LONG
    direction = "SHORT" if side == "SELL" else "LONG"
    return RouteSpec(family="ETH_CONVEX", rule_name=f"CONT_{direction}_SL{int(sl)}_TP{int(tp)}",
                     symbol=symbol, liq_side=side, direction=direction, threshold_usd=float(threshold),
                     tp_bps=float(tp), sl_bps=float(sl), be_bps=NO_BE, entry_delay_sec=0, max_horizon_sec=int(horizon_sec))


def collect(conn, marks, symbol, side, threshold, *, bucket_sec, min_gap_sec, accel_window_sec, min_vdepth_bps):
    liqs = load_liquidations(conn, symbol, side, None, None)
    anchors = reconstruct_anchors(liqs, bucket_sec=bucket_sec, min_gap_sec=min_gap_sec,
                                  thresholds=(float(threshold),), accel_window_sec=accel_window_sec)
    out = []
    for a in anchors:
        if float(a.threshold_usd) != float(threshold):
            continue
        start = marks.at_or_after(int(a.first_ts_ms))
        anc = marks.at_or_after(int(a.anchor_ts_ms))
        if not start or not anc or float(start[1]) <= 0:
            continue
        depth = ((float(start[1]) - float(anc[1])) if side == "SELL" else (float(anc[1]) - float(start[1]))) / float(start[1]) * 10_000.0
        if depth >= float(min_vdepth_bps):
            out.append(a)
    return out


def run_combo(conn, marks, anchors, side, sl, tp, horizon_sec, *, symbol, threshold, fee_bps_side, max_book_staleness_sec, holdout_frac):
    spec = cont_spec(side, sl, tp, horizon_sec, symbol, threshold)
    trades = []
    for a in anchors:
        sim = simulate_route(conn, marks, spec, a, fee_bps_side=fee_bps_side, max_book_staleness_sec=max_book_staleness_sec)
        if sim.get("status") != "FILLED" or sim.get("net_bps") is None:
            continue
        trades.append((int(a.anchor_ts_ms), float(sim["net_bps"]), sim.get("exit_reason")))
    if not trades:
        return None
    trades.sort(key=lambda t: t[0])
    cut = trades[int(len(trades) * (1.0 - holdout_frac))][0]
    exits: dict[str, int] = {}
    for _, _, rsn in trades:
        exits[str(rsn)] = exits.get(str(rsn), 0) + 1
    cal = [n for ts, n, _ in trades if ts < cut]
    hold = [n for ts, n, _ in trades if ts >= cut]
    return {"sl_bps": sl, "tp_bps": (None if tp >= NO_TP else tp), "filled_n": len(trades), "exit_reasons": exits,
            "cal": metrics(cal), "hold": metrics(hold),
            "lead": (metrics(cal)["sum"] > 0 and metrics(hold)["sum"] > 0 and len(cal) >= 20 and len(hold) >= 10)}


def render_md(report):
    cfg = report["config"]
    lines = [
        "# S34 Convexity Flip (continuation + tight stop, long the tail)",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  {cfg['symbol']} {cfg['side']} deep-V>= {cfg['min_vdepth_bps']}bps "
        f"{int(cfg['threshold']/1000)}K {cfg['horizon_hr']}h, fee {cfg['fee_bps_side']}/side  |  anchors={report['anchor_n']}",
        "",
        "Trade WITH the cascade (continuation) with a tight stop and no/far TP. Positive skew: many small losses, "
        "rare big wins. `**` = cal sum>0 AND hold sum>0. Judge on SUM + win-rate + max_win/max_loss (NOT T3R).",
        "",
        "| SL | TP | Filled | cal N | cal sum | cal win | cal maxW | cal maxL | hold N | hold sum | hold win | hold maxW | hold maxL | exits | |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for c in report["combos"]:
        cc, hh = c["cal"], c["hold"]
        cw = lambda m: None if m["win_rate"] is None else r1(m["win_rate"] * 100.0)
        ex = ",".join(f"{k}:{v}" for k, v in sorted(c["exit_reasons"].items()))
        flag = "**" if c["lead"] else ""
        lines.append(f"| {int(c['sl_bps'])} | {'none' if c['tp_bps'] is None else int(c['tp_bps'])} | {c['filled_n']} | "
                     f"{cc['n']} | {cc['sum']} | {cw(cc)} | {cc['max_win']} | {cc['max_loss']} | "
                     f"{hh['n']} | {hh['sum']} | {cw(hh)} | {hh['max_win']} | {hh['max_loss']} | {ex} | {flag} |")
    lines.append("")
    leads = [c for c in report["combos"] if c["lead"]]
    lines.append(f"## Leads (cal sum>0 AND hold sum>0): {len(leads)}")
    for c in leads:
        lines.append(f"- SL={int(c['sl_bps'])} TP={'none' if c['tp_bps'] is None else int(c['tp_bps'])}: "
                     f"cal sum={c['cal']['sum']} (win {r1(c['cal']['win_rate']*100)}%), hold sum={c['hold']['sum']} (win {r1(c['hold']['win_rate']*100)}%)")
    if not leads:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Convexity flip: deep-V continuation with tight stop (long the unpredictable tail).")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--side", default="SELL")
    p.add_argument("--threshold", type=float, default=200_000.0)
    p.add_argument("--min-vdepth-bps", type=float, default=28.0)
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
        anchors = collect(conn, marks, args.symbol, args.side, float(args.threshold), bucket_sec=int(args.bucket_sec),
                          min_gap_sec=int(args.min_gap_sec), accel_window_sec=int(args.accel_window_sec),
                          min_vdepth_bps=float(args.min_vdepth_bps))
        combos = []
        for sl in SWEEP_SL:
            for tp in SWEEP_TP:
                c = run_combo(conn, marks, anchors, args.side, sl, tp, horizon_sec, symbol=args.symbol,
                              threshold=float(args.threshold), fee_bps_side=float(args.fee_bps_side),
                              max_book_staleness_sec=int(args.max_book_staleness_sec), holdout_frac=float(args.holdout_frac))
                if c:
                    combos.append(c)
    report = {"generated_at_utc": utc_now(),
              "config": {"symbol": args.symbol, "side": args.side, "threshold": float(args.threshold),
                         "min_vdepth_bps": float(args.min_vdepth_bps), "horizon_hr": int(args.horizon_hr), "fee_bps_side": float(args.fee_bps_side)},
              "anchor_n": len(anchors), "combos": combos}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
