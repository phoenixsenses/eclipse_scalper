"""S34 V-Shape Conditioning.

Tests the intuition that the SHAPE of the cascade -- specifically how deep price
overshot during the spike (the down-leg of the V) -- predicts the reversal. Not
"a cascade happened" but "price capitulated hard, so it springs back harder".

For each cascade anchor the knowable-at-entry V-depth is the cascade-direction
move from cascade start to the threshold cross:
    SELL-liq (down spike): (mark_start - mark_anchor)/mark_start
    BUY-liq  (up spike):   (mark_anchor - mark_start)/mark_start
We then take the FADE return (LONG after a down spike, SHORT after an up spike)
at swing horizons, bin events by V-depth terciles derived ON CALIBRATION, and
apply the cuts to the chronological holdout. The hypothesis is confirmed only if
the DEEP-depth bin is positive on BOTH splits and stronger than the shallow bin
(a monotone depth -> reversal response).

Uses the 200K threshold (max sample) and pools BUY+SELL; depth binning naturally
isolates the big-overshoot subset. Mark-to-mark returns minus round-trip cost.
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
    signed_return_bps,
)

DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_VSHAPE_CONDITIONING.json"
OUT_MD = OUT_DIR / "S34_VSHAPE_CONDITIONING.md"

HORIZONS_SEC = {"1h": 3600, "4h": 14400, "24h": 86400}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summ(vals: list[float], cost: float) -> dict[str, Any]:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {"n": 0, "median": None, "sum": 0.0, "win_rate": None, "net_median": None}
    m = pctile(vals, 0.5)
    return {"n": len(vals), "median": r1(m), "net_median": r1(m - cost), "sum": r1(sum(vals)),
            "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals))}


def build_events(conn, symbol, threshold, *, bucket_sec, min_gap_sec, accel_window_sec):
    marks = load_mark_index(conn, symbol)
    events = []
    for side, fade_dir in (("BUY", "SHORT"), ("SELL", "LONG")):
        liqs = load_liquidations(conn, symbol, side, None, None)
        anchors = reconstruct_anchors(liqs, bucket_sec=bucket_sec, min_gap_sec=min_gap_sec,
                                      thresholds=(float(threshold),), accel_window_sec=accel_window_sec)
        for a in anchors:
            if float(a.threshold_usd) != float(threshold):
                continue
            start = marks.at_or_after(int(a.first_ts_ms))
            anc = marks.at_or_after(int(a.anchor_ts_ms))
            if not start or not anc or float(start[1]) <= 0:
                continue
            # knowable V-depth = cascade-direction move from start to cross (positive = deeper spike)
            if side == "SELL":
                depth = (float(start[1]) - float(anc[1])) / float(start[1]) * 10_000.0
            else:
                depth = (float(anc[1]) - float(start[1])) / float(start[1]) * 10_000.0
            fwd = {}
            for hl, hs in HORIZONS_SEC.items():
                ex = marks.at_or_after(int(a.anchor_ts_ms) + hs * 1000)
                fwd[hl] = signed_return_bps(fade_dir, float(anc[1]), float(ex[1])) if ex else None
            events.append({"ts_ms": int(a.anchor_ts_ms), "depth_bps": depth, "fade": fwd})
    events.sort(key=lambda e: e["ts_ms"])
    return events


def screen(events, cost, holdout_frac):
    if len(events) < 30:
        return {"status": "THIN", "n": len(events)}
    cut = events[int(len(events) * (1.0 - holdout_frac))]["ts_ms"]
    cal_depths = sorted(e["depth_bps"] for e in events if e["ts_ms"] < cut)
    q33, q66 = pctile(cal_depths, 1 / 3), pctile(cal_depths, 2 / 3)

    def dbin(v):
        return "shallow" if v <= q33 else ("mid" if v <= q66 else "deep")

    out = {"status": "SCREENED", "n": len(events), "cut_depth_low": r1(q33), "cut_depth_high": r1(q66), "horizons": {}}
    for hl in HORIZONS_SEC:
        bins = {}
        for label in ("shallow", "mid", "deep"):
            cal = [e["fade"][hl] for e in events if e["ts_ms"] < cut and dbin(e["depth_bps"]) == label and e["fade"][hl] is not None]
            hold = [e["fade"][hl] for e in events if e["ts_ms"] >= cut and dbin(e["depth_bps"]) == label and e["fade"][hl] is not None]
            cs, hs = summ(cal, cost), summ(hold, cost)
            bins[label] = {"cal": cs, "hold": hs,
                           "stable_pos": (cs["n"] >= 20 and hs["n"] >= 10 and (cs["net_median"] or -1) > 0 and (hs["net_median"] or -1) > 0)}
        out["horizons"][hl] = bins
    return out


def render_md(report):
    cfg = report["config"]
    lines = [
        "# S34 V-Shape Conditioning (does spike depth predict the reversal?)",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  {cfg['symbol']} {int(cfg['threshold']/1000)}K, cost {cfg['cost']}bps, holdout {cfg['holdout_frac']}",
        "",
        "Fade return binned by knowable V-depth (cascade-direction overshoot at the cross). Terciles from calibration, "
        "applied to holdout. Hypothesis holds only if the DEEP bin is net-positive on BOTH splits and beats shallow "
        "(monotone depth->reversal). `**` = deep bin stable-positive both splits.",
        "",
    ]
    sc = report["screen"]
    if sc.get("status") != "SCREENED":
        lines.append(f"Insufficient events (n={sc.get('n')}).")
        return "\n".join(lines)
    lines.append(f"V-depth tercile cuts (bps): low<= {sc['cut_depth_low']}, high> {sc['cut_depth_high']}  |  total events: {sc['n']}")
    lines.append("")
    for hl, bins in sc["horizons"].items():
        lines.append(f"## {hl}")
        lines.append("")
        lines.append("| Depth bin | cal N | cal net med | cal win | hold N | hold net med | hold win | |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for label in ("shallow", "mid", "deep"):
            b = bins[label]
            cw = None if b["cal"]["win_rate"] is None else r1(b["cal"]["win_rate"] * 100.0)
            hw = None if b["hold"]["win_rate"] is None else r1(b["hold"]["win_rate"] * 100.0)
            flag = "**" if (label == "deep" and b["stable_pos"]) else ("+" if b["stable_pos"] else "")
            lines.append(f"| {label} | {b['cal']['n']} | {b['cal']['net_median']} | {cw} | {b['hold']['n']} | {b['hold']['net_median']} | {hw} | {flag} |")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Condition the liquidation-fade reversal on knowable V-shape (spike depth).")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--threshold", type=float, default=200_000.0)
    p.add_argument("--bucket-sec", type=int, default=300)
    p.add_argument("--min-gap-sec", type=int, default=900)
    p.add_argument("--accel-window-sec", type=int, default=30)
    p.add_argument("--cost-bps-rt", type=float, default=6.1)
    p.add_argument("--holdout-frac", type=float, default=0.30)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        events = build_events(conn, args.symbol, float(args.threshold), bucket_sec=int(args.bucket_sec),
                              min_gap_sec=int(args.min_gap_sec), accel_window_sec=int(args.accel_window_sec))
        sc = screen(events, float(args.cost_bps_rt), float(args.holdout_frac))
    report = {"generated_at_utc": utc_now(),
              "config": {"symbol": args.symbol, "threshold": float(args.threshold), "cost": float(args.cost_bps_rt), "holdout_frac": float(args.holdout_frac)},
              "screen": sc}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
