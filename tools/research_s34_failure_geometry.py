"""S34 Failure Geometry / Feedback.

The static seed does not separate reverters from runaways (butterfly result). But
the DYNAMICS just after entry might: a reverter starts recovering within minutes;
a runaway keeps falling (fails to reclaim) while its liquidation chain ACCELERATES
(positive feedback). Both are knowable in real time (not lookahead) and -- unlike
binary "resumption" which fired on everyone -- a reclaim/feedback condition may be
SPECIFIC to the losers.

On the 4-month bridged deep-V SELL fade (LONG), per event we compute:
  - reclaim at tau: is mark(entry+tau) back >= entry (fade working early)?
  - mae_5m: worst adverse excursion in the first 5 min.
  - liq feedback: same-side liq notional in [entry,5m] vs [5m,10m] (accelerating chain?)
Then we test a RECLAIM-STOP: hold to 4h only if recovering at tau, else cut at tau.
The win: cut the -400 tail while keeping the 4-month cal+hold edge. We also report
the reverter-vs-runaway separation on each dynamic feature.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
import sys
from collections import defaultdict
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
OUT_JSON = OUT_DIR / "S34_FAILURE_GEOMETRY.json"
OUT_MD = OUT_DIR / "S34_FAILURE_GEOMETRY.md"

HORIZON_SEC = 4 * 3600
TAUS_MIN = (5, 10, 15, 30)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def month_of(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m")


def metrics(vals: list[float]) -> dict[str, Any]:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {"n": 0, "sum": 0.0, "median": None, "win_rate": None, "max_loss": None}
    return {"n": len(vals), "sum": r1(sum(vals)), "median": r1(pctile(vals, 0.5)),
            "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)), "max_loss": r1(min(vals))}


def split(rows, key, holdout_frac):
    rows = sorted([r for r in rows if r.get(key) is not None], key=lambda r: r["entry_ts_ms"])
    if not rows:
        return {"cal": metrics([]), "hold": metrics([])}
    cut = rows[int(len(rows) * (1.0 - holdout_frac))]["entry_ts_ms"]
    return {"cal": metrics([r[key] for r in rows if r["entry_ts_ms"] < cut]),
            "hold": metrics([r[key] for r in rows if r["entry_ts_ms"] >= cut])}


def med(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return r1(pctile(vals, 0.5)) if vals else None


def build(conn, symbol, side, threshold, *, bucket_sec, min_gap_sec, accel_window_sec, min_vdepth_bps, cost):
    marks = load_mark_index(conn, symbol)
    liqs = load_liquidations(conn, symbol, side, None, None)
    liq_ts = [int(r["ts_ms"]) for r in liqs]
    anchors = reconstruct_anchors(liqs, bucket_sec=bucket_sec, min_gap_sec=min_gap_sec,
                                  thresholds=(float(threshold),), accel_window_sec=accel_window_sec)

    def liq_notional(a_ts, b_ts):
        lo = bisect.bisect_right(liq_ts, int(a_ts))
        hi = bisect.bisect_right(liq_ts, int(b_ts))
        return sum(float(liqs[i]["notional"]) for i in range(lo, hi))

    rows = []
    for a in anchors:
        if float(a.threshold_usd) != float(threshold):
            continue
        start = marks.at_or_after(int(a.first_ts_ms))
        anc = marks.at_or_after(int(a.anchor_ts_ms))
        if not start or not anc or float(start[1]) <= 0:
            continue
        depth = (float(start[1]) - float(anc[1])) / float(start[1]) * 10_000.0 if side == "SELL" else (float(anc[1]) - float(start[1])) / float(start[1]) * 10_000.0
        if depth < float(min_vdepth_bps):
            continue
        entry_ts = int(a.anchor_ts_ms)
        entry_px = float(anc[1])
        ex4 = marks.at_or_after(entry_ts + HORIZON_SEC * 1000)
        if not ex4:
            continue
        baseline = signed_return_bps("LONG", entry_px, float(ex4[1])) - cost

        # reclaim-stop variants: hold to 4h only if recovering at tau, else cut at tau
        tau_nets = {}
        reclaimed = {}
        for tm in TAUS_MIN:
            mk = marks.at_or_after(entry_ts + tm * 60 * 1000)
            if not mk:
                tau_nets[tm] = baseline
                reclaimed[tm] = None
                continue
            rec = float(mk[1]) >= entry_px
            reclaimed[tm] = rec
            tau_nets[tm] = baseline if rec else (signed_return_bps("LONG", entry_px, float(mk[1])) - cost)

        # dynamics for separation
        path5 = marks.slice_range(entry_ts, entry_ts + 5 * 60 * 1000)
        mae5 = min((signed_return_bps("LONG", entry_px, float(px)) for _, px in path5), default=0.0)
        liq_0_5 = liq_notional(entry_ts, entry_ts + 5 * 60 * 1000)
        liq_5_10 = liq_notional(entry_ts + 5 * 60 * 1000, entry_ts + 10 * 60 * 1000)
        accel = (liq_5_10 / liq_0_5) if liq_0_5 > 0 else 0.0

        rows.append({
            "entry_ts_ms": entry_ts, "month": month_of(entry_ts), "baseline_net": baseline,
            "reclaimed_5m": reclaimed.get(5), "mae_5m_bps": r1(mae5),
            "liq_0_5_k": r1(liq_0_5 / 1000.0), "liq_accel_5to10": r1(accel),
            **{f"reclaim_stop_{tm}m_net": tau_nets[tm] for tm in TAUS_MIN},
        })
    return rows


def render_md(report):
    cfg = report["config"]
    lines = [
        "# S34 Failure Geometry / Feedback (cut the tail by dynamics, not seed)",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  {cfg['symbol']} {cfg['side']} deep-V>= {cfg['min_vdepth_bps']}bps "
        f"{int(cfg['threshold']/1000)}K 4h, cost {cfg['cost']}bps RT, bridged span",
        "",
        f"Events: {report['event_n']}  |  winners: {report['n_win']}  runaways(net<-100): {report['n_runaway']}",
        "",
        "## Reverter vs runaway separation (dynamic, knowable in real time)",
        "",
        "| Feature | winners med | runaways med | separates? |",
        "| --- | ---: | ---: | --- |",
    ]
    for f, d in report["separation"].items():
        lines.append(f"| {f} | {d['winner']} | {d['runaway']} | {d['note']} |")
    lines.append("")
    lines.append("## Reclaim-stop variants (hold to 4h only if recovering at tau)")
    lines.append("")
    lines.append("| Variant | cal N | cal sum | cal win | cal maxL | hold N | hold sum | hold win | hold maxL | pos months |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, d in report["variants"].items():
        s = d["split"]
        cw = lambda m: None if m["win_rate"] is None else r1(m["win_rate"] * 100.0)
        lines.append(f"| {name} | {s['cal']['n']} | {s['cal']['sum']} | {cw(s['cal'])} | {s['cal']['max_loss']} | "
                     f"{s['hold']['n']} | {s['hold']['sum']} | {cw(s['hold'])} | {s['hold']['max_loss']} | {d['pos_months']}/{report['n_months']} |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Dynamic failure-geometry / feedback tail-cut on the bridged deep-V fade.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--side", default="SELL")
    p.add_argument("--threshold", type=float, default=200_000.0)
    p.add_argument("--min-vdepth-bps", type=float, default=28.0)
    p.add_argument("--bucket-sec", type=int, default=300)
    p.add_argument("--min-gap-sec", type=int, default=900)
    p.add_argument("--accel-window-sec", type=int, default=30)
    p.add_argument("--fee-bps-side", type=float, default=3.05)
    p.add_argument("--modeled-spread-bps", type=float, default=2.0)
    p.add_argument("--holdout-frac", type=float, default=0.30)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cost = 2.0 * float(args.fee_bps_side) + float(args.modeled_spread_bps)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        rows = build(conn, args.symbol, args.side, float(args.threshold), bucket_sec=int(args.bucket_sec),
                     min_gap_sec=int(args.min_gap_sec), accel_window_sec=int(args.accel_window_sec),
                     min_vdepth_bps=float(args.min_vdepth_bps), cost=cost)
    win = [r for r in rows if r["baseline_net"] > 0]
    runaway = [r for r in rows if r["baseline_net"] < -100.0]
    sep_feats = ("reclaimed_5m", "mae_5m_bps", "liq_0_5_k", "liq_accel_5to10")
    separation = {}
    for f in sep_feats:
        wv = [(1.0 if r[f] else 0.0) if f == "reclaimed_5m" else r[f] for r in win]
        rv = [(1.0 if r[f] else 0.0) if f == "reclaimed_5m" else r[f] for r in runaway]
        wm, rm = med(wv), med(rv)
        note = ""
        if wm is not None and rm is not None:
            note = "yes" if abs(wm - rm) > 0.25 * (abs(wm) + abs(rm) + 1e-9) else "weak"
        separation[f] = {"winner": wm, "runaway": rm, "note": note}

    months = sorted({r["month"] for r in rows})
    hf = float(args.holdout_frac)
    variants = {}
    for key, name in [("baseline_net", "baseline_4h")] + [(f"reclaim_stop_{tm}m_net", f"reclaim_stop_{tm}m") for tm in TAUS_MIN]:
        sp = split(rows, key, hf)
        pos = sum(1 for m in months if metrics([r[key] for r in rows if r["month"] == m and r.get(key) is not None])["sum"] > 0)
        variants[name] = {"split": sp, "pos_months": pos}

    report = {
        "generated_at_utc": utc_now(),
        "config": {"symbol": args.symbol, "side": args.side, "threshold": float(args.threshold),
                   "min_vdepth_bps": float(args.min_vdepth_bps), "cost": r1(cost)},
        "event_n": len(rows), "n_win": len(win), "n_runaway": len(runaway), "n_months": len(months),
        "separation": separation, "variants": variants, "rows": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
