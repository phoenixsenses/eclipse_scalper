"""S34 Confirmation Entry (wait for the reclaim, then enter).

The failure-geometry result: deep-V SELL fades that reclaim the entry within tau
minutes are strongly positive (mean +28bps, ~64% win, cal+hold both positive),
while non-reclaimers are negative. But you cannot know at the cross that it will
reclaim. The tradeable form is a CONFIRMATION ENTRY: don't catch the falling
knife -- wait tau minutes, and only enter (LONG) if price has reclaimed the cross
level, entering at the (higher) reclaimed price, then hold to 4h. You give up the
very bottom for a much higher win rate; the question is whether the remaining
upside nets out.

Mark-based + modeled spread over the full bridged span (Feb-Jun). Sweeps the
confirmation delay tau. Reports per-month and chronological holdout. This is the
direct, knowable, tradeable test of the lead.
"""

from __future__ import annotations

import argparse
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
OUT_JSON = OUT_DIR / "S34_CONFIRMATION_ENTRY.json"
OUT_MD = OUT_DIR / "S34_CONFIRMATION_ENTRY.md"

HORIZON_SEC = 4 * 3600
TAUS_MIN = (5, 10, 15)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def month_of(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m")


def metrics(vals: list[float]) -> dict[str, Any]:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {"n": 0, "sum": 0.0, "mean": None, "median": None, "win_rate": None, "max_loss": None}
    return {"n": len(vals), "sum": r1(sum(vals)), "mean": r1(mean(vals)), "median": r1(pctile(vals, 0.5)),
            "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)), "max_loss": r1(min(vals))}


def build(conn, symbol, side, threshold, *, bucket_sec, min_gap_sec, accel_window_sec, min_vdepth_bps, cost):
    marks = load_mark_index(conn, symbol)
    liqs = load_liquidations(conn, symbol, side, None, None)
    anchors = reconstruct_anchors(liqs, bucket_sec=bucket_sec, min_gap_sec=min_gap_sec,
                                  thresholds=(float(threshold),), accel_window_sec=accel_window_sec)
    events = []
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
        cross_ts = int(a.anchor_ts_ms)
        cross_px = float(anc[1])
        ex4 = marks.at_or_after(cross_ts + HORIZON_SEC * 1000)
        if not ex4:
            continue
        rec = {"entry_ts_ms": cross_ts, "month": month_of(cross_ts),
               "baseline_net": signed_return_bps("LONG", cross_px, float(ex4[1])) - cost}
        for tm in TAUS_MIN:
            mk = marks.at_or_after(cross_ts + tm * 60 * 1000)
            # confirmation: enter only if reclaimed cross level by tau; entry at the tau price; exit 4h from cross
            if mk and float(mk[1]) >= cross_px:
                rec[f"conf_{tm}m_net"] = signed_return_bps("LONG", float(mk[1]), float(ex4[1])) - cost
            else:
                rec[f"conf_{tm}m_net"] = None  # skipped (no confirmation)
        events.append(rec)
    return events


def split(rows, key, holdout_frac):
    rows = sorted([r for r in rows if r.get(key) is not None], key=lambda r: r["entry_ts_ms"])
    if not rows:
        return {"cal": metrics([]), "hold": metrics([])}
    cut = rows[int(len(rows) * (1.0 - holdout_frac))]["entry_ts_ms"]
    return {"cal": metrics([r[key] for r in rows if r["entry_ts_ms"] < cut]),
            "hold": metrics([r[key] for r in rows if r["entry_ts_ms"] >= cut])}


def render_md(report):
    cfg = report["config"]
    lines = [
        "# S34 Confirmation Entry (wait for reclaim, then fade)",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  {cfg['symbol']} {cfg['side']} deep-V>= {cfg['min_vdepth_bps']}bps "
        f"{int(cfg['threshold']/1000)}K, exit 4h from cross, cost {cfg['cost']}bps RT, bridged Feb-Jun",
        "",
        f"Total deep-V events: {report['event_n']}. Confirmation entry = enter at tau ONLY if price reclaimed the cross level.",
        "",
        "| Variant | taken | cal N | cal mean | cal win | cal sum | hold N | hold mean | hold win | hold sum | hold maxL | pos months |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, d in report["variants"].items():
        s = d["split"]
        cw = lambda m: None if m["win_rate"] is None else r1(m["win_rate"] * 100.0)
        lines.append(f"| {name} | {d['taken']} | {s['cal']['n']} | {s['cal']['mean']} | {cw(s['cal'])} | {s['cal']['sum']} | "
                     f"{s['hold']['n']} | {s['hold']['mean']} | {cw(s['hold'])} | {s['hold']['sum']} | {s['hold']['max_loss']} | {d['pos_months']}/{report['n_months']} |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Confirmation-entry backtest: wait for reclaim then fade (bridged, 4-month).")
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
    months = sorted({r["month"] for r in rows})
    hf = float(args.holdout_frac)
    variants = {}
    for key, name in [("baseline_net", "baseline_no_wait")] + [(f"conf_{tm}m_net", f"confirm_{tm}m") for tm in TAUS_MIN]:
        sp = split(rows, key, hf)
        taken = sum(1 for r in rows if r.get(key) is not None)
        pos = sum(1 for m in months if metrics([r[key] for r in rows if r["month"] == m and r.get(key) is not None])["sum"] > 0)
        variants[name] = {"split": sp, "taken": taken, "pos_months": pos}
    report = {
        "generated_at_utc": utc_now(),
        "config": {"symbol": args.symbol, "side": args.side, "threshold": float(args.threshold),
                   "min_vdepth_bps": float(args.min_vdepth_bps), "cost": r1(cost)},
        "event_n": len(rows), "n_months": len(months), "variants": variants, "rows": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
