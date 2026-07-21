"""S34 Butterfly Seed.

Butterfly effect, grounded: does a small knowable difference in the cascade's
SEED (its first seconds / shape at the threshold cross) determine the divergent
outcome -- a small revert (winner) vs a -300 bps runaway (the tail)?

For each deep-V SELL fade event we take the outcome (baseline 4h taker net) and
the knowable-at-entry seed features carried by the anchor: acceleration (still
building vs exhausting), elapsed (fast violent flush vs slow grind), single-liq
dominance (one whale vs distributed), liq count, overshoot depth, intensity,
and BTC state. We compare winners vs runaways on each feature, then test the most
promising ENTRY filters (no post-entry whipsaw) for whether they cut the -330
tail while keeping T3R/sum positive on calibration AND holdout.

N is small (deep-V SELL, ~54 events, ~8 runaways) -- separations are suggestive,
not conclusive; report honestly.
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
OUT_JSON = OUT_DIR / "S34_BUTTERFLY_SEED.json"
OUT_MD = OUT_DIR / "S34_BUTTERFLY_SEED.md"

HORIZON_SEC = 4 * 3600
SEED_FEATURES = ("accel", "elapsed_sec", "dominance_pct", "liq_count", "depth_bps", "intensity_k_per_sec", "btc_ret_bps")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def metrics(vals: list[float]) -> dict[str, Any]:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {"n": 0, "sum": 0.0, "median": None, "win_rate": None, "max_loss": None, "t3r": None}
    s = sorted(vals, reverse=True)
    return {"n": len(vals), "sum": r1(sum(vals)), "median": r1(pctile(vals, 0.5)),
            "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)),
            "max_loss": r1(min(vals)), "t3r": r1(sum(s[3:])) if len(s) > 3 else r1(sum(s))}


def split_metrics(rows: list[dict[str, Any]], holdout_frac: float) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: r["entry_ts_ms"])
    if not rows:
        return {"cal": metrics([]), "hold": metrics([])}
    cut = rows[int(len(rows) * (1.0 - holdout_frac))]["entry_ts_ms"]
    return {"cal": metrics([r["net"] for r in rows if r["entry_ts_ms"] < cut]),
            "hold": metrics([r["net"] for r in rows if r["entry_ts_ms"] >= cut])}


def med(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return r1(pctile(vals, 0.5)) if vals else None


def build(conn, symbol, side, threshold, *, bucket_sec, min_gap_sec, accel_window_sec, min_vdepth_bps,
          fee_bps_side, max_book_staleness_sec):
    fade_dir = "LONG" if side == "SELL" else "SHORT"
    marks = load_mark_index(conn, symbol)
    btc = load_mark_index(conn, "BTCUSDT")
    liqs = load_liquidations(conn, symbol, side, None, None)
    anchors = reconstruct_anchors(liqs, bucket_sec=bucket_sec, min_gap_sec=min_gap_sec,
                                  thresholds=(float(threshold),), accel_window_sec=accel_window_sec)
    rows = []
    for a in anchors:
        if float(a.threshold_usd) != float(threshold):
            continue
        start = marks.at_or_after(int(a.first_ts_ms))
        anc = marks.at_or_after(int(a.anchor_ts_ms))
        if not start or not anc or float(start[1]) <= 0:
            continue
        depth = ((float(start[1]) - float(anc[1])) if side == "SELL" else (float(anc[1]) - float(start[1]))) / float(start[1]) * 10_000.0
        if depth < float(min_vdepth_bps):
            continue
        entry_ts = int(a.anchor_ts_ms)
        eb = book_at(conn, symbol, entry_ts, max_book_staleness_sec)
        xb = book_at(conn, symbol, entry_ts + HORIZON_SEC * 1000, max_book_staleness_sec)
        if not eb or not xb:
            continue
        entry_px = eb.ask if fade_dir == "LONG" else eb.bid
        exit_px = xb.bid if fade_dir == "LONG" else xb.ask
        net = signed_return_bps(fade_dir, float(entry_px), float(exit_px)) - 2.0 * float(fee_bps_side)
        btc_ret = btc.ret_bps(entry_ts - 900_000, entry_ts)
        rows.append({
            "entry_ts_ms": entry_ts, "net": net,
            "accel": float(a.running_accel),
            "accel_bucket": a.acceleration_bucket,
            "elapsed_sec": float(a.elapsed_since_first_sec),
            "dominance_pct": float(a.running_single_liq_dominance),
            "liq_count": int(a.running_liq_count),
            "depth_bps": float(depth),
            "intensity_k_per_sec": float(a.running_rate) / 1000.0,
            "btc_ret_bps": (float(btc_ret) if btc_ret is not None else None),
        })
    return rows


def render_md(report):
    cfg = report["config"]
    lines = [
        "# S34 Butterfly Seed (does the cascade seed predict revert vs runaway?)",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  {cfg['symbol']} {cfg['side']} deep-V>= {cfg['min_vdepth_bps']}bps {int(cfg['threshold']/1000)}K 4h",
        "",
        f"Events: {report['event_n']}  winners(net>0): {report['n_win']}  losers(net<0): {report['n_lose']}  runaways(net<-100): {report['n_runaway']}",
        "",
        "## Seed feature: winners vs runaways (median)",
        "",
        "| Feature | winners med | runaways med | separates? |",
        "| --- | ---: | ---: | --- |",
    ]
    for f, d in report["separation"].items():
        lines.append(f"| {f} | {d['winner_med']} | {d['runaway_med']} | {d['note']} |")
    lines.append("")
    lines.append("## Entry-filter tail test (cut runaways without whipsaw)")
    lines.append("")
    lines.append("| Filter | cal N | cal sum | cal med | cal win% | cal max_loss | cal T3R | hold N | hold sum | hold med | hold max_loss | hold T3R |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, sm in report["filters"].items():
        c, h = sm["cal"], sm["hold"]
        cw = None if c["win_rate"] is None else r1(c["win_rate"] * 100.0)
        lines.append(f"| {name} | {c['n']} | {c['sum']} | {c['median']} | {cw} | {c['max_loss']} | {c['t3r']} | "
                     f"{h['n']} | {h['sum']} | {h['median']} | {h['max_loss']} | {h['t3r']} |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Butterfly-seed discrimination of revert vs runaway for the deep-V fade.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--side", default="SELL")
    p.add_argument("--threshold", type=float, default=200_000.0)
    p.add_argument("--min-vdepth-bps", type=float, default=28.0)
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
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        rows = build(conn, args.symbol, args.side, float(args.threshold),
                     bucket_sec=int(args.bucket_sec), min_gap_sec=int(args.min_gap_sec),
                     accel_window_sec=int(args.accel_window_sec), min_vdepth_bps=float(args.min_vdepth_bps),
                     fee_bps_side=float(args.fee_bps_side), max_book_staleness_sec=int(args.max_book_staleness_sec))
    win = [r for r in rows if r["net"] > 0]
    runaway = [r for r in rows if r["net"] < -100.0]
    separation = {}
    for f in SEED_FEATURES:
        wm = med([r[f] for r in win])
        rm = med([r[f] for r in runaway])
        note = ""
        if wm is not None and rm is not None:
            note = "yes" if abs(wm - rm) > 0.25 * (abs(wm) + abs(rm) + 1e-9) else "weak"
        separation[f] = {"winner_med": wm, "runaway_med": rm, "note": note}

    hf = float(args.holdout_frac)
    em = sorted([r["accel"] for r in rows])
    accel_med = pctile(em, 0.5) if em else 0.0
    elapsed_med = pctile(sorted([r["elapsed_sec"] for r in rows]), 0.5) if rows else 0.0
    filters = {
        "baseline_all": split_metrics(rows, hf),
        "decelerating(accel<0)": split_metrics([r for r in rows if r["accel"] < 0], hf),
        "decel_bucket": split_metrics([r for r in rows if r["accel_bucket"] == "decelerating"], hf),
        "high_dominance>=80": split_metrics([r for r in rows if r["dominance_pct"] >= 80.0], hf),
        "slow_build(elapsed>=med)": split_metrics([r for r in rows if r["elapsed_sec"] >= elapsed_med], hf),
        "fast_build(elapsed<med)": split_metrics([r for r in rows if r["elapsed_sec"] < elapsed_med], hf),
    }
    report = {
        "generated_at_utc": utc_now(),
        "config": {"symbol": args.symbol, "side": args.side, "threshold": float(args.threshold),
                   "min_vdepth_bps": float(args.min_vdepth_bps), "fee_bps_side": float(args.fee_bps_side)},
        "event_n": len(rows), "n_win": len(win), "n_lose": sum(1 for r in rows if r["net"] < 0), "n_runaway": len(runaway),
        "separation": separation, "filters": filters,
        "accel_median": r1(accel_med), "elapsed_median": r1(elapsed_med), "rows": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
