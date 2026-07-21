"""S34 Synchronization Gate.

Butterfly showed the revert-vs-runaway outcome is NOT in ETH's own seed. The
user's connection vision suggests the separator is in the EDGES, not the node:
an idiosyncratic ETH flush (BTC/SOL calm) is a local capitulation that reverts;
a SYNCHRONIZED cascade (BTC and SOL also being sell-liquidated at the same time)
is a real market-wide deleveraging that continues -- the -410 tail.

For each ETH deep-V SELL fade event we measure, knowable at entry, the CONCURRENT
cross-asset sell-liquidation in the prior 10 min: BTC and SOL sell-liq notional,
and BTC's concurrent return. We split events into IDIOSYNCRATIC (low concurrent
cross-asset liquidation) vs SYNCHRONIZED (high), and compare each bucket's P&L,
win rate, max loss (the tail), T3R, per-month, and chronological holdout. If the
idiosyncratic bucket is positive with a small tail and the synchronized bucket
carries the runaways, synchronization is the tail-separator we have been hunting.
"""

from __future__ import annotations

import argparse
import bisect
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
OUT_JSON = OUT_DIR / "S34_SYNCHRONIZATION_GATE.json"
OUT_MD = OUT_DIR / "S34_SYNCHRONIZATION_GATE.md"

HORIZON_SEC = 4 * 3600
SYNC_WINDOW_SEC = 600  # 10 min concurrent window


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def month_of(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m")


def metrics(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return {"n": 0, "sum": 0.0, "mean": None, "win_rate": None, "max_loss": None, "t3r": None}
    s = sorted(vals, reverse=True)
    return {"n": len(vals), "sum": r1(sum(vals)), "mean": r1(mean(vals)),
            "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)), "max_loss": r1(min(vals)),
            "t3r": r1(sum(s[3:])) if len(s) > 3 else r1(sum(s))}


def split(rows, holdout_frac):
    rows = sorted(rows, key=lambda r: r["entry_ts_ms"])
    if not rows:
        return {"cal": metrics([]), "hold": metrics([])}
    cut = rows[int(len(rows) * (1.0 - holdout_frac))]["entry_ts_ms"]
    return {"cal": metrics([r["net"] for r in rows if r["entry_ts_ms"] < cut]),
            "hold": metrics([r["net"] for r in rows if r["entry_ts_ms"] >= cut])}


def med(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return r1(pctile(vals, 0.5)) if vals else None


def window_liq(liq_ts, liq_rows, a, b):
    lo = bisect.bisect_right(liq_ts, int(a))
    hi = bisect.bisect_right(liq_ts, int(b))
    return sum(float(liq_rows[i]["notional"]) for i in range(lo, hi))


def build(conn, threshold, *, bucket_sec, min_gap_sec, accel_window_sec, min_vdepth_bps, cost):
    eth = load_mark_index(conn, "ETHUSDT")
    btc_m = load_mark_index(conn, "BTCUSDT")
    eth_liq = load_liquidations(conn, "ETHUSDT", "SELL", None, None)
    btc_liq = load_liquidations(conn, "BTCUSDT", "SELL", None, None)
    sol_liq = load_liquidations(conn, "SOLUSDT", "SELL", None, None)
    btc_ts = [int(r["ts_ms"]) for r in btc_liq]
    sol_ts = [int(r["ts_ms"]) for r in sol_liq]
    anchors = reconstruct_anchors(eth_liq, bucket_sec=bucket_sec, min_gap_sec=min_gap_sec,
                                  thresholds=(float(threshold),), accel_window_sec=accel_window_sec)
    rows = []
    for a in anchors:
        if float(a.threshold_usd) != float(threshold):
            continue
        start = eth.at_or_after(int(a.first_ts_ms))
        anc = eth.at_or_after(int(a.anchor_ts_ms))
        if not start or not anc or float(start[1]) <= 0:
            continue
        depth = (float(start[1]) - float(anc[1])) / float(start[1]) * 10_000.0
        if depth < float(min_vdepth_bps):
            continue
        t = int(a.anchor_ts_ms)
        ex = eth.at_or_after(t + HORIZON_SEC * 1000)
        if not ex:
            continue
        net = signed_return_bps("LONG", float(anc[1]), float(ex[1])) - cost
        win_a = t - SYNC_WINDOW_SEC * 1000
        btc_liq_k = window_liq(btc_ts, btc_liq, win_a, t) / 1000.0
        sol_liq_k = window_liq(sol_ts, sol_liq, win_a, t) / 1000.0
        btc_ret = btc_m.ret_bps(win_a, t)
        rows.append({"entry_ts_ms": t, "month": month_of(t), "net": net,
                     "btc_liq_k": r1(btc_liq_k), "sol_liq_k": r1(sol_liq_k),
                     "market_concurrent_k": r1(btc_liq_k + sol_liq_k), "btc_ret_10m": (r1(btc_ret) if btc_ret is not None else None)})
    return rows


def render_md(report):
    cfg = report["config"]
    lines = [
        "# S34 Synchronization Gate (is the tail a market-wide synchronized cascade?)",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  ETH SELL deep-V>= {cfg['min_vdepth_bps']}bps {int(cfg['threshold']/1000)}K 4h fade, "
        f"cost {cfg['cost']}bps RT, sync window {SYNC_WINDOW_SEC//60}m, sync threshold {cfg['sync_threshold_k']}K cross-asset sell-liq",
        "",
        f"Events: {report['event_n']}  (idiosyncratic {report['n_idio']} / synchronized {report['n_sync']})",
        "",
        "## Winner vs runaway: concurrent cross-asset sell-liq (median)",
        f"- market_concurrent_k: winners={report['sep']['win']} vs runaways={report['sep']['runaway']}  ({report['sep']['note']})",
        f"- btc_ret_10m: winners={report['sep']['win_btcret']} vs runaways={report['sep']['runaway_btcret']}",
        "",
        "## Buckets",
        "",
        "| Bucket | N | sum | mean | win | max_loss | T3R | cal N | cal sum | cal win | hold N | hold sum | hold win | cal&hold + | " + " | ".join(report["months"]) + " |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | " + " | ".join("---:" for _ in report["months"]) + " |",
    ]
    for name, d in report["buckets"].items():
        m, s = d["overall"], d["split"]
        cw = lambda x: None if x["win_rate"] is None else r1(x["win_rate"] * 100.0)
        both = "YES" if ((s["cal"]["sum"] or 0) > 0 and (s["hold"]["sum"] or 0) > 0 and s["cal"]["n"] >= 15 and s["hold"]["n"] >= 8) else ""
        mm = " | ".join(str(d["by_month"].get(mo, 0.0)) for mo in report["months"])
        lines.append(f"| {name} | {m['n']} | {m['sum']} | {m['mean']} | {cw(m)} | {m['max_loss']} | {m['t3r']} | "
                     f"{s['cal']['n']} | {s['cal']['sum']} | {cw(s['cal'])} | {s['hold']['n']} | {s['hold']['sum']} | {cw(s['hold'])} | {both} | {mm} |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Synchronization (idiosyncratic vs market-wide) gate for the ETH deep-V fade.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--threshold", type=float, default=200_000.0)
    p.add_argument("--min-vdepth-bps", type=float, default=28.0)
    p.add_argument("--bucket-sec", type=int, default=300)
    p.add_argument("--min-gap-sec", type=int, default=900)
    p.add_argument("--accel-window-sec", type=int, default=30)
    p.add_argument("--fee-bps-side", type=float, default=3.05)
    p.add_argument("--modeled-spread-bps", type=float, default=2.0)
    p.add_argument("--sync-threshold-k", type=float, default=200.0)
    p.add_argument("--holdout-frac", type=float, default=0.30)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cost = 2.0 * float(args.fee_bps_side) + float(args.modeled_spread_bps)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        rows = build(conn, float(args.threshold), bucket_sec=int(args.bucket_sec), min_gap_sec=int(args.min_gap_sec),
                     accel_window_sec=int(args.accel_window_sec), min_vdepth_bps=float(args.min_vdepth_bps), cost=cost)
    thr = float(args.sync_threshold_k)
    idio = [r for r in rows if r["market_concurrent_k"] < thr]
    sync = [r for r in rows if r["market_concurrent_k"] >= thr]
    months = sorted({r["month"] for r in rows})
    win = [r for r in rows if r["net"] > 0]
    runaway = [r for r in rows if r["net"] < -100.0]
    wm, rm = med([r["market_concurrent_k"] for r in win]), med([r["market_concurrent_k"] for r in runaway])
    buckets = {}
    for name, sub in (("idiosyncratic(<thr)", idio), ("synchronized(>=thr)", sync), ("all", rows)):
        buckets[name] = {"overall": metrics([r["net"] for r in sub]), "split": split(sub, float(args.holdout_frac)),
                         "by_month": {mo: metrics([r["net"] for r in sub if r["month"] == mo])["sum"] for mo in months}}
    report = {
        "generated_at_utc": utc_now(),
        "config": {"threshold": float(args.threshold), "min_vdepth_bps": float(args.min_vdepth_bps),
                   "cost": r1(cost), "sync_threshold_k": thr},
        "event_n": len(rows), "n_idio": len(idio), "n_sync": len(sync), "months": months,
        "sep": {"win": wm, "runaway": rm, "note": ("yes" if (wm is not None and rm is not None and abs(wm - rm) > 0.25 * (abs(wm) + abs(rm) + 1e-9)) else "weak"),
                "win_btcret": med([r["btc_ret_10m"] for r in win]), "runaway_btcret": med([r["btc_ret_10m"] for r in runaway])},
        "buckets": buckets, "rows": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
