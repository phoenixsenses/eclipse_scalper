"""S34 Regime Gate.

Across ETH/BTC/SOL the deep-V fade lost in April and won in June -> the apparent
edge is a regime effect. This asks: is there a KNOWABLE-at-entry regime variable
that distinguishes the winning regime from the losing one? If yes, condition on
it (a regime gate) and the edge should turn positive in BOTH April and June -- not
just June. If no knowable variable separates them (only the calendar does), the
strategy is regime-luck.

For each deep-V SELL fade event (ETH, bridged Feb-Jun, mark-based + modeled
spread) we compute knowable regime features at entry:
  - eth_rv24_bps : ETH realized vol (std of hourly returns over the prior 24h)
  - btc_abs24_bps: |BTC return over the prior 24h| (trending vs range)
  - btc_ret24_bps: signed BTC prior-24h return
  - eth_day_trend_bps: ETH return from UTC day start to entry
Then: per-month medians (is June a different regime?), winner-vs-loser medians,
and median-split GATES reporting each half's per-month P&L (does one half win in
BOTH Apr and Jun?).
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
OUT_JSON = OUT_DIR / "S34_REGIME_GATE.json"
OUT_MD = OUT_DIR / "S34_REGIME_GATE.md"

HORIZON_SEC = 4 * 3600
FEATURES = ("eth_rv24_bps", "btc_abs24_bps", "btc_ret24_bps", "eth_day_trend_bps")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def month_of(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m")


def day_start_ms(ts_ms: int) -> int:
    dt = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc)
    return int(datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).timestamp() * 1000)


def realized_vol_bps(marks, end_ts, hours=24):
    pts = []
    for h in range(hours + 1):
        mk = marks.at_or_before(end_ts - (hours - h) * 3600 * 1000)
        if mk:
            pts.append(float(mk[1]))
    rets = [math.log(pts[i] / pts[i - 1]) for i in range(1, len(pts)) if pts[i - 1] > 0 and pts[i] > 0]
    if len(rets) < 3:
        return None
    mu = sum(rets) / len(rets)
    var = sum((x - mu) ** 2 for x in rets) / (len(rets) - 1)
    return math.sqrt(var) * 10_000.0


def metrics(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return {"n": 0, "sum": 0.0, "median": None, "win_rate": None}
    return {"n": len(vals), "sum": r1(sum(vals)), "median": r1(pctile(vals, 0.5)),
            "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals))}


def med(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return r1(pctile(vals, 0.5)) if vals else None


def build(conn, symbol, side, threshold, *, bucket_sec, min_gap_sec, accel_window_sec, min_vdepth_bps, cost):
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
        depth = (float(start[1]) - float(anc[1])) / float(start[1]) * 10_000.0 if side == "SELL" else (float(anc[1]) - float(start[1])) / float(start[1]) * 10_000.0
        if depth < float(min_vdepth_bps):
            continue
        entry_ts = int(a.anchor_ts_ms)
        entry_px = float(anc[1])
        ex = marks.at_or_after(entry_ts + HORIZON_SEC * 1000)
        if not ex:
            continue
        net = signed_return_bps("LONG", entry_px, float(ex[1])) - cost
        day0 = marks.at_or_after(day_start_ms(entry_ts))
        eth_day = ((entry_px - float(day0[1])) / float(day0[1]) * 10_000.0) if day0 and float(day0[1]) > 0 else None
        btc24 = btc.ret_bps(entry_ts - 24 * 3600 * 1000, entry_ts)
        rows.append({
            "entry_ts_ms": entry_ts, "month": month_of(entry_ts), "net": net,
            "eth_rv24_bps": realized_vol_bps(marks, entry_ts),
            "btc_abs24_bps": (abs(btc24) if btc24 is not None else None),
            "btc_ret24_bps": (r1(btc24) if btc24 is not None else None),
            "eth_day_trend_bps": (r1(eth_day) if eth_day is not None else None),
        })
    return rows


def render_md(report):
    cfg = report["config"]
    lines = [
        "# S34 Regime Gate (is the June-vs-April split knowable?)",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  {cfg['symbol']} {cfg['side']} deep-V>= {cfg['min_vdepth_bps']}bps "
        f"{int(cfg['threshold']/1000)}K 4h fade, cost {cfg['cost']}bps RT, bridged",
        "",
        f"Events: {report['event_n']}",
        "",
        "## Regime feature by month (does June differ from April?)",
        "",
        "| Month | N | net sum | net win | " + " | ".join(FEATURES) + " |",
        "| --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in FEATURES) + " |",
    ]
    for m, d in report["by_month"].items():
        lines.append(f"| {m} | {d['n']} | {d['net']['sum']} | {None if d['net']['win_rate'] is None else r1(d['net']['win_rate']*100)} | "
                     + " | ".join(str(d["feat"][f]) for f in FEATURES) + " |")
    lines.append("")
    lines.append("## Winner vs loser (median regime feature)")
    lines.append("")
    lines.append("| Feature | winners | losers | separates? |")
    lines.append("| --- | ---: | ---: | --- |")
    for f, d in report["winloss"].items():
        lines.append(f"| {f} | {d['winner']} | {d['loser']} | {d['note']} |")
    lines.append("")
    lines.append("## Median-split gates (per-month P&L of the favorable half)")
    lines.append("")
    lines.append("| Gate | half | N | sum | win | " + " | ".join(report["months"]) + " | Apr&Jun both+ |")
    lines.append("| --- | --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in report["months"]) + " | --- |")
    for g in report["gates"]:
        mm = " | ".join(str(g["by_month"].get(m, 0.0)) for m in report["months"])
        lines.append(f"| {g['feature']} | {g['half']} | {g['n']} | {g['sum']} | {g['win']} | {mm} | {'YES' if g['apr_jun_both_pos'] else ''} |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Regime-gate analysis of the bridged deep-V fade (is June-vs-April knowable?).")
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
    by_month = {}
    for m in months:
        mr = [r for r in rows if r["month"] == m]
        by_month[m] = {"n": len(mr), "net": metrics([r["net"] for r in mr]),
                       "feat": {f: med([r[f] for r in mr]) for f in FEATURES}}
    win = [r for r in rows if r["net"] > 0]
    lose = [r for r in rows if r["net"] <= 0]
    winloss = {}
    for f in FEATURES:
        wm, lm = med([r[f] for r in win]), med([r[f] for r in lose])
        note = "yes" if (wm is not None and lm is not None and abs(wm - lm) > 0.25 * (abs(wm) + abs(lm) + 1e-9)) else "weak"
        winloss[f] = {"winner": wm, "loser": lm, "note": note}
    gates = []
    for f in FEATURES:
        vals = sorted([r[f] for r in rows if r[f] is not None])
        if len(vals) < 10:
            continue
        cutv = pctile(vals, 0.5)
        for half, pred in (("high", lambda v: v is not None and v >= cutv), ("low", lambda v: v is not None and v < cutv)):
            sub = [r for r in rows if pred(r[f])]
            mm = {m: metrics([r["net"] for r in sub if r["month"] == m])["sum"] for m in months}
            apr_jun = (mm.get("2026-04", 0.0) > 0 and mm.get("2026-06", 0.0) > 0)
            mt = metrics([r["net"] for r in sub])
            gates.append({"feature": f, "half": half, "n": mt["n"], "sum": mt["sum"],
                          "win": None if mt["win_rate"] is None else r1(mt["win_rate"] * 100.0),
                          "by_month": {m: r1(mm[m]) for m in months}, "apr_jun_both_pos": apr_jun})
    report = {
        "generated_at_utc": utc_now(),
        "config": {"symbol": args.symbol, "side": args.side, "threshold": float(args.threshold),
                   "min_vdepth_bps": float(args.min_vdepth_bps), "cost": r1(cost)},
        "event_n": len(rows), "months": months, "by_month": by_month, "winloss": winloss, "gates": gates, "rows": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
