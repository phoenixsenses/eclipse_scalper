"""S34 Regime Gate Validation.

The regime-gate scan suggested a mechanistically-sensible filter: only fade a
SELL cascade when the asset is already DOWN (a capitulation flush that bounces),
not when it is up (the start of a real selloff that continues). But that gate was
in-sample selected. This validates it properly:
  - fixed binary gates (own day-trend < 0; BTC prior-24h return < 0; both),
  - across ETH / BTC / SOL (does the mechanism generalize, or is it ETH-luck?),
  - with a chronological holdout over the bridged 4-month span (is it OOS-stable?),
  - per-month P&L (does it fix the losing regime?).
Bridged mark-based + modeled spread. A real gate shows cal>0 AND hold>0 on
MULTIPLE assets, not just one.
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
OUT_JSON = OUT_DIR / "S34_REGIME_GATE_VALIDATE.json"
OUT_MD = OUT_DIR / "S34_REGIME_GATE_VALIDATE.md"

HORIZON_SEC = 4 * 3600
SYMBOLS = ("ETHUSDT", "BTCUSDT", "SOLUSDT")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def month_of(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m")


def day_start_ms(ts_ms):
    dt = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc)
    return int(datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).timestamp() * 1000)


def metrics(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return {"n": 0, "sum": 0.0, "win_rate": None, "median": None}
    return {"n": len(vals), "sum": r1(sum(vals)), "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)),
            "median": r1(pctile(vals, 0.5))}


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
        day_trend = ((entry_px - float(day0[1])) / float(day0[1]) * 10_000.0) if day0 and float(day0[1]) > 0 else None
        btc24 = btc.ret_bps(entry_ts - 24 * 3600 * 1000, entry_ts)
        rows.append({"entry_ts_ms": entry_ts, "month": month_of(entry_ts), "net": net,
                     "day_trend": day_trend, "btc_ret24": btc24})
    return rows


GATES = {
    "ungated": lambda r: True,
    "asset_down(daytrend<0)": lambda r: r["day_trend"] is not None and r["day_trend"] < 0,
    "btc_falling(btc24<0)": lambda r: r["btc_ret24"] is not None and r["btc_ret24"] < 0,
    "double(both<0)": lambda r: (r["day_trend"] is not None and r["day_trend"] < 0 and r["btc_ret24"] is not None and r["btc_ret24"] < 0),
}


def evaluate(rows, holdout_frac):
    months = sorted({r["month"] for r in rows})
    out = {}
    for name, pred in GATES.items():
        sub = sorted([r for r in rows if pred(r)], key=lambda r: r["entry_ts_ms"])
        if not sub:
            out[name] = {"all": metrics([]), "cal": metrics([]), "hold": metrics([]), "by_month": {}}
            continue
        cut = sub[int(len(sub) * (1.0 - holdout_frac))]["entry_ts_ms"]
        out[name] = {
            "all": metrics([r["net"] for r in sub]),
            "cal": metrics([r["net"] for r in sub if r["entry_ts_ms"] < cut]),
            "hold": metrics([r["net"] for r in sub if r["entry_ts_ms"] >= cut]),
            "by_month": {m: metrics([r["net"] for r in sub if r["month"] == m])["sum"] for m in months},
        }
    return out, months


def render_md(report):
    cfg = report["config"]
    lines = [
        "# S34 Regime Gate Validation (cross-asset, chronological holdout)",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  SELL deep-V>= {cfg['min_vdepth_bps']}bps {int(cfg['threshold']/1000)}K 4h fade, "
        f"cost {cfg['cost']}bps RT, bridged, modeled spread",
        "",
        "Gate = enter only when the regime condition holds at entry. A real gate: cal>0 AND hold>0 on MULTIPLE assets.",
        "",
    ]
    for sym in report["symbols"]:
        lines.append(f"## {sym['symbol']} (months: {', '.join(sym['months'])})")
        lines.append("")
        lines.append("| Gate | all N | all sum | cal N | cal sum | cal win | hold N | hold sum | hold win | cal&hold + | " + " | ".join(sym["months"]) + " |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | " + " | ".join("---:" for _ in sym["months"]) + " |")
        for name, d in sym["gates"].items():
            both = "YES" if ((d["cal"]["sum"] or 0) > 0 and (d["hold"]["sum"] or 0) > 0 and d["cal"]["n"] >= 15 and d["hold"]["n"] >= 8) else ""
            cw = lambda m: None if m["win_rate"] is None else r1(m["win_rate"] * 100.0)
            mm = " | ".join(str(d["by_month"].get(m, 0.0)) for m in sym["months"])
            lines.append(f"| {name} | {d['all']['n']} | {d['all']['sum']} | {d['cal']['n']} | {d['cal']['sum']} | {cw(d['cal'])} | "
                         f"{d['hold']['n']} | {d['hold']['sum']} | {cw(d['hold'])} | {both} | {mm} |")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Validate the regime gate cross-asset with a chronological holdout.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--side", default="SELL")
    p.add_argument("--threshold", type=float, default=200_000.0)
    p.add_argument("--min-vdepth-bps", type=float, default=28.0)
    p.add_argument("--bucket-sec", type=int, default=300)
    p.add_argument("--min-gap-sec", type=int, default=900)
    p.add_argument("--accel-window-sec", type=int, default=30)
    p.add_argument("--fee-bps-side", type=float, default=3.05)
    p.add_argument("--modeled-spread-bps", type=float, default=2.0)
    p.add_argument("--holdout-frac", type=float, default=0.30)
    p.add_argument("--symbols", default=",".join(SYMBOLS))
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cost = 2.0 * float(args.fee_bps_side) + float(args.modeled_spread_bps)
    syms = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    out = []
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        for sym in syms:
            rows = build(conn, sym, args.side, float(args.threshold), bucket_sec=int(args.bucket_sec),
                         min_gap_sec=int(args.min_gap_sec), accel_window_sec=int(args.accel_window_sec),
                         min_vdepth_bps=float(args.min_vdepth_bps), cost=cost)
            gates, months = evaluate(rows, float(args.holdout_frac))
            out.append({"symbol": sym, "event_n": len(rows), "months": months, "gates": gates})
    report = {"generated_at_utc": utc_now(),
              "config": {"side": args.side, "threshold": float(args.threshold), "min_vdepth_bps": float(args.min_vdepth_bps), "cost": r1(cost)},
              "symbols": out}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
