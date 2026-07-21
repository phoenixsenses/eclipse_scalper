"""S34 Bridge Backtest.

Pushback: we do NOT need to collect more months -- we can recover months we
already have. The real backtest was capped at ~2 months (Apr+Jun) only because it
required book_ticker fills. But mark_prices spans the full liquidation history
(Feb-Jun). So we BRIDGE: use mark-based fills with a uniform MODELED spread (a
fixed bps haircut) over the WHOLE span, trading some fill realism for 2 extra
months of regime diversity -- the direct antidote to the 2-regime cal/hold
instability.

It runs both structural sides on the deep-V SELL cascade:
  - FADE:        LONG, fixed 4h exit (bet on revert).
  - CONTINUATION: SHORT with a tight mark-path stop (convexity, long the tail).
Cost per round trip = 2*fee + modeled_spread_bps. Reports a PER-MONTH P&L
breakdown (the whole point: is the edge positive across many months, or does it
flip every regime?) plus a chronological holdout over the full span.
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
OUT_JSON = OUT_DIR / "S34_BRIDGE_BACKTEST.json"
OUT_MD = OUT_DIR / "S34_BRIDGE_BACKTEST.md"

HORIZON_SEC = 4 * 3600


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


def cont_path_net(marks, entry_ts, entry_px, sl_bps, tp_bps, cost):
    """SHORT continuation with mark-path stop. SHORT profits when price falls."""
    path = marks.slice_range(int(entry_ts), int(entry_ts) + HORIZON_SEC * 1000)
    sl_price = entry_px * (1.0 + sl_bps / 10_000.0)   # stop above (adverse for short)
    tp_price = entry_px * (1.0 - tp_bps / 10_000.0)   # target below
    exit_px = entry_px
    for ts, px in path:
        if int(ts) <= int(entry_ts):
            continue
        if px >= sl_price:
            exit_px = sl_price
            break
        if tp_bps < 50_000 and px <= tp_price:
            exit_px = tp_price
            break
        exit_px = px
    return signed_return_bps("SHORT", entry_px, exit_px) - cost


def build(conn, symbol, side, threshold, *, bucket_sec, min_gap_sec, accel_window_sec, min_vdepth_bps,
          cost, cont_sl_bps):
    marks = load_mark_index(conn, symbol)
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
        entry_px = float(anc[1])
        ex = marks.at_or_after(entry_ts + HORIZON_SEC * 1000)
        if not ex:
            continue
        fade_net = signed_return_bps("LONG", entry_px, float(ex[1])) - cost
        cont_net = cont_path_net(marks, entry_ts, entry_px, cont_sl_bps, 100_000.0, cost)
        rows.append({"entry_ts_ms": entry_ts, "month": month_of(entry_ts), "fade_net": fade_net, "cont_net": cont_net})
    return rows


def split(rows, key, holdout_frac):
    rows = sorted(rows, key=lambda r: r["entry_ts_ms"])
    if not rows:
        return {"cal": metrics([]), "hold": metrics([])}
    cut = rows[int(len(rows) * (1.0 - holdout_frac))]["entry_ts_ms"]
    return {"cal": metrics([r[key] for r in rows if r["entry_ts_ms"] < cut]),
            "hold": metrics([r[key] for r in rows if r["entry_ts_ms"] >= cut])}


def render_md(report):
    cfg = report["config"]
    lines = [
        "# S34 Bridge Backtest (mark-based + modeled spread, full Feb-Jun span)",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  {cfg['symbol']} {cfg['side']} deep-V>= {cfg['min_vdepth_bps']}bps "
        f"{int(cfg['threshold']/1000)}K 4h, cost {cfg['cost']}bps RT (2*fee+spread), cont SL {cfg['cont_sl_bps']}bps",
        "",
        f"Events: {report['event_n']}  |  months present: {', '.join(report['months_present'])}",
        "",
        "Per-month P&L. The test of the bridge: is the edge positive across MANY months, or does it flip every regime?",
        "",
        "| Month | N | FADE sum | FADE win | FADE maxL | CONT sum | CONT win | CONT maxL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in report["months_present"]:
        d = report["by_month"][m]
        fw = None if d["fade"]["win_rate"] is None else r1(d["fade"]["win_rate"] * 100.0)
        cw = None if d["cont"]["win_rate"] is None else r1(d["cont"]["win_rate"] * 100.0)
        lines.append(f"| {m} | {d['fade']['n']} | {d['fade']['sum']} | {fw} | {d['fade']['max_loss']} | "
                     f"{d['cont']['sum']} | {cw} | {d['cont']['max_loss']} |")
    lines.append("")
    lines.append(f"FADE positive months: {report['fade_pos_months']}/{len(report['months_present'])}  |  "
                 f"CONT positive months: {report['cont_pos_months']}/{len(report['months_present'])}")
    lines.append("")
    lines.append("## Chronological holdout (full span)")
    for name in ("fade", "cont"):
        s = report["split"][name]
        lines.append(f"- {name.upper()}: cal sum={s['cal']['sum']} (N={s['cal']['n']}, win {None if s['cal']['win_rate'] is None else r1(s['cal']['win_rate']*100)}%) | "
                     f"hold sum={s['hold']['sum']} (N={s['hold']['n']}, win {None if s['hold']['win_rate'] is None else r1(s['hold']['win_rate']*100)}%)")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Bridge backtest: mark-based + modeled spread over the full liquidation span.")
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
    p.add_argument("--cont-sl-bps", type=float, default=20.0)
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
                     min_vdepth_bps=float(args.min_vdepth_bps), cost=cost, cont_sl_bps=float(args.cont_sl_bps))
    by_month_rows = defaultdict(list)
    for r in rows:
        by_month_rows[r["month"]].append(r)
    months = sorted(by_month_rows)
    by_month = {m: {"fade": metrics([r["fade_net"] for r in by_month_rows[m]]),
                    "cont": metrics([r["cont_net"] for r in by_month_rows[m]])} for m in months}
    report = {
        "generated_at_utc": utc_now(),
        "config": {"symbol": args.symbol, "side": args.side, "threshold": float(args.threshold),
                   "min_vdepth_bps": float(args.min_vdepth_bps), "cost": r1(cost), "cont_sl_bps": float(args.cont_sl_bps)},
        "event_n": len(rows), "months_present": months,
        "by_month": by_month,
        "fade_pos_months": sum(1 for m in months if by_month[m]["fade"]["sum"] > 0),
        "cont_pos_months": sum(1 for m in months if by_month[m]["cont"]["sum"] > 0),
        "split": {"fade": split(rows, "fade_net", float(args.holdout_frac)), "cont": split(rows, "cont_net", float(args.holdout_frac))},
        "rows": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
