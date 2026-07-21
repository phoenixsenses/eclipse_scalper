"""S34 Reversal Regime Diagnostic.

The fixed-horizon ETH-1M fade backtest showed the edge concentrated in the
holdout (late) period while calibration was flat/negative. That is the signature
of a regime effect, not a stable edge. This bins the per-trade P&L by calendar
month to answer: is the fade edge present across the whole history, or only in a
mean-reverting window? A real edge should be positive in most months on both
sides of the calibration/holdout boundary, not loaded into a few late months.
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

from tools.research_s34_knowable_anchor_continuation import mean, pctile, r1, r3
from tools.research_s34_reversal_backtest import collect_events, trade_pnl

DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_REVERSAL_REGIME_DIAG.json"
OUT_MD = OUT_DIR / "S34_REVERSAL_REGIME_DIAG.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def month_of(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m")


def summ(vals: list[float]) -> dict[str, Any]:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {"n": 0, "sum": 0.0, "median": None, "win_rate": None}
    return {"n": len(vals), "sum": r1(sum(vals)), "median": r1(pctile(vals, 0.5)),
            "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals))}


def run_horizon(conn, symbol, events, horizon_hr, *, fee_bps_side, max_book_staleness_sec, holdout_frac):
    trades = []
    for ev in events:
        t = trade_pnl(conn, symbol, ev, horizon_hr, fee_bps_side=fee_bps_side, max_book_staleness_sec=max_book_staleness_sec)
        if t is not None:
            trades.append(t)
    if not trades:
        return {"horizon_hr": horizon_hr, "filled_n": 0, "months": {}}
    cut_ts = trades[int(len(trades) * (1.0 - holdout_frac))]["entry_ts_ms"]
    by_month: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        by_month[month_of(t["entry_ts_ms"])].append(t["net_bps"])
    months = {}
    pos = neg = 0
    cal_pos = cal_tot = hold_pos = hold_tot = 0
    for m in sorted(by_month):
        s = summ(by_month[m])
        is_hold = any(True for t in trades if month_of(t["entry_ts_ms"]) == m and t["entry_ts_ms"] >= cut_ts)
        s["split"] = "hold" if is_hold else "cal"
        months[m] = s
        if s["sum"] > 0:
            pos += 1
        else:
            neg += 1
        if s["split"] == "cal":
            cal_tot += 1
            cal_pos += 1 if s["sum"] > 0 else 0
        else:
            hold_tot += 1
            hold_pos += 1 if s["sum"] > 0 else 0
    return {
        "horizon_hr": horizon_hr, "filled_n": len(trades),
        "positive_months": pos, "negative_months": neg,
        "cal_months_positive": f"{cal_pos}/{cal_tot}", "hold_months_positive": f"{hold_pos}/{hold_tot}",
        "cut_month": month_of(cut_ts),
        "months": months,
    }


def render_md(report):
    cfg = report["config"]
    lines = [
        "# S34 Reversal Regime Diagnostic",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  {cfg['symbol']} {int(cfg['threshold']/1000)}K fade, "
        f"fee {cfg['fee_bps_side']}bps/side",
        "",
        "Per-month net P&L of the fixed-horizon fade. If the edge is real it is positive across most months on both "
        "sides of the cal/hold cut; if it is a regime fluke it loads into a few late months.",
        "",
    ]
    for h in report["horizons"]:
        if h.get("filled_n", 0) == 0:
            continue
        lines.append(f"## {h['horizon_hr']}h horizon (filled={h['filled_n']}, cut={h['cut_month']})")
        lines.append(f"- positive months: {h['positive_months']} / {h['positive_months']+h['negative_months']}  "
                     f"| calibration months positive: {h['cal_months_positive']}  | holdout months positive: {h['hold_months_positive']}")
        lines.append("")
        lines.append("| Month | split | N | sum bps | median | win% |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for m, s in h["months"].items():
            wr = None if s["win_rate"] is None else r1(s["win_rate"] * 100.0)
            lines.append(f"| {m} | {s['split']} | {s['n']} | {s['sum']} | {s['median']} | {wr} |")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Month-by-month regime diagnostic of the liquidation-fade backtest.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--threshold", type=float, default=1_000_000.0)
    p.add_argument("--bucket-sec", type=int, default=300)
    p.add_argument("--min-gap-sec", type=int, default=900)
    p.add_argument("--accel-window-sec", type=int, default=30)
    p.add_argument("--fee-bps-side", type=float, default=3.05)
    p.add_argument("--max-book-staleness-sec", type=int, default=10)
    p.add_argument("--holdout-frac", type=float, default=0.30)
    p.add_argument("--horizons-hr", default="4,24")
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    horizons = [int(x) for x in str(args.horizons_hr).split(",") if x.strip()]
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        events = collect_events(conn, args.symbol, float(args.threshold),
                                bucket_sec=int(args.bucket_sec), min_gap_sec=int(args.min_gap_sec),
                                accel_window_sec=int(args.accel_window_sec))
        results = [run_horizon(conn, args.symbol, events, h, fee_bps_side=float(args.fee_bps_side),
                               max_book_staleness_sec=int(args.max_book_staleness_sec), holdout_frac=float(args.holdout_frac))
                   for h in horizons]
    report = {"generated_at_utc": utc_now(),
              "config": {"symbol": args.symbol, "threshold": float(args.threshold), "fee_bps_side": float(args.fee_bps_side)},
              "horizons": results}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
