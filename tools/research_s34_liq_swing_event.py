"""S34 Liquidation Swing Event.

Reframes the liquidation cascade not as a scalp but as the CAUSE/marker of a
larger swing-scale move (hours), in either direction: an exhaustion spike that
reverses, or a capitulation that keeps running. Scalp horizons (<=1h) were dead;
this looks at 1h..48h, where the spread/timing cost that killed the scalp is
negligible.

Beta control (critical): "price went up 24h after a BUY-liq" could just be market
drift. So we measure the RAW (unsigned) forward return after BUY-side and
SELL-side liquidation events separately, and the signal is their DIFFERENCE:

    signal_diff = median(raw | BUY-liq) - median(raw | SELL-liq)

Market beta hits both sides equally and cancels in the difference. signal_diff > 0
=> continuation (spike direction persists); < 0 => reversal (the user's
"spike -> reversal"); ~0 => only beta, no directional event. We also report the
combined continuation-signed net return (LONG after BUY-liq, SHORT after SELL-liq),
which is the tradeable, beta-cancelling P&L, with a chronological holdout split.
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
OUT_JSON = OUT_DIR / "S34_LIQ_SWING_EVENT.json"
OUT_MD = OUT_DIR / "S34_LIQ_SWING_EVENT.md"

SYMBOLS = ("ETHUSDT", "SOLUSDT", "BTCUSDT")
THRESHOLDS_USD = (200_000.0, 500_000.0, 1_000_000.0)
HORIZONS_SEC = (3600, 7200, 14400, 28800, 43200, 86400, 172800)
HORIZON_LABEL = {3600: "1h", 7200: "2h", 14400: "4h", 28800: "8h", 43200: "12h", 86400: "24h", 172800: "48h"}
SIDES = (("BUY", "LONG"), ("SELL", "SHORT"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def med(vals: list[float]) -> float | None:
    vals = [v for v in vals if math.isfinite(v)]
    return r1(pctile(vals, 0.5)) if vals else None


def win(vals: list[float]) -> float | None:
    vals = [v for v in vals if math.isfinite(v)]
    return r3(sum(1 for v in vals if v > 0) / len(vals)) if vals else None


def raw_return_bps(marks, ts_ms: int, h_sec: int) -> float | None:
    a = marks.at_or_after(ts_ms)
    b = marks.at_or_after(ts_ms + h_sec * 1000)
    if not a or not b or float(a[1]) <= 0:
        return None
    return (float(b[1]) - float(a[1])) / float(a[1]) * 10_000.0


def eval_symbol(conn, symbol, *, bucket_sec, min_gap_sec, accel_window_sec, holdout_frac, cost_bps_rt) -> dict[str, Any]:
    marks = load_mark_index(conn, symbol)
    # anchors per side, grouped by threshold
    events: dict[str, dict[float, list[int]]] = {"BUY": {}, "SELL": {}}
    for side, _ in SIDES:
        liqs = load_liquidations(conn, symbol, side, None, None)
        anchors = reconstruct_anchors(
            liqs, bucket_sec=bucket_sec, min_gap_sec=min_gap_sec,
            thresholds=THRESHOLDS_USD, accel_window_sec=accel_window_sec,
        )
        for th in THRESHOLDS_USD:
            events[side][th] = sorted(int(a.anchor_ts_ms) for a in anchors if float(a.threshold_usd) == th)

    thresholds_out = {}
    for th in THRESHOLDS_USD:
        buy_ts = events["BUY"][th]
        sell_ts = events["SELL"][th]
        if not buy_ts or not sell_ts:
            continue
        # chronological holdout cut on the combined event timeline
        all_ts = sorted(buy_ts + sell_ts)
        cut = all_ts[int(len(all_ts) * (1.0 - holdout_frac))] if len(all_ts) > 1 else all_ts[-1] + 1
        horizons_out = {}
        for h in HORIZONS_SEC:
            buy_raw = [raw_return_bps(marks, ts, h) for ts in buy_ts]
            sell_raw = [raw_return_bps(marks, ts, h) for ts in sell_ts]
            buy_raw = [v for v in buy_raw if v is not None]
            sell_raw = [v for v in sell_raw if v is not None]
            if not buy_raw or not sell_raw:
                continue
            # continuation-signed (tradeable): LONG after BUY-liq, SHORT after SELL-liq
            cont = ([signed_return_bps("LONG", 1.0, 1.0 + v / 10_000.0) for v in buy_raw]
                    + [signed_return_bps("SHORT", 1.0, 1.0 + v / 10_000.0) for v in sell_raw])
            cont_cal = ([signed_return_bps("LONG", 1.0, 1.0 + (raw_return_bps(marks, ts, h) or 0) / 10_000.0)
                         for ts in buy_ts if ts < cut and raw_return_bps(marks, ts, h) is not None]
                        + [signed_return_bps("SHORT", 1.0, 1.0 + (raw_return_bps(marks, ts, h) or 0) / 10_000.0)
                           for ts in sell_ts if ts < cut and raw_return_bps(marks, ts, h) is not None])
            cont_hold = ([signed_return_bps("LONG", 1.0, 1.0 + (raw_return_bps(marks, ts, h) or 0) / 10_000.0)
                          for ts in buy_ts if ts >= cut and raw_return_bps(marks, ts, h) is not None]
                         + [signed_return_bps("SHORT", 1.0, 1.0 + (raw_return_bps(marks, ts, h) or 0) / 10_000.0)
                            for ts in sell_ts if ts >= cut and raw_return_bps(marks, ts, h) is not None])
            bm, sm = med(buy_raw), med(sell_raw)
            diff = (bm - sm) if (bm is not None and sm is not None) else None
            horizons_out[HORIZON_LABEL[h]] = {
                "n_buy": len(buy_raw), "n_sell": len(sell_raw),
                "buy_raw_median": bm, "sell_raw_median": sm,
                "signal_diff_bps": r1(diff) if diff is not None else None,
                "direction": (None if diff is None else ("CONTINUATION" if diff > 0 else "REVERSAL")),
                "cont_net_median_bps": r1((med(cont) or 0) - cost_bps_rt),
                "cont_win_rate": win(cont),
                "cont_cal_net_median_bps": r1((med(cont_cal) or 0) - cost_bps_rt) if cont_cal else None,
                "cont_hold_net_median_bps": r1((med(cont_hold) or 0) - cost_bps_rt) if cont_hold else None,
            }
        thresholds_out[f"{int(th/1000)}K"] = {"n_buy_events": len(buy_ts), "n_sell_events": len(sell_ts), "horizons": horizons_out}
    return {"symbol": symbol, "thresholds": thresholds_out}


def render_md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    lines = [
        "# S34 Liquidation Swing Event (beta-controlled, 1h-48h)",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  cost `{cfg['cost_bps_rt']}`bps, holdout `{cfg['holdout_frac']}`",
        "",
        "`signal_diff` = median(raw return | BUY-liq) - median(raw | SELL-liq); beta cancels. >0 CONTINUATION, <0 REVERSAL. "
        "`cont_net` = combined continuation-signed net P&L (LONG after BUY-liq, SHORT after SELL-liq), beta-cancelling, after cost. "
        "Note: long horizons overlap (not independent); read the BUY-vs-SELL difference and cal/hold stability, not single windows.",
        "",
    ]
    for sym in report["symbols"]:
        for thr, d in sym["thresholds"].items():
            lines.append(f"## {sym['symbol']} {thr}  (BUY events={d['n_buy_events']}, SELL events={d['n_sell_events']})")
            lines.append("")
            lines.append("| Horizon | BUY raw med | SELL raw med | signal_diff | dir | cont_net | cont_win% | cal_net | hold_net |")
            lines.append("| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
            for hl, h in d["horizons"].items():
                wr = None if h["cont_win_rate"] is None else r1(h["cont_win_rate"] * 100.0)
                lines.append(
                    f"| {hl} | {h['buy_raw_median']} | {h['sell_raw_median']} | {h['signal_diff_bps']} | "
                    f"{h['direction']} | {h['cont_net_median_bps']} | {wr} | {h['cont_cal_net_median_bps']} | {h['cont_hold_net_median_bps']} |"
                )
            lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Beta-controlled swing-scale (1h-48h) response to liquidation cascade events.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--bucket-sec", type=int, default=300)
    p.add_argument("--min-gap-sec", type=int, default=900)
    p.add_argument("--accel-window-sec", type=int, default=30)
    p.add_argument("--holdout-frac", type=float, default=0.30)
    p.add_argument("--cost-bps-rt", type=float, default=6.1)
    p.add_argument("--symbols", default=",".join(SYMBOLS))
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    syms = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        symbols = [
            eval_symbol(conn, sym, bucket_sec=int(args.bucket_sec), min_gap_sec=int(args.min_gap_sec),
                        accel_window_sec=int(args.accel_window_sec), holdout_frac=float(args.holdout_frac),
                        cost_bps_rt=float(args.cost_bps_rt))
            for sym in syms
        ]
    report = {
        "generated_at_utc": utc_now(),
        "config": {"holdout_frac": float(args.holdout_frac), "cost_bps_rt": float(args.cost_bps_rt)},
        "symbols": symbols,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
