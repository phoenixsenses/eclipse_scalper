# encoding: utf-8
"""
S34 Research: Cascade gap / cooldown analysis

Finding: BUY signal #3-4 of day has SL=28-29%. SELL signal #1 is weakest.
Question: Is the real driver "intraday ordinal" OR "time since last cascade"?

If gap-based: "wait Xs after a cascade before re-entering" is actionable.
If ordinal-based: "skip Nth signal" is the right filter.

Also tests: same-direction gap vs opposite-direction gap.
  e.g. ETH BUY after recent ETH BUY vs ETH BUY after recent ETH SELL

Tests ETH BUY and ETH SELL.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

ROOT     = Path("D:/eclipse_scalper")
MICRO_DB = ROOT / "data" / "microstructure.db"

BUCKET_SEC = 30
FEE_BPS    = 8.0

ROUTES = [
    dict(label="ETH BUY  $500K", symbol="ETHUSDT", side="BUY",
         threshold=500_000, cnt_min=8, tp=60.0, sl=40.0, be=30.0,
         hold=510, direction="LONG"),
    dict(label="ETH SELL $500K", symbol="ETHUSDT", side="SELL",
         threshold=500_000, cnt_min=8, tp=60.0, sl=40.0, be=40.0,
         hold=510, direction="SHORT"),
]

def mark_at(micro, symbol, ts_ms):
    r = micro.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (symbol, ts_ms)).fetchone()
    return r[0] if r else None

def simulate(micro, symbol, entry_ms, tp, sl, be, hold, direction):
    p0 = mark_at(micro, symbol, entry_ms)
    if not p0:
        return None, "MISS"
    if direction == "LONG":
        p_tp = p0 * (1 + tp / 10000)
        p_sl = p0 * (1 - sl / 10000)
        p_be = p0 * (1 + be / 10000)
    else:
        p_tp = p0 * (1 - tp / 10000)
        p_sl = p0 * (1 + sl / 10000)
        p_be = p0 * (1 - be / 10000)
    be_on = False
    rows = micro.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms>? AND ts_ms<=? ORDER BY ts_ms",
        (symbol, entry_ms, entry_ms + hold * 1000)).fetchall()
    for (mp,) in rows:
        if direction == "LONG":
            if mp >= p_tp: return float(tp - FEE_BPS), "TP"
            if be_on and mp <= p0: return float(-FEE_BPS), "BE"
            if not be_on and mp <= p_sl: return float(-sl - FEE_BPS), "SL"
            if mp >= p_be: be_on = True
        else:
            if mp <= p_tp: return float(tp - FEE_BPS), "TP"
            if be_on and mp >= p0: return float(-FEE_BPS), "BE"
            if not be_on and mp >= p_sl: return float(-sl - FEE_BPS), "SL"
            if mp <= p_be: be_on = True
    p_end = mark_at(micro, symbol, entry_ms + hold * 1000) or p0
    if direction == "LONG":
        return float((p_end - p0) / p0 * 10000 - FEE_BPS), "TIME"
    else:
        return float((p0 - p_end) / p0 * 10000 - FEE_BPS), "TIME"

def build_cascades(micro, symbol, side, threshold, cnt_min):
    BM = BUCKET_SEC * 1000
    rows = micro.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol=? AND side=? ORDER BY ts_ms",
        (symbol, side)).fetchall()
    buckets = defaultdict(lambda: [0.0, 0, None])
    for ts, notional in rows:
        bk = (ts // BM) * BM
        buckets[bk][0] += notional
        buckets[bk][1] += 1
        if buckets[bk][2] is None:
            buckets[bk][2] = ts
    return sorted(
        [(d[2], d[0], d[1]) for _, d in buckets.items()
         if d[0] >= threshold and d[1] >= cnt_min and d[2] is not None],
        key=lambda x: x[0])

def st(label, nets, width=35):
    if not nets:
        return f"  {label:{width}} N=  0"
    s = sorted(nets)
    sl  = sum(1 for x in nets if x < -20)
    wr  = sum(1 for x in nets if x > 0) / len(nets)
    med = s[len(s) // 2]
    mn  = sum(nets) / len(nets)
    return (f"  {label:{width}} N={len(nets):>4}  WR={wr*100:>4.0f}%  "
            f"med={med:>+7.1f}  mean={mn:>+6.1f}  SL={sl}({sl/len(nets)*100:>3.0f}%)")

def bucket_gap(gap_sec):
    """Label for gap-since-last-cascade bucket."""
    if gap_sec is None:
        return "first_ever"
    if gap_sec < 300:
        return "<5min"
    if gap_sec < 900:
        return "5-15min"
    if gap_sec < 1800:
        return "15-30min"
    if gap_sec < 3600:
        return "30-60min"
    if gap_sec < 7200:
        return "1-2h"
    if gap_sec < 14400:
        return "2-4h"
    return "4h+"

def main():
    micro = sqlite3.connect(f"file:{MICRO_DB}?mode=ro", uri=True)

    # Pre-build opposite-direction cascade sets for gap measurement
    eth_buy_all  = build_cascades(micro, "ETHUSDT", "BUY",  500_000, 1)
    eth_sell_all = build_cascades(micro, "ETHUSDT", "SELL", 500_000, 1)

    print("S34 CASCADE GAP / COOLDOWN ANALYSIS")
    print("Question: Is intraday degradation driven by TIME gap or ordinal position?")

    for route in ROUTES:
        cascades = build_cascades(micro, route["symbol"], route["side"],
                                  route["threshold"], route["cnt_min"])

        opp_cascades = eth_sell_all if route["side"] == "BUY" else eth_buy_all

        results = []
        day_last: dict[str, int] = {}  # day -> last cascade ts

        for i, (ts, total, cnt) in enumerate(cascades):
            net, exit_r = simulate(micro, route["symbol"], ts,
                                   route["tp"], route["sl"], route["be"],
                                   route["hold"], route["direction"])
            if net is None:
                continue

            dk = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

            # Gap since last SAME-direction cascade (any cnt)
            same_gap = None
            if i > 0:
                same_gap = (ts - cascades[i - 1][0]) / 1000

            # Gap since last OPPOSITE-direction cascade
            opp_gap = None
            for (ot, *_) in reversed(opp_cascades):
                if ot < ts:
                    opp_gap = (ts - ot) / 1000
                    break

            # Intraday ordinal (how many same-dir cascades today before this one)
            nth = 1
            if dk in day_last:
                # count cascades today before this ts
                nth = sum(1 for (ct, *_) in cascades if
                          datetime.fromtimestamp(ct/1000, tz=timezone.utc).strftime("%Y-%m-%d") == dk
                          and ct < ts) + 1
            day_last[dk] = ts

            results.append({
                "ts": ts, "net": net, "exit": exit_r,
                "same_gap_sec": same_gap,
                "opp_gap_sec": opp_gap,
                "nth": nth,
            })

        split_idx = int(len(results) * 0.70)
        split_ts  = results[split_idx]["ts"] if split_idx < len(results) else 0
        split_dt  = datetime.fromtimestamp(split_ts/1000, tz=timezone.utc).strftime("%Y-%m-%d")
        test = results[split_idx:]

        print(f"\n{'='*72}")
        print(f"{route['label']}  |  OOS: {split_dt}  test N={len(test)}")
        print(f"{'='*72}")

        # --- Gap since last same-direction cascade (OOS) ---
        print("OOS: Gap since last SAME-direction cascade")
        gap_order = ["<5min", "5-15min", "15-30min", "30-60min", "1-2h", "2-4h", "4h+", "first_ever"]
        for gb in gap_order:
            nets = [r["net"] for r in test if bucket_gap(r["same_gap_sec"]) == gb]
            print(st(gb, nets))
        print(st("ALL", [r["net"] for r in test]))

        # --- Gap since last OPPOSITE-direction cascade (OOS) ---
        print("\nOOS: Gap since last OPPOSITE-direction cascade")
        for gb in gap_order:
            nets = [r["net"] for r in test if bucket_gap(r["opp_gap_sec"]) == gb]
            print(st(gb, nets))

        # --- Intraday Nth (OOS) ---
        print("\nOOS: Intraday ordinal (Nth cascade of day)")
        for n in [1, 2, 3, 4, 5]:
            nets = [r["net"] for r in test if r["nth"] == n]
            label = f"#{n} of day"
            print(st(label, nets))
        nets_6p = [r["net"] for r in test if r["nth"] >= 6]
        print(st("#6+ of day", nets_6p))

        # --- Continuous gap vs performance (all data, correlate) ---
        print("\nALL DATA: Gap buckets vs performance (same-dir gap)")
        for gb in gap_order:
            nets = [r["net"] for r in results if bucket_gap(r["same_gap_sec"]) == gb]
            print(st(gb, nets))

        # Key question: is gap <30min correlated with bad outcome?
        print("\nALL DATA: <30min cooldown vs >=30min")
        short_gap = [r["net"] for r in results
                     if r["same_gap_sec"] is not None and r["same_gap_sec"] < 1800]
        long_gap  = [r["net"] for r in results
                     if r["same_gap_sec"] is None or r["same_gap_sec"] >= 1800]
        print(st("Gap <30min (too soon?)", short_gap))
        print(st("Gap >=30min or first",   long_gap))

    micro.close()
    print("\nNOTE: Shadow research. N>=50 per bucket before action.")

if __name__ == "__main__":
    main()
