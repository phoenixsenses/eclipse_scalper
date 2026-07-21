# encoding: utf-8
"""
S34 Research: BTC micro-trend at ETH/SOL SELL cascade entry

For BUY cascades: BTC 10s return 2-5 bps sweet spot (WR=70%, SL=12%)
Question: does BTC negative momentum improve SELL cascade (SHORT) performance?
Logic: SELL cascade + BTC also falling = market-wide bearish = stronger SHORT

Tests:
  - ETH $500K SELL cascade -> SHORT
  - BTC 10s return segmentation: neg (<0), 0-2, 2-5 bps, 5-10, 10+
    (For SELL, negative BTC = momentum aligned with SHORT)
  - Also checks BTC 30s return (slightly longer window)
  - OOS 70/30 split
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
    dict(label="ETH SELL $500K", symbol="ETHUSDT", side="SELL",
         threshold=500_000, cnt_min=8,
         tp=60.0, sl=40.0, be=40.0, hold=510, direction="SHORT"),
    dict(label="SOL SELL $200K", symbol="SOLUSDT", side="SELL",
         threshold=200_000, cnt_min=8,
         tp=60.0, sl=30.0, be=30.0, hold=510, direction="SHORT"),
]

BTC_WINDOWS = [10, 30]  # seconds

def mark_at(micro, symbol, ts_ms):
    r = micro.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (symbol, ts_ms)).fetchone()
    return r[0] if r else None

def simulate_short(micro, symbol, entry_ms, tp_bps, sl_bps, be_bps, hold_sec):
    p0 = mark_at(micro, symbol, entry_ms)
    if not p0:
        return None, "MISS"
    p_tp = p0 * (1 - tp_bps / 10000)
    p_sl = p0 * (1 + sl_bps / 10000)
    p_be = p0 * (1 - be_bps / 10000)
    be_on = False
    rows = micro.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms>? AND ts_ms<=? ORDER BY ts_ms",
        (symbol, entry_ms, entry_ms + hold_sec * 1000)).fetchall()
    for (mp,) in rows:
        if mp <= p_tp: return float(tp_bps - FEE_BPS), "TP"
        if be_on and mp >= p0: return float(-FEE_BPS), "BE"
        if not be_on and mp >= p_sl: return float(-sl_bps - FEE_BPS), "SL"
        if mp <= p_be: be_on = True
    p_end = mark_at(micro, symbol, entry_ms + hold_sec * 1000) or p0
    return float((p0 - p_end) / p0 * 10000 - FEE_BPS), "TIME"

def bps_ret(p1, p2):
    if not p1 or not p2 or p1 == 0:
        return None
    return (p2 - p1) / p1 * 10000

def build_cascades(micro, symbol, side, threshold, cnt_min):
    BUCKET_MS = BUCKET_SEC * 1000
    rows = micro.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol=? AND side=? ORDER BY ts_ms",
        (symbol, side)).fetchall()
    buckets = defaultdict(lambda: [0.0, 0, None])
    for ts, notional in rows:
        bk = (ts // BUCKET_MS) * BUCKET_MS
        buckets[bk][0] += notional
        buckets[bk][1] += 1
        if buckets[bk][2] is None:
            buckets[bk][2] = ts
    return sorted(
        [(data[2], data[0], data[1])
         for bk, data in buckets.items()
         if data[0] >= threshold and data[1] >= cnt_min and data[2] is not None],
        key=lambda x: x[0]
    )

def st(label, nets):
    if not nets:
        return f"  {label:35} N=  0"
    s = sorted(nets)
    sl  = sum(1 for x in nets if x < -20)
    wr  = sum(1 for x in nets if x > 0) / len(nets)
    med = s[len(s) // 2]
    mn  = sum(nets) / len(nets)
    return (f"  {label:35} N={len(nets):>4}  WR={wr*100:>4.0f}%  "
            f"med={med:>+7.1f}  mean={mn:>+6.1f}  SL={sl}({sl/len(nets)*100:>3.0f}%)")

# For SELL: negative BTC = aligned with SHORT direction
BTC_BANDS_SELL = [
    ("BTC neg  (<-2 bps)",  lambda r: r is not None and r < -2),
    ("BTC flat (-2 to 0)",  lambda r: r is not None and -2 <= r < 0),
    ("BTC 0-2  bps",        lambda r: r is not None and 0 <= r < 2),
    ("BTC 2-5  bps",        lambda r: r is not None and 2 <= r < 5),
    ("BTC 5+   bps",        lambda r: r is not None and r >= 5),
    ("BTC miss",            lambda r: r is None),
]

def main():
    micro = sqlite3.connect(f"file:{MICRO_DB}?mode=ro", uri=True)

    print("S34 SELL-SIDE BTC MICRO-TREND ANALYSIS")
    print("Hypothesis: BTC falling at SELL cascade entry -> stronger SHORT")
    print("Symmetric to BUY: neg BTC = aligned momentum for SHORT")

    for route in ROUTES:
        cascades = build_cascades(micro, route["symbol"], route["side"],
                                  route["threshold"], route["cnt_min"])
        if not cascades:
            print(f"\nNo cascades for {route['label']}")
            continue

        # Simulate all and collect BTC returns
        results = []
        for ts, total, cnt in cascades:
            net, exit_r = simulate_short(micro, route["symbol"], ts,
                                         route["tp"], route["sl"],
                                         route["be"], route["hold"])
            if net is None:
                continue
            row = {"ts": ts, "total": total, "cnt": cnt, "net": net, "exit": exit_r}
            for win in BTC_WINDOWS:
                b_before = mark_at(micro, "BTCUSDT", ts - win * 1000)
                b_at     = mark_at(micro, "BTCUSDT", ts)
                row[f"btc_{win}s"] = bps_ret(b_before, b_at)
            results.append(row)

        split_idx = int(len(results) * 0.70)
        split_ts  = results[split_idx]["ts"] if split_idx < len(results) else 0
        split_dt  = datetime.fromtimestamp(split_ts/1000, tz=timezone.utc).strftime("%Y-%m-%d")
        train = results[:split_idx]
        test  = results[split_idx:]

        print(f"\n{'='*70}")
        print(f"{route['label']}  (OOS split: {split_dt})")
        print(f"Total N={len(results)}  train={len(train)}  test={len(test)}")
        print(f"{'='*70}")

        for win in BTC_WINDOWS:
            key = f"btc_{win}s"
            print(f"\n--- BTC {win}s return segmentation ---")

            for split_label, subset in [("TRAIN", train), ("TEST (OOS)", test), ("ALL", results)]:
                print(f"  [{split_label}]")
                for band_label, fn in BTC_BANDS_SELL:
                    nets = [r["net"] for r in subset if fn(r[key])]
                    print(st(band_label, nets))
                print(st("ALL", [r["net"] for r in subset]))

        # SL detail
        print(f"\n--- SL trades by BTC 10s band ---")
        for band_label, fn in BTC_BANDS_SELL:
            key = "btc_10s"
            bad = [r for r in results if fn(r[key]) and r["net"] < -20]
            for r in bad:
                dt = datetime.fromtimestamp(r["ts"]/1000, tz=timezone.utc).strftime("%m/%d %H:%M")
                btc_s = f"{r[key]:+.1f}" if r[key] is not None else "N/A"
                print(f"  {band_label:25} SL: {dt}  btc={btc_s}  cnt={r['cnt']}  total=${r['total']/1e3:.0f}K")

        # Distribution summary: what % of cascades have BTC negative at entry?
        btc_signs = [r["btc_10s"] for r in results if r["btc_10s"] is not None]
        neg = sum(1 for x in btc_signs if x < 0)
        pos = sum(1 for x in btc_signs if x >= 0)
        print(f"\n--- BTC 10s sign at SELL cascade entry ---")
        print(f"  BTC negative: {neg}/{len(btc_signs)} ({neg/len(btc_signs)*100:.0f}%)")
        print(f"  BTC positive: {pos}/{len(btc_signs)} ({pos/len(btc_signs)*100:.0f}%)")

    micro.close()
    print("\nNOTE: Shadow research. N>=50 per band required before action.")

if __name__ == "__main__":
    main()
