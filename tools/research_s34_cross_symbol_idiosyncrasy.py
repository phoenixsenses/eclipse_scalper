# encoding: utf-8
"""
S34 Research: Cross-symbol co-cascade idiosyncrasy analysis

Hypothesis: ETH cascade quality depends on whether it's ETH-specific
or market-wide (co-movement with BTC and/or SOL).

  Idiosyncratic: Only ETH cascades -> cleanest signal
  Systemic:      ETH + BTC/SOL cascade simultaneously -> bounce risk

Tests all 4 ETH/SOL routes, classifying each cascade as:
  - ETH_ONLY:     No BTC or SOL co-cascade within +-60s
  - ETH_BTC:      ETH + BTC co-cascade (same direction)
  - ETH_SOL:      ETH + SOL co-cascade (same direction)
  - ETH_BTC_SOL:  All three (full systemic)

Also tests opposite-direction co-cascade (divergence signal).

OOS 70/30 split.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

ROOT     = Path("D:/eclipse_scalper")
MICRO_DB = ROOT / "data" / "microstructure.db"

BUCKET_SEC   = 30
CO_WINDOW_MS = 60_000   # look +-60s for co-cascades
FEE_BPS      = 8.0

# BTC/SOL cascade thresholds (lower to detect meaningful co-movement)
BTC_CO_THRESHOLD = 1_000_000   # $1M BTC cascade
SOL_CO_THRESHOLD =   100_000   # $100K SOL cascade

ROUTES = [
    dict(label="ETH BUY  $500K", symbol="ETHUSDT", side="BUY",
         threshold=500_000, cnt_min=8,
         tp=60.0, sl=40.0, be=30.0, hold=510, direction="LONG"),
    dict(label="ETH SELL $500K", symbol="ETHUSDT", side="SELL",
         threshold=500_000, cnt_min=8,
         tp=60.0, sl=40.0, be=40.0, hold=510, direction="SHORT"),
    dict(label="SOL BUY  $200K", symbol="SOLUSDT", side="BUY",
         threshold=200_000, cnt_min=8,
         tp=60.0, sl=40.0, be=30.0, hold=510, direction="LONG"),
    dict(label="SOL SELL $200K", symbol="SOLUSDT", side="SELL",
         threshold=200_000, cnt_min=8,
         tp=60.0, sl=30.0, be=30.0, hold=510, direction="SHORT"),
]

def mark_at(micro, symbol, ts_ms):
    r = micro.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (symbol, ts_ms)).fetchone()
    return r[0] if r else None

def simulate(micro, symbol, entry_ms, tp_bps, sl_bps, be_bps, hold_sec, direction):
    p0 = mark_at(micro, symbol, entry_ms)
    if not p0:
        return None, "MISS"
    if direction == "LONG":
        p_tp = p0 * (1 + tp_bps / 10000)
        p_sl = p0 * (1 - sl_bps / 10000)
        p_be = p0 * (1 + be_bps / 10000)
    else:
        p_tp = p0 * (1 - tp_bps / 10000)
        p_sl = p0 * (1 + sl_bps / 10000)
        p_be = p0 * (1 - be_bps / 10000)
    be_on = False
    rows = micro.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms>? AND ts_ms<=? ORDER BY ts_ms",
        (symbol, entry_ms, entry_ms + hold_sec * 1000)).fetchall()
    for (mp,) in rows:
        if direction == "LONG":
            if mp >= p_tp: return float(tp_bps - FEE_BPS), "TP"
            if be_on and mp <= p0: return float(-FEE_BPS), "BE"
            if not be_on and mp <= p_sl: return float(-sl_bps - FEE_BPS), "SL"
            if mp >= p_be: be_on = True
        else:
            if mp <= p_tp: return float(tp_bps - FEE_BPS), "TP"
            if be_on and mp >= p0: return float(-FEE_BPS), "BE"
            if not be_on and mp >= p_sl: return float(-sl_bps - FEE_BPS), "SL"
            if mp <= p_be: be_on = True
    p_end = mark_at(micro, symbol, entry_ms + hold_sec * 1000) or p0
    if direction == "LONG":
        return float((p_end - p0) / p0 * 10000 - FEE_BPS), "TIME"
    else:
        return float((p0 - p_end) / p0 * 10000 - FEE_BPS), "TIME"

def build_cascade_set(micro, symbol, side, threshold, cnt_min):
    """Returns sorted list of (ts_ms, total, cnt) cascade events."""
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

def has_co_cascade(cascade_list, ts_ms, window_ms):
    """Binary: did a cascade in cascade_list fire within window of ts_ms?"""
    lo = ts_ms - window_ms
    hi = ts_ms + window_ms
    for (ct, *_) in cascade_list:
        if lo <= ct <= hi:
            return True
        if ct > hi:
            break
    return False

def st(label, nets):
    if not nets:
        return f"  {label:30} N=  0"
    s = sorted(nets)
    sl  = sum(1 for x in nets if x < -20)
    wr  = sum(1 for x in nets if x > 0) / len(nets)
    med = s[len(s) // 2]
    mn  = sum(nets) / len(nets)
    cum = sum(nets)
    return (f"  {label:30} N={len(nets):>4}  WR={wr*100:>4.0f}%  "
            f"med={med:>+7.1f}  mean={mn:>+6.1f}  cum={cum:>+7.0f}  "
            f"SL={sl}({sl/len(nets)*100:>3.0f}%)")

def main():
    micro = sqlite3.connect(f"file:{MICRO_DB}?mode=ro", uri=True)

    print("S34 CROSS-SYMBOL CO-CASCADE IDIOSYNCRASY ANALYSIS")
    print(f"Co-cascade window: +-{CO_WINDOW_MS//1000}s")
    print(f"BTC threshold: ${BTC_CO_THRESHOLD/1e6:.0f}M  |  SOL threshold: ${SOL_CO_THRESHOLD/1e3:.0f}K")

    # Pre-build co-cascade lookup sets
    btc_buy_cascades  = build_cascade_set(micro, "BTCUSDT", "BUY",  BTC_CO_THRESHOLD, 1)
    btc_sell_cascades = build_cascade_set(micro, "BTCUSDT", "SELL", BTC_CO_THRESHOLD, 1)
    sol_buy_cascades  = build_cascade_set(micro, "SOLUSDT", "BUY",  SOL_CO_THRESHOLD, 1)
    sol_sell_cascades = build_cascade_set(micro, "SOLUSDT", "SELL", SOL_CO_THRESHOLD, 1)
    eth_buy_cascades  = build_cascade_set(micro, "ETHUSDT", "BUY",  500_000, 1)
    eth_sell_cascades = build_cascade_set(micro, "ETHUSDT", "SELL", 500_000, 1)

    print(f"\nCo-cascade pool sizes:")
    print(f"  BTC BUY $1M+:  {len(btc_buy_cascades)}")
    print(f"  BTC SELL $1M+: {len(btc_sell_cascades)}")
    print(f"  SOL BUY $100K+:{len(sol_buy_cascades)}")
    print(f"  SOL SELL $100K+:{len(sol_sell_cascades)}")

    for route in ROUTES:
        cascades = build_cascade_set(micro, route["symbol"], route["side"],
                                     route["threshold"], route["cnt_min"])
        if not cascades:
            continue

        # For each cascade: determine co-cascade type
        # Same-direction co-cascade = systemic
        # Opposite-direction co-cascade = divergence
        if route["side"] == "BUY":
            same_btc  = btc_buy_cascades
            opp_btc   = btc_sell_cascades
            same_sol  = sol_buy_cascades  if route["symbol"] == "ETHUSDT" else eth_buy_cascades
            opp_sol   = sol_sell_cascades if route["symbol"] == "ETHUSDT" else eth_sell_cascades
        else:
            same_btc  = btc_sell_cascades
            opp_btc   = btc_buy_cascades
            same_sol  = sol_sell_cascades if route["symbol"] == "ETHUSDT" else eth_sell_cascades
            opp_sol   = sol_buy_cascades  if route["symbol"] == "ETHUSDT" else eth_buy_cascades

        results = []
        for ts, total, cnt in cascades:
            net, exit_r = simulate(micro, route["symbol"], ts,
                                   route["tp"], route["sl"], route["be"],
                                   route["hold"], route["direction"])
            if net is None:
                continue

            has_btc_same = has_co_cascade(same_btc, ts, CO_WINDOW_MS)
            has_sol_same = has_co_cascade(same_sol, ts, CO_WINDOW_MS)
            has_btc_opp  = has_co_cascade(opp_btc,  ts, CO_WINDOW_MS)
            has_sol_opp  = has_co_cascade(opp_sol,  ts, CO_WINDOW_MS)

            if has_btc_same and has_sol_same:
                co_type = "SYSTEMIC (BTC+SOL)"
            elif has_btc_same:
                co_type = "ETH+BTC"
            elif has_sol_same:
                co_type = "ETH+SOL"
            else:
                co_type = "IDIOSYNCRATIC"

            divergence = has_btc_opp or has_sol_opp

            results.append({
                "ts": ts, "net": net, "exit": exit_r,
                "co_type": co_type, "divergence": divergence,
                "has_btc_same": has_btc_same, "has_sol_same": has_sol_same,
            })

        split_idx = int(len(results) * 0.70)
        split_ts  = results[split_idx]["ts"] if split_idx < len(results) else 0
        split_dt  = datetime.fromtimestamp(split_ts/1000, tz=timezone.utc).strftime("%Y-%m-%d")
        test = results[split_idx:]

        print(f"\n{'='*72}")
        print(f"{route['label']}  |  OOS: {split_dt}  |  total N={len(results)}  test N={len(test)}")
        print(f"{'='*72}")

        co_types = ["IDIOSYNCRATIC", "ETH+BTC", "ETH+SOL", "SYSTEMIC (BTC+SOL)"]

        print("ALL DATA:")
        for ct in co_types:
            nets = [r["net"] for r in results if r["co_type"] == ct]
            print(st(ct, nets))
        print(st("ALL", [r["net"] for r in results]))

        print("\nOOS (TEST) ONLY:")
        for ct in co_types:
            nets = [r["net"] for r in test if r["co_type"] == ct]
            print(st(ct, nets))
        print(st("ALL", [r["net"] for r in test]))

        # Divergence (opposite-direction BTC/SOL co-cascade)
        div_nets  = [r["net"] for r in test if r["divergence"]]
        ndiv_nets = [r["net"] for r in test if not r["divergence"]]
        if div_nets or ndiv_nets:
            print("\nOOS — DIVERGENCE (opposite-dir co-cascade):")
            print(st("With divergence",    div_nets))
            print(st("No divergence",      ndiv_nets))

        # Composition breakdown
        print(f"\nComposition (all data):")
        total_n = len(results)
        for ct in co_types:
            n = sum(1 for r in results if r["co_type"] == ct)
            print(f"  {ct:30} {n:>4} ({n/total_n*100:>4.0f}%)")

    micro.close()
    print("\nNOTE: Shadow research. N>=50 per bucket before action.")

if __name__ == "__main__":
    main()
