# encoding: utf-8
"""
S34 Research: BTC $1M+ cascade -> ETH momentum scalp (cross-symbol lead-lag)

Hypothesis:
  When BTC has a large ($1M+) liquidation cascade BUT ETH does NOT have
  a simultaneous cascade, does ETH price still bounce?
  If yes: enter ETH long on BTC cascade signal, hold 60s.

Logic:
  BTC cascade -> BTC price bounces -> cross-market momentum -> ETH follows
  This is a "pre-ETH-cascade" entry using BTC as the leading signal.

Compared to current approach:
  Current: enter ETH on ETH's OWN $500K+ cascade (N=1.8/day)
  New:     enter ETH when BTC has $1M+ cascade (additional frequency?)

Test:
  - Pull all BTC $1M+ liquidation cascades from microstructure.db
  - Exclude cascades within 60s of an ETH $500K+ cascade (avoid double-counting)
  - Simulate ETH long entry at BTC cascade time, hold 60s, TP=60, SL=40
  - Compare to current ETH 500K signal performance
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

ROOT     = Path("D:/eclipse_scalper")
MICRO_DB = ROOT / "data" / "microstructure.db"
FEAT_DB  = ROOT / "data" / "s34_feature_factory.db"

BTC_THRESHOLD  = 1_000_000   # $1M BTC cascade
ETH_THRESHOLD  = 500_000     # $500K ETH cascade (our current signal)
HOLD_SEC       = 60
TP_BPS         = 60.0
SL_BPS         = 40.0
BE_BPS         = 30.0        # break-even trigger
EXCLUSION_SEC  = 60          # exclude if ETH cascade within this window
MIN_CNT        = 8           # liq_count filter (same as current)
FEE_BPS        = 8.0         # 4 bps each side taker

def bps(p1, p2):
    if not p1 or not p2 or p1 == 0:
        return None
    return (p2 - p1) / p1 * 10000

def stats(nets, label=""):
    if not nets:
        return f"{label}: N=0"
    s = sorted(nets)
    sl = sum(1 for x in nets if x < -20)
    wr = sum(1 for x in nets if x > 0) / len(nets)
    med = s[len(s)//2]
    mn = sum(nets) / len(nets)
    cum = sum(nets)
    nf_adj = f"net={mn-FEE_BPS:+.1f}(after fee)"
    return (f"{label}: N={len(nets)} WR={wr*100:.0f}% med={med:+.1f} "
            f"mean={mn:+.1f} SL={sl}({sl/len(nets)*100:.0f}%) cum={cum:+.0f} | {nf_adj}")

def mark_price_at(micro, symbol, ts_ms):
    row = micro.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (symbol, ts_ms)).fetchone()
    return row[0] if row else None

def simulate_trade(micro, symbol, entry_ms, hold_sec, tp_bps, sl_bps, be_bps):
    p_entry = mark_price_at(micro, symbol, entry_ms)
    if not p_entry:
        return None, "MISS"
    p_tp = p_entry * (1 + tp_bps / 10000)
    p_sl = p_entry * (1 - sl_bps / 10000)
    p_be = p_entry * (1 + be_bps / 10000)  # once reached, move SL to entry
    be_triggered = False
    final_price = None
    exit_reason = "TIME"

    # Walk mark prices from entry to entry+hold_sec
    rows = micro.execute("""
        SELECT ts_ms, mark_price FROM mark_prices
        WHERE symbol=? AND ts_ms > ? AND ts_ms <= ?
        ORDER BY ts_ms ASC
    """, (symbol, entry_ms, entry_ms + hold_sec * 1000)).fetchall()

    for ts, mp in rows:
        if mp >= p_tp:
            final_price = p_tp
            exit_reason = "TP"
            break
        if be_triggered and mp <= p_entry:
            final_price = p_entry
            exit_reason = "BE"
            break
        if not be_triggered and mp <= p_sl:
            final_price = p_sl
            exit_reason = "SL"
            break
        if mp >= p_be:
            be_triggered = True

    if final_price is None:
        final_price = mark_price_at(micro, symbol, entry_ms + hold_sec * 1000) or p_entry
        exit_reason = "TIME"

    net = bps(p_entry, final_price) - FEE_BPS
    return net, exit_reason


def main():
    micro = sqlite3.connect(f"file:{MICRO_DB}?mode=ro", uri=True)

    # --- STEP 1: Identify BTC $1M+ cascade events ---
    # A "cascade" = a period where BTC liquidations >= $1M within a 30s window
    # Use liquidations table, group by 30s buckets
    btc_liqs = micro.execute("""
        SELECT ts_ms, side, notional
        FROM liquidations WHERE symbol='BTCUSDT' AND ts_ms >= 1776000000000
        ORDER BY ts_ms
    """).fetchall()

    # Build 30-second cascade buckets
    BUCKET_MS = 30_000
    btc_buckets = defaultdict(lambda: {"total": 0, "cnt": 0, "ts": None})
    for ts, side, notional in btc_liqs:
        bucket_key = (ts // BUCKET_MS) * BUCKET_MS
        btc_buckets[bucket_key]["total"] += notional
        btc_buckets[bucket_key]["cnt"] += 1
        if btc_buckets[bucket_key]["ts"] is None:
            btc_buckets[bucket_key]["ts"] = ts

    btc_cascades = [(data["ts"], data["total"], data["cnt"])
                    for _, data in btc_buckets.items()
                    if data["total"] >= BTC_THRESHOLD]
    btc_cascades.sort()

    # --- STEP 2: Identify ETH $500K+ cascades to exclude ---
    eth_liqs = micro.execute("""
        SELECT ts_ms, notional
        FROM liquidations WHERE symbol='ETHUSDT' AND ts_ms >= 1776000000000
        ORDER BY ts_ms
    """).fetchall()

    eth_buckets = defaultdict(lambda: {"total": 0, "cnt": 0})
    for ts, notional in eth_liqs:
        bucket_key = (ts // BUCKET_MS) * BUCKET_MS
        eth_buckets[bucket_key]["total"] += notional
        eth_buckets[bucket_key]["cnt"] += 1

    eth_cascade_times = set(
        bucket_key for bucket_key, data in eth_buckets.items()
        if data["total"] >= ETH_THRESHOLD
    )

    print("=" * 70)
    print("BTC CASCADE -> ETH MOMENTUM SCALP (cross-symbol lead-lag)")
    print(f"BTC threshold: ${BTC_THRESHOLD/1e6:.0f}M  |  ETH threshold: ${ETH_THRESHOLD/1e3:.0f}K")
    print(f"Hold: {HOLD_SEC}s  |  TP={TP_BPS}  SL={SL_BPS}  BE={BE_BPS}  Fee={FEE_BPS} bps")
    print("=" * 70)
    print()
    print(f"BTC $1M+ cascades found: {len(btc_cascades)}")
    print(f"ETH $500K+ cascades found: {len(eth_cascade_times)}")
    print()

    # --- STEP 3: Filter BTC cascades, run simulation ---
    pure_btc = []  # BTC cascade without nearby ETH cascade
    mixed    = []  # BTC cascade WITH nearby ETH cascade
    no_fill  = 0

    for ts, total, cnt in btc_cascades:
        # Check if any ETH cascade within exclusion window
        eth_nearby = any(
            abs(et - ts) <= EXCLUSION_SEC * 1000
            for et in eth_cascade_times
        )
        # Also apply cnt filter equivalent for BTC
        if cnt < MIN_CNT:
            continue

        net, exit_r = simulate_trade(micro, 'ETHUSDT', ts, HOLD_SEC, TP_BPS, SL_BPS, BE_BPS)
        if net is None:
            no_fill += 1
            continue

        entry = (net, exit_r, ts, total, cnt)
        if eth_nearby:
            mixed.append(entry)
        else:
            pure_btc.append(entry)

    print(f"After cnt>={MIN_CNT} filter: {len(pure_btc)+len(mixed)} qualifying BTC cascades")
    print(f"  Pure BTC (no ETH cascade nearby): {len(pure_btc)}")
    print(f"  Mixed (BTC + ETH cascade):        {len(mixed)}")
    print(f"  No fill (no price data):           {no_fill}")
    print()

    pure_nets = [x[0] for x in pure_btc]
    mixed_nets = [x[0] for x in mixed]
    all_nets = pure_nets + mixed_nets

    print(stats(pure_nets, "PURE BTC (ETH entry, no ETH cascade)"))
    print(stats(mixed_nets, "MIXED (BTC+ETH cascade both)"))
    print(stats(all_nets, "ALL BTC-triggered ETH entries"))
    print()

    # Compare to current ETH 500K signal from feature_factory
    try:
        feat = sqlite3.connect(f"file:{FEAT_DB}?mode=ro", uri=True)
        eth500_events = feat.execute("""
            SELECT f.ts_ms FROM s34_features f
            WHERE f.symbol='ETHUSDT' AND f.liq_side='BUY'
            AND f.cluster_notional >= 500000
            AND f.cluster_liq_count >= 8
            AND (f.day_trend_bps IS NULL OR f.day_trend_bps >= 0)
            AND f.ts_ms >= 1776000000000
            ORDER BY f.ts_ms
        """).fetchall()
        feat.close()

        eth_nets = []
        for (ts,) in eth500_events[:100]:  # limit to avoid slow run
            net, _ = simulate_trade(micro, 'ETHUSDT', ts, HOLD_SEC, TP_BPS, SL_BPS, BE_BPS)
            if net is not None:
                eth_nets.append(net)

        print(stats(eth_nets[:100], f"ETH 500K own signal (sample N<={len(eth_nets)})"))
    except Exception as e:
        print(f"  ETH 500K comparison: {e}")

    micro.close()

    print()
    if pure_nets:
        print("PURE BTC cascade exits breakdown:")
        ec = {"TP": 0, "SL": 0, "BE": 0, "TIME": 0}
        for net, exit_r, *_ in pure_btc:
            ec[exit_r] = ec.get(exit_r, 0) + 1
        print(f"  {ec}")


if __name__ == "__main__":
    main()
