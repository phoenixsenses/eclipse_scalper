# encoding: utf-8
"""
S34 Research: ETH SELL cascade threshold timing comparison
250K vs 300K vs 500K vs 1M early-warning analysis

Hypothesis: 500K threshold triggers at the "confirmed" point of a SELL cascade.
Lower thresholds (250K/300K) may trigger earlier, capturing more of the initial
price drop. Test whether earlier entry = better SHORT outcome.

Method:
- Build 30s cascade buckets from SELL liquidations
- For each cascade, find WHEN each threshold is first crossed (early warning time)
- Calculate: delay between 250K trigger and 500K trigger on same cascade
- Simulate SHORT entry at each threshold's trigger point
- Compare TP/SL/timing outcomes
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

ROOT     = Path("D:/eclipse_scalper")
MICRO_DB = ROOT / "data" / "microstructure.db"

THRESHOLDS   = [250_000, 300_000, 500_000, 1_000_000]
BUCKET_SEC   = 30
TP_BPS       = 80.0   # current ETH SELL 1M rule
SL_BPS       = 40.0
BE_BPS       = 40.0
HOLD_SEC     = 510
FEE_BPS      = 8.0
MERGE_GAP_MS = 60_000  # cascades within 60s = same event

def mark_at(micro, symbol, ts_ms):
    r = micro.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (symbol, ts_ms)).fetchone()
    return r[0] if r else None

def simulate_short(micro, entry_ms, tp_bps, sl_bps, be_bps, hold_sec):
    p0 = mark_at(micro, 'ETHUSDT', entry_ms)
    if not p0:
        return None, 'MISS'
    p_tp = p0 * (1 - tp_bps / 10000)
    p_sl = p0 * (1 + sl_bps / 10000)
    p_be = p0 * (1 - be_bps / 10000)
    be_on = False
    rows = micro.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>? AND ts_ms<=? ORDER BY ts_ms",
        (entry_ms, entry_ms + hold_sec * 1000)).fetchall()
    for (mp,) in rows:
        if mp <= p_tp:
            return (p0 - p_tp) / p0 * 10000 - FEE_BPS, 'TP'
        if be_on and mp >= p0:
            return -FEE_BPS, 'BE'
        if not be_on and mp >= p_sl:
            return -((p_sl - p0) / p0 * 10000) - FEE_BPS, 'SL'
        if mp <= p_be:
            be_on = True
    p_end = mark_at(micro, 'ETHUSDT', entry_ms + hold_sec * 1000) or p0
    return (p0 - p_end) / p0 * 10000 - FEE_BPS, 'TIME'

def st(nets, label=""):
    if not nets:
        return f"  {label}: N=0"
    s = sorted(nets)
    sl = sum(1 for x in nets if x < -20)
    wr = sum(1 for x in nets if x > 0) / len(nets)
    med = s[len(s) // 2]
    mn = sum(nets) / len(nets)
    return (f"  {label:30} N={len(nets):>4}  WR={wr*100:.0f}%  "
            f"med={med:>+7.1f}  mean={mn:>+6.1f}  SL={sl}({sl/len(nets)*100:.0f}%)")

def main():
    micro = sqlite3.connect(f"file:{MICRO_DB}?mode=ro", uri=True)

    # Pull all ETH SELL liquidations
    eth_sell = micro.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' ORDER BY ts_ms"
    ).fetchall()

    # Build 30s cascade buckets: running total per bucket
    BUCKET_MS = BUCKET_SEC * 1000
    raw_buckets = defaultdict(lambda: [0.0, 0, None])
    for ts, notional in eth_sell:
        bk = (ts // BUCKET_MS) * BUCKET_MS
        raw_buckets[bk][0] += notional
        raw_buckets[bk][1] += 1
        if raw_buckets[bk][2] is None:
            raw_buckets[bk][2] = ts

    # Merge adjacent buckets within MERGE_GAP_MS into events
    # For each event, track cumulative notional over time (to find when each threshold crossed)
    sorted_bks = sorted(raw_buckets.items())

    # Group into events (consecutive buckets within 60s gap)
    events = []
    current_event = []
    for bk, data in sorted_bks:
        if not current_event or (bk - current_event[-1][0]) <= MERGE_GAP_MS:
            current_event.append((bk, data))
        else:
            events.append(current_event)
            current_event = [(bk, data)]
    if current_event:
        events.append(current_event)

    print("=" * 70)
    print("ETH SELL CASCADE THRESHOLD TIMING COMPARISON")
    print(f"250K vs 300K vs 500K vs 1M early-warning analysis")
    print(f"SHORT: TP={TP_BPS} SL={SL_BPS} BE={BE_BPS} hold={HOLD_SEC}s fee={FEE_BPS}")
    print("=" * 70)
    print(f"Total SELL liq events (merged): {len(events)}")
    print()

    # For each event, find when each threshold is FIRST crossed
    threshold_entries = {t: [] for t in THRESHOLDS}  # ts of first crossing
    threshold_delays  = []  # delay between 250K and 500K on same cascade

    cascade_data = []  # (event_start_ts, {thr: trigger_ts})

    for event in events:
        cum = 0.0
        crossed = {}
        for bk, (total, cnt, first_ts) in event:
            cum += total
            for thr in THRESHOLDS:
                if thr not in crossed and cum >= thr:
                    crossed[thr] = first_ts or bk
        if crossed:
            cascade_data.append(crossed)
            for thr, ts in crossed.items():
                threshold_entries[thr].append(ts)

    # Cascades that reach 500K threshold
    cascades_500k = [c for c in cascade_data if 500_000 in c]
    print(f"Cascades reaching each threshold:")
    for thr in THRESHOLDS:
        n = len([c for c in cascade_data if thr in c])
        print(f"  ${thr/1000:.0f}K: {n} cascades")
    print()

    # Timing analysis: how much earlier does 250K/300K trigger vs 500K?
    delays_250_500 = []
    delays_300_500 = []
    for c in cascades_500k:
        if 250_000 in c:
            delays_250_500.append((c[500_000] - c[250_000]) / 1000)
        if 300_000 in c:
            delays_300_500.append((c[500_000] - c[300_000]) / 1000)

    def median(lst):
        if not lst: return 0
        s = sorted(lst)
        return s[len(s)//2]

    print("ENTRY TIMING ADVANTAGE (how many seconds EARLIER vs 500K trigger):")
    if delays_250_500:
        s = sorted(delays_250_500)
        print(f"  250K before 500K: med={median(delays_250_500):.1f}s  p25={s[len(s)//4]:.1f}s  p75={s[int(len(s)*0.75)]:.1f}s  max={s[-1]:.1f}s")
    if delays_300_500:
        s = sorted(delays_300_500)
        print(f"  300K before 500K: med={median(delays_300_500):.1f}s  p25={s[len(s)//4]:.1f}s  p75={s[int(len(s)*0.75)]:.1f}s  max={s[-1]:.1f}s")
    print()

    # Performance simulation: SHORT at each threshold trigger time
    print("SHORT PERFORMANCE BY THRESHOLD (TP=80 SL=40 BE=40):")
    results_by_thr = {t: [] for t in THRESHOLDS}

    for c in cascade_data:
        for thr in THRESHOLDS:
            if thr not in c:
                continue
            entry_ts = c[thr]
            net, exit_r = simulate_short(micro, entry_ts, TP_BPS, SL_BPS, BE_BPS, HOLD_SEC)
            if net is not None:
                results_by_thr[thr].append((net, exit_r))

    for thr in THRESHOLDS:
        nets = [x[0] for x in results_by_thr[thr]]
        label = f"${thr/1000:.0f}K threshold"
        print(st(nets, label))

    print()

    # OOS split at 70%
    cascades_sorted = sorted(cascade_data, key=lambda c: min(c.values()))
    split_idx = int(len(cascades_sorted) * 0.70)
    split_ts  = min(cascades_sorted[split_idx].values()) if split_idx < len(cascades_sorted) else 0
    split_dt  = datetime.fromtimestamp(split_ts/1000, tz=timezone.utc).strftime("%Y-%m-%d")

    train_cascades = cascades_sorted[:split_idx]
    test_cascades  = cascades_sorted[split_idx:]

    print(f"OOS SPLIT: {split_dt} (train={len(train_cascades)} test={len(test_cascades)} cascades)")
    print()
    print("TEST performance (OOS only):")
    test_results = {t: [] for t in THRESHOLDS}
    for c in test_cascades:
        for thr in THRESHOLDS:
            if thr not in c:
                continue
            net, exit_r = simulate_short(micro, c[thr], TP_BPS, SL_BPS, BE_BPS, HOLD_SEC)
            if net is not None:
                test_results[thr].append((net, exit_r))

    for thr in THRESHOLDS:
        nets = [x[0] for x in test_results[thr]]
        label = f"${thr/1000:.0f}K threshold"
        print(st(nets, label))

    # Also: 500K vs 250K same-cascade comparison
    print()
    print("SAME-CASCADE COMPARISON: 250K vs 500K entry on identical events:")
    paired_250 = []
    paired_500 = []
    for c in test_cascades:
        if 250_000 not in c or 500_000 not in c:
            continue
        n250, _ = simulate_short(micro, c[250_000], TP_BPS, SL_BPS, BE_BPS, HOLD_SEC)
        n500, _ = simulate_short(micro, c[500_000], TP_BPS, SL_BPS, BE_BPS, HOLD_SEC)
        if n250 is not None and n500 is not None:
            paired_250.append(n250)
            paired_500.append(n500)

    if paired_250:
        print(st(paired_250, "250K (early)"))
        print(st(paired_500, "500K (confirm)"))
        avg_diff = sum(paired_250) / len(paired_250) - sum(paired_500) / len(paired_500)
        print(f"  Early entry advantage: {avg_diff:+.1f} bps mean (250K - 500K)")

    micro.close()
    print()
    print("NOTE: Shadow research. Separate pre-registration required before live.")


if __name__ == "__main__":
    main()
