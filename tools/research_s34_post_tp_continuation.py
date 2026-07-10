# encoding: utf-8
"""
S34 Research: Post-TP continuation scalp
After ETH 500K hits TP (+60 bps), does price continue up?

Tests:
  1. How far does price go BEYOND TP in the next 60/120/300 seconds?
  2. What % of trades would profit from a re-entry after TP?
  3. Is the post-TP continuation predictable (BTC trend, cascade size, cnt)?
  4. Also: funding rate context - does high funding = stronger cascade?
"""
from __future__ import annotations
import sqlite3, json, sys
from pathlib import Path
from datetime import datetime, timezone

ROOT     = Path("D:/eclipse_scalper")
MICRO_DB = ROOT / "data" / "microstructure.db"
INTEL_DB = ROOT / "data" / "s34_intelligence.db"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.storage import production as PR
from ami.storage import research_reader as RR
ETH_RULE = "ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30"
TP_BPS   = 60.0
HOLD_AFTER_TP = [30, 60, 120, 300]  # seconds

def mark_price_at(micro, symbol, ts_ms):
    row = micro.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (symbol, ts_ms)).fetchone()
    return row[0] if row else None

def bps_from(p_ref, p_now):
    if not p_ref or not p_now or p_ref == 0:
        return None
    return (p_now - p_ref) / p_ref * 10000

def stats(vals):
    if not vals:
        return None
    s = sorted(vals)
    pos = sum(1 for x in vals if x > 0)
    return {"n": len(vals), "wr": pos/len(vals), "med": s[len(s)//2],
            "mean": sum(vals)/len(vals), "p75": s[int(len(s)*0.75)],
            "p25": s[int(len(s)*0.25)]}


def _find_tp_crossing(micro, gt_ms, le_ms, price_threshold):
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `_find_tp_crossing_v2` (BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-
    CONSUMER-MIGRATION-V8). No longer called by main()'s live path.
    Symbol is hardcoded to 'ETHUSDT' via a SQL literal (not a
    parameter) -- this file's only caller never uses another symbol."""
    return micro.execute("""
        SELECT ts_ms, mark_price FROM mark_prices
        WHERE symbol='ETHUSDT' AND ts_ms > ? AND ts_ms <= ? AND mark_price >= ?
        ORDER BY ts_ms ASC LIMIT 1
    """, (gt_ms, le_ms, price_threshold)).fetchone()


def _find_tp_crossing_v2(root, gt_ms, le_ms, price_threshold, source_db_path=None):
    """Reader-backed replacement for `_find_tp_crossing`, via
    `plan_read`/`execute_read`. Symbol is hardcoded to `'ETHUSDT'`,
    matching the oracle's own embedded SQL literal -- not a genuine
    runtime parameter in this file (the only call site never passes
    another symbol).

    Range semantics: same exclusive-lower/inclusive-upper mapping as
    `research_s34_hold_sweep.py`/`research_s34_session_analysis.py`'s
    identically-shaped range clause -- `start_ms=gt_ms+1,
    end_ms=le_ms+1` reproduces `ts_ms>gt_ms AND ts_ms<=le_ms`
    bit-for-bit.

    LIMIT-1-with-extra-filter semantics: `execute_read` streams rows in
    canonical `(ts_ms ASC, id ASC)` order -- a strict refinement of the
    oracle's `ORDER BY ts_ms ASC` (no tie-break), proven equivalent by
    parity test -- with the `mark_price>=price_threshold` predicate
    applied per-row via `filters=`. The FIRST qualifying row from that
    ordered, filtered stream is exactly the oracle's
    `ORDER BY ts_ms ASC LIMIT 1` result: this function returns as soon
    as it finds one, short-circuiting the stream rather than
    materializing the whole range."""
    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=int(gt_ms) + 1, end_ms=int(le_ms) + 1)
    result = RR.execute_read(plan, columns=("ts_ms", "mark_price"),
                              filters=(("mark_price", ">=", float(price_threshold)),),
                              source_db_path=source_db_path)
    for ts, px in result.iter_rows():
        return (int(ts), float(px))
    return None

def main():
    intel = sqlite3.connect(f"file:{INTEL_DB}?mode=ro", uri=True)
    micro = sqlite3.connect(f"file:{MICRO_DB}?mode=ro", uri=True)
    root, _ = PR.resolve_production_root()
    source_db_path = str(MICRO_DB)

    trades = intel.execute("""
        SELECT trade_id, entry_ts_ms, net_bps, exit_reason, trade_json
        FROM s34_trades
        WHERE rule_name=? AND status='CLOSED' AND net_bps IS NOT NULL
        ORDER BY entry_ts_ms
    """, (ETH_RULE,)).fetchall()
    intel.close()

    tp_trades = [(tid, ts, float(net), exit_r, tj)
                 for tid, ts, net, exit_r, tj in trades
                 if exit_r and 'TP' in exit_r]

    print("=" * 70)
    print("POST-TP CONTINUATION SCALP ANALYSIS")
    print(f"Rule: {ETH_RULE}")
    print(f"TP trades: N={len(tp_trades)} / {len(trades)} total")
    print("=" * 70)
    print()

    # For each TP trade: simulate re-entry at TP price, hold for HOLD_AFTER_TP seconds
    results = []
    for tid, entry_ms, net, exit_r, tj in tp_trades:
        sig = json.loads(tj).get('signal') or {} if tj else {}
        cnt = sig.get('liq_count')
        total_k = (sig.get('liq_total_notional') or 0) / 1000

        p_entry = mark_price_at(micro, 'ETHUSDT', entry_ms)
        if not p_entry:
            continue

        # Estimate TP exit time and price
        # TP price = entry + 60 bps
        p_tp = p_entry * (1 + TP_BPS / 10000)

        # Find approximate TP exit timestamp from mark_prices
        # Look for when price first crossed TP
        tp_row = _find_tp_crossing_v2(root, entry_ms, entry_ms + 510000, p_tp * 0.9995, source_db_path=source_db_path)

        if not tp_row:
            continue

        tp_ts, tp_price = tp_row
        hold_bps = {}
        for hold_s in HOLD_AFTER_TP:
            p_after = mark_price_at(micro, 'ETHUSDT', tp_ts + hold_s * 1000)
            ret = bps_from(p_tp, p_after) if p_after else None
            hold_bps[hold_s] = ret

        # Funding rate at entry
        fr_row = micro.execute("""
            SELECT funding_rate FROM funding_rates
            WHERE symbol='ETHUSDT' AND ts_ms <= ? ORDER BY ts_ms DESC LIMIT 1
        """, (entry_ms,)).fetchone()
        funding = fr_row[0] * 10000 if fr_row else None  # convert to bps

        results.append({
            "tid": tid, "entry_ms": entry_ms, "net_bps": net,
            "cnt": cnt, "total_k": total_k,
            "hold_bps": hold_bps, "funding_bps": funding,
            "tp_to_entry_sec": (tp_ts - entry_ms) / 1000,
        })

    print(f"TP trades with price path data: N={len(results)}")
    print()

    # Overall post-TP analysis
    print("POST-TP RETURNS (re-entry at TP, exit after N seconds):")
    print(f"  {'Hold':>8}  {'N':>4}  {'WR':>5}  {'med':>7}  {'mean':>7}  {'p75':>7}  profitable?")
    for h in HOLD_AFTER_TP:
        vals = [r['hold_bps'][h] for r in results if r['hold_bps'].get(h) is not None]
        s = stats(vals)
        if s:
            verdict = "YES" if s['med'] > 8 else ("MAYBE" if s['med'] > 0 else "NO")
            print(f"  {h:>5}s      {s['n']:>4}  {s['wr']*100:.0f}%  {s['med']:>+7.1f}  "
                  f"{s['mean']:>+7.1f}  {s['p75']:>+7.1f}  {verdict}")

    print()
    print("TIME TO TP (how quickly do TP trades hit 60 bps):")
    tp_times = [r['tp_to_entry_sec'] for r in results]
    if tp_times:
        s = sorted(tp_times)
        print(f"  Median: {s[len(s)//2]:.0f}s  p25: {s[len(s)//4]:.0f}s  p75: {s[int(len(s)*0.75)]:.0f}s  fastest: {s[0]:.0f}s")

    # Funding rate context
    print()
    print("FUNDING RATE at entry vs ETH outcome (ALL trades incl SL/BE):")
    intel2 = sqlite3.connect(f"file:{INTEL_DB}?mode=ro", uri=True)
    all_trades = intel2.execute("""
        SELECT entry_ts_ms, net_bps FROM s34_trades
        WHERE rule_name=? AND status='CLOSED' AND net_bps IS NOT NULL
    """, (ETH_RULE,)).fetchall()
    intel2.close()

    fr_data = {"hi": [], "med": [], "lo": []}
    for ts, net in all_trades:
        fr_row = micro.execute("""
            SELECT funding_rate FROM funding_rates
            WHERE symbol='ETHUSDT' AND ts_ms <= ? ORDER BY ts_ms DESC LIMIT 1
        """, (ts,)).fetchone()
        if fr_row:
            fr_bps = fr_row[0] * 10000
            if fr_bps > 1.0:
                fr_data["hi"].append(float(net))
            elif fr_bps > 0:
                fr_data["med"].append(float(net))
            else:
                fr_data["lo"].append(float(net))

    print(f"  {'Funding':>15}  {'N':>4}  {'WR':>5}  {'med':>7}  {'SL%':>5}")
    for label, key, desc in [("high (>1 bps)", "hi", "funding > 1 bps"),
                               ("med (0-1 bps)", "med", "funding 0-1 bps"),
                               ("low/neg (<0)", "lo", "funding <= 0")]:
        nets = fr_data[key]
        if not nets:
            print(f"  {label:>15}: N=0")
            continue
        s = sorted(nets)
        sl = sum(1 for x in nets if x < -20)
        wr = sum(1 for x in nets if x > 0) / len(nets)
        med = s[len(s)//2]
        print(f"  {label:>15}: N={len(nets):>3}  WR={wr*100:.0f}%  med={med:>+7.1f}  SL={sl}({sl/len(nets)*100:.0f}%)")

    micro.close()
    print()
    print("NOTE: Shadow research. No runner change without N>=50 OOS evidence.")


if __name__ == "__main__":
    main()
