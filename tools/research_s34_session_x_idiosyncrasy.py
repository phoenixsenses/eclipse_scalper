# encoding: utf-8
"""
S34 Research: Session x Idiosyncrasy interaction

Two strong findings so far:
  1. Europe 08-14 UTC best for ETH BUY (OOS WR=82%)
  2. Systemic cascades (ETH+BTC+SOL) best for SELL (OOS WR=75%)
     Idiosyncratic SELL worst (OOS WR=36%)

Question: Do these compound?
  - Europe + Systemic = super strong?
  - Asia + Idiosyncratic = worst of both worlds?
  - Does session matter MORE for idiosyncratic (no market anchor)?

Also: intraday cascade frequency — does Nth cascade on same day degrade?
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

ROOT     = Path("D:/eclipse_scalper")
MICRO_DB = ROOT / "data" / "microstructure.db"

BUCKET_SEC      = 30
CO_WINDOW_MS    = 60_000
BTC_CO_THRESHOLD = 1_000_000
SOL_CO_THRESHOLD =   100_000
FEE_BPS         = 8.0

SESSIONS = [
    ("Asia  ", 0,  8),
    ("Europe", 8,  14),
    ("US    ", 14, 22),
    ("Late  ", 22, 24),
]

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

def has_co(cascade_list, ts_ms):
    lo, hi = ts_ms - CO_WINDOW_MS, ts_ms + CO_WINDOW_MS
    for (ct, *_) in cascade_list:
        if lo <= ct <= hi: return True
        if ct > hi: break
    return False

def session_of(ts_ms):
    h = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour
    for name, h0, h1 in SESSIONS:
        if h0 <= h < h1:
            return name
    return "Late  "

def day_key(ts_ms):
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")

def st(label, nets, width=32):
    if not nets:
        return f"  {label:{width}} N=  0"
    s = sorted(nets)
    sl  = sum(1 for x in nets if x < -20)
    wr  = sum(1 for x in nets if x > 0) / len(nets)
    med = s[len(s) // 2]
    mn  = sum(nets) / len(nets)
    return (f"  {label:{width}} N={len(nets):>4}  WR={wr*100:>4.0f}%  "
            f"med={med:>+7.1f}  mean={mn:>+6.1f}  SL={sl}({sl/len(nets)*100:>3.0f}%)")

def main():
    micro = sqlite3.connect(f"file:{MICRO_DB}?mode=ro", uri=True)

    btc_buy  = build_cascades(micro, "BTCUSDT", "BUY",  BTC_CO_THRESHOLD, 1)
    btc_sell = build_cascades(micro, "BTCUSDT", "SELL", BTC_CO_THRESHOLD, 1)
    sol_buy  = build_cascades(micro, "SOLUSDT", "BUY",  SOL_CO_THRESHOLD, 1)
    sol_sell = build_cascades(micro, "SOLUSDT", "SELL", SOL_CO_THRESHOLD, 1)

    print("S34 SESSION x IDIOSYNCRASY INTERACTION")
    print("Combining: session timing + co-cascade type")

    for route in ROUTES:
        cascades = build_cascades(micro, route["symbol"], route["side"],
                                  route["threshold"], route["cnt_min"])

        same_btc = btc_buy  if route["side"] == "BUY" else btc_sell
        same_sol = sol_buy  if route["side"] == "BUY" else sol_sell

        # Day-cascade counter: which Nth signal is this on the same day?
        day_counts: dict[str, int] = {}

        results = []
        for ts, total, cnt in cascades:
            net, exit_r = simulate(micro, route["symbol"], ts,
                                   route["tp"], route["sl"], route["be"],
                                   route["hold"], route["direction"])
            if net is None:
                continue

            has_btc = has_co(same_btc, ts)
            has_sol = has_co(same_sol, ts)

            if has_btc and has_sol:
                co_type = "SYSTEMIC"
            elif has_btc or has_sol:
                co_type = "PARTIAL"
            else:
                co_type = "IDIO"

            sess = session_of(ts)
            dk = day_key(ts)
            day_counts[dk] = day_counts.get(dk, 0) + 1
            nth = day_counts[dk]  # 1st, 2nd, 3rd... signal of the day

            results.append({
                "ts": ts, "net": net, "exit": exit_r,
                "co_type": co_type, "session": sess, "nth": nth,
            })

        split_idx = int(len(results) * 0.70)
        split_ts  = results[split_idx]["ts"] if split_idx < len(results) else 0
        split_dt  = datetime.fromtimestamp(split_ts/1000, tz=timezone.utc).strftime("%Y-%m-%d")
        test = results[split_idx:]

        print(f"\n{'='*72}")
        print(f"{route['label']}  |  OOS: {split_dt}  test N={len(test)}")
        print(f"{'='*72}")

        # --- SESSION x CO_TYPE grid (OOS only) ---
        print("OOS: SESSION x CO_TYPE grid")
        print(f"  {'':32} {'IDIO':>22}  {'PARTIAL':>22}  {'SYSTEMIC':>22}")
        for sname, h0, h1 in SESSIONS:
            row_parts = [f"  {sname} ({h0:02d}-{h1:02d} UTC)           "]
            for ct in ["IDIO", "PARTIAL", "SYSTEMIC"]:
                nets = [r["net"] for r in test
                        if r["session"] == sname and r["co_type"] == ct]
                if not nets:
                    row_parts.append(f"{'N=0':>22}")
                else:
                    s = sorted(nets)
                    wr = sum(1 for x in nets if x > 0) / len(nets)
                    med = s[len(s)//2]
                    sl  = sum(1 for x in nets if x < -20)
                    row_parts.append(f"N={len(nets):>2} WR={wr*100:.0f}% med={med:>+5.0f} SL={sl}")
            print("".join(row_parts))

        # --- SESSION only (OOS) ---
        print("\nOOS: Session breakdown")
        for sname, h0, h1 in SESSIONS:
            nets = [r["net"] for r in test if r["session"] == sname]
            print(st(f"{sname} ({h0:02d}-{h1:02d})", nets))

        # --- CO_TYPE only (OOS) ---
        print("\nOOS: Co-cascade type breakdown")
        for ct in ["IDIO", "PARTIAL", "SYSTEMIC"]:
            nets = [r["net"] for r in test if r["co_type"] == ct]
            print(st(ct, nets))
        print(st("ALL", [r["net"] for r in test]))

        # --- Intraday Nth signal degradation (all data) ---
        print("\nINTRADAY: Nth cascade of the day (all data)")
        for n in [1, 2, 3, 4]:
            nets = [r["net"] for r in results if r["nth"] == n]
            print(st(f"Signal #{n} of day", nets))
        nets_5plus = [r["net"] for r in results if r["nth"] >= 5]
        print(st("Signal #5+ of day", nets_5plus))

        # --- Best combo vs worst combo (OOS) ---
        print("\nOOS: Best/worst combos")
        combos = {}
        for r in test:
            key = f"{r['session'].strip()}+{r['co_type']}"
            combos.setdefault(key, []).append(r["net"])
        ranked = sorted(combos.items(), key=lambda x: (
            sum(x[1]) / len(x[1]) if x[1] else -999), reverse=True)
        for key, nets in ranked:
            if len(nets) >= 3:
                print(st(key, nets, width=22))

    micro.close()
    print("\nNOTE: Shadow research. No action without N>=50 per cell.")

if __name__ == "__main__":
    main()
