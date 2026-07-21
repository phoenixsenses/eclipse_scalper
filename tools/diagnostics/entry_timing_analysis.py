# encoding: utf-8
"""
Entry timing analysis: does delaying entry improve performance?
Measures MAE (max adverse excursion) and hypothetical delay-entry prices.
"""
import sqlite3, json
from pathlib import Path
from datetime import datetime, timezone

ROOT     = Path("D:/eclipse_scalper")
INTEL_DB = ROOT / "data" / "s34_intelligence.db"
MICRO_DB = ROOT / "data" / "microstructure.db"

RULES = [
    "ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30",
    "SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30",
]

OFFSETS_SEC = [0, 10, 20, 30, 45, 60, 90, 120, 180, 300]
MAE_WINDOW_SEC = 600   # look for worst drawdown in first 10 min

intel = sqlite3.connect(f"file:{INTEL_DB}?mode=ro", uri=True)
micro = sqlite3.connect(f"file:{MICRO_DB}?mode=ro", uri=True)

rows = intel.execute(
    "SELECT trade_id, rule_name, entry_ts_ms, exit_ts_ms, net_bps, exit_reason, trade_json "
    "FROM s34_trades WHERE status='CLOSED' AND net_bps IS NOT NULL "
    "AND rule_name IN (?,?) ORDER BY entry_ts_ms",
    RULES
).fetchall()
intel.close()

def get_mark(symbol, ts_ms, window_ms=4000):
    """Nearest mark price within +-window around ts_ms."""
    r = micro.execute(
        "SELECT mark_price FROM mark_prices "
        "WHERE symbol=? AND ts_ms BETWEEN ? AND ? "
        "ORDER BY ABS(ts_ms - ?) LIMIT 1",
        (symbol, ts_ms - window_ms, ts_ms + window_ms, ts_ms)
    ).fetchone()
    return r[0] if r else None

def get_min_price(symbol, ts_start_ms, ts_end_ms):
    """Minimum mark price in a window (worst case for a long)."""
    r = micro.execute(
        "SELECT MIN(mark_price) FROM mark_prices "
        "WHERE symbol=? AND ts_ms BETWEEN ? AND ?",
        (symbol, ts_start_ms, ts_end_ms)
    ).fetchone()
    return r[0] if r else None

def get_max_price(symbol, ts_start_ms, ts_end_ms):
    r = micro.execute(
        "SELECT MAX(mark_price) FROM mark_prices "
        "WHERE symbol=? AND ts_ms BETWEEN ? AND ?",
        (symbol, ts_start_ms, ts_end_ms)
    ).fetchone()
    return r[0] if r else None

# ── Build per-trade timing data ──────────────────────────────────────────────
trades = []
for tid, rule, entry_ms, exit_ms, net_bps, exit_r, tj in rows:
    try:
        t = json.loads(tj)
        symbol = t.get("symbol") or ("ETHUSDT" if "ETH" in rule else "SOLUSDT")
        ep = float(t.get("entry_price") or 0)
        if ep <= 0:
            continue

        # Mark price at each delay offset
        delay_prices = {}
        for sec in OFFSETS_SEC:
            px = get_mark(symbol, entry_ms + sec * 1000)
            delay_prices[sec] = px

        # MAE: worst mark price in first MAE_WINDOW_SEC
        mae_low  = get_min_price(symbol, entry_ms, entry_ms + MAE_WINDOW_SEC * 1000)
        mae_high = get_max_price(symbol, entry_ms, entry_ms + MAE_WINDOW_SEC * 1000)

        mae_bps = None
        if mae_low and ep:
            mae_bps = (mae_low - ep) / ep * 10000  # negative = adverse for long

        is_sl = exit_r and "SL" in exit_r

        trades.append({
            "tid": tid, "rule": rule, "symbol": symbol,
            "entry_ms": entry_ms, "entry_px": ep,
            "net_bps": float(net_bps), "exit_r": exit_r, "is_sl": is_sl,
            "delay_prices": delay_prices,
            "mae_low": mae_low, "mae_high": mae_high, "mae_bps": mae_bps,
        })
    except Exception:
        pass

micro.close()

N = len(trades)
print("=" * 70)
print(f"ENTRY TIMING ANALYSIS  N={N} closed trades")
print("=" * 70)

# ── 1. MAE Distribution (all trades) ─────────────────────────────────────────
print("\n--- MAX ADVERSE EXCURSION (first 10 min after entry, LONG) ---\n")
mae_vals = [t["mae_bps"] for t in trades if t["mae_bps"] is not None]
mae_vals_sl = [t["mae_bps"] for t in trades if t["mae_bps"] is not None and t["is_sl"]]
mae_vals_win = [t["mae_bps"] for t in trades if t["mae_bps"] is not None and not t["is_sl"]]

def pct(vals, p):
    if not vals: return None
    s = sorted(vals)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s)-1)]

if mae_vals:
    print(f"  All trades   N={len(mae_vals):>3}  "
          f"median={pct(mae_vals,50):+.1f} bps  "
          f"p10={pct(mae_vals,10):+.1f}  p25={pct(mae_vals,25):+.1f}  p75={pct(mae_vals,75):+.1f}")
if mae_vals_win:
    print(f"  Winners      N={len(mae_vals_win):>3}  "
          f"median={pct(mae_vals_win,50):+.1f} bps  "
          f"p10={pct(mae_vals_win,10):+.1f}  p25={pct(mae_vals_win,25):+.1f}  p75={pct(mae_vals_win,75):+.1f}")
if mae_vals_sl:
    print(f"  SL trades    N={len(mae_vals_sl):>3}  "
          f"median={pct(mae_vals_sl,50):+.1f} bps  "
          f"p10={pct(mae_vals_sl,10):+.1f}  p25={pct(mae_vals_sl,25):+.1f}  p75={pct(mae_vals_sl,75):+.1f}")

# ── 2. Delay benefit table ────────────────────────────────────────────────────
print("\n--- HYPOTHETICAL DELAY ENTRY: avg price improvement vs t=0 ---\n")
print(f"  {'Delay':>8}  {'N w/data':>8}  {'Avg px chg (bps)':>18}  {'Better %':>9}  {'SL avoided?':>12}")
print("  " + "-" * 62)

t0_prices = {t["tid"]: t["delay_prices"].get(0) or t["entry_px"] for t in trades}

for sec in OFFSETS_SEC[1:]:
    deltas = []
    better = 0
    for t in trades:
        px0 = t["entry_px"]
        pxd = t["delay_prices"].get(sec)
        if pxd and px0:
            delta_bps = (pxd - px0) / px0 * 10000  # negative = cheaper entry for long
            deltas.append(delta_bps)
            if delta_bps < 0:  # lower price = better for BUY
                better += 1
    if not deltas:
        continue
    avg = sum(deltas) / len(deltas)
    better_pct = better / len(deltas) * 100
    print(f"  {sec:>6}s    {len(deltas):>8}  {avg:>+17.1f} bps  {better_pct:>8.0f}%")

# ── 3. SL trade deep dive ─────────────────────────────────────────────────────
sl_trades = [t for t in trades if t["is_sl"]]
print(f"\n--- SL TRADES DEEP DIVE (N={len(sl_trades)}) ---\n")
if sl_trades:
    print(f"  {'Date':>5}  {'Rule':>3}  {'MAE':>7}  {'Net':>7}  "
          f"{'px@30s vs entry':>16}  {'px@60s vs entry':>16}")
    print("  " + "-" * 62)
    for t in sl_trades:
        dt = datetime.fromtimestamp(t["entry_ms"]/1000, tz=timezone.utc).strftime("%m/%d")
        sym = "ETH" if "ETH" in t["rule"] else "SOL"
        mae_s = f"{t['mae_bps']:+.1f}" if t["mae_bps"] is not None else "   ?"
        p30 = t["delay_prices"].get(30)
        p60 = t["delay_prices"].get(60)
        d30 = f"{(p30-t['entry_px'])/t['entry_px']*10000:+.1f}" if p30 else "   ?"
        d60 = f"{(p60-t['entry_px'])/t['entry_px']*10000:+.1f}" if p60 else "   ?"
        print(f"  {dt:>5}  {sym:>3}  {mae_s:>7}  {t['net_bps']:>+7.1f}  "
              f"{d30:>16} bps  {d60:>16} bps")

# ── 4. Rule breakdown ─────────────────────────────────────────────────────────
print("\n--- BY RULE ---\n")
for rule in RULES:
    rt = [t for t in trades if t["rule"] == rule]
    if not rt:
        continue
    short = "ETH_500K" if "500K" in rule else "SOL_200K"
    mae_r = [t["mae_bps"] for t in rt if t["mae_bps"] is not None]
    sl_r  = [t for t in rt if t["is_sl"]]
    mae_med = f"{pct(mae_r,50):+.1f}" if mae_r else "?"
    mae_p10 = f"{pct(mae_r,10):+.1f}" if mae_r else "?"
    print(f"  {short}  N={len(rt)}  SL={len(sl_r)}  MAE median={mae_med} bps  MAE p10={mae_p10} bps")

# ── 5. Spike pattern: entry-to-peak-to-trough ─────────────────────────────────
print("\n--- SPIKE PATTERN: first 2 min after entry ---\n")
print("  How often does price go UP first, then DOWN below entry?")
spike_count = 0
direct_down = 0
neutral = 0
coverage = 0
for t in trades:
    p30 = t["delay_prices"].get(30)
    p60 = t["delay_prices"].get(60)
    p120 = t["delay_prices"].get(120)
    ep  = t["entry_px"]
    if not (p30 and p60 and p120):
        continue
    coverage += 1
    peak_early = max(p30, p60)
    trough_late = min(p60, p120)
    if peak_early > ep * 1.0002 and trough_late < ep:
        spike_count += 1  # went up then came back below entry
    elif p120 < ep and p30 < ep:
        direct_down += 1  # went straight down
    else:
        neutral += 1

if coverage:
    print(f"  N w/coverage={coverage}")
    print(f"  Up-then-below-entry spike: {spike_count} ({spike_count/coverage*100:.0f}%)")
    print(f"  Straight down from entry:  {direct_down} ({direct_down/coverage*100:.0f}%)")
    print(f"  Held above entry at 2min:  {neutral} ({neutral/coverage*100:.0f}%)")

print()
print("=" * 70)
print("VERDICT")
print("=" * 70)
if mae_vals:
    med_mae = pct(mae_vals, 50)
    p10_mae = pct(mae_vals, 10)
    print(f"Typical immediate dip after entry: {med_mae:+.1f} bps (median MAE)")
    print(f"Worst 10% of entries see:          {p10_mae:+.1f} bps dip in first 10 min")
    if len(mae_vals) >= 20:
        # Check if delay 30s or 60s gives systematic edge
        d30_vals = []
        d60_vals = []
        for t in trades:
            ep = t["entry_px"]
            p30 = t["delay_prices"].get(30)
            p60 = t["delay_prices"].get(60)
            if p30 and ep:
                d30_vals.append((p30 - ep) / ep * 10000)
            if p60 and ep:
                d60_vals.append((p60 - ep) / ep * 10000)
        avg30 = sum(d30_vals) / len(d30_vals) if d30_vals else 0
        avg60 = sum(d60_vals) / len(d60_vals) if d60_vals else 0
        print(f"Avg price at t+30s vs entry:       {avg30:+.1f} bps ({'>0=worse entry' if avg30>0 else '<0=better entry'})")
        print(f"Avg price at t+60s vs entry:       {avg60:+.1f} bps ({'>0=worse entry' if avg60>0 else '<0=better entry'})")
        if avg30 < -3:
            print("-> 30s delay shows meaningful entry improvement. Worth testing.")
        elif avg30 > 3:
            print("-> Price typically rises after signal. Delay hurts entry price.")
        else:
            print("-> No strong systematic delay benefit detected.")
