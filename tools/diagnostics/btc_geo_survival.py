# encoding: utf-8
"""BTC 1M geometry analysis + survival simulation. $35 start, 85% x 40x."""
import sqlite3, json
from pathlib import Path
from datetime import datetime, timezone

ROOT     = Path("D:/eclipse_scalper")
INTEL_DB = ROOT / "data" / "s34_intelligence.db"
BTC_RULE = "BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30"

conn = sqlite3.connect("file:" + str(INTEL_DB) + "?mode=ro", uri=True)
rows = conn.execute(
    "SELECT trade_id, entry_ts_ms, net_bps, exit_reason, trade_json "
    "FROM s34_trades WHERE status='CLOSED' AND net_bps IS NOT NULL "
    "AND rule_name=? ORDER BY entry_ts_ms",
    (BTC_RULE,)
).fetchall()
conn.close()

trades = []
for tid, ts_ms, net_bps, exit_r, tj in rows:
    try:
        t = json.loads(tj)
        sig = t.get("signal") or {}
        cas   = sig.get("liq_total_notional")
        cnt   = sig.get("liq_count")
        mx    = sig.get("liq_max_notional")
        share = (mx / cas * 100) if (cas and mx and cas > 0) else None
        ep    = float(t.get("entry_price") or 0)
        trades.append({
            "tid": tid, "ts_ms": ts_ms,
            "net": float(net_bps), "exit": exit_r,
            "cas": cas, "cnt": cnt, "share": share, "ep": ep,
        })
    except Exception:
        pass

print("=" * 65)
print(f"BTC 1M DISTRIBUTED — N={len(trades)} closed trades")
print("=" * 65)

def stats(group):
    nets = [t["net"] for t in group]
    if not nets:
        return None
    wr  = sum(1 for n in nets if n > 0) / len(nets) * 100
    med = sorted(nets)[len(nets)//2]
    return {"n": len(nets), "wr": wr, "median": med, "cum": sum(nets)}

print("\n--- GEOMETRY BREAKDOWN ---\n")

# Cascade notional
print("Cascade notional split:")
print(f"  {'Bucket':>10}  {'N':>3}  {'WR':>5}  {'Median':>7}  {'Cum':>8}")
for label, fn in [
    ("1M-2M",  lambda t: t["cas"] and 1_000_000 <= t["cas"] < 2_000_000),
    ("2M-5M",  lambda t: t["cas"] and 2_000_000 <= t["cas"] < 5_000_000),
    ("5M-10M", lambda t: t["cas"] and 5_000_000 <= t["cas"] < 10_000_000),
    (">10M",   lambda t: t["cas"] and t["cas"] >= 10_000_000),
    ("unknown",lambda t: t["cas"] is None),
]:
    g = [t for t in trades if fn(t)]
    s = stats(g)
    if s:
        print(f"  {label:>10}  {s['n']:>3}  {s['wr']:>4.0f}%  {s['median']:>+7.1f}  {s['cum']:>+8.1f}")

# liq_count
print("\nliq_count split:")
print(f"  {'Bucket':>10}  {'N':>3}  {'WR':>5}  {'Median':>7}  {'Cum':>8}")
for label, fn in [
    ("<=5",    lambda t: t["cnt"] is not None and t["cnt"] <= 5),
    ("6-10",   lambda t: t["cnt"] is not None and 6 <= t["cnt"] <= 10),
    ("11-20",  lambda t: t["cnt"] is not None and 11 <= t["cnt"] <= 20),
    (">20",    lambda t: t["cnt"] is not None and t["cnt"] > 20),
    ("unknown",lambda t: t["cnt"] is None),
]:
    g = [t for t in trades if fn(t)]
    s = stats(g)
    if s:
        print(f"  {label:>10}  {s['n']:>3}  {s['wr']:>4.0f}%  {s['median']:>+7.1f}  {s['cum']:>+8.1f}")

# single share
print("\nsingle dominant share split:")
print(f"  {'Bucket':>10}  {'N':>3}  {'WR':>5}  {'Median':>7}  {'Cum':>8}")
for label, fn in [
    ("<50%",   lambda t: t["share"] is not None and t["share"] < 50),
    ("50-80%", lambda t: t["share"] is not None and 50 <= t["share"] < 80),
    (">=80%",  lambda t: t["share"] is not None and t["share"] >= 80),
    ("unknown",lambda t: t["share"] is None),
]:
    g = [t for t in trades if fn(t)]
    s = stats(g)
    if s:
        print(f"  {label:>10}  {s['n']:>3}  {s['wr']:>4.0f}%  {s['median']:>+7.1f}  {s['cum']:>+8.1f}")

# Per-trade detail
print("\nPer-trade detail:")
print(f"  {'Date':>5}  {'Net':>6}  {'Exit':>2}  {'Cascade':>10}  {'LiqCnt':>6}  {'Share':>6}  {'BTC px':>9}")
for t in trades:
    dt    = datetime.fromtimestamp(t["ts_ms"]/1000, tz=timezone.utc).strftime("%m/%d")
    cas_s = f"${t['cas']/1_000_000:.2f}M" if t["cas"] else "?"
    cnt_s = str(t["cnt"]) if t["cnt"] is not None else "?"
    sh_s  = f"{t['share']:.0f}%" if t["share"] else "?"
    ep_s  = f"${t['ep']:,.0f}" if t["ep"] else "?"
    flag  = " <SL" if t["exit"] and "SL" in t["exit"] else ""
    ex2   = (t["exit"] or "")[:2]
    print(f"  {dt:>5}  {t['net']:>+6.1f}  {ex2:>2}  {cas_s:>10}  {cnt_s:>6}  {sh_s:>6}  {ep_s:>9}{flag}")

# ── SURVIVAL SIM ───────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("BTC 1M SURVIVAL SIM — $35 start, 85% x 40x")
print("=" * 65)

MARGIN_PCT = 0.85
LEVERAGE   = 40
bal = 35.0
peak = 35.0
max_dd = 0.0
sl_count = be_count = tp_count = 0
consec_sl = max_consec_sl = 0

print(f"\n  {'Date':>5}  {'Exit':>2}  {'bps':>6}  {'Before':>9}  {'PnL':>8}  {'After':>9}")
print("  " + "-" * 53)
for t in trades:
    margin   = bal * MARGIN_PCT
    notional = margin * LEVERAGE
    pnl      = notional * t["net"] / 10000
    new_bal  = max(bal + pnl, 0.01)
    dt = datetime.fromtimestamp(t["ts_ms"]/1000, tz=timezone.utc).strftime("%m/%d")
    ex2 = (t["exit"] or "")[:2]

    if t["exit"] and "SL" in t["exit"]:
        sl_count += 1; consec_sl += 1
        max_consec_sl = max(max_consec_sl, consec_sl)
    elif t["exit"] and "BE" in t["exit"]:
        be_count += 1; consec_sl = 0
    else:
        tp_count += 1; consec_sl = 0

    dd = (peak - new_bal) / peak * 100 if peak > 0 else 0
    peak = max(peak, new_bal)
    max_dd = max(max_dd, dd)

    marker = " <-- SL" if t["exit"] and "SL" in t["exit"] else ""
    print(f"  {dt:>5}  {ex2:>2}  {t['net']:>+6.1f}  ${bal:>8.2f}  ${pnl:>+7.2f}  ${new_bal:>8.2f}{marker}")
    bal = new_bal

n = len(trades)
print("  " + "-" * 53)
print(f"  Start: $35.00  ->  Final: ${bal:.2f}")
print(f"  TP={tp_count}  BE={be_count}  SL={sl_count}  |  Max consec SL={max_consec_sl}")
print(f"  Max drawdown={max_dd:.1f}%  |  WR={(tp_count+be_count)/n*100:.0f}%" if n else "  N=0")
print(f"  Total gain: ${bal-35:+.2f}  ({(bal/35-1)*100:+.0f}%)")

# ── VERDICT ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("VERDICT")
print("=" * 65)
print(f"N={n} — {'PRELIMINARY (N<30)' if n < 30 else 'SUFFICIENT'}")
print("Geometry signals: check liq_count split for clean/weak pattern")
print("Live candidate: only if N grows and edge holds in OOS forward data")
