# encoding: utf-8
"""3-way comparison: unfiltered / current live geo / current + ETH count+share filter."""
import sqlite3, json
from pathlib import Path
from datetime import datetime, timezone

ROOT     = Path("D:/eclipse_scalper")
INTEL_DB = ROOT / "data" / "s34_intelligence.db"

RULES = [
    "ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30",
    "SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30",
]

intel = sqlite3.connect(f"file:{INTEL_DB}?mode=ro", uri=True)
weak_ids = {r[0] for r in intel.execute(
    "SELECT trade_id FROM s34_shadow_geometry_tags WHERE tag='SOL_WEAK_GEOMETRY_SHADOW'"
).fetchall()}
rows = intel.execute(
    "SELECT trade_id, rule_name, entry_ts_ms, net_bps, exit_reason, trade_json "
    "FROM s34_trades WHERE status='CLOSED' AND net_bps IS NOT NULL "
    "AND rule_name IN (?,?) ORDER BY entry_ts_ms", RULES
).fetchall()
intel.close()

trades = []
for tid, rule, entry_ms, net_bps, exit_r, tj in rows:
    try:
        t   = json.loads(tj)
        sig = t.get("signal") or {}
        cnt   = sig.get("liq_count")
        cas   = sig.get("liq_total_notional")
        mx    = sig.get("liq_max_notional")
        share = (mx / cas * 100) if (cas and mx and cas > 0) else None
        sym   = "ETH" if "ETH" in rule else "SOL"
        dt    = datetime.fromtimestamp(entry_ms/1000, tz=timezone.utc).strftime("%m/%d")
        is_sol = "SOL" in rule
        is_eth = "ETH" in rule

        # Filter A: current live geo (SOL only)
        geo_blocked = tid in weak_ids  # SOL weak geometry

        # Filter B: proposed new (ETH cnt<=7 OR ETH share>=80%)
        eth_cnt_weak   = is_eth and cnt  is not None and cnt  <= 7
        eth_share_weak = is_eth and share is not None and share >= 80.0
        new_blocked = eth_cnt_weak or eth_share_weak

        trades.append({
            "tid": tid, "sym": sym, "dt": dt, "rule": rule,
            "net": float(net_bps), "exit_r": exit_r or "",
            "cnt": cnt, "share": share, "cas": cas,
            "geo_blocked": geo_blocked,
            "new_blocked": new_blocked,
            "is_sl": bool(exit_r and "SL" in exit_r),
        })
    except Exception:
        pass

MARGIN_PCT = 0.85
LEVERAGE   = 40
START      = 35.0

def run_sim(trade_list):
    bal = START; peak = START; max_dd = 0.0
    tp = be = sl = 0
    for t in trade_list:
        notional = bal * MARGIN_PCT * LEVERAGE
        pnl = notional * t["net"] / 10000
        new_bal = max(bal + pnl, 0.01)
        if "SL" in t["exit_r"]: sl += 1
        elif "BE" in t["exit_r"]: be += 1
        else: tp += 1
        dd = (peak - new_bal) / peak * 100 if peak > 0 else 0
        if new_bal > peak: peak = new_bal
        max_dd = max(max_dd, dd)
        bal = new_bal
    n = tp + be + sl
    wr = (tp + be) / n * 100 if n else 0
    return {"bal": bal, "n": n, "tp": tp, "be": be, "sl": sl,
            "wr": wr, "dd": max_dd, "ret": (bal/START-1)*100}

s1 = run_sim(trades)                                                    # all
s2 = run_sim([t for t in trades if not t["geo_blocked"]])               # current live
s3 = run_sim([t for t in trades if not t["geo_blocked"] and not t["new_blocked"]])  # current + new

print("=" * 72)
print("3-WAY COMPARISON  —  $35 start, 85% x 40x")
print("=" * 72)
print(f"\n  {'':28} {'1.UNFILT':>10} {'2.CURRENT':>10} {'3.CURRENT+NEW':>14}")
print("  " + "-" * 64)
for label, k, fmt in [
    ("Trades", "n", "d"), ("TP", "tp", "d"), ("BE", "be", "d"), ("SL", "sl", "d"),
    ("Win rate", "wr", ".0f%"), ("Max drawdown", "dd", ".1f%"),
    ("Final balance", "bal", "$.2f"), ("Return", "ret", ".0f%"),
]:
    v1, v2, v3 = s1[k], s2[k], s3[k]
    if "%" in fmt:
        r1 = f"{v1:{fmt[:-1]}}" + "%"
        r2 = f"{v2:{fmt[:-1]}}" + "%"
        r3 = f"{v3:{fmt[:-1]}}" + "%"
    elif fmt == "d":
        r1, r2, r3 = str(int(v1)), str(int(v2)), str(int(v3))
    else:
        r1, r2, r3 = f"${v1:.2f}", f"${v2:.2f}", f"${v3:.2f}"
    print(f"  {label:28} {r1:>10} {r2:>10} {r3:>14}")

# SL detail per scenario
print(f"\n--- WHICH SLs REMAIN IN EACH SCENARIO ---\n")
for label, fn in [
    ("1. UNFILT", lambda t: True),
    ("2. CURRENT", lambda t: not t["geo_blocked"]),
    ("3. CURRENT+NEW", lambda t: not t["geo_blocked"] and not t["new_blocked"]),
]:
    sl_ts = [t for t in trades if t["is_sl"] and fn(t)]
    print(f"  {label}: {len(sl_ts)} SL(s)")
    for t in sl_ts:
        cnt_s  = str(t["cnt"]) if t["cnt"] is not None else "?"
        sh_s   = f"{t['share']:.0f}%" if t["share"] else "?"
        print(f"    {t['dt']} {t['sym']}  cnt={cnt_s}  share={sh_s}  net={t['net']:+.1f}")

# New filter: what gets additionally blocked
new_add = [t for t in trades if not t["geo_blocked"] and t["new_blocked"]]
print(f"\n--- ADDITIONALLY BLOCKED BY NEW ETH FILTER (N={len(new_add)}) ---\n")
print(f"  {'Date':>5}  {'Sym':>3}  {'cnt':>4}  {'Share':>6}  {'Exit':>2}  {'Net':>7}")
print("  " + "-" * 40)
for t in new_add:
    flag = "SL" if "SL" in t["exit_r"] else ("BE" if "BE" in t["exit_r"] else "TP")
    cnt_s = str(t["cnt"]) if t["cnt"] is not None else "?"
    sh_s  = f"{t['share']:.0f}%" if t["share"] else "?"
    print(f"  {t['dt']:>5}  {t['sym']:>3}  {cnt_s:>4}  {sh_s:>6}  {flag:>2}  {t['net']:>+7.1f}")
add_sl  = sum(1 for t in new_add if t["is_sl"])
add_win = sum(1 for t in new_add if not t["is_sl"])
add_net = sum(t["net"] for t in new_add)
print(f"\n  Blocked additionally: {len(new_add)} trades  (SL={add_sl}, win/BE={add_win})")
print(f"  Net bps of blocked group: {add_net:+.1f}")
