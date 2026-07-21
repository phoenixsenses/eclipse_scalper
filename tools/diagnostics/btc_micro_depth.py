"""BTC micro-trend magnitude vs ETH outcome depth analysis"""
import sqlite3
from pathlib import Path

ROOT     = Path("D:/eclipse_scalper")
MICRO_DB = ROOT / "data" / "microstructure.db"
INTEL_DB = ROOT / "data" / "s34_intelligence.db"
ETH_RULE = "ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30"

def bps(p1, p2):
    if not p1 or not p2 or p1 == 0:
        return None
    return (p2 - p1) / p1 * 10000

def stats(nets):
    if not nets:
        return None
    s = sorted(nets)
    sl = sum(1 for x in nets if x < -20)
    return {"n": len(nets), "wr": sum(1 for x in nets if x > 0)/len(nets),
            "med": s[len(s)//2], "mean": sum(nets)/len(nets),
            "sl": sl, "sl_pct": sl/len(nets)*100}

intel = sqlite3.connect(f"file:{INTEL_DB}?mode=ro", uri=True)
micro = sqlite3.connect(f"file:{MICRO_DB}?mode=ro", uri=True)

trades = intel.execute("""
    SELECT trade_id, entry_ts_ms, net_bps FROM s34_trades
    WHERE rule_name=? AND status='CLOSED' AND net_bps IS NOT NULL
    ORDER BY entry_ts_ms
""", (ETH_RULE,)).fetchall()
intel.close()

buckets = {"0-2": [], "2-5": [], "5-10": [], "10+": [], "neg": []}
detail = []

for tid, entry_ms, net in trades:
    w_start = entry_ms - 10000
    r_before = micro.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (w_start,)).fetchone()
    r_entry = micro.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (entry_ms,)).fetchone()
    btc = bps(r_before[0] if r_before else None, r_entry[0] if r_entry else None)
    net_f = float(net)
    if btc is None:
        continue
    detail.append((btc, net_f, tid))
    if btc < 0:
        buckets["neg"].append(net_f)
    elif btc < 2:
        buckets["0-2"].append(net_f)
    elif btc < 5:
        buckets["2-5"].append(net_f)
    elif btc < 10:
        buckets["5-10"].append(net_f)
    else:
        buckets["10+"].append(net_f)

micro.close()

print("BTC 10s return magnitude vs ETH outcome:")
print(f"  {'BTC range':>10}  {'N':>4}  {'WR':>5}  {'med':>7}  {'mean':>7}  {'SL%':>5}")
for label, nets in [("neg (<0)", buckets["neg"]), ("0-2 bps", buckets["0-2"]),
                     ("2-5 bps", buckets["2-5"]), ("5-10 bps", buckets["5-10"]),
                     ("10+ bps", buckets["10+"])]:
    s = stats(nets)
    if not s:
        print(f"  {label:>10}  N=0")
        continue
    print(f"  {label:>10}  {s['n']:>4}  {s['wr']*100:.0f}%  {s['med']:>+7.1f}  {s['mean']:>+7.1f}  {s['sl_pct']:>4.0f}%")

print()
print("All trades (btc_ret vs eth_net):")
for btc, net, tid in sorted(detail, key=lambda x: x[0]):
    flag = "SL" if net < -20 else ("BE" if net < 5 else "TP")
    print(f"  btc={btc:>+7.2f} bps  eth={net:>+7.1f}  {flag}  {tid}")
