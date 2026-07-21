import sqlite3
from datetime import datetime, timezone

conn = sqlite3.connect("data/s34_intelligence.db")

rows = conn.execute(
    "SELECT rule_name, entry_ts_ms, net_bps FROM s34_trades "
    "WHERE status='CLOSED' AND net_bps IS NOT NULL "
    "AND rule_name IN ("
    "'ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30',"
    "'SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30'"
    ") ORDER BY entry_ts_ms"
).fetchall()

# SOL clean geo tag IDs
tag_ids = {r[0] for r in conn.execute(
    "SELECT trade_id FROM s34_shadow_geometry_tags WHERE tag='SOL_WEAK_GEOMETRY_SHADOW'"
).fetchall()}
sol_trade_ids = {r[0] for r in conn.execute(
    "SELECT trade_id FROM s34_trades WHERE status='CLOSED' "
    "AND rule_name='SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30'"
).fetchall()}

conn.close()

first_ms = rows[0][1]
last_ms  = rows[-1][1]
days = (last_ms - first_ms) / 86_400_000
first_dt = datetime.fromtimestamp(first_ms/1000, tz=timezone.utc)
last_dt  = datetime.fromtimestamp(last_ms/1000,  tz=timezone.utc)

print(f"Live paper range: {first_dt.date()} to {last_dt.date()} ({days:.1f} days)")
print()

for rule, short in [
    ("ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30", "ETH_500K"),
    ("SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30",           "SOL_200K_ALL"),
]:
    r = [x for x in rows if x[0] == rule]
    nets = [float(x[2]) for x in r]
    freq = len(r) / days
    mean = sum(nets) / len(nets)
    wr   = sum(1 for n in nets if n > 0) / len(nets)
    print(f"{short}: N={len(r)}  freq={freq:.3f}/day ({freq*30:.1f}/month)  mean={mean:+.1f}bps  WR={wr*100:.0f}%")

# SOL clean only
sol_rows = [x for x in rows if x[0] == "SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30"]
# need trade IDs - re-fetch
conn2 = sqlite3.connect("data/s34_intelligence.db")
sol_full = conn2.execute(
    "SELECT trade_id, entry_ts_ms, net_bps FROM s34_trades "
    "WHERE status='CLOSED' AND net_bps IS NOT NULL "
    "AND rule_name='SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30' ORDER BY entry_ts_ms"
).fetchall()
conn2.close()

clean_rows = [(tid, ts, nb) for tid, ts, nb in sol_full if tid not in tag_ids]
if clean_rows:
    nets_c = [float(r[2]) for r in clean_rows]
    freq_c = len(clean_rows) / days
    mean_c = sum(nets_c) / len(nets_c)
    wr_c   = sum(1 for n in nets_c if n > 0) / len(nets_c)
    print(f"SOL_200K_CLEAN: N={len(clean_rows)}  freq={freq_c:.3f}/day ({freq_c*30:.1f}/month)  mean={mean_c:+.1f}bps  WR={wr_c*100:.0f}%")
