import sqlite3, json

conn = sqlite3.connect('file:data/s34_intelligence.db?mode=ro', uri=True)
rows = conn.execute("""
    SELECT rule_name, net_bps, exit_reason, trade_json
    FROM s34_trades WHERE status='CLOSED' AND net_bps IS NOT NULL
    AND rule_name IN (
        'ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30',
        'SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30'
    )
""").fetchall()
conn.close()

data = []
for rule, net, exit_r, tj in rows:
    sig = json.loads(tj).get('signal') or {}
    total = sig.get('liq_total_notional')
    cnt = sig.get('liq_count')
    sym = 'ETH' if 'ETH' in rule else 'SOL'
    if total:
        data.append({'sym': sym, 'total_k': total/1000, 'cnt': cnt, 'net': float(net),
                     'sl': float(net) < -20})

def st(arr, label):
    if not arr:
        return
    nets = [x['net'] for x in arr]
    sl = sum(x['sl'] for x in arr)
    s = sorted(nets)
    print(f'  {label:40} N={len(nets):>3} WR={sum(1 for x in nets if x>0)/len(nets)*100:.0f}% SL={sl}({sl/len(nets)*100:.0f}%) med={s[len(s)//2]:+.1f}')

# SOL: finer threshold sweep for liq_total
print('SOL 200K — cascade SIZE upper bound sweep:')
thresholds = [300, 400, 500, 600, 700, 800, 1000, 9999]
sol = [x for x in data if x['sym'] == 'SOL']
for thr in thresholds:
    below = [x for x in sol if x['total_k'] < thr]
    above = [x for x in sol if x['total_k'] >= thr]
    if not below:
        continue
    print(f'  total < {thr:>5}K:', end='')
    nets = [x['net'] for x in below]
    sl = sum(x['sl'] for x in below)
    s = sorted(nets)
    blocked_sl = sum(x['sl'] for x in above)
    blocked_win = sum(1 for x in above if x['net'] > 0)
    print(f' N={len(nets):>3} WR={sum(1 for x in nets if x>0)/len(nets)*100:.0f}% SL={sl}({sl/len(nets)*100:.0f}%) med={s[len(s)//2]:+.1f} | blocked: {len(above)} ({blocked_win}W/{blocked_sl}SL)')

print()
print('ETH 500K — cascade SIZE upper bound sweep:')
eth = [x for x in data if x['sym'] == 'ETH']
thresholds_eth = [1000, 1500, 2000, 2500, 3000, 9999]
for thr in thresholds_eth:
    below = [x for x in eth if x['total_k'] < thr]
    above = [x for x in eth if x['total_k'] >= thr]
    if not below:
        continue
    nets = [x['net'] for x in below]
    sl = sum(x['sl'] for x in below)
    s = sorted(nets)
    blocked_sl = sum(x['sl'] for x in above)
    blocked_win = sum(1 for x in above if x['net'] > 0)
    print(f'  total < {thr:>5}K: N={len(nets):>3} WR={sum(1 for x in nets if x>0)/len(nets)*100:.0f}% SL={sl}({sl/len(nets)*100:.0f}%) med={s[len(s)//2]:+.1f} | blocked: {len(above)} ({blocked_win}W/{blocked_sl}SL)')

print()
print('SOL SL detail:')
for x in sorted(sol, key=lambda x: x['total_k']):
    if x['sl']:
        print(f'  {x["total_k"]:>8.0f}K  cnt={x["cnt"]}  net={x["net"]:+.1f}  SL')

print()
print('ETH SL detail:')
for x in sorted(eth, key=lambda x: x['total_k']):
    if x['sl']:
        print(f'  {x["total_k"]:>8.0f}K  cnt={x["cnt"]}  net={x["net"]:+.1f}  SL')
