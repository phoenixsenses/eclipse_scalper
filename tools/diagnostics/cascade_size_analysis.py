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

for sym in ['ETH', 'SOL']:
    d = sorted([x for x in data if x['sym'] == sym], key=lambda x: x['total_k'])
    if not d:
        continue
    mid = len(d) // 2

    def st(arr, label):
        nets = [x['net'] for x in arr]
        sl = sum(x['sl'] for x in arr)
        s = sorted(nets)
        print(f'  {label}: N={len(nets)} WR={sum(1 for x in nets if x>0)/len(nets)*100:.0f}% SL={sl}({sl/len(nets)*100:.0f}%) med={s[len(s)//2]:+.1f} range=${arr[0]["total_k"]:.0f}K-${arr[-1]["total_k"]:.0f}K')

    print(f'{sym} cascade size split (median={d[mid]["total_k"]:.0f}K):')
    st(d[:mid], 'SMALL cascade')
    st(d[mid:], 'LARGE cascade')
    print()

print('ETH 500K by cascade size:')
eth_data = [(x['total_k'], x['cnt'], x['net'], x['sl']) for x in data if x['sym'] == 'ETH']
for total_k, cnt, net, sl in sorted(eth_data, key=lambda x: x[0]):
    flag = 'SL' if sl else ('BE' if net < 5 else 'TP')
    print(f'  {total_k:>8.0f}K  cnt={str(cnt):>3}  {net:>+7.1f}  {flag}')
