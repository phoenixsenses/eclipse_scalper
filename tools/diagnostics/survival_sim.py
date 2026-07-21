# encoding: utf-8
"""
Survival simulation: $35 start, 85% margin x40 leverage.
Uses ACTUAL paper trades for ETH_500K + SOL_200K (clean geo only).
Chronological order. Shows: would we have survived? SL hits? Balance today?
"""
import sqlite3, json
from pathlib import Path
from datetime import datetime, timezone

ROOT     = Path("D:/eclipse_scalper")
INTEL_DB = ROOT / "data" / "s34_intelligence.db"

MARGIN_PCT = 0.85
LEVERAGE   = 40
START_BAL  = 35.0
MIN_BAL    = 5.0   # below this = effectively bust

ETH_RULE = "ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30"
SOL_RULE = "SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30"

conn = sqlite3.connect("file:" + str(INTEL_DB) + "?mode=ro", uri=True)

# All closed trades for both rules
rows = conn.execute(
    "SELECT trade_id, rule_name, entry_ts_ms, exit_ts_ms, net_bps, exit_reason, trade_json "
    "FROM s34_trades WHERE status='CLOSED' AND net_bps IS NOT NULL "
    "AND rule_name IN (?,?) ORDER BY entry_ts_ms",
    (ETH_RULE, SOL_RULE)
).fetchall()

# Weak geo tag IDs (to exclude from SOL)
weak_ids = {r[0] for r in conn.execute(
    "SELECT trade_id FROM s34_shadow_geometry_tags WHERE tag='SOL_WEAK_GEOMETRY_SHADOW'"
).fetchall()}
conn.close()

print("=" * 65)
print("SURVIVAL SIMULATION — $35 start, 85% x 40x leverage")
print("=" * 65)

# ── Sim 1: ETH_500K + SOL_200K ALL (current setup before geo filter) ──────
def run_sim(label, trades):
    bal = START_BAL
    peak = START_BAL
    max_dd = 0.0
    sl_count = 0
    be_count = 0
    tp_count = 0
    consec_sl = 0
    max_consec_sl = 0
    bust = False
    log = []

    for tid, rule, entry_ms, exit_ms, net_bps, exit_r, tj in trades:
        if bal < MIN_BAL:
            bust = True
            break
        margin   = bal * MARGIN_PCT
        notional = margin * LEVERAGE
        pnl      = notional * float(net_bps) / 10000
        new_bal  = bal + pnl
        new_bal  = max(new_bal, 0.01)

        short_rule = "ETH" if "ETH" in rule else "SOL"
        dt = datetime.fromtimestamp(entry_ms/1000, tz=timezone.utc).strftime("%m/%d")

        if exit_r and "SL" in exit_r:
            sl_count += 1
            consec_sl += 1
            max_consec_sl = max(max_consec_sl, consec_sl)
            flag = "SL"
        elif exit_r and "BE" in exit_r:
            be_count += 1
            consec_sl = 0
            flag = "BE"
        else:
            tp_count += 1
            consec_sl = 0
            flag = "TP"

        dd = (peak - new_bal) / peak * 100 if peak > 0 else 0
        if new_bal > peak:
            peak = new_bal
        max_dd = max(max_dd, dd)

        log.append((dt, short_rule, flag, float(net_bps), bal, pnl, new_bal))
        bal = new_bal

    n = len(log)
    print(f"\n--- {label} (N={n} trades) ---")
    print(f"{'Date':>5}  {'Rule':>3}  {'Exit':>2}  {'bps':>6}  {'Before':>9}  {'PnL':>8}  {'After':>9}")
    print("-" * 58)
    for dt, rule, flag, bps, before, pnl, after in log:
        marker = " <-- SL" if flag == "SL" else ""
        print(f"{dt:>5}  {rule:>3}  {flag:>2}  {bps:>+6.1f}  ${before:>8.2f}  ${pnl:>+7.2f}  ${after:>8.2f}{marker}")

    print("-" * 58)
    print(f"Start: ${START_BAL:.2f}  ->  Final: ${bal:.2f}  ({'BUST' if bust else 'ALIVE'})")
    print(f"TP={tp_count}  BE={be_count}  SL={sl_count}  |  Max consec SL={max_consec_sl}  |  Max drawdown={max_dd:.1f}%")
    if n > 0:
        wr = (tp_count + be_count) / n * 100
        print(f"WR={wr:.0f}%  |  Total gain: ${bal-START_BAL:+.2f}  ({(bal/START_BAL-1)*100:+.0f}%)")
    return bal

# Sim A: ALL SOL trades (no geo filter — old behavior)
all_trades = [(r[0],r[1],r[2],r[3],r[4],r[5],r[6]) for r in rows]
bal_a = run_sim("SIM A: ETH_500K + SOL_200K ALL (no geo filter)", all_trades)

# Sim B: Clean geo only for SOL
clean_trades = [(r[0],r[1],r[2],r[3],r[4],r[5],r[6])
                for r in rows if not (r[1]==SOL_RULE and r[0] in weak_ids)]
bal_b = run_sim("SIM B: ETH_500K + SOL_200K CLEAN geo only", clean_trades)

print("\n" + "=" * 65)
print(f"SIM A (all SOL):    $35 -> ${bal_a:.2f}  ({(bal_a/35-1)*100:+.0f}%)")
print(f"SIM B (clean only): $35 -> ${bal_b:.2f}  ({(bal_b/35-1)*100:+.0f}%)")
print("=" * 65)
