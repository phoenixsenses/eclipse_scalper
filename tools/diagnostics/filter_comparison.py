# encoding: utf-8
"""
Filter comparison: unfiltered vs cnt>7 AND share<80% filter.
$35 start, 85% margin x 40x leverage.
"""
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
rows = intel.execute(
    "SELECT trade_id, rule_name, entry_ts_ms, net_bps, exit_reason, trade_json "
    "FROM s34_trades WHERE status='CLOSED' AND net_bps IS NOT NULL "
    "AND rule_name IN (?,?) ORDER BY entry_ts_ms",
    RULES
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

        # Filter decision
        weak_count = (cnt is not None and cnt <= 7)
        weak_share = (share is not None and share >= 80.0)
        filtered_out = weak_count or weak_share

        trades.append({
            "tid": tid, "sym": sym, "dt": dt,
            "net": float(net_bps), "exit_r": exit_r or "",
            "cnt": cnt, "share": share,
            "filtered_out": filtered_out,
            "reason": ("cnt<=7" if weak_count else "") + ("+" if (weak_count and weak_share) else "") + ("share>=80%" if weak_share else ""),
        })
    except Exception:
        pass

MARGIN_PCT = 0.85
LEVERAGE   = 40
START      = 35.0

def run_sim(label, trade_list):
    bal  = START
    peak = START
    max_dd = 0.0
    tp = be = sl = 0
    log = []
    for t in trade_list:
        margin   = bal * MARGIN_PCT
        notional = margin * LEVERAGE
        pnl      = notional * t["net"] / 10000
        new_bal  = max(bal + pnl, 0.01)
        flag = "SL" if "SL" in t["exit_r"] else ("BE" if "BE" in t["exit_r"] else "TP")
        if flag == "SL": sl += 1
        elif flag == "BE": be += 1
        else: tp += 1
        dd = (peak - new_bal) / peak * 100 if peak > 0 else 0
        if new_bal > peak: peak = new_bal
        max_dd = max(max_dd, dd)
        log.append((t["dt"], t["sym"], flag, t["net"], bal, pnl, new_bal))
        bal = new_bal
    n = len(log)
    wr = (tp + be) / n * 100 if n else 0
    return bal, n, tp, be, sl, wr, max_dd, log

all_trades      = trades
filtered_trades = [t for t in trades if not t["filtered_out"]]
blocked_trades  = [t for t in trades if t["filtered_out"]]

bal_a, n_a, tp_a, be_a, sl_a, wr_a, dd_a, log_a = run_sim("UNFILTERED", all_trades)
bal_b, n_b, tp_b, be_b, sl_b, wr_b, dd_b, log_b = run_sim("FILTERED",   filtered_trades)

print("=" * 70)
print("FILTER COMPARISON  —  $35 start, 85% x 40x")
print("=" * 70)

# ── Side-by-side summary ──────────────────────────────────────────────────────
print(f"\n  {'':30} {'UNFILTERED':>12} {'FILTERED':>12} {'DIFF':>10}")
print("  " + "-" * 66)
print(f"  {'Trades':30} {n_a:>12} {n_b:>12} {n_b-n_a:>+10}")
print(f"  {'TP':30} {tp_a:>12} {tp_b:>12} {tp_b-tp_a:>+10}")
print(f"  {'BE':30} {be_a:>12} {be_b:>12} {be_b-be_a:>+10}")
print(f"  {'SL':30} {sl_a:>12} {sl_b:>12} {sl_b-sl_a:>+10}")
print(f"  {'Win rate':30} {wr_a:>11.0f}% {wr_b:>11.0f}% {wr_b-wr_a:>+9.0f}%")
print(f"  {'Max drawdown':30} {dd_a:>11.1f}% {dd_b:>11.1f}% {dd_b-dd_a:>+9.1f}%")
print(f"  {'Final balance':30} ${bal_a:>11.2f} ${bal_b:>11.2f} ${bal_b-bal_a:>+9.2f}")
print(f"  {'Total return':30} {(bal_a/START-1)*100:>11.0f}% {(bal_b/START-1)*100:>11.0f}% {(bal_b-bal_a)/START*100:>+9.0f}%")

# ── Blocked trades detail ─────────────────────────────────────────────────────
print(f"\n--- BLOCKED TRADES (N={len(blocked_trades)}) ---\n")
print(f"  {'Date':>5}  {'Sym':>3}  {'cnt':>4}  {'Share':>6}  {'Exit':>2}  {'Net':>7}  Reason")
print("  " + "-" * 52)
blocked_tp = blocked_be = blocked_sl = 0
for t in blocked_trades:
    flag = "SL" if "SL" in t["exit_r"] else ("BE" if "BE" in t["exit_r"] else "TP")
    if flag == "SL": blocked_sl += 1
    elif flag == "BE": blocked_be += 1
    else: blocked_tp += 1
    cnt_s  = str(t["cnt"]) if t["cnt"] is not None else "?"
    sh_s   = f"{t['share']:.0f}%" if t["share"] else "?"
    print(f"  {t['dt']:>5}  {t['sym']:>3}  {cnt_s:>4}  {sh_s:>6}  {flag:>2}  {t['net']:>+7.1f}  {t['reason']}")
print(f"\n  Blocked: TP={blocked_tp}  BE={blocked_be}  SL={blocked_sl}")
blocked_net = sum(t["net"] for t in blocked_trades)
print(f"  Net bps foregone: {blocked_net:+.1f}  (positive = would have won, negative = would have lost)")

# ── Chronological trade log (filtered) ───────────────────────────────────────
print(f"\n--- FILTERED SIM: trade-by-trade ---\n")
print(f"  {'Date':>5}  {'Sym':>3}  {'Exit':>2}  {'bps':>6}  {'Before':>9}  {'PnL':>8}  {'After':>9}")
print("  " + "-" * 58)
for dt, sym, flag, bps, before, pnl, after in log_b:
    marker = "  <-- SL" if flag == "SL" else ""
    print(f"  {dt:>5}  {sym:>3}  {flag:>2}  {bps:>+6.1f}  ${before:>8.2f}  ${pnl:>+7.2f}  ${after:>8.2f}{marker}")
print("  " + "-" * 58)
print(f"  Start: $35.00  ->  Final: ${bal_b:.2f}  (+{(bal_b/START-1)*100:.0f}%)")
