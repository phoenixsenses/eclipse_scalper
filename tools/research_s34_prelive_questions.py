"""
S34 Pre-Live Research: 8 questions before building state machine executor.
A: SILENCE entry lag (T+0 vs T+30min price)
B: NEITHER confirmation time distribution
C: Overlap / same-position conflicts (LONG-on-LONG, SHORT-on-SHORT)
D: 1-position-at-a-time rule impact on WR and N
E: Day-of-week filter (Mon-Fri WR breakdown)
F: NEITHER SHORT session breakdown
G: BTC threshold sensitivity (300K / 500K / 750K / 1M)
H: Holdout-only compound simulation
"""
import bisect, json, math, sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path("D:/eclipse_scalper")
NAV  = ROOT / "reports/research/s34/S34_NAVIGATION_EVENTS.jsonl"
DB   = ROOT / "data/microstructure.db"
PRICES_DB = ROOT / "data/microstructure.db"
OUT_DIR = ROOT / "reports/research/s34"

LIVE_THRESH = 200_000.0
SIL_LO_MS  = 60_000
SIL_HI_MS  = 30 * 60_000
PROP_THRESH = 50_000.0
BTC_THRESH  = 500_000.0
FEE_BPS     = 5.0
SYNC_WIN_MS = 10 * 60_000
LEV         = 40.0
START_EQ    = 35.0
CAL_FRAC    = 0.70

# ── helpers ───────────────────────────────────────────────────────────────────
def wcnt(ts, v, lo, hi, thr):
    a = bisect.bisect_left(ts, lo); b = bisect.bisect_right(ts, hi)
    return sum(1 for i in range(a, b) if v[i] >= thr)

def wsum(ts, v, lo, hi):
    a = bisect.bisect_left(ts, lo); b = bisect.bisect_right(ts, hi)
    return sum(v[i] for i in range(a, b))

def first_above(ts, v, lo, hi, thr):
    a = bisect.bisect_left(ts, lo); b = bisect.bisect_right(ts, hi)
    for i in range(a, b):
        if v[i] >= thr: return int(ts[i])
    return None

def load_liq(conn, sym, side):
    r = conn.execute(
        "SELECT ts_ms,notional FROM liquidations WHERE symbol=? AND side=? ORDER BY ts_ms",
        (sym, side)).fetchall()
    return [int(x[0]) for x in r], [float(x[1]) for x in r]

def mark_at(conn, sym, ts_ms, window_ms=5000):
    """Mark price nearest to ts_ms within ±window_ms."""
    r = conn.execute(
        "SELECT price FROM mark_prices WHERE symbol=? AND ts_ms BETWEEN ? AND ? ORDER BY ABS(ts_ms - ?) LIMIT 1",
        (sym, ts_ms - window_ms, ts_ms + window_ms, ts_ms)).fetchone()
    return float(r[0]) if r else None

def win_rate(vals):
    if not vals: return float("nan")
    return sum(1 for v in vals if v > 0) / len(vals)

def stats_str(vals):
    if not vals: return "N=0"
    return f"N={len(vals)} WR={win_rate(vals):.1%} mean={mean(vals):+.1f} median={median(vals):+.1f}bps"

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading events...")
events = []
with NAV.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            try: events.append(json.loads(line))
            except: pass

valid_all = [
    r for r in events
    if float(r.get("threshold_usd") or 0) >= LIVE_THRESH
    and r.get("net_2h_bps") and math.isfinite(float(r["net_2h_bps"]))
]
valid_all.sort(key=lambda r: int(r["signal_ts_ms"]))
n_all = len(valid_all)
cal_n = int(n_all * CAL_FRAC)
HOLD_CUTOFF_MS = int(valid_all[cal_n]["signal_ts_ms"])

print(f"  Total >=200K: {n_all}  Cal: {cal_n}  Hold: {n_all-cal_n}")

print("Loading liquidation arrays...")
with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
    eth_ts, eth_not = load_liq(conn, "ETHUSDT", "SELL")
    btc_ts, btc_not = load_liq(conn, "BTCUSDT", "SELL")
    sol_ts, sol_not = load_liq(conn, "SOLUSDT", "SELL")

    # Check if mark_prices table exists
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    has_mark = "mark_prices" in tables
    print(f"  mark_prices table: {'YES' % () if has_mark else 'NO'}")

results = {}

# ── classify helper ──────────────────────────────────────────────────────────
def classify_row(row):
    ts   = int(row["signal_ts_ms"])
    thr  = float(row.get("threshold_usd") or 0)
    net2 = float(row.get("net_2h_bps") or "nan")
    net4v= row.get("net_4h_bps")
    net4 = float(net4v) if net4v is not None else net2
    tags = row.get("tags") or []
    if not math.isfinite(net2) or thr < LIVE_THRESH: return None
    if "BULL_PULLBACK" in tags: return None

    n_prop  = wcnt(eth_ts, eth_not, ts + SIL_LO_MS, ts + SIL_HI_MS, PROP_THRESH)
    sil_eth = n_prop == 0
    btc_1st = first_above(btc_ts, btc_not, ts + SIL_LO_MS, ts + SIL_HI_MS, BTC_THRESH)
    sil_btc = btc_1st is None
    b4h     = float(row.get("btc4h_bps") or 0)
    vd      = float(row.get("vdepth_bps") or 0)
    ts_dt   = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    hour    = ts_dt.hour
    sess_us = 13 <= hour < 21
    sync_k  = wsum(btc_ts, btc_not, ts - SYNC_WIN_MS, ts) + wsum(sol_ts, sol_not, ts - SYNC_WIN_MS, ts)
    n2h     = wcnt(eth_ts, eth_not, ts - 2*3600_000, ts - 1000, PROP_THRESH)
    score   = sum([int(sil_eth), int(n2h >= 3), int(b4h < 0), int(vd >= 30), int(sess_us), int(sync_k >= 200_000)])
    session = ("ASIA" if 0 <= hour < 7 else "EUROPE" if 7 <= hour < 13
               else "US" if 13 <= hour < 21 else "OFF")

    return {
        "ts": ts, "ts_dt": ts_dt, "net2": net2, "net4": net4,
        "sil_eth": sil_eth, "sil_btc": sil_btc, "btc_1st": btc_1st,
        "score": score, "session": session, "b4h": b4h, "vd": vd,
        "sess_us": sess_us, "sync_k": sync_k, "n2h": n2h,
        "is_holdout": ts >= HOLD_CUTOFF_MS,
        "hour": hour, "dow": ts_dt.weekday(),  # 0=Mon 6=Sun
    }

print("Classifying events...")
classified = [c for row in valid_all if (c := classify_row(row)) is not None]
print(f"  Classified: {len(classified)}")

# ─────────────────────────────────────────────────────────────────────────────
# Q-A: SILENCE entry lag
# Mark prices not always available; use btc4h drift as proxy for T+30 price shift
# Instead: look at net_4h vs hypothetical net if we enter 30min late
# We approximate: if the avg ETH price move in first 30min after cascade
# can be inferred from net_4h_bps (T=0 to T+4h) vs a proxy.
# More directly: report the distribution of ETH 30min-to-4h drift
# using btc price data if available.
print("\n=== A: SILENCE entry lag ===")
sil_rows = [c for c in classified if c["sil_eth"] and c["session"] != "EUROPE"]
# The backtest uses net_4h measured from cascade time (T=0 to T+4h).
# Live entry at T+30min means we hold T+30min to T+4h = 3.5h hold.
# We don't have per-minute marks but we can estimate:
# net_4h = price_move(T, T+4h); if entry is T+30min, we miss the first 30min drift.
# We'll look at net_2h vs net_4h as a rough proxy for early vs late gain capture.
# Actually best we can do: flag this as a key implementation question.

# We CAN compute: among SILENCE LONG wins, do wins tend to move fast (in first 30min)?
# Use: if btc_4h_bps > 0 (BTC falling = positive for ETH LONG) at signal time,
# does the first 30min matter?

# Proxy: net_4h - net_2h = "gain in hours 2-4"
# If most gain is in hours 2-4, T+30min entry still captures it.
a_vals_4h = [c["net4"] - FEE_BPS for c in sil_rows]
a_vals_2h = [c["net2"] - FEE_BPS for c in sil_rows]  # 2h as rough T+30min proxy
# net_2h from cascade time T: 30min late entry misses first ~25% of the 2h hold
# Rough approximation: 30min late = net_4h measured from T+30min ≈ (net_4h * 3.5/4.0)
a_vals_lag = [(c["net4"] * 3.5/4.0) - FEE_BPS for c in sil_rows]

a_stats = {
    "n": len(sil_rows),
    "t0_entry_4h": stats_str(a_vals_4h),
    "t30min_proxy_3h30m": stats_str(a_vals_lag),
    "wr_t0": f"{win_rate(a_vals_4h):.1%}",
    "wr_t30min_proxy": f"{win_rate(a_vals_lag):.1%}",
    "mean_diff_bps": f"{mean(a_vals_lag) - mean(a_vals_4h):+.1f}",
    "note": ("T+30min entry approximated as 3.5/4 of net_4h. "
             "Real test needs per-minute mark prices. "
             "CRITICAL: live must enter at T+30min (silence confirmed), not T+0."),
}
print(f"  T=0 entry:      {a_stats['t0_entry_4h']}")
print(f"  T+30 proxy:     {a_stats['t30min_proxy_3h30m']}")
print(f"  Mean diff:      {a_stats['mean_diff_bps']} bps")
results["A_silence_entry_lag"] = a_stats

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== B: NEITHER confirmation time ===")
# For NEITHER: we need BOTH eth_follow_on AND btc_cascade in [T+1min, T+30min].
# ETH follow-on time = first ETH >=50K after T+1min
# BTC cascade time = first BTC >=500K after T+1min
# Confirmation time = max(eth_fo_time, btc_time) - T
# Entry = BTC cascade time (not confirmation time)

neither_rows = [c for c in classified if not c["sil_eth"] and not c["sil_btc"] and c["score"] >= 3]

eth_fo_delays = []
btc_delays    = []
confirm_delays= []
entry_delays  = []  # BTC cascade time - T

for c in neither_rows:
    ts = c["ts"]
    eth_fo = first_above(eth_ts, eth_not, ts + SIL_LO_MS, ts + SIL_HI_MS, PROP_THRESH)
    btc_1  = c["btc_1st"]
    if eth_fo is not None and btc_1 is not None:
        eth_delay = (eth_fo - ts) / 60_000
        btc_delay = (btc_1 - ts) / 60_000
        confirm   = max(eth_fo, btc_1)
        conf_delay= (confirm - ts) / 60_000
        eth_fo_delays.append(eth_delay)
        btc_delays.append(btc_delay)
        confirm_delays.append(conf_delay)
        entry_delays.append(btc_delay)  # entry = first BTC

def pct(lst, p):
    s = sorted(lst)
    idx = int(len(s)*p/100)
    return s[min(idx, len(s)-1)]

b_stats = {
    "n_neither_score3": len(neither_rows),
    "n_with_timing": len(eth_fo_delays),
    "eth_fo_min_delay_p25_p50_p75_p90": (
        f"{pct(eth_fo_delays,25):.1f} / {pct(eth_fo_delays,50):.1f} / "
        f"{pct(eth_fo_delays,75):.1f} / {pct(eth_fo_delays,90):.1f} min"
    ) if eth_fo_delays else "N/A",
    "btc_cascade_delay_p25_p50_p75_p90": (
        f"{pct(btc_delays,25):.1f} / {pct(btc_delays,50):.1f} / "
        f"{pct(btc_delays,75):.1f} / {pct(btc_delays,90):.1f} min"
    ) if btc_delays else "N/A",
    "confirmation_delay_p25_p50_p75_p90": (
        f"{pct(confirm_delays,25):.1f} / {pct(confirm_delays,50):.1f} / "
        f"{pct(confirm_delays,75):.1f} / {pct(confirm_delays,90):.1f} min"
    ) if confirm_delays else "N/A",
    "pct_entry_within_5min": f"{sum(1 for d in btc_delays if d<=5)/len(btc_delays):.1%}" if btc_delays else "N/A",
    "pct_entry_within_15min": f"{sum(1 for d in btc_delays if d<=15)/len(btc_delays):.1%}" if btc_delays else "N/A",
    "pct_entry_within_5min_label": "BTC entry <=5min",
    "pct_entry_within_15min_label": "BTC entry <=15min",
    "note": "Entry = first BTC cascade time. Confirmation = max(eth_fo, btc) — when we KNOW it's NEITHER.",
}
print(f"  ETH follow-on delay: {b_stats['eth_fo_min_delay_p25_p50_p75_p90']}")
print(f"  BTC cascade delay:   {b_stats['btc_cascade_delay_p25_p50_p75_p90']}")
print(f"  Confirmation delay:  {b_stats['confirmation_delay_p25_p50_p75_p90']}")
print(f"  BTC entry <=5min:    {b_stats['pct_entry_within_5min']}")
print(f"  BTC entry <=15min:   {b_stats['pct_entry_within_15min']}")
results["B_neither_confirmation_time"] = b_stats

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== C: Overlap / same-signal conflicts ===")
# Build full signal stream (all SILENCE LONG ex-Europe + NEITHER SHORT score>=3)
# then count how many fire while a position is still open.
signal_stream = []
for c in classified:
    if c["sil_eth"] and c["session"] != "EUROPE":
        signal_stream.append({"ts": c["ts"], "type": "LONG", "hold_ms": 4*3600_000,
                               "net_bps": c["net4"] - FEE_BPS, "c": c})
    elif not c["sil_eth"] and not c["sil_btc"] and c["score"] >= 3:
        entry_ts = c["btc_1st"] if c["btc_1st"] else c["ts"] + SIL_LO_MS
        signal_stream.append({"ts": entry_ts, "type": "SHORT", "hold_ms": 2*3600_000,
                               "net_bps": -c["net2"] - FEE_BPS, "c": c})
signal_stream.sort(key=lambda x: x["ts"])

long_on_long  = []   # LONG fired while another LONG open
short_on_short= []   # SHORT fired while another SHORT open
flip_events   = []   # SHORT fired while LONG open
ignored_ll    = []   # would be ignored (LONG-on-LONG)
ignored_ss    = []   # would be ignored (SHORT-on-SHORT)

cur_end = None
cur_type = None
taken = []
blocked = []

for sig in signal_stream:
    if cur_end is None or sig["ts"] >= cur_end:
        # No active position — take it
        cur_type = sig["type"]
        cur_end  = sig["ts"] + sig["hold_ms"]
        taken.append(sig)
    else:
        # Position active
        if sig["type"] == "LONG" and cur_type == "LONG":
            long_on_long.append(sig)
            blocked.append(sig)
        elif sig["type"] == "SHORT" and cur_type == "SHORT":
            short_on_short.append(sig)
            blocked.append(sig)
        elif sig["type"] == "SHORT" and cur_type == "LONG":
            # Flip: close LONG early, open SHORT
            flip_events.append(sig)
            cur_type = "SHORT"
            cur_end  = sig["ts"] + sig["hold_ms"]
            taken.append(sig)  # counts as taken (flip)
        # SHORT-on-LONG is a flip (handled above); LONG-on-SHORT → block
        elif sig["type"] == "LONG" and cur_type == "SHORT":
            blocked.append(sig)

c_stats = {
    "total_raw_signals": len(signal_stream),
    "long_signals": sum(1 for s in signal_stream if s["type"]=="LONG"),
    "short_signals": sum(1 for s in signal_stream if s["type"]=="SHORT"),
    "taken_after_1pos_rule": len(taken),
    "blocked_signals": len(blocked),
    "long_on_long_conflicts": len(long_on_long),
    "short_on_short_conflicts": len(short_on_short),
    "flip_events": len(flip_events),
    "pct_signals_taken": f"{len(taken)/len(signal_stream):.1%}" if signal_stream else "N/A",
}
print(f"  Raw signals:    LONG={c_stats['long_signals']} SHORT={c_stats['short_signals']}")
print(f"  After 1-pos rule: taken={c_stats['taken_after_1pos_rule']} blocked={c_stats['blocked_signals']}")
print(f"  LONG-on-LONG:   {c_stats['long_on_long_conflicts']}")
print(f"  SHORT-on-SHORT: {c_stats['short_on_short_conflicts']}")
print(f"  Flips (L->S):   {c_stats['flip_events']}")
results["C_overlap_conflicts"] = c_stats

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== D: 1-position-at-a-time impact ===")
# Compare WR and N with vs without the blocking rule
taken_bps = [s["net_bps"] for s in taken]
all_bps_long  = [s["net_bps"] for s in signal_stream if s["type"]=="LONG"]
all_bps_short = [s["net_bps"] for s in signal_stream if s["type"]=="SHORT"]
taken_long  = [s["net_bps"] for s in taken if s["type"]=="LONG"]
taken_short = [s["net_bps"] for s in taken if s["type"]=="SHORT"]

d_stats = {
    "no_rule_long":  stats_str(all_bps_long),
    "no_rule_short": stats_str(all_bps_short),
    "no_rule_all":   stats_str(all_bps_long + all_bps_short),
    "with_rule_long":  stats_str(taken_long),
    "with_rule_short": stats_str(taken_short),
    "with_rule_all":   stats_str(taken_bps),
    "note": "1-pos rule blocks LONG-on-LONG and SHORT-on-SHORT; allows flip (L->S).",
}
print(f"  Without rule — ALL:   {d_stats['no_rule_all']}")
print(f"  With rule    — ALL:   {d_stats['with_rule_all']}")
print(f"  Without rule — LONG:  {d_stats['no_rule_long']}")
print(f"  With rule    — LONG:  {d_stats['with_rule_long']}")
print(f"  Without rule — SHORT: {d_stats['no_rule_short']}")
print(f"  With rule    — SHORT: {d_stats['with_rule_short']}")
results["D_one_pos_rule_impact"] = d_stats

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== E: Day-of-week breakdown ===")
DOW = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
dow_long  = defaultdict(list)
dow_short = defaultdict(list)
for s in taken:
    ts_dt = s["c"]["ts_dt"]
    dow   = ts_dt.weekday()
    if s["type"] == "LONG":  dow_long[dow].append(s["net_bps"])
    if s["type"] == "SHORT": dow_short[dow].append(s["net_bps"])

e_stats = {}
print(f"  {'Day':<4} {'LONG':>30} {'SHORT':>30}")
for d in range(7):
    ll = dow_long.get(d,[])
    ss = dow_short.get(d,[])
    l_str = f"N={len(ll)} WR={win_rate(ll):.0%} m={mean(ll):+.0f}" if ll else "N=0"
    s_str = f"N={len(ss)} WR={win_rate(ss):.0%} m={mean(ss):+.0f}" if ss else "N=0"
    print(f"  {DOW[d]:<4} {l_str:>30} {s_str:>30}")
    e_stats[DOW[d]] = {"long": stats_str(ll), "short": stats_str(ss)}
results["E_day_of_week"] = e_stats

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== F: NEITHER SHORT session breakdown ===")
sess_short = defaultdict(list)
for c in classified:
    if not c["sil_eth"] and not c["sil_btc"] and c["score"] >= 3:
        sess_short[c["session"]].append(-c["net2"] - FEE_BPS)

f_stats = {}
print(f"  {'Session':<8} {'Stats':>50}")
for sess in ["ASIA","EUROPE","US","OFF"]:
    vals = sess_short.get(sess, [])
    s = stats_str(vals)
    print(f"  {sess:<8} {s}")
    f_stats[sess] = s
results["F_neither_short_session"] = f_stats

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== G: BTC threshold sensitivity ===")
thresholds = [200_000, 300_000, 500_000, 750_000, 1_000_000, 1_500_000]
g_stats = {}
print(f"  {'BTC_THRESH':>12}  {'N':>4}  {'WR':>6}  {'Mean':>8}  {'Hold WR':>8}  {'Hold N':>6}")
for thr in thresholds:
    vals_all  = []
    vals_hold = []
    for c in classified:
        if c["sil_eth"]: continue  # only NEITHER candidates
        n_prop = 0  # already know sil_eth=False => n_prop>0 for NEITHER candidates
        btc_1  = first_above(btc_ts, btc_not, c["ts"] + SIL_LO_MS, c["ts"] + SIL_HI_MS, thr)
        if btc_1 is None: continue  # no BTC cascade above this threshold
        # recompute score with this new threshold (only sil_btc changes)
        # sil_eth=False, sil_btc depends on threshold
        # score components: sil_eth(0) + n2h>=3 + b4h<0 + vd>=30 + sess_us + sync_k>=200K
        score_no_sil = sum([int(c["n2h"]>=3), int(c["b4h"]<0), int(c["vd"]>=30),
                            int(c["sess_us"]), int(c["sync_k"]>=200_000)])
        # sil_eth=0 so max score=5; we keep score>=3
        if score_no_sil < 3: continue
        net_bps = -c["net2"] - FEE_BPS
        vals_all.append(net_bps)
        if c["is_holdout"]: vals_hold.append(net_bps)
    wr   = win_rate(vals_all)
    wr_h = win_rate(vals_hold)
    m    = mean(vals_all) if vals_all else float("nan")
    label = f"{thr/1000:.0f}K"
    print(f"  {label:>12}  {len(vals_all):>4}  {wr:.1%}  {m:>+8.1f}  {wr_h:.1%}  {len(vals_hold):>6}")
    g_stats[label] = {"n": len(vals_all), "wr": f"{wr:.1%}", "mean": f"{m:+.1f}",
                      "hold_n": len(vals_hold), "hold_wr": f"{wr_h:.1%}"}
results["G_btc_threshold_sensitivity"] = g_stats

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== H: Holdout-only compound ===")
hold_long  = [s for s in taken if s["type"]=="LONG"  and s["c"]["is_holdout"]]
hold_short = [s for s in taken if s["type"]=="SHORT" and s["c"]["is_holdout"]]
hold_all   = sorted(hold_long + hold_short, key=lambda x: x["ts"])
hold_bps   = [s["net_bps"] for s in hold_all]

hl_bps = [s["net_bps"] for s in hold_long]
hs_bps = [s["net_bps"] for s in hold_short]

print(f"  Holdout: {len(hold_all)} trades | LONG={len(hold_long)} SHORT={len(hold_short)}")
if hold_long:  print(f"  LONG:  {stats_str(hl_bps)}")
if hold_short: print(f"  SHORT: {stats_str(hs_bps)}")
if hold_all:   print(f"  ALL:   {stats_str(hold_bps)}")

# Compound
eq = START_EQ; pk = eq; max_dd_usd=0.0; max_dd_pct=0.0
milestones = [50,75,100,150,200,500,1000,2000,5000]
mt_idx = 0
mt_hits = []
for i, s in enumerate(hold_all):
    pnl_pct = s["net_bps"] / 10_000.0 * LEV
    eq = max(0.001, eq + eq * pnl_pct)
    if eq > pk: pk = eq
    dd_usd = pk - eq; dd_pct = dd_usd / pk * 100
    if dd_usd > max_dd_usd: max_dd_usd = dd_usd
    if dd_pct > max_dd_pct: max_dd_pct = dd_pct
    while mt_idx < len(milestones) and eq >= milestones[mt_idx]:
        dt_str = hold_all[i]["c"]["ts_dt"].strftime("%Y-%m-%d")
        print(f"  ${milestones[mt_idx]:<6} at trade {i+1:>3} ({dt_str})")
        mt_hits.append({"milestone": milestones[mt_idx], "trade": i+1})
        mt_idx += 1

simple_pnl = sum(hold_bps) * (START_EQ * LEV / 10_000.0) if hold_bps else 0.0
h_stats = {
    "n_trades": len(hold_all),
    "long_n": len(hold_long), "short_n": len(hold_short),
    "long_stats": stats_str(hl_bps),
    "short_stats": stats_str(hs_bps),
    "all_stats": stats_str(hold_bps),
    "compound_end_equity": round(eq, 2),
    "compound_pct": f"{(eq/START_EQ-1)*100:.1f}%",
    "max_dd_usd": round(max_dd_usd, 2),
    "max_dd_pct": f"{max_dd_pct:.1f}%",
    "simple_pnl_usd": round(simple_pnl, 2),
    "simple_final": round(START_EQ + simple_pnl, 2),
    "milestones_hit": mt_hits,
}
print(f"\n  Compound end:  ${eq:,.2f} ({h_stats['compound_pct']})")
print(f"  Max DD:        ${max_dd_usd:,.2f} ({h_stats['max_dd_pct']})")
print(f"  Sabit margin:  ${START_EQ} -> ${START_EQ + simple_pnl:.2f} (+${simple_pnl:.2f})")
results["H_holdout_compound"] = h_stats

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Saving results ===")
out_path = OUT_DIR / "S34_PRELIVE_RESEARCH.json"
import json as _json
out_path.write_text(_json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"  Saved: {out_path}")

# Summary
print("\n" + "="*62)
print("SUMMARY")
print(f"A entry lag:    T+30min proxy WR={results['A_silence_entry_lag']['wr_t30min_proxy']} vs T=0 WR={results['A_silence_entry_lag']['wr_t0']}")
print(f"B confirm time: BTC entry median={b_stats['btc_cascade_delay_p25_p50_p75_p90']}")
print(f"C conflicts:    flips={c_stats['flip_events']} blocked={c_stats['blocked_signals']}")
print(f"D 1pos rule:    {d_stats['with_rule_all']}")
print(f"G BTC optimal:  see table above")
print(f"H holdout only: {h_stats['all_stats']}")
