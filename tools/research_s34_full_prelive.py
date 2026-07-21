"""
S34 Full Pre-Live Research — all 35 questions.
S1-S10: Signal quality | H1-H6: Hold & exit | E1-E4: Entry precision
P1-P5:  Portfolio/sizing | R1-R6: Regime | RS1-RS5: Risk/system
"""
import bisect, json, math, random, sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev

ROOT    = Path("D:/eclipse_scalper")
NAV     = ROOT / "reports/research/s34/S34_NAVIGATION_EVENTS.jsonl"
DB      = ROOT / "data/microstructure.db"
OUT     = ROOT / "reports/research/s34/S34_FULL_PRELIVE.json"

LIVE_THRESH  = 200_000.0
SIL_LO_MS   = 60_000
SIL_HI_MS   = 30 * 60_000
PROP_THRESH  = 50_000.0
BTC_THRESH   = 500_000.0
FEE_BPS      = 5.0
SYNC_WIN_MS  = 10 * 60_000
CASCADE_WIN  = 5 * 60_000
LEV          = 40.0
START_EQ     = 35.0
CAL_FRAC     = 0.70
PERM_N       = 1000

results = {}

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

def get_mark(conn, sym, ts_ms, window_ms=90_000):
    r = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms BETWEEN ? AND ?"
        " ORDER BY ABS(ts_ms - ?) LIMIT 1",
        (sym, ts_ms - window_ms, ts_ms + window_ms, ts_ms)).fetchone()
    return float(r[0]) if r else None

def wr(vals):
    return sum(1 for v in vals if v > 0) / len(vals) if vals else float("nan")

def si(vals):
    if not vals: return "N=0"
    w = wr(vals)
    m = mean(vals)
    md = median(vals)
    return f"N={len(vals)} WR={w:.1%} mean={m:+.1f} med={md:+.1f}bps"

def pct(lst, p):
    if not lst: return float("nan")
    s = sorted(lst); idx = max(0, min(int(len(s)*p/100), len(s)-1))
    return s[idx]

def compound_sim(bps_list):
    eq = START_EQ; pk = eq; max_dd_pct = 0.0
    for b in bps_list:
        eq = max(0.001, eq + eq * b / 10000.0 * LEV)
        if eq > pk: pk = eq
        dd = (pk - eq) / pk * 100
        if dd > max_dd_pct: max_dd_pct = dd
    simple = sum(bps_list) * START_EQ * LEV / 10000.0
    return {
        "end_eq": round(eq, 2),
        "simple_final": round(START_EQ + simple, 2),
        "simple_gain": round(simple, 2),
        "max_dd_pct": round(max_dd_pct, 1),
    }

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading NAV events...")
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
print(f"  {n_all} events | cal={cal_n} hold={n_all-cal_n} | cutoff={datetime.fromtimestamp(HOLD_CUTOFF_MS/1000,tz=timezone.utc).date()}")

print("Loading liq arrays...")
with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
    eth_ts, eth_not = load_liq(conn, "ETHUSDT", "SELL")
    btc_ts, btc_not = load_liq(conn, "BTCUSDT", "SELL")
    sol_ts, sol_not = load_liq(conn, "SOLUSDT", "SELL")
    # Check mark_prices columns
    mp_cols = [r[1] for r in conn.execute("PRAGMA table_info(mark_prices)").fetchall()]
print(f"  ETH={len(eth_ts):,} BTC={len(btc_ts):,} SOL={len(sol_ts):,}")
print(f"  mark_prices cols: {mp_cols}")

# ── classify ─────────────────────────────────────────────────────────────────
print("Classifying...")

def classify(row, btc_thresh=BTC_THRESH, cascade_win=CASCADE_WIN):
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
    btc_1st = first_above(btc_ts, btc_not, ts + SIL_LO_MS, ts + SIL_HI_MS, btc_thresh)
    sil_btc = btc_1st is None
    b4h     = float(row.get("btc4h_bps") or 0)
    vd      = float(row.get("vdepth_bps") or 0)
    ts_dt   = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    hour    = ts_dt.hour
    dow     = ts_dt.weekday()
    sess_us = 13 <= hour < 21
    session = ("ASIA" if 0<=hour<7 else "EUROPE" if 7<=hour<13
               else "US" if 13<=hour<21 else "OFF")
    sync_k  = wsum(btc_ts, btc_not, ts-SYNC_WIN_MS, ts) + wsum(sol_ts, sol_not, ts-SYNC_WIN_MS, ts)
    n2h     = wcnt(eth_ts, eth_not, ts-2*3600_000, ts-1000, PROP_THRESH)
    score   = sum([int(sil_eth), int(n2h>=3), int(b4h<0), int(vd>=30), int(sess_us), int(sync_k>=200_000)])

    # Cascade size bucket
    eth_5min= wcnt(eth_ts, eth_not, ts-cascade_win, ts, PROP_THRESH)  # prior cascades in window
    sol_lead= wcnt(sol_ts, sol_not, ts-10*60_000, ts, 50_000)         # SOL cascade in -10min

    return {
        "ts": ts, "ts_dt": ts_dt, "net2": net2, "net4": net4,
        "sil_eth": sil_eth, "sil_btc": sil_btc, "btc_1st": btc_1st,
        "score": score, "session": session, "hour": hour, "dow": dow,
        "b4h": b4h, "vd": vd, "sess_us": sess_us, "sync_k": sync_k,
        "n2h": n2h, "thr": thr, "n_prop": n_prop,
        "sol_lead": sol_lead,
        "is_hold": ts >= HOLD_CUTOFF_MS,
    }

classified = [c for row in valid_all if (c := classify(row)) is not None]
print(f"  Classified: {len(classified)}")

# Baseline signal builders
def build_baseline(cs, btc_thr=BTC_THRESH, score_thr=3, dow_excl_long=(), dow_excl_short=(),
                   sync_thr=200_000, n2h_thr=3):
    sigs = []
    for c in cs:
        sc = sum([int(c["sil_eth"]), int(c["n2h"]>=n2h_thr), int(c["b4h"]<0),
                  int(c["vd"]>=30), int(c["sess_us"]), int(c["sync_k"]>=sync_thr)])
        if c["sil_eth"] and c["session"] != "EUROPE":
            if c["dow"] in dow_excl_long: continue
            sigs.append({"ts": c["ts"], "type": "LONG",
                         "net_bps": c["net4"]-FEE_BPS, "c": c, "score": sc})
        elif not c["sil_eth"] and c["btc_1st"] is not None:
            btc_ok = first_above(btc_ts, btc_not, c["ts"]+SIL_LO_MS, c["ts"]+SIL_HI_MS, btc_thr)
            if btc_ok is None: continue
            if sc < score_thr: continue
            if c["dow"] in dow_excl_short: continue
            sigs.append({"ts": btc_ok, "type": "SHORT",
                         "net_bps": -c["net2"]-FEE_BPS, "c": c, "score": sc})
    return sorted(sigs, key=lambda x: x["ts"])

def apply_1pos_rule(stream):
    taken, blocked = [], []
    cur_end, cur_type = None, None
    for s in stream:
        if cur_end is None or s["ts"] >= cur_end:
            cur_type = s["type"]
            cur_end  = s["ts"] + (4*3600_000 if s["type"]=="LONG" else 2*3600_000)
            taken.append(s)
        else:
            if s["type"] == "SHORT" and cur_type == "LONG":
                cur_type = "SHORT"
                cur_end  = s["ts"] + 2*3600_000
                taken.append(s)
            else:
                blocked.append(s)
    return taken, blocked

baseline_stream = build_baseline(classified)
baseline_taken, baseline_blocked = apply_1pos_rule(baseline_stream)

print(f"  Baseline: raw={len(baseline_stream)} taken={len(baseline_taken)} blocked={len(baseline_blocked)}")
DOW_NAMES = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL QUALITY  S1-S10
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== S1: DOW filter LONG (excl Mon+Wed) ===")
s1_stream = build_baseline(classified, dow_excl_long=(0, 2))
s1_taken, _ = apply_1pos_rule(s1_stream)
s1_long = [s["net_bps"] for s in s1_taken if s["type"]=="LONG"]
s1_short= [s["net_bps"] for s in s1_taken if s["type"]=="SHORT"]
s1_all  = [s["net_bps"] for s in s1_taken]
bl_long = [s["net_bps"] for s in baseline_taken if s["type"]=="LONG"]
bl_short= [s["net_bps"] for s in baseline_taken if s["type"]=="SHORT"]
bl_all  = [s["net_bps"] for s in baseline_taken]
print(f"  Baseline LONG:        {si(bl_long)}")
print(f"  Excl Mon+Wed LONG:    {si(s1_long)}")
print(f"  Baseline ALL:         {si(bl_all)}")
print(f"  Excl Mon+Wed ALL:     {si(s1_all)}")
results["S1_dow_excl_monwed"] = {
    "baseline_long": si(bl_long), "filtered_long": si(s1_long),
    "baseline_all": si(bl_all), "filtered_all": si(s1_all),
    "hold_long_base": si([s["net_bps"] for s in baseline_taken if s["type"]=="LONG" and s["c"]["is_hold"]]),
    "hold_long_filt": si([s["net_bps"] for s in s1_taken if s["type"]=="LONG" and s["c"]["is_hold"]]),
}

print("\n=== S2: DOW filter SHORT (excl Sun) ===")
s2_stream = build_baseline(classified, dow_excl_short=(6,))
s2_taken, _ = apply_1pos_rule(s2_stream)
s2_short = [s["net_bps"] for s in s2_taken if s["type"]=="SHORT"]
s2_all   = [s["net_bps"] for s in s2_taken]
print(f"  Baseline SHORT:       {si(bl_short)}")
print(f"  Excl Sun SHORT:       {si(s2_short)}")
print(f"  Excl Sun ALL:         {si(s2_all)}")
results["S2_dow_excl_sun_short"] = {
    "baseline_short": si(bl_short), "filtered_short": si(s2_short),
    "baseline_all": si(bl_all), "filtered_all": si(s2_all),
    "hold_short_base": si([s["net_bps"] for s in baseline_taken if s["type"]=="SHORT" and s["c"]["is_hold"]]),
    "hold_short_filt": si([s["net_bps"] for s in s2_taken if s["type"]=="SHORT" and s["c"]["is_hold"]]),
}

print("\n=== S3: BTC 750K threshold — combined DOW + threshold ===")
s3_stream_750 = build_baseline(classified, btc_thr=750_000)
s3_taken_750, _ = apply_1pos_rule(s3_stream_750)
# Also: 750K + Mon/Wed LONG excl + Sun SHORT excl
s3_stream_best = build_baseline(classified, btc_thr=750_000, dow_excl_long=(0,2), dow_excl_short=(6,))
s3_taken_best, _ = apply_1pos_rule(s3_stream_best)
s3_hold_750 = [s["net_bps"] for s in s3_taken_750 if s["c"]["is_hold"]]
s3_hold_best= [s["net_bps"] for s in s3_taken_best if s["c"]["is_hold"]]
print(f"  750K only:            {si([s['net_bps'] for s in s3_taken_750])}")
print(f"  750K holdout:         {si(s3_hold_750)}")
print(f"  750K+DOW best:        {si([s['net_bps'] for s in s3_taken_best])}")
print(f"  750K+DOW holdout:     {si(s3_hold_best)}")
results["S3_btc750k_dow_combined"] = {
    "btc750_all": si([s["net_bps"] for s in s3_taken_750]),
    "btc750_hold": si(s3_hold_750),
    "btc750_dow_all": si([s["net_bps"] for s in s3_taken_best]),
    "btc750_dow_hold": si(s3_hold_best),
    "btc750_simple": compound_sim([s["net_bps"] for s in s3_taken_750]),
    "btc750_dow_simple": compound_sim([s["net_bps"] for s in s3_taken_best]),
}

print("\n=== S4: SILENCE LONG score breakdown ===")
sil_rows = [c for c in classified if c["sil_eth"] and c["session"] != "EUROPE"]
by_score = defaultdict(list)
for c in sil_rows:
    sc = sum([int(c["sil_eth"]),int(c["n2h"]>=3),int(c["b4h"]<0),int(c["vd"]>=30),int(c["sess_us"]),int(c["sync_k"]>=200_000)])
    by_score[sc].append(c["net4"]-FEE_BPS)
s4_stats = {}
for sc in sorted(by_score.keys()):
    vals = by_score[sc]
    print(f"  score={sc}: {si(vals)}")
    s4_stats[f"score_{sc}"] = si(vals)
# Also >=2 and >=3 filter
s4_ge2 = [v for sc,vs in by_score.items() if sc>=2 for v in vs]
s4_ge3 = [v for sc,vs in by_score.items() if sc>=3 for v in vs]
print(f"  score>=2: {si(s4_ge2)}")
print(f"  score>=3: {si(s4_ge3)}")
s4_stats["score_ge2"] = si(s4_ge2); s4_stats["score_ge3"] = si(s4_ge3)
results["S4_silence_score_breakdown"] = s4_stats

print("\n=== S5: n2h threshold (3 vs 4) for NEITHER SHORT ===")
for n2h_thr in [2, 3, 4, 5]:
    stream_tmp = build_baseline(classified, n2h_thr=n2h_thr)
    taken_tmp, _ = apply_1pos_rule(stream_tmp)
    short_tmp = [s["net_bps"] for s in taken_tmp if s["type"]=="SHORT"]
    hold_tmp  = [s["net_bps"] for s in taken_tmp if s["type"]=="SHORT" and s["c"]["is_hold"]]
    print(f"  n2h>={n2h_thr}: {si(short_tmp)} | hold={si(hold_tmp)}")
results["S5_n2h_threshold"] = "see console output above"

print("\n=== S6: sync_k threshold (200K vs 400K) ===")
for skt in [0, 100_000, 200_000, 400_000, 600_000]:
    stream_tmp = build_baseline(classified, sync_thr=skt)
    taken_tmp, _ = apply_1pos_rule(stream_tmp)
    all_tmp  = [s["net_bps"] for s in taken_tmp]
    hold_tmp = [s["net_bps"] for s in taken_tmp if s["c"]["is_hold"]]
    label = f"{skt//1000}K"
    print(f"  sync_k>={label}: {si(all_tmp)} | hold={si(hold_tmp)}")
results["S6_sync_threshold"] = "see console output above"

print("\n=== S7: SILENCE early entry T+15min (mark_prices) ===")
sil_taken_15 = []
with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
    for c in classified:
        if not c["sil_eth"] or c["session"] == "EUROPE": continue
        ts = c["ts"]
        # Check silence at T+15min: no ETH follow-on in T+1min to T+15min
        n_prop_15 = wcnt(eth_ts, eth_not, ts+SIL_LO_MS, ts+15*60_000, PROP_THRESH)
        if n_prop_15 > 0: continue  # noisy already by 15min — can't enter early
        # Get mark price at T+0 and T+15min
        p0   = get_mark(conn, "ETHUSDT", ts)
        p15  = get_mark(conn, "ETHUSDT", ts + 15*60_000)
        p4h  = get_mark(conn, "ETHUSDT", ts + 4*3600_000)
        if p0 and p15 and p4h:
            lag_bps_15 = (p15 - p0) / p0 * 10000       # cost of 15min lag
            net_from_15 = (p4h - p15) / p15 * 10000 - FEE_BPS   # P&L from T+15 to T+4h (3h45m)
            net_from_00 = (p4h - p0)  / p0  * 10000 - FEE_BPS   # P&L from T+0 (backtest)
            sil_taken_15.append({
                "lag_15": lag_bps_15,
                "net_t0": net_from_00,
                "net_t15": net_from_15,
                "is_hold": c["is_hold"],
            })

if sil_taken_15:
    t0_vals  = [r["net_t0"]  for r in sil_taken_15]
    t15_vals = [r["net_t15"] for r in sil_taken_15]
    lags     = [r["lag_15"]  for r in sil_taken_15]
    print(f"  Samples with mark prices: {len(sil_taken_15)}")
    print(f"  T+0  entry: {si(t0_vals)}")
    print(f"  T+15 entry: {si(t15_vals)}")
    print(f"  Avg 15min lag cost: {mean(lags):+.1f} bps")
    print(f"  Holdout T+0:  {si([r['net_t0']  for r in sil_taken_15 if r['is_hold']])}")
    print(f"  Holdout T+15: {si([r['net_t15'] for r in sil_taken_15 if r['is_hold']])}")
    results["S7_silence_t15_entry"] = {
        "n_with_marks": len(sil_taken_15),
        "t0_entry":  si(t0_vals), "t15_entry": si(t15_vals),
        "avg_lag_15_bps": round(mean(lags), 2),
        "hold_t0":  si([r["net_t0"]  for r in sil_taken_15 if r["is_hold"]]),
        "hold_t15": si([r["net_t15"] for r in sil_taken_15 if r["is_hold"]]),
    }

print("\n=== S8: ETH 5min drift filter ===")
# If ETH moved UP >15bps in first 5min after cascade, bounce has started — skip LONG
drift_data = []
with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
    for c in classified:
        if not c["sil_eth"] or c["session"] == "EUROPE": continue
        ts = c["ts"]
        p0 = get_mark(conn, "ETHUSDT", ts)
        p5 = get_mark(conn, "ETHUSDT", ts + 5*60_000)
        if p0 and p5:
            drift5 = (p5 - p0) / p0 * 10000
            drift_data.append({"drift5": drift5, "net4": c["net4"]-FEE_BPS, "is_hold": c["is_hold"]})

if drift_data:
    # Buckets by 5min drift
    for thr_lo, thr_hi, label in [(-999,-10,"bounce<-10"),(-10,10,"flat"),( 10, 999,"bounce>+10")]:
        bucket = [r["net4"] for r in drift_data if thr_lo <= r["drift5"] < thr_hi]
        print(f"  5min drift {label:>14}: {si(bucket)}")
    # Filter: skip if 5min drift > +15bps (price bounced strongly)
    long_no_filter = [r["net4"] for r in drift_data]
    long_filtered  = [r["net4"] for r in drift_data if r["drift5"] < 15]
    print(f"  All:              {si(long_no_filter)}")
    print(f"  Excl drift>+15:   {si(long_filtered)}")
    results["S8_eth5min_drift"] = {
        "n_with_marks": len(drift_data),
        "all": si(long_no_filter), "excl_bounce_gt15": si(long_filtered),
        "buckets": {
            "bounce_neg10": si([r["net4"] for r in drift_data if r["drift5"] < -10]),
            "flat":         si([r["net4"] for r in drift_data if -10 <= r["drift5"] < 10]),
            "bounce_pos10": si([r["net4"] for r in drift_data if r["drift5"] >= 10]),
        }
    }

print("\n=== S9: Blocked LONG-on-LONG signals (27) independent WR ===")
# Re-run 1-pos rule tracking blocked LONG-on-LONG specifically
s9_stream = build_baseline(classified)
cur_end2, cur_type2 = None, None
s9_blocked_ll = []
s9_blocked_ss = []
for s in s9_stream:
    if cur_end2 is None or s["ts"] >= cur_end2:
        cur_type2 = s["type"]
        cur_end2  = s["ts"] + (4*3600_000 if s["type"]=="LONG" else 2*3600_000)
    else:
        if s["type"] == "LONG" and cur_type2 == "LONG":
            s9_blocked_ll.append(s["net_bps"])
        elif s["type"] == "SHORT" and cur_type2 == "SHORT":
            s9_blocked_ss.append(s["net_bps"])
        elif s["type"] == "SHORT" and cur_type2 == "LONG":
            cur_type2 = "SHORT"; cur_end2 = s["ts"] + 2*3600_000
print(f"  Blocked LONG-on-LONG ({len(s9_blocked_ll)}): {si(s9_blocked_ll)}")
print(f"  Blocked SHORT-on-SHORT ({len(s9_blocked_ss)}): {si(s9_blocked_ss)}")
print(f"  Comparison taken LONG: {si(bl_long)}")
results["S9_blocked_signals_wr"] = {
    "blocked_long_on_long": si(s9_blocked_ll),
    "blocked_short_on_short": si(s9_blocked_ss),
    "taken_long_ref": si(bl_long), "taken_short_ref": si(bl_short),
}

print("\n=== S10: SHORT OFF session filter ===")
s10_off = [s["net_bps"] for s in baseline_taken if s["type"]=="SHORT" and s["c"]["session"]=="OFF"]
s10_no_off = [s["net_bps"] for s in baseline_taken if s["type"]=="SHORT" and s["c"]["session"]!="OFF"]
print(f"  OFF session SHORT ({len(s10_off)}): {si(s10_off)}")
print(f"  Non-OFF SHORT:        {si(s10_no_off)}")
results["S10_short_off_session"] = {
    "off_short": si(s10_off), "non_off_short": si(s10_no_off),
    "off_hold": si([s["net_bps"] for s in baseline_taken if s["type"]=="SHORT" and s["c"]["session"]=="OFF" and s["c"]["is_hold"]]),
}

# ═══════════════════════════════════════════════════════════════════════════════
# HOLD & EXIT  H1-H6
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== H1: LONG hold 3h vs 4h (mark_prices) ===")
h1_data = []
with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
    for s in baseline_taken:
        if s["type"] != "LONG": continue
        ts = s["c"]["ts"]
        p0 = get_mark(conn, "ETHUSDT", ts)
        p3 = get_mark(conn, "ETHUSDT", ts + 3*3600_000)
        if p0 and p3:
            net3 = (p3 - p0) / p0 * 10000 - FEE_BPS
            h1_data.append({"net3": net3, "net4": s["net_bps"], "is_hold": s["c"]["is_hold"]})
if h1_data:
    n3  = [r["net3"] for r in h1_data]
    n4  = [r["net4"] for r in h1_data]
    print(f"  3h hold: {si(n3)}")
    print(f"  4h hold: {si(n4)}")
    print(f"  3h holdout: {si([r['net3'] for r in h1_data if r['is_hold']])}")
    print(f"  4h holdout: {si([r['net4'] for r in h1_data if r['is_hold']])}")
    results["H1_long_hold_3h_vs_4h"] = {
        "hold3h": si(n3), "hold4h": si(n4),
        "hold3h_holdout": si([r["net3"] for r in h1_data if r["is_hold"]]),
        "hold4h_holdout": si([r["net4"] for r in h1_data if r["is_hold"]]),
    }

print("\n=== H2: SHORT hold 1.5h vs 2h (mark_prices) ===")
h2_data = []
with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
    for s in baseline_taken:
        if s["type"] != "SHORT": continue
        ts = s["ts"]  # entry = BTC cascade time
        p0  = get_mark(conn, "ETHUSDT", ts)
        p15 = get_mark(conn, "ETHUSDT", ts + int(1.5*3600_000))
        if p0 and p15:
            net15 = -(p15 - p0) / p0 * 10000 - FEE_BPS  # SHORT: negative price move = profit
            h2_data.append({"net15": net15, "net2": s["net_bps"], "is_hold": s["c"]["is_hold"]})
if h2_data:
    n15 = [r["net15"] for r in h2_data]
    n2  = [r["net2"]  for r in h2_data]
    print(f"  1.5h hold: {si(n15)}")
    print(f"  2h hold:   {si(n2)}")
    print(f"  1.5h holdout: {si([r['net15'] for r in h2_data if r['is_hold']])}")
    print(f"  2h holdout:   {si([r['net2']  for r in h2_data if r['is_hold']])}")
    results["H2_short_hold_1h5_vs_2h"] = {
        "hold1h5": si(n15), "hold2h": si(n2),
        "hold1h5_holdout": si([r["net15"] for r in h2_data if r["is_hold"]]),
        "hold2h_holdout":  si([r["net2"]  for r in h2_data if r["is_hold"]]),
    }

print("\n=== H3: BE stop (approx: if net2h>50, floor net4h at -FEE) ===")
# Approximation: if trade was up >50bps at 2h mark (proxy), BE stop activated.
# If net4 then went negative, we exit at ~-FEE instead.
# Note: this uses mark prices at 2h as proxy.
h3_data = []
with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
    for s in baseline_taken:
        if s["type"] != "LONG": continue
        ts = s["c"]["ts"]
        p0 = get_mark(conn, "ETHUSDT", ts)
        p2 = get_mark(conn, "ETHUSDT", ts + 2*3600_000)
        if p0 and p2:
            net2 = (p2 - p0) / p0 * 10000
            net4 = s["net_bps"]
            # BE stop: if mid-hold > +50bps, worst case = -FEE
            net4_be = max(-FEE_BPS, net4) if net2 > 50 else net4
            h3_data.append({"net4": net4, "net4_be": net4_be, "net2_proxy": net2,
                             "be_triggered": net2 > 50, "is_hold": s["c"]["is_hold"]})
if h3_data:
    n4_base = [r["net4"] for r in h3_data]
    n4_be   = [r["net4_be"] for r in h3_data]
    be_count= sum(1 for r in h3_data if r["be_triggered"])
    print(f"  Baseline LONG:      {si(n4_base)}")
    print(f"  With BE stop:       {si(n4_be)}")
    print(f"  BE triggered:       {be_count}/{len(h3_data)} ({be_count/len(h3_data):.0%})")
    print(f"  BE holdout base:    {si([r['net4'] for r in h3_data if r['is_hold']])}")
    print(f"  BE holdout be:      {si([r['net4_be'] for r in h3_data if r['is_hold']])}")
    results["H3_breakeven_stop"] = {
        "baseline": si(n4_base), "with_be": si(n4_be),
        "be_triggered_pct": f"{be_count/len(h3_data):.0%}",
        "hold_base": si([r["net4"] for r in h3_data if r["is_hold"]]),
        "hold_be":   si([r["net4_be"] for r in h3_data if r["is_hold"]]),
        "note": "Approx: BE stop floor = -FEE when 2h mark >+50bps",
    }

print("\n=== H4: Trailing stop 100bps (approx using net2h as peak proxy) ===")
# Approx: if net4 < net2 - 100bps, trailing would have triggered at roughly net2-100
h4_data = []
for s in baseline_taken:
    if s["type"] != "LONG": continue
    net4 = s["net_bps"]
    net2 = s["c"]["net2"] - FEE_BPS  # 2h P&L as mid-hold proxy
    # Trailing: stop = highest_so_far - 100bps
    # If net4 < net2 - 100: stopped at net2 - 100
    net4_trail = max(net4, net2 - 100) if net2 > 0 else net4
    h4_data.append({"net4": net4, "net4_trail": net4_trail, "is_hold": s["c"]["is_hold"]})
if h4_data:
    base = [r["net4"] for r in h4_data]
    trail= [r["net4_trail"] for r in h4_data]
    print(f"  Baseline LONG:   {si(base)}")
    print(f"  Trail 100bps:    {si(trail)}")
    print(f"  Hold baseline:   {si([r['net4'] for r in h4_data if r['is_hold']])}")
    print(f"  Hold trail:      {si([r['net4_trail'] for r in h4_data if r['is_hold']])}")
    results["H4_trailing_100bps"] = {
        "baseline": si(base), "trailing": si(trail),
        "hold_base": si([r["net4"] for r in h4_data if r["is_hold"]]),
        "hold_trail": si([r["net4_trail"] for r in h4_data if r["is_hold"]]),
        "note": "Approx: if net4 < net2h-100bps, trail stops at net2h-100",
    }

print("\n=== H5: SHORT partial exit at +80bps (mark T+1h) ===")
h5_data = []
with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
    for s in baseline_taken:
        if s["type"] != "SHORT": continue
        ts = s["ts"]
        p0 = get_mark(conn, "ETHUSDT", ts)
        p1 = get_mark(conn, "ETHUSDT", ts + 3600_000)
        if p0 and p1:
            net1h = -(p1 - p0) / p0 * 10000
            # If >80bps at 1h: close 50% at 1h, let other 50% run to 2h
            net2 = s["net_bps"]  # full 2h P&L
            if net1h > 80:
                net_partial = (net1h + net2) / 2 - FEE_BPS
            else:
                net_partial = net2
            h5_data.append({"net2": net2, "net1h": net1h, "net_partial": net_partial,
                             "partial_triggered": net1h > 80, "is_hold": s["c"]["is_hold"]})
if h5_data:
    base  = [r["net2"] for r in h5_data]
    part  = [r["net_partial"] for r in h5_data]
    trig  = sum(1 for r in h5_data if r["partial_triggered"])
    print(f"  Baseline SHORT:   {si(base)}")
    print(f"  Partial exit:     {si(part)} (triggered {trig}/{len(h5_data)})")
    print(f"  Hold base:        {si([r['net2'] for r in h5_data if r['is_hold']])}")
    print(f"  Hold partial:     {si([r['net_partial'] for r in h5_data if r['is_hold']])}")
    results["H5_short_partial_exit"] = {
        "baseline": si(base), "partial_50pct_at_1h": si(part),
        "trigger_rate": f"{trig/len(h5_data):.0%}",
        "hold_base": si([r["net2"] for r in h5_data if r["is_hold"]]),
        "hold_partial": si([r["net_partial"] for r in h5_data if r["is_hold"]]),
    }

print("\n=== H6: Silence gate — 4h extension effect ===")
# Already in backtest via net_4h. Check: trades where silence gate would fire
# (no follow-on in full 4h window) vs those where it fires and we extend.
# Proxy: if net_2h < net_4h significantly, the extension helped.
h6_long = [s for s in baseline_taken if s["type"]=="LONG"]
helped = [s for s in h6_long if s["net_bps"] > s["c"]["net2"] - FEE_BPS]
print(f"  LONG trades where 4h > 2h: {len(helped)}/{len(h6_long)} ({len(helped)/len(h6_long):.0%})")
avg_gain = mean([s["net_bps"] - (s["c"]["net2"]-FEE_BPS) for s in h6_long]) if h6_long else 0
print(f"  Avg extra bps from 2h->4h hold: {avg_gain:+.1f}")
results["H6_silence_gate_hold_extension"] = {
    "pct_4h_better_than_2h": f"{len(helped)/len(h6_long):.0%}" if h6_long else "N/A",
    "avg_2h_to_4h_gain": round(avg_gain, 1),
    "2h_stats": si([s["c"]["net2"]-FEE_BPS for s in h6_long]),
    "4h_stats": si([s["net_bps"] for s in h6_long]),
}

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY PRECISION  E1-E4
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== E1: SILENCE T+30min real mark price entry ===")
e1_data = []
with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
    for c in classified:
        if not c["sil_eth"] or c["session"] == "EUROPE": continue
        ts = c["ts"]
        p0  = get_mark(conn, "ETHUSDT", ts)
        p30 = get_mark(conn, "ETHUSDT", ts + SIL_HI_MS)  # T+30min
        p4h = get_mark(conn, "ETHUSDT", ts + 4*3600_000)
        if p0 and p30 and p4h:
            lag30 = (p30 - p0) / p0 * 10000
            net_t0  = (p4h - p0)  / p0  * 10000 - FEE_BPS
            net_t30 = (p4h - p30) / p30 * 10000 - FEE_BPS
            e1_data.append({"lag30": lag30, "net_t0": net_t0, "net_t30": net_t30,
                             "is_hold": c["is_hold"]})
if e1_data:
    lags = [r["lag30"] for r in e1_data]
    t0v  = [r["net_t0"]  for r in e1_data]
    t30v = [r["net_t30"] for r in e1_data]
    print(f"  Samples: {len(e1_data)}")
    print(f"  30min lag bps: mean={mean(lags):+.1f} p25={pct(lags,25):+.1f} p50={pct(lags,50):+.1f} p75={pct(lags,75):+.1f}")
    print(f"  T+0  entry:   {si(t0v)}")
    print(f"  T+30 entry:   {si(t30v)}")
    print(f"  Hold T+0:     {si([r['net_t0']  for r in e1_data if r['is_hold']])}")
    print(f"  Hold T+30:    {si([r['net_t30'] for r in e1_data if r['is_hold']])}")
    results["E1_silence_t30_real_entry"] = {
        "n": len(e1_data),
        "lag30_mean": round(mean(lags),2), "lag30_p50": round(pct(lags,50),2),
        "t0_entry": si(t0v), "t30_entry": si(t30v),
        "hold_t0": si([r["net_t0"]  for r in e1_data if r["is_hold"]]),
        "hold_t30": si([r["net_t30"] for r in e1_data if r["is_hold"]]),
    }

print("\n=== E2: NEITHER entry price (BTC cascade time ETH price) ===")
e2_data = []
with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
    for s in baseline_taken:
        if s["type"] != "SHORT": continue
        c = s["c"]; btc_ts_entry = c["btc_1st"]
        if btc_ts_entry is None: continue
        ts0 = c["ts"]           # ETH cascade time
        p_eth_cascade = get_mark(conn, "ETHUSDT", ts0)
        p_eth_entry   = get_mark(conn, "ETHUSDT", btc_ts_entry)
        p_eth_exit    = get_mark(conn, "ETHUSDT", btc_ts_entry + 2*3600_000)
        if p_eth_cascade and p_eth_entry and p_eth_exit:
            slippage = (p_eth_entry - p_eth_cascade) / p_eth_cascade * 10000  # >0 means worse for SHORT
            net_from_entry = -(p_eth_exit - p_eth_entry) / p_eth_entry * 10000 - FEE_BPS
            e2_data.append({"slippage": slippage, "net": net_from_entry, "is_hold": c["is_hold"]})
if e2_data:
    slip = [r["slippage"] for r in e2_data]
    nets = [r["net"] for r in e2_data]
    print(f"  Samples: {len(e2_data)}")
    print(f"  ETH price shift (cascade -> BTC entry): mean={mean(slip):+.1f} p50={pct(slip,50):+.1f}bps")
    print(f"  Net from BTC entry: {si(nets)}")
    print(f"  Hold: {si([r['net'] for r in e2_data if r['is_hold']])}")
    results["E2_neither_btc_entry_slippage"] = {
        "n": len(e2_data),
        "eth_price_shift_mean_bps": round(mean(slip),2),
        "eth_price_shift_p50_bps": round(pct(slip,50),2),
        "net_from_btc_entry": si(nets),
        "hold": si([r["net"] for r in e2_data if r["is_hold"]]),
    }

print("\n=== E3: CASCADE_WIN 5min vs 10min (event count) ===")
for cwin_min in [3, 5, 10, 15]:
    cwin_ms = cwin_min * 60_000
    classified_cwin = [c for row in valid_all if (c := classify(row, cascade_win=cwin_ms)) is not None]
    stream_cwin = build_baseline(classified_cwin)
    taken_cwin, _ = apply_1pos_rule(stream_cwin)
    all_v  = [s["net_bps"] for s in taken_cwin]
    hold_v = [s["net_bps"] for s in taken_cwin if s["c"]["is_hold"]]
    print(f"  CASCADE_WIN={cwin_min}min: raw={len(stream_cwin)} taken={len(taken_cwin)} | {si(all_v)} | hold={si(hold_v)}")
results["E3_cascade_win_sensitivity"] = "see console output above"

print("\n=== E4: Flip timing (L->S) — price at flip moment ===")
e4_flips = []
with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
    cur_end3, cur_type3, cur_entry_ts, cur_entry_net4 = None, None, None, None
    for s in baseline_stream:
        if cur_end3 is None or s["ts"] >= cur_end3:
            cur_type3 = s["type"]
            cur_end3  = s["ts"] + (4*3600_000 if s["type"]=="LONG" else 2*3600_000)
            cur_entry_ts = s["ts"]; cur_entry_net4 = s["net_bps"]
        else:
            if s["type"] == "SHORT" and cur_type3 == "LONG":
                # Flip event
                long_elapsed_ms = s["ts"] - cur_entry_ts
                p_long_start = get_mark(conn, "ETHUSDT", cur_entry_ts)
                p_flip       = get_mark(conn, "ETHUSDT", s["ts"])
                if p_long_start and p_flip:
                    long_pnl_at_flip = (p_flip - p_long_start) / p_long_start * 10000
                    e4_flips.append({
                        "long_elapsed_min": long_elapsed_ms / 60_000,
                        "long_pnl_at_flip": long_pnl_at_flip,
                        "short_net": s["net_bps"],
                    })
                cur_type3 = "SHORT"; cur_end3 = s["ts"] + 2*3600_000
if e4_flips:
    print(f"  Flip events: {len(e4_flips)}")
    for f in e4_flips:
        print(f"    elapsed={f['long_elapsed_min']:.0f}min LONG_pnl={f['long_pnl_at_flip']:+.0f}bps SHORT_net={f['short_net']:+.0f}bps")
    results["E4_flip_timing"] = {
        "n_flips": len(e4_flips),
        "avg_long_elapsed_min": round(mean([f["long_elapsed_min"] for f in e4_flips]),1),
        "avg_long_pnl_at_flip": round(mean([f["long_pnl_at_flip"] for f in e4_flips]),1),
        "avg_short_net": round(mean([f["short_net"] for f in e4_flips]),1),
        "flips": e4_flips,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO / SIZING  P1-P5
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== P1: Score-based position sizing ===")
# score>=5: 2x, score>=4: 1.5x, else 1x
eq_score = START_EQ; pk_s = eq_score; max_dd_s = 0.0
eq_flat  = START_EQ; pk_f = eq_flat;  max_dd_f = 0.0
for s in baseline_taken:
    sc = s["score"]
    mult = 2.0 if sc >= 5 else 1.5 if sc >= 4 else 1.0
    # Scored compound
    pnl_s = s["net_bps"] / 10000.0 * LEV * mult
    eq_score = max(0.001, eq_score + eq_score * pnl_s)
    if eq_score > pk_s: pk_s = eq_score
    dd = (pk_s - eq_score) / pk_s * 100
    if dd > max_dd_s: max_dd_s = dd
    # Flat compound
    pnl_f = s["net_bps"] / 10000.0 * LEV
    eq_flat = max(0.001, eq_flat + eq_flat * pnl_f)
    if eq_flat > pk_f: pk_f = eq_flat
    dd = (pk_f - eq_flat) / pk_f * 100
    if dd > max_dd_f: max_dd_f = dd

simple_score = sum(s["net_bps"] * (2.0 if s["score"]>=5 else 1.5 if s["score"]>=4 else 1.0) for s in baseline_taken) * START_EQ * LEV / 10000.0
simple_flat  = sum(s["net_bps"] for s in baseline_taken) * START_EQ * LEV / 10000.0

print(f"  Flat sizing:  simple ${START_EQ}->${START_EQ+simple_flat:.0f}  max_dd={max_dd_f:.1f}%")
print(f"  Score-sized:  simple ${START_EQ}->${START_EQ+simple_score:.0f}  max_dd={max_dd_s:.1f}%")
results["P1_score_sizing"] = {
    "flat_simple_final": round(START_EQ + simple_flat, 2),
    "score_simple_final": round(START_EQ + simple_score, 2),
    "flat_max_dd_pct": round(max_dd_f, 1),
    "score_max_dd_pct": round(max_dd_s, 1),
}

print("\n=== P2: Daily max trade cap ===")
for cap in [2, 3, 5, 999]:
    daily_count = defaultdict(int)
    capped = []
    for s in baseline_stream:
        d = s["c"]["ts_dt"].date().isoformat()
        if daily_count[d] < cap:
            capped.append(s); daily_count[d] += 1
    taken_cap, _ = apply_1pos_rule(capped)
    all_v = [s["net_bps"] for s in taken_cap]
    label = f"cap={cap}" if cap < 999 else "no_cap"
    print(f"  {label}: N={len(taken_cap)} | {si(all_v)}")
results["P2_daily_cap"] = "see console output above"

print("\n=== P3: Fresh signal after 3+ trades same day ===")
# Count how many times we'd have a 4th+ trade on same day
date_trades = defaultdict(list)
for s in baseline_taken:
    date_trades[s["c"]["ts_dt"].date().isoformat()].append(s["net_bps"])
days_over3 = {d: vs for d, vs in date_trades.items() if len(vs) >= 3}
print(f"  Days with 3+ trades: {len(days_over3)}")
for d, vs in sorted(days_over3.items()):
    print(f"    {d}: N={len(vs)} sum={sum(vs):+.0f}bps WR={wr(vs):.0%}")
results["P3_same_day_3plus"] = {
    "days_with_3plus": len(days_over3),
    "detail": {d: {"n": len(vs), "wr": f"{wr(vs):.0%}", "sum": round(sum(vs),0)} for d,vs in days_over3.items()},
}

print("\n=== P4: Holdout compound — 750K + DOW filter best config ===")
s3_hold_bps = [s["net_bps"] for s in s3_taken_best if s["c"]["is_hold"]]
s3_hold_long = [s["net_bps"] for s in s3_taken_best if s["c"]["is_hold"] and s["type"]=="LONG"]
s3_hold_srt  = [s["net_bps"] for s in s3_taken_best if s["c"]["is_hold"] and s["type"]=="SHORT"]
print(f"  Config: BTC=750K + excl Mon/Wed LONG + excl Sun SHORT")
print(f"  Holdout ALL:   {si(s3_hold_bps)}")
print(f"  Holdout LONG:  {si(s3_hold_long)}")
print(f"  Holdout SHORT: {si(s3_hold_srt)}")
print(f"  Simple:        {compound_sim(s3_hold_bps)}")
results["P4_holdout_best_config"] = {
    "config": "BTC=750K + excl Mon/Wed LONG + excl Sun SHORT",
    "hold_all": si(s3_hold_bps), "hold_long": si(s3_hold_long), "hold_short": si(s3_hold_srt),
    "hold_simple": compound_sim(s3_hold_bps),
}

print("\n=== P5: Blocked signal detailed analysis ===")
cur_end5, cur_type5, cur_end_ts5 = None, None, None
p5_blocked = []
for s in baseline_stream:
    hold_ms = 4*3600_000 if s["type"]=="LONG" else 2*3600_000
    if cur_end5 is None or s["ts"] >= cur_end5:
        cur_type5 = s["type"]; cur_end5 = s["ts"] + hold_ms; cur_end_ts5 = s["ts"]
    else:
        conflict = f"{s['type']}_on_{cur_type5}"
        time_into_pos = (s["ts"] - cur_end_ts5) / 3600_000
        p5_blocked.append({
            "type": s["type"], "conflict": conflict, "net_bps": s["net_bps"],
            "time_into_pos_h": round(time_into_pos, 2), "is_hold": s["c"]["is_hold"],
        })
        if s["type"] == "SHORT" and cur_type5 == "LONG":
            cur_type5 = "SHORT"; cur_end5 = s["ts"] + 2*3600_000; cur_end_ts5 = s["ts"]

by_conflict = defaultdict(list)
for b in p5_blocked: by_conflict[b["conflict"]].append(b["net_bps"])
for conf, vs in by_conflict.items():
    print(f"  {conf}: {si(vs)}")
avg_time = mean([b["time_into_pos_h"] for b in p5_blocked]) if p5_blocked else 0
print(f"  Avg time-into-position when blocked: {avg_time:.1f}h")
results["P5_blocked_detail"] = {
    conf: si(vs) for conf, vs in by_conflict.items()
}
results["P5_blocked_detail"]["avg_time_into_pos_h"] = round(avg_time, 2)

# ═══════════════════════════════════════════════════════════════════════════════
# REGIME  R1-R6
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== R1: Cascade density regime ===")
for s in baseline_taken:
    ts = s["c"]["ts"]
    s["density"] = wcnt(eth_ts, eth_not, ts - 2*3600_000, ts - 1000, PROP_THRESH)

dense_long  = [s["net_bps"] for s in baseline_taken if s["type"]=="LONG"  and s["density"] >= 5]
normal_long = [s["net_bps"] for s in baseline_taken if s["type"]=="LONG"  and s["density"] <  5]
dense_short = [s["net_bps"] for s in baseline_taken if s["type"]=="SHORT" and s["density"] >= 5]
normal_short= [s["net_bps"] for s in baseline_taken if s["type"]=="SHORT" and s["density"] <  5]
print(f"  LONG  dense(>=5 in 2h): {si(dense_long)}")
print(f"  LONG  normal(<5):       {si(normal_long)}")
print(f"  SHORT dense:            {si(dense_short)}")
print(f"  SHORT normal:           {si(normal_short)}")
results["R1_cascade_density"] = {
    "long_dense_5plus": si(dense_long), "long_normal": si(normal_long),
    "short_dense_5plus": si(dense_short), "short_normal": si(normal_short),
}

print("\n=== R2: SKIP — no BTC dominance data ===")
results["R2_btc_dominance"] = "SKIPPED — no dominance data in DB"

print("\n=== R3: ETH/BTC sync_k breakdown ===")
sync_buckets = [(0, 100_000, "0-100K"), (100_000, 200_000, "100-200K"),
                (200_000, 500_000, "200-500K"), (500_000, 999_999_999, "500K+")]
r3 = {}
for lo, hi, label in sync_buckets:
    lv = [s["net_bps"] for s in baseline_taken if s["type"]=="LONG"  and lo <= s["c"]["sync_k"] < hi]
    sv = [s["net_bps"] for s in baseline_taken if s["type"]=="SHORT" and lo <= s["c"]["sync_k"] < hi]
    print(f"  sync_k {label}: LONG={si(lv)} | SHORT={si(sv)}")
    r3[label] = {"long": si(lv), "short": si(sv)}
results["R3_sync_breakdown"] = r3

print("\n=== R4: Consecutive SILENCE (prior event also SILENCE) ===")
prev_sil = False
r4_consec, r4_first = [], []
for c in classified:
    if not c["sil_eth"] or c["session"] == "EUROPE":
        prev_sil = c["sil_eth"]
        continue
    net = c["net4"] - FEE_BPS
    if prev_sil:
        r4_consec.append(net)
    else:
        r4_first.append(net)
    prev_sil = True
print(f"  First SILENCE (prev not sil): {si(r4_first)}")
print(f"  Consecutive SILENCE:          {si(r4_consec)}")
results["R4_consecutive_silence"] = {
    "first_silence": si(r4_first), "consecutive_silence": si(r4_consec)
}

print("\n=== R5: SOL lead signal ===")
# SOL cascade in -10min before ETH cascade = SOL led
r5_sol_lead    = [s["net_bps"] for s in baseline_taken if s["c"]["sol_lead"] > 0]
r5_no_sol_lead = [s["net_bps"] for s in baseline_taken if s["c"]["sol_lead"] == 0]
r5_lead_long   = [s["net_bps"] for s in baseline_taken if s["type"]=="LONG"  and s["c"]["sol_lead"] > 0]
r5_lead_short  = [s["net_bps"] for s in baseline_taken if s["type"]=="SHORT" and s["c"]["sol_lead"] > 0]
print(f"  SOL lead (>=1 cascade pre-10min): {si(r5_sol_lead)}")
print(f"  No SOL lead:                      {si(r5_no_sol_lead)}")
print(f"  SOL lead LONG:  {si(r5_lead_long)}")
print(f"  SOL lead SHORT: {si(r5_lead_short)}")
results["R5_sol_lead"] = {
    "sol_lead": si(r5_sol_lead), "no_sol_lead": si(r5_no_sol_lead),
    "sol_lead_long": si(r5_lead_long), "sol_lead_short": si(r5_lead_short),
}

print("\n=== R6: ETH cascade size filter ===")
for thr_lo, thr_hi, label in [(200_000, 300_000, "200-300K"), (300_000, 500_000, "300-500K"),
                                (500_000, 750_000, "500-750K"), (750_000, 999_999_999, "750K+")]:
    lv = [s["net_bps"] for s in baseline_taken if s["type"]=="LONG"  and thr_lo<=s["c"]["thr"]<thr_hi]
    sv = [s["net_bps"] for s in baseline_taken if s["type"]=="SHORT" and thr_lo<=s["c"]["thr"]<thr_hi]
    print(f"  ETH cascade {label}: LONG={si(lv)} | SHORT={si(sv)}")
results["R6_eth_cascade_size"] = {
    label: {"long": si([s["net_bps"] for s in baseline_taken if s["type"]=="LONG"  and tlo<=s["c"]["thr"]<thi]),
            "short":si([s["net_bps"] for s in baseline_taken if s["type"]=="SHORT" and tlo<=s["c"]["thr"]<thi])}
    for tlo,thi,label in [(200_000,300_000,"200-300K"),(300_000,500_000,"300-500K"),
                           (500_000,750_000,"500-750K"),(750_000,9e9,"750K+")]
}

# ═══════════════════════════════════════════════════════════════════════════════
# RISK / SYSTEM  RS1-RS5
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== RS1: Daily -200bps stop ===")
eq_base  = START_EQ; eq_dd  = START_EQ
day_pnl  = defaultdict(float)
for s in baseline_taken:
    day_pnl[s["c"]["ts_dt"].date().isoformat()] += s["net_bps"]

halted_days = set()
eq_dd_val = START_EQ
for d in sorted(day_pnl):
    if day_pnl[d] < -200:
        halted_days.add(d)

# Re-run taking only trades on non-halted days
taken_rs1 = [s for s in baseline_taken if s["c"]["ts_dt"].date().isoformat() not in halted_days]
print(f"  Days halted (daily loss > -200bps): {len(halted_days)} -> {sorted(halted_days)}")
print(f"  Trades taken without halt: {len(baseline_taken)}")
print(f"  Trades with -200 halt:     {len(taken_rs1)}")
print(f"  Without halt: {si(bl_all)}")
print(f"  With halt:    {si([s['net_bps'] for s in taken_rs1])}")
results["RS1_daily_drawdown_stop"] = {
    "halted_days": sorted(halted_days),
    "trades_no_halt": si(bl_all),
    "trades_with_halt": si([s["net_bps"] for s in taken_rs1]),
}

print("\n=== RS2: NEITHER within NEITHER (SHORT-on-SHORT) ===")
# When a second NEITHER fires while SHORT active: replace or ignore?
rs2_replace_stream = []
rs2_cur_end = None; rs2_cur_type = None
for s in baseline_stream:
    hold_ms = 4*3600_000 if s["type"]=="LONG" else 2*3600_000
    if rs2_cur_end is None or s["ts"] >= rs2_cur_end:
        rs2_cur_type = s["type"]; rs2_cur_end = s["ts"] + hold_ms
        rs2_replace_stream.append(s)
    elif s["type"] == "SHORT" and rs2_cur_type == "SHORT":
        # Replace: close current, open new
        rs2_cur_end = s["ts"] + 2*3600_000
        rs2_replace_stream.append(s)
    elif s["type"] == "SHORT" and rs2_cur_type == "LONG":
        rs2_cur_type = "SHORT"; rs2_cur_end = s["ts"] + 2*3600_000
        rs2_replace_stream.append(s)

rs2_bps = [s["net_bps"] for s in rs2_replace_stream]
bl_bps  = bl_all
print(f"  Ignore SHORT-on-SHORT (baseline): {si(bl_bps)}")
print(f"  Replace SHORT-on-SHORT:           {si(rs2_bps)}")
results["RS2_short_on_short_replace"] = {
    "ignore": si(bl_bps), "replace": si(rs2_bps),
    "n_replace": len(rs2_replace_stream), "n_baseline": len(baseline_taken),
}

print("\n=== RS3: Permutation test (combined portfolio, 1000 shuffles) ===")
real_bps = [s["net_bps"] for s in baseline_taken]
real_wr  = wr(real_bps)
real_mean= mean(real_bps) if real_bps else 0
rng = random.Random(42)
perm_wr_dist   = []
perm_mean_dist = []
for _ in range(PERM_N):
    shuffled = real_bps[:]
    rng.shuffle(shuffled)
    perm_wr_dist.append(wr(shuffled))
    perm_mean_dist.append(mean(shuffled))
p_wr   = sum(1 for v in perm_wr_dist   if v >= real_wr)   / PERM_N
p_mean = sum(1 for v in perm_mean_dist if v >= real_mean)  / PERM_N
print(f"  Real WR={real_wr:.1%} mean={real_mean:+.1f}bps (N={len(real_bps)})")
print(f"  Perm WR   p-value: {p_wr:.3f} (p<0.05 = significant)")
print(f"  Perm mean p-value: {p_mean:.3f}")
print(f"  Perm WR dist: p5={pct(perm_wr_dist,5):.1%} p50={pct(perm_wr_dist,50):.1%} p95={pct(perm_wr_dist,95):.1%}")
results["RS3_permutation_test"] = {
    "real_wr": f"{real_wr:.1%}", "real_mean": round(real_mean, 2),
    "perm_n": PERM_N,
    "p_value_wr":   round(p_wr,   4),
    "p_value_mean": round(p_mean, 4),
    "perm_wr_p5":  f"{pct(perm_wr_dist,5):.1%}",
    "perm_wr_p95": f"{pct(perm_wr_dist,95):.1%}",
    "significant_wr":   p_wr   < 0.05,
    "significant_mean": p_mean < 0.05,
}

print("\n=== RS4: Walk-forward (cal split into 3 blocks) ===")
cal_events = [c for c in classified if not c["is_hold"]]
cal_events.sort(key=lambda c: c["ts"])
block_size = len(cal_events) // 3
rs4_results = {}
for blk in range(3):
    lo = blk * block_size
    hi = (blk+1)*block_size if blk < 2 else len(cal_events)
    block_cs = cal_events[lo:hi]
    blk_stream = build_baseline(block_cs)
    blk_taken, _ = apply_1pos_rule(blk_stream)
    blk_bps = [s["net_bps"] for s in blk_taken]
    dt_lo = datetime.fromtimestamp(block_cs[0]["ts"]/1000,tz=timezone.utc).strftime("%Y-%m-%d")
    dt_hi = datetime.fromtimestamp(block_cs[-1]["ts"]/1000,tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"  Block {blk+1} ({dt_lo} to {dt_hi}): {si(blk_bps)}")
    rs4_results[f"block_{blk+1}"] = {"date_range": f"{dt_lo} to {dt_hi}", "stats": si(blk_bps)}
results["RS4_walkforward_3block"] = rs4_results

print("\n=== RS5: Washout filter (ETH back to cascade level in 10min) ===")
rs5_data = []
with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
    for s in baseline_taken:
        if s["type"] != "LONG": continue
        ts = s["c"]["ts"]
        p0   = get_mark(conn, "ETHUSDT", ts)
        p10m = get_mark(conn, "ETHUSDT", ts + 10*60_000)
        if p0 and p10m:
            move10 = (p10m - p0) / p0 * 10000
            # Washout: price recovered >80% of initial cascade drop (or went flat/up by 10min)
            washout = move10 > 10  # ETH bounced back up >10bps — cascade was temporary
            rs5_data.append({"move10": move10, "washout": washout, "net4": s["net_bps"],
                              "is_hold": s["c"]["is_hold"]})

if rs5_data:
    washout_n = sum(1 for r in rs5_data if r["washout"])
    keep_bps  = [r["net4"] for r in rs5_data if not r["washout"]]
    wash_bps  = [r["net4"] for r in rs5_data if r["washout"]]
    all_bps   = [r["net4"] for r in rs5_data]
    print(f"  Samples: {len(rs5_data)}  Washouts (10min bounce>+10bps): {washout_n} ({washout_n/len(rs5_data):.0%})")
    print(f"  All:             {si(all_bps)}")
    print(f"  Washout trades:  {si(wash_bps)}")
    print(f"  Non-washout:     {si(keep_bps)}")
    print(f"  Hold non-wash:   {si([r['net4'] for r in rs5_data if not r['washout'] and r['is_hold']])}")
    results["RS5_washout_filter"] = {
        "n": len(rs5_data), "washout_rate": f"{washout_n/len(rs5_data):.0%}",
        "all": si(all_bps), "washout_trades": si(wash_bps), "non_washout": si(keep_bps),
        "hold_non_washout": si([r["net4"] for r in rs5_data if not r["washout"] and r["is_hold"]]),
    }

# ── Save ──────────────────────────────────────────────────────────────────────
OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved: {OUT}")
print("DONE")
