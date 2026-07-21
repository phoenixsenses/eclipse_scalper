"""
S34 State Machine — New Mind-Occupying Questions
NQ1: Provisional T=0 entry with NEITHER-flip simulation (most critical)
NQ2: SHORT score filter — does score>=3 gate improve NEITHER WR?
NQ3: April dead-zone fingerprint — what differs in Block 3?
NQ4: Funding cost — realistic 4h LONG holding cost at 40x
"""
import bisect, json, math, sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

ROOT  = Path("D:/eclipse_scalper")
NAV   = ROOT / "reports/research/s34/S34_NAVIGATION_EVENTS.jsonl"
DB    = ROOT / "data/microstructure.db"
OUT   = ROOT / "reports/research/s34/S34_NEW_QUESTIONS.json"

LIVE_THRESH  = 200_000.0
SIL_LO_MS   = 60_000
SIL_HI_MS   = 30 * 60_000
PROP_THRESH  = 50_000.0
BTC_THRESH   = 750_000.0
FEE_BPS      = 5.0          # round-trip: 2 * 2.5 taker bps
SYNC_WIN_MS  = 10 * 60_000
LEV          = 40.0
CAL_FRAC     = 0.70

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

def get_mark(conn, sym, ts_ms, window_ms=120_000):
    r = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms BETWEEN ? AND ?"
        " ORDER BY ABS(ts_ms - ?) LIMIT 1",
        (sym, ts_ms - window_ms, ts_ms + window_ms, ts_ms)).fetchone()
    return float(r[0]) if r else None

def wr(vals):
    return sum(1 for v in vals if v > 0) / len(vals) if vals else float("nan")

def si(vals):
    if not vals: return "N=0"
    w = wr(vals); m = mean(vals); md = median(vals)
    return f"N={len(vals)} WR={w:.1%} mean={m:+.1f} med={md:+.1f}bps"

# ── load ──────────────────────────────────────────────────────────────────────
print("Loading events and DB...")
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
print(f"  {n_all} events | cal={cal_n} hold={n_all-cal_n}")

with sqlite3.connect(str(DB), timeout=10) as conn:
    eth_ts, eth_not = load_liq(conn, "ETHUSDT", "SELL")
    btc_ts, btc_not = load_liq(conn, "BTCUSDT", "SELL")
    sol_ts, sol_not = load_liq(conn, "SOLUSDT", "SELL")
print(f"  ETH={len(eth_ts):,} BTC={len(btc_ts):,} SOL={len(sol_ts):,}")

# ── classify ──────────────────────────────────────────────────────────────────
def classify(row):
    ts   = int(row["signal_ts_ms"])
    thr  = float(row.get("threshold_usd") or 0)
    net2 = float(row.get("net_2h_bps") or "nan")
    net4v= row.get("net_4h_bps")
    net4 = float(net4v) if net4v is not None else net2
    if not math.isfinite(net2) or thr < LIVE_THRESH: return None
    tags = row.get("tags") or []
    if "BULL_PULLBACK" in tags: return None

    n_prop   = wcnt(eth_ts, eth_not, ts + SIL_LO_MS, ts + SIL_HI_MS, PROP_THRESH)
    sil_eth  = n_prop == 0
    btc_1st  = first_above(btc_ts, btc_not, ts + SIL_LO_MS, ts + SIL_HI_MS, BTC_THRESH)
    b4h      = float(row.get("btc4h_bps") or 0)
    vd       = float(row.get("vdepth_bps") or 0)
    ts_dt    = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    hour     = ts_dt.hour
    dow      = ts_dt.weekday()  # 0=Mon
    sess_us  = 13 <= hour < 21
    session  = ("ASIA" if 0<=hour<7 else "EUROPE" if 7<=hour<13
                else "US" if 13<=hour<21 else "OFF")
    sync_k   = (wsum(btc_ts, btc_not, ts-SYNC_WIN_MS, ts) +
                wsum(sol_ts, sol_not, ts-SYNC_WIN_MS, ts))
    n2h      = wcnt(eth_ts, eth_not, ts-2*3600_000, ts-1000, PROP_THRESH)

    # Score: sil_eth + n2h>=3 + b4h<0 + vd>=30 + sess_us + sync_k>=200K
    score      = sum([int(sil_eth), int(n2h>=3), int(b4h<0), int(vd>=30),
                      int(sess_us), int(sync_k>=200_000)])
    prov_score = sum([int(n2h>=3), int(b4h<0), int(vd>=30),
                      int(sess_us), int(sync_k>=200_000)])  # without sil_eth

    # First ETH follow-on time in silence window
    eth_fo_ts = first_above(eth_ts, eth_not, ts+SIL_LO_MS, ts+SIL_HI_MS, PROP_THRESH)

    return {
        "ts": ts, "ts_dt": ts_dt, "net2": net2, "net4": net4,
        "sil_eth": sil_eth, "btc_1st": btc_1st, "eth_fo_ts": eth_fo_ts,
        "score": score, "prov_score": prov_score,
        "session": session, "hour": hour, "dow": dow,
        "b4h": b4h, "vd": vd, "sess_us": sess_us, "sync_k": sync_k,
        "n2h": n2h, "thr": thr,
        "is_hold": ts >= HOLD_CUTOFF_MS,
    }

print("Classifying...")
classified = [c for row in valid_all if (c := classify(row)) is not None]
print(f"  Classified: {len(classified)}")

# ── NQ1: Provisional T=0 entry with NEITHER flip ──────────────────────────────
print("\n=== NQ1: Provisional T=0 entry + NEITHER flip simulation ===")
#
# Architecture:
#   1. Cascade detected: provisional_score >= 2, non-Europe, non Mon/Wed
#      → Enter LONG at anchor mark price (T=0)
#   2. Monitor 30min:
#      NEITHER: ETH follow-on >=50K + BTC >=750K BOTH in [T+60s, T+30min]
#        → Close LONG at BTC cascade time price
#        → Open SHORT at BTC cascade time price, hold 2h
#        → NET = (btc_price / anchor_price - 1)*10000 * direction + short_net
#      NOISY: ETH follow-on but no BTC >=750K
#        → Close LONG at ETH follow-on time price
#        → NET = (fo_price / anchor_price - 1)*10000 (likely negative)
#      SILENCE: no follow-on
#        → Hold LONG 4h from anchor
#        → NET = net4 - FEE_BPS
#
# For SILENCE: net result = net_4h_bps (pre-computed) - fee
# For NOISY: need mark prices at T=0 and T=follow-on
# For NEITHER: need mark prices at T=0, T=btc_cascade, T=btc+2h
#
# Since mark price queries are slow, we'll:
#   - For SILENCE events: use pre-computed net4
#   - For NOISY/NEITHER: query mark prices from DB (sample)
#
with sqlite3.connect(str(DB), timeout=10) as conn:
    nq1_silence = []   # net bps for SILENCE path
    nq1_noisy   = []   # net bps for NOISY path (no BTC)
    nq1_neither = []   # net bps for NEITHER path (flip)
    nq1_total   = []   # combined
    nq1_hold_silence = []
    nq1_hold_noisy   = []
    nq1_hold_neither = []
    nq1_hold_total   = []

    # Filters: provisional_score>=2, non-Europe, non Mon/Wed LONG
    candidates = [c for c in classified
                  if c["prov_score"] >= 2
                  and c["session"] != "EUROPE"
                  and c["dow"] not in (0, 2)]  # Mon=0, Wed=2

    print(f"  Candidates (prov>=2, non-Europe, non Mon/Wed): N={len(candidates)}")

    for c in candidates:
        ts = c["ts"]
        is_hold = c["is_hold"]

        if c["sil_eth"] and c["btc_1st"] is None:
            # SILENCE: no follow-on, no BTC → hold 4h LONG
            net = c["net4"] - FEE_BPS
            nq1_silence.append(net)
            nq1_total.append(net)
            if is_hold:
                nq1_hold_silence.append(net)
                nq1_hold_total.append(net)

        elif not c["sil_eth"] and c["btc_1st"] is not None:
            # NEITHER path: ETH follow-on + BTC cascade
            # Get prices: anchor (T=0), BTC cascade time, BTC+2h
            anchor_price = get_mark(conn, "ETHUSDT", ts)
            btc_ts_ms    = c["btc_1st"]
            btc_price    = get_mark(conn, "ETHUSDT", btc_ts_ms)
            short_exit   = get_mark(conn, "ETHUSDT", btc_ts_ms + 2*3600_000)

            if anchor_price and btc_price and short_exit:
                # LONG gain: from anchor to BTC cascade (we exit LONG here)
                long_gain_bps = (btc_price / anchor_price - 1) * 10_000
                # SHORT gain: from BTC cascade to BTC+2h (we're short, so positive = down)
                short_gain_bps = (btc_price / short_exit - 1) * 10_000
                # Net = long_gain + short_gain - 2 round-trip fees
                net = long_gain_bps + short_gain_bps - 2 * FEE_BPS
                nq1_neither.append(net)
                nq1_total.append(net)
                if is_hold:
                    nq1_hold_neither.append(net)
                    nq1_hold_total.append(net)

        elif not c["sil_eth"] and c["btc_1st"] is None:
            # NOISY: ETH follow-on but no BTC → exit LONG at follow-on time
            fo_ts = c["eth_fo_ts"]
            if fo_ts:
                anchor_price = get_mark(conn, "ETHUSDT", ts)
                fo_price     = get_mark(conn, "ETHUSDT", fo_ts)
                if anchor_price and fo_price:
                    net = (fo_price / anchor_price - 1) * 10_000 - FEE_BPS
                    nq1_noisy.append(net)
                    nq1_total.append(net)
                    if is_hold:
                        nq1_hold_noisy.append(net)
                        nq1_hold_total.append(net)
            else:
                # Follow-on detected in classification but fo_ts=None (edge)
                pass
        # else: NOISY+NEITHER edge case - skip

results["NQ1_provisional_entry"] = {
    "silence": si(nq1_silence),
    "noisy":   si(nq1_noisy),
    "neither": si(nq1_neither),
    "total":   si(nq1_total),
    "hold_silence": si(nq1_hold_silence),
    "hold_noisy":   si(nq1_hold_noisy),
    "hold_neither": si(nq1_hold_neither),
    "hold_total":   si(nq1_hold_total),
    "n_candidates": len(candidates),
    "description": "Provisional T=0 entry: SILENCE=hold4h, NEITHER=flip to SHORT, NOISY=exit LONG only",
}

print(f"  SILENCE  (hold4h):    {si(nq1_silence)}")
print(f"  NOISY    (exit only): {si(nq1_noisy)}")
print(f"  NEITHER  (flip):      {si(nq1_neither)}")
print(f"  TOTAL:                {si(nq1_total)}")
print(f"  Holdout total:        {si(nq1_hold_total)}")

# ── NQ2: SHORT score filter ────────────────────────────────────────────────────
print("\n=== NQ2: SHORT (NEITHER) score filter ===")
# Score for SHORT excludes sil_eth (=0 for NEITHER events), so score=prov_score
# Since BTC>=750K implies sync_k>=200K is likely (but not guaranteed - BTC liq is BTC not combined)
# We use the FULL score (with sil_eth=0) for SHORT

short_all = [c for c in classified if not c["sil_eth"] and c["btc_1st"] is not None]
print(f"  Total NEITHER events (BTC>=750K, any score): N={len(short_all)}")
print(f"  Holdout: N={sum(1 for c in short_all if c['is_hold'])}")

# NEITHER excludes Sunday per research
for dow_excl_short in [(), (6,)]:
    label = "no_sun" if 6 in dow_excl_short else "all_days"
    subset = [c for c in short_all if c["dow"] not in dow_excl_short]
    vals_all = [c["net2"] - FEE_BPS for c in subset]
    hold_all = [c["net2"] - FEE_BPS for c in subset if c["is_hold"]]
    print(f"\n  [{label}] all scores: {si(vals_all)} | hold: {si(hold_all)}")
    for sc_thr in [1, 2, 3, 4]:
        filt = [c for c in subset if c["score"] >= sc_thr]
        hold_filt = [c for c in filt if c["is_hold"]]
        v_filt = [c["net2"] - FEE_BPS for c in filt]
        h_filt = [c["net2"] - FEE_BPS for c in hold_filt]
        print(f"    score>={sc_thr}: {si(v_filt)} | hold: {si(h_filt)}")

    # Prov_score >= 2 (= full score >=2 since sil_eth=0 for NEITHER)
    prov2 = [c for c in subset if c["prov_score"] >= 2]
    hold_prov2 = [c["net2"] - FEE_BPS for c in prov2 if c["is_hold"]]
    v_prov2 = [c["net2"] - FEE_BPS for c in prov2]
    print(f"    prov>=2 (=score>=2 for NEITHER): {si(v_prov2)} | hold: {si(hold_prov2)}")

results["NQ2_short_score_filter"] = {
    "n_neither": len(short_all),
    "note": "NEITHER = sil_eth=False + BTC_1st found. score=prov_score (sil_eth=0).",
    "all_no_sun": si([c["net2"]-FEE_BPS for c in short_all if c["dow"]!=6]),
    "score_ge2_no_sun": si([c["net2"]-FEE_BPS for c in short_all
                            if c["dow"]!=6 and c["prov_score"]>=2]),
    "score_ge3_no_sun": si([c["net2"]-FEE_BPS for c in short_all
                            if c["dow"]!=6 and c["score"]>=3]),
    "hold_all_no_sun": si([c["net2"]-FEE_BPS for c in short_all
                           if c["dow"]!=6 and c["is_hold"]]),
    "hold_score_ge3_no_sun": si([c["net2"]-FEE_BPS for c in short_all
                                 if c["dow"]!=6 and c["score"]>=3 and c["is_hold"]]),
}

# ── NQ3: April dead-zone fingerprint ──────────────────────────────────────────
print("\n=== NQ3: Block 3 dead-zone (April) fingerprint ===")
# Block 3: 2026-04-14 to 2026-06-10 per RS4
# Block 1: 2026-02-15 to 2026-03-09
# Block 2: 2026-03-09 to 2026-04-13

B1_end = datetime(2026, 3, 9,  tzinfo=timezone.utc).timestamp() * 1000
B2_end = datetime(2026, 4, 14, tzinfo=timezone.utc).timestamp() * 1000
B3_end = datetime(2026, 6, 10, tzinfo=timezone.utc).timestamp() * 1000

def block(c):
    ts = c["ts"]
    if ts < B1_end:   return 1
    if ts < B2_end:   return 2
    if ts < B3_end:   return 3
    return 4  # holdout

all_long  = [c for c in classified if c["sil_eth"] and c["session"] != "EUROPE" and c["dow"] not in (0,2)]
all_short = [c for c in classified if not c["sil_eth"] and c["btc_1st"] is not None and c["dow"] != 6]

for b in [1, 2, 3, 4]:
    bl = [c for c in all_long  if block(c) == b]
    bs = [c for c in all_short if block(c) == b]
    vl = [c["net4"] - FEE_BPS for c in bl]
    vs = [c["net2"] - FEE_BPS for c in bs]

    avg_b4h   = mean([c["b4h"]    for c in bl+bs]) if bl+bs else float("nan")
    avg_sync  = mean([c["sync_k"] for c in bl+bs]) if bl+bs else float("nan")
    avg_n2h   = mean([c["n2h"]    for c in bl+bs]) if bl+bs else float("nan")
    avg_score = mean([c["score"]  for c in bl+bs]) if bl+bs else float("nan")

    print(f"\n  Block {b}: LONG {si(vl)} | SHORT {si(vs)}")
    print(f"    avg_b4h={avg_b4h:+.1f} avg_sync={avg_sync/1000:.0f}K avg_n2h={avg_n2h:.1f} avg_score={avg_score:.1f}")

results["NQ3_regime_fingerprint"] = {}
for b in [1, 2, 3, 4]:
    bl = [c for c in all_long  if block(c) == b]
    bs = [c for c in all_short if block(c) == b]
    vl = [c["net4"] - FEE_BPS for c in bl]
    vs = [c["net2"] - FEE_BPS for c in bs]
    avg_b4h   = round(mean([c["b4h"]    for c in bl+bs]), 1) if bl+bs else None
    avg_sync  = round(mean([c["sync_k"] for c in bl+bs]) / 1000, 0) if bl+bs else None
    avg_n2h   = round(mean([c["n2h"]    for c in bl+bs]), 1) if bl+bs else None
    avg_score = round(mean([c["score"]  for c in bl+bs]), 1) if bl+bs else None
    results["NQ3_regime_fingerprint"][f"block_{b}"] = {
        "long":  si(vl), "short": si(vs),
        "avg_b4h": avg_b4h, "avg_sync_K": avg_sync,
        "avg_n2h": avg_n2h, "avg_score": avg_score,
    }

# ── NQ4: Funding cost ─────────────────────────────────────────────────────────
print("\n=== NQ4: Funding cost for 4h LONG at 40x ===")
with sqlite3.connect(str(DB), timeout=10) as conn:
    rows = conn.execute(
        "SELECT ts_ms, funding_rate FROM mark_prices "
        "WHERE symbol='ETHUSDT' AND funding_rate IS NOT NULL "
        "ORDER BY ts_ms"
    ).fetchall()

fr_vals = [float(r[1]) for r in rows if r[1] is not None and abs(float(r[1])) < 0.01]
if fr_vals:
    avg_fr  = mean(fr_vals)
    med_fr  = median(fr_vals)
    # Each funding period = 8h; 4h hold = 0.5 periods on average
    # Funding cost in bps = funding_rate * 0.5 * LEV * 10000
    cost_4h_bps   = avg_fr * 0.5 * LEV * 10_000  # bps cost for LONG position
    cost_4h_worst = max(fr_vals) * 0.5 * LEV * 10_000
    pct_positive  = sum(1 for v in fr_vals if v > 0) / len(fr_vals)
    print(f"  Funding rate samples: N={len(fr_vals)}")
    print(f"  avg={avg_fr*100:.4f}%  med={med_fr*100:.4f}%  pct_positive={pct_positive:.1%}")
    print(f"  4h LONG 40x cost: avg={cost_4h_bps:+.2f} bps  worst={cost_4h_worst:+.2f} bps")
    results["NQ4_funding_cost"] = {
        "n_samples": len(fr_vals),
        "avg_funding_rate": round(avg_fr * 100, 4),
        "med_funding_rate": round(med_fr * 100, 4),
        "pct_funding_positive": round(pct_positive, 3),
        "avg_4h_LONG_40x_cost_bps": round(cost_4h_bps, 2),
        "worst_4h_LONG_40x_cost_bps": round(cost_4h_worst, 2),
        "note": "Positive funding = LONG pays SHORT. 4h = 0.5 funding periods.",
    }
else:
    print("  No funding rate data found")
    results["NQ4_funding_cost"] = "no data"

# ── NQ5: Score threshold comparison for LONG ─────────────────────────────────
print("\n=== NQ5: LONG score breakdown (DOW+Europe filtered) ===")
long_filtered = [c for c in classified
                 if c["sil_eth"]
                 and c["session"] != "EUROPE"
                 and c["dow"] not in (0, 2)]
for sc_thr in [2, 3, 4, 5]:
    filt = [c for c in long_filtered if c["score"] >= sc_thr]
    hold = [c for c in filt if c["is_hold"]]
    v    = [c["net4"] - FEE_BPS for c in filt]
    h    = [c["net4"] - FEE_BPS for c in hold]
    print(f"  score>={sc_thr}: all={si(v)} | hold={si(h)}")

results["NQ5_long_score_comparison"] = {
    f"score_ge{sc}": {
        "all":  si([c["net4"]-FEE_BPS for c in long_filtered if c["score"]>=sc]),
        "hold": si([c["net4"]-FEE_BPS for c in long_filtered if c["score"]>=sc and c["is_hold"]])
    }
    for sc in [2, 3, 4, 5]
}

# ── NQ6: Session breakdown for SHORT ─────────────────────────────────────────
print("\n=== NQ6: SHORT session breakdown ===")
for sess in ["ASIA", "EUROPE", "US", "OFF"]:
    sub = [c for c in all_short if c["session"] == sess]
    hold = [c for c in sub if c["is_hold"]]
    v    = [c["net2"] - FEE_BPS for c in sub]
    h    = [c["net2"] - FEE_BPS for c in hold]
    print(f"  {sess}: {si(v)} | hold: {si(h)}")

results["NQ6_short_session"] = {
    s: {"all": si([c["net2"]-FEE_BPS for c in all_short if c["session"]==s]),
        "hold": si([c["net2"]-FEE_BPS for c in all_short if c["session"]==s and c["is_hold"]])}
    for s in ["ASIA", "EUROPE", "US", "OFF"]
}

# ── save ──────────────────────────────────────────────────────────────────────
OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved -> {OUT}")
