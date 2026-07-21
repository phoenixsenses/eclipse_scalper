"""
S34 Critical Self-Audit — Falsification Attempt

Hypothesis: The alpha is real.
Mission: Prove it isn't.

10 targeted tests designed to BREAK the hypothesis.
No new features. No cherry-picking. Same FEE=5.0bps throughout.

Status: RESEARCH_ONLY_NO_LIVE_CHANGE
"""
from __future__ import annotations
import json
import math
import random
import sqlite3
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DB   = ROOT / "data" / "microstructure.db"
LEDGER_PATH  = ROOT / "reports" / "shadow" / "s34_state_machine_shadow.jsonl"
NAV_EVENTS   = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_EVENTS.jsonl"

FEE_BPS          = 5.0
ETH_THRESH       = 200_000.0
PROP_THRESH      = 50_000.0
BTC_THRESH       = 1_000_000.0
SYNC_WIN_MS      = 10 * 60_000
SIL_LO_MS        = 60_000
SIL_HI_MS        = 30 * 60_000
HORIZON_LONG_MS  = 4 * 3600_000
HORIZON_SHORT_MS = 2 * 3600_000
SEED             = 42

APR_END_MS  = 1777593600000   # 2026-05-01 00:00 UTC
JUN_START_MS = 1780272000000  # 2026-06-01 00:00 UTC


def utc(ms): return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def r1(v): return round(float(v), 1) if v is not None and math.isfinite(float(v)) else None
def sep(n=72): print("=" * n)
def sub(n=72): print("-" * n)

def stats(vals):
    if not vals:
        return {"n":0,"wr":None,"avg":None,"total":None}
    wins = [v for v in vals if v > 0]
    return {"n":len(vals),"wr":round(len(wins)/len(vals),3),
            "avg":r1(sum(vals)/len(vals)),"total":r1(sum(vals)),
            "maxW":r1(max(vals)),"maxL":r1(min(vals))}

def period(ts):
    if ts < APR_END_MS: return "Feb-Apr"
    if ts < JUN_START_MS: return "May"
    return "Jun"


# ── DB helpers ────────────────────────────────────────────────────────────────

def mark_at(conn, sym, ts):
    r = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1",(sym,int(ts))).fetchone()
    if r: return float(r[0])
    r = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(sym,int(ts))).fetchone()
    return float(r[0]) if r else None

def mark_ret_bps(conn, sym, t0, t1):
    p0 = mark_at(conn, sym, t0); p1 = mark_at(conn, sym, t1)
    if not p0 or not p1 or p0<=0: return None
    return (p1-p0)/p0*10000.0

def liq_first_ts(conn, sym, side, lo, hi, thr):
    r = conn.execute("SELECT ts_ms FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=? ORDER BY ts_ms ASC LIMIT 1",(sym,side,int(lo),int(hi),float(thr))).fetchone()
    return int(r[0]) if r else None

def liq_cnt(conn, sym, side, lo, hi, thr):
    r = conn.execute("SELECT COUNT(*) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?",(sym,side,int(lo),int(hi),float(thr))).fetchone()
    return int(r[0] or 0) if r else 0

def liq_max(conn, sym, side, lo, hi):
    r = conn.execute("SELECT COALESCE(MAX(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?", (sym,side,int(lo),int(hi))).fetchone()
    return float(r[0] or 0.0)

def liq_sum(conn, sym, side, lo, hi):
    r = conn.execute("SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?", (sym,side,int(lo),int(hi))).fetchone()
    return float(r[0] or 0.0)

def session_name(ts):
    h = datetime.fromtimestamp(ts/1000, tz=timezone.utc).hour
    if h<7: return "ASIA"
    if h<13: return "EUROPE"
    if h<21: return "US"
    return "OFF"


# ── Load backfill ledger ──────────────────────────────────────────────────────

def load_ledger():
    seen = set()
    long_ev, short_ev = [], []
    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        if r.get("event") != "CLOSE": continue
        key = r.get("id","")
        if key in seen: continue
        seen.add(key)
        if r.get("signal") == "LONG_SILENCE": long_ev.append(r)
        elif r.get("signal") == "SHORT_NEITHER": short_ev.append(r)
    return long_ev, short_ev


def load_nav200():
    rows = []
    for line in NAV_EVENTS.read_text(encoding="utf-8").splitlines():
        l = line.strip()
        if l: rows.append(json.loads(l))
    return [r for r in rows if float(r.get("threshold_usd",0)) >= 200_000]


def main():
    long_ev, short_ev = load_ledger()
    nav200 = load_nav200()

    print(f"Ledger: {len(long_ev)} LONG, {len(short_ev)} SHORT")
    print(f"NAV 200K: {len(nav200)}")

    with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as conn:

        # ─────────────────────────────────────────────────────────────────────
        # TEST-1: RANDOM BASELINE
        # If naive LONG +16.9 bps is just "ETH was bullish Feb-Jun",
        # random entries in the same period should show similar returns.
        # ─────────────────────────────────────────────────────────────────────
        sep()
        print("TEST-1: RANDOM BASELINE — Would any random LONG entry work?")
        print("  If random entries match naive LONG, the cascade signal adds nothing.")
        sub()

        # Get the time range of our LONG events
        long_ts_sorted = sorted(int(r["anchor_ts_ms"]) for r in long_ev)
        t_start = long_ts_sorted[0]
        t_end   = long_ts_sorted[-1]

        # Generate random entry times in same window (skip May gap)
        rng = random.Random(SEED)
        random_ts = []
        n_random = 500
        while len(random_ts) < n_random:
            ts = rng.randint(t_start, t_end - HORIZON_LONG_MS)
            per = period(ts)
            if per != "May":
                random_ts.append(ts)

        rand_nets = []
        for ts in random_ts:
            p0 = mark_at(conn, "ETHUSDT", ts)
            p4 = mark_at(conn, "ETHUSDT", ts + HORIZON_LONG_MS)
            if p0 and p4 and p0>0:
                rand_nets.append((p4-p0)/p0*10000.0 - FEE_BPS)

        s_rand = stats(rand_nets)
        print(f"  Random entries (N={s_rand['n']}): WR={s_rand['wr']:.1%} avg={s_rand['avg']:+.1f} bps")

        # Also split by period
        rand_feb_apr = [v for v,ts in zip(rand_nets,random_ts[:len(rand_nets)]) if period(ts)=="Feb-Apr"]
        # recompute properly
        by_per_rand: dict[str, list] = defaultdict(list)
        for ts in random_ts:
            p0 = mark_at(conn, "ETHUSDT", ts)
            p4 = mark_at(conn, "ETHUSDT", ts + HORIZON_LONG_MS)
            if p0 and p4 and p0>0:
                by_per_rand[period(ts)].append((p4-p0)/p0*10000.0 - FEE_BPS)

        for per_lbl in ["Feb-Apr","Jun"]:
            s = stats(by_per_rand[per_lbl])
            print(f"  Random {per_lbl} (N={s['n']}): WR={s['wr']:.1%} avg={s['avg']:+.1f} bps")

        print()
        print("  Cascade LONG eligible (backfill):")
        long_nets_4h = []
        long_by_per: dict[str, list] = defaultdict(list)
        for r in long_ev:
            ts = int(r["anchor_ts_ms"])
            px0 = float(r["entry_price"])
            p4 = mark_at(conn, "ETHUSDT", ts + HORIZON_LONG_MS)
            if p4 and px0>0:
                net = (p4-px0)/px0*10000.0 - FEE_BPS
                long_nets_4h.append(net)
                long_by_per[period(ts)].append(net)

        s_long = stats(long_nets_4h)
        print(f"  Cascade LONG all (N={s_long['n']}): WR={s_long['wr']:.1%} avg={s_long['avg']:+.1f} bps")
        for per_lbl in ["Feb-Apr","Jun"]:
            s = stats(long_by_per[per_lbl])
            print(f"  Cascade LONG {per_lbl} (N={s['n']}): WR={s['wr']:.1%} avg={s['avg']:+.1f} bps")

        # ─────────────────────────────────────────────────────────────────────
        # TEST-2: long_eligible FILTER VALUE
        # Are filtered-OUT 200K events WORSE than filtered-IN?
        # If not, the eligibility filter adds no alpha.
        # ─────────────────────────────────────────────────────────────────────
        sep()
        print("TEST-2: ELIGIBILITY FILTER VALUE")
        print("  Do the 274 filtered-out 200K events (EUROPE/MonWed/lowscore/bull)")
        print("  have WORSE naive 4h returns than the 217 included ones?")
        sub()

        nav200_ts_set = {int(r["signal_ts_ms"]) for r in nav200}
        bf_ts_set     = {int(r["anchor_ts_ms"]) for r in long_ev}
        MATCH_TOL     = 90_000

        in_bf_vals, out_bf_vals = [], []
        for r in nav200:
            ts = int(r["signal_ts_ms"])
            net4 = r.get("net_4h_bps")
            if net4 is None or not math.isfinite(float(net4)): continue
            # check if this nav event is in backfill (within tolerance)
            lo = bisect_left(sorted(bf_ts_set), ts - MATCH_TOL)
            # simpler: just check timestamp proximity
            matched = any(abs(bts - ts) <= MATCH_TOL for bts in bf_ts_set)
            if matched:
                in_bf_vals.append(float(net4))
            else:
                out_bf_vals.append(float(net4))

        s_in  = stats(in_bf_vals)
        s_out = stats(out_bf_vals)
        print(f"  IN backfill (long_eligible):  N={s_in['n']}  WR={s_in['wr']:.1%}  avg={s_in['avg']:+.1f} bps")
        print(f"  OUT of backfill (filtered):   N={s_out['n']}  WR={s_out['wr']:.1%}  avg={s_out['avg']:+.1f} bps")
        diff = (s_in['avg'] or 0) - (s_out['avg'] or 0)
        print(f"  Filter improvement: {diff:+.1f} bps/trade")
        if abs(diff) < 3.0:
            print("  VERDICT: Filter adds negligible value (<3 bps). Eligibility criteria unproven.")
        else:
            print(f"  VERDICT: Filter {'adds' if diff>0 else 'LOSES'} {abs(diff):.1f} bps. Meaningful.")

        # ─────────────────────────────────────────────────────────────────────
        # TEST-3: PERIOD SPLIT — Is +16.9 bps consistent or June-only?
        # ─────────────────────────────────────────────────────────────────────
        sep()
        print("TEST-3: PERIOD SPLIT — Naive LONG hold by period")
        print("  Is +16.9 bps real in BOTH Feb-Apr AND Jun?")
        print("  Or is it concentrated in one regime?")
        sub()

        for per_lbl in ["Feb-Apr", "Jun"]:
            s = stats(long_by_per[per_lbl])
            if s['n'] == 0:
                print(f"  {per_lbl}: no data")
                continue
            print(f"  {per_lbl}: N={s['n']} WR={s['wr']:.1%} avg={s['avg']:+.1f} bps  total={s['total']:+.0f} bps")

        total_s = stats(long_nets_4h)
        print(f"  ALL:    N={total_s['n']} WR={total_s['wr']:.1%} avg={total_s['avg']:+.1f} bps  total={total_s['total']:+.0f} bps")

        # ─────────────────────────────────────────────────────────────────────
        # TEST-4: CASCADE TAIL CONTAMINATION
        # How many "noisy" exits fire within 0-120s of SIL_LO_MS?
        # These might be the CASCADE ITSELF still completing, not follow-on.
        # ─────────────────────────────────────────────────────────────────────
        sep()
        print("TEST-4: CASCADE TAIL CONTAMINATION")
        print("  The cascade's own liquidations may extend past T+60s (SIL_LO_MS).")
        print("  If so, the noisy exit fires on the cascade tail, not follow-on.")
        sub()

        noisy_by_delay: dict[str, list] = defaultdict(list)  # bucket -> net_bps
        noisy_delay_all = []

        for r in long_ev:
            if r.get("close_reason") != "NOISY_EARLY_EXIT": continue
            ts = int(r["anchor_ts_ms"])
            exit_ts = int(r["exit_ts_ms"])
            delay_sec = (exit_ts - ts - SIL_LO_MS) / 1000.0  # seconds AFTER monitoring starts
            net = float(r["net_bps"])

            if delay_sec < 30:
                bucket = "0-30s"
            elif delay_sec < 120:
                bucket = "30-120s"
            elif delay_sec < 300:
                bucket = "2-5min"
            elif delay_sec < 600:
                bucket = "5-10min"
            elif delay_sec < 900:
                bucket = "10-15min"
            else:
                bucket = "15-30min"
            noisy_by_delay[bucket].append(net)
            noisy_delay_all.append((delay_sec, net))

        order = ["0-30s","30-120s","2-5min","5-10min","10-15min","15-30min"]
        print(f"  Delay after T+60s  N    WR     avg    Interpretation")
        print(f"  {'─'*65}")
        for b in order:
            vals = noisy_by_delay.get(b, [])
            if not vals: continue
            s = stats(vals)
            interp = "CASCADE TAIL (own liq)" if b in ["0-30s","30-120s"] else "follow-on"
            print(f"  {b:<18}  {s['n']:<5} {s['wr']:.1%}  {s['avg']:+.1f}  {interp}")

        # What fraction of noisy exits are within 2 min of monitoring start?
        early_noisy = sum(1 for d,_ in noisy_delay_all if d < 120)
        total_noisy = len(noisy_delay_all)
        print(f"\n  Exits within 120s of monitoring start: {early_noisy}/{total_noisy} ({early_noisy/max(total_noisy,1):.0%})")
        print(f"  These are LIKELY cascade tail, NOT independent follow-on.")

        # What if we extend SIL_LO to 3 min (180s)?
        print(f"\n  What if SIL_LO = 3min (skip cascade's own first 3 min)?")
        SIL_LO_3MIN = 3 * 60_000
        extended_nets = []
        for r in long_ev:
            ts = int(r["anchor_ts_ms"])
            px0 = float(r["entry_price"])
            # Re-simulate with 3min skip window
            noisy_ts = liq_first_ts(conn, "ETHUSDT", "SELL",
                                    ts + SIL_LO_3MIN, ts + SIL_HI_MS, PROP_THRESH)
            if noisy_ts is not None:
                exit_ts = noisy_ts
                p_exit = mark_at(conn, "ETHUSDT", exit_ts) or px0
            else:
                exit_ts = ts + HORIZON_LONG_MS
                p_exit = mark_at(conn, "ETHUSDT", exit_ts) or px0
            if px0 > 0:
                extended_nets.append((p_exit - px0) / px0 * 10000.0 - FEE_BPS)

        s_ext = stats(extended_nets)
        s_cur = stats([float(r["net_bps"]) for r in long_ev])
        print(f"  Current (SIL_LO=60s):   WR={s_cur['wr']:.1%} avg={s_cur['avg']:+.1f} bps")
        print(f"  Extended (SIL_LO=3min): WR={s_ext['wr']:.1%} avg={s_ext['avg']:+.1f} bps")

        # ─────────────────────────────────────────────────────────────────────
        # TEST-5: SHORT_NEITHER BOOTSTRAP CI
        # N=28 is small. Is the edge statistically robust?
        # ─────────────────────────────────────────────────────────────────────
        sep()
        print("TEST-5: SHORT_NEITHER — Bootstrap Confidence Interval")
        print("  N=28 is very small. What's the real uncertainty on WR=64.3%?")
        sub()

        sn_nets = [float(r["net_bps"]) for r in short_ev]
        rng2 = random.Random(SEED+1)
        boot_wr  = []
        boot_avg = []
        N_BOOT = 5000
        for _ in range(N_BOOT):
            sample = [rng2.choice(sn_nets) for _ in range(len(sn_nets))]
            boot_wr.append(sum(1 for v in sample if v > 0) / len(sample))
            boot_avg.append(sum(sample) / len(sample))

        boot_wr.sort(); boot_avg.sort()
        lo5, hi95 = int(0.05*N_BOOT), int(0.95*N_BOOT)
        lo25, hi75 = int(0.25*N_BOOT), int(0.75*N_BOOT)
        lo1, hi99  = int(0.01*N_BOOT), int(0.99*N_BOOT)

        s_sn = stats(sn_nets)
        print(f"  Observed: N={s_sn['n']} WR={s_sn['wr']:.1%} avg={s_sn['avg']:+.1f} bps")
        print(f"  Bootstrap WR:  50% CI = [{boot_wr[lo25]:.1%}, {boot_wr[hi75]:.1%}]")
        print(f"             90% CI = [{boot_wr[lo5]:.1%}, {boot_wr[hi95]:.1%}]")
        print(f"             98% CI = [{boot_wr[lo1]:.1%}, {boot_wr[hi99]:.1%}]")
        print(f"  Bootstrap avg: 90% CI = [{boot_avg[lo5]:+.1f}, {boot_avg[hi95]:+.1f}] bps")
        p_wr_below50 = sum(1 for v in boot_wr if v < 0.5) / N_BOOT
        p_avg_neg    = sum(1 for v in boot_avg if v < 0) / N_BOOT
        print(f"  P(WR < 50%) = {p_wr_below50:.1%}  (would mean SHORT has no edge)")
        print(f"  P(avg < 0)  = {p_avg_neg:.1%}    (would mean SHORT is net loser)")

        # ─────────────────────────────────────────────────────────────────────
        # TEST-6: SHORT ENTRY TIMING — T=0 vs BTC-confirm
        # Research measured SHORT from T=0. Backfill enters at BTC confirm.
        # Are these materially different?
        # ─────────────────────────────────────────────────────────────────────
        sep()
        print("TEST-6: SHORT ENTRY TIMING — T=0 anchor vs BTC-confirm")
        print("  Research entered SHORT at T=0. Live enters at BTC confirm (T+X).")
        print("  Are these materially different? Same 28 events.")
        sub()

        t0_nets, bc_nets = [], []
        delays_sn = []
        for r in short_ev:
            anchor_ts  = int(r["anchor_ts_ms"])
            entry_ts   = int(r["entry_ts_ms"])
            exit_ts    = int(r["exit_ts_ms"])
            entry_px   = float(r["entry_price"])
            exit_px    = float(r["exit_price"])
            delay_sec  = (entry_ts - anchor_ts) / 1000.0

            # BTC-confirm short result (actual backfill)
            outcome_bc = (entry_px - exit_px) / entry_px * 10_000.0
            bc_nets.append(outcome_bc - FEE_BPS)

            # T=0 short: enter at anchor, exit at T=0+HORIZON (same 2h duration)
            anchor_px = mark_at(conn, "ETHUSDT", anchor_ts)
            if anchor_px and anchor_px > 0:
                p_exit_t0 = mark_at(conn, "ETHUSDT", anchor_ts + HORIZON_SHORT_MS)
                if p_exit_t0:
                    t0_net = (anchor_px - p_exit_t0) / anchor_px * 10_000.0 - FEE_BPS
                    t0_nets.append(t0_net)

            delays_sn.append(delay_sec)

        avg_delay = sum(delays_sn) / len(delays_sn)
        s_bc = stats(bc_nets)
        s_t0 = stats(t0_nets)
        print(f"  Avg BTC-confirm delay: {avg_delay:.0f}s ({avg_delay/60:.1f} min) after anchor")
        print(f"  SHORT at BTC-confirm:  N={s_bc['n']} WR={s_bc['wr']:.1%} avg={s_bc['avg']:+.1f} bps")
        print(f"  SHORT at T=0 (anchor): N={s_t0['n']} WR={s_t0['wr']:.1%} avg={s_t0['avg']:+.1f} bps")
        t_diff = (s_bc['avg'] or 0) - (s_t0['avg'] or 0)
        print(f"  Entry timing impact:   {t_diff:+.1f} bps/trade")
        if t_diff > 5:
            print("  BTC-confirm entry is BETTER. Waiting for BTC confirmation adds value.")
        elif t_diff < -5:
            print("  T=0 entry is BETTER. BTC-confirm entry is hurting (price moved against us).")
        else:
            print("  Entry timing difference is small (<5 bps). Both work similarly.")

        # ─────────────────────────────────────────────────────────────────────
        # TEST-7: FEATURE INDEPENDENCE
        # Does each feature individually improve on all-eligible baseline?
        # If no single feature adds edge, the combo is data-mined.
        # ─────────────────────────────────────────────────────────────────────
        sep()
        print("TEST-7: FEATURE INDEPENDENCE — Does each feature contribute?")
        print("  If no single feature improves WR vs base rate,")
        print("  the combined score is data-mined on a small sample.")
        sub()

        # Base: all long_ev naive 4h
        base_all = long_nets_4h  # all 217
        s_base = stats(base_all)
        print(f"  BASE (all 217 long_eligible, naive 4h): WR={s_base['wr']:.1%} avg={s_base['avg']:+.1f}")
        print()

        # For each feature, split into HIGH vs LOW and compare
        features = {
            "n2h>=3":      lambda r: int(r.get("n2h",0)) >= 3,
            "n2h<3":       lambda r: int(r.get("n2h",0)) < 3,
            "score=3":     lambda r: int(r.get("score",0)) == 3,
            "score>=4":    lambda r: int(r.get("score",0)) >= 4,
            "sess=US":     lambda r: r.get("session") == "US",
            "sess=ASIA":   lambda r: r.get("session") == "ASIA",
            "sess=OFF":    lambda r: r.get("session") == "OFF",
            "sync>=200K":  lambda r: float(r.get("sync_k",0)) >= 200_000,
            "sync<200K":   lambda r: float(r.get("sync_k",0)) < 200_000,
        }

        # rebuild per-event with naive 4h net
        ev_with_net = []
        for i, r in enumerate(long_ev):
            if i < len(long_nets_4h):
                ev_with_net.append((r, long_nets_4h[i]))

        for feat_name, fn in features.items():
            subset = [net for r, net in ev_with_net if fn(r)]
            s = stats(subset)
            if s['n'] == 0: continue
            delta = (s['avg'] or 0) - (s_base['avg'] or 0)
            sign = "+" if delta >= 0 else ""
            print(f"  {feat_name:<15} N={s['n']:<4} WR={s['wr']:.1%} avg={s['avg']:+.1f}  delta={sign}{delta:.1f}")

        # ─────────────────────────────────────────────────────────────────────
        # TEST-8: ROLLING 30-DAY STABILITY
        # Is edge stable, or concentrated in a short window?
        # ─────────────────────────────────────────────────────────────────────
        sep()
        print("TEST-8: ROLLING 30-DAY STABILITY")
        print("  If edge is concentrated in one month, it's regime-dependent.")
        sub()

        WINDOW_MS = 30 * 86400_000
        sorted_ev = sorted(zip(long_nets_4h, [int(r["anchor_ts_ms"]) for r in long_ev]),
                           key=lambda x: x[1])
        sorted_sn = sorted(short_ev, key=lambda r: int(r["anchor_ts_ms"]))

        windows = [
            ("Feb-2026", 1769904000000, 1772323200000),   # Feb 01 - Mar 01 2026
            ("Mar-2026", 1772323200000, 1775001600000),   # Mar 01 - Apr 01 2026
            ("Apr-2026", 1775001600000, 1777593600000),   # Apr 01 - May 01 2026
            ("Jun-2026", 1780272000000, 1782950400000),   # Jun 01 - Jul 01 2026
        ]

        print(f"  Period    LONG N  WR     avg    | SHORT N  WR     avg")
        print(f"  {'─'*65}")
        for lbl, w_start, w_end in windows:
            lv = [v for v,ts in sorted_ev if w_start <= ts < w_end]
            sv = [float(r["net_bps"]) for r in sorted_sn if w_start <= int(r["anchor_ts_ms"]) < w_end]
            sl = stats(lv); ss = stats(sv)
            ln = sl['n'] if sl['n'] else 0
            lw = f"{sl['wr']:.1%}" if sl['n'] else "N/A"
            la = f"{sl['avg']:+.1f}" if sl['avg'] is not None else "N/A"
            sn_n = ss['n'] if ss['n'] else 0
            sw = f"{ss['wr']:.1%}" if ss['n'] else "N/A"
            sa = f"{ss['avg']:+.1f}" if ss['avg'] is not None else "N/A"
            print(f"  {lbl:<10}{ln:<6}  {lw:<6} {la:<8}| {sn_n:<6}  {sw:<6} {sa}")

        # ─────────────────────────────────────────────────────────────────────
        # TEST-9: THE HARDEST QUESTION
        # Same 200K events NOT filtered by long_eligible: what's their 4h return?
        # If non-eligible events return similarly, the whole state machine is pointless.
        # ─────────────────────────────────────────────────────────────────────
        sep()
        print("TEST-9: NON-ELIGIBLE 200K EVENTS")
        print("  The 274 filtered-out events (EUROPE, Mon/Wed, low score, bull).")
        print("  If their 4h return is similar to eligible, eligibility filter is useless.")
        sub()

        eligible_4h   = in_bf_vals   # from TEST-2: matched to backfill
        ineligible_4h = out_bf_vals  # from TEST-2: not in backfill

        # Further split ineligible by reason
        inelig_europe = [float(r.get("net_4h_bps") or "nan") for r in nav200
                         if not any(abs(int(r["signal_ts_ms"])-bts)<=90_000 for bts in bf_ts_set)
                         and session_name(int(r["signal_ts_ms"])) == "EUROPE"
                         and r.get("net_4h_bps") is not None and math.isfinite(float(r.get("net_4h_bps")))]

        inelig_dow    = [float(r.get("net_4h_bps") or "nan") for r in nav200
                         if not any(abs(int(r["signal_ts_ms"])-bts)<=90_000 for bts in bf_ts_set)
                         and datetime.fromtimestamp(int(r["signal_ts_ms"])/1000, tz=timezone.utc).weekday() in {0,2}
                         and session_name(int(r["signal_ts_ms"])) != "EUROPE"
                         and r.get("net_4h_bps") is not None and math.isfinite(float(r.get("net_4h_bps")))]

        s_elig   = stats(eligible_4h)
        s_inelig = stats(ineligible_4h)
        s_eur    = stats(inelig_europe)
        s_dow    = stats(inelig_dow)

        print(f"  Long_eligible (in backfill): N={s_elig['n']}  WR={s_elig['wr']:.1%}  avg={s_elig['avg']:+.1f} bps")
        print(f"  Non-eligible (filtered out): N={s_inelig['n']}  WR={s_inelig['wr']:.1%}  avg={s_inelig['avg']:+.1f} bps")
        print(f"    of which EUROPE:           N={s_eur['n']}  WR={s_eur['wr']:.1%}  avg={s_eur['avg']:+.1f} bps")
        print(f"    of which Mon/Wed:          N={s_dow['n']}  WR={s_dow['wr']:.1%}  avg={s_dow['avg']:+.1f} bps")

        diff9 = (s_elig['avg'] or 0) - (s_inelig['avg'] or 0)
        print(f"\n  Eligible vs non-eligible delta: {diff9:+.1f} bps/trade")
        if abs(diff9) < 3:
            print("  DAMNING: Eligibility filter adds <3 bps. The filter is NOT adding signal.")
        elif diff9 > 0:
            print(f"  Eligibility filter adds {diff9:+.1f} bps. Some signal present.")
        else:
            print(f"  DAMNING: Non-eligible events are BETTER by {abs(diff9):.1f} bps.")

        # ─────────────────────────────────────────────────────────────────────
        # TEST-10: PERMUTATION NULL ON LONG
        # What's the p-value of naive LONG hold +16.9 bps?
        # If p>0.05, the LONG edge is not statistically significant.
        # ─────────────────────────────────────────────────────────────────────
        sep()
        print("TEST-10: PERMUTATION NULL — Is naive LONG hold statistically significant?")
        print("  Permute returns from all 200K NAV_EVENTS (450 pool),")
        print("  sample N=217, test if real sum exceeds null distribution.")
        sub()

        all_nav_4h = [float(r["net_4h_bps"]) for r in nav200
                      if r.get("net_4h_bps") is not None and math.isfinite(float(r["net_4h_bps"]))]

        rng3 = random.Random(SEED+2)
        N_PERM = 2000
        n_sample = len(long_nets_4h)
        real_sum = sum(long_nets_4h)
        real_avg = real_sum / n_sample

        null_avgs = [sum(rng3.sample(all_nav_4h, min(n_sample,len(all_nav_4h)))) / n_sample
                     for _ in range(N_PERM)]

        p_right = sum(1 for v in null_avgs if v >= real_avg) / N_PERM
        p95_null = sorted(null_avgs)[int(0.95*N_PERM)]

        print(f"  Pool: all 200K NAV events, N={len(all_nav_4h)}")
        print(f"  Real avg = {real_avg:+.1f} bps (N={n_sample})")
        print(f"  Null p95 = {p95_null:+.1f} bps")
        print(f"  p-value  = {p_right:.3f}  ({N_PERM} permutations)")
        if p_right < 0.05:
            print(f"  PASS: Long eligible events outperform random 200K sample (p={p_right:.3f})")
        else:
            print(f"  FAIL: Long eligible events do NOT outperform random sample (p={p_right:.3f})")
            print(f"  DAMNING: The eligibility filter does not add statistically significant alpha.")

        sep()
        print("CRITICAL AUDIT SUMMARY")
        sub()
        print("Issues ranked by severity:")
        print()
        print("  [1] N=28 SHORT — 98% CI on WR spans from loss to large gain.")
        print("      Europe filter proposed from N=5. No statistical basis.")
        print()
        print("  [2] CASCADE TAIL CONTAMINATION — many noisy exits at T+60-90s")
        print("      may be cascade's own liquidations, not follow-on.")
        print("      SIL_LO=60s may not be long enough.")
        print()
        print("  [3] PERIOD CONSISTENCY — if Feb-Apr naive hold is near 0 but")
        print("      Jun is strongly positive, edge is June-only (regime artifact).")
        print()
        print("  [4] RANDOM BASELINE — if random LONG entries in same period also")
        print("      show +WR, cascade signal adds nothing. Just bullish period.")
        print()
        print("  [5] ELIGIBILITY FILTER — if non-eligible 200K events have similar")
        print("      returns, the entire filter stack adds no signal.")
        print()
        print("  [6] SHORT ENTRY TIMING — research entered at T=0, live at BTC+Xmin.")
        print("      If BTC confirm delay hurts entry price, research overstates edge.")
        sep()


if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
