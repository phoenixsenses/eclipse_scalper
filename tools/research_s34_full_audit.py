"""
research_s34_full_audit.py
Full audit: 135-day backfill, all key questions answered.

TEST-A: Period split (Feb-Apr vs Jun) — is edge consistent?
TEST-B: SHORT_NEITHER permutation null — statistically significant?
TEST-C: Naive hold vs state machine on same population
TEST-D: LONG_SILENCE filter ladder (n2h/session/sync/score)
TEST-E: SHORT_NEITHER horizon scan (1h/2h/3h/4h)
TEST-F: SHORT_NEITHER alone — is it enough to run live?
TEST-G: Combined strategy P&L with LONG disabled
"""
from __future__ import annotations
import json, math, random, sqlite3, statistics
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "microstructure.db"
LEDGER  = ROOT / "reports" / "shadow" / "s34_state_machine_shadow.jsonl"
FEE_BPS = 5.0
HORIZON_SHORT_MS = 2 * 3600_000
N_PERM  = 2000
SEED    = 42

# ── helpers ──────────────────────────────────────────────────────────────────

def load_closes() -> list[dict]:
    rows = []
    with LEDGER.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("event") == "CLOSE":
                rows.append(obj)
    return rows

def mark_at(conn, symbol, ts_ms):
    row = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (symbol, ts_ms)).fetchone()
    return float(row[0]) if row else None

def mark_after(conn, symbol, ts_ms):
    row = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1",
        (symbol, ts_ms)).fetchone()
    return float(row[0]) if row else None

def stats(vals):
    if not vals: return {"n": 0, "wr": None, "avg": None, "total": None, "med": None}
    n = len(vals)
    wins = sum(1 for v in vals if v > 0)
    return {"n": n, "wr": round(wins/n*100,1), "avg": round(sum(vals)/n,1),
            "total": round(sum(vals),1), "med": round(statistics.median(vals),1)}

def pprint(label, d):
    if not d or d["n"] == 0:
        print(f"  {label:50s}  N=0")
        return
    print(f"  {label:50s}  N={d['n']:4d}  WR={d['wr']:5.1f}%  avg={d['avg']:+7.1f}  total={d['total']:+8.1f}  bps")

def permtest(observed_vals, null_pool, n_perm=N_PERM, seed=SEED):
    if not observed_vals or not null_pool: return None
    rng = random.Random(seed)
    obs_mean = sum(observed_vals) / len(observed_vals)
    n = len(observed_vals)
    beats = sum(1 for _ in range(n_perm)
                if sum(rng.choice(null_pool) for _ in range(n)) / n >= obs_mean * n)
    return round(beats / n_perm, 4)

def sep(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")

# ── TEST-A: Period split ──────────────────────────────────────────────────────

def test_a(closes):
    sep("TEST-A · Period split: Feb-Apr vs Jun")
    # May is missing from DB; Feb15-Apr30 vs Jun1-Jun30
    APR_END_MS = 1777593600000   # 2026-05-01 00:00 UTC
    JUN_START_MS = 1780272000000  # 2026-06-01 00:00 UTC

    for sig in ["LONG_SILENCE", "SHORT_NEITHER"]:
        sub = [c for c in closes if c.get("signal") == sig]
        early = [c for c in sub if int(c.get("anchor_ts_ms",0)) < APR_END_MS]
        late  = [c for c in sub if int(c.get("anchor_ts_ms",0)) >= JUN_START_MS]
        vals_e = [float(c["net_bps"]) for c in early if c.get("net_bps") is not None]
        vals_l = [float(c["net_bps"]) for c in late  if c.get("net_bps") is not None]
        print(f"\n  {sig}")
        pprint("Feb-Apr", stats(vals_e))
        pprint("Jun", stats(vals_l))
        pprint("ALL", stats(vals_e + vals_l))

# ── TEST-B: SHORT_NEITHER permutation null ────────────────────────────────────

def test_b(closes):
    sep("TEST-B · SHORT_NEITHER permutation null (N_PERM=2000)")
    sn = [c for c in closes if c.get("signal") == "SHORT_NEITHER"]
    vals = [float(c["net_bps"]) for c in sn if c.get("net_bps") is not None]
    pprint("SHORT_NEITHER observed", stats(vals))

    # Null pool: LONG_SILENCE (similar trade universe, different signal)
    ls_vals = [float(c["net_bps"]) for c in closes
               if c.get("signal") == "LONG_SILENCE" and c.get("net_bps") is not None]

    # Permutation: randomly draw N=len(vals) from null pool
    p = permtest(vals, ls_vals)
    obs_t3r = sum(sorted(vals)[max(0,len(vals)//10):-max(1,len(vals)//10)]) if len(vals) > 4 else sum(vals)
    print(f"  p-value (mean >= observed, vs LONG_SILENCE null): {p}")
    print(f"  trimmed sum (t3r proxy): {obs_t3r:.1f} bps")

    # Period split
    APR_END_MS = 1746057600000
    JUN_START_MS = 1748736000000
    early = [float(c["net_bps"]) for c in sn if int(c.get("anchor_ts_ms",0)) < APR_END_MS and c.get("net_bps") is not None]
    late  = [float(c["net_bps"]) for c in sn if int(c.get("anchor_ts_ms",0)) >= JUN_START_MS and c.get("net_bps") is not None]
    pprint("SHORT_NEITHER Feb-Apr", stats(early))
    pprint("SHORT_NEITHER Jun", stats(late))

    # DOW breakdown
    day_names = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    print("\n  by DOW:")
    for dow in sorted(set(c.get("dow",0) for c in sn)):
        sub = [float(c["net_bps"]) for c in sn if c.get("dow")==dow and c.get("net_bps") is not None]
        pprint(f"  DOW={day_names.get(dow,dow)}", stats(sub))

    # Session
    print("\n  by session:")
    for sess in sorted(set(c.get("session","?") for c in sn)):
        sub = [float(c["net_bps"]) for c in sn if c.get("session")==sess and c.get("net_bps") is not None]
        pprint(f"  session={sess}", stats(sub))

# ── TEST-C: Naive hold vs state machine ──────────────────────────────────────

def test_c(closes, conn):
    sep("TEST-C · Naive hold to T+4h vs state machine early exit")
    ls = [c for c in closes if c.get("signal") == "LONG_SILENCE"]

    sm_vals, naive_vals, skipped = [], [], 0
    for c in ls:
        sm_net = c.get("net_bps")
        if sm_net is None: continue

        anchor = int(c.get("anchor_ts_ms", 0))
        entry_px = c.get("entry_price")
        if not entry_px or not anchor: skipped += 1; continue

        exit_px_4h = mark_at(conn, "ETHUSDT", anchor + 4*3600_000)
        if exit_px_4h is None:
            exit_px_4h = mark_after(conn, "ETHUSDT", anchor + 4*3600_000)
        if exit_px_4h is None: skipped += 1; continue

        naive_gross = (float(exit_px_4h) - float(entry_px)) / float(entry_px) * 10_000
        naive_net = naive_gross - FEE_BPS
        sm_vals.append(float(sm_net))
        naive_vals.append(naive_net)

    print(f"  (skipped {skipped}/{len(ls)} due to missing price)")
    pprint("State machine (early exit on noisy)", stats(sm_vals))
    pprint("Naive hold T=0 -> T+4h (same population)", stats(naive_vals))

    # Delta per trade
    if sm_vals and naive_vals:
        deltas = [s - n for s, n in zip(sm_vals, naive_vals)]
        pprint("Delta (SM - naive)", stats(deltas))
        wins = sum(1 for d in deltas if d > 0)
        print(f"  SM better than naive on {wins}/{len(deltas)} trades ({wins/len(deltas)*100:.0f}%)")

# ── TEST-D: LONG_SILENCE filter ladder ────────────────────────────────────────

def test_d(closes):
    sep("TEST-D · LONG_SILENCE filter ladder")
    ls = [c for c in closes if c.get("signal") == "LONG_SILENCE"]
    base = [float(c["net_bps"]) for c in ls if c.get("net_bps") is not None]
    pprint("Base (all LONG_SILENCE)", stats(base))

    # Individual filters
    filters = [
        ("n2h >= 3",      lambda c: (c.get("n2h") or 0) >= 3),
        ("n2h >= 5",      lambda c: (c.get("n2h") or 0) >= 5),
        ("ASIA only",     lambda c: c.get("session") == "ASIA"),
        ("US only",       lambda c: c.get("session") == "US"),
        ("score=3",       lambda c: (c.get("long_score") or c.get("score",0)) == 3),
        ("score>=4",      lambda c: (c.get("long_score") or c.get("score",0)) >= 4),
        ("sync_k<300K",   lambda c: (c.get("sync_k") or 0) < 300_000),
        ("sync_k<500K",   lambda c: (c.get("sync_k") or 0) < 500_000),
        ("Sun or Fri",    lambda c: c.get("dow") in {4, 6}),
        ("not Thu",       lambda c: c.get("dow") != 3),
        ("TIME_EXIT only (silence confirmed)", lambda c: c.get("close_reason") == "TIME_EXIT"),
    ]
    print()
    for label, fn in filters:
        sub = [float(c["net_bps"]) for c in ls if fn(c) and c.get("net_bps") is not None]
        pprint(label, stats(sub))

    # Best combos
    print("\n  --- Combinations ---")
    combos = [
        ("n2h>=3 + ASIA",         lambda c: (c.get("n2h") or 0) >= 3 and c.get("session") == "ASIA"),
        ("n2h>=3 + score=3",      lambda c: (c.get("n2h") or 0) >= 3 and (c.get("long_score") or c.get("score",0)) == 3),
        ("ASIA + score=3",        lambda c: c.get("session") == "ASIA" and (c.get("long_score") or c.get("score",0)) == 3),
        ("ASIA + n2h>=3 + score=3", lambda c: c.get("session")=="ASIA" and (c.get("n2h") or 0)>=3 and (c.get("long_score") or c.get("score",0))==3),
        ("n2h>=3 + sync_k<300K",  lambda c: (c.get("n2h") or 0) >= 3 and (c.get("sync_k") or 0) < 300_000),
        ("ASIA + sync_k<300K",    lambda c: c.get("session")=="ASIA" and (c.get("sync_k") or 0) < 300_000),
        ("not Thu + n2h>=3",      lambda c: c.get("dow") != 3 and (c.get("n2h") or 0) >= 3),
    ]
    for label, fn in combos:
        sub = [float(c["net_bps"]) for c in ls if fn(c) and c.get("net_bps") is not None]
        pprint(label, stats(sub))

# ── TEST-E: SHORT_NEITHER horizon scan ────────────────────────────────────────

def test_e(closes, conn):
    sep("TEST-E · SHORT_NEITHER horizon scan (1h/2h/3h/4h)")
    sn = [c for c in closes if c.get("signal") == "SHORT_NEITHER"]

    for horizon_h in [1, 2, 3, 4]:
        horizon_ms = horizon_h * 3600_000
        vals, skipped = [], 0
        for c in sn:
            # SHORT_NEITHER entry is at btc_confirm time, not cascade T=0
            entry_ts  = int(c.get("entry_ts_ms") or c.get("anchor_ts_ms", 0))
            entry_px  = c.get("entry_price")
            if not entry_px or not entry_ts: skipped += 1; continue

            exit_ts = entry_ts + horizon_ms
            exit_px = mark_at(conn, "ETHUSDT", exit_ts)
            if exit_px is None: exit_px = mark_after(conn, "ETHUSDT", exit_ts)
            if exit_px is None: skipped += 1; continue

            gross = (float(entry_px) - exit_px) / float(entry_px) * 10_000
            net   = gross - FEE_BPS
            vals.append(net)

        pprint(f"SHORT_NEITHER {horizon_h}h hold  (skipped={skipped})", stats(vals))

# ── TEST-F: SHORT_NEITHER standalone live viability ──────────────────────────

def test_f(closes):
    sep("TEST-F · SHORT_NEITHER standalone — live viability")
    sn = [c for c in closes if c.get("signal") == "SHORT_NEITHER" and c.get("net_bps") is not None]

    vals = [float(c["net_bps"]) for c in sn]
    pprint("ALL", stats(vals))

    # Cumulative P&L curve (print key checkpoints)
    cum = 0.0
    drawdown_peak = 0.0
    max_dd = 0.0
    for v in vals:
        cum += v
        if cum > drawdown_peak: drawdown_peak = cum
        dd = drawdown_peak - cum
        if dd > max_dd: max_dd = dd

    print(f"  cumulative total: {sum(vals):+.1f} bps")
    print(f"  max drawdown: -{max_dd:.1f} bps")
    print(f"  avg per trade: {sum(vals)/len(vals):+.1f} bps")

    # Trade frequency
    APR_END_MS  = 1746057600000
    JUN_START_MS = 1748736000000
    early = [c for c in sn if int(c.get("anchor_ts_ms",0)) < APR_END_MS]
    late  = [c for c in sn if int(c.get("anchor_ts_ms",0)) >= JUN_START_MS]
    print(f"  Feb-Apr: {len(early)} trades over ~75 days = {len(early)/75:.2f}/day")
    print(f"  Jun:     {len(late)}  trades over ~30 days = {len(late)/30:.2f}/day")
    print(f"  Overall: {len(sn)} trades over 135 days = {len(sn)/135:.2f}/day  (~{len(sn)/135*30:.0f}/month)")

    # Worst/best trade
    worst = min(vals)
    best  = max(vals)
    print(f"  worst trade: {worst:+.1f} bps")
    print(f"  best trade:  {best:+.1f} bps")

    # P(tail) — trades worse than -100 bps
    tails = [v for v in vals if v <= -100]
    print(f"  tail trades (<=−100 bps): {len(tails)}/{len(vals)} = {len(tails)/len(vals)*100:.0f}%")

    # Permutation vs random
    print(f"\n  Expected per trade if zero-alpha: 0 bps")
    print(f"  Observed: {sum(vals)/len(vals):+.1f} bps")

    # Kelly fraction estimate (rough)
    w = sum(1 for v in vals if v > 0) / len(vals)
    avg_win  = (sum(v for v in vals if v > 0) / max(sum(1 for v in vals if v > 0), 1))
    avg_loss = abs(sum(v for v in vals if v <= 0) / max(sum(1 for v in vals if v <= 0), 1))
    b = avg_win / avg_loss if avg_loss > 0 else 0
    kelly = w - (1-w)/b if b > 0 else 0
    print(f"\n  Kelly fraction (rough): {kelly*100:.1f}%  (avg_win={avg_win:.0f} bps  avg_loss={avg_loss:.0f} bps)")

# ── TEST-G: Disable LONG, SHORT only ─────────────────────────────────────────

def test_g(closes):
    sep("TEST-G · Strategy variants: LONG only / SHORT only / both")
    ls_vals = [float(c["net_bps"]) for c in closes
               if c.get("signal") == "LONG_SILENCE" and c.get("net_bps") is not None]
    sn_vals = [float(c["net_bps"]) for c in closes
               if c.get("signal") == "SHORT_NEITHER" and c.get("net_bps") is not None]

    total_days = 135  # Feb-Jun (ex May gap)
    pprint("LONG_SILENCE only", stats(ls_vals))
    pprint("SHORT_NEITHER only", stats(sn_vals))
    pprint("Both combined", stats(ls_vals + sn_vals))

    print(f"\n  Per-day economics (SHORT_NEITHER only):")
    sn_freq = len(sn_vals) / total_days
    sn_avg  = sum(sn_vals) / len(sn_vals) if sn_vals else 0
    print(f"  {sn_freq:.3f} trades/day × {sn_avg:+.1f} bps/trade = {sn_freq*sn_avg:+.1f} bps/day")
    print(f"  At $1000 notional: ${sn_freq*sn_avg*0.1:.2f}/day expected")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("S34 Full Audit — 135-day backfill")
    closes = load_closes()
    ls = [c for c in closes if c.get("signal") == "LONG_SILENCE"]
    sn = [c for c in closes if c.get("signal") == "SHORT_NEITHER"]
    print(f"Loaded: LONG_SILENCE={len(ls)}  SHORT_NEITHER={len(sn)}")

    test_a(closes)
    test_b(closes)
    test_d(closes)
    test_g(closes)

    print("\nConnecting DB for price queries...")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        test_c(closes, conn)
        test_e(closes, conn)
    finally:
        conn.close()

    test_f(closes)
    print("\n" + "="*70 + "\nDone.")

if __name__ == "__main__":
    main()
