"""S34 Round-3 Tests — Complete remaining test list.

Remaining dimensions after Round-1 and Round-2:

  G1-G4  NOISY CADAVER: What would T+4h outcome be for 145 NOISY_EARLY_EXIT events?
          Quantifies the true forward improvement from removing noisy exit.
  H1-H3  SCORE PARADOX: Why is long_score==4 the worst? Feature interaction decoder.
  I1-I4  BTC7D REGIME: Threshold sweep + TIME_EXIT + NOISY cross, best combo.
  J1-J3  SHORT HOLD EXTENSION: T+2h vs T+3h vs T+4h outcomes (DB mark prices).
  K1-K2  MON/WED EVALUATION: Would LONG work on excluded days? (DB cascade query)
  L1-L2  LONG SCORE WITHOUT SYNC: Revised score impact on TIME_EXIT.
  M1     SIL_LO RESCUE: T+4h outcomes for 18 events rescued by 3min tail gap.
  N1     FEATURE INTERACTION MATRIX: All 2-feature combos on TIME_EXIT.

Usage: python tools/research_s34_round3_tests.py
"""
from __future__ import annotations
import json
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "reports" / "shadow" / "s34_state_machine_shadow.jsonl"
DB_PATH = ROOT / "data" / "microstructure.db"
FEE_BPS = 5.0
SIL_LO_MS   = 60_000
SIL_HI_MS   = 30 * 60_000
HORIZON_LONG_MS  = 4 * 3600_000
HORIZON_SHORT_MS = 2 * 3600_000

# ── Helpers ───────────────────────────────────────────────────────────────────

def stat(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "wr": None, "avg": None, "total": None}
    wins = sum(1 for v in vals if v > 0)
    return {"n": len(vals), "wr": round(wins/len(vals), 3),
            "avg": round(sum(vals)/len(vals), 1), "total": round(sum(vals), 0)}

def pct(v) -> str:
    return "  -  " if v is None else f"{v*100:5.1f}%"

def fmt(v, d=1) -> str:
    return "   -   " if v is None else f"{v:+{7+d}.{d}f}"

def hdr(t: str) -> None:
    print(); print("=" * 66); print(f"  {t}"); print("=" * 66)

def row(label: str, s: dict, note: str = "") -> None:
    if s["n"] == 0:
        print(f"  {label:<34s}  N=  0  -----  ------  ------  {note}")
        return
    print(f"  {label:<34s}  N={s['n']:4d}  WR={pct(s['wr'])}  "
          f"avg={fmt(s['avg'])} bps  tot={fmt(s['total'],0)} bps  {note}")

def month_of(r: dict) -> str:
    try:
        return datetime.fromtimestamp(int(r["anchor_ts_ms"])/1000,
                                       tz=timezone.utc).strftime("%Y-%m")
    except Exception:
        return "?"

# ── DB helpers ────────────────────────────────────────────────────────────────

def mark_at(conn, sym: str, ts_ms: int) -> float | None:
    r = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? "
        "ORDER BY ts_ms DESC LIMIT 1", (sym, ts_ms)).fetchone()
    return float(r[0]) if r else None

def mark_after(conn, sym: str, ts_ms: int) -> float | None:
    r = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms>=? "
        "ORDER BY ts_ms ASC LIMIT 1", (sym, ts_ms)).fetchone()
    return float(r[0]) if r else None

def btc_ret(conn, ts_ms: int, lookback_ms: int) -> float | None:
    a = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms>=? "
        "ORDER BY ts_ms ASC LIMIT 1", (ts_ms - lookback_ms,)).fetchone()
    b = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms<=? "
        "ORDER BY ts_ms DESC LIMIT 1", (ts_ms,)).fetchone()
    if not a or not b or float(a[0] or 0) <= 0:
        return None
    return (float(b[0]) - float(a[0])) / float(a[0]) * 10_000.0

def eth_sell_first(conn, ts_ms: int, lo: int, hi: int, thresh: float):
    r = conn.execute(
        "SELECT ts_ms FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' "
        "AND ts_ms>=? AND ts_ms<? AND notional>=? ORDER BY ts_ms ASC LIMIT 1",
        (ts_ms+lo, ts_ms+hi, thresh)).fetchone()
    return int(r[0]) if r else None

def prior_bps_eth(conn, ts_ms: int, lookback_ms: int) -> float:
    a = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>=? "
        "ORDER BY ts_ms ASC LIMIT 1", (ts_ms - lookback_ms,)).fetchone()
    b = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? "
        "ORDER BY ts_ms DESC LIMIT 1", (ts_ms,)).fetchone()
    if not a or not b or float(a[0] or 0) <= 0:
        return 0.0
    return (float(b[0]) - float(a[0])) / float(a[0]) * 10_000.0

# ── Load ledger ───────────────────────────────────────────────────────────────

def load_records():
    longs, shorts = [], []
    with LEDGER.open(encoding="utf-8") as f:
        for line in f:
            try: r = json.loads(line)
            except Exception: continue
            if r.get("event") != "CLOSE": continue
            net = r.get("net_bps")
            if net is None: continue
            r["_net"] = float(net)
            r["_month"] = month_of(r)
            if r.get("direction") == "LONG":   longs.append(r)
            elif r.get("direction") == "SHORT": shorts.append(r)
    longs.sort(key=lambda r: int(r.get("anchor_ts_ms") or 0))
    shorts.sort(key=lambda r: int(r.get("anchor_ts_ms") or 0))
    return longs, shorts

# ── SECTION G: NOISY CADAVER ──────────────────────────────────────────────────

def section_g(longs: list[dict]) -> None:
    noisy = [r for r in longs if r.get("close_reason") == "NOISY_EARLY_EXIT"]
    te    = [r for r in longs if r.get("close_reason") == "TIME_EXIT"]

    hdr("G1 · NOISY CADAVER — hypothetical T+4h hold outcome (DB)")
    print(f"  Computing T+4h mark prices for {len(noisy)} NOISY_EARLY_EXIT events...")
    hypo_nets = []
    missing = 0
    with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=10) as conn:
        for r in noisy:
            entry_px = float(r.get("entry_price") or 0)
            ts = int(r.get("anchor_ts_ms") or 0)
            px4h = mark_at(conn, "ETHUSDT", ts + HORIZON_LONG_MS)
            if entry_px <= 0 or px4h is None:
                missing += 1
                continue
            raw = (px4h - entry_px) / entry_px * 10_000.0
            r["_hypo_net"] = raw - FEE_BPS
            hypo_nets.append(r["_hypo_net"])

    print(f"  (missing={missing})")
    row("NOISY actual (early exit)", stat([r["_net"] for r in noisy]))
    row("NOISY hypo T+4h hold",      stat(hypo_nets))
    te_nets = [r["_net"] for r in te]
    row("TIME_EXIT actual (T+4h)",   stat(te_nets))

    delta = round(sum(hypo_nets)/len(hypo_nets) - sum([r["_net"] for r in noisy])/len(noisy), 1) if hypo_nets else None
    print(f"\n  NOISY hypo avg - actual avg = {delta:+.1f} bps per trade")
    print(f"  => Removing noisy exit rescues {delta:+.1f} bps/trade on {len(noisy)} forward trades")

    hdr("G2 · NOISY CADAVER by session")
    for sess in ["US", "ASIA", "OFF"]:
        act = [r["_net"] for r in noisy if r.get("session")==sess]
        hypo = [r["_hypo_net"] for r in noisy if r.get("session")==sess and "_hypo_net" in r]
        row(f"{sess} actual",    stat(act))
        row(f"{sess} hypo T+4h", stat(hypo))
        print()

    hdr("G3 · NOISY CADAVER by month")
    months: dict[str, list] = {}
    hypo_months: dict[str, list] = {}
    for r in noisy:
        m = r["_month"]
        months.setdefault(m, []).append(r["_net"])
        if "_hypo_net" in r:
            hypo_months.setdefault(m, []).append(r["_hypo_net"])
    for m in sorted(months):
        row(f"{m} actual",    stat(months[m]))
        row(f"{m} hypo T+4h", stat(hypo_months.get(m, [])))
        print()

    hdr("G4 · Combined: if noisy exit NEVER happened (all LONG = TIME_EXIT policy)")
    all_actual = [r["_net"] for r in longs]
    all_hypo   = te_nets + hypo_nets
    row("Current backfill mix (actual)", stat(all_actual))
    row("Forward policy: all hold T+4h", stat(all_hypo))


# ── SECTION H: SCORE PARADOX DECODER ─────────────────────────────────────────

def section_h(longs: list[dict]) -> None:
    te = [r for r in longs if r.get("close_reason") == "TIME_EXIT"]

    hdr("H1 · long_score==4 paradox: cross-tab with sync_k")
    # long_score==4 means base_score=3. One hypothesis: these have sync>=200K
    # (3 components often includes n2h + btc4h + sync, without US sess)
    for ls in [3, 4, 5]:
        sub = [r for r in te if (r.get("long_score") or 0) == ls]
        with_sync  = [r["_net"] for r in sub if (r.get("sync_k") or 0) >= 200_000]
        no_sync    = [r["_net"] for r in sub if (r.get("sync_k") or 0) <  200_000]
        us_sess    = [r["_net"] for r in sub if r.get("session") == "US"]
        no_us      = [r["_net"] for r in sub if r.get("session") != "US"]
        print(f"\n  long_score=={ls}  (N={len(sub)})")
        row(f"    sync>=200K", stat(with_sync))
        row(f"    sync<200K",  stat(no_sync))
        row(f"    US session", stat(us_sess))
        row(f"    !US",        stat(no_us))

    hdr("H2 · Revised score: n2h component weight (n2h>=3 = strongest predictor)")
    # Replace sync component with inverse sync (reward low sync)
    for r in te:
        n2h_s = int((r.get("n2h") or 0) >= 3)
        sync_old = int((r.get("sync_k") or 0) >= 200_000)
        sync_inv = int((r.get("sync_k") or 0) < 200_000)
        us_s  = int(r.get("session") == "US")
        r["_rev_score"] = n2h_s + sync_inv + us_s  # max 3 (no btc4h/vdepth — not in backfill)
        r["_orig_score_proxy"] = n2h_s + sync_old + us_s

    for thr in [0, 1, 2, 3]:
        sub_rev  = [r["_net"] for r in te if (r.get("_rev_score") or 0) >= thr]
        sub_orig = [r["_net"] for r in te if (r.get("_orig_score_proxy") or 0) >= thr]
        print(f"\n  threshold >= {thr}")
        row(f"  original proxy score >= {thr}", stat(sub_orig))
        row(f"  revised (inv-sync) >= {thr}",   stat(sub_rev))

    hdr("H3 · Best pure n2h-based gate vs original score gate (TIME_EXIT)")
    combos = [
        ("n2h>=3 only (score-agnostic)", lambda r: (r.get("n2h") or 0) >= 3),
        ("n2h>=4 only",                  lambda r: (r.get("n2h") or 0) >= 4),
        ("n2h>=3 AND sync<200K",         lambda r: (r.get("n2h") or 0) >= 3 and (r.get("sync_k") or 0) < 200_000),
        ("n2h>=4 AND sync<200K",         lambda r: (r.get("n2h") or 0) >= 4 and (r.get("sync_k") or 0) < 200_000),
        ("n2h>=3 AND sync<500K",         lambda r: (r.get("n2h") or 0) >= 3 and (r.get("sync_k") or 0) < 500_000),
        ("long_score>=3 (current)",      lambda r: (r.get("long_score") or 0) >= 3),
    ]
    for label, fn in combos:
        sub = [r["_net"] for r in te if fn(r)]
        row(label, stat(sub))


# ── SECTION I: BTC7D REGIME PRECISION ─────────────────────────────────────────

def section_i(longs: list[dict]) -> None:
    te    = [r for r in longs if r.get("close_reason") == "TIME_EXIT"]
    noisy = [r for r in longs if r.get("close_reason") == "NOISY_EARLY_EXIT"]

    print("\n  Loading btc7d for all LONG events...")
    with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=10) as conn:
        for r in longs:
            ts = int(r.get("anchor_ts_ms") or 0)
            r["_btc7d"] = btc_ret(conn, ts, 7*24*3600_000)

    hdr("I1 · BTC 7d trend threshold sweep (TIME_EXIT only)")
    for thr in [-500, -300, -200, -100, -50, 0, 100, 300]:
        sub_pass = [r["_net"] for r in te if r.get("_btc7d") is not None and r["_btc7d"] < thr]
        sub_fail = [r["_net"] for r in te if r.get("_btc7d") is not None and r["_btc7d"] >= thr]
        print(f"\n  btc7d < {thr:+4d} bps:")
        row(f"  PASS (btc7d < {thr})", stat(sub_pass))
        row(f"  FAIL (btc7d >= {thr})", stat(sub_fail))

    hdr("I2 · btc7d distribution: TIME_EXIT vs NOISY vs April")
    te_b7 = [r["_btc7d"] for r in te if r.get("_btc7d") is not None]
    n_b7  = [r["_btc7d"] for r in noisy if r.get("_btc7d") is not None]
    apr_b7 = [r["_btc7d"] for r in longs
              if r["_month"]=="2026-04" and r.get("_btc7d") is not None]
    def _dist(vals):
        if not vals: return "N=0"
        avg = sum(vals)/len(vals)
        pneg = sum(1 for v in vals if v<0)/len(vals)*100
        return f"N={len(vals)}  avg={avg:+.0f} bps  pct<0={pneg:.0f}%"
    print(f"  TIME_EXIT btc7d:  {_dist(te_b7)}")
    print(f"  NOISY btc7d:      {_dist(n_b7)}")
    print(f"  April btc7d:      {_dist(apr_b7)}")

    hdr("I3 · Best btc7d combo for TIME_EXIT")
    combos = [
        ("TIME_EXIT baseline",              lambda r: True),
        ("btc7d < 0",                       lambda r: (r.get("_btc7d") or 1) < 0),
        ("btc7d < 0 AND n2h>=3",            lambda r: (r.get("_btc7d") or 1) < 0 and (r.get("n2h") or 0) >= 3),
        ("btc7d < 0 AND n2h>=4",            lambda r: (r.get("_btc7d") or 1) < 0 and (r.get("n2h") or 0) >= 4),
        ("btc7d < 0 AND sync<200K",         lambda r: (r.get("_btc7d") or 1) < 0 and (r.get("sync_k") or 999999) < 200_000),
        ("btc7d < 0 AND n2h>=3 AND sync<200K",
                                            lambda r: (r.get("_btc7d") or 1)<0 and (r.get("n2h") or 0)>=3 and (r.get("sync_k") or 999999)<200_000),
        ("btc7d < 0 AND n2h>=3 AND !US",    lambda r: (r.get("_btc7d") or 1)<0 and (r.get("n2h") or 0)>=3 and r.get("session")!="US"),
    ]
    for label, fn in combos:
        sub = [r["_net"] for r in te if fn(r)]
        row(label, stat(sub))

    hdr("I4 · April rescue attempt (any combo that makes April positive?)")
    april_te = [r for r in te if r["_month"] == "2026-04"]
    print(f"  April TIME_EXIT total: N={len(april_te)}, avg={sum(r['_net'] for r in april_te)/max(len(april_te),1):+.1f} bps")
    combos_apr = [
        ("April ALL",            lambda r: True),
        ("April btc7d<0",        lambda r: (r.get("_btc7d") or 1) < 0),
        ("April n2h>=4",         lambda r: (r.get("n2h") or 0) >= 4),
        ("April sync<200K",      lambda r: (r.get("sync_k") or 999999) < 200_000),
        ("April !US",            lambda r: r.get("session") != "US"),
        ("April n2h>=4+sync<200K", lambda r: (r.get("n2h") or 0)>=4 and (r.get("sync_k") or 999999)<200_000),
    ]
    for label, fn in combos_apr:
        sub = [r["_net"] for r in april_te if fn(r)]
        row(label, stat(sub))


# ── SECTION J: SHORT HOLD EXTENSION ──────────────────────────────────────────

def section_j(shorts: list[dict]) -> None:
    hdr("J1 · SHORT hold duration: T+2h (current) vs T+3h vs T+4h")
    print(f"  Computing mark prices for {len(shorts)} SHORT entries at T+2h, T+3h, T+4h...")
    nets_2h, nets_3h, nets_4h = [], [], []
    with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=10) as conn:
        for r in shorts:
            ep = float(r.get("entry_price") or 0)
            et = int(r.get("entry_ts_ms") or 0)
            if ep <= 0 or et <= 0:
                continue
            for h, nets in [(2, nets_2h), (3, nets_3h), (4, nets_4h)]:
                px = mark_at(conn, "ETHUSDT", et + h*3600_000)
                if px is None:
                    continue
                raw = -(px - ep) / ep * 10_000.0  # SHORT: positive if price falls
                nets.append(raw - FEE_BPS)
                r[f"_net_{h}h"] = raw - FEE_BPS

    row("SHORT T+2h (current)",  stat(nets_2h))
    row("SHORT T+3h",            stat(nets_3h))
    row("SHORT T+4h",            stat(nets_4h))

    hdr("J2 · SHORT hold extension by score")
    for score_thr in [3, 4]:
        sub = [r for r in shorts if (r.get("score") or 0) >= score_thr]
        print(f"\n  score >= {score_thr}:")
        row(f"  T+2h", stat([r["_net_2h"] for r in sub if "_net_2h" in r]))
        row(f"  T+3h", stat([r["_net_3h"] for r in sub if "_net_3h" in r]))
        row(f"  T+4h", stat([r["_net_4h"] for r in sub if "_net_4h" in r]))

    hdr("J3 · SHORT hold extension by session")
    for sess in ["US", "ASIA"]:
        sub = [r for r in shorts if r.get("session")==sess]
        print(f"\n  session={sess}:")
        row(f"  T+2h", stat([r["_net_2h"] for r in sub if "_net_2h" in r]))
        row(f"  T+3h", stat([r["_net_3h"] for r in sub if "_net_3h" in r]))
        row(f"  T+4h", stat([r["_net_4h"] for r in sub if "_net_4h" in r]))


# ── SECTION K: MON/WED EVALUATION ─────────────────────────────────────────────

def section_k() -> None:
    hdr("K1 · Mon/Wed LONG — would it have worked? (raw DB cascade query)")
    PROP_THRESH = 50_000.0
    ETH_THRESH  = 200_000.0
    BUCKET_SEC  = 300
    GAP_SEC     = 900

    print("  Scanning DB for Mon/Wed ETH SELL >=200K cascades Feb-Jun 2026...")
    FEB_START = 1769904000000
    JUL_START = 1782950400000

    with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=30) as conn:
        # Bucket liquidations by 5min
        rows = conn.execute(
            """
            SELECT (ts_ms / 300000) * 300000 AS bucket, SUM(notional) AS total
            FROM liquidations
            WHERE symbol='ETHUSDT' AND side='SELL'
              AND ts_ms >= ? AND ts_ms < ?
            GROUP BY bucket
            HAVING total >= ?
            ORDER BY bucket ASC
            """,
            (FEB_START, JUL_START, ETH_THRESH)).fetchall()

        # Apply min-gap filter (same as anchor reconstruction)
        anchors = []
        last_ts = 0
        for bucket_ms, total in rows:
            if int(bucket_ms) - last_ts >= GAP_SEC * 1000:
                dt = datetime.fromtimestamp(int(bucket_ms)/1000, tz=timezone.utc)
                dow = dt.weekday()
                hour = dt.hour
                anchors.append({
                    "ts_ms": int(bucket_ms), "notional": float(total),
                    "dow": dow, "hour": hour,
                    "session": "EUROPE" if 7<=hour<13 else ("US" if 13<=hour<21 else "OFF"),
                })
                last_ts = int(bucket_ms)

        # Filter Mon/Wed (would have been excluded)
        mon_wed = [a for a in anchors if a["dow"] in {0, 2}]
        other   = [a for a in anchors if a["dow"] not in {0, 2}]
        print(f"  Total anchors: {len(anchors)}  "
              f"(Mon/Wed: {len(mon_wed)}, other: {len(other)})")

        # For each Mon/Wed anchor: compute LONG T+4h outcome
        def is_bull(conn, ts_ms):
            eth1h = prior_bps_eth(conn, ts_ms, 3600_000)
            b4h = btc_ret(conn, ts_ms, 4*3600_000)
            return (b4h is not None and eth1h > 20.0 and b4h > 50.0)

        def sync_k(conn, ts_ms):
            r = conn.execute(
                "SELECT COALESCE(SUM(notional),0) FROM liquidations "
                "WHERE symbol IN ('BTCUSDT','SOLUSDT') AND side='SELL' "
                "AND ts_ms>=? AND ts_ms<?", (ts_ms - 10*60_000, ts_ms)).fetchone()
            return float(r[0] or 0)

        def n2h_cnt(conn, ts_ms):
            r = conn.execute(
                "SELECT COUNT(*) FROM liquidations "
                "WHERE symbol='ETHUSDT' AND side='SELL' "
                "AND ts_ms>=? AND ts_ms<? AND notional>=?",
                (ts_ms - 2*3600_000, ts_ms - 1000, PROP_THRESH)).fetchone()
            return int(r[0] or 0)

        mon_wed_results, other_check = [], []
        for a in mon_wed:
            ts = a["ts_ms"]
            if is_bull(conn, ts):
                continue
            if a["session"] == "EUROPE":
                continue
            sk = sync_k(conn, ts)
            n2h = n2h_cnt(conn, ts)
            long_score = 1 + int(n2h>=3) + int(btc_ret(conn,ts,4*3600_000) or 0 < 0) + int(sk>=200_000)
            if long_score < 3:
                continue
            px_entry = mark_after(conn, "ETHUSDT", ts)
            px_exit  = mark_at(conn, "ETHUSDT", ts + HORIZON_LONG_MS)
            if px_entry and px_exit and px_entry > 0:
                raw = (px_exit - px_entry) / px_entry * 10_000.0
                mon_wed_results.append(raw - FEE_BPS)

        # Same for Tue-Fri-Sat-Sun (all non-excluded days) for comparison
        for a in other:
            ts = a["ts_ms"]
            if is_bull(conn, ts): continue
            if a["session"] == "EUROPE": continue
            if a["dow"] == 6: continue  # Sunday only has LONG
            sk = sync_k(conn, ts)
            n2h = n2h_cnt(conn, ts)
            long_score = 1 + int(n2h>=3) + int(btc_ret(conn,ts,4*3600_000) or 0 < 0) + int(sk>=200_000)
            if long_score < 3: continue
            px_entry = mark_after(conn, "ETHUSDT", ts)
            px_exit  = mark_at(conn, "ETHUSDT", ts + HORIZON_LONG_MS)
            if px_entry and px_exit and px_entry > 0:
                raw = (px_exit - px_entry) / px_entry * 10_000.0
                other_check.append(raw - FEE_BPS)

    row("Mon/Wed LONG T+4h (naive)", stat(mon_wed_results))
    row("Other days LONG T+4h (naive)", stat(other_check))
    print("\n  NOTE: naive = no vdepth or silence filter — directional comparison only")


# ── SECTION L: SYNC_K SCORE REVISION ─────────────────────────────────────────

def section_l(longs: list[dict]) -> None:
    te = [r for r in longs if r.get("close_reason") == "TIME_EXIT"]

    hdr("L1 · Sync_k exclusion zones (TIME_EXIT only)")
    # Which sync_k range is the real problem?
    buckets = [
        (0,    50,   "sync 0-50K"),
        (50,   100,  "sync 50-100K"),
        (100,  200,  "sync 100-200K"),
        (200,  300,  "sync 200-300K"),
        (300,  500,  "sync 300-500K"),
        (500,  1000, "sync 500K-1M"),
        (1000, 2000, "sync 1M-2M"),
        (2000, 9999, "sync >=2M"),
    ]
    for lo, hi, label in buckets:
        sub = [r["_net"] for r in te if lo*1000 <= (r.get("sync_k") or 0) < hi*1000]
        row(label, stat(sub))

    hdr("L2 · Revised eligibility: remove sync from score, adjust threshold")
    # Current: long_score = score+1 >= 3, score includes sync>=200K
    # Revised: score_no_sync = n2h>=3 + btc4h_proxy + US_sess (max 3 computable from ledger)
    # Use: n2h>=3 + US_sess as proxy (2 components) + silence (+1) = long_score_proxy
    # Events that are currently in backfill won't change since gate already passed
    # But we can see: of those that passed, which ones would pass a revised 2-component threshold?

    for r in te:
        n2h_c = int((r.get("n2h") or 0) >= 3)
        us_c  = int(r.get("session") == "US")
        sync_c = int((r.get("sync_k") or 0) >= 200_000)
        # original long_score = some_base + sync + n2h + us + btc4h + vdepth + 1
        # proxy without sync: n2h + us + 1 (silence)
        r["_score_no_sync_proxy"] = n2h_c + us_c + 1  # max 3

    # How does population shift?
    for thr in [1, 2, 3]:
        with_sync    = [r["_net"] for r in te if (r.get("long_score") or 0) >= thr+1]
        without_sync = [r["_net"] for r in te if (r.get("_score_no_sync_proxy") or 0) >= thr]
        print(f"\n  threshold analogue = {thr}:")
        row(f"  current long_score>={thr+1}", stat(with_sync))
        row(f"  no-sync proxy>={thr}",        stat(without_sync))


# ── SECTION M: SIL_LO RESCUE T+4h ────────────────────────────────────────────

def section_m(longs: list[dict]) -> None:
    hdr("M1 · SIL_LO=3min rescue: T+4h outcomes for reclassified events")
    noisy = [r for r in longs if r.get("close_reason") == "NOISY_EARLY_EXIT"]
    print(f"  Re-identifying 18 rescued events from {len(noisy)} NOISY records...")

    rescued, still_noisy = [], []
    with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=10) as conn:
        for r in noisy:
            ts = int(r.get("anchor_ts_ms") or 0)
            early = eth_sell_first(conn, ts, SIL_LO_MS, 180_000, 50_000.0)
            if early is None:
                still_noisy.append(r)
                continue
            # Triggered in [60s, 180s]. Is [180s, 30min] also noisy?
            late = eth_sell_first(conn, ts, 180_000, SIL_HI_MS, 50_000.0)
            if late is not None:
                still_noisy.append(r)
                continue
            # This event would be rescued by SIL_LO=3min
            ep = float(r.get("entry_price") or 0)
            px4h = mark_at(conn, "ETHUSDT", ts + HORIZON_LONG_MS)
            if ep > 0 and px4h is not None:
                r["_rescued_net"] = (px4h - ep) / ep * 10_000.0 - FEE_BPS
                rescued.append(r)

    print(f"  Rescued (SIL_LO=3min): N={len(rescued)}, still noisy: N={len(still_noisy)}")
    row("Rescued events: actual early exit",  stat([r["_net"] for r in rescued]))
    row("Rescued events: hypo T+4h hold",     stat([r["_rescued_net"] for r in rescued if "_rescued_net" in r]))
    rescued_hypo = [r["_rescued_net"] for r in rescued if "_rescued_net" in r]
    if rescued_hypo:
        delta = round(sum(rescued_hypo)/len(rescued_hypo) - sum(r["_net"] for r in rescued)/max(len(rescued),1), 1)
        print(f"  Delta per rescued event: {delta:+.1f} bps")
    row("Still noisy: actual",  stat([r["_net"] for r in still_noisy]))


# ── SECTION N: FEATURE INTERACTION MATRIX ─────────────────────────────────────

def section_n(longs: list[dict]) -> None:
    te = [r for r in longs if r.get("close_reason") == "TIME_EXIT"]

    hdr("N1 · Feature interaction matrix (TIME_EXIT — all 2-feature combos)")
    features = {
        "n2h>=3":   lambda r: (r.get("n2h") or 0) >= 3,
        "n2h>=4":   lambda r: (r.get("n2h") or 0) >= 4,
        "sync<200K":lambda r: (r.get("sync_k") or 999999) < 200_000,
        "sync<500K":lambda r: (r.get("sync_k") or 999999) < 500_000,
        "!US":      lambda r: r.get("session") != "US",
        "ASIA+OFF": lambda r: r.get("session") in {"ASIA","OFF"},
        "btc7d<0":  lambda r: (r.get("_btc7d") or 1) < 0,
        "Fri":      lambda r: r.get("dow") == 4,
        "!Sat":     lambda r: r.get("dow") != 5,
    }
    keys = list(features.keys())
    single = {}
    for k, fn in features.items():
        sub = [r["_net"] for r in te if fn(r)]
        single[k] = stat(sub)
        row(f"  {k}", stat(sub))

    print("\n  --- 2-feature combos (showing improvements over baseline) ---")
    baseline = stat([r["_net"] for r in te])
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            ka, kb = keys[i], keys[j]
            fa, fb = features[ka], features[kb]
            sub = [r["_net"] for r in te if fa(r) and fb(r)]
            s = stat(sub)
            if s["n"] >= 8 and s["avg"] is not None and s["avg"] > 60:
                print(f"  ** {ka} AND {kb}")
                row(f"     combo", s)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\nLoading backfill ledger...")
    longs, shorts = load_records()
    te = [r for r in longs if r.get("close_reason")=="TIME_EXIT"]
    print(f"LONG: {len(longs)} ({len(te)} TIME_EXIT, {len(longs)-len(te)} NOISY)")
    print(f"SHORT: {len(shorts)}")

    section_g(longs)
    section_h(longs)
    section_i(longs)
    section_j(shorts)
    section_k()
    section_l(longs)
    section_m(longs)
    section_n(longs)

    print()
    print("=" * 66)
    print("  ALL ROUND-3 TESTS COMPLETE")
    print("=" * 66)

if __name__ == "__main__":
    main()
