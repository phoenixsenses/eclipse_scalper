"""
research_s34_silence_edge.py
6-test silence-vs-noisy edge research suite.

TEST-1: Feature separability — session / score / n2h / sync_k at T=0
TEST-2: NOISY_EARLY_EXIT → SHORT conversion (noisy_ts entry, 2h hold)
TEST-3: Score gate tightening (long_score >= 3/4/5)
TEST-4: Entry delay (T+5min / T+10min / T+15min vs T=0)
TEST-5: Session-based silence rate breakdown
TEST-6: BTC 5-min bps at cascade as silence predictor
"""
from __future__ import annotations

import json
import sqlite3
import statistics
from pathlib import Path
from typing import Any

ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "microstructure.db"
LEDGER  = ROOT / "reports" / "shadow" / "s34_state_machine_shadow.jsonl"

FEE_BPS         = 5.0
HORIZON_SHORT_MS = 2 * 3600_000


# ── helpers ──────────────────────────────────────────────────────────────────

def load_closes() -> list[dict]:
    rows = []
    with LEDGER.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("event") == "CLOSE":
                rows.append(obj)
    return rows


def mark_at(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> float | None:
    row = conn.execute(
        "SELECT mark_price FROM mark_prices "
        "WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (symbol, ts_ms),
    ).fetchone()
    return float(row[0]) if row else None


def mark_after(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> float | None:
    row = conn.execute(
        "SELECT mark_price FROM mark_prices "
        "WHERE symbol=? AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1",
        (symbol, ts_ms),
    ).fetchone()
    return float(row[0]) if row else None


def stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    n    = len(vals)
    wins = sum(1 for v in vals if v > 0)
    avg  = sum(vals) / n
    return {
        "n":     n,
        "wr":    round(wins / n * 100, 1),
        "avg":   round(avg, 1),
        "total": round(sum(vals), 1),
        "med":   round(statistics.median(vals), 1),
    }


def pprint(label: str, d: dict) -> None:
    n = d.get("n", 0)
    if n == 0:
        print(f"  {label:40s}  N=0")
        return
    print(
        f"  {label:40s}  N={n:4d}  WR={d['wr']:5.1f}%  "
        f"avg={d['avg']:+7.1f}  total={d['total']:+8.1f}  med={d['med']:+7.1f}  bps"
    )


# ── TEST-1 / TEST-3 / TEST-5  (ledger only) ──────────────────────────────────

def run_ledger_tests(closes: list[dict]) -> None:
    ls = [c for c in closes if c.get("signal") == "LONG_SILENCE"]

    silence = [c for c in ls if c.get("close_reason") == "TIME_EXIT"]
    noisy   = [c for c in ls if c.get("close_reason") == "NOISY_EARLY_EXIT"]

    # ── TEST-5: Session silence rate ─────────────────────────────────────────
    print("\n" + "═"*70)
    print("TEST-5 · Session silence rate")
    print("═"*70)
    sessions = sorted(set(c.get("session","?") for c in ls))
    for sess in sessions:
        sub  = [c for c in ls if c.get("session") == sess]
        sil  = sum(1 for c in sub if c.get("close_reason") == "TIME_EXIT")
        rate = sil / len(sub) * 100 if sub else 0
        vals = [float(c["net_bps"]) for c in sub if c.get("net_bps") is not None]
        pprint(f"LONG_SILENCE session={sess} (silence_rate={rate:.0f}%)", stats(vals))

    # ── TEST-1: Feature separability ─────────────────────────────────────────
    print("\n" + "═"*70)
    print("TEST-1 · Feature separability (SILENCE vs NOISY)")
    print("═"*70)

    def bucket_score(cs: list[dict], label: str) -> None:
        print(f"\n  [{label}]  n_silence={len([c for c in cs if c.get('close_reason')=='TIME_EXIT'])}  "
              f"n_noisy={len([c for c in cs if c.get('close_reason')=='NOISY_EARLY_EXIT'])}  "
              f"silence_rate={len([c for c in cs if c.get('close_reason')=='TIME_EXIT'])/len(cs)*100:.0f}%  "
              f"  avg_net={sum(float(c['net_bps']) for c in cs if c.get('net_bps') is not None)/max(len(cs),1):.1f} bps"
              if cs else f"\n  [{label}]  N=0")

    # By long_score (base_score+1 — the silence-gate value)
    print("\n  — by long_score (silence gate threshold) —")
    for sc in sorted(set(c.get("long_score", c.get("score")) for c in ls)):
        sub = [c for c in ls if (c.get("long_score") or c.get("score")) == sc]
        bucket_score(sub, f"long_score={sc}")

    # By session split: silence vs noisy avg
    print("\n  — silence rate and avg net by session —")
    for sess in sessions:
        sub = [c for c in ls if c.get("session") == sess]
        if not sub:
            continue
        sil_sub   = [c for c in sub if c.get("close_reason") == "TIME_EXIT"]
        noisy_sub = [c for c in sub if c.get("close_reason") == "NOISY_EARLY_EXIT"]
        sil_rate  = len(sil_sub) / len(sub) * 100
        sil_avg   = sum(float(c["net_bps"]) for c in sil_sub if c.get("net_bps") is not None) / max(len(sil_sub), 1)
        noisy_avg = sum(float(c["net_bps"]) for c in noisy_sub if c.get("net_bps") is not None) / max(len(noisy_sub), 1)
        print(f"  session={sess:6s}  N={len(sub):3d}  silence_rate={sil_rate:4.0f}%  "
              f"silence_avg={sil_avg:+6.1f} bps  noisy_avg={noisy_avg:+6.1f} bps")

    # By n2h buckets
    print("\n  — by n2h (2h cascade count) —")
    for lo, hi in [(0,2),(3,5),(6,10),(11,999)]:
        sub = [c for c in ls if lo <= (c.get("n2h") or 0) <= hi]
        if not sub:
            continue
        bucket_score(sub, f"n2h={lo}-{hi}")

    # By sync_k buckets
    print("\n  — by sync_k (cascade notional) —")
    for lo, hi in [(0,300_000),(300_000,600_000),(600_000,1_500_000),(1_500_000,999_999_999)]:
        sub = [c for c in ls if lo <= (c.get("sync_k") or 0) < hi]
        if not sub:
            continue
        bucket_score(sub, f"sync_k={lo//1000}K-{hi//1000 if hi<999_999_999 else '∞'}K")

    # By DOW
    print("\n  — by day of week —")
    day_names = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    for dow in sorted(set(c.get("dow",0) for c in ls)):
        sub = [c for c in ls if c.get("dow") == dow]
        if not sub:
            continue
        bucket_score(sub, f"DOW={day_names.get(dow,dow)}")

    # ── TEST-3: Score gate ────────────────────────────────────────────────────
    print("\n" + "═"*70)
    print("TEST-3 · Score gate tightening")
    print("═"*70)
    for min_score in [3, 4, 5]:
        sub = [c for c in ls if (c.get("long_score") or c.get("score", 0)) >= min_score]
        vals = [float(c["net_bps"]) for c in sub if c.get("net_bps") is not None]
        sil_n = sum(1 for c in sub if c.get("close_reason") == "TIME_EXIT")
        sil_rate = sil_n / len(sub) * 100 if sub else 0
        pprint(f"long_score >= {min_score}  (silence_rate={sil_rate:.0f}%)", stats(vals))

    # Best combo: score + session
    print("\n  — score × session combinations —")
    for min_score in [3, 4]:
        for sess in sessions:
            sub = [c for c in ls
                   if (c.get("long_score") or c.get("score", 0)) >= min_score
                   and c.get("session") == sess]
            if len(sub) < 4:
                continue
            vals = [float(c["net_bps"]) for c in sub if c.get("net_bps") is not None]
            sil_n = sum(1 for c in sub if c.get("close_reason") == "TIME_EXIT")
            sil_rate = sil_n / len(sub) * 100 if sub else 0
            pprint(f"score>={min_score} + {sess:6s}  (sil={sil_rate:.0f}%)", stats(vals))


# ── TEST-2: NOISY → SHORT ─────────────────────────────────────────────────────

def run_noisy_short(closes: list[dict], conn: sqlite3.Connection) -> None:
    print("\n" + "═"*70)
    print("TEST-2 · NOISY_EARLY_EXIT → SHORT conversion (2h hold)")
    print("═"*70)

    noisy = [c for c in closes
             if c.get("signal") == "LONG_SILENCE"
             and c.get("close_reason") == "NOISY_EARLY_EXIT"]

    results: list[dict] = []
    skipped = 0
    for c in noisy:
        entry_ts  = int(c.get("exit_ts_ms", 0))
        entry_px  = c.get("exit_price")
        exit_ts   = entry_ts + HORIZON_SHORT_MS
        if not entry_px or not entry_ts:
            skipped += 1
            continue
        exit_px = mark_at(conn, "ETHUSDT", exit_ts)
        if exit_px is None:
            exit_px = mark_after(conn, "ETHUSDT", exit_ts)
        if exit_px is None:
            skipped += 1
            continue
        # SHORT: profit when price falls
        outcome = (float(entry_px) - exit_px) / float(entry_px) * 10_000
        net     = outcome - FEE_BPS
        results.append({
            "net_bps":  net,
            "session":  c.get("session"),
            "long_score": c.get("long_score") or c.get("score"),
            "anchor_ts_ms": c.get("anchor_ts_ms"),
        })

    print(f"  (skipped {skipped}/{len(noisy)} due to missing price data)")
    vals = [r["net_bps"] for r in results]
    pprint("NOISY → SHORT (all)", stats(vals))

    # By session
    for sess in sorted(set(r.get("session","?") for r in results)):
        sub = [r["net_bps"] for r in results if r.get("session") == sess]
        pprint(f"NOISY → SHORT  session={sess}", stats(sub))

    # By long_score
    for sc in sorted(set(r.get("long_score") for r in results if r.get("long_score"))):
        sub = [r["net_bps"] for r in results if r.get("long_score") == sc]
        pprint(f"NOISY → SHORT  long_score={sc}", stats(sub))

    # Combined: what if we do BOTH original LONG_SILENCE + NOISY SHORT?
    ls_vals = [float(c["net_bps"]) for c in closes
               if c.get("signal") == "LONG_SILENCE" and c.get("net_bps") is not None]
    combined = ls_vals + vals
    n_orig   = len(ls_vals)
    print(f"\n  Combined strategy (LONG_SILENCE + NOISY→SHORT harvested):")
    pprint(f"  LONG_SILENCE only (N={n_orig})", stats(ls_vals))
    pprint(f"  + NOISY→SHORT (N={len(vals)})", stats(combined))


# ── TEST-4: Entry delay ────────────────────────────────────────────────────────

def run_entry_delay(closes: list[dict], conn: sqlite3.Connection) -> None:
    print("\n" + "═"*70)
    print("TEST-4 · Entry delay (T+5 / T+10 / T+15 min vs T=0)")
    print("═"*70)

    ls = [c for c in closes if c.get("signal") == "LONG_SILENCE"]

    for delay_min in [0, 5, 10, 15]:
        delay_ms = delay_min * 60_000
        vals: list[float] = []
        skipped = 0
        for c in ls:
            anchor  = int(c.get("anchor_ts_ms", 0))
            exit_px = c.get("exit_price")
            exit_ts = int(c.get("exit_ts_ms", 0))
            if not exit_px or not exit_ts:
                skipped += 1
                continue
            if delay_ms == 0:
                entry_px = c.get("entry_price")
            else:
                entry_ts_delayed = anchor + delay_ms
                # If delayed entry is past the actual exit, skip
                if entry_ts_delayed >= exit_ts:
                    skipped += 1
                    continue
                entry_px = mark_at(conn, "ETHUSDT", entry_ts_delayed)
                if entry_px is None:
                    skipped += 1
                    continue
            if not entry_px:
                skipped += 1
                continue
            outcome = (float(exit_px) - float(entry_px)) / float(entry_px) * 10_000
            net     = outcome - FEE_BPS
            vals.append(net)
        label = f"T+{delay_min}min entry  (skipped={skipped})"
        pprint(label, stats(vals))

    # For SILENCE_CONFIRMED only (TIME_EXIT) to see if delay helps/hurts on winners
    print("\n  — SILENCE_CONFIRMED trades only (TIME_EXIT) —")
    silence_ls = [c for c in ls if c.get("close_reason") == "TIME_EXIT"]
    for delay_min in [0, 5, 10, 15, 30]:
        delay_ms = delay_min * 60_000
        vals: list[float] = []
        skipped = 0
        for c in silence_ls:
            anchor  = int(c.get("anchor_ts_ms", 0))
            exit_px = c.get("exit_price")
            exit_ts = int(c.get("exit_ts_ms", 0))
            if not exit_px or not exit_ts:
                skipped += 1
                continue
            if delay_ms == 0:
                entry_px = c.get("entry_price")
            else:
                entry_ts_delayed = anchor + delay_ms
                if entry_ts_delayed >= exit_ts:
                    skipped += 1
                    continue
                entry_px = mark_at(conn, "ETHUSDT", entry_ts_delayed)
                if entry_px is None:
                    skipped += 1
                    continue
            if not entry_px:
                skipped += 1
                continue
            outcome = (float(exit_px) - float(entry_px)) / float(entry_px) * 10_000
            net     = outcome - FEE_BPS
            vals.append(net)
        label = f"SILENCE T+{delay_min}min  (skipped={skipped})"
        pprint(label, stats(vals))


# ── TEST-6: BTC 5-min bps at cascade ─────────────────────────────────────────

def run_btc_predictor(closes: list[dict], conn: sqlite3.Connection) -> None:
    print("\n" + "═"*70)
    print("TEST-6 · BTC 5-min bps at cascade as silence predictor")
    print("═"*70)

    ls = [c for c in closes if c.get("signal") == "LONG_SILENCE"]

    btc_groups: dict[str, list] = {"BTC_UP": [], "BTC_DOWN": [], "BTC_FLAT": []}
    skipped = 0
    annotated: list[dict] = []
    for c in ls:
        anchor = int(c.get("anchor_ts_ms", 0))
        btc_now  = mark_at(conn, "BTCUSDT", anchor)
        btc_prev = mark_at(conn, "BTCUSDT", anchor - 5 * 60_000)
        if not btc_now or not btc_prev:
            skipped += 1
            continue
        btc_bps = (btc_now - btc_prev) / btc_prev * 10_000
        if btc_bps >= 10:
            group = "BTC_UP"
        elif btc_bps <= -10:
            group = "BTC_DOWN"
        else:
            group = "BTC_FLAT"
        annotated.append({**c, "_btc_5m_bps": btc_bps, "_btc_group": group})

    print(f"  (skipped {skipped}/{len(ls)} due to missing BTC price)")

    for group in ["BTC_UP", "BTC_FLAT", "BTC_DOWN"]:
        sub = [c for c in annotated if c.get("_btc_group") == group]
        if not sub:
            continue
        vals = [float(c["net_bps"]) for c in sub if c.get("net_bps") is not None]
        sil_n    = sum(1 for c in sub if c.get("close_reason") == "TIME_EXIT")
        sil_rate = sil_n / len(sub) * 100 if sub else 0
        pprint(f"{group:12s}  (silence_rate={sil_rate:.0f}%)", stats(vals))

    # More granular: BTC bps buckets
    print("\n  — BTC 5-min bps buckets —")
    buckets = [(-9999,-50,"<-50"),(-50,-10,"-50..-10"),(-10,10,"-10..+10"),(10,50,"+10..+50"),(50,9999,">+50")]
    for lo, hi, label in buckets:
        sub = [c for c in annotated if lo <= c.get("_btc_5m_bps", 0) < hi]
        if not sub:
            continue
        vals = [float(c["net_bps"]) for c in sub if c.get("net_bps") is not None]
        sil_n    = sum(1 for c in sub if c.get("close_reason") == "TIME_EXIT")
        sil_rate = sil_n / len(sub) * 100 if sub else 0
        pprint(f"btc_5m {label:12s}  (sil={sil_rate:.0f}%)", stats(vals))

    # BTC_DOWN + score/session cross-cut
    print("\n  — BTC_DOWN subgroups (most interesting for cascade LONG) —")
    btc_down = [c for c in annotated if c.get("_btc_group") == "BTC_DOWN"]
    sessions = sorted(set(c.get("session","?") for c in btc_down))
    for sess in sessions:
        sub = [c for c in btc_down if c.get("session") == sess]
        if not sub:
            continue
        vals = [float(c["net_bps"]) for c in sub if c.get("net_bps") is not None]
        sil_n    = sum(1 for c in sub if c.get("close_reason") == "TIME_EXIT")
        sil_rate = sil_n / len(sub) * 100 if sub else 0
        pprint(f"BTC_DOWN + {sess:6s}  (sil={sil_rate:.0f}%)", stats(vals))


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("S34 Silence Edge Research — 6 tests")
    print(f"Ledger: {LEDGER}")
    print(f"DB:     {DB_PATH}")

    closes = load_closes()
    ls = [c for c in closes if c.get("signal") == "LONG_SILENCE"]
    sn = [c for c in closes if c.get("signal") == "SHORT_NEITHER"]
    print(f"\nLoaded: LONG_SILENCE={len(ls)}  SHORT_NEITHER={len(sn)}  total_closes={len(closes)}")

    # Ledger-only tests (fast)
    run_ledger_tests(closes)

    # DB-dependent tests
    print("\nConnecting to DB for price queries...")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        run_noisy_short(closes, conn)
        run_entry_delay(closes, conn)
        run_btc_predictor(closes, conn)
    finally:
        conn.close()

    print("\n" + "═"*70)
    print("Done.")


if __name__ == "__main__":
    main()
