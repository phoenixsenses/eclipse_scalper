"""
S34 Forward Validation — Bucket Analysis
61 pre-registration trades, cross-referenced with detector_signals.
Metric: outcome_15m_return (bps proxy) + outcome_15m_win (binary).
"""
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MICRO_DB = ROOT / "data" / "microstructure.db"
PAPER_DB = ROOT / "data" / "paper_trades.db"


def bucket_table(label, rows, key_fn, min_n=2):
    buckets = defaultdict(list)
    for r in rows:
        buckets[key_fn(r)].append(r["ret"])

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  {'Bucket':<26} {'N':>4}  {'WinR':>6}  {'AvgRet':>8}  {'TotRet':>9}  Flag")
    print(f"  {'-'*26}  {'-'*4}  {'-'*6}  {'-'*8}  {'-'*9}  ----")

    order = sorted(buckets.keys(), key=lambda x: (x is None, str(x)))
    for bkt in order:
        rets = buckets[bkt]
        if len(rets) < min_n:
            continue
        wins = sum(1 for r in rets if r > 0)
        winr = wins / len(rets) * 100
        avg = sum(rets) / len(rets)
        tot = sum(rets)
        flag = "<<< KILL" if avg < -0.05 else ("WEAK" if avg < 0 else ("OK" if avg < 0.10 else "GOOD"))
        print(f"  {str(bkt):<26}  {len(rets):>4}  {winr:>5.1f}%  {avg:>+8.4f}  {tot:>+9.4f}  {flag}")

    all_rets = [r for rets in buckets.values() for r in rets]
    wins = sum(1 for r in all_rets if r > 0)
    avg = sum(all_rets) / len(all_rets) if all_rets else 0
    print(f"  {'TOPLAM':<26}  {len(all_rets):>4}  {wins/len(all_rets)*100:>5.1f}%  {avg:>+8.4f}  {sum(all_rets):>+9.4f}")


def score_quartile(val, vals):
    if val is None:
        return "None"
    sorted_vals = sorted(v for v in vals if v is not None)
    n = len(sorted_vals)
    if n == 0:
        return "None"
    p25 = sorted_vals[n // 4]
    p50 = sorted_vals[n // 2]
    p75 = sorted_vals[3 * n // 4]
    if val <= p25:
        return f"Q1 (lo={p25:.1f})"
    elif val <= p50:
        return f"Q2 ({p25:.1f}-{p50:.1f})"
    elif val <= p75:
        return f"Q3 ({p50:.1f}-{p75:.1f})"
    else:
        return f"Q4 (hi={p75:.1f})"


def main():
    micro = sqlite3.connect(str(MICRO_DB))
    micro.row_factory = sqlite3.Row
    paper = sqlite3.connect(str(PAPER_DB))
    paper.row_factory = sqlite3.Row

    fv = paper.execute("""
        SELECT signal_ts_ms, spread_ratio, eclipse_score, regime_at_entry,
               session_bucket, purity_score, outcome_15m_return, outcome_15m_win,
               group_label, entry_tier
        FROM forward_validation ORDER BY signal_ts_ms
    """).fetchall()

    sigs = micro.execute("""
        SELECT signal_ts_ms, confidence_band, entry_tier as sig_tier,
               build_bucket, signal_type, eth_vol_regime,
               liq_composition, geometry_type, fingerprint_class,
               purity_cohort, entropy_cohort, fragility_zone,
               first_of_day_cascade, btc_lead_before_cascade,
               vol_decile_at_entry, cascade_rise_time_sec
        FROM detector_signals ORDER BY signal_ts_ms
    """).fetchall()
    sig_map = {s["signal_ts_ms"]: dict(s) for s in sigs}

    # Build enriched rows
    rows = []
    for r in fv:
        sig = sig_map.get(r["signal_ts_ms"], {})
        rows.append({
            "ret": r["outcome_15m_return"],
            "win": r["outcome_15m_win"],
            "regime": r["regime_at_entry"],
            "session": r["session_bucket"],
            "eclipse": r["eclipse_score"],
            "spread_ratio": r["spread_ratio"],
            "group": r["group_label"],
            "tier": r["entry_tier"],
            "conf": sig.get("confidence_band") if sig else None,
            "build": sig.get("build_bucket") if sig else None,
            "sig_type": sig.get("signal_type") if sig else None,
            "vol_regime": sig.get("eth_vol_regime") if sig else None,
            "liq_comp": sig.get("liq_composition") if sig else None,
            "geometry": sig.get("geometry_type") if sig else None,
            "fingerprint": sig.get("fingerprint_class") if sig else None,
            "purity": sig.get("purity_cohort") if sig else None,
            "entropy": sig.get("entropy_cohort") if sig else None,
            "frag_zone": sig.get("fragility_zone") if sig else None,
            "first_of_day": sig.get("first_of_day_cascade") if sig else None,
            "btc_led": sig.get("btc_lead_before_cascade") if sig else None,
            "vol_decile": sig.get("vol_decile_at_entry") if sig else None,
            "rise_time": sig.get("cascade_rise_time_sec") if sig else None,
        })

    all_eclipse = [r["eclipse"] for r in rows if r["eclipse"] is not None]
    all_spread = [r["spread_ratio"] for r in rows if r["spread_ratio"] is not None]
    all_rets = [r["ret"] for r in rows if r["ret"] is not None]

    print(f"\n{'#'*65}")
    print(f"  S34 FORWARD VALIDATION — BUCKET ANALİZİ")
    print(f"  N={len(rows)} trade | Metrik: 15m return")
    print(f"  Genel WinR: {sum(r['win'] for r in rows)/len(rows)*100:.1f}%")
    print(f"  Ortalama ret: {sum(all_rets)/len(all_rets):+.4f}")
    print(f"{'#'*65}")

    # 1. Entry tier
    bucket_table("1. ENTRY TIER", rows, lambda r: r["tier"])

    # 2. Confidence band
    bucket_table("2. CONFIDENCE BAND", rows, lambda r: r["conf"])

    # 3. Build bucket
    bucket_table("3. BUILD BUCKET", rows, lambda r: r["build"])

    # 4. Regime
    bucket_table("4. REGIME AT ENTRY", rows, lambda r: r["regime"])

    # 5. Session bucket
    bucket_table("5. SESSION (UTC)", rows, lambda r: r["session"])

    # 6. Signal type
    bucket_table("6. SIGNAL TYPE", rows, lambda r: r["sig_type"])

    # 7. Eclipse score quartiles
    bucket_table("7. ECLIPSE SCORE (quartile)",
                 rows, lambda r: score_quartile(r["eclipse"], all_eclipse))

    # 8. Spread ratio quartiles
    bucket_table("8. SPREAD RATIO (quartile)",
                 rows, lambda r: score_quartile(r["spread_ratio"], all_spread))

    # 9. Vol regime
    bucket_table("9. ETH VOL REGIME", rows, lambda r: r["vol_regime"])

    # 10. Purity cohort
    bucket_table("10. PURITY COHORT", rows, lambda r: r["purity"])

    # 11. First of day cascade
    bucket_table("11. FIRST-OF-DAY CASCADE", rows, lambda r: bool(r["first_of_day"]))

    # 12. BTC-led
    bucket_table("12. BTC LED", rows, lambda r: bool(r["btc_led"]))

    # 13. Vol decile
    bucket_table("13. VOL DECILE AT ENTRY", rows,
                 lambda r: f"D{r['vol_decile']}" if r["vol_decile"] is not None else "None", min_n=1)

    # 14. Group label
    bucket_table("14. GROUP LABEL", rows, lambda r: r["group"])

    # 15. Fragility zone
    bucket_table("15. FRAGILITY ZONE", rows, lambda r: r["frag_zone"])

    # --- Best/worst combo ---
    print(f"\n{'='*65}")
    print("  TOP KOMBINASYONLAR (N>=3)")
    print(f"{'='*65}")
    combos = defaultdict(list)
    for r in rows:
        key = (r["conf"], r["build"], r["tier"])
        combos[key].append(r["ret"])

    combo_stats = []
    for key, rets in combos.items():
        if len(rets) < 3:
            continue
        avg = sum(rets) / len(rets)
        wins = sum(1 for r in rets if r > 0)
        combo_stats.append((avg, len(rets), wins, key))

    print(f"  {'conf/build/tier':<30} {'N':>4}  {'WinR':>6}  {'AvgRet':>8}")
    for avg, n, wins, key in sorted(combo_stats, reverse=True):
        print(f"  {str(key):<30}  {n:>4}  {wins/n*100:>5.1f}%  {avg:>+8.4f}")

    micro.close()
    paper.close()


if __name__ == "__main__":
    main()
