"""S34 Pre-Cascade Dip-Recovery Pattern — "liq'ten önce 2-3 kere düşüş+çıkış" hipotezi.

Hipotez (operatör, 2026-07-06): Likidasyon cascade anchor'ından ÖNCE fiyatta
2-3 kez tamamlanmış düşüş→toparlanma (dip→recovery) döngüsü olması gerekir;
bu ön-desen cascade sonrası LONG reversion sonucunu ayrıştırır.

Tasarım (point-in-time, lookahead YOK — tüm feature'lar T0 öncesi):
  - Universe: ETH SELL 200K anchors (bucket=300s, min_gap=900s) — core alfa evreni.
  - Feature: T0'dan geriye N bar'lık pencerede zigzag ile sayılan TAMAMLANMIŞ
    dip→recovery döngüsü. Dip = son tepeden >= amp_bps düşüş; recovery = dip
    dibinden düşüşün >= %60'ını geri alma. T0'a giren tamamlanmamış son düşüş
    (cascade bacağının kendisi) SAYILMAZ.
  - Grid ("time framelerde"): bar TF {1m, 3m, 5m, 15m} × lookback {kısa, uzun}
    × amp {10, 20, 35, 60 bps} — hepsi TEK proseste SERİ taranır.
  - Sonuç: LONG hold 1h/4h/6h gross (ana=4h), net = gross - 5bps.
  - Protokol: kronolojik %60/%40 TRAIN/TEST; config seçimi TRAIN'de
    (2-3 bucket avg - rest avg maksimize, bucket N>=15); TEST'te rapor +
    label-shuffle permütasyon p (2000 iter) + sign-flip mc_p.
  - Sekonder: core gate (not bull, not EU, silence) altında aynı okuma.

Çıktı:
  reports/research/s34/S34_PRE_CASCADE_DIP_RECOVERY.json
  reports/research/s34/S34_PRE_CASCADE_DIP_RECOVERY.md
"""
from __future__ import annotations

import json
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    load_liquidations,
    load_mark_index,
    reconstruct_anchors,
)

DB_PATH  = ROOT / "data" / "microstructure.db"
OUT_DIR  = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_PRE_CASCADE_DIP_RECOVERY.json"
OUT_MD   = OUT_DIR / "S34_PRE_CASCADE_DIP_RECOVERY.md"

ETH_THRESH   = 200_000.0
PROP_THRESH  =  50_000.0
LOOKBACK_MS  = 400 * 24 * 3600_000
FEE_BPS      = 5.0
MC_ITER      = 2000
RETRACE_FRAC = 0.6
TRAIN_FRAC   = 0.6
MIN_BUCKET_N = 15

# TF (dakika) → bar sayısı listesi (kısa/uzun lookback)
TF_GRID = {
    1:  (60, 120),   # 1h / 2h
    3:  (40, 80),    # 2h / 4h
    5:  (48, 96),    # 4h / 8h
    15: (32, 64),    # 8h / 16h
}
AMP_GRID = (10.0, 20.0, 35.0, 60.0)
HOLDS_H  = (1, 4, 6)
PRIMARY_HOLD = 4

random.seed(42)

# ---------------------------------------------------------------------------
# Helpers (alpha_attribution konvansiyonları)
# ---------------------------------------------------------------------------

def long_gross(marks, ts, hold_ms):
    r0 = marks.at_or_after(ts)
    r1 = marks.at_or_before(ts + hold_ms)
    if not r0 or not r1 or float(r0[1]) <= 0:
        return None
    return (float(r1[1]) - float(r0[1])) / float(r0[1]) * 10_000.0

def mark_bps(conn, sym, ts_ms, lookback_ms):
    r0 = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? "
        "ORDER BY ts_ms DESC LIMIT 1", (sym, ts_ms - lookback_ms)).fetchone()
    r1 = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? "
        "ORDER BY ts_ms DESC LIMIT 1", (sym, ts_ms)).fetchone()
    if r0 and r1 and float(r0[0]) > 0:
        return (float(r1[0]) - float(r0[0])) / float(r0[0]) * 10_000.0
    return 0.0

def liq_first_ts(conn, sym, side, lo, hi, thr):
    row = conn.execute(
        "SELECT ts_ms FROM liquidations "
        "WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?"
        " ORDER BY ts_ms ASC LIMIT 1", (sym, side, lo, hi, thr)).fetchone()
    return int(row[0]) if row else None

def session_name(ts_ms):
    h = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour
    if 7 <= h < 13:
        return "EUROPE"
    if 13 <= h < 21:
        return "US"
    return "OFF"

def _mc_p(vals, avg):
    if len(vals) < 4:
        return None
    rng = random.Random(0)
    ct = sum(1 for _ in range(MC_ITER)
             if sum(rng.choice([-1, 1]) * abs(v) for v in vals) / len(vals) >= avg)
    return round(ct / MC_ITER, 3)

def stats(gross_vals, label="", months=None, fee=FEE_BPS):
    m = months or 1.0
    if not gross_vals:
        return {"label": label, "n": 0}
    net = [v - fee for v in gross_vals]
    n = len(net)
    wins = sum(1 for v in net if v > 0)
    sv = sorted(net)
    avg = sum(net) / n
    total = sum(net)
    return {
        "label": label, "n": n,
        "wr": round(100 * wins / n, 1),
        "avg": round(avg, 1),
        "total": round(total, 0),
        "per_month": round(n / m, 1),
        "pnl_per_month": round(total / m, 0),
        "worst": round(sv[0], 1),
        "tail_n": sum(1 for v in net if v < -100),
        "mc_p": _mc_p(net, avg),
    }

# ---------------------------------------------------------------------------
# Dip→recovery sayacı (zigzag, tamamlanmış döngüler)
# ---------------------------------------------------------------------------

def count_dip_recoveries(closes, amp_bps, retrace_frac=RETRACE_FRAC):
    """closes: eskiden yeniye bar kapanışları (son bar ~T0).
    Tamamlanmış dip→recovery döngüsü sayısı. T0'a giren tamamlanmamış
    düşüş sayılmaz (in_dip halinde biterse o bacak yok sayılır)."""
    if len(closes) < 3:
        return None
    n = 0
    last_high = closes[0]
    in_dip = False
    dip_low = None
    for px in closes:
        if px <= 0:
            return None
        if not in_dip:
            if px >= last_high:
                last_high = px
            elif (last_high - px) / last_high * 10_000.0 >= amp_bps:
                in_dip = True
                dip_low = px
        else:
            if px < dip_low:
                dip_low = px
            elif (px - dip_low) >= retrace_frac * (last_high - dip_low):
                n += 1
                in_dip = False
                last_high = px
    return n

def bar_closes(marks, t0_ms, tf_min, n_bars):
    """T0'dan geriye n_bars adet TF-dakikalık bar kapanışı (eskiden yeniye).
    Bar kapanışı = bar sonundaki at_or_before mark. Veri penceresi eksikse None."""
    tf_ms = tf_min * 60_000
    start_needed = t0_ms - n_bars * tf_ms
    first = marks.at_or_before(start_needed)
    if first is None:
        return None  # pencerenin başında veri yok
    closes = []
    for k in range(n_bars, 0, -1):
        r = marks.at_or_before(t0_ms - (k - 1) * tf_ms - 1)
        if r is None:
            return None
        # bar kapanışı bar sonundan 1 TF'den eskiyse pencere delikli say
        if (t0_ms - (k - 1) * tf_ms) - r[0] > 2 * tf_ms:
            return None
        closes.append(float(r[1]))
    return closes

# ---------------------------------------------------------------------------
# Permütasyon testi: 2-3 bucket vs rest (label shuffle)
# ---------------------------------------------------------------------------

def perm_p_diff(pairs, in_bucket_fn, iters=MC_ITER):
    """pairs: [(dip_count, net_ret)]. Gözlenen diff = mean(bucket) - mean(rest).
    p = shuffle'da diff >= gözlenen olma oranı (tek yönlü, hipotez yönü)."""
    labels = [c for c, _ in pairs]
    rets   = [r for _, r in pairs]
    def diff(lbls):
        b  = [r for l, r in zip(lbls, rets) if in_bucket_fn(l)]
        nb = [r for l, r in zip(lbls, rets) if not in_bucket_fn(l)]
        if not b or not nb:
            return None
        return sum(b) / len(b) - sum(nb) / len(nb)
    obs = diff(labels)
    if obs is None:
        return None, None
    rng = random.Random(7)
    ct = 0
    valid = 0
    for _ in range(iters):
        sh = labels[:]
        rng.shuffle(sh)
        d = diff(sh)
        if d is None:
            continue
        valid += 1
        if d >= obs:
            ct += 1
    return round(obs, 1), (round(ct / valid, 3) if valid else None)

# ---------------------------------------------------------------------------
# Event inşası
# ---------------------------------------------------------------------------

def build_events(conn, anchors, marks_eth):
    events = []
    max_bars_ms = max(tf * 60_000 * bars[1] for tf, bars in TF_GRID.items())
    for anc in anchors:
        ts = int(anc.anchor_ts_ms)
        if marks_eth.at_or_after(ts) is None:
            continue
        if marks_eth.at_or_before(ts - max_bars_ms) is None:
            continue  # en uzun pencere için tarih öncesi veri yok
        ev = {"ts": ts}
        ok = True
        for hh in HOLDS_H:
            g = long_gross(marks_eth, ts, hh * 3600_000)
            if g is None and hh == PRIMARY_HOLD:
                ok = False
            ev["l%dh" % hh] = g
        if not ok:
            continue
        # gate feature'ları (hepsi T0 öncesi / T0 sonrası silence 30m sonra biliniyor —
        # silence deployed rotada zaten kullanılan post-entry yönetim gate'i)
        btc4h = mark_bps(conn, "BTCUSDT", ts, 4 * 3600_000)
        eth1h = mark_bps(conn, "ETHUSDT", ts, 3600_000)
        ev["bull"]  = eth1h > 20 and btc4h > 50
        ev["sess"]  = session_name(ts)
        ev["noisy"] = liq_first_ts(conn, "ETHUSDT", "SELL",
                                   ts + 60_000, ts + 30 * 60_000, PROP_THRESH) is not None
        # dip sayıları: her TF için en uzun pencerenin kapanışları bir kez,
        # kısa pencere = suffix
        ev["dips"] = {}
        for tf, (nb_short, nb_long) in TF_GRID.items():
            closes = bar_closes(marks_eth, ts, tf, nb_long)
            for nb in (nb_short, nb_long):
                key = "tf%d_n%d" % (tf, nb)
                sub = closes[-nb:] if closes is not None else None
                for amp in AMP_GRID:
                    ck = "%s_a%d" % (key, int(amp))
                    ev["dips"][ck] = (count_dip_recoveries(sub, amp)
                                      if sub is not None else None)
        events.append(ev)
    return events

# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def config_keys():
    for tf, (nb_s, nb_l) in TF_GRID.items():
        for nb in (nb_s, nb_l):
            for amp in AMP_GRID:
                yield tf, nb, amp, "tf%d_n%d_a%d" % (tf, nb, int(amp))

def in_23(c):
    return c in (2, 3)

def bucket_of(c):
    if c is None:
        return None
    if c >= 4:
        return "4+"
    return str(c)

def eval_config(events, ck, months):
    """Config için bucket istatistikleri + 2-3 vs rest diff."""
    pairs = [(ev["dips"][ck], ev["l%dh" % PRIMARY_HOLD] - FEE_BPS)
             for ev in events
             if ev["dips"].get(ck) is not None and ev["l%dh" % PRIMARY_HOLD] is not None]
    out = {"n": len(pairs)}
    by_bucket = {}
    for c, r in pairs:
        by_bucket.setdefault(bucket_of(c), []).append(r + FEE_BPS)  # stats gross bekler
    out["buckets"] = {b: stats(v, b, months) for b, v in sorted(by_bucket.items())}
    b23  = [r for c, r in pairs if in_23(c)]
    rest = [r for c, r in pairs if not in_23(c)]
    out["n_23"], out["n_rest"] = len(b23), len(rest)
    if b23 and rest:
        out["avg_23"]   = round(sum(b23) / len(b23), 1)
        out["avg_rest"] = round(sum(rest) / len(rest), 1)
        out["diff"]     = round(out["avg_23"] - out["avg_rest"], 1)
    return out, pairs

def main():
    print("=== S34 Pre-Cascade Dip-Recovery (2-3 dusus+cikis hipotezi) ===")
    with sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True) as conn:
        conn.execute("PRAGMA cache_size=-128000")
        conn.execute("PRAGMA temp_store=MEMORY")
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        start_ms = now_ms - LOOKBACK_MS

        print("Loading ETH SELL liqs ...")
        liqs = load_liquidations(conn, "ETHUSDT", "SELL", start_ms, now_ms)
        print("Reconstructing 200K anchors ...")
        anchors = reconstruct_anchors(liqs, bucket_sec=300, min_gap_sec=900,
                                      thresholds=(ETH_THRESH,), accel_window_sec=30)
        print("  anchors: %d" % len(anchors))
        print("Loading ETH mark index ...")
        marks_eth = load_mark_index(conn, "ETHUSDT")
        print("Building events (dip counts, seri grid) ...")
        events = build_events(conn, anchors, marks_eth)
    print("  events: %d" % len(events))
    if len(events) < 40:
        print("YETERSIZ EVENT — rapor yine yazilacak ama sonuc gecersiz sayilmali.")

    events.sort(key=lambda e: e["ts"])
    span_days = (events[-1]["ts"] - events[0]["ts"]) / 86_400_000 if len(events) > 1 else 30
    months = max(1.0, span_days / 30.0)
    cut = int(len(events) * TRAIN_FRAC)
    train, test = events[:cut], events[cut:]
    m_train = months * TRAIN_FRAC
    m_test  = months * (1 - TRAIN_FRAC)
    print("  span=%.0f gun, TRAIN n=%d / TEST n=%d" % (span_days, len(train), len(test)))

    # --- TRAIN taraması (seri) ---
    print("\n=== TRAIN taramasi (TF x lookback x amp) ===")
    grid = {}
    best = None
    for tf, nb, amp, ck in config_keys():
        res, _ = eval_config(train, ck, m_train)
        grid[ck] = res
        d = res.get("diff")
        if d is not None and res["n_23"] >= MIN_BUCKET_N:
            print("  %-16s N=%3d  n23=%3d avg23=%+7.1f rest=%+7.1f diff=%+7.1f"
                  % (ck, res["n"], res["n_23"], res["avg_23"], res["avg_rest"], d))
            if best is None or d > grid[best].get("diff", -1e9):
                best = ck
    if best is None:
        print("TRAIN'de MIN_BUCKET_N saglayan config yok — hipotez test edilemedi.")

    # --- TEST raporu (TRAIN'de secilen config) ---
    result = {"selected": best, "train_grid": grid}
    if best is not None:
        print("\n=== TEST (secilen config: %s) ===" % best)
        test_res, test_pairs = eval_config(test, best, m_test)
        obs, p = perm_p_diff(test_pairs, in_23)
        test_res["perm_diff"], test_res["perm_p"] = obs, p
        result["test"] = test_res
        for b, s in test_res["buckets"].items():
            print("  bucket %-3s n=%3d wr=%s avg=%s total=%s mc_p=%s"
                  % (b, s["n"], s.get("wr"), s.get("avg"), s.get("total"), s.get("mc_p")))
        print("  2-3 vs rest: diff=%s perm_p=%s (n23=%d rest=%d)"
              % (obs, p, test_res["n_23"], test_res["n_rest"]))

        # TRAIN'deki aynı config (referans) + tüm holdlar TEST'te
        result["train_selected"] = grid[best]
        holds = {}
        for hh in HOLDS_H:
            pairs_h = [(ev["dips"][best], ev["l%dh" % hh])
                       for ev in test
                       if ev["dips"].get(best) is not None and ev.get("l%dh" % hh) is not None]
            b23  = [r for c, r in pairs_h if in_23(c)]
            rest = [r for c, r in pairs_h if not in_23(c)]
            holds["l%dh" % hh] = {
                "b23": stats(b23, "2-3 dips", m_test),
                "rest": stats(rest, "rest", m_test),
            }
        result["test_holds"] = holds

        # Sekonder: core gate altında (not bull, not EU, silence)
        gated = [ev for ev in test
                 if not ev["bull"] and ev["sess"] != "EUROPE" and not ev["noisy"]]
        gres, gpairs = eval_config(gated, best, m_test)
        gobs, gp = perm_p_diff(gpairs, in_23)
        gres["perm_diff"], gres["perm_p"] = gobs, gp
        result["test_core_gated"] = gres
        print("  [core gate] n=%d, 2-3 vs rest diff=%s perm_p=%s" % (gres["n"], gobs, gp))

    meta = {
        "hypothesis": "liq oncesi 2-3 kere dusus+cikis (dip->recovery) sonucu ayirir",
        "universe": "ETHUSDT SELL 200K anchors, bucket=300s, min_gap=900s",
        "primary_outcome": "LONG %dh hold net (FEE=%.0fbps)" % (PRIMARY_HOLD, FEE_BPS),
        "retrace_frac": RETRACE_FRAC,
        "train_frac": TRAIN_FRAC,
        "n_events": len(events),
        "months": round(months, 2),
        "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "result": result}, f, indent=2, default=str)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(make_md(meta, result))
    print("\nJSON: %s\nMD:   %s" % (OUT_JSON, OUT_MD))

def make_md(meta, result):
    L = []
    L.append("# S34 Pre-Cascade Dip-Recovery Pattern (2-3 dusus+cikis hipotezi)\n")
    L.append("- Tarih: %s" % meta["generated_utc"])
    L.append("- Universe: %s" % meta["universe"])
    L.append("- Outcome: %s" % meta["primary_outcome"])
    L.append("- Events: %d (%.2f ay), TRAIN %.0f%%" % (meta["n_events"], meta["months"], meta["train_frac"] * 100))
    L.append("- Feature: T0 oncesi tamamlanmis dip->recovery dongusu (zigzag, retrace>=%.0f%%); "
             "T0'a giren tamamlanmamis dusus sayilmaz.\n" % (meta["retrace_frac"] * 100))
    best = result.get("selected")
    if best is None:
        L.append("**SONUC: TRAIN'de yeterli N'li config yok — hipotez test edilemedi.**\n")
        return "\n".join(L)
    L.append("## Secilen config (TRAIN'de secildi): `%s`\n" % best)
    tr = result.get("train_selected", {})
    L.append("TRAIN: n23=%s avg23=%s rest=%s diff=%s\n"
             % (tr.get("n_23"), tr.get("avg_23"), tr.get("avg_rest"), tr.get("diff")))
    te = result.get("test", {})
    L.append("## TEST sonucu\n")
    L.append("| bucket | N | WR% | avg net | total | worst | mc_p |")
    L.append("|---|---|---|---|---|---|---|")
    for b, s in te.get("buckets", {}).items():
        if s.get("n", 0) == 0:
            continue
        L.append("| %s | %d | %s | %s | %s | %s | %s |"
                 % (b, s["n"], s.get("wr"), s.get("avg"), s.get("total"),
                    s.get("worst"), s.get("mc_p")))
    L.append("\n**2-3 vs rest (TEST):** diff=%s bps, perm_p=%s (n23=%s, rest=%s)\n"
             % (te.get("perm_diff"), te.get("perm_p"), te.get("n_23"), te.get("n_rest")))
    L.append("### Hold duyarliligi (TEST, secilen config)\n")
    L.append("| hold | 2-3: N/WR/avg/total | rest: N/WR/avg/total |")
    L.append("|---|---|---|")
    for hk, hv in result.get("test_holds", {}).items():
        b, r = hv["b23"], hv["rest"]
        L.append("| %s | %s/%s/%s/%s | %s/%s/%s/%s |"
                 % (hk, b.get("n"), b.get("wr"), b.get("avg"), b.get("total"),
                    r.get("n"), r.get("wr"), r.get("avg"), r.get("total")))
    g = result.get("test_core_gated", {})
    L.append("\n### Core gate altinda (not bull, not EU, silence) — TEST\n")
    L.append("n=%s, 2-3 vs rest diff=%s perm_p=%s (n23=%s rest=%s)\n"
             % (g.get("n"), g.get("perm_diff"), g.get("perm_p"), g.get("n_23"), g.get("n_rest")))
    L.append("\n### TRAIN grid ozeti (min bucket N saglayanlar)\n")
    L.append("| config | N | n23 | avg23 | rest | diff |")
    L.append("|---|---|---|---|---|---|")
    for ck, res in result.get("train_grid", {}).items():
        if res.get("diff") is None or res.get("n_23", 0) < MIN_BUCKET_N:
            continue
        L.append("| %s | %d | %d | %s | %s | %s |"
                 % (ck, res["n"], res["n_23"], res["avg_23"], res["avg_rest"], res["diff"]))
    L.append("\n> Knowledge Object notu: kanit seviyesi = tek-universe TRAIN/TEST + perm testi; "
             "kapsam = ETH SELL 200K cascade LONG reversion; curutme kosulu = TEST perm_p > 0.05 "
             "veya diff isareti TRAIN/TEST arasi tutarsiz.\n")
    return "\n".join(L)

if __name__ == "__main__":
    main()
