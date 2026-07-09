"""S34 Mechanism Taxonomy — Faz 2: ayiricilar gated route uzerinde.

1. Store'a gate kolonlari yazar (btc4h/btc7d/bull/gated — ana DB'den, tek sefer).
2. hour17+regime GATED altkumede top mekanizma ayiricilarini test eder
   (esikler UNGATED TRAIN'den — kucuk-N gated'de esik secmek overfit olur).
3. Mekanizma kompozit skoru (0-7): funding<0, rv hi, retail lo, ofi hi,
   two_sided hi, refill hi, pull hi -> gated + ungated, TRAIN/TEST.
4. Funding seviye vs velocity ortak split (hangisi bagimsiz katki).

Girdi: mechanism_store.sqlite (+ ana DB gate icin, salt-okunur)
Cikti: reports/research/s34/S34_MECHANISM_TAXONOMY.json + .md
"""
from __future__ import annotations
import json, random, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.storage import production as PR
from ami.storage import research_reader as RR
STORE = ROOT / "reports" / "research" / "s34" / "mechanism_store.sqlite"
DB = ROOT / "data" / "microstructure.db"
OJ = ROOT / "reports" / "research" / "s34" / "S34_MECHANISM_TAXONOMY.json"
OM = ROOT / "reports" / "research" / "s34" / "S34_MECHANISM_TAXONOMY.md"
FEE = 5.0; MC = 500; TRAIN = 0.70
random.seed(42)

MECH = [("fund_rate", "lo"), ("px_rv", "hi"), ("fl_pre10_avg_sz", "lo"), ("fl_pre10_ofi", "hi"),
        ("liq_two_sided_1h", "hi"), ("bk_refill", "hi"), ("bk_pull", "hi"), ("fund_vel_1h", "lo")]


def med(x):
    s = sorted(v for v in x if v is not None)
    return s[len(s) // 2] if s else None


def mcp(v, a):
    if len(v) < 4: return None
    r = random.Random(0)
    ct = sum(1 for _ in range(MC) if sum(r.choice([-1, 1]) * abs(x) for x in v) / len(v) >= a)
    return round(ct / MC, 3)


def stat(g):
    net = [x - FEE for x in g if x is not None]
    if not net: return {"n": 0}
    n = len(net); w = sum(1 for x in net if x > 0); a = sum(net) / n
    return {"n": n, "wr": round(100 * w / n, 1), "avg": round(a, 1),
            "total": round(sum(net), 0), "worst": round(min(net), 1), "mc_p": mcp(net, a)}


def ps(k, v):
    if not v or v.get("n", 0) == 0:
        print("    %-40s N=0" % k[:40]); return
    print("    %-40s N=%-4d WR=%-6s avg=%-8s TOT=%-8s worst=%-8s mc=%s"
          % (k[:40], v["n"], str(v["wr"]) + "%", str(v["avg"]), str(v.get("total")),
             str(v.get("worst")), v.get("mc_p", "?")))


def mbps(main, s, ts, lb):
    """Direct-SQL oracle -- kept as the parity reference for `mbps_v2`
    (BATCH-STORAGE-ROTATION-RETENTION-ASOF-LOOKUP-CONSUMER-MIGRATION-V4).
    No longer called by ensure_gate_cols(); the reader-backed path is
    used instead. `main` is the read-only microstructure.db connection."""
    a = main.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (s, ts - lb)).fetchone()
    b = main.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (s, ts)).fetchone()
    return (float(b[0]) - float(a[0])) / float(a[0]) * 1e4 if a and b and float(a[0]) > 0 else None


def mbps_v2(root, s, ts, lb, source_db_path=None):
    """Reader-backed replacement for `mbps`, via lookup_latest_at_or_before
    for both endpoints. ETHUSDT resolves through the real mark_prices/
    ETHUSDT/2026-05 archive for historical timestamps; BTCUSDT has no
    mark_prices archive and resolves SQLITE_ONLY."""
    a = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol=s, ts_ms=ts - lb,
                                       columns=("mark_price",), source_db_path=source_db_path)
    b = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol=s, ts_ms=ts,
                                       columns=("mark_price",), source_db_path=source_db_path)
    if a.found and b.found and float(a.row[0]) > 0:
        return (float(b.row[0]) - float(a.row[0])) / float(a.row[0]) * 1e4
    return None


def ensure_gate_cols(st, main, root, source_db_path=None):
    cols = [r[1] for r in st.execute("PRAGMA table_info(events)")]
    for c in ("btc4h", "btc7d", "eth1h_gate", "bull", "gated"):
        if c not in cols:
            st.execute(f"ALTER TABLE events ADD COLUMN {c} REAL")
    if "gated" in cols:
        n = st.execute("SELECT COUNT(*) FROM events WHERE gated IS NOT NULL").fetchone()[0]
        if n > 0:
            print("  gate kolonlari zaten dolu"); return
    rows = st.execute("SELECT ts_ms, is_event, hour FROM events").fetchall()
    for i, (ts, is_ev, hour) in enumerate(rows):
        b4 = mbps_v2(root, "BTCUSDT", ts, 4 * 3600_000, source_db_path=source_db_path) or 0.0
        b7 = mbps_v2(root, "BTCUSDT", ts, 7 * 24 * 3600_000, source_db_path=source_db_path) or 0.0
        e1 = mbps_v2(root, "ETHUSDT", ts, 3600_000, source_db_path=source_db_path) or 0.0
        bull = 1.0 if (e1 > 20 and b4 > 50) else 0.0
        gated = 1.0 if (is_ev == 1 and hour >= 17 and (b4 < 0 or b7 < 0) and not bull) else 0.0
        st.execute("UPDATE events SET btc4h=?, btc7d=?, eth1h_gate=?, bull=?, gated=? WHERE ts_ms=?",
                   (b4, b7, e1, bull, gated, ts))
        if (i + 1) % 200 == 0: print(f"    gate {i+1}/{len(rows)}")
    st.commit()
    print(f"  gate kolonlari yazildi ({len(rows)} satir)")


def main():
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    print("=== S34 Mechanism Taxonomy (Faz 2) ===")
    st = sqlite3.connect(STORE)
    main_db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    root, _root_source = PR.resolve_production_root()
    ensure_gate_cols(st, main_db, root, source_db_path=str(DB))
    st.row_factory = sqlite3.Row
    ev = [dict(r) for r in st.execute("SELECT * FROM events WHERE is_event=1 AND y_6h IS NOT NULL ORDER BY ts_ms")]
    gated = [r for r in ev if r.get("gated") == 1.0]
    print(f"  event={len(ev)} gated(hour17+regime)={len(gated)}")
    cut = int(len(ev) * TRAIN); tr = ev[:cut]
    cut_ts = ev[cut]["ts_ms"] if cut < len(ev) else 0
    R = {"meta": {"n_event": len(ev), "n_gated": len(gated)}}

    # esikler UNGATED TRAIN medyani
    thr = {f: med([r[f] for r in tr if r.get(f) is not None]) for f, _ in MECH}
    R["thresholds"] = thr

    def hit(r, f, d):
        v = r.get(f)
        if v is None or thr[f] is None: return None
        return (v >= thr[f]) if d == "hi" else (v < thr[f])

    # 2 — gated altkumede tek-ayiricilar
    print("\n=== Gated (hour17+regime) altkumede mekanizma ayiricilari ===")
    G = {}
    for f, d in MECH:
        fav = [r["y_6h"] for r in gated if hit(r, f, d) is True]
        anti = [r["y_6h"] for r in gated if hit(r, f, d) is False]
        sf, sa = stat(fav), stat(anti)
        G[f] = {"dir": d, "fav": sf, "anti": sa,
                "delta": round((sf.get("avg") or 0) - (sa.get("avg") or 0), 1)}
        print("    %-20s %-3s favN=%-3d WR=%-6s avg=%-8s antiN=%-3d anti_avg=%-8s D=%s"
              % (f, d, sf.get("n", 0), str(sf.get("wr")) + "%", sf.get("avg"),
                 sa.get("n", 0), sa.get("avg"), G[f]["delta"]))
    R["gated_single"] = G

    # 3 — mekanizma kompoziti
    print("\n=== Mekanizma kompozit skoru (0-7, fund_vel haric) ===")
    def mscore(r):
        s = 0
        for f, d in MECH[:7]:
            if hit(r, f, d) is True: s += 1
        return s
    K = {}
    for uni_name, uni in (("ungated", ev), ("gated", gated)):
        for kmin in (3, 4, 5):
            sub = [r for r in uni if mscore(r) >= kmin]
            K[f"{uni_name}_m{kmin}"] = stat([r["y_6h"] for r in sub])
            K[f"{uni_name}_m{kmin}_TEST"] = stat([r["y_6h"] for r in sub if r["ts_ms"] >= cut_ts])
            ps(f"{uni_name} mech>={kmin}", K[f"{uni_name}_m{kmin}"])
            ps(f"{uni_name} mech>={kmin} TEST", K[f"{uni_name}_m{kmin}_TEST"])
    R["mech_composite"] = K

    # 4 — funding seviye vs velocity ortak split
    print("\n=== Funding: seviye vs velocity (2x2, ungated) ===")
    F = {}
    for lv in ("lo", "hi"):
        for vl in ("lo", "hi"):
            sub = [r["y_6h"] for r in ev
                   if hit(r, "fund_rate", "lo") is (lv == "lo") and hit(r, "fund_vel_1h", "lo") is (vl == "lo")]
            F[f"rate_{lv}_vel_{vl}"] = stat(sub)
            ps(f"rate={lv} vel={vl}", F[f"rate_{lv}_vel_{vl}"])
    R["funding_2x2"] = F

    st.close(); main_db.close()
    OJ.write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    lines = ["# S34 Mechanism Taxonomy (Faz 2)", "",
             f"> event={len(ev)} gated={len(gated)}. Esikler ungated-TRAIN. {datetime.now(timezone.utc):%Y-%m-%d}", "",
             "## Gated altkume tek-ayiricilar", ""]
    for f, v in sorted(G.items(), key=lambda kv: abs(kv[1]["delta"]), reverse=True):
        s = v["fav"]
        if s.get("n", 0) >= 8:
            lines.append("- **%s** (%s): favN=%d WR=%s%% avg=%+.1f (anti %+.1f) D=%+.1f mc=%s"
                         % (f, v["dir"], s["n"], s["wr"], s["avg"], v["anti"].get("avg") or 0,
                            v["delta"], s.get("mc_p")))
    lines += ["", "## Mekanizma kompoziti", ""]
    for k, v in K.items():
        if v.get("n", 0) > 0:
            lines.append("- **%s**: N=%d WR=%s%% avg=%+.1f TOT=%s worst=%s mc=%s"
                         % (k, v["n"], v["wr"], v["avg"], v.get("total"), v.get("worst"), v.get("mc_p")))
    lines += ["", "## Funding 2x2 (seviye x velocity)", ""]
    for k, v in F.items():
        if v.get("n", 0) > 0:
            lines.append("- **%s**: N=%d WR=%s%% avg=%+.1f mc=%s" % (k, v["n"], v["wr"], v["avg"], v.get("mc_p")))
    lines += ["", "---", "*Script: tools/s34_mechanism_taxonomy.py*"]
    OM.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nJSON:{OJ}\nMD:  {OM}\nDone.")


if __name__ == "__main__":
    main()
