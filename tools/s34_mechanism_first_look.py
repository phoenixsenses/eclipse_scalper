"""S34 Mechanism First Look — Faz 2 on-analizi (store uzerinde, hizli).

A  continuation vs reversal: hangi mekanizma degiskeni ayiriyor?
   (pull/refill/impact/OFI/funding-velocity/basis/imbalance) TRAIN median -> TEST
B  pre-cascade prediction on-izleme: pre10m ozellikleri event'i kontrolden ayiriyor mu?
C  execution grid: giris gecikme egrisi (2s..15m) + reversal-kosullu

Girdi: reports/research/s34/mechanism_store.sqlite
Cikti: reports/research/s34/S34_MECHANISM_FIRST_LOOK.json + .md
"""
from __future__ import annotations
import json, random, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "reports" / "research" / "s34" / "mechanism_store.sqlite"
OJ = ROOT / "reports" / "research" / "s34" / "S34_MECHANISM_FIRST_LOOK.json"
OM = ROOT / "reports" / "research" / "s34" / "S34_MECHANISM_FIRST_LOOK.md"
FEE = 5.0; MC = 500; TRAIN = 0.70
random.seed(42)

MECH_FEATS = ["bk_pull", "bk_refill", "fl_pre10_impact", "fl_post1_impact", "fl_post5_impact",
              "fl_pre10_ofi", "fl_post1_ofi", "fund_rate", "fund_vel_1h", "fund_vel_8h",
              "basis_spot_bps", "basis_spot_slope", "bk_pre10_imb", "bk_pre1_imb", "bk_pre10_imb_slope",
              "bk_pre10_spread", "bk_post1_spread_max", "fl_pre10_avg_sz", "fl_post1_avg_sz",
              "liq_two_sided_1h", "liq_btc_sync", "px_rv", "px_ret_1h"]
PRE_FEATS = [f for f in MECH_FEATS if f.startswith(("bk_pre", "fl_pre", "fund", "basis", "px_", "liq_"))] + ["bk_pull"]


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
            "total": round(sum(net), 0), "mc_p": mcp(net, a)}


def main():
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    print("=== S34 Mechanism First Look ===")
    c = sqlite3.connect(f"file:{STORE}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    rows = [dict(r) for r in c.execute("SELECT * FROM events ORDER BY ts_ms")]
    ev = [r for r in rows if r["is_event"] == 1 and r.get("y_6h") is not None]
    ctl = [r for r in rows if r["is_event"] == 0]
    print(f"  event={len(ev)} kontrol={len(ctl)}")
    cut = int(len(ev) * TRAIN); tr, te = ev[:cut], ev[cut:]
    R = {"meta": {"n_event": len(ev), "n_control": len(ctl)}}

    # A — continuation vs reversal + LONG y6h ayrimi
    print("\n=== A: mekanizma degiskenleri (TRAIN median -> TEST, hedef y_6h LONG) ===")
    A = {}
    for f in MECH_FEATS:
        vals = [r[f] for r in tr if r.get(f) is not None]
        if len(vals) < 30: continue
        thr = med(vals)
        hi_tr = [r["y_6h"] for r in tr if r.get(f) is not None and r[f] >= thr]
        lo_tr = [r["y_6h"] for r in tr if r.get(f) is not None and r[f] < thr]
        if not hi_tr or not lo_tr: continue
        fav = "hi" if sum(hi_tr) / len(hi_tr) > sum(lo_tr) / len(lo_tr) else "lo"
        pick = (lambda r, t=thr: r.get(f) is not None and r[f] >= t) if fav == "hi" else \
               (lambda r, t=thr: r.get(f) is not None and r[f] < t)
        s_fav = stat([r["y_6h"] for r in te if pick(r)])
        s_anti = stat([r["y_6h"] for r in te if r.get(f) is not None and not pick(r)])
        # continuation orani farki (mekanizma yorumu)
        cont_fav = [r["lbl_continuation"] for r in te if pick(r) and r.get("lbl_continuation") is not None]
        cont_anti = [r["lbl_continuation"] for r in te if r.get(f) is not None and not pick(r) and r.get("lbl_continuation") is not None]
        cf = round(100 * sum(cont_fav) / len(cont_fav), 1) if cont_fav else None
        ca = round(100 * sum(cont_anti) / len(cont_anti), 1) if cont_anti else None
        delta = (s_fav.get("avg") or 0) - (s_anti.get("avg") or 0)
        A[f] = {"fav": fav, "thr": thr, "TEST_fav": s_fav, "TEST_anti": s_anti,
                "delta": round(delta, 1), "cont_fav_pct": cf, "cont_anti_pct": ca}
        if s_fav.get("n", 0) >= 10:
            print("  %-22s fav=%-3s TESTn=%-3d WR=%-6s avg=%-7s anti_avg=%-7s D=%-7s cont%%: %s vs %s"
                  % (f, fav, s_fav["n"], str(s_fav["wr"]) + "%", s_fav["avg"], s_anti.get("avg"),
                     round(delta, 1), cf, ca))
    R["A"] = A

    # B — event vs kontrol ayrimi (pre-only)
    print("\n=== B: pre-cascade prediction on-izleme (event vs kontrol, pre-ozellikler) ===")
    B = {}
    for f in sorted(set(PRE_FEATS)):
        eV = [r[f] for r in ev if r.get(f) is not None]
        cV = [r[f] for r in ctl if r.get(f) is not None]
        if len(eV) < 30 or len(cV) < 30: continue
        me_, mc_ = med(eV), med(cV)
        # basit ayirma gucu: kontrol medyani esiginde event'lerin hangi orani ayni tarafta
        hi_rate_e = sum(1 for v in eV if v >= mc_) / len(eV)
        sep = abs(hi_rate_e - 0.5) * 2  # 0=ayrim yok, 1=tam ayrim
        B[f] = {"med_event": me_, "med_control": mc_, "sep": round(sep, 3),
                "n_e": len(eV), "n_c": len(cV)}
    for f, v in sorted(B.items(), key=lambda kv: kv[1]["sep"], reverse=True)[:12]:
        print("  %-22s sep=%-6s med_ev=%-12s med_ctl=%s" % (f, v["sep"], round(v["med_event"], 6) if isinstance(v["med_event"], float) else v["med_event"], round(v["med_control"], 6) if isinstance(v["med_control"], float) else v["med_control"]))
    R["B"] = B

    # C — execution grid
    print("\n=== C: giris gecikme egrisi (event'ler, y 6h-hold) ===")
    Cx = {}
    for lbl in ("2s", "5s", "10s", "30s", "1m", "5m", "15m"):
        Cx[f"grid_{lbl}"] = stat([r.get(f"yg_{lbl}") for r in ev])
        s = Cx[f"grid_{lbl}"]
        ent = [r.get(f"ent_{lbl}_bps") for r in ev if r.get(f"ent_{lbl}_bps") is not None]
        Cx[f"grid_{lbl}"]["ent_drift_bps"] = round(sum(ent) / len(ent), 1) if ent else None
        print("  +%-4s N=%-4d WR=%-6s avg=%-7s TOT=%-8s giris-drift=%-6s mc=%s"
              % (lbl, s.get("n", 0), str(s.get("wr")) + "%", s.get("avg"), s.get("total"),
                 Cx[f"grid_{lbl}"]["ent_drift_bps"], s.get("mc_p")))
    # reversal-kosullu grid (dusuk impact + refill hizli altkumesi ornegi TRAIN esikli)
    thr_ref = med([r["bk_refill"] for r in tr if r.get("bk_refill") is not None])
    if thr_ref is not None:
        sub = [r for r in ev if r.get("bk_refill") is not None and r["bk_refill"] >= thr_ref]
        for lbl in ("10s", "1m", "5m"):
            s = stat([r.get(f"yg_{lbl}") for r in sub])
            Cx[f"grid_refillhi_{lbl}"] = s
            print("  refill-hi +%-4s N=%-4d WR=%-6s avg=%-7s mc=%s" % (lbl, s.get("n", 0), str(s.get("wr")) + "%", s.get("avg"), s.get("mc_p")))
    R["C"] = Cx

    OJ.write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    lines = ["# S34 Mechanism First Look", "",
             f"> event={len(ev)} kontrol={len(ctl)}. {datetime.now(timezone.utc):%Y-%m-%d}", "",
             "## A — continuation vs reversal ayiricilar (TEST)", ""]
    for f, v in sorted(A.items(), key=lambda kv: abs(kv[1]["delta"]), reverse=True):
        t = v["TEST_fav"]
        if t.get("n", 0) >= 10:
            lines.append("- **%s** fav=%s: TEST N=%d WR=%s%% avg=%+.1f (anti %+.1f, D=%+.1f) cont%% %s→%s"
                         % (f, v["fav"], t["n"], t["wr"], t["avg"], v["TEST_anti"].get("avg") or 0,
                            v["delta"], v["cont_anti_pct"], v["cont_fav_pct"]))
    lines += ["", "## B — event-vs-kontrol ayrimi (pre-cascade)", ""]
    for f, v in sorted(B.items(), key=lambda kv: kv[1]["sep"], reverse=True)[:12]:
        lines.append(f"- **{f}**: sep={v['sep']} (med_ev={v['med_event']}, med_ctl={v['med_control']})")
    lines += ["", "## C — giris gecikme egrisi", ""]
    for k, v in Cx.items():
        if v.get("n", 0) > 0:
            lines.append("- **%s**: N=%d WR=%s%% avg=%+.1f mc=%s" % (k, v["n"], v.get("wr"), v.get("avg") or 0, v.get("mc_p")))
    lines += ["", "---", "*Script: tools/s34_mechanism_first_look.py*"]
    OM.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nJSON:{OJ}\nMD:  {OM}\nDone.")


if __name__ == "__main__":
    main()
