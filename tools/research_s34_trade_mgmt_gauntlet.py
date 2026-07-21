"""S34 Trade Management Gauntlet — giris olgun, simdi YONETIM.

Evren: hour17 composite (200K, tam gecmis 4.5 ay) + 100K route (s>=3) — LONG.
Her event icin 1m cozunurluklu 12h mark path. TRAIN/TEST 70/30, MC, no-overlap mdd.
KURAL: hicbir varyant baseline mdd'sini asamaz (kullanici kisiti).

M1  MFE-giveback anatomisi (peak zamani, giveback, conviction'a gore)
M2  Hold suresi grid (2..12h) x conviction (s7 2-3 vs 4+)
M3  Exit-saat etkisi + saat-hedefli cikis (23:00 / 07:00 UTC)
M4  Partial exit: %50@+100 rest 6h; %50@3h; 1/3+1/3+1/3; conviction-kosullu
M5  Profit-lock (arm/lock): 100/50, 150/75, 200/100, 300/150
M6  Loser time-stop: t={1,2,3}h x thr={-25,-50,-75,-100}
M7  Scale-in @-75/-100 ilk 2h (cift birim, 6h cikis) — mdd hesabiyla
M8  Conviction-politikasi: hi=full+uzun, lo=yarim/partial — uniform'a karsi
M9  Portfoy: tek-slot vs 2-slot (200K+100K+SHORT1317), conviction-oncelik,
    gunluk kayip throttle, after-loss yarim boyut
M10 Bar-trailing tum timeframe'ler: 1m/5m/15m/1h/4h N-bar-low cikisi

Cikti: reports/research/s34/S34_TRADE_MGMT.json + .md
"""
from __future__ import annotations
import json, random, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools.research_s34_knowable_anchor_continuation import load_liquidations, load_mark_index, reconstruct_anchors

DB = ROOT / "data" / "microstructure.db"; OUT = ROOT / "reports" / "research" / "s34"
OJ = OUT / "S34_TRADE_MGMT.json"; OM = OUT / "S34_TRADE_MGMT.md"
LB = 400 * 24 * 3600_000; FEE = 5.0; MC = 500; TRAIN = 0.70
H6 = 360; PATH_MIN = 721  # dakika
CT = {"sync": 0.5421, "rvp": 0.0026337, "d24": 5.0, "be_lo": 0.2195, "be_hi": 2.0,
      "shelf": 2_775_000.0, "whale": 6440.0}
random.seed(42)
import math


def _s(c, q, p=()):
    r = c.execute(q, p).fetchone(); return float(r[0]) if r and r[0] is not None else 0.0
def lsum(c, s, sd, lo, hi): return _s(c, "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?", (s, sd, lo, hi))
def lmax(c, s, sd, lo, hi): return _s(c, "SELECT COALESCE(MAX(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?", (s, sd, lo, hi))
def lcnt(c, s, sd, lo, hi, t): return int(_s(c, "SELECT COUNT(*) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?", (s, sd, lo, hi, t)))
def lfirst(c, s, sd, lo, hi, t):
    r = c.execute("SELECT ts_ms FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=? ORDER BY ts_ms ASC LIMIT 1", (s, sd, lo, hi, t)).fetchone()
    return int(r[0]) if r else None
def mbps(c, s, ts, lb):
    a = c.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (s, ts - lb)).fetchone()
    b = c.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (s, ts)).fetchone()
    return (float(b[0]) - float(a[0])) / float(a[0]) * 1e4 if a and b and float(a[0]) > 0 else None
def ofir(c, lo, hi):
    r = c.execute("SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END),SUM(notional),COUNT(*) FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?", (lo, hi)).fetchone()
    if not r or r[0] is None: return None, None
    b, se = float(r[0]), float(r[1]); t = b + se
    return ((b - se) / t if t > 0 else 0.0), ((float(r[2]) / int(r[3])) if r[3] else None)
def nextfund(c, ts):
    r = c.execute("SELECT next_funding_time_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND next_funding_time_ms IS NOT NULL ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    return int(r[0]) if r and r[0] else None
def hod(ts): return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).hour
def dowf(ts): return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).weekday()
def sxn(ts):
    h = hod(ts); return "EUROPE" if 7 <= h < 13 else ("US" if 13 <= h < 21 else "OFF")
def rv_proxy(m, ts):
    px = []
    for k in range(5, -1, -1):
        r = m.at_or_before(ts - k * 60_000)
        if r is None: return None
        px.append(float(r[1]))
    rets = [math.log(px[i + 1] / px[i]) for i in range(5) if px[i] > 0]
    return math.sqrt(sum(x * x for x in rets)) if len(rets) == 5 else None


def mcp(v, a):
    if len(v) < 4: return None
    r = random.Random(0)
    ct = sum(1 for _ in range(MC) if sum(r.choice([-1, 1]) * abs(x) for x in v) / len(v) >= a)
    return round(ct / MC, 3)
def mdd(vals):
    eq = peak = d = 0.0
    for v in vals:
        eq += v; peak = max(peak, eq); d = min(d, eq - peak)
    return round(d, 1)
def stat(g, fee=FEE):
    net = [x - fee for x in g if x is not None]
    if not net: return {"n": 0}
    n = len(net); w = sum(1 for x in net if x > 0); a = sum(net) / n
    return {"n": n, "wr": round(100 * w / n, 1), "avg": round(a, 1), "total": round(sum(net), 0),
            "worst": round(min(net), 1), "mdd": mdd(net), "mc_p": mcp(net, a)}
def ps(k, v):
    if not v or v.get("n", 0) == 0:
        print("    %-40s N=0" % k[:40]); return
    print("    %-40s N=%-4d WR=%-6s avg=%-8s TOT=%-8s worst=%-8s mdd=%-8s mc=%s"
          % (k[:40], v["n"], str(v["wr"]) + "%", str(v["avg"]), str(v.get("total")),
             str(v.get("worst")), str(v.get("mdd")), v.get("mc_p", "?")))


def s7_of(conn, m, ts, rn):
    of, whale = ofir(conn, ts - 5 * 60_000, ts)
    e = m.at_or_after(ts); px = float(e[1]) if e else None
    shelf = _s(conn, "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<? AND price>=? AND price<=?", (ts - 24 * 3600_000, ts, px * 0.98, px)) if px else 0
    sk = lsum(conn, "BTCUSDT", "SELL", ts - 600_000, ts) + lsum(conn, "SOLUSDT", "SELL", ts - 600_000, ts)
    rvp = rv_proxy(m, ts)
    return sum([(sk / rn if rn > 0 else 0) >= CT["sync"],
                rvp is not None and rvp >= CT["rvp"],
                lcnt(conn, "ETHUSDT", "SELL", ts - 24 * 3600_000, ts - 300_000, 200_000) >= CT["d24"],
                of is not None and of >= 0,
                CT["be_lo"] <= (lmax(conn, "BTCUSDT", "SELL", ts - 600_000, ts) / rn if rn > 0 else 0) < CT["be_hi"],
                shelf >= CT["shelf"],
                whale is not None and whale < CT["whale"]])


def build_universe(conn, m, now, start, thresh, smin):
    ancs = reconstruct_anchors(load_liquidations(conn, "ETHUSDT", "SELL", start, now),
                               bucket_sec=300, min_gap_sec=900, thresholds=(thresh,), accel_window_sec=30)
    ev = []
    for a in ancs:
        ts = int(a.anchor_ts_ms); rn = float(a.running_notional)
        if rn < thresh or (thresh == 100_000.0 and rn >= 200_000.0): continue
        if m.at_or_after(ts) is None: continue
        b4 = mbps(conn, "BTCUSDT", ts, 4 * 3600_000) or 0
        b7 = mbps(conn, "BTCUSDT", ts, 7 * 24 * 3600_000) or 0
        if ((mbps(conn, "ETHUSDT", ts, 3600_000) or 0) > 20 and b4 > 50) or sxn(ts) == "EUROPE" \
                or not (b4 < 0 or b7 < 0) or hod(ts) < 17:
            continue
        nf = nextfund(conn, ts); m2f = ((nf - ts) / 60_000) if nf else None
        if m2f is not None and m2f < 60: continue
        s7 = s7_of(conn, m, ts, rn)
        if s7 < smin: continue
        e0 = m.at_or_after(ts)
        p0 = float(e0[1])
        path = []
        ok = True
        for k in range(PATH_MIN):
            r = m.at_or_before(ts + k * 60_000) if k else e0
            if r is None: ok = False; break
            path.append((float(r[1]) - p0) / p0 * 1e4)
        if not ok or len(path) < 361: continue
        ev.append({"ts": ts, "s7": s7, "dow": dowf(ts), "hour": hod(ts),
                   "route": "200k" if thresh == 200_000.0 else "100k", "path": path})
    ev.sort(key=lambda x: x["ts"])
    return ev


def noov(evs, hold_min=H6):
    busy = -1; o = []
    for e in evs:
        if e["ts"] >= busy: o.append(e); busy = e["ts"] + hold_min * 60_000
    return o


def ret_at(e, k): return e["path"][min(k, len(e["path"]) - 1)]


def main():
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    print("=== S34 Trade Management Gauntlet ===")
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.execute("PRAGMA cache_size=-200000")
    now = int(datetime.now(tz=timezone.utc).timestamp() * 1000); start = now - LB
    m = load_mark_index(conn, "ETHUSDT")
    print("build 200k...")
    u200 = build_universe(conn, m, now, start, 200_000.0, 2)
    print(f"  200k s>=2: {len(u200)}")
    print("build 100k...")
    u100 = build_universe(conn, m, now, start, 100_000.0, 3)
    print(f"  100k s>=3: {len(u100)}")
    allev = sorted(u200 + u100, key=lambda x: x["ts"])
    adm = noov(allev)          # tek-slot gerceklik, baseline admission
    n = len(adm); cut = int(n * TRAIN); cut_ts = adm[cut]["ts"] if cut < n else 0
    TM = max(1.0, (adm[-1]["ts"] - adm[0]["ts"]) / 86_400_000 / 30.0)
    print(f"  admitted(no-overlap): {n}  months={TM:.2f}")
    R = {"meta": {"n_200k": len(u200), "n_100k": len(u100), "n_admitted": n, "months": round(TM, 2)}}
    te = [e for e in adm if e["ts"] >= cut_ts]

    base = stat([ret_at(e, H6) for e in adm])
    base_te = stat([ret_at(e, H6) for e in te])
    print("\n  BASELINE (6h hold):"); ps("baseline", base); ps("baseline TEST", base_te)
    R["baseline"] = base; R["baseline_TEST"] = base_te
    BASE_MDD = abs(base["mdd"])

    # ---- M1 MFE-giveback anatomi
    print("\n=== M1: MFE-giveback anatomisi ===")
    m1 = {}
    for lbl, grp in (("all", adm), ("hi_conv", [e for e in adm if e["s7"] >= 4]),
                     ("lo_conv", [e for e in adm if e["s7"] < 4])):
        mfes = []; gives = []; tpk = []
        for e in grp:
            w = e["path"][:H6 + 1]
            pk = max(w); fin = w[H6] if len(w) > H6 else w[-1]
            mfes.append(pk); gives.append(pk - fin); tpk.append(w.index(pk))
        if mfes:
            m1[lbl] = {"n": len(grp), "avg_mfe": round(sum(mfes) / len(mfes), 1),
                       "avg_giveback": round(sum(gives) / len(gives), 1),
                       "med_peak_min": sorted(tpk)[len(tpk) // 2]}
            print("    %-8s N=%-4d MFE=%-7s giveback=%-7s medyan-peak=%s dk"
                  % (lbl, len(grp), m1[lbl]["avg_mfe"], m1[lbl]["avg_giveback"], m1[lbl]["med_peak_min"]))
    R["M1"] = m1

    # ---- M2 hold grid x conviction
    print("\n=== M2: hold suresi x conviction ===")
    m2 = {}
    for hmin, hl in ((120, "2h"), (180, "3h"), (240, "4h"), (360, "6h"), (480, "8h"), (600, "10h"), (720, "12h")):
        for cl, grp in (("hi", [e for e in adm if e["s7"] >= 4]), ("lo", [e for e in adm if e["s7"] < 4])):
            m2[f"{hl}_{cl}"] = stat([ret_at(e, hmin) for e in grp])
        a, b = m2[f"{hl}_hi"], m2[f"{hl}_lo"]
        print("    %-4s hi: N=%-3d avg=%-7s mdd=%-8s | lo: N=%-3d avg=%-7s mdd=%s"
              % (hl, a.get("n", 0), a.get("avg"), a.get("mdd"), b.get("n", 0), b.get("avg"), b.get("mdd")))
    R["M2"] = m2

    # ---- M3 exit-saat
    print("\n=== M3: exit-saat etkisi + saat-hedefli cikis ===")
    m3 = {}
    byh = {}
    for e in adm:
        xh = hod(e["ts"] + H6 * 60_000)
        byh.setdefault(xh // 4 * 4, []).append(ret_at(e, H6))
    for hb in sorted(byh):
        m3[f"exit_h{hb:02d}"] = stat(byh[hb])
        print("    exit %02d-%02d UTC: N=%-3d avg=%-7s" % (hb, hb + 4, m3[f"exit_h{hb:02d}"]["n"], m3[f"exit_h{hb:02d}"]["avg"]))
    def clock_exit(e, target_h, min_hold=120, max_hold=720):
        for k in range(min_hold, max_hold + 1):
            if hod(e["ts"] + k * 60_000) == target_h:
                return ret_at(e, k)
        return ret_at(e, H6)
    for th in (23, 3, 7):
        m3[f"clock_{th:02d}utc"] = stat([clock_exit(e, th) for e in adm])
        ps(f"clock-exit {th:02d}:00 UTC", m3[f"clock_{th:02d}utc"])
    R["M3"] = m3

    # ---- M4 partial exits
    print("\n=== M4: partial exit varyantlari ===")
    def pol_half_at(e, trig, rest_min=H6):
        w = e["path"]
        for k in range(1, rest_min + 1):
            if w[min(k, len(w) - 1)] >= trig:
                return 0.5 * trig + 0.5 * ret_at(e, rest_min)
        return ret_at(e, rest_min)
    def pol_half_time(e, t_min):
        return 0.5 * ret_at(e, t_min) + 0.5 * ret_at(e, H6)
    def pol_thirds(e):
        w = e["path"]; got = []; rem = 1.0
        for trig in (100.0, 200.0):
            hit = next((k for k in range(1, H6 + 1) if w[min(k, len(w) - 1)] >= trig), None)
            if hit is not None: got.append((1 / 3) * trig); rem -= 1 / 3
        return sum(got) + rem * ret_at(e, H6)
    m4 = {}
    m4["half_at_100"] = stat([pol_half_at(e, 100.0) for e in adm])
    m4["half_at_150"] = stat([pol_half_at(e, 150.0) for e in adm])
    m4["half_at_3h"] = stat([pol_half_time(e, 180) for e in adm])
    m4["thirds_100_200"] = stat([pol_thirds(e) for e in adm])
    m4["conv_cond"] = stat([ret_at(e, H6) if e["s7"] >= 4 else pol_half_at(e, 100.0) for e in adm])
    for k, v in m4.items(): ps(f"M4 {k}", v)
    R["M4"] = m4

    # ---- M5 profit-lock
    print("\n=== M5: profit-lock (arm/lock) ===")
    def pol_lock(e, arm, lock):
        w = e["path"]; armed = False
        for k in range(1, H6 + 1):
            r = w[min(k, len(w) - 1)]
            if not armed and r >= arm: armed = True
            if armed and r <= lock: return lock
        return ret_at(e, H6)
    m5 = {}
    for arm, lock in ((100, 50), (150, 75), (200, 100), (300, 150)):
        m5[f"lock_{arm}_{lock}"] = stat([pol_lock(e, arm, lock) for e in adm])
        ps(f"M5 lock {arm}/{lock}", m5[f"lock_{arm}_{lock}"])
    R["M5"] = m5

    # ---- M6 loser time-stop
    print("\n=== M6: loser time-stop ===")
    def pol_tstop(e, t_min, thr):
        if ret_at(e, t_min) <= thr: return ret_at(e, t_min)
        return ret_at(e, H6)
    m6 = {}
    for t_min, tl in ((60, "1h"), (120, "2h"), (180, "3h")):
        for thr in (-25.0, -50.0, -75.0, -100.0):
            key = f"ts_{tl}_{int(abs(thr))}"
            m6[key] = stat([pol_tstop(e, t_min, thr) for e in adm])
    top6 = sorted(m6.items(), key=lambda kv: kv[1].get("avg") or -999, reverse=True)[:4]
    for k, v in top6: ps(f"M6 {k}", v)
    R["M6"] = m6

    # ---- M7 scale-in
    print("\n=== M7: scale-in dip (ilk 2h) ===")
    def pol_scalein(e, dip):
        w = e["path"]
        hit = next((k for k in range(1, 121) if w[min(k, len(w) - 1)] <= dip), None)
        fin = ret_at(e, H6)
        if hit is None: return fin, 1.0
        return fin + (fin - dip), 2.0   # 2 birim pnl (birim-bps toplami)
    m7 = {}
    for dip in (-75.0, -100.0):
        res = [pol_scalein(e, dip) for e in adm]
        pnl = [r - FEE * u for r, u in res]
        units = sum(u for _, u in res)
        st_ = {"n": len(pnl), "w_total": round(sum(pnl), 0), "per_unit": round(sum(pnl) / units, 1),
               "w_worst": round(min(pnl), 1), "w_mdd": mdd(pnl)}
        m7[f"scalein_{int(abs(dip))}"] = st_
        print("    M7 dip=%-5s wTOT=%-8s perU=%-6s wWorst=%-8s wMDD=%s"
              % (dip, st_["w_total"], st_["per_unit"], st_["w_worst"], st_["w_mdd"]))
    R["M7"] = m7

    # ---- M10 bar-trailing tum timeframe'ler
    print("\n=== M10: N-bar-low trailing (1m/5m/15m/1h/4h) ===")
    m10 = {}
    for tf, tl in ((1, "1m"), (5, "5m"), (15, "15m"), (60, "1h"), (240, "4h")):
        def pol_trail(e, tf=tf, nbar=3):
            w = e["path"]; closes = [w[min(k, len(w) - 1)] for k in range(0, H6 + 1, tf)]
            for i in range(nbar + 1, len(closes)):
                if closes[i] < min(closes[i - nbar:i]) and closes[i] > -900:
                    if i * tf >= 30:  # ilk 30dk'da cikma (dip bolgesi)
                        return closes[i]
            return ret_at(e, H6)
        m10[f"trail_{tl}"] = stat([pol_trail(e) for e in adm])
        ps(f"M10 trail {tl} (3-bar-low)", m10[f"trail_{tl}"])
    R["M10"] = m10

    # ---- M8 conviction-politikasi + TEST dogrulama (en iyi adaylar)
    print("\n=== M8: politika sentezi (TEST) ===")
    m8 = {}
    m8["uniform_6h_TEST"] = stat([ret_at(e, H6) for e in te])
    cands = {
        "hold8h_hi_6h_lo": lambda e: ret_at(e, 480) if e["s7"] >= 4 else ret_at(e, H6),
        "half100_lo_full_hi": lambda e: ret_at(e, H6) if e["s7"] >= 4 else pol_half_at(e, 100.0),
        "clock23_all": lambda e: clock_exit(e, 23),
    }
    # M2'de TRAIN'de hi icin en iyi hold'u sec
    tr_adm = [e for e in adm if e["ts"] < cut_ts]
    best_h_hi = max(((h, stat([ret_at(e, h) for e in tr_adm if e["s7"] >= 4]).get("avg") or -999)
                     for h in (240, 360, 480, 600, 720)), key=lambda x: x[1])[0]
    best_h_lo = max(((h, stat([ret_at(e, h) for e in tr_adm if e["s7"] < 4]).get("avg") or -999)
                     for h in (120, 180, 240, 360, 480)), key=lambda x: x[1])[0]
    m8["train_best_hold"] = {"hi": best_h_hi, "lo": best_h_lo}
    cands[f"trainpick_h{best_h_hi}hi_h{best_h_lo}lo"] = lambda e: ret_at(e, best_h_hi if e["s7"] >= 4 else best_h_lo)
    for k, fn in cands.items():
        m8[k + "_TEST"] = stat([fn(e) for e in te])
        ps(f"M8 {k} TEST", m8[k + "_TEST"])
    ps("M8 uniform 6h TEST", m8["uniform_6h_TEST"])
    R["M8"] = m8

    # ---- M9 portfoy: slot/throttle/after-loss
    print("\n=== M9: portfoy yonetimi ===")
    m9 = {}
    def run_port(evs, slots=1, throttle_day_bps=None, halfsize_after_loss=False):
        evs = sorted(evs, key=lambda x: x["ts"])
        busy = []; pnl = []; day_pnl = {}; last_loss = False
        for e in evs:
            busy = [b for b in busy if b > e["ts"]]
            day = e["ts"] // 86_400_000
            if throttle_day_bps is not None and day_pnl.get(day, 0.0) <= throttle_day_bps: continue
            if len(busy) >= slots: continue
            u = 0.5 if (halfsize_after_loss and last_loss) else 1.0
            r = (ret_at(e, H6) - FEE) * u
            pnl.append(r); day_pnl[day] = day_pnl.get(day, 0.0) + r
            last_loss = r < 0
            busy.append(e["ts"] + H6 * 60_000)
        if not pnl: return {"n": 0}
        n_ = len(pnl); w = sum(1 for x in pnl if x > 0)
        return {"n": n_, "per_month": round(n_ / TM, 1), "wr": round(100 * w / n_, 1),
                "avg": round(sum(pnl) / n_, 1), "total": round(sum(pnl), 0),
                "worst": round(min(pnl), 1), "mdd": mdd(pnl)}
    m9["slot1"] = run_port(allev, 1)
    m9["slot2"] = run_port(allev, 2)
    m9["slot2_throttle150"] = run_port(allev, 2, throttle_day_bps=-150.0)
    m9["slot1_throttle150"] = run_port(allev, 1, throttle_day_bps=-150.0)
    m9["slot1_halfafterloss"] = run_port(allev, 1, halfsize_after_loss=True)
    for k, v in m9.items():
        if v.get("n"):
            print("    %-24s N=%-4d /ay=%-5s WR=%-6s avg=%-7s TOT=%-8s worst=%-8s mdd=%s"
                  % (k, v["n"], v["per_month"], str(v["wr"]) + "%", v["avg"], v["total"], v["worst"], v["mdd"]))
    R["M9"] = m9

    conn.close()
    OUT.mkdir(parents=True, exist_ok=True)
    OJ.write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    lines = ["# S34 Trade Management Gauntlet", "",
             f"> admitted={n} ({len(u200)}x200k + {len(u100)}x100k, no-overlap) {TM:.1f} ay. "
             f"Baseline 6h: avg={base['avg']} mdd={base['mdd']}. {datetime.now(timezone.utc):%Y-%m-%d}", ""]
    def emit(name, sec):
        lines.append(f"## {name}"); lines.append("")
        for k, v in sec.items():
            if isinstance(v, dict) and v.get("n", 0) > 0 and "wr" in v:
                lines.append("- **%s**: N=%d WR=%s%% avg=%+.1f TOT=%s worst=%s mdd=%s mc=%s"
                             % (k, v["n"], v["wr"], v["avg"], v.get("total"), v.get("worst"),
                                v.get("mdd"), v.get("mc_p", "?")))
            elif isinstance(v, dict):
                lines.append(f"- **{k}**: {json.dumps(v, default=str)}")
        lines.append("")
    for name in ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M10", "M8", "M9"):
        emit(name, R[name])
    lines += ["---", "*Script: tools/research_s34_trade_mgmt_gauntlet.py*"]
    OM.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nJSON:{OJ}\nMD:  {OM}\nDone.")


if __name__ == "__main__":
    main()
