"""S34 Mechanism Feature Store — Faz 1.

Her ETH SELL anchor (>=100K, book kapsami 11 Nis'ten itibaren) + saat-eslenmis
kontrol ani icin pre/at/post pencere ozellikleri cikarir ve kompakt SQLite'a yazar.
Sonraki fazlar (taksonomi, pre-cascade prediction, execution opt) 650GB ana DB'ye
dokunmadan bu store uzerinde calisir.

Cikti: reports/research/s34/mechanism_store.sqlite  (tablo: events)
Ayrica ozet: reports/research/s34/S34_MECHANISM_STORE_BUILD.md

Guardrail: ana DB salt-okunur; lookahead yok (ozellikler t<=T0, etiketler t>T0);
tek proses; canliya dokunmaz.
"""
from __future__ import annotations
import json, math, random, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools.research_s34_knowable_anchor_continuation import load_liquidations, load_mark_index, reconstruct_anchors

DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
STORE = OUT_DIR / "mechanism_store.sqlite"
OM = OUT_DIR / "S34_MECHANISM_STORE_BUILD.md"
ANCHOR_MIN = 100_000.0
BOOK_START_PAD_MS = 30 * 60_000     # book kapsaminin ilk 30dk'sini pre-pencere icin atla
HOLD6 = 6 * 3600_000
random.seed(42)

W = {"pre10": (-600_000, 0), "pre1": (-60_000, 0),
     "post1": (0, 60_000), "post5": (0, 300_000), "post10": (0, 600_000)}
OFFSETS = [("2s", 2_000), ("5s", 5_000), ("10s", 10_000), ("30s", 30_000),
           ("1m", 60_000), ("5m", 300_000), ("15m", 900_000)]


def _s(c, q, p=()):
    r = c.execute(q, p).fetchone(); return float(r[0]) if r and r[0] is not None else 0.0


def book_win(c, lo, hi):
    mid = (lo + hi) // 2
    r = c.execute(
        "SELECT COUNT(*), AVG(bid_qty), MIN(bid_qty), AVG(ask_qty), AVG(spread_pct), MAX(spread_pct),"
        " AVG(book_imbalance), AVG(bid_depth_usd),"
        " AVG(CASE WHEN ts_ms< ? THEN book_imbalance END), AVG(CASE WHEN ts_ms>=? THEN book_imbalance END)"
        " FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?",
        (mid, mid, lo, hi)).fetchone()
    if not r or not r[0]:
        return None
    f = lambda x: None if x is None else float(x)
    slope = (f(r[9]) - f(r[8])) if (r[8] is not None and r[9] is not None) else None
    return {"n": int(r[0]), "bidq": f(r[1]), "bidq_min": f(r[2]), "askq": f(r[3]),
            "spread": f(r[4]), "spread_max": f(r[5]), "imb": f(r[6]), "bdep": f(r[7]),
            "imb_slope": slope}


def flow_win(c, lo, hi):
    r = c.execute(
        "SELECT COUNT(*), SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),"
        " SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END), AVG(notional), MAX(notional)"
        " FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?", (lo, hi)).fetchone()
    if not r or not r[0]:
        return None
    buy, sell = float(r[1] or 0), float(r[2] or 0); tot = buy + sell
    return {"n": int(r[0]), "buy": buy, "sell": sell,
            "ofi": ((buy - sell) / tot) if tot > 0 else None,
            "avg_sz": float(r[3]) if r[3] else None, "max_sz": float(r[4]) if r[4] else None,
            "tot": tot}


def fund_at(c, ts):
    r = c.execute("SELECT funding_rate FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND funding_rate IS NOT NULL ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    return float(r[0]) if r and r[0] is not None else None


def nextfund(c, ts):
    r = c.execute("SELECT next_funding_time_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND next_funding_time_ms IS NOT NULL ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    return int(r[0]) if r and r[0] else None


def spot_at(c, ts):
    r = c.execute("SELECT spot_price, ts_ms FROM spot_prices WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    if r and r[0] is not None and (ts - int(r[1])) <= 10 * 60_000:
        return float(r[0])
    return None


def rv_proxy(m, ts):
    px = []
    for k in range(5, -1, -1):
        r = m.at_or_before(ts - k * 60_000)
        if r is None: return None
        px.append(float(r[1]))
    rets = [math.log(px[i + 1] / px[i]) for i in range(5) if px[i] > 0]
    return math.sqrt(sum(x * x for x in rets)) if len(rets) == 5 else None


def mret(m, t0, t1):
    a = m.at_or_before(t0); b = m.at_or_before(t1)
    if not a or not b or float(a[1]) <= 0: return None
    return (float(b[1]) - float(a[1])) / float(a[1]) * 1e4


def extract(conn, m, ts, is_event, rn, nliq, dur_s):
    row = {"ts_ms": ts, "is_event": int(is_event), "rn": rn, "nliq": nliq, "dur_s": dur_s,
           "hour": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).hour,
           "dow": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).weekday()}
    e0 = m.at_or_after(ts)
    if not e0 or float(e0[1]) <= 0: return None
    p0 = float(e0[1])
    # BOOK + FLOW pencereleri
    for wn, (a, b) in W.items():
        bw = book_win(conn, ts + a, ts + b)
        fw = flow_win(conn, ts + a, ts + b)
        if bw is None and wn == "pre10":  # book kapsami disinda -> at
            return None
        for k in ("bidq", "bidq_min", "askq", "spread", "spread_max", "imb", "imb_slope", "bdep", "n"):
            row[f"bk_{wn}_{k}"] = None if bw is None else bw.get(k)
        for k in ("ofi", "avg_sz", "max_sz", "tot", "n"):
            row[f"fl_{wn}_{k}"] = None if fw is None else fw.get(k)
        r_w = mret(m, ts + a, ts + b)
        row[f"px_{wn}_ret"] = r_w
        row[f"fl_{wn}_impact"] = (abs(r_w) / (fw["tot"] / 1e6)) if (fw and fw.get("tot") and fw["tot"] > 0 and r_w is not None) else None
    # pull & refill turevleri
    if row.get("bk_pre10_bidq") and row.get("bk_pre1_bidq_min") is not None and row["bk_pre10_bidq"] > 0:
        row["bk_pull"] = row["bk_pre1_bidq_min"] / row["bk_pre10_bidq"]
    else:
        row["bk_pull"] = None
    if row.get("bk_pre10_bidq") and row.get("bk_post5_bidq") is not None and row["bk_pre10_bidq"] > 0:
        row["bk_refill"] = row["bk_post5_bidq"] / row["bk_pre10_bidq"]
    else:
        row["bk_refill"] = None
    # PRICE gecmis + rv
    row["px_ret_1h"] = mret(m, ts - 3600_000, ts)
    row["px_ret_4h"] = mret(m, ts - 4 * 3600_000, ts)
    row["px_rv"] = rv_proxy(m, ts)
    # LIQ baglami
    row["liq_prebuild_1h"] = int(_s(conn, "SELECT COUNT(*) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<? AND notional>=50000", (ts - 3600_000, ts - 60_000)))
    row["liq_two_sided_1h"] = _s(conn, "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='BUY' AND ts_ms>=? AND ts_ms<?", (ts - 3600_000, ts))
    row["liq_btc_sync"] = _s(conn, "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='BTCUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<?", (ts - 600_000, ts))
    row["liq_sol_sync"] = _s(conn, "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='SOLUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<?", (ts - 600_000, ts))
    # FUNDING seviye + velocity
    f0 = fund_at(conn, ts)
    row["fund_rate"] = f0
    f1 = fund_at(conn, ts - 3600_000); f8 = fund_at(conn, ts - 8 * 3600_000)
    row["fund_vel_1h"] = (f0 - f1) if (f0 is not None and f1 is not None) else None
    row["fund_vel_8h"] = (f0 - f8) if (f0 is not None and f8 is not None) else None
    nf = nextfund(conn, ts)
    row["fund_min_to"] = ((nf - ts) / 60_000) if nf else None
    # BASIS
    sp = spot_at(conn, ts)
    row["basis_spot_bps"] = ((p0 - sp) / sp * 1e4) if sp else None
    sp1 = spot_at(conn, ts - 3600_000)
    m1 = m.at_or_before(ts - 3600_000)
    if sp and sp1 and m1 and float(m1[1]) > 0:
        row["basis_spot_slope"] = row["basis_spot_bps"] - (float(m1[1]) - sp1) / sp1 * 1e4
    else:
        row["basis_spot_slope"] = None
    # POST yapisi (etiket tarafi)
    path30 = m.slice_range(int(e0[0]), ts + 1800_000)
    if path30:
        mn = min(path30, key=lambda r: r[1]); mx = max(path30, key=lambda r: r[1])
        row["post_min_bps_30m"] = (float(mn[1]) - p0) / p0 * 1e4
        row["post_max_bps_30m"] = (float(mx[1]) - p0) / p0 * 1e4
        row["post_t_min_s"] = (int(mn[0]) - ts) / 1000.0
    else:
        row["post_min_bps_30m"] = row["post_max_bps_30m"] = row["post_t_min_s"] = None
    path6h = m.slice_range(int(e0[0]), ts + HOLD6)
    if path6h:
        row["mfe_6h"] = (max(float(r[1]) for r in path6h) - p0) / p0 * 1e4
        row["mae_6h"] = (min(float(r[1]) for r in path6h) - p0) / p0 * 1e4
    else:
        row["mfe_6h"] = row["mae_6h"] = None
    # getiriler T0'dan
    for lbl, hz in (("30m", 1800_000), ("1h", 3600_000), ("2h", 2 * 3600_000), ("6h", HOLD6)):
        row[f"y_{lbl}"] = mret(m, ts, ts + hz)
    # EXECUTION GRID: offset girisli 6h hold
    for lbl, off in OFFSETS:
        ei = m.at_or_after(ts + off)
        if ei and float(ei[1]) > 0:
            xo = m.at_or_before(ts + off + HOLD6)
            row[f"yg_{lbl}"] = ((float(xo[1]) - float(ei[1])) / float(ei[1]) * 1e4) if xo else None
            row[f"ent_{lbl}_bps"] = (float(ei[1]) - p0) / p0 * 1e4
        else:
            row[f"yg_{lbl}"] = row[f"ent_{lbl}_bps"] = None
    # etiketler
    row["lbl_continuation"] = None if row["y_30m"] is None else int(row["y_30m"] < 0)
    row["lbl_profit_long_6h"] = None if row["y_6h"] is None else int(row["y_6h"] - 5.0 > 0)
    return row


def main():
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    print("=== S34 Mechanism Feature Store (Faz 1) ===")
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.execute("PRAGMA cache_size=-200000"); conn.execute("PRAGMA temp_store=MEMORY")
    now = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    bs = conn.execute("SELECT MIN(ts_ms) FROM book_ticker WHERE symbol='ETHUSDT'").fetchone()[0]
    start = int(bs) + BOOK_START_PAD_MS
    print(f"  evren: {datetime.fromtimestamp(start/1000, tz=timezone.utc):%Y-%m-%d} -> simdi (book kapsami)")
    m = load_mark_index(conn, "ETHUSDT")
    ancs = reconstruct_anchors(load_liquidations(conn, "ETHUSDT", "SELL", start, now),
                               bucket_sec=300, min_gap_sec=900, thresholds=(ANCHOR_MIN,), accel_window_sec=30)
    events = [(int(a.anchor_ts_ms), float(a.running_notional), int(getattr(a, "liq_count", 0) or 0))
              for a in ancs if float(a.running_notional) >= ANCHOR_MIN]
    print(f"  anchor >=100K: {len(events)}")
    rows = []
    for i, (ts, rn, nl) in enumerate(events):
        r = extract(conn, m, ts, True, rn, nl, None)
        if r: rows.append(r)
        if (i + 1) % 100 == 0: print(f"    event {i+1}/{len(events)}")
    n_ev = len(rows)
    # kontroller: saat-eslenmis, cascade'siz anlar
    ev_hours = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc).hour for ts, _, _ in events]
    n_ctl = 0; tries = 0
    while n_ctl < n_ev and tries < n_ev * 30:
        tries += 1
        h = random.choice(ev_hours)
        day = random.randint(0, (now - start) // 86_400_000 - 1)
        ts = start + day * 86_400_000
        ts = ts - (ts % 86_400_000) + h * 3600_000 + random.randint(0, 3_599_000)
        if ts <= start or ts >= now - HOLD6: continue
        near = _s(conn, "SELECT COALESCE(MAX(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<?", (ts - 1800_000, ts + 600_000))
        if near >= 50_000: continue
        r = extract(conn, m, ts, False, 0.0, 0, None)
        if r: rows.append(r); n_ctl += 1
        if n_ctl % 100 == 0: print(f"    kontrol {n_ctl}/{n_ev}")
    print(f"  toplam satir: {len(rows)} (event {n_ev} + kontrol {n_ctl})")
    # store yaz
    STORE.unlink(missing_ok=True)
    out = sqlite3.connect(STORE)
    cols = sorted({k for r in rows for k in r})
    defs = ", ".join(f'"{c}" REAL' if c not in ("ts_ms", "is_event", "hour", "dow", "nliq") else f'"{c}" INTEGER' for c in cols)
    out.execute(f"CREATE TABLE events ({defs})")
    for r in rows:
        ks = sorted(r.keys())
        out.execute(f'INSERT INTO events ({",".join(chr(34)+k+chr(34) for k in ks)}) VALUES ({",".join("?"*len(ks))})',
                    [r[k] for k in ks])
    out.commit()
    out.execute("CREATE INDEX idx_ev_ts ON events(ts_ms)")
    out.commit(); out.close(); conn.close()
    OM.write_text("\n".join([
        "# S34 Mechanism Store Build", "",
        f"> {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        f"- Evren: book kapsami ({datetime.fromtimestamp(start/1000, tz=timezone.utc):%Y-%m-%d}) -> simdi",
        f"- Event (ETH SELL >=100K anchor): **{n_ev}**",
        f"- Kontrol (saat-eslenmis, cascade'siz): **{n_ctl}**",
        f"- Kolon sayisi: {len(cols)}",
        f"- Store: `{STORE.relative_to(ROOT)}`",
        "", "Pencereler: pre10/pre1/post1/post5/post10 — book(L1)/flow/impact;",
        "pull/refill turevleri; funding velocity; spot basis; execution grid (2s..15m);",
        "etiketler: continuation, profit_long_6h.", "",
        "*Script: tools/s34_mechanism_feature_store.py*"]), encoding="utf-8")
    print(f"  store: {STORE}\n  ozet: {OM}\nDone.")


if __name__ == "__main__":
    main()
