r"""LANE C, round 53 -- sweep this lane for the unit defect, as I told lane B to do and did not.

C-T52's `to B` said: "when a unit or metric defect is established, what ELSE in the atlas was
computed in that unit?" I gave that advice and applied it to exactly one number, R_inf. Nearly
every quantity this lane has published is in the aggTrade clock.

AND THE CORPUS SPECIFIES THE CLOCK EXPLICITLY, TWICE. Read with corpus_text_v1:

    sec. 11.4  "Throughout this section, we again choose to work in MARKET ORDER TIME, whereby we
                advance t by 1 FOR EACH MARKET ORDER ARRIVAL."
    ch. 13     "we restrict our attention to MARKET ORDER TIME, in which we advance the clock by a
                single unit FOR EACH MARKET ORDER ARRIVAL."

Those are the two chapters this lane took the aggregate-impact collapse, the propagator and the
response function from. C-T50 measured that 10.75% of BTC and 13.81% of ETH market orders occupy
SEVERAL aggTrades. So the clock I used is not the clock the book specifies, and that is a stated
specification I did not follow rather than a subtlety.

WHAT THIS ROUND DOES. Not re-run twenty studies -- measure the EXPOSURE of the lane's core
quantities, so the estate knows which results carry the defect and which are invariant to it:

    chi          sign-memory exponent          C-T23 / C-T28 / C-T31 / C-T39
    H            price exponent                C-T29 / C-T38
    h(q) fall    multifractal curvature        C-T41 / C-T46
    delta        aggregate-impact concavity    C-T33

Each is computed in BOTH clocks with the identical estimator, and each carries its own shuffle
null in its own clock, because the two clocks have different N and a null computed in one does not
transfer to the other.

PREDICTION, stated before measurement. Exponents of partial sums should be MORE robust than levels
in bps: a walk contributes a fixed-length run of same-signed steps, which perturbs short lags and
should wash out of a scaling exponent fitted over a decade. R_inf, a level, halved (C-T52). If the
exponents move as much as the level did, far more of this lane falls than one number.

SAMPLE BY ARTIFACT PATH: data/microstructure_02.db :: agg_trades, first 2,000,000 rows by ts_ms
per symbol.
THRESHOLD: event = consecutive aggTrades with identical ts_ms AND side. C-T52 already swept the
looser ts_ms-only boundary and found it not load-bearing, so it is not re-swept here.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "microstructure_02.db"
OUT = ROOT / "reports" / "atlas"
SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
NROWS = 2_000_000
WT = (20, 50, 100, 200, 500, 1000)
QS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
NULL_REPS = 20
SEED = 20260827


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def slope(T, S):
    x, y = np.log(np.asarray(T, float)), np.log(np.asarray(S, float))
    A = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(b[1])


def exp_of(x, wt=WT):
    T, S = [], []
    for t in wt:
        m = len(x) // t
        if m < 200:
            continue
        T.append(float(t))
        S.append(float(np.std(x[:m * t].reshape(m, t).sum(axis=1), ddof=1)))
    return slope(T, S) if len(T) >= 4 else float("nan")


def zeta_fall(x):
    n = len(x)
    Ts = np.unique(np.round(np.geomspace(8, min(4000, n // 300), 14)).astype(int))
    h = []
    for q in QS:
        T, M = [], []
        for t in Ts:
            m = n // t
            if m < 300:
                continue
            s = np.abs(x[:m * t].reshape(m, t).sum(axis=1))
            mom = float(np.mean(s ** q))
            if mom > 0:
                T.append(float(t))
                M.append(mom)
        if len(T) >= 6:
            h.append(slope(T, M) / q)
    return (h[0] - h[-1]) if len(h) >= 2 else float("nan")


def delta_of(lp, flow, T=100, nbin=21):
    n = len(lp)
    m = (n - 1) // T
    i0 = np.arange(1, m) * T
    dv = flow[:m * T].reshape(m, T).sum(axis=1)[1:]
    dp = (lp[i0 + T - 1] - lp[i0 - 1]) * 1e4
    q = np.quantile(dv, np.linspace(0, 1, nbin + 1))
    q[0] -= 1e-9
    q[-1] += 1e-9
    idx = np.clip(np.searchsorted(q, dv, side="right") - 1, 0, nbin - 1)
    xs, ys = [], []
    for b in range(nbin):
        mk = idx == b
        if mk.sum() < 30:
            continue
        mv, mr = float(dv[mk].mean()), float(dp[mk].mean())
        if mv == 0 or mr == 0 or np.sign(mv) != np.sign(mr):
            continue
        xs.append(np.log(abs(mv)))
        ys.append(np.log(abs(mr)))
    if len(xs) < 5:
        return float("nan")
    A = np.column_stack([np.ones(len(xs)), np.asarray(xs)])
    b, *_ = np.linalg.lstsq(A, np.asarray(ys), rcond=None)
    return float(b[1])


def collapse(ts, px, bm):
    ch = np.empty(len(ts), dtype=bool)
    ch[0] = True
    ch[1:] = (ts[1:] != ts[:-1]) | (bm[1:] != bm[:-1])
    st = np.flatnonzero(ch)
    en = np.append(st[1:] - 1, len(ts) - 1)
    return en, st


def main() -> int:
    rng = np.random.default_rng(SEED)
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per = {}
    try:
        for sym in SYMS:
            a = np.array(con.execute(
                "select ts_ms,price,notional,is_buyer_maker from agg_trades where symbol=? "
                "order by ts_ms limit ?", (sym, NROWS)).fetchall(), dtype=np.float64)
            ts, px, vol, bm = a[:, 0], a[:, 1], a[:, 2], a[:, 3]

            # --- aggTrade clock
            lp_a = np.log(px)
            eps_a = np.where(bm > 0.5, -1.0, 1.0)
            fl_a = eps_a * vol
            r_a = np.empty_like(lp_a)
            r_a[0] = 0.0
            r_a[1:] = np.diff(lp_a)

            # --- market-order clock: one row per (ts_ms, side) run
            en, st = collapse(ts, px, bm)
            lp_e = np.log(px[en])
            eps_e = np.where(bm[st] > 0.5, -1.0, 1.0)
            fl_e = np.array([vol[s:e + 1].sum() for s, e in zip(st, en)]) * eps_e
            r_e = np.empty_like(lp_e)
            r_e[0] = 0.0
            r_e[1:] = np.diff(lp_e)

            def pack(eps, r, lp, fl, label):
                chi = exp_of(eps)
                H = exp_of(r)
                fall = zeta_fall(r)
                dl = delta_of(lp, fl)
                nchi = float(np.mean([exp_of(eps[rng.permutation(len(eps))])
                                      for _ in range(NULL_REPS)]))
                nH = float(np.mean([exp_of(r[rng.permutation(len(r))])
                                    for _ in range(NULL_REPS)]))
                return {"clock": label, "n": int(len(eps)),
                        "chi_sign": round(chi, 4), "chi_null": round(nchi, 4),
                        "H_price": round(H, 4), "H_null": round(nH, 4),
                        "hq_fall": round(fall, 4), "delta_T100": round(dl, 4)}

            A_ = pack(eps_a, r_a, lp_a, fl_a, "aggTrade")
            E_ = pack(eps_e, r_e, lp_e, fl_e, "market_order")
            per[sym] = {
                "aggTrade": A_, "market_order": E_,
                "shift": {k: round(E_[k] - A_[k], 4)
                          for k in ("chi_sign", "H_price", "hq_fall", "delta_T100")},
                "n_ratio": round(E_["n"] / A_["n"], 4),
            }
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    art = {"study": "C-T53", "lane": "C", "utc": _utc(),
           "corpus_specification": ("Bouchaud sec. 11.4 and ch. 13 both state the clock advances "
                                    "by one unit FOR EACH MARKET ORDER ARRIVAL"),
           "defect": ("this lane used the aggTrade clock; 10.75% of BTC and 13.81% of ETH market "
                      "orders occupy several aggTrades (C-T50)"),
           "sample": ("data/microstructure_02.db :: agg_trades, first 2,000,000 rows by ts_ms "
                      "per symbol"),
           "prediction": ("exponents of partial sums should be more robust than levels in bps; "
                          "R_inf halved in C-T52"),
           "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C53_UNIT_EXPOSURE_SWEEP_V1.json").write_text(json.dumps(art, indent=2),
                                                         encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    w("%-9s %-14s %9s %10s %10s %10s %11s %11s" % (
        "sym", "clock", "n", "chi", "chi null", "H", "h(q) fall", "delta T100"))
    for s in SYMS:
        for k in ("aggTrade", "market_order"):
            v = per[s][k]
            w("%-9s %-14s %9d %10s %10s %10s %11s %11s" % (
                s, v["clock"], v["n"], v["chi_sign"], v["chi_null"], v["H_price"],
                v["hq_fall"], v["delta_T100"]))
        sh = per[s]["shift"]
        w("%-9s %-14s n ratio %-6s  SHIFT  chi %+0.4f   H %+0.4f   h(q) %+0.4f   delta %+0.4f" % (
            s, "", per[s]["n_ratio"], sh["chi_sign"], sh["H_price"], sh["hq_fall"],
            sh["delta_T100"]))
        w("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
