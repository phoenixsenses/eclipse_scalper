# -*- coding: utf-8 -*-
"""C-T33 -- THE NEGATIVE-LAG RESPONSE: DOES THE PRICE MOVE BEFORE THE TRADE?

Sec 14.2 names the propagator model's second weakness and it is the one this lane's finding
points at:

    "Another weakness of the propagator model concerns the shape of the negative-lag response
     function, i.e. R(l) for l < 0.  ... only the positive side of R(l) is needed to calibrate
     G(l).  Once this is known, R(l < 0) can be predicted without any additional parameters.
     ... In all cases, the empirical R(l < 0) lies ABOVE the theoretical prediction of the
     propagator model."

with a footnote that carries a DIRECTION on the tick axis:

    "The discrepancy tends to be smaller for small-tick assets than for large-tick assets."

C-T30 measured that 76-83 percent of the lag-dependent diffusion coefficient is carried by
intervals containing NO TRADE, and that this component is autocorrelated.  If the price moves
first and trades follow it, R(l < 0) should be large -- trades arrive after the move, not
before it.  So the two sources make OPPOSITE ordering predictions:

    corpus footnote        discrepancy smaller for small-tick   =>  SOL  >  BTC, ETH
    C-T30's non-trade share  0.821 (BTC) 0.826 (ETH) 0.761 (SOL) =>  BTC, ETH  >  SOL

WHAT IS AND IS NOT MEASURED.  The book's statement is about the residual between the
empirical R(l<0) and a propagator PREDICTION, and forming that residual needs G(l), which
needs C(l)'s LEVEL -- convention-dependent on this feed (H-T8: C(1) flips sign under
merging).  So the residual is not available here and is not attempted.

What IS available, and is all that is claimed, is the RATIO

    A(l) := R(-l) / R(+l)

compared ACROSS symbols.  Both sides carry the same sign-autocorrelation structure, so the
ratio divides much of it out, and the cross-symbol ORDERING is the quantity the two
predictions disagree about.  It is a comparative statistic, not a decomposition, and no
attempt is made to read A(l) as "the fraction of the move that anticipates the trade".

Null: trade signs shuffled within (day, hour), which sends both R(+l) and R(-l) to zero and
so calibrates the ratio's noise floor.  De-fragmented at 200 ms throughout (H-T5).

ESTIMATION.  Ceiling: MECHANISM_CHARACTERISATION.

  python -m tools.ct33_the_negative_lag_response --i-have-approval
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np

from tools import h2_response_shape_driver as H2
from tools import s66_cascade_process_driver as D
from tools import hb4_is_a_liquidation_special as B4

OUT = "reports/atlas"
DAYS = ("2026-08-07", "2026-08-08", "2026-08-09", "2026-08-10",
        "2026-08-11", "2026-08-12", "2026-08-13")
LAGS = (1, 2, 5, 10, 20, 50, 100)
MERGE_MS = 200
N_SHUF = 12
RNG_SEED = 20260827
NON_TRADE_SHARE_CT30 = {"BTCUSDT": 0.821, "ETHUSDT": 0.826, "SOLUSDT": 0.761}
TICK_CLASS = {"BTCUSDT": "SMALL", "ETHUSDT": "SMALL", "SOLUSDT": "LARGE"}


def responses(eps, mid, lags):
    """R(+l) = <eps_t (m_{t+l} - m_t)>  and  R(-l) = <eps_t (m_t - m_{t-l})>, in bps"""
    out = {}
    n = len(eps)
    for L in lags:
        if n <= 2 * L + 10:
            continue
        fwd = eps[:n - L] * (mid[L:] / mid[:n - L] - 1.0) * 1e4
        bwd = eps[L:] * (mid[L:] / mid[:n - L] - 1.0) * 1e4
        out[L] = (float(np.sum(fwd)), len(fwd), float(np.sum(bwd)), len(bwd))
    return out


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    rng = np.random.default_rng(RNG_SEED)
    res = {"days": list(DAYS), "lags": list(LAGS), "merge_ms": MERGE_MS,
           "book": "Sec 14.2: empirical R(l<0) lies ABOVE the propagator prediction; "
                   "footnote: discrepancy smaller for small-tick",
           "competing_predictions": {
               "corpus_footnote": "SOL > BTC, ETH",
               "CT30_non_trade_share": "BTC, ETH > SOL"},
           "non_trade_share_CT30": NON_TRADE_SHARE_CT30, "tick_class": TICK_CLASS,
           "per_symbol": {}, "ceiling": "MECHANISM_CHARACTERISATION"}

    for sym in H2.SYMBOLS:
        acc, shuf_acc = {}, [dict() for _ in range(N_SHUF)]
        for day in DAYS:
            d0 = dt.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
            lo = int(d0.timestamp() * 1000)
            hi = int((d0 + dt.timedelta(days=1)).timestamp() * 1000)
            con = D._con()
            rows = con.execute(
                "SELECT ts_ms,bid_price,ask_price FROM book_ticker WHERE symbol=? "
                "AND ts_ms>=? AND ts_ms<? AND bid_price>0 AND ask_price>0 ORDER BY ts_ms",
                (sym, lo - 2000, hi + 2000)).fetchall()
            con.close()
            if len(rows) < 10000:
                continue
            bts = np.array([r[0] for r in rows], np.int64)
            mid = np.array([0.5 * (r[1] + r[2]) for r in rows], float)
            del rows
            ts, px, eps, qty = B4.load_raw_with_qty(sym, (day,))
            new = np.concatenate([[True], (ts[1:] != ts[:-1]) | (eps[1:] != eps[:-1])])
            idx = np.flatnonzero(new)
            ots0, oeps0 = ts[idx], eps[idx]
            del ts, px, eps, qty
            keep = np.concatenate([[True], (np.diff(ots0) >= MERGE_MS)
                                   | (oeps0[1:] != oeps0[:-1])])
            j = np.flatnonzero(keep)
            ots, oeps = ots0[j], oeps0[j].astype(float)
            ib = np.searchsorted(bts, ots, side="left") - 1
            ok = ib >= 0
            ots, oeps, ib = ots[ok], oeps[ok], ib[ok]
            m = mid[ib]

            for L, v in responses(oeps, m, LAGS).items():
                a = acc.get(L, (0.0, 0, 0.0, 0))
                acc[L] = (a[0] + v[0], a[1] + v[1], a[2] + v[2], a[3] + v[3])

            hour = (ots // 3600000).astype(np.int64)
            _, inv = np.unique(hour, return_inverse=True)
            slots = np.argsort(inv, kind="stable")
            for s in range(N_SHUF):
                order = np.lexsort((rng.random(len(oeps)), inv))
                e2 = np.empty_like(oeps)
                e2[slots] = oeps[order]
                for L, v in responses(e2, m, LAGS).items():
                    a = shuf_acc[s].get(L, (0.0, 0, 0.0, 0))
                    shuf_acc[s][L] = (a[0] + v[0], a[1] + v[1], a[2] + v[2], a[3] + v[3])
            del bts, mid

        if not acc:
            continue
        Rp = {L: acc[L][0] / acc[L][1] for L in acc}
        Rm = {L: acc[L][2] / acc[L][3] for L in acc}
        ratio = {L: (Rm[L] / Rp[L]) if Rp[L] != 0 else None for L in acc}
        sh_ratio = {L: [] for L in acc}
        for s in range(N_SHUF):
            for L in acc:
                if L in shuf_acc[s]:
                    a = shuf_acc[s][L]
                    p, mm = a[0] / a[1], a[2] / a[3]
                    if p != 0:
                        sh_ratio[L].append(mm / p)
        r_med = float(np.median([ratio[L] for L in ratio if ratio[L] is not None]))
        out = {"R_plus": {str(L): Rp[L] for L in sorted(Rp)},
               "R_minus": {str(L): Rm[L] for L in sorted(Rm)},
               "ratio_Rminus_over_Rplus": {str(L): ratio[L] for L in sorted(ratio)},
               "ratio_median": r_med,
               "null_ratio": {str(L): ({"mean": float(np.mean(sh_ratio[L])),
                                        "sd": float(np.std(sh_ratio[L]))}
                                       if sh_ratio[L] else None) for L in sorted(sh_ratio)},
               "tick_class": TICK_CLASS[sym],
               "non_trade_share_CT30": NON_TRADE_SHARE_CT30[sym]}
        res["per_symbol"][sym] = out
        print("=== %s  (%s tick, non-trade share %.3f)"
              % (sym, TICK_CLASS[sym], NON_TRADE_SHARE_CT30[sym]), flush=True)
        print("    R(+l): " + "  ".join("%d %+.4f" % (L, Rp[L]) for L in sorted(Rp)),
              flush=True)
        print("    R(-l): " + "  ".join("%d %+.4f" % (L, Rm[L]) for L in sorted(Rm)),
              flush=True)
        print("    ratio: " + "  ".join("%d %.3f" % (L, ratio[L]) for L in sorted(ratio)
                                        if ratio[L] is not None)
              + "   median %.3f" % r_med, flush=True)
        nz = [out["null_ratio"][str(L)] for L in sorted(sh_ratio)
              if out["null_ratio"][str(L)]]
        if nz:
            print("    null ratio (signs shuffled): mean %+.3f  sd %.3f"
                  % (float(np.mean([v["mean"] for v in nz])),
                     float(np.mean([v["sd"] for v in nz]))), flush=True)

    P = res["per_symbol"]
    if len(P) == 3:
        order = sorted(P, key=lambda s: -P[s]["ratio_median"])
        res["ordering_by_ratio"] = order
        small = [s for s in P if TICK_CLASS[s] == "SMALL"]
        large = [s for s in P if TICK_CLASS[s] == "LARGE"]
        sm = float(np.mean([P[s]["ratio_median"] for s in small]))
        lg = float(np.mean([P[s]["ratio_median"] for s in large]))
        res["small_tick_mean"] = sm
        res["large_tick_mean"] = lg
        res["supports"] = ("CT30_non_trade_share" if sm > lg else "corpus_footnote")
        print("ORDERING by R(-l)/R(+l): %s" % order, flush=True)
        print("    small-tick mean %.3f   large-tick mean %.3f   => supports %s"
              % (sm, lg, res["supports"]), flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "CT33_NEGATIVE_LAG_V1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print("written", flush=True)


if __name__ == "__main__":
    main()
