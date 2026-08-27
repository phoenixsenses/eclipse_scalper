# -*- coding: utf-8 -*-
"""C-T38 -- SEC 14.4's "ABOUT TEN EVENTS", MEASURED IN THE BOOK'S OWN EVENT CLOCK.

C-T37 tried this and was void for three reasons, two of them mine (ERR-HU-037, ERR-HU-038):

    the trade indicator was built from the 200 ms MERGED order series, so every merged-away
    trade left its interval labelled trade-free -- the symptom being T(k) constant in k;
    and k counted L1 BOOK UPDATES, about 160 per second on BTC, so k = 60 was 0.4 seconds
    while the corpus's "ten events" means ten LOB events in a market with a few per second.

Both are fixed here, and the fix for the second is not an arbitrary rescaling: the corpus's
event clock is market orders + limit orders + cancellations, which is exactly the INTERTWINED
series C-T35 built.  So k is counted in intertwined events, the corpus's own unit.

    "After a market order of either type MO0 or MO1, the flow of limit orders and
     cancellations first pushes the price in the same direction for about ten events, then
     reverses and opposes the market order flow, PARTICULARLY THROUGH LO-TYPE EVENTS, which
     correspond to liquidity refill."

The last clause is testable too, so the book-carried accumulation is split by event type:
AL (price-improving quotes -- the refill) against CX (quotes retreating).  The corpus says the
reversal comes particularly through the AL side.

MEASURED, from every price-moving market order, over the following k intertwined events:
    B_AL(k), B_CX(k)   mid change across intervals with NO trade, by which quote event it was
    T(k)               mid change across intervals containing a trade, from the UNMERGED
                       order series
all signed by the market order's direction, in bps.

NULL on the LOCATION as well as the level: market-order signs shuffled within (day, hour),
8 draws, and the argmax position of each draw reported -- an argmax always exists, and this
lane has published four verdicts its own numbers did not license.

ESTIMATION.  Ceiling: MECHANISM_CHARACTERISATION.

  python -m tools.ct38_ten_events_in_the_books_own_clock --i-have-approval
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
KMAX = 100
MERGE_MS = 200
N_SHUF = 8
RNG_SEED = 20260827
BOOK_TURN = 10


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    rng = np.random.default_rng(RNG_SEED)
    res = {"days": list(DAYS), "kmax": KMAX, "clock": "intertwined LOB events (MO + LO + CA)",
           "book": "Sec 14.4: same direction about ten events, then reverses, "
                   "particularly through LO-type events",
           "book_turn": BOOK_TURN, "fixes": ["ERR-HU-037", "ERR-HU-038"],
           "per_symbol": {}, "ceiling": "MECHANISM_CHARACTERISATION"}

    for sym in H2.SYMBOLS:
        accAL = np.zeros(KMAX + 1)
        accCX = np.zeros(KMAX + 1)
        accT = np.zeros(KMAX + 1)
        shuf = np.zeros((N_SHUF, KMAX + 1))
        n_ev = 0
        for day in DAYS:
            d0 = dt.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
            lo = int(d0.timestamp() * 1000)
            hi = int((d0 + dt.timedelta(days=1)).timestamp() * 1000)
            con = D._con()
            rows = con.execute(
                "SELECT ts_ms,bid_price,ask_price FROM book_ticker WHERE symbol=? "
                "AND ts_ms>=? AND ts_ms<? AND bid_price>0 AND ask_price>0 ORDER BY ts_ms",
                (sym, lo, hi)).fetchall()
            con.close()
            if len(rows) < 10000:
                continue
            bts = np.array([r[0] for r in rows], np.int64)
            bid = np.array([r[1] for r in rows], float)
            ask = np.array([r[2] for r in rows], float)
            del rows
            mid = 0.5 * (bid + ask)

            ts, px, eps, qty = B4.load_raw_with_qty(sym, (day,))
            new = np.concatenate([[True], (ts[1:] != ts[:-1]) | (eps[1:] != eps[:-1])])
            idx = np.flatnonzero(new)
            raw_t, raw_e = ts[idx], eps[idx]           # UNMERGED -- ERR-HU-037 fix
            del ts, px, eps, qty
            keep = np.concatenate([[True], (np.diff(raw_t) >= MERGE_MS)
                                   | (raw_e[1:] != raw_e[:-1])])
            mg = np.flatnonzero(keep)
            mo_t, mo_e = raw_t[mg], raw_e[mg].astype(float)

            dm = np.zeros(len(bts))
            dm[1:] = (mid[1:] / mid[:-1] - 1.0) * 1e4
            ntr = np.zeros(len(bts))
            ntr[1:] = (np.searchsorted(raw_t, bts[1:], side="right")
                       - np.searchsorted(raw_t, bts[:-1], side="right"))
            tr = ntr > 0
            da, db = np.diff(ask), np.diff(bid)
            is_AL = np.zeros(len(bts), bool)
            is_CX = np.zeros(len(bts), bool)
            is_AL[1:] = (~tr[1:]) & (((da < 0) & (db == 0)) | ((db > 0) & (da == 0)))
            is_CX[1:] = (~tr[1:]) & (((da > 0) & (db == 0)) | ((db < 0) & (da == 0)))

            # the intertwined event clock: every MO plus every AL/CX book event
            ev_t = np.concatenate([mo_t, bts[is_AL], bts[is_CX]])
            ev_t.sort()
            if len(ev_t) < 1000:
                continue
            # cumulative price legs indexed by BOOK index
            cAL = np.concatenate([[0.0], np.cumsum(np.where(is_AL, dm, 0.0))])
            cCX = np.concatenate([[0.0], np.cumsum(np.where(is_CX, dm, 0.0))])
            cT = np.concatenate([[0.0], np.cumsum(np.where(tr, dm, 0.0))])

            ib = np.searchsorted(bts, mo_t, side="left") - 1
            ia = np.searchsorted(bts, mo_t, side="right")
            ok = (ib >= 0) & (ia < len(bts))
            moved = np.zeros(len(mo_t), bool)
            moved[ok] = mid[ia[ok]] != mid[ib[ok]]
            sel = ok & moved
            if sel.sum() < 500:
                continue
            t0 = mo_t[sel]
            e0 = mo_e[sel]
            # position of each anchor on the EVENT clock
            p0 = np.searchsorted(ev_t, t0, side="right")
            good = (p0 + KMAX) < len(ev_t)
            t0, e0, p0 = t0[good], e0[good], p0[good]
            if len(t0) < 500:
                continue
            n_ev += len(t0)
            # map event-clock position k -> book index, then read the cumulative legs
            for k in range(1, KMAX + 1):
                tk = ev_t[p0 + k - 1]
                jk = np.searchsorted(bts, tk, side="right")
                j0 = np.searchsorted(bts, t0, side="right")
                accAL[k] += float(np.sum(e0 * (cAL[jk] - cAL[j0])))
                accCX[k] += float(np.sum(e0 * (cCX[jk] - cCX[j0])))
                accT[k] += float(np.sum(e0 * (cT[jk] - cT[j0])))

            hour = (t0 // 3600000).astype(np.int64)
            _, inv = np.unique(hour, return_inverse=True)
            slots = np.argsort(inv, kind="stable")
            j0 = np.searchsorted(bts, t0, side="right")
            for s in range(N_SHUF):
                order = np.lexsort((rng.random(len(e0)), inv))
                e2 = np.empty_like(e0)
                e2[slots] = e0[order]
                for k in range(1, KMAX + 1):
                    jk = np.searchsorted(bts, ev_t[p0 + k - 1], side="right")
                    shuf[s, k] += float(np.sum(e2 * ((cAL[jk] - cAL[j0])
                                                     + (cCX[jk] - cCX[j0]))))
            del bts, bid, ask, mid

        if n_ev == 0:
            continue
        AL, CX, T = accAL / n_ev, accCX / n_ev, accT / n_ev
        BK = AL + CX
        SH = shuf / n_ev
        kpk = int(np.argmax(BK[1:]) + 1)
        neg = [k for k in range(1, KMAX + 1) if BK[k] < BK[kpk] * 0.0]
        first_fall = next((k for k in range(kpk + 1, KMAX + 1) if BK[k] < BK[kpk]), None)
        null_pk = [int(np.argmax(SH[s, 1:]) + 1) for s in range(N_SHUF)]
        out = {"n_anchors": n_ev,
               "B_book": {str(k): float(BK[k]) for k in range(1, KMAX + 1)},
               "B_AL": {str(k): float(AL[k]) for k in range(1, KMAX + 1)},
               "B_CX": {str(k): float(CX[k]) for k in range(1, KMAX + 1)},
               "T_trade": {str(k): float(T[k]) for k in range(1, KMAX + 1)},
               "argmax_k": kpk, "peak": float(BK[kpk]),
               "first_k_below_peak": first_fall,
               "null_argmax_positions": null_pk,
               "null_peak_mean": float(np.mean([SH[s, int(np.argmax(SH[s, 1:]) + 1)]
                                                for s in range(N_SHUF)])),
               "book_turn": BOOK_TURN}
        res["per_symbol"][sym] = out
        ks = [1, 2, 3, 5, 8, 10, 12, 15, 20, 30, 50, 100]
        print("=== %s   anchors %d" % (sym, n_ev), flush=True)
        print("    B_book(k): " + "  ".join("%d %+.4f" % (k, BK[k]) for k in ks), flush=True)
        print("    B_AL(k)  : " + "  ".join("%d %+.4f" % (k, AL[k]) for k in ks), flush=True)
        print("    B_CX(k)  : " + "  ".join("%d %+.4f" % (k, CX[k]) for k in ks), flush=True)
        print("    T(k)     : " + "  ".join("%d %+.4f" % (k, T[k]) for k in ks), flush=True)
        print("    argmax k = %d (book ~%d)  peak %+.4f  first k below peak = %s"
              % (kpk, BOOK_TURN, BK[kpk], first_fall), flush=True)
        print("    NULL argmax positions %s   null peak %+.5f"
              % (null_pk, out["null_peak_mean"]), flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "CT38_TEN_EVENTS_V1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print("written", flush=True)


if __name__ == "__main__":
    main()
