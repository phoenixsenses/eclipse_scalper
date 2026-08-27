# -*- coding: utf-8 -*-
"""C-T37 -- DOES THE BOOK FLOW TURN AGAINST THE TRADE AFTER ABOUT TEN EVENTS?

Sec 14.4 lists four patterns; C-T34 confirmed one of them (price-improving quotes attract the
opposite side, 6/6 at 0.5 s).  This tests the one that carries a NUMBER:

    "After a market order of either type MO0 or MO1, the flow of limit orders and
     cancellations FIRST PUSHES THE PRICE IN THE SAME DIRECTION FOR ABOUT TEN EVENTS, then
     REVERSES and opposes the market order flow, particularly through LO-type events, which
     correspond to liquidity refill."

It matters here because C-T30 measured that 76-83 percent of the lag-dependent diffusion
coefficient is carried by intervals containing NO trade.  If that book-carried component
first amplifies a trade and then opposes it, the turning point is the timescale on which the
refill takes over -- and this lane has never measured it, only its intensity (H-T6's AM->AL
= 5.65 / 4.14 / 7.58).

ESTIMAND.  For every price-moving market order at book index i0 with sign eps, accumulate the
BOOK-CARRIED mid change -- the change across book updates whose preceding interval contained
NO trade -- over the following k book events, signed by eps:

    B(k) = < eps * sum_{j=1..k, no trade in (j-1, j]} (m_j - m_{j-1}) >   in bps

The corpus predicts B(k) rises, peaks near k = 10, then falls and goes negative.  The TURNING
POINT is argmax B(k), and it is reported per symbol against the book's "about ten".

Also reported for contrast, from the same events: the TRADE-carried accumulation T(k), which
the corpus does not make this claim about.

NULL: market-order signs shuffled within (day, hour), 8 draws, which sends B(k) to zero at
every k and so calibrates both the peak height and the position of the argmax under noise --
an argmax always exists, so a null on the LOCATION is required, not just on the level.  This
lane has published four verdicts that its own numbers did not license; a bare argmax without
a location null would be the fifth.

De-fragmented at 200 ms (H-T5).  ESTIMATION.  Ceiling: MECHANISM_CHARACTERISATION.

  python -m tools.ct37_when_does_the_book_turn_against_the_trade --i-have-approval
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
KMAX = 60
MERGE_MS = 200
N_SHUF = 8
RNG_SEED = 20260827
BOOK_TURNING_POINT = 10


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    rng = np.random.default_rng(RNG_SEED)
    res = {"days": list(DAYS), "kmax": KMAX, "merge_ms": MERGE_MS,
           "book": "Sec 14.4: same direction for about ten events, then reverses",
           "book_turning_point": BOOK_TURNING_POINT,
           "per_symbol": {}, "ceiling": "MECHANISM_CHARACTERISATION"}

    for sym in H2.SYMBOLS:
        accB = np.zeros(KMAX + 1)
        accT = np.zeros(KMAX + 1)
        nB = 0
        shufB = np.zeros((N_SHUF, KMAX + 1))
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

            dm = np.zeros(len(bts))
            dm[1:] = (mid[1:] / mid[:-1] - 1.0) * 1e4
            ntr = np.zeros(len(bts))
            ntr[1:] = (np.searchsorted(ots, bts[1:], side="right")
                       - np.searchsorted(ots, bts[:-1], side="right"))
            tr = ntr > 0
            cB = np.concatenate([[0.0], np.cumsum(np.where(~tr, dm, 0.0))])
            cT = np.concatenate([[0.0], np.cumsum(np.where(tr, dm, 0.0))])

            ib = np.searchsorted(bts, ots, side="left") - 1
            ia = np.searchsorted(bts, ots, side="right")
            ok = (ib >= 0) & (ia < len(bts)) & (ia + KMAX < len(bts))
            moved = np.zeros(len(ots), bool)
            moved[ok] = mid[ia[ok]] != mid[ib[ok]]
            sel = ok & moved
            i0 = ia[sel]
            e0 = oeps[sel]
            if len(i0) < 500:
                continue
            nB += len(i0)
            for k in range(1, KMAX + 1):
                accB[k] += float(np.sum(e0 * (cB[i0 + k] - cB[i0])))
                accT[k] += float(np.sum(e0 * (cT[i0 + k] - cT[i0])))

            hour = (ots[sel] // 3600000).astype(np.int64)
            _, inv = np.unique(hour, return_inverse=True)
            slots = np.argsort(inv, kind="stable")
            for s in range(N_SHUF):
                order = np.lexsort((rng.random(len(e0)), inv))
                e2 = np.empty_like(e0)
                e2[slots] = e0[order]
                for k in range(1, KMAX + 1):
                    shufB[s, k] += float(np.sum(e2 * (cB[i0 + k] - cB[i0])))
            del bts, mid

        if nB == 0:
            continue
        B = accB / nB
        T = accT / nB
        SB = shufB / nB
        kpeak = int(np.argmax(B[1:]) + 1)
        neg = [k for k in range(1, KMAX + 1) if B[k] < 0]
        kzero = neg[0] if neg else None
        peaks_null = [int(np.argmax(SB[s, 1:]) + 1) for s in range(N_SHUF)]
        out = {"n_events": nB,
               "B_book_carried": {str(k): float(B[k]) for k in range(1, KMAX + 1)},
               "T_trade_carried": {str(k): float(T[k]) for k in range(1, KMAX + 1)},
               "argmax_k": kpeak, "peak_value": float(B[kpeak]),
               "first_negative_k": kzero,
               "null_argmax_k": peaks_null,
               "null_peak_mean": float(np.mean([SB[s, int(np.argmax(SB[s, 1:]) + 1)]
                                                for s in range(N_SHUF)])),
               "null_peak_sd": float(np.std([SB[s, int(np.argmax(SB[s, 1:]) + 1)]
                                             for s in range(N_SHUF)])),
               "book_says": BOOK_TURNING_POINT}
        res["per_symbol"][sym] = out
        print("=== %s   n=%d" % (sym, nB), flush=True)
        ks = [1, 2, 3, 5, 8, 10, 12, 15, 20, 30, 40, 60]
        print("    B(k) book-carried : " + "  ".join("%d %+.4f" % (k, B[k]) for k in ks),
              flush=True)
        print("    T(k) trade-carried: " + "  ".join("%d %+.4f" % (k, T[k]) for k in ks),
              flush=True)
        print("    argmax k = %d  (book says about %d)   peak %+.4f   first negative k = %s"
              % (kpeak, BOOK_TURNING_POINT, B[kpeak], kzero), flush=True)
        print("    NULL argmax positions %s   null peak %+.5f +- %.5f"
              % (peaks_null, out["null_peak_mean"], out["null_peak_sd"]), flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "CT37_BOOK_TURNS_V1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print("written", flush=True)


if __name__ == "__main__":
    main()
