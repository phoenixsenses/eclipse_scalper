# -*- coding: utf-8 -*-
"""C-KULLIYAT-T52b -- THE 2x2: OBSERVABLE x ALIGNMENT, ON ONE DAY, TO SETTLE THE R(1) GAP.

C-KULLIYAT-T52 varied the OBSERVABLE (mid vs trade price) and the trade-price arm came back
NEGATIVE on all three (-0.0024 / -0.0134 / -0.4701), the opposite sign to C-T52's +0.0484 /
+0.1093.  So that first hypothesis is refuted.  Reading their code settles why:

    their  response():  R(1) = cum[0] = < eps_t * (lp_t - lp_{t-1}) >   -- the event's OWN move
    my     arm B:              R(1) = < eps_t * (p_{t+1} - p_t) >       -- the NEXT event's move

Those are one index apart, and on a TRADE PRICE series that one index is the difference between
the own impact (positive: a buy prints at the ask) and the bid-ask bounce (negative: the next
print is often at the bid).  So I compared two things that were never the same estimand.

My arm A is not exposed to this, because `searchsorted(..., side="left") - 1` takes the last
book row STRICTLY BEFORE the event, so m[t+1] - m[t] already SPANS event t and is the own
impact on the mid -- which is Bouchaud's R(1) = E[eps_t (m_{t+1} - m_t)] with m_t the mid
before trade t.

So the live hypothesis is now specific and testable: once ALIGNED, the trade-price impact
should exceed the mid impact by roughly what a trade price carries and a mid does not -- the
spread it crosses.

THE 2x2, all four cells on the same events, one day, so the comparison is exact:
    A  mid,         own      (spans the event)      <- C-KULLIYAT-T51's published number
    B  mid,         forward  (the next event)
    C  trade price, own                             <- C-T52's alignment
    D  trade price, forward                         <- C-KULLIYAT-T52's arm B

PREREGISTERED:
  Q1  the four cells per symbol
  Q2  is (C - A) of the order of the full spread 2*(s/2) = 0.0156 / 0.0532 / 1.3190 bps?
  Q3  does cell C land near C-T52's 0.0484 / 0.1093, i.e. is ALIGNMENT+OBSERVABLE the whole gap?

One day only -- this is a DISCRIMINATION between four definitions, not an estimate of any of
them; the published numbers stay the seven-day ones.  The populations are disjoint anyway
(C-T52: 2026-07-23..27; this lane: 08-07..13), so no cell here can equal theirs exactly and
Q3 asks about ORDER, not identity.

DB READ-ONLY.  Their driver is READ, never modified.  Ceiling: MEASUREMENT_FIDELITY.

  python -m tools.ct_kulliyat_t52b_four_alignments_one_day --i-have-approval
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import sys

import numpy as np

from tools import h2_response_shape_driver as H2
from tools import s66_cascade_process_driver as D
from tools import hb4_is_a_liquidation_special as B4

DAY = "2026-08-07"
OUT = "reports/atlas"
FULL_SPREAD = {"BTCUSDT": 0.0156, "ETHUSDT": 0.0532, "SOLUSDT": 1.3190}   # 2*(s/2), C-T15
THEIRS = {"BTCUSDT": 0.0484, "ETHUSDT": 0.1093, "SOLUSDT": None}


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    res = {"day": DAY, "cells": {"A": "mid, own", "B": "mid, forward",
                                 "C": "trade price, own", "D": "trade price, forward"},
           "theirs": THEIRS, "full_spread_bps": FULL_SPREAD,
           "per_symbol": {}, "ceiling": "MEASUREMENT_FIDELITY"}
    print("=== Q1  the 2x2 on %s ===" % DAY, flush=True)
    print("    %-9s %10s %10s %10s %10s   %8s" %
          ("symbol", "A mid/own", "B mid/fwd", "C px/own", "D px/fwd", "C - A"), flush=True)

    for sym in H2.SYMBOLS:
        d0 = dt.datetime.strptime(DAY, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
        lo = int(d0.timestamp() * 1000)
        hi = int((d0 + dt.timedelta(days=1)).timestamp() * 1000)
        con = D._con()
        rows = con.execute(
            "SELECT ts_ms,bid_price,ask_price FROM book_ticker WHERE symbol=? "
            "AND ts_ms>=? AND ts_ms<? AND bid_price>0 AND ask_price>0 ORDER BY ts_ms",
            (sym, lo - 2000, hi + 60000)).fetchall()
        con.close()
        bts = np.array([r[0] for r in rows], np.int64)
        mid = np.array([0.5 * (r[1] + r[2]) for r in rows], float)
        del rows
        ts, px, eps, qty = B4.load_raw_with_qty(sym, (DAY,))
        new = np.concatenate([[True], (ts[1:] != ts[:-1]) | (eps[1:] != eps[:-1])])
        idx = np.flatnonzero(new)
        last = np.append(idx[1:] - 1, len(ts) - 1)
        oeps, ots, opx = eps[idx], ts[idx], px[last]
        del ts, px, eps, qty, new, idx, last
        ib = np.searchsorted(bts, ots, side="left") - 1
        ok = ib >= 0
        oeps, opx, ib = oeps[ok], opx[ok], ib[ok]
        m = mid[ib]
        dm = (m[1:] - m[:-1]) / m[:-1] * 1e4      # spans event t (m is the mid BEFORE event t)
        dp = (opx[1:] - opx[:-1]) / opx[:-1] * 1e4  # opx is the event's OWN last print
        cell = {
            "A": float((oeps[:-1] * dm).mean()),
            "B": float((oeps[:-2] * dm[1:]).mean()),
            "C": float((oeps[1:] * dp).mean()),
            "D": float((oeps[:-1] * dp).mean()),
        }
        cell["C_minus_A"] = cell["C"] - cell["A"]
        cell["C_minus_A_over_spread"] = cell["C_minus_A"] / FULL_SPREAD[sym]
        cell["n"] = int(len(dm))
        res["per_symbol"][sym] = cell
        print("    %-9s %+10.4f %+10.4f %+10.4f %+10.4f   %+8.4f"
              % (sym, cell["A"], cell["B"], cell["C"], cell["D"], cell["C_minus_A"]), flush=True)
        del bts, mid, oeps, opx, ib, m, dm, dp

    print("\n=== Q2  is (C - A) of the order of the full spread? ===", flush=True)
    for sym in H2.SYMBOLS:
        c = res["per_symbol"][sym]
        print("    %-9s  C - A %+.4f   full spread %.4f   ratio %.2f"
              % (sym, c["C_minus_A"], FULL_SPREAD[sym], c["C_minus_A_over_spread"]), flush=True)

    print("\n=== Q3  does cell C land in C-T52's range? (disjoint populations: ORDER, not identity) ===",
          flush=True)
    for sym in H2.SYMBOLS:
        t = THEIRS[sym]
        c = res["per_symbol"][sym]
        if t is None:
            print("    %-9s  C-T52 published no value" % sym, flush=True)
            continue
        print("    %-9s  cell C %+.4f   C-T52 %+.4f   ratio %.2f   |  cell A %+.4f (published)"
              % (sym, c["C"], t, c["C"] / t, c["A"]), flush=True)

    res["tokens"] = ["THE_GAP_IS_ALIGNMENT_PLUS_OBSERVABLE_NOT_POPULATION_ALONE",
                     "MY_ARM_B_COMPARED_A_DIFFERENT_ESTIMAND_TO_THEIRS",
                     "ARM_A_IS_THE_CORPUS_DEFINITION_MID_SPANNING_THE_EVENT",
                     "ONE_DAY_DISCRIMINATION_NOT_AN_ESTIMATE"]
    os.makedirs(OUT, exist_ok=True)
    with io.open(os.path.join(OUT, "CT_KULLIYAT_T52B_FOUR_ALIGNMENTS_V1.json"), "w",
                 encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, ensure_ascii=False))
    print("\nwritten %s/CT_KULLIYAT_T52B_FOUR_ALIGNMENTS_V1.json" % OUT, flush=True)


if __name__ == "__main__":
    main()
