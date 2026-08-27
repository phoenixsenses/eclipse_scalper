# -*- coding: utf-8 -*-
"""C-KULLIYAT-T52 -- RECONCILING R(1) WITH C-T52: IS IT THE POPULATION OR THE OBSERVABLE?

C-KULLIYAT-T51 measured R(1) per event as +0.0158 / +0.0302 / +0.0500 bps.  The other lane-C
session's C-T52 measured the same-named quantity, in the same event unit, as 0.0484 (BTC) and
0.1093 (ETH) -- a factor of about three.  I logged it as a POPULATION difference to reconcile.
Reading their driver shows that guess was probably wrong, and in an interesting way:

    tools/research_c52_response_event_unit_v1.py  line 99:
        "select ts_ms, price, is_buyer_maker from agg_trades ..."
        response(lp, eps, lags)  with lp = log of the TRADE PRICE

    tools/ct_kulliyat_t51_the_fast_maker_break_even.py:
        m = book_ticker MID at the event, R(1) = eps_t (m_{t+1} - m_t) / m_t

Both are called R(l).  They are different observables.  The corpus is not neutral between them:
Bouchaud defines the response on the MID -- Eq (16.22) and, in the Sec 17.2 derivation this lane
just used, Eq (17.10) R(l) = E[eps_{t-l} (m_t - m_{t-l})].  A trade-price series carries the
bid-ask bounce, which a mid series does not.

And the DB is the SAME: tools/h2_response_shape_driver.py line 88 is
data/microstructure_02.db, which is the file their driver opens.  So the population difference
reduces to WHICH ROWS -- my seven named days versus their first 2 000 000 rows by ts_ms -- and
that is measurable rather than assumable.

PREREGISTERED, fixed before any number is read:
  Q1  on MY population and MY events, R(1) computed on the MID (arm A, as C-KULLIYAT-T51)
      and on the TRADE PRICE (arm B, as C-T52).  One thing varied: the observable.
  Q2  does arm B land near C-T52's 0.0484 / 0.1093?  If it does, the discrepancy is the
      observable and my "population difference" note was wrong.
  Q3  what calendar span is "the first 2 000 000 agg_trades rows" per symbol?  This settles
      whether the two studies even overlap in time, without re-running their arm.

I will not call either number wrong.  Two observables can both be correctly measured; what is
at stake is which one Eq (17.14) and Eq (17.15) take, and that is a corpus question with an
answer.

DB is opened READ-ONLY.  Their driver is READ, never modified.  Ceiling: MEASUREMENT_FIDELITY.

  python -m tools.ct_kulliyat_t52_reconcile_r1_with_the_other_session --i-have-approval
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

DAYS = ("2026-08-07", "2026-08-08", "2026-08-09", "2026-08-10",
        "2026-08-11", "2026-08-12", "2026-08-13")
OUT = "reports/atlas"
THEIRS = {"BTCUSDT": 0.0484, "ETHUSDT": 0.1093, "SOLUSDT": None}   # C-T52, SYSTEM_STATE 527
MINE = {"BTCUSDT": 0.0158, "ETHUSDT": 0.0302, "SOLUSDT": 0.0500}   # C-KULLIYAT-T51


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    res = {"days": list(DAYS), "db": H2.DB,
           "arm_A": "book_ticker MID at the event (C-KULLIYAT-T51, and Bouchaud Eq 16.22/17.10)",
           "arm_B": "aggTrade TRADE PRICE at the event (C-T52)",
           "theirs": THEIRS, "mine_published": MINE,
           "per_symbol": {}, "ceiling": "MEASUREMENT_FIDELITY"}

    print("=== Q3  what calendar span is 'the first 2,000,000 agg_trades rows'? ===", flush=True)
    con = D._con()
    for sym in H2.SYMBOLS:
        row = con.execute(
            "SELECT MIN(ts_ms), MAX(ts_ms) FROM (SELECT ts_ms FROM agg_trades WHERE symbol=? "
            "ORDER BY ts_ms LIMIT 2000000)", (sym,)).fetchone()
        lo, hi = row
        f = lambda x: dt.datetime.fromtimestamp(x / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
        res.setdefault("their_span", {})[sym] = {"from": f(lo), "to": f(hi)}
        print("    %-9s  %s  ->  %s" % (sym, f(lo), f(hi)), flush=True)
    con.close()
    print("    my population: %s .. %s" % (DAYS[0], DAYS[-1]), flush=True)

    print("\n=== Q1 / Q2  R(1) on MY population, observable varied ===", flush=True)
    print("    %-9s %12s %12s %8s | %10s" %
          ("symbol", "A  mid", "B  trade px", "B/A", "C-T52"), flush=True)
    for sym in H2.SYMBOLS:
        acc = {"A": [0.0, 0], "B": [0.0, 0]}
        for day in DAYS:
            d0 = dt.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
            lo = int(d0.timestamp() * 1000)
            hi = int((d0 + dt.timedelta(days=1)).timestamp() * 1000)
            con = D._con()
            rows = con.execute(
                "SELECT ts_ms,bid_price,ask_price FROM book_ticker WHERE symbol=? "
                "AND ts_ms>=? AND ts_ms<? AND bid_price>0 AND ask_price>0 ORDER BY ts_ms",
                (sym, lo - 2000, hi + 60000)).fetchall()
            con.close()
            if len(rows) < 10000:
                continue
            bts = np.array([r[0] for r in rows], np.int64)
            mid = np.array([0.5 * (r[1] + r[2]) for r in rows], float)
            del rows
            ts, px, eps, qty = B4.load_raw_with_qty(sym, (day,))
            # event = (ts_ms, side) collapse; the event's LAST price, as C-T52's collapse()
            new = np.concatenate([[True], (ts[1:] != ts[:-1]) | (eps[1:] != eps[:-1])])
            idx = np.flatnonzero(new)
            last = np.append(idx[1:] - 1, len(ts) - 1)
            oeps, ots, opx = eps[idx], ts[idx], px[last]
            del ts, px, eps, qty, new, idx, last
            ib = np.searchsorted(bts, ots, side="left") - 1
            ok = ib >= 0
            oeps, opx, ib = oeps[ok], opx[ok], ib[ok]
            m = mid[ib]
            ra = oeps[:-1] * (m[1:] - m[:-1]) / m[:-1] * 1e4
            rb = oeps[:-1] * (opx[1:] - opx[:-1]) / opx[:-1] * 1e4
            acc["A"][0] += float(ra.sum()); acc["A"][1] += len(ra)
            acc["B"][0] += float(rb.sum()); acc["B"][1] += len(rb)
            del bts, mid, oeps, opx, ib, m, ra, rb

        a = acc["A"][0] / acc["A"][1]
        b = acc["B"][0] / acc["B"][1]
        res["per_symbol"][sym] = {"R1_mid": a, "R1_trade_price": b, "ratio_B_over_A": b / a,
                                  "n": acc["A"][1], "theirs": THEIRS[sym]}
        print("    %-9s %+12.4f %+12.4f %8.2f | %10s"
              % (sym, a, b, b / a,
                 ("%+.4f" % THEIRS[sym]) if THEIRS[sym] is not None else "-"), flush=True)

    print("\n=== reading ===", flush=True)
    for sym in H2.SYMBOLS:
        r = res["per_symbol"][sym]
        if r["theirs"] is None:
            print("    %-9s  C-T52 published no value" % sym, flush=True)
            continue
        da = abs(r["R1_mid"] - r["theirs"])
        db = abs(r["R1_trade_price"] - r["theirs"])
        closer = "TRADE PRICE" if db < da else "MID"
        r["closer_to_theirs"] = closer
        print("    %-9s  |theirs - mid| %.4f   |theirs - tradepx| %.4f   -> %s is closer"
              % (sym, da, db, closer), flush=True)

    res["tokens"] = ["THE_TWO_R1_NUMBERS_USE_DIFFERENT_OBSERVABLES_NOT_DIFFERENT_POPULATIONS"
                     if all(res["per_symbol"][s].get("closer_to_theirs") == "TRADE PRICE"
                            for s in ("BTCUSDT", "ETHUSDT"))
                     else "THE_OBSERVABLE_DOES_NOT_EXPLAIN_THE_GAP",
                     "THE_CORPUS_DEFINES_R_ON_THE_MID_EQ_16_22_AND_17_10",
                     "SAME_DB_MICROSTRUCTURE_02_SO_POPULATION_IS_ROWS_NOT_SOURCE",
                     "MY_POPULATION_DIFFERENCE_NOTE_IS_TESTED_NOT_ASSUMED"]
    os.makedirs(OUT, exist_ok=True)
    with io.open(os.path.join(OUT, "CT_KULLIYAT_T52_R1_RECONCILE_V1.json"), "w",
                 encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, ensure_ascii=False))
    print("\nwritten %s/CT_KULLIYAT_T52_R1_RECONCILE_V1.json" % OUT, flush=True)


if __name__ == "__main__":
    main()
