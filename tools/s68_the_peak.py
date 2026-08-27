# -*- coding: utf-8 -*-
"""S68 -- where is the peak, and does it revert?  The post-liquidation path at 1 second.

WHAT THE CORPUS DEMANDS AND A-S54 COULD NOT SEE
-----------------------------------------------
Bouchaud TQP, Figure 12.1, in as many words:

    "Over the course of its execution, a buy metaorder pushes the price up, until it
     reaches a PEAK impact.  Upon completion, the buying pressure stops and the price
     REVERTS ABRUPTLY.  Some impact is however still observable long after the metaorder
     execution is completed, and sometimes persists permanently."

That is a shape with two features -- a peak and a reversion -- and A-S54 measured the
post-liquidation path on ONE-MINUTE bars, where neither is resolvable.  A-S54 nonetheless
read its t+1 point as impact:

    "the liquidation's own impact IS visible: a transient -11.8 bps at t+1 recovering by
     t+10, the same order as the 5.24 bps A-S53's law predicted.  Two studies, two
     directions, one answer."

Two lanes have since said that reading cannot stand at that resolution:

    C-T44  single-trade response saturation is 0.2-0.6 MINUTES -- so by t+1 the object
           A-S54 claimed to measure has already saturated and begun to revert.
    C-T47  "price discovery is decisively resolved at 1 s and COMPLETELY UNRESOLVED at
           60 s ... if any of your work runs on minute bars, this is the scale at which
           that class of question stops carrying information."

THE MEASUREMENT
---------------
`data/raw_trades_v1` (DL-002) holds RAW trades for BTC/ETH/SOL over 2026-08-07..08-13,
seven days that sit INSIDE the liquidation window (2026-07-23..08-20).  3,416 BTCUSDT
liquidations fall in the overlap.  So the same path A-S54 drew at one minute can be drawn
at ONE SECOND, on the same events, and the two features the corpus names either appear or
they do not.

THE NULL, CALIBRATED BEFORE THE TEST IS READ
--------------------------------------------
The estimator is run against MATCHED PLACEBO seconds -- a uniformly drawn second from the
same symbol-day, given a random orientation -- so that whatever shape the estimator
produces on nothing is on the page beside the shape it produces on events.  Liquidations
follow moves, so the event path will not be flat before t0 and the placebo will be; that
difference is selection, not response, and it is left visible rather than argued away.
"""

import io
import json
import math
import os
import random
import sqlite3
import sys
import zipfile

LIQ = "file:data/microstructure_02.db?mode=ro"
DIR = "data/raw_trades_v1"
OUT = "reports/research/h2_response_shape_v1/S68_THE_PEAK_V1.json"
PRE, POST = 60, 300                      # seconds either side
FLOOR = 500000.0                         # D-E2's published floor, as in A-S65
SEED = 20260827


def day_prices(sym, day):
    """Last trade price per SECOND for one symbol-day, and the day's first ms."""
    p = "%s/%s-trades-%s.zip" % (DIR, sym, day)
    if not os.path.exists(p):
        return None
    z = zipfile.ZipFile(p)
    px = {}
    with z.open(z.namelist()[0]) as fh:
        fh.readline()
        for line in fh:
            f = line.split(b",")
            px[int(f[4]) // 1000] = float(f[1])       # last trade in that second wins
    return px


def liqs(sym, lo, hi, floor):
    c = sqlite3.connect(LIQ, uri=True)
    r = c.execute("SELECT ts_ms, side, notional FROM liquidations WHERE symbol=? AND "
                  "ts_ms>=? AND ts_ms<? AND notional>=? ORDER BY ts_ms",
                  (sym, lo, hi, floor)).fetchall()
    c.close()
    return [(int(t) // 1000, (1.0 if s == "BUY" else -1.0), float(n)) for t, s, n in r]


def path(px, evs):
    """Oriented cumulative log return from t0, in bps, per second offset.

    A second with no trade is a GAP, not a zero: the offset is skipped for that event
    rather than carried forward, so a quiet second cannot manufacture a flat path."""
    n = PRE + POST + 1
    acc, cnt = [0.0] * n, [0] * n
    used = 0
    for s0, sgn, _sz in evs:
        p0 = px.get(s0)
        if not p0:
            continue
        used += 1
        for i, k in enumerate(range(-PRE, POST + 1)):
            p = px.get(s0 + k)
            if p:
                acc[i] += sgn * math.log(p / p0) * 1e4
                cnt[i] += 1
    return [(acc[i] / cnt[i]) if cnt[i] else float("nan") for i in range(n)], cnt, used


def main():
    days = ["2026-08-%02d" % d for d in range(7, 14)]
    syms = sys.argv[1:] or ["BTCUSDT"]
    rnd = random.Random(SEED)
    res = {}
    print("THE POST-LIQUIDATION PATH AT ONE SECOND  (TQP Fig 12.1: a PEAK, then a REVERSION)")
    print("  A-S54 drew this at ONE MINUTE and read t+1 as impact.  C-T44 puts single-trade")
    print("  saturation at 0.2-0.6 min and C-T47 calls 60 s unresolved for this class.")
    print("  floor $%s (D-E2's published floor, as in A-S65)   null = matched placebo seconds"
          % format(int(FLOOR), ","))

    for sym in syms:
        allp, alle, allplacebo = {}, [], []
        for d in days:
            px = day_prices(sym, d)
            if not px:
                continue
            secs = sorted(px)
            lo = secs[0] * 1000
            hi = (secs[-1] + 1) * 1000
            ev = liqs(sym, lo, hi, FLOOR)
            alle += ev
            # matched placebo: one uniform second per event, random orientation
            for _ in ev:
                allplacebo.append((rnd.choice(secs), rnd.choice((1.0, -1.0)), 0.0))
            allp.update(px)
        if not alle:
            print("\n  %s: no events in the overlap at this floor" % sym)
            continue
        m, c, used = path(allp, alle)
        mp, cp, usedp = path(allp, allplacebo)
        pk = max(range(PRE, PRE + POST + 1), key=lambda i: m[i] if m[i] == m[i] else -1e9)
        print()
        print("  %s   events %s used %s   placebo %s   seconds priced %s"
              % (sym, format(len(alle), ","), format(used, ","), format(usedp, ","),
                 format(len(allp), ",")))
        print("    %-9s %s" % ("t (s)", "".join("%9s" % x for x in
                                                ("-60", "-10", "-1", "0", "+1", "+5",
                                                 "+15", "+60", "+180", "+300"))))
        idx = [PRE - 60, PRE - 10, PRE - 1, PRE, PRE + 1, PRE + 5,
               PRE + 15, PRE + 60, PRE + 180, PRE + 300]
        print("    %-9s %s" % ("EVENT", "".join("%9.2f" % m[i] for i in idx)))
        print("    %-9s %s" % ("placebo", "".join("%9.2f" % mp[i] for i in idx)))
        print("    peak of the EVENT path after t0: %+.2f bps at t+%d s"
              % (m[pk], pk - PRE))
        rev = m[pk] - m[PRE + POST]
        print("    from the peak to t+%d s: %+.2f bps  (%.0f%% of the peak given back)"
              % (POST, -rev, 100.0 * rev / m[pk] if m[pk] else float("nan")))
        res[sym] = {"n_events": len(alle), "used": used, "path": m, "counts": c,
                    "placebo": mp, "peak_s": pk - PRE, "peak_bps": m[pk],
                    "at_300s": m[PRE + POST]}

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"study": "S68_THE_PEAK", "pre_s": PRE, "post_s": POST, "floor": FLOOR,
         "days": days, "seed": SEED, "symbols": res}, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
