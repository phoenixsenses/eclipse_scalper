# -*- coding: utf-8 -*-
"""S72 -- pricing A-S71's unstated condition, without conditioning on anything.

THE CONDITION
-------------
A-S71 found the frontier rests on something nobody had written down: that a position can be
HELD THROUGH the interruption D's mu_tau times.  Read at the accumulation clock the edge is
negative on both majors (-2.571 BTC, -0.945 ETH against a 4.0 bps maker fee); read at the
holding clock it is positive (+2.718, +6.070).  Everything the frontier claims lives in the
increment between eighteen minutes and sixty.

    UNCONDITIONALLY, from A-S54's own path:
        BTC  G(60) - G(18.1) = +5.289 bps
        ETH                  = +7.015
        SOL                  = +3.262

That increment is a MEAN and it needs no conditioning, so it carries none of A-S67's
collider problem.  But a mean is not the condition.  The condition is whether a position
can SIT through the interval, and what decides that is the EXCURSION along the way, not
its endpoint.

WHAT IS MEASURED
----------------
Per event, oriented by the forced-flow direction, over the interval [t+18, t+60]:

    MAE   the worst ADVERSE excursion relative to t+18   (how far underwater you go)
    MFE   the best favourable one
    END   the terminal increment G(60) - G(18)

No conditioning on anything after t0.  Every event with prices at both ends is included,
so the collider path A-S67 named is not opened.  Distributions are reported, not means
alone, because the condition is about the tail and a mean cannot express it.

PRIOR ART, HONESTLY
-------------------
`--who` returned nothing across four queries in two languages ("excursion drawdown holding",
"olumsuz sapma tutma", "MAE MFE tail", "adverse excursion").  That is a CLAIM, not a
default, and it is qualified: this lane's own memory records §162/§163 (4h tail 21.7%
irreducible) and a SHORT re-mine that measured MFE/MAE -- on ROUTE EXITS, not on the
hold-through condition, and at a different horizon.  Related object, different question.
"""

import io
import json
import math
import sqlite3

LIQ = "file:data/microstructure_02.db?mode=ro"
PANEL = "file:data/xsec_klines_ext.db?mode=ro"
CUT = 1787270400000
OUT = "reports/research/h2_response_shape_v1/S72_CAN_YOU_HOLD_V1.json"
SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
MIN_MS = 60000
T0, T1 = 18, 60                       # the interval the frontier depends on
FLOORS = (0.0, 500000.0)              # D-E2's two published floors
MAKER_RT = 4.0


def closes(sym, lo, hi):
    c = sqlite3.connect(PANEL, uri=True)
    r = c.execute("SELECT open_time/%d, close FROM klines WHERE symbol=? AND open_time>=? "
                  "AND open_time<?" % MIN_MS, (sym, lo, hi)).fetchall()
    c.close()
    return {int(b): float(p) for b, p in r if p and p > 0}


def events(sym, lo, hi, floor):
    c = sqlite3.connect(LIQ, uri=True)
    r = c.execute("SELECT ts_ms, side, notional FROM liquidations WHERE symbol=? AND "
                  "ts_ms>=? AND ts_ms<? AND notional>=? ORDER BY ts_ms",
                  (sym, lo, hi, floor)).fetchall()
    c.close()
    return [(int(t) // MIN_MS, (1.0 if s == "BUY" else -1.0), float(n)) for t, s, n in r]


def q(v, p):
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def main():
    c = sqlite3.connect(LIQ, uri=True)
    llo, lhi = c.execute("SELECT MIN(ts_ms), MAX(ts_ms) FROM liquidations WHERE ts_ms<?",
                         (CUT,)).fetchone()
    c.close()
    p = sqlite3.connect(PANEL, uri=True)
    plo, phi = p.execute("SELECT MIN(open_time), MAX(open_time) FROM klines WHERE symbol=?",
                         ("BTCUSDT",)).fetchone()
    p.close()
    lo, hi = max(llo, plo), min(lhi, phi)

    print("CAN A POSITION BE HELD FROM t+%d TO t+%d?  (A-S71's unstated condition)" % (T0, T1))
    print("  oriented by the forced flow, relative to t+%d.  NO conditioning on anything" % T0)
    print("  after t0, so A-S67's collider path is not opened.")
    print("  the frontier needs the END increment; the CONDITION is about the MAE.")

    res = {}
    for floor in FLOORS:
        print()
        print("  ===== SIZE FLOOR $%s =====" % format(int(floor), ","))
        for s in SYMS:
            px = closes(s, lo, hi)
            ev = events(s, lo, hi, floor)
            mae, mfe, end = [], [], []
            for b, sgn, _n in ev:
                p0 = px.get(b + T0)
                if not p0:
                    continue
                path = []
                for k in range(T0, T1 + 1):
                    pk = px.get(b + k)
                    if pk:
                        path.append(sgn * math.log(pk / p0) * 1e4)
                if len(path) < (T1 - T0) // 2:
                    continue
                mae.append(min(path))
                mfe.append(max(path))
                end.append(path[-1])
            if len(end) < 100:
                print("  %-9s insufficient (%d)" % (s, len(end)))
                continue
            mae.sort(); mfe.sort(); end.sort()
            n = len(end)
            m_end = sum(end) / n
            m_mae = sum(mae) / n
            print()
            print("  %s   n = %s" % (s, format(n, ",")))
            print("    %-22s %9s %9s %9s %9s %9s"
                  % ("", "mean", "p05", "p25", "p50", "p95"))
            for lab, v in (("END  G(60)-G(18)", end), ("MAE  worst adverse", mae),
                           ("MFE  best favourable", mfe)):
                print("    %-22s %9.2f %9.2f %9.2f %9.2f %9.2f"
                      % (lab, sum(v) / n, q(v, 0.05), q(v, 0.25), q(v, 0.50), q(v, 0.95)))
            ratio = -m_mae / m_end if m_end else float("nan")
            worse = 100.0 * sum(1 for a, e in zip(mae, end) if -a > m_end) / n
            print("    mean MAE is %.2fx the mean END  --  you sit through %.2f bps to earn %.2f"
                  % (ratio, -m_mae, m_end))
            print("    share of events whose ADVERSE excursion exceeds the MEAN gain: %.1f%%"
                  % worse)
            print("    and the fee is %.1f bps, so the END must clear it: mean %.2f, p50 %.2f"
                  % (MAKER_RT, m_end, q(end, 0.50)))
            res["%s|%.0f" % (s, floor)] = {
                "n": n, "end_mean": m_end, "end_p50": q(end, 0.50),
                "mae_mean": m_mae, "mae_p05": q(mae, 0.05), "mfe_mean": sum(mfe) / n,
                "mae_over_end": ratio, "share_mae_exceeds_mean_gain": worse}

    print()
    print("WHAT THIS PRICES")
    print("  The frontier's condition is not 'is the increment positive' -- it is -- but")
    print("  'can the position sit through the excursion that comes with it'.  Those are")
    print("  different questions and only the first had an answer before this.")

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"study": "S72_CAN_YOU_HOLD", "t0": T0, "t1": T1, "floors": list(FLOORS),
         "maker_rt": MAKER_RT, "cells": res}, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
