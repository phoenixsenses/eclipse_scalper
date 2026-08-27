# -*- coding: utf-8 -*-
"""S74 -- the PSR on the STRATEGY's payoff, which A-S51 could not compute and flagged.

THE CAVEAT THIS CLOSES
----------------------
A-S51 applied Lopez de Prado's Probabilistic Sharpe Ratio and found the non-normality
correction negligible.  It closed with its own limitation, in its own words:

    "PSR wants the moments of the STRATEGY's returns, not the market's; and the prereg
     deliberately does not name a sign.  Under a symmetric sign skew(s*r) = 0 but the
     KURTOSIS carries over -- and it enters as ((g4-1)/4)*SR^2 = 2.7e-5, i.e. nothing."

The whole argument rested on SR = 0.003483, the frontier's per-trade Sharpe at f_design.
A-S73 has now measured an actual SIGNED payoff distribution, and its per-event Sharpe is
z = 0.1996 / 0.1734 / 0.0818 -- FIFTY-SEVEN TIMES larger than the number A-S51's dismissal
rested on.  At SR = 0.20 A-S51's own sensitivity table puts the PSR factor at 1.27 with a
skew of -3.  So the correction it dismissed may bite on this object, and that has to be
measured rather than argued either way.

WHAT IS COMPUTED
----------------
On the same END distribution A-S73 used (hold t+18 to t+60, oriented, no conditioning
after t0):

    SR      mean / sd, per event, directly from the full distribution
    g3, g4  the payoff's own skewness and kurtosis
    PSR     z[0] = SR*sqrt(T-1) / sqrt(1 - g3*SR + ((g4-1)/4)*SR^2)   (MLAM s.107)
    and the same z under the Normal assumption, for the ratio

AND A CROSS-CHECK ON A-S73
--------------------------
A-S73 computed z from a TWO-POINT summary (p, pi_u, pi_d).  The full distribution gives SR
directly.  If the two disagree materially, A-S73's two-point z is the approximation and
this says by how much.
"""

import io
import json
import math
import sqlite3

LIQ = "file:data/microstructure_02.db?mode=ro"
PANEL = "file:data/xsec_klines_ext.db?mode=ro"
CUT = 1787270400000
OUT = "reports/research/h2_response_shape_v1/S74_PSR_ON_THE_PAYOFF_V1.json"
SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
MIN_MS = 60000
T0, T1 = 18, 60
FLOORS = (0.0, 500000.0)
MAKER_RT = 4.0
A_S73_Z = {"BTCUSDT|0": 0.1996, "ETHUSDT|0": 0.1734, "SOLUSDT|0": 0.0818,
           "BTCUSDT|500000": 0.8446, "ETHUSDT|500000": 0.7242}


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


def moments(v):
    n = len(v)
    m = sum(v) / n
    m2 = sum((x - m) ** 2 for x in v) / n
    m3 = sum((x - m) ** 3 for x in v) / n
    m4 = sum((x - m) ** 4 for x in v) / n
    sd = math.sqrt(m2)
    return m, sd, m3 / m2 ** 1.5, m4 / m2 ** 2


def psr_factor(sr, g3, g4):
    v = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr
    return math.sqrt(v) if v > 0 else float("nan")


def main():
    c = sqlite3.connect(LIQ, uri=True)
    llo, lhi = c.execute("SELECT MIN(ts_ms), MAX(ts_ms) FROM liquidations WHERE ts_ms<?",
                         (CUT,)).fetchone()
    c.close()
    p_ = sqlite3.connect(PANEL, uri=True)
    plo, phi = p_.execute("SELECT MIN(open_time), MAX(open_time) FROM klines WHERE symbol=?",
                          ("BTCUSDT",)).fetchone()
    p_.close()
    lo, hi = max(llo, plo), min(lhi, phi)

    print("THE PSR ON THE STRATEGY'S OWN PAYOFF  (closing A-S51's stated caveat)")
    print("  A-S51 dismissed the correction at SR = 0.003483.  A-S73's payoff has")
    print("  SR ~ 0.20, fifty-seven times larger, so the dismissal has to be re-run.")
    print("  fee %.1f bps is applied to the payoff before the moments are taken." % MAKER_RT)

    res = {}
    for floor in FLOORS:
        print()
        print("  ===== SIZE FLOOR $%s  (gross, then net of the %.1f bps fee) ====="
              % (format(int(floor), ","), MAKER_RT))
        print("    %-9s %8s %8s %8s %9s %9s %10s %10s"
              % ("symbol", "n", "SR", "A-S73 z", "skew", "kurtosis", "PSR fac", "z / z_norm"))
        for s in SYMS:
            px = closes(s, lo, hi)
            ev = events(s, lo, hi, floor)
            end = []
            for b, sgn, _n in ev:
                a = px.get(b + T0)
                z = px.get(b + T1)
                if a and z:
                    end.append(sgn * math.log(z / a) * 1e4)
            if len(end) < 100:
                print("    %-9s %8d  insufficient" % (s, len(end)))
                continue
            for tag, v in (("gross", end), ("net", [x - MAKER_RT for x in end])):
                m, sd, g3, g4 = moments(v)
                sr = m / sd if sd else float("nan")
                fac = psr_factor(sr, g3, g4)
                key = "%s|%.0f" % (s, floor)
                z73 = A_S73_Z.get(key, float("nan"))
                print("    %-9s %8s %8.4f %8.4f %9.3f %9.2f %10.4f %10.4f"
                      % (s if tag == "gross" else "  net", format(len(v), ","), sr,
                         z73 if tag == "gross" else float("nan"), g3, g4, fac, 1.0 / fac))
                res["%s|%s" % (key, tag)] = {
                    "n": len(v), "SR": sr, "skew": g3, "kurt": g4,
                    "psr_factor": fac, "z_over_z_norm": 1.0 / fac,
                    "a_s73_two_point_z": z73 if tag == "gross" else None}

    print()
    print("WHAT IT SAYS")
    print("  A-S51's dismissal was correct FOR ITS OWN OBJECT and does not transfer.  The")
    print("  correction enters as g3*SR and SR here is two orders larger, so the factor is")
    print("  no longer 1.000.  Whether it helps or hurts depends on the SIGN of the skew:")
    print("  a POSITIVE skew makes the denominator smaller than one, which INFLATES the")
    print("  corrected z rather than deflating it -- the opposite of the hedge-fund case")
    print("  MLAM uses as its example.")
    print()
    print("  and the cross-check on A-S73's two-point z:")
    for k, v in res.items():
        if not k.endswith("|gross") or v["a_s73_two_point_z"] != v["a_s73_two_point_z"]:
            continue
        d = v["SR"] - v["a_s73_two_point_z"]
        print("    %-22s full-dist SR %.4f  vs two-point %.4f   diff %+.4f"
              % (k.replace("|gross", ""), v["SR"], v["a_s73_two_point_z"], d))
    print("    a two-point summary keeps the mean and the variance of a two-outcome bet,")
    print("    not of this one; the gap is the price of that compression.")

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"study": "S74_PSR_ON_THE_PAYOFF", "t0": T0, "t1": T1, "maker_rt": MAKER_RT,
         "closes": "A-S51's stated caveat: PSR wants the STRATEGY's moments",
         "cells": res}, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
