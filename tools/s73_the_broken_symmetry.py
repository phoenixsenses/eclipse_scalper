# -*- coding: utf-8 -*-
"""S73 -- how badly is LdP's symmetry assumption broken here, and what is the real break-even?

INHERITED, NOT RE-DERIVED
-------------------------
`--who bet sizing symmetric payoff` returned §457 (this lane's own A-S29), which already
opened MLAM ch.5 and closed it:

    "§5.5.1 opens: '...for some SYMMETRIC PAYOFF of magnitude pi > 0.  mu = pi(2p-1) ...
     z = (p-1/2)/sqrt(p(1-p)), m = 2Z[z]-1'
     'a symmetric payoff of magnitude pi' is exactly the assumption S20's identity breaks."
    verdict: THE TOOL IS NOT VALID HERE.

That verdict is INHERITED with citation.  This does not re-open it.

WHAT A-S72 ADDS THAT §457 COULD NOT HAVE HAD
--------------------------------------------
§457 argued the assumption is broken IN PRINCIPLE.  A-S72 measured the payoff distribution
for the first time, so the violation can be QUANTIFIED and the corrected break-even
computed.  With an asymmetric payoff the arithmetic is not LdP's:

    symmetric    mu = pi(2p-1)                     break-even  p* = 1/2
    asymmetric   mu = p*pi_u - (1-p)*pi_d          break-even  p* = pi_d / (pi_u + pi_d)
                 sigma^2 = p*pi_u^2 + (1-p)*pi_d^2 - mu^2
                 z = mu / sigma

AND THE PAYOFFS MUST NOT BE MFE AND MAE
---------------------------------------
Using the best and worst excursions as pi_u and pi_d would price a TRIPLE-BARRIER rule --
exit at whichever extreme comes first.  `CLAUDE.md`'s graveyard closes tight stops and
partial exits, and this study proposes no rule.  The applicable payoffs are the TERMINAL
ones: what a position that simply holds to t+60 actually collects, split by sign.

    p     = P(END > 0)
    pi_u  =  E[END | END > 0]
    pi_d  = -E[END | END < 0]

all measured on the same events A-S72 used, with no conditioning after t0.
"""

import io
import json
import math
import sqlite3

LIQ = "file:data/microstructure_02.db?mode=ro"
PANEL = "file:data/xsec_klines_ext.db?mode=ro"
CUT = 1787270400000
OUT = "reports/research/h2_response_shape_v1/S73_BROKEN_SYMMETRY_V1.json"
SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
MIN_MS = 60000
T0, T1 = 18, 60
FLOORS = (0.0, 500000.0)
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


def z_asym(p, pu, pd):
    mu = p * pu - (1 - p) * pd
    var = p * pu * pu + (1 - p) * pd * pd - mu * mu
    sd = math.sqrt(var) if var > 0 else float("nan")
    return mu, sd, (mu / sd if sd == sd and sd > 0 else float("nan"))


def z_sym(p):
    return (p - 0.5) / math.sqrt(p * (1 - p)) if 0 < p < 1 else float("nan")


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

    print("HOW BADLY IS LdP's SYMMETRY BROKEN, AND WHAT IS THE REAL BREAK-EVEN?")
    print("  §457 (this lane) already ruled the tool invalid; that is INHERITED.")
    print("  what is new is the MEASUREMENT of the violation and the corrected p*.")
    print("  payoffs are TERMINAL (hold to t+%d), not MFE/MAE -- barriers are graveyarded." % T1)

    res = {}
    for floor in FLOORS:
        print()
        print("  ===== SIZE FLOOR $%s =====" % format(int(floor), ","))
        print("    %-9s %8s %8s %9s %9s %9s %9s %9s"
              % ("symbol", "n", "p", "pi_u", "pi_d", "p* asym", "z asym", "z LdP"))
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
            up = [x for x in end if x > 0]
            dn = [x for x in end if x < 0]
            n = len(end)
            p = len(up) / float(n)
            pu = sum(up) / len(up) if up else 0.0
            pd = -sum(dn) / len(dn) if dn else 0.0
            pstar = pd / (pu + pd) if (pu + pd) else float("nan")
            mu, sd, za = z_asym(p, pu, pd)
            zs = z_sym(p)
            print("    %-9s %8s %8.4f %9.2f %9.2f %9.4f %9.4f %9.4f"
                  % (s, format(n, ","), p, pu, pd, pstar, za, zs))
            res["%s|%.0f" % (s, floor)] = {
                "n": n, "p": p, "pi_u": pu, "pi_d": pd, "p_star": pstar,
                "mu": mu, "sd": sd, "z_asym": za, "z_ldp": zs,
                "asymmetry": pu / pd if pd else float("nan"),
                "margin_p_minus_pstar": p - pstar}

    print()
    print("WHAT THE NUMBERS SAY")
    print("  %-16s %10s %10s %10s %12s"
          % ("cell", "pi_u/pi_d", "p", "p*", "p - p*"))
    for k, v in res.items():
        print("  %-16s %10.3f %10.4f %10.4f %+12.4f"
              % (k, v["asymmetry"], v["p"], v["p_star"], v["margin_p_minus_pstar"]))
    print()
    print("  LdP's formula assumes p* = 0.5 by construction.  Measured p* is where the")
    print("  asymmetry actually puts it, and the two differ on every cell.  Using the")
    print("  symmetric z where the payoff is asymmetric mis-states the bet size in a")
    print("  direction that depends on which side is fatter -- it is not a rescaling.")
    print()
    print("  And the fee is %.1f bps, which enters as a shift of pi_u down and pi_d up:" % MAKER_RT)
    print("  %-16s %12s %12s %12s" % ("cell", "p* net of fee", "p", "clears?"))
    for k, v in res.items():
        pun, pdn = v["pi_u"] - MAKER_RT, v["pi_d"] + MAKER_RT
        ps = pdn / (pun + pdn) if (pun + pdn) > 0 else float("nan")
        ok = "yes" if v["p"] > ps else "NO"
        print("  %-16s %12.4f %12.4f %12s" % (k, ps, v["p"], ok))
        v["p_star_net"] = ps
        v["clears_net"] = ok == "yes"

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"study": "S73_BROKEN_SYMMETRY", "t0": T0, "t1": T1, "maker_rt": MAKER_RT,
         "inherits": "§457 A-S29 -- LdP ch.5 bet sizing NOT VALID here",
         "cells": res}, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
