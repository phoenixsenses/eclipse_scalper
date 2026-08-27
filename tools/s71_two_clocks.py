# -*- coding: utf-8 -*-
"""S71 -- the frontier runs on two clocks and nobody has checked they are compatible.

WHAT IS UNEXAMINED
------------------
A-S57 computed the room with ONE window, w = 50 min, used for two different purposes:

    IMPACT      k ~ sqrt(w)          -- w is how long you take to ACCUMULATE the position
    EDGE        a = G(w) - fee       -- w is how long you HOLD it

A-S69 then replaced w with lane D's measured mu_tau = 18.10 min in the SIZE channel and
left the edge at G(60).  That was not argued; it was where the two numbers happened to sit.
And A-S69's own invariance result -- the room does not depend on w at the revenue-optimal
POV -- meant the mismatch produced no numerical difference and so was never noticed.

    A frontier insensitive to a distinction it is getting wrong is not robust.  It is blind.

THE TWO CLOCKS, NAMED
---------------------
    ACCUMULATION  mu_tau = 18.10 min (D-E10), the expected time the episode stays open.
                  This is the right clock for IMPACT: it sets what fraction of the flow
                  you are, and D's construction X = ADV*POV*mu_tau says so explicitly.
    HOLDING       60 min, where A-S54's path is read and A-S62's lag sits (40-60 min).
                  This is the right clock for the EDGE: it is how long the continuation
                  takes to accrue.

They are ALLOWED to differ -- you build a position in eighteen minutes and hold it for an
hour -- but only if the frontier says so, and it does not.  This makes the pairing explicit
and prices the alternative in which they are forced equal.

AND THE QUESTION THAT DECIDES IT
--------------------------------
If the edge must be read at the ACCUMULATION clock -- if you cannot hold past the point
where the episode is interrupted -- then a = G(18) - fee, and G(18) is read off A-S54's own
path.  Whether that is positive at all is what this measures.

A-S65 bears directly on it: the continuation lives ENTIRELY in the windows where another
liquidation arrives.  So the interruption D's mu_tau times is not an exit signal -- it is
where the edge is.  That argues for holding through, but it is an argument and the number
should be on the page beside it.
"""

import io
import json
import math

PATH = "reports/research/h2_response_shape_v1/S54_THE_PATH_V1.json"
OUT = "reports/research/h2_response_shape_v1/S71_TWO_CLOCKS_V1.json"

MU_TAU = 18.10                     # D-E10, accumulation clock
HOLD = 60                          # A-S54's path horizon, holding clock
MAKER_RT = 4.0
Y, DELTA = 0.5, 0.5
SIGMA_D = {"BTCUSDT": 0.02015, "ETHUSDT": 0.03534, "SOLUSDT": 0.02541}
ADV = {"BTCUSDT": 7296912640.0, "ETHUSDT": 5218467948.0, "SOLUSDT": 996048951.0}


def G(path, K, minutes):
    """The oriented cumulative move at h minutes, interpolated between whole minutes."""
    lo = int(math.floor(minutes))
    hi = min(K, lo + 1)
    f = minutes - lo
    return path[K + lo] * (1 - f) + path[K + hi] * f


def room(a, sig, adv, w_impact, w_size):
    """Revenue-optimal room.  w_impact sets k; w_size sets how much notional fits."""
    if a <= 0:
        return None
    k = 2.0 * 1e4 * Y * sig * ((w_impact / 1440.0) ** DELTA)
    u = 2.0 * a / (3.0 * k)
    pov = u * u
    return {"a": a, "pov": pov, "net_bps": a - k * u,
            "size_usd": adv * pov * w_size / 1440.0}


def main():
    p = json.load(io.open(PATH, encoding="utf-8"))
    K = p["K_minutes"]

    print("THE FRONTIER'S TWO CLOCKS, MADE EXPLICIT")
    print("  ACCUMULATION  mu_tau = %.2f min (D-E10)  -> sets IMPACT and the size" % MU_TAU)
    print("  HOLDING       %d min (A-S54's path)      -> sets the EDGE" % HOLD)
    print("  A-S57 used ONE w = 50 for both.  A-S69 moved the size to mu_tau and left the")
    print("  edge at 60, and its own invariance result hid that the two were mismatched.")

    print()
    print("  %-9s %10s %10s %12s %12s"
          % ("symbol", "G(18.1)", "G(60)", "a @18.1", "a @60"))
    res = {}
    for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        path = p["symbols"][s]["all"]["path"]
        g18, g60 = G(path, K, MU_TAU), G(path, K, HOLD)
        print("  %-9s %10.3f %10.3f %12.3f %12.3f"
              % (s, g18, g60, g18 - MAKER_RT, g60 - MAKER_RT))
        res[s] = {"G_mu_tau": g18, "G_hold": g60,
                  "a_mu_tau": g18 - MAKER_RT, "a_hold": g60 - MAKER_RT}
    print()
    print("  AT THE ACCUMULATION CLOCK THE EDGE IS NEGATIVE ON BTC (-2.571) AND ETH")
    print("  (-0.945), AND BARELY POSITIVE ON SOL (+0.109).  The maker fee alone is %.1f"
          % MAKER_RT)
    print("  bps and the continuation has not accrued that far by 18 minutes.")
    print("  (I wrote 'negative on all three' into this driver before running it.  SOL is")
    print("   positive.  The corrected reading is sharper anyway: SOL's +0.109 bps yields a")
    print("   revenue-optimal room of 0.0363 bps on EIGHTY-ONE DOLLARS -- positive and")
    print("   economically nothing, which is a more useful statement than a wrong sign.)")

    print()
    print("SO THE FRONTIER ONLY EXISTS IF YOU HOLD PAST THE INTERRUPTION")
    print("  %-9s %-26s %10s %12s %14s"
          % ("symbol", "pairing", "POV*", "net bps", "size $"))
    for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        r = res[s]
        rows = [
            ("A-S57 as published (50/50)", 50.0, 50.0, r["G_hold"] - MAKER_RT),
            ("A-S69 (size mu_tau, edge 60)", MU_TAU, MU_TAU, r["a_hold"]),
            ("forced equal at mu_tau", MU_TAU, MU_TAU, r["a_mu_tau"]),
            ("forced equal at 60", 60.0, 60.0, r["a_hold"]),
        ]
        for lab, wi, ws, a in rows:
            rr = room(a, SIGMA_D[s], ADV[s], wi, ws)
            if rr is None:
                print("  %-9s %-26s %10s %12s %14s"
                      % (s if lab.startswith("A-S57") else "", lab, "-", "NEGATIVE", "-"))
                continue
            print("  %-9s %-26s %9.4f%% %12.4f %14s"
                  % (s if lab.startswith("A-S57") else "", lab, 100 * rr["pov"],
                     rr["net_bps"], "$" + format(int(rr["size_usd"]), ",")))
            res[s][lab] = rr
        print()

    print("WHAT THIS SETTLES")
    print("  1  The two clocks are NOT interchangeable and the frontier needs both.  Forcing")
    print("     them equal at mu_tau makes the edge negative and the room vanish; forcing")
    print("     them equal at 60 is A-S57's published row under a different name.")
    print("  2  A-S69's invariance is now a WARNING, not a comfort: the room is unchanged")
    print("     across every pairing that keeps the EDGE at 60, so the frontier cannot")
    print("     detect which accumulation clock it is using.  It was blind, not robust.")
    print("  3  The frontier therefore RESTS ON AN ASSUMPTION nobody had stated: that a")
    print("     position can be held through the interruption D's mu_tau times.  A-S65")
    print("     supports it -- the continuation lives entirely in the interrupted windows --")
    print("     but that is an argument from another study, not a property of the frontier.")
    print("     It is now written down as a condition rather than assumed silently.")

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"study": "S71_TWO_CLOCKS", "mu_tau": MU_TAU, "hold": HOLD,
         "maker_rt": MAKER_RT, "symbols": res}, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
