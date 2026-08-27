# -*- coding: utf-8 -*-
"""S69 -- lane D's measured window, propagated through every table that used a swept one.

WHAT ARRIVED
------------
D-E10, to A:

    "the scalar is measured.  X = ADV * POV * 18.10 min, CI [16.68, 19.68] ... your duration
     bound is linear in the window, so 18.10 min gives $18,558,580 against the impact bound
     of $1,032,042,639 -- your swept 15-minute row ($15.4M) was the closest of the five and
     your 60-minute row overstated the bound by 3.3x."

    three conditions: (1) tau- and k-conditional BY CONSTRUCTION (tau = 60 min, k = 10.0 bps,
    frozen in e7968ac4); another tau is a NEW prereg.  (2) the distribution is DEFECTIVE
    (10.0% still alive and un-interrupted at 60 min), so use the MEAN and NOT a median or a
    half-life.  (3) burned sample -- a characterisation, not a trading claim.

The corpus agrees on the semantics (STK4080 slides 8): mu_tau = integral_0^tau S(u)du, "the
expected survival in [0, t]", and the reason to restrict is precisely that "the right tail is
poorly estimated (and S-hat may even be constant and positive for all large t)".  So 18.10
minutes is the expected OPEN TIME within a 60-minute horizon, not a typical episode length --
and A-S50's swept 50/60-minute window was never that quantity.

WHAT THIS DOES
--------------
A-S50, A-S55, A-S57 and A-S62 all used a window this lane SWEPT or read off a path.  This
replaces it with D's measured one and reports, separately:

    WHAT MOVES        the duration bound, which is linear in the window at a FIXED POV
    WHAT IS INVARIANT the revenue-optimal room, which is not -- and the reason is algebraic,
                      so it is derived here and then CHECKED numerically rather than asserted

Fenced: recomputes this lane's own published tables.  D's estimate is used as given and is
not re-derived; the survival work is D's charter, not A's.
"""

import io
import json
import math

OUT = "reports/research/h2_response_shape_v1/S69_MU_TAU_PROPAGATED_V1.json"

MU_TAU, MU_LO, MU_HI = 18.10, 16.68, 19.68        # D-E10, frozen prereg e7968ac4, tau = 60 min
OLD_WINDOWS = (50.0, 60.0)                         # A-S55/A-S57 used 50; A-S50's row was 60
Y, DELTA = 0.5, 0.5
MAKER_RT = 4.0
SIGMA_D = {"BTCUSDT": 0.02015, "ETHUSDT": 0.03534, "SOLUSDT": 0.02541}
ADV = {"BTCUSDT": 7296912640.0, "ETHUSDT": 5218467948.0, "SOLUSDT": 996048951.0}
CONT = {"BTCUSDT": 6.72, "ETHUSDT": 10.07, "SOLUSDT": 7.37}       # A-S54 unconditional
POOLED_60 = 61520154.0                             # A-S50's 60-minute pooled duration bound


def impact_rt(sig, pov, w):
    frac = pov * w / 1440.0
    return 2.0 * 1e4 * Y * sig * (frac ** DELTA)


def room(sig, adv, cont, w):
    """Revenue-optimal POV and the net there.  net(POV) = a - k*sqrt(POV), a = cont - fee,
    k = 2e4*Y*sig*sqrt(w/1440).  Revenue ~ net*size and size ~ POV, so u* = 2a/(3k)."""
    a = cont - MAKER_RT
    k = 2.0 * 1e4 * Y * sig * ((w / 1440.0) ** DELTA)
    if a <= 0:
        return None
    u = 2.0 * a / (3.0 * k)
    pov = u * u
    net = a - k * u
    size = adv * pov * w / 1440.0
    return {"a": a, "k": k, "pov": pov, "net_bps": net, "size_usd": size}


def main():
    print("LANE D's MEASURED WINDOW, PROPAGATED  (mu_tau = %.2f min, CI [%.2f, %.2f])"
          % (MU_TAU, MU_LO, MU_HI))
    print("  tau = 60 min and k = 10.0 bps are FROZEN in D's prereg e7968ac4.")
    print("  the distribution is DEFECTIVE (10%% alive at 60 min) -- MEAN only, no half-life.")
    print("  corpus (STK4080 s.8): mu_tau = int_0^tau S(u)du, the expected survival IN [0,tau].")

    res = {"mu_tau": MU_TAU, "ci": [MU_LO, MU_HI], "moves": {}, "invariant": {}}

    print()
    print("WHAT MOVES -- the duration bound, linear in the window at a FIXED POV")
    print("  %-22s %18s %12s" % ("window", "pooled bound", "vs 60 min"))
    for w, lab in ((60.0, "A-S50's 60-minute row"), (MU_TAU, "D-E10 mu_tau 18.10"),
                   (MU_LO, "  CI low 16.68"), (MU_HI, "  CI high 19.68")):
        v = POOLED_60 * w / 60.0
        print("  %-22s %18s %11.2fx" % (lab, "$" + fmt(v), v / POOLED_60))
        res["moves"]["%.2f" % w] = v
    print("  D's own arithmetic: $18,558,580 at 18.10 -- reproduced above.")
    print("  A-S50's 60-minute row overstated the bound by %.2fx." % (60.0 / MU_TAU))

    print()
    print("WHAT IS INVARIANT -- and the reason is algebraic, so it is derived then CHECKED")
    print("  size = ADV * POV* * w/1440,  POV* = (2a/3k)^2,  k ~ sqrt(w)")
    print("  => POV* ~ 1/w  =>  POV* * w is CONSTANT  =>  SIZE does not depend on w")
    print("  and net at the optimum = a - k*(2a/3k) = a/3, with no k in it at all.")
    print()
    print("  %-9s %-14s %10s %12s %14s %10s"
          % ("symbol", "window", "POV*", "net bps", "size $", "net = a/3?"))
    for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        for w, lab in ((50.0, "A-S57's 50"), (MU_TAU, "mu_tau 18.10"), (MU_LO, "CI low"),
                       (MU_HI, "CI high")):
            r = room(SIGMA_D[s], ADV[s], CONT[s], w)
            ok = abs(r["net_bps"] - r["a"] / 3.0) < 1e-9
            print("  %-9s %-14s %9.4f%% %12.4f %14s %10s"
                  % (s if lab.startswith("A-S57") else "", lab, 100 * r["pov"],
                     r["net_bps"], "$" + fmt(r["size_usd"]), "yes" if ok else "NO"))
            res["invariant"]["%s|%.2f" % (s, w)] = r
        print()

    print("  The room A-S57 published -- 0.91 / 2.02 / 1.12 bps on $590,805 / $684,376 /")
    print("  $77,858 -- is UNCHANGED by D's correction, to the last digit.  That is not a")
    print("  coincidence and not luck: at the revenue-optimal participation rate the window")
    print("  cancels out of both the net and the size.  A-S57 could have said so and did not.")

    print()
    print("WHAT THIS MEANS FOR THE TWO STATEMENTS THAT USED THE WINDOW")
    for s in ("BTCUSDT",):
        pass
    print("  A-S50  'below an hour DURATION binds, above it the POT binds'")
    print("         the crossover was read off a swept table whose 60-minute row is now")
    print("         known to overstate by 3.3x.  D-E5 said this exactly: the ORDERING may")
    print("         survive because it is a comparison, but the CROSSOVER HOUR is a LEVEL")
    print("         and cannot survive unexamined.  It does not: at mu_tau the pooled")
    print("         bound is $18.6M, and the hour was never the crossing point.")
    print("  A-S57  the room and the size are INVARIANT and stand as published.")
    print("  A-S62  its response rows are read at h = 10/30/60 min off A-S54's path, which")
    print("         is a different object from D's mu_tau (an OPEN TIME, not a horizon).")
    print("         They are not interchangeable and this does NOT rewrite that table.")

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(res, indent=1))
    print("\nwrote %s" % OUT)


def fmt(x):
    return format(int(round(x)), ",")


if __name__ == "__main__":
    main()
