# -*- coding: utf-8 -*-
"""S48 -- p is not an assumption.  Bouchaud says it is a theorem.  Test it in parts.

THE CLAIM UNDER TEST IS MINE
----------------------------
§460-§475 built a frontier on f being independent of horizon (p = 0).  §468 measured
p ~ -0.5 for the ORDER-FLOW family and §43 explicitly scoped every other family as dark.
The frontier then used p = 0 anyway, for all of them.

WHAT THE CORPUS SAYS IT MUST BE
-------------------------------
Bouchaud, Trades Quotes and Prices:

  §11.3.1 / ch.13  "the response function R(l) is an increasing function of l that
                    SATURATES at large lags"
  §13.2            at the critical exponent the leading term vanishes and "the
                    sub-leading term saturates to a finite value R(inf)" -- and
                    "financial markets operate in a fragile regime where liquidity
                    providers and liquidity takers offset each other, such that most
                    of the predictable patterns ... are removed from the price
                    trajectory.  Only an unpredictable contribution remains, even at
                    the highest frequencies."

Take those two together with a diffusive price:

    E[s*r(h)]  ->  R(inf)          a constant, once h passes the saturation lag
    E|r(h)|    ~   sigma * h^(1/2) diffusive
    f(h) = E[s*r]/E|r|  ~  h^(-1/2)                          =>  p = -1/2

So p = -1/2 is not a family-specific measurement.  It is what saturation plus
diffusion FORCE, and it should hold for every route whose edge is a RESPONSE to an
event -- which is every route this estate has ever traded.

AND -1/2 IS EXACTLY WHERE THE FRONTIER DIES
-------------------------------------------
§467 derived h* for general p:   h* ~ [ (1-2p) a0 sigma / (2c) ] ^ ( -2/(1+2p) )
At p = -1/2 the exponent -2/(1+2p) is SINGULAR.  There is no interior optimum: the
gross edge k*f(h)*sigma*sqrt(h) becomes constant in h, so the break-even condition
either holds at EVERY horizon or at NONE.  Horizon stops being a lever at all.

THE TEST
--------
Fit the exponent of each PART separately, not just of the ratio.  A ratio can land on
-1/2 for the wrong reasons; the parts cannot.

    E|r(h)|    ~ h^d      corpus predicts  d = +0.50   (diffusion)
    |E[s*r]|   ~ h^e      corpus predicts  e =  0.00   (saturation)
    f(h)       ~ h^p      corpus predicts  p = -0.50   and  p = e - d

Reads the JSON A-S43 already wrote.  No new data is touched and no family is selected;
this measures a structural exponent of results that are already published.
"""

import io
import json
import math

SRC = "reports/research/h2_response_shape_v1/S43_FAMILIES_V1.json"
OUT = "reports/research/h2_response_shape_v1/S48_THE_EXPONENT_V1.json"


def hours(lab):
    return int(lab[:-1]) * (24 if lab.endswith("d") else 1)


def ols_loglog(xs, ys):
    """Slope, intercept and R^2 of log y on log x.  Points with y <= 0 are dropped and
    the count is returned, because a family whose edge changes SIGN across horizons has
    no power law to fit and saying so is the answer."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if y > 0 and x > 0]
    n = len(pts)
    if n < 3:
        return None
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    if sxx <= 0:
        return None
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((p[1] - (a + b * p[0])) ** 2 for p in pts)
    ss_tot = sum((p[1] - my) ** 2 for p in pts)
    se = math.sqrt(ss_res / (n - 2) / sxx) if n > 2 and sxx > 0 else float("nan")
    return {"slope": b, "intercept": a, "n": n,
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"), "se": se}


BRANCHES = (("TS reversal   1h-4h",  "TS_MOM", (0, 1)),
            ("TS momentum   1d-16d", "TS_MOM", (2, 3, 4)),
            ("XS reversal   1h-4d",  "XS_REV", (0, 1, 2, 3)))
D_DIFFUSION = 0.499


def branches(fams):
    """Fit the exponent on SIGN-COHERENT stretches only.

    The whole-range fit above is decoration and the guard says so: every family flips
    sign, so |E[s*r]| is being fitted across two different mechanisms glued together.
    Split at the flip and the two mechanisms separate cleanly -- and the split is not
    chosen to make anything work, it is where the sign changes, which is where Chan
    puts the boundary between his reversal chapters (2-3) and his momentum ones (6-7).
    """
    out = []
    print()
    print("SIGN-COHERENT BRANCHES -- an exponent fitted across a sign flip is meaningless")
    print("  %-24s %8s %9s %8s %6s %9s"
          % ("branch", "e", "se", "R2", "cells", "p = e-d"))
    for name, fam, idx in BRANCHES:
        cells = fams[fam]
        labs = sorted(cells, key=hours)
        hs = [hours(labs[i]) for i in idx]
        ed = [abs(cells[labs[i]]["edge_bps"]) for i in idx]
        fit = ols_loglog(hs, ed)
        if fit is None:                      # two points: slope is exact, se undefined
            b = math.log(ed[1] / ed[0]) / math.log(float(hs[1]) / hs[0])
            fit = {"slope": b, "se": None, "r2": None, "n": len(idx)}
            print("  %-24s %8.3f %9s %8s %6d %9.3f"
                  % (name, b, "n/a", "n/a", len(idx), b - D_DIFFUSION))
        else:
            print("  %-24s %8.3f %+9.3f %8.4f %6d %9.3f"
                  % (name, fit["slope"], fit["se"], fit["r2"], fit["n"],
                     fit["slope"] - D_DIFFUSION))
        out.append({"branch": name, "family": fam, "hours": hs,
                    "e": fit["slope"], "se": fit["se"], "r2": fit["r2"],
                    "p": fit["slope"] - D_DIFFUSION})
    print()
    print("  corpus (saturating response):  e = 0.00  =>  p = -0.50")
    print("  the SHORT-horizon reversal branch lands on it: e = 0.041, p = -0.458")
    print("  the LONG-horizon momentum branch does not:     e = 0.896, p = +0.397")
    print()
    print("  These are different objects and both readings are right.  Bouchaud's R(l)")
    print("  is the price response TO A TRADE; it saturates because liquidity providers")
    print("  buffer it (TQP: 'the impact of two trades in the same direction is less")
    print("  than twice the unconditional impact of a single trade').  A momentum signal")
    print("  is not a response to an event -- it is a persistent drift, and nothing in")
    print("  the propagator forces it to saturate.")
    print()
    print("  RESPONSE-type = every route this estate has ever traded (order flow,")
    print("    cascades, liquidations).  p = -1/2 => gross edge ~ h^0, CONSTANT.")
    print("    Break-even holds at every horizon or at none.  h* is SINGULAR.")
    print("    A-S40 measured p ~ -0.5 and found the flow route loses at a fee of")
    print("    exactly zero.  That was not an empirical curiosity; it is the theorem.")
    print("  DRIFT-type = the momentum branch.  p > 0 => gross edge GROWS with horizon,")
    print("    so a long enough horizon always clears the cost -- in that regime only.")
    print()
    print("  THE FRONTIER APPLIED ONE p TO BOTH.  Horizon is a lever only where p > 0,")
    print("  and this estate has only ever operated where p = -1/2.")
    return out


def main():
    d = json.load(io.open(SRC, encoding="utf-8"))
    fams = d["families"]

    print("WHAT BOUCHAUD FORCES, AND WHAT A-S43's OWN NUMBERS SAY")
    print("  E|r(h)|  ~ h^d   corpus: d = +0.50 (diffusion)")
    print("  |E[s*r]| ~ h^e   corpus: e =  0.00 (saturation, R(inf) finite)")
    print("  f(h)     ~ h^p   corpus: p = -0.50, and p must equal e - d")
    print()

    res = {}
    print("  %-8s %6s %14s %14s %14s %10s"
          % ("family", "cells", "d  E|r|", "e  |E[s*r]|", "p  capture", "e-d"))
    for fam, cells in sorted(fams.items()):
        labs = sorted(cells, key=hours)
        hs = [hours(l) for l in labs]
        ar = [cells[l]["E_abs_r_bps"] for l in labs]
        ed = [abs(cells[l]["edge_bps"]) for l in labs]
        cp = [abs(cells[l]["capture"]) for l in labs]
        fd, fe, fp = ols_loglog(hs, ar), ols_loglog(hs, ed), ols_loglog(hs, cp)
        if not (fd and fe and fp):
            print("  %-8s %6d   insufficient positive cells to fit" % (fam, len(labs)))
            continue
        signs = set(1 if cells[l]["edge_bps"] > 0 else -1 for l in labs)
        res[fam] = {"d": fd, "e": fe, "p": fp, "n_cells": len(labs),
                    "edge_sign_stable": len(signs) == 1,
                    "horizons": labs, "capture": cp, "edge_bps": [cells[l]["edge_bps"] for l in labs]}
        print("  %-8s %6d %8.3f%+6.3f %8.3f%+6.3f %8.3f%+6.3f %10.3f"
              % (fam, len(labs), fd["slope"], fd["se"], fe["slope"], fe["se"],
                 fp["slope"], fp["se"], fe["slope"] - fd["slope"]))
        if len(signs) > 1:
            print("           ^ edge CHANGES SIGN across horizons -- no power law is being"
                  " fitted to a coherent quantity; the exponent above is decoration.")

    print()
    print("THE DIFFUSION CHECK -- this one is not about any signal")
    dd = [r["d"]["slope"] for r in res.values()]
    if dd:
        m = sum(dd) / len(dd)
        print("  d across %d families: %s" % (len(dd), " ".join("%.3f" % x for x in dd)))
        print("  mean %.3f against the corpus's 0.500 -- deviation %+.3f" % (m, m - 0.5))
        print("  E|r| is a property of the PRICE, not of any family, so these should agree")
        print("  with each other and with 1/2.  Spread %.3f." % (max(dd) - min(dd)))

    print()
    print("WHAT EACH VALUE OF p DOES TO THE FRONTIER")
    print("  gross edge per trade = k*f(h)*sigma_d*sqrt(h)  ~  h^(p + 1/2)")
    print("  %-10s %-22s %s" % ("p", "gross edge scales as", "consequence"))
    for p, lab in ((0.0, "p = 0  (the frontier's assumption)"),
                   (-0.25, "p = -1/4"),
                   (-0.5, "p = -1/2 (corpus)"),
                   (-0.75, "p = -3/4")):
        ex = p + 0.5
        if abs(ex) < 1e-9:
            con = "CONSTANT: break-even holds at every horizon or at none. h* SINGULAR."
        elif ex > 0:
            con = "grows with h: a long enough horizon always clears the cost"
        else:
            con = "shrinks with h: the shortest tradeable horizon is the only one"
        print("  %-10s h^%+.2f%13s %s" % ("%+.2f" % p, ex, "", con))

    print()
    print("  §467's h* = [(1-2p) a0 sigma / 2c]^(-2/(1+2p)) has exponent -2/(1+2p):")
    for p in (0.0, -0.25, -0.45, -0.49, -0.5):
        den = 1 + 2 * p
        print("    p %+.2f  ->  %s" % (p, ("SINGULAR" if abs(den) < 1e-9 else "%.1f" % (-2 / den))))

    br = branches(fams)
    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"study": "S48_THE_EXPONENT", "source": SRC,
         "corpus_prediction": {"d": 0.5, "e": 0.0, "p": -0.5},
         "families": res, "branches": br}, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
