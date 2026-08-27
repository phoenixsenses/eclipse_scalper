# -*- coding: utf-8 -*-
"""S70 -- A-S62 says larger events saturate FASTER; D-E11 says larger episodes last LONGER.

THE TENSION
-----------
D-E11, to A:

    "the scalar is not one number -- it SCALES WITH SIZE ... small episodes (median $89k)
     mu_tau 14.14 min, mid (median $302k) 19.35 min, large (median $1.62M) 20.81 min ...
     using the pooled figure OVERSTATES capacity for small episodes by 28% and understates
     it for large ones by 15%."

A-S62 measured the opposite ordering on the same covariate:

    saturation lag, 5%-per-10-min criterion:
        all events   BTC 60   ETH 60   SOL 50 min
        p99 largest  BTC 40   ETH 40   SOL 20 min      -> LARGER SATURATES FASTER

A-S69 argued these are different objects -- mine a HORIZON, D's an OPEN TIME -- and that
argument stands.  But A-S65 then showed that what A-S62 called saturation IS the arrival
process, which is the same family of object D is timing.  Two arrival-process quantities
should not order oppositely in size without a reason.

THE SUSPICION, STATED BEFORE THE MEASUREMENT
--------------------------------------------
A-S62's criterion was RELATIVE: "the first h beyond which G gains less than 5% OF ITS t+60
VALUE per additional 10 minutes".  A larger event has a larger t+60 value, so the same
ABSOLUTE growth clears a relative bar sooner.  If that is the whole story, the ordering
flips under an absolute criterion and A-S62's size-dependence is an artefact of its own
denominator -- which would also dissolve the tension with D-E11 and withdraw the constraint
A-S62 sent to lane C.

Two criteria are run side by side.  Neither is preferred in advance; both are published.

AND ONE CHECK LANE C's C-T49 IMPLIES FOR A-S68
----------------------------------------------
    "On a large-tick instrument 23.3% of 50-trade forward moves are EXACTLY zero, against
     7.5% on BTC and 4.2% on ETH.  Any hit rate, win rate or sign test that does not say
     what it does with those is reporting a different quantity."

A-S68 measured a path at ONE SECOND, where most seconds carry no price change at all.  Its
conclusion ("no peak") could be a resolution artefact if the second-by-second returns are
overwhelmingly ties.  The tie fraction was never reported, so it is reported here.
"""

import io
import json
import math
import os
import sys
import zipfile

PATH = "reports/research/h2_response_shape_v1/S54_THE_PATH_V1.json"
DIR = "data/raw_trades_v1"
OUT = "reports/research/h2_response_shape_v1/S70_RELATIVE_OR_ABSOLUTE_V1.json"
SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
REL = 0.05                      # A-S62's criterion, as published
ABS_BPS = 0.5                   # absolute: fewer than 0.5 bps gained per 10 minutes


def lag(path, K, rel=None, abs_bps=None):
    """First h at which the path stops growing, by the stated criterion."""
    g60 = path[K + 60]
    for h in range(10, 60, 5):
        gain = path[K + min(60, h + 10)] - path[K + h]
        if rel is not None:
            if g60 != 0 and gain / g60 < rel:
                return h
        else:
            if gain < abs_bps:
                return h
    return 60


def tie_fraction(sym, days):
    """Share of consecutive priced SECONDS with an exactly zero log return."""
    tot = z = 0
    for d in days:
        p = "%s/%s-trades-%s.zip" % (DIR, sym, d)
        if not os.path.exists(p):
            continue
        zf = zipfile.ZipFile(p)
        px = {}
        with zf.open(zf.namelist()[0]) as fh:
            fh.readline()
            for line in fh:
                f = line.split(b",")
                px[int(f[4]) // 1000] = f[1]
        secs = sorted(px)
        for a, b in zip(secs, secs[1:]):
            if b == a + 1:
                tot += 1
                if px[a] == px[b]:
                    z += 1
    return z, tot


def main():
    p = json.load(io.open(PATH, encoding="utf-8"))
    K = p["K_minutes"]

    print("PART 1 -- IS A-S62's SIZE ORDERING AN ARTEFACT OF ITS OWN DENOMINATOR?")
    print("  A-S62: 'gains less than 5%% OF ITS t+60 VALUE per 10 min'  -- RELATIVE")
    print("  here also: 'gains less than %.1f bps per 10 min'          -- ABSOLUTE" % ABS_BPS)
    print("  %-9s %-12s %10s %12s %12s %12s"
          % ("symbol", "set", "G(t+60)", "lag RELATIVE", "lag ABSOLUTE", "flip?"))
    res = {"lags": {}}
    for s in SYMS:
        rows = {}
        for tag in ("all", "p99 largest"):
            path = p["symbols"][s][tag]["path"]
            lr = lag(path, K, rel=REL)
            la = lag(path, K, abs_bps=ABS_BPS)
            rows[tag] = {"g60": path[K + 60], "lag_rel": lr, "lag_abs": la}
            print("  %-9s %-12s %10.2f %12d %12d" % (s, tag, path[K + 60], lr, la))
        f_rel = "faster" if rows["p99 largest"]["lag_rel"] < rows["all"]["lag_rel"] else \
                ("slower" if rows["p99 largest"]["lag_rel"] > rows["all"]["lag_rel"] else "same")
        f_abs = "faster" if rows["p99 largest"]["lag_abs"] < rows["all"]["lag_abs"] else \
                ("slower" if rows["p99 largest"]["lag_abs"] > rows["all"]["lag_abs"] else "same")
        print("  %-9s %-12s %10s %12s %12s %12s"
              % ("", "-> larger is", "", f_rel, f_abs, "FLIPS" if f_rel != f_abs else "no"))
        res["lags"][s] = rows
        print()

    print("  D-E11 measures the OPEN TIME and finds larger episodes last LONGER")
    print("  (14.14 -> 19.35 -> 20.81 min).  If the absolute criterion also says 'slower',")
    print("  the two lanes agree and A-S62's relative reading was the odd one out.")

    print()
    print("PART 2 -- C-T49's CHECK APPLIED TO A-S68: HOW MANY ONE-SECOND RETURNS ARE TIES?")
    print("  A-S68 concluded 'no peak' from a one-second path and never reported this.")
    days = ["2026-08-%02d" % d for d in range(7, 14)]
    ties = {}
    for s in sys.argv[1:] or ["BTCUSDT"]:
        z, tot = tie_fraction(s, days)
        pct = 100.0 * z / tot if tot else float("nan")
        print("  %-9s consecutive priced seconds %s   EXACT ties %s  (%.1f%%)"
              % (s, format(tot, ","), format(z, ","), pct))
        ties[s] = {"pairs": tot, "ties": z, "pct": pct}
    res["ties"] = ties

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"study": "S70_RELATIVE_OR_ABSOLUTE", "rel": REL, "abs_bps": ABS_BPS,
         "d_e11_mu_tau": {"small": 14.14, "mid": 19.35, "large": 20.81}, **res}, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
