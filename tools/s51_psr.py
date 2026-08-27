# -*- coding: utf-8 -*-
"""S51 -- does non-normality invalidate the frozen prereg's bar?

THE OBJECTION
-------------
LANE_A_PREREG_V1 §7 sets the bar at t >= 2.0 and §8 derives N_required = 329,726 from
(2.0 / S_per_trade)^2.  Both assume Normal returns.  This estate is full of fat tails --
echo's 14 catastrophic tails with a worst of -338.9, and a 4h tail rate of 21.7% that
§162/§163 found IRREDUCIBLE.  A bar built on Normality looks like the wrong bar.

THE CORPUS'S CORRECTION
-----------------------
Lopez de Prado MLAM §8, the Probabilistic Sharpe Ratio:

    z[SR*] = (SR - SR*) * sqrt(T-1) / sqrt( 1 - g3*SR + ((g4-1)/4)*SR^2 )

with g3 = skewness and g4 = kurtosis of the returns, SR NON-ANNUALISED at the
observation frequency.  LdP's own worked example (skew -3, kurtosis 10, T = 1250) puts
the familywise error at 0.0608 where Normality would have said 0.0261 -- non-Normality
alone costs a factor of 2.33, and he calls assuming Normality "a gross underestimation
of the type I error probability".

WHAT THIS DRIVER CHECKS
-----------------------
Whether that factor bites HERE.  Two reasons to suspect it may not, both of which have
to be measured rather than asserted:

  1  the correction enters through g3*SR, and this estate's per-trade Sharpe is 0.003483
     (§475: k*f/2, symbol-independent).  A skew of -3 against an SR of 0.0035 moves the
     denominator by about half a per cent.  The correction is PROPORTIONAL to SR, and a
     tiny SR cannot be rescued or ruined by skewness.
  2  aggregational gaussianity: returns get closer to Normal as the horizon grows, and
     the frontier's h* is 30 days at the median.  Both effects push the same way.

If both hold, the fat-tail objection to the frozen bar FAILS -- and that is worth having
in writing, because it is an objection anyone would raise and nobody had answered.

Measures skew and kurtosis on the estate's own returns at every horizon rather than
importing LdP's hedge-fund figures.  No outcome is read.
"""

import io
import json
import math
import sqlite3

DB = "file:data/xsec_klines_ext.db?mode=ro"
CUT = 1787270400000
OUT = "reports/research/h2_response_shape_v1/S51_PSR_V1.json"

S_PER_TRADE = 0.6966 * 0.010 / 2.0       # prereg §8: k*f_design/2, symbol-independent
T_BAR = 2.0
HORIZONS_H = (1, 4, 24, 96, 384, 720)    # 1h .. 30d, spanning the frontier's h* range
MIN_SIGMA_D_BPS = 50.0


def moments(hours):
    """Pooled skewness and excess-kurtosis-plus-3 of log returns at one horizon.

    Pooled across symbols after standardising each symbol by its OWN sd: otherwise the
    pooled kurtosis measures the dispersion of volatilities across symbols rather than
    the tail of any return distribution.  That mixture effect is itself fat-tailed and
    would be mistaken for the thing being measured.
    """
    ms = hours * 3600000
    c = sqlite3.connect(DB, uri=True)
    rows = c.execute(
        "SELECT symbol, open_time/%d AS b, MAX(open_time), close FROM klines "
        "WHERE open_time < %d GROUP BY symbol, b ORDER BY symbol, b" % (ms, CUT)).fetchall()
    c.close()
    by = {}
    for s, b, _t, cl in rows:
        if cl and cl > 0:
            by.setdefault(s, []).append((int(b), float(cl)))
    z = []
    nsym = 0
    for s, v in by.items():
        r = [math.log(b[1] / a[1]) for a, b in zip(v, v[1:]) if b[0] == a[0] + 1 and a[1] > 0]
        if len(r) < 100:
            continue
        m = sum(r) / len(r)
        sd = math.sqrt(sum((x - m) ** 2 for x in r) / (len(r) - 1))
        if sd <= 0:
            continue
        nsym += 1
        z.extend((x - m) / sd for x in r)
    n = len(z)
    if n < 1000:
        return None
    m2 = sum(x * x for x in z) / n
    m3 = sum(x ** 3 for x in z) / n
    m4 = sum(x ** 4 for x in z) / n
    return {"n": n, "n_sym": nsym, "skew": m3 / m2 ** 1.5, "kurt": m4 / m2 ** 2}


def psr_factor(sr, g3, g4):
    """The denominator of LdP's z.  >1 means the Normal bar was too LENIENT."""
    v = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr
    return math.sqrt(v) if v > 0 else float("nan")


def main():
    print("DOES NON-NORMALITY MOVE THE FROZEN BAR?")
    print("  prereg bar t >= %.1f   per-trade Sharpe %.6f (k*f/2, symbol-independent)"
          % (T_BAR, S_PER_TRADE))
    print("  LdP correction:  z = (SR-SR*)*sqrt(T-1) / sqrt(1 - g3*SR + ((g4-1)/4)*SR^2)")
    print()
    print("  %-8s %10s %8s %10s %12s %14s %12s"
          % ("horizon", "n returns", "symbols", "skew", "kurtosis", "PSR factor", "N_required"))

    res = {}
    for h in HORIZONS_H:
        m = moments(h)
        if not m:
            continue
        f = psr_factor(S_PER_TRADE, m["skew"], m["kurt"])
        nreq = (T_BAR * f / S_PER_TRADE) ** 2
        lab = ("%dh" % h) if h < 24 else ("%dd" % (h // 24))
        print("  %-8s %10s %8d %10.3f %12.2f %14.5f %12s"
              % (lab, format(m["n"], ","), m["n_sym"], m["skew"], m["kurt"], f,
                 format(int(nreq), ",")))
        res[lab] = dict(m, psr_factor=f, n_required=nreq)

    base = (T_BAR / S_PER_TRADE) ** 2
    print()
    print("  Normal-assumption N_required (the frozen value): %s" % format(int(base), ","))
    if res:
        worst = max(res.values(), key=lambda r: r["psr_factor"])
        print("  worst horizon multiplies it by %.5f -- i.e. by %.2f%%"
              % (worst["psr_factor"] ** 2, 100 * (worst["psr_factor"] ** 2 - 1)))

    print()
    print("WHY IT DOES NOT BITE, AND WHERE IT WOULD")
    print("  The correction enters as g3*SR.  It is PROPORTIONAL to the Sharpe ratio, so")
    print("  a tiny per-trade Sharpe cannot be rescued or ruined by skewness.")
    print("  %-14s %12s %12s %12s" % ("per-trade SR", "at skew -1", "skew -3", "skew -6"))
    for sr, lab in ((S_PER_TRADE, "0.0035 HERE"), (0.05, "0.05"), (0.20, "0.20"),
                    (0.50, "0.50"), (1.00, "1.00")):
        row = [psr_factor(sr, g, 10.0) for g in (-1.0, -3.0, -6.0)]
        print("  %-14s %12.4f %12.4f %12.4f" % (lab, row[0], row[1], row[2]))
    print()
    print("  LdP's own example sits at SR = 0.0791 with skew -3 and kurtosis 10, and there")
    print("  the factor is %.4f -- enough to move a p-value from 0.026 to 0.061."
          % psr_factor(0.0791, -3.0, 10.0))
    print("  This estate's per-trade Sharpe is %.1fx smaller." % (0.0791 / S_PER_TRADE))

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"study": "S51_PSR", "s_per_trade": S_PER_TRADE, "bar": T_BAR,
         "n_required_normal": base, "horizons": res,
         "ldp_example_factor": psr_factor(0.0791, -3.0, 10.0)}, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
