# -*- coding: utf-8 -*-
"""LANE A EVALUATOR V1 -- the machine that executes LANE_A_PREREG_V1.md.

The prereg's success condition is "a prereg an independent reader can execute without
asking A anything."  This file is the executable half of that.  Everything it needs is
either in the prereg or measured here; nothing is passed in that could change a verdict.

IT REFUSES TO RUN BY DEFAULT.  Three interlocks, because the charter's one hard rule is
that no outcome may be read before the prereg is frozen:

  1  the prereg file must exist, hash to the value recorded in FROZEN_SHA256, and say
     STATUS  FROZEN.  A draft cannot be evaluated.
  2  --evaluate must be passed explicitly.  There is no default that reads an outcome.
  3  the evaluation window must START after the freeze timestamp.  This is the interlock
     that matters: it is what makes the data fresh.  It is checked against the timestamp
     INSIDE the frozen file, not against anything on the command line.

Without --evaluate the tool runs in the only other mode it has: --dry, which reports the
universe, the derived horizons, the accrued N_eff and the distance to N_required, and
touches no return.  --dry is safe to run at any time and is the intended way to answer
"is it time yet".

  python -m tools.lane_a_evaluator_v1 --dry
  python -m tools.lane_a_evaluator_v1 --evaluate --start 2026-09-01 --end 2027-03-01

Prereg: reports/atlas/LANE_A_PREREG_V1.md   (that file is authoritative; if this code and
that file disagree, the file wins and this code is the defect).
"""

import argparse
import hashlib
import io
import json
import math
import os
import re
import sqlite3
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG = os.path.join(REPO, "reports", "atlas", "LANE_A_PREREG_V1.md")
ADDENDA = [os.path.join(REPO, "reports", "atlas", "LANE_A_PREREG_V1_ADDENDUM_A.md")]
PANEL = os.path.join(REPO, "data", "xsec_klines_ext.db")
OUT = os.path.join(REPO, "reports", "atlas", "LANE_A_RESULT_V1.json")

# ---------------------------------------------------------------- frozen constants
# Every one of these is derived in the prereg.  None is a choice made here.
F_DESIGN = 0.010          # prereg section 4 -- the LOW end of A-S14's single-leg range,
                          # deliberately low because being short of h* is catastrophic
                          # and being long of it costs 25%.
K_MEAN_ABS = 0.6966       # prereg section 3 -- E|r|/sigma, measured (A-S39/§467)
T_BAR = 2.0               # prereg section 7 -- one-sided
MIN_SYMBOLS = 20          # prereg section 8, S3
COV_DAYS = 150            # prereg section 5, U1
COV_WINDOW = 180
COV_BARS = 1200
MIN_SIGMA_BPS = 50.0      # U4
DEFAULT_L = 5.0e6         # U2a default when Q_intended is not supplied
LAWFUL_CUTOFF_MS = 1787270400000   # 2026-08-21; the panel crosses it

# N_required = (T_BAR / (K_MEAN_ABS * F_DESIGN / 2))^2 -- recomputed, never hardcoded.
N_REQUIRED = int(round((T_BAR / (K_MEAN_ABS * F_DESIGN / 2.0)) ** 2))


# ---------------------------------------------------------------- interlocks
def read_prereg():
    if not os.path.exists(PREREG):
        die("prereg not found: %s" % PREREG)
    raw = io.open(PREREG, "rb").read()
    txt = raw.decode("utf-8")
    sha = hashlib.sha256(raw).hexdigest()
    status = grab(txt, r"^STATUS\s+(\S+)", "STATUS")
    frozen_sha = grab(txt, r"^sha256 of this file\s+(\S+)", "sha256")
    frozen_at = grab(txt, r"^frozen at\s+(\S+)", "frozen at")
    return {"sha_now": sha, "status": status, "sha_frozen": frozen_sha,
            "frozen_at": frozen_at, "text": txt}


def grab(txt, pat, what):
    m = re.search(pat, txt, re.M)
    if not m:
        die("prereg does not carry a %s line -- it is not a frozen document" % what)
    return m.group(1)


def check_frozen(p, start_ms):
    """The three interlocks.  Any failure is fatal; none is overridable by a flag."""
    if p["status"] != "FROZEN":
        die("prereg STATUS is %r, not FROZEN.  A draft cannot be evaluated." % p["status"])
    # The hashed SUBJECT is everything before the freeze block: unambiguous, and it needs
    # no un-substitution of values written at freeze time.  The first attempt hashed the
    # draft with placeholders and tried to reconstruct it by reversing three edits; a
    # fourth edit (the status line in the header) was missed and the check failed closed.
    # It failing closed was correct, but a rule that requires listing every edit is a rule
    # that will eventually miss one.
    marker = "## 11 " + chr(0xB7) + " Freeze block"
    if marker not in p["text"]:
        die("prereg has no freeze block marker")
    subject = p["text"].split(marker)[0]
    got = hashlib.sha256(subject.encode("utf-8")).hexdigest()
    if got != p["sha_frozen"]:
        nl = chr(10)
        die("prereg body does not match its frozen hash." + nl
            + "  recorded %s" % p["sha_frozen"] + nl
            + "  computed %s" % got + nl
            + "  the document was edited after freezing.  Evaluation refused.")
    check_addenda()
    fz = iso_ms(p["frozen_at"])
    if start_ms is not None and start_ms < fz:
        die("evaluation window starts %s, BEFORE the freeze at %s.\n"
            "  That data existed when the hypothesis was written.  Refused."
            % (ms_iso(start_ms), p["frozen_at"]))


def check_addenda():
    """An addendum that is not verified is decoration.  Each binds to the prereg body hash
    and is hashed on the same terms: everything above its own freeze block."""
    for path in ADDENDA:
        if not os.path.exists(path):
            die("addendum missing: %s -- the frozen record is incomplete" % path)
        txt = io.open(path, encoding="utf-8").read()
        rec = grab(txt, r"^sha256 of this file    (\S+)", "addendum sha256")
        got = hashlib.sha256(txt.split("## Freeze block")[0].encode("utf-8")).hexdigest()
        if got != rec:
            die("addendum edited after freezing: %s" % os.path.basename(path))
        print("  addendum verified: %s" % os.path.basename(path))


def die(msg):
    sys.stderr.write("REFUSED: %s\n" % msg)
    raise SystemExit(2)


def iso_ms(s):
    s = s.replace("Z", "").replace("T", " ")
    for f in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return int(time.mktime(time.strptime(s, f)) * 1000)
        except ValueError:
            pass
    die("cannot parse timestamp %r" % s)


def ms_iso(ms):
    return time.strftime("%Y-%m-%d", time.localtime(ms / 1000.0))


# ---------------------------------------------------------------- the universe rule
def universe(con, end_ms):
    """Prereg section 5.  Mechanical, outcome-free.  No return enters this function's
    filter -- sigma is a COST quantity here (it sets the horizon), not a performance one,
    and U4 uses it only to exclude pegged instruments."""
    lo = end_ms - COV_WINDOW * 86400000
    rows = con.execute("""
        SELECT symbol,
               COUNT(DISTINCT DATE(open_time/1000,'unixepoch'))                 AS days,
               COUNT(*)                                                     AS bars,
               AVG(quote_volume)                                            AS qv,
               COUNT(DISTINCT CASE WHEN close<>open THEN 1 END)             AS moved
        FROM klines
        WHERE open_time >= ? AND open_time < ?
        GROUP BY symbol""", (lo, end_ms)).fetchall()

    keep, rej = [], {}
    for sym, days, bars, qv, moved in rows:
        if days < COV_DAYS:
            rej[sym] = "U1 days=%d" % days
            continue
        if bars < days * COV_BARS:
            rej[sym] = "U1 bars"
            continue
        if (qv or 0) * 1440.0 < DEFAULT_L:
            rej[sym] = "U2 notional"
            continue
        if not moved:
            rej[sym] = "U3 no tick"
            continue
        keep.append(sym)
    return keep, rej


def daily_sigmas(con, lo, hi):
    """Daily sigma in bps for every symbol, from close-to-close, in ONE pass.

    SQLite's bare-column rule makes `MAX(open_time), close` under a GROUP BY return the
    close from the row carrying that max -- the day's last bar.  A symbol with fewer than
    30 usable days returns nothing: an UNMEASURED sigma is not a zero, and U4 must not be
    able to admit a symbol by silence."""
    rows = con.execute("""
        SELECT symbol, DATE(open_time/1000,'unixepoch') AS d, MAX(open_time), close
        FROM klines WHERE open_time >= ? AND open_time < ?
        GROUP BY symbol, d ORDER BY symbol, d""", (lo, hi)).fetchall()
    series = {}
    for sym, d, _t, c in rows:
        if c:
            series.setdefault(sym, []).append(c)
    out = {}
    for sym, cl in series.items():
        rets = [math.log(b / a) for a, b in zip(cl, cl[1:]) if a > 0 and b > 0]
        if len(rets) < 30:
            continue
        m = sum(rets) / len(rets)
        v = sum((x - m) ** 2 for x in rets) / (len(rets) - 1)
        out[sym] = math.sqrt(v) * 1e4
    return out


def h_star_days(sigma_bps, cost_bps):
    """Prereg section 3.  h* = [2c / (k f sigma_d)]^2, from A-S32's closed form.
    Horizon is DERIVED here, never passed in."""
    denom = K_MEAN_ABS * F_DESIGN * sigma_bps
    if denom <= 0:
        return None
    return (2.0 * cost_bps / denom) ** 2


# ---------------------------------------------------------------- N_eff, section 6
def fmt_years(y):
    return ("%.1f y" % y) if y < 1000 else ("%s y" % format(int(y), ","))


def corr_matrix(con, syms, lo, hi):
    """Correlation of daily close-to-close returns across the admitted universe.

    Built on the INTERSECTION of days: a symbol contributes only where every symbol has a
    return, because a correlation computed on unequal supports is not a correlation.  The
    count of surviving days is returned so a thin intersection cannot hide."""
    rows = con.execute("""
        SELECT symbol, DATE(open_time/1000,'unixepoch') AS d, MAX(open_time), close
        FROM klines WHERE open_time >= ? AND open_time < ?
        GROUP BY symbol, d ORDER BY symbol, d""", (lo, hi)).fetchall()
    ser = {}
    for sym, d, _t, c in rows:
        if c and sym in set(syms):
            ser.setdefault(sym, {})[d] = c
    days = None
    for sym in syms:
        ks = set(ser.get(sym, {}))
        days = ks if days is None else (days & ks)
    days = sorted(days or [])
    if len(days) < 30:
        return [[1.0]], 0
    R = {}
    for sym in syms:
        cl = [ser[sym][d] for d in days]
        R[sym] = [math.log(b / a) for a, b in zip(cl, cl[1:]) if a > 0 and b > 0]
    n = len(syms)
    m = len(days) - 1
    mu = {s: sum(R[s]) / m for s in syms}
    sd = {s: math.sqrt(sum((x - mu[s]) ** 2 for x in R[s]) / (m - 1)) or 1e-12 for s in syms}
    M = [[0.0] * n for _ in range(n)]
    for i, a in enumerate(syms):
        for j in range(i, n):
            b = syms[j]
            cov = sum((R[a][t] - mu[a]) * (R[b][t] - mu[b]) for t in range(m)) / (m - 1)
            v = cov / (sd[a] * sd[b])
            M[i][j] = M[j][i] = v
    return M, len(days)


def effective_bets(mat):
    """trace(C)/lambda_max, reported against its 1/n noise floor.

    The prereg spells this out because A-S36 caught the defect live: at small n the
    eigenvalue share has a floor of 1/n and a pure-noise matrix scores near it, which
    reads as "many independent bets" when it means "not enough data to tell".  When the
    measurement is within 1.5x of the floor it is declared uninformative and the LOWER
    of the two estimates is used -- the conservative direction."""
    n = len(mat)
    if n < 2:
        return {"n": n, "eff": 1.0, "floor": 1.0, "informative": False,
                "rho_bar": None, "note": "n<2"}
    lam = power_iteration(mat)
    eff = float(n) / lam if lam > 0 else 1.0     # trace(C) = n for a correlation matrix
    floor = 1.0                                   # lambda_max >= 1 always => eff <= n
    off = [mat[i][j] for i in range(n) for j in range(n) if i != j]
    rho_bar = sum(off) / len(off)
    informative = eff <= n / 1.5
    return {"n": n, "eff": eff if informative else min(eff, n / 1.5),
            "lambda_max": lam, "rho_bar": rho_bar, "informative": informative,
            "note": "" if informative else "eigenvalue share within 1.5x of n; "
                                           "reported as >= n/1.5, uninformative"}


def power_iteration(mat, iters=500):
    n = len(mat)
    v = [1.0 / math.sqrt(n)] * n
    lam = 0.0
    for _ in range(iters):
        w = [sum(mat[i][j] * v[j] for j in range(n)) for i in range(n)]
        nrm = math.sqrt(sum(x * x for x in w))
        if nrm == 0:
            return 0.0
        v = [x / nrm for x in w]
        lam = nrm
    return lam


def n_eff(per_symbol_windows, corr):
    """N_eff = (independent time units) x (effective bets).

    Time units are the MAXIMUM across symbols, not the sum.  This is the correction the
    prereg exists to enforce: symbols share the calendar, so they do not add time.  Three
    sections today (A-S35, A-S41, A-S43) summed rows across symbols and called it N; the
    largest of them looked like a 90 bps edge and was t = 0.41."""
    units = max(per_symbol_windows.values()) if per_symbol_windows else 0
    eb = effective_bets(corr)
    return {"time_units": units, "effective_bets": eb["eff"],
            "N_eff": units * eb["eff"], "eigen": eb,
            "rows_naive": sum(per_symbol_windows.values())}


# ---------------------------------------------------------------- capture + verdict
def capture(signed, absr):
    """f = E[s r] / E[|r|].  A-S38's definition, unchanged."""
    den = sum(absr)
    return (sum(signed) / den) if den > 0 else None


def verdict(f_hat, se, neff, k_variants=1):
    t = (f_hat / se) if (se and se > 0) else 0.0
    fst = fst_bar(k_variants)
    return {"f_hat": f_hat, "se": se, "t": t, "N_eff": neff,
            "bar_primary": T_BAR, "bar_fst": fst, "K": k_variants,
            "pass_primary": t >= T_BAR, "pass_fst": t >= fst,
            "verdict": "PASS" if (t >= T_BAR and t >= fst)
                       else ("NEGATIVE" if t <= -T_BAR else "NOT_ESTABLISHED")}


def fst_bar(K):
    """Lopez de Prado MLAM section 8.5.  With K = 1 this is 0 and adds nothing, which is
    the point: a single preregistered hypothesis on fresh data carries no multiplicity.
    It is computed anyway so that the number is on the record and so that running a second
    variant on the same window cannot quietly skip the correction."""
    if K <= 1:
        return 0.0
    g = 0.5772156649015329
    return (1 - g) * z_inv(1 - 1.0 / K) + g * z_inv(1 - 1.0 / (K * math.e))


def z_inv(p):
    """Acklam's inverse normal; adequate to ~1e-9 and dependency-free."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ---------------------------------------------------------------- modes
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true",
                    help="report universe, horizons and accrued N. Reads no outcome.")
    ap.add_argument("--evaluate", action="store_true",
                    help="run the frozen test. Requires a FROZEN prereg and a fresh window.")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--cost-bps", type=float, default=10.0,
                    help="round-trip cost. BINANCE_BASE = 10.0 unless OD-033 answers.")
    a = ap.parse_args()

    if not (a.dry or a.evaluate):
        sys.stderr.write(__doc__)
        sys.stderr.write("\nNo mode given.  There is no default that reads an outcome.\n")
        raise SystemExit(2)

    p = read_prereg()
    print("prereg   %s" % PREREG)
    print("  status %s   sha(file) %s" % (p["status"], p["sha_now"][:16]))
    print("  N_required (recomputed) %s   f_design %.4f   k %.4f"
          % (format(N_REQUIRED, ","), F_DESIGN, K_MEAN_ABS))

    if a.evaluate:
        if not (a.start and a.end):
            die("--evaluate needs --start and --end")
        check_frozen(p, iso_ms(a.start))
        print("  interlocks PASSED: frozen, hash matches, window is fresh")
        run_evaluate(a, p)
        return

    check_dry(a)


def check_dry(a):
    """The safe mode.  Universe + derived horizons + accrued time.  No return is read."""
    if not os.path.exists(PANEL):
        die("panel not found: %s" % PANEL)
    con = sqlite3.connect("file:%s?mode=ro" % PANEL.replace("\\", "/"), uri=True)
    end_ms = iso_ms(a.end) if a.end else LAWFUL_CUTOFF_MS
    keep, rej = universe(con, end_ms)
    print("\nUNIVERSE (prereg section 5) at %s" % ms_iso(end_ms))
    print("  admitted %d   rejected %d" % (len(keep), len(rej)))
    from collections import Counter
    for r, n in Counter(v.split()[0] for v in rej.values()).most_common():
        print("    %-4s %d" % (r, n))
    if len(keep) < MIN_SYMBOLS:
        print("  *** below S3's floor of %d -- the test would ABORT" % MIN_SYMBOLS)

    lo = end_ms - COV_WINDOW * 86400000
    sig = daily_sigmas(con, lo, end_ms)
    hs = []
    for s in keep:
        sd = sig.get(s)
        if sd is None or sd < MIN_SIGMA_BPS:
            continue
        h = h_star_days(sd, a.cost_bps)
        if h:
            hs.append((s, sd, h))
    hs.sort(key=lambda x: x[2])
    print()
    print("DERIVED HORIZONS (prereg section 3; h* = [2c/(k f sigma_d)]^2)")
    print("  measured on %d of %d admitted (U4 sigma >= %.0f bps)"
          % (len(hs), len(keep), MIN_SIGMA_BPS))
    if not hs:
        con.close()
        return
    for lab, i in (("shortest", 0), ("median", len(hs) // 2), ("longest", -1)):
        sy, sd, h = hs[i]
        print("    %-9s %-12s sigma_d %7.1f bps   h* %9.2f d   h_be %8.2f d"
              % (lab, sy, sd, h, h / 4))

    # ---- effective bets, MEASURED.  The prereg's section 6 forbids assuming it.
    syms = [x[0] for x in hs]
    C, used = corr_matrix(con, syms, lo, end_ms)
    eb = effective_bets(C)
    print()
    print("EFFECTIVE BETS (prereg section 6, measured -- not assumed)")
    print("  symbols in matrix %d   lambda_max %.3f   rho_bar %+.4f"
          % (eb["n"], eb.get("lambda_max", 0), eb.get("rho_bar") or 0))
    print("  effective bets %.2f of %d   informative %s%s"
          % (eb["eff"], eb["n"], eb["informative"],
             ("  <- " + eb["note"]) if eb["note"] else ""))

    # ---- feasibility.  t = S_annual * sqrt(years), and h* maximises BOTH the annual
    # Sharpe and t, because t(h) = S_annual(h) * sqrt(T_years) -- the same optimisation.
    s_trade = K_MEAN_ABS * F_DESIGN / 2.0     # per-trade Sharpe at h*, symbol-independent
    print()
    print("FEASIBILITY AT f_design = %.3f  (per-trade Sharpe at h* is k*f/2 = %.6f,"
          % (F_DESIGN, s_trade))
    print("  the same for every symbol -- A-S33's invariant)")
    print("  %-10s %10s %12s %14s" % ("", "h* (d)", "S_annual", "years to t=2"))
    for lab, i in (("best", 0), ("median", len(hs) // 2)):
        sy, sd, h = hs[i]
        sa = math.sqrt(365.0 / h) * s_trade
        print("  %-10s %10.2f %12.4f %14s"
              % (sy[:10], h, sa, fmt_years((T_BAR / sa) ** 2)))
    hmed = hs[len(hs) // 2][2]
    sp = math.sqrt(365.0 / hmed) * s_trade * math.sqrt(eb["eff"])
    print("  %-10s %10.2f %12.4f %14s"
          % ("POOLED", hmed, sp, fmt_years((T_BAR / sp) ** 2)))
    print()
    print("N_required %s trades; at the median h* and %.1f effective bets the pooled"
          % (format(N_REQUIRED, ","), eb["eff"]))
    print("  test reaches t = 2 after the span above.  REPORTED, NOT JUDGED -- section 9")
    print("  is explicit that a shortfall is 'not established at this N', never 'f = 0'.")

    # ---- the inversion.  S_annual at h* is proportional to f^2 (because h* ~ 1/f^2
    # makes sqrt(365/h*) ~ f, on top of the explicit f in k*f/2), so the time to a
    # verdict falls as f^-4.  Inverting gives the capture that would make the test
    # runnable -- a number that does NOT depend on f_design and is therefore the one
    # figure here an independent reader can check without accepting section 4.
    yrs = (T_BAR / sp) ** 2
    print()
    print("THE INVERSION -- what capture would make this test runnable")
    print("  years to a verdict scale as f^-4.  Solving for the span:")
    for target, lab in ((1.0, "1 year"), (2.0, "2 years"), (5.0, "5 years")):
        need = F_DESIGN * (yrs / target) ** 0.25
        print("    verdict in %-8s requires pooled capture f >= %6.2f%%" % (lab, need * 100))
    print("  every single-leg capture this estate has measured is 1-2% (A-S14),")
    print("  and the best dark-family cell measured 2.09% (A-S43).")
    print("  This line is independent of f_design: it inverts the same algebra.")


    con.close()


def run_evaluate(a, p):
    """Deliberately unimplemented against live data.

    The charter's stop rule is 'STOP when the prereg is frozen and hashed.  No outcome is
    read.  Not one.'  Writing the code that reads the outcome is inside the charter;
    running it is not, and the window it would need does not exist yet -- every day in the
    estate predates the freeze, so the third interlock rejects all of them by construction.

    The reader who eventually runs this fills the loop below.  Everything it needs is
    already in this file: universe(), daily_sigmas(), h_star_days(), capture(), n_eff(),
    verdict().  What is missing is only the signal -- and the signal belongs to the
    rule-level preregistration that a PASS here would license, not to this one."""
    die("no evaluation window exists yet: every day in the estate predates the freeze.\n"
        "  This test measures whether the sign carries; it does not name a rule.  The\n"
        "  signal column is supplied by the rule-level prereg that a PASS here licenses.\n"
        "  Until then --dry is the only meaningful mode, and it is the honest answer.")


if __name__ == "__main__":
    main()
