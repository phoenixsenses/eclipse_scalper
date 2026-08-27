r"""LANE C, round 47 -- Hasbrouck's information-share BOUNDS, and the one demand I wrongly closed.

C-T43 tabulated fifteen demands the corpus makes of this estate and marked two OPEN. One of them:

    #4  "are your information shares BOUNDS, and does aggregation widen them?"   Hasbrouck
        status recorded by C-T43: BLOCKED -- "needs two venues, the estate has one"

That status was an ASSUMPTION, not a measurement. I never opened the schema. `microstructure_02.db`
carries `mark_prices` alongside `agg_trades`: the perpetual's own transaction price and Binance's
index-based mark, two observable series tracking one efficient price, updating every 0.12 s and
1.03 s respectively. Hasbrouck's framework is constructible here and always was.

`--who information share price discovery` returns one irrelevant hit; the Turkish search returns
none, and the tool's own warning is that an empty result is a CLAIM. Both languages and
discriminating terms were tried. Nobody in this estate has measured price discovery.

THE PREDICTION, stated verbatim before any measurement:

    "Time aggregation may make events that are actually separated in time to appear
     contemporaneous. This leads to larger off-diagonal covariances in Omega... As the off-diagonal
     elements increase in size, the information shares become more sensitive to the causal ordering
     imposed by the Cholesky factorizations. OVER ALL CAUSAL PERMUTATIONS, THE UPPER AND LOWER
     BOUNDS FOR THE INFORMATION SHARES WILL BECOME WIDER. A shorter time interval is desirable,
     therefore, because it generally implies tighter bounds."          Hasbrouck, EMM

Two things are testable and both are tested: (a) the bounds widen with the sampling interval, and
(b) the stated MECHANISM -- the off-diagonal correlation of Omega rises with it.

CALIBRATION BEFORE THE TEST. The estimator is run first on a synthetic pair with a KNOWN answer: a
common efficient random walk where series 1 sees it immediately and series 2 sees it one step late
plus independent noise, so the true information share of series 1 is 1. If the estimator does not
recover that at the finest sampling, nothing measured on the real pair is readable.

SAMPLE BY ARTIFACT PATH: data/microstructure_02.db :: agg_trades and mark_prices, first 1,500,000
rows of each by ts_ms per symbol, inner-joined on a common calendar grid.
THRESHOLD DECLARED AND SWEPT: the bar width, at 1 / 5 / 30 / 60 / 300 seconds.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "microstructure_02.db"
OUT = ROOT / "reports" / "atlas"
SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
NROWS = 1_500_000
BARS = (1, 5, 30, 60, 300)
NLAG = 2
SEED = 20260827


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def bar_last(ts, val, t0, sec, nb):
    """last observation in each calendar bar, forward-filled; returns series and empty share"""
    idx = ((ts - t0) // (sec * 1000)).astype(np.int64)
    ok = (idx >= 0) & (idx < nb)
    out = np.full(nb, np.nan)
    out[idx[ok]] = val[ok]
    m = np.isnan(out)
    if m.all():
        return None, 1.0
    first = int(np.flatnonzero(~m)[0])
    out = out[first:]
    m = np.isnan(out)
    ix = np.where(~m, np.arange(len(out)), 0)
    np.maximum.accumulate(ix, out=ix)
    return out[ix], float(m.mean())


def vecm_is(p1, p2, nlag=NLAG):
    """Hasbrouck information-share bounds for a bivariate system with cointegrating vector (1,-1).

    z_t = p1_t - p2_t; dp_t = alpha z_{t-1} + sum Gamma_k dp_{t-k} + eps_t.
    Common-factor weights psi are orthogonal to alpha: psi = (a2, -a1)/(a2 - a1).
    IS_j under a Cholesky ordering = (psi F)_j^2 / (psi Omega psi'), F lower-triangular chol.
    """
    d1 = np.diff(p1)
    d2 = np.diff(p2)
    z = (p1 - p2)[:-1]
    n = len(d1)
    s = nlag
    Y = np.column_stack([d1[s:], d2[s:]])
    cols = [np.ones(n - s), z[s:]]
    for k in range(1, nlag + 1):
        cols.append(d1[s - k:n - k])
        cols.append(d2[s - k:n - k])
    X = np.column_stack(cols)
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    E = Y - X @ B
    Om = np.cov(E.T, ddof=1)
    a1, a2 = float(B[1, 0]), float(B[1, 1])
    den = a2 - a1
    if abs(den) < 1e-14:
        return None
    psi = np.array([a2, -a1]) / den
    var = float(psi @ Om @ psi)
    if var <= 0:
        return None
    shares = []
    for order in ((0, 1), (1, 0)):
        P = np.eye(2)[list(order)]
        Op = P @ Om @ P.T
        try:
            F = np.linalg.cholesky(Op)
        except np.linalg.LinAlgError:
            return None
        pf = (psi @ P.T) @ F
        s_ord = (pf ** 2) / var
        # map back: position of series 0 in this ordering
        shares.append(float(s_ord[list(order).index(0)]))
    lo, hi = min(shares), max(shares)
    rho = float(Om[0, 1] / np.sqrt(Om[0, 0] * Om[1, 1]))
    return {"IS1_lower": round(lo, 4), "IS1_upper": round(hi, 4),
            "bound_width": round(hi - lo, 4), "IS1_midpoint": round((lo + hi) / 2, 4),
            "omega_offdiag_corr": round(rho, 4),
            "alpha_trade": round(a1, 6), "alpha_mark": round(a2, 6), "n_obs": int(n - s)}


def main() -> int:
    rng = np.random.default_rng(SEED)

    # ---- CALIBRATION: series 1 leads, series 2 follows one step late; true IS1 = 1
    N = 200_000
    w = rng.standard_normal(N) * 1e-4
    eff = np.cumsum(w)
    lead = eff.copy()
    follow = np.concatenate([[eff[0]], eff[:-1]]) + rng.standard_normal(N) * 2e-5
    cal = {}
    for k in (1, 5, 30, 60, 300):
        a = vecm_is(lead[::k], follow[::k])
        if a:
            cal["thin_by_{0}".format(k)] = a
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per = {}
    try:
        for sym in SYMS:
            tr = np.array(con.execute(
                "select ts_ms,price from agg_trades where symbol=? order by ts_ms limit ?",
                (sym, NROWS)).fetchall(), dtype=np.float64)
            mk = np.array(con.execute(
                "select ts_ms,mark_price from mark_prices where symbol=? order by ts_ms limit ?",
                (sym, NROWS)).fetchall(), dtype=np.float64)
            t0 = max(tr[0, 0], mk[0, 0])
            t1 = min(tr[-1, 0], mk[-1, 0])
            if t1 <= t0:
                per[sym] = {"error": "no overlapping span"}
                continue
            d = {}
            for sec in BARS:
                nb = int((t1 - t0) // (sec * 1000)) + 1
                if nb < 3000:
                    continue
                p1, e1 = bar_last(tr[:, 0], np.log(tr[:, 1]), t0, sec, nb)
                p2, e2 = bar_last(mk[:, 0], np.log(mk[:, 1]), t0, sec, nb)
                if p1 is None or p2 is None:
                    continue
                k = min(len(p1), len(p2))
                r = vecm_is(p1[:k], p2[:k])
                if r:
                    r["bar_seconds"] = sec
                    r["empty_bar_share_trade"] = round(e1, 4)
                    r["empty_bar_share_mark"] = round(e2, 4)
                    d["bar_{0}s".format(sec)] = r
            widths = [v["bound_width"] for v in d.values()]
            rhos = [v["omega_offdiag_corr"] for v in d.values()]
            per[sym] = {
                "overlap_hours": round((t1 - t0) / 3.6e6, 1),
                "by_bar": d,
                "width_1s": d.get("bar_1s", {}).get("bound_width"),
                "width_300s": d.get("bar_300s", {}).get("bound_width"),
                "width_widens_with_interval": (bool(widths[-1] > widths[0])
                                               if len(widths) >= 2 else None),
                "rho_rises_with_interval": (bool(abs(rhos[-1]) > abs(rhos[0]))
                                            if len(rhos) >= 2 else None),
            }
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    art = {"study": "C-T47", "lane": "C", "utc": _utc(),
           "corpus_demand": ("Hasbrouck EMM: information shares are BOUNDS over causal "
                             "permutations, and time aggregation WIDENS them"),
           "status_correction": ("C-T43 recorded this demand as BLOCKED for want of a second "
                                 "price series. mark_prices was always in the same database; "
                                 "the block was an assumption, not a measurement."),
           "sample": ("data/microstructure_02.db :: agg_trades + mark_prices, first 1,500,000 "
                      "rows each by ts_ms per symbol, common calendar grid"),
           "threshold_swept": "bar width 1 / 5 / 30 / 60 / 300 seconds",
           "calibration_true_IS1_is_1": cal,
           "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C47_INFORMATION_SHARE_BOUNDS_V1.json").write_text(json.dumps(art, indent=2),
                                                              encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w_(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    w_("CALIBRATION -- series 1 leads by one step, true IS1 = 1")
    w_("%-14s %10s %10s %10s %10s" % ("thinning", "IS1 lower", "IS1 upper", "width", "rho(Om)"))
    for k, v in cal.items():
        w_("%-14s %10s %10s %10s %10s" % (k, v["IS1_lower"], v["IS1_upper"], v["bound_width"],
                                          v["omega_offdiag_corr"]))
    w_("")
    w_("REAL -- trade price vs mark price")
    w_("%-9s %8s %9s %10s %10s %10s %11s %9s" % ("sym", "bar", "n", "IS1 lo", "IS1 hi",
                                                 "width", "rho(Omega)", "empty tr"))
    for s in SYMS:
        p = per[s]
        if "by_bar" not in p:
            w_("%-9s %s" % (s, p.get("error")))
            continue
        for k, v in p["by_bar"].items():
            w_("%-9s %8s %9d %10s %10s %10s %11s %9s" % (
                s, k, v["n_obs"], v["IS1_lower"], v["IS1_upper"], v["bound_width"],
                v["omega_offdiag_corr"], v["empty_bar_share_trade"]))
        w_("%-9s  widens with interval: %-6s   rho rises: %s" % (
            s, p["width_widens_with_interval"], p["rho_rises_with_interval"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
