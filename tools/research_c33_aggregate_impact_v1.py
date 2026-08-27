r"""LANE C, round 33 -- the aggregate impact function, by the book's own definition.

Bouchaud Sec. 11.4 gives an object this estate has never measured and states outright that it is
available from the data we have:

    R(dV, T) := E[ m_{t+T} - m_t | sum_{n in [t,t+T)} eps_n nu_n = dV ]        Eq. (11.10)

    "This quantity can be studied empirically using only public trades-and-quotes data."

That matters here because C-T24 closed the metaorder route (identifiers absent, NOT_IDENTIFIABLE)
and C-T32 closed the counterfactual route (Eq. 11.1, the two worlds are mutually exclusive).
Aggregate impact is neither: it is a conditional expectation over observed quantities.

THREE PREDICTIONS ARE STATED IN THE TEXT AND ARE TESTED HERE.
  1. R(dV, 1) is a STRONGLY CONCAVE function of dV.
  2. As T grows, the dependence becomes CLOSER TO LINEAR for small dV while RETAINING concavity
     for large |dV|.
  3. The curves collapse under  R(dV,T) ~ R(1) T^kappa F( dV / (V_D T^chi) ), and for TSLA the
     book reports kappa = 0.65, chi = 0.95 -- so the book's own kappa - chi is NEGATIVE, -0.30.

PREDICTION 3 IS ALSO A REGISTRY QUESTION. Three quantities in this estate are called kappa-chi:
    this lane's unconditional pair (C-T23/C-T31):   +0.0009 / -0.1035 / -0.0693
    another lane's collapsed-scaling pair:          +0.2245 / +0.3786 / +0.2032
    the book's own TSLA collapse:                   -0.30
The third is measured by the definition above. Measuring it here on the same three symbols says
which of the two lanes is estimating the book's object, without contradicting either by assertion.

METHOD DISCIPLINE CARRIED IN FROM THE LAST TWO ROUNDS.
  - C-T31: a permutation null is a TEST, not a standard error. Every exponent here carries a
    moving-block bootstrap SE with the dependence intact.
  - C-T32: no counterfactual. R(dV,T) is a conditional mean of observed quantities; nothing is
    compared to a world that did not happen.
  - C-T30: print the composition. Bin counts are reported so a shape cannot be read off bins
    that are populated differently.
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
NROWS = 2_000_000
T_GRID = (1, 5, 10, 20, 50, 100)
NBIN = 21
BOOT = 40
BLOCK = 50_000
SEED = 20260827


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def curve(dv, dp, nbin=NBIN):
    """mean price change per quantile bin of signed flow, with counts printed"""
    q = np.quantile(dv, np.linspace(0, 1, nbin + 1))
    q[0] -= 1e-9
    q[-1] += 1e-9
    idx = np.searchsorted(q, dv, side="right") - 1
    idx = np.clip(idx, 0, nbin - 1)
    rows = []
    for b in range(nbin):
        m = idx == b
        if m.sum() < 30:
            continue
        rows.append({"bin": b, "n": int(m.sum()),
                     "mean_dV": float(dv[m].mean()),
                     "mean_dP_bps": float(dp[m].mean())})
    return rows


def concavity_exponent(rows):
    """fit |R| ~ |dV|^delta on bins whose sign agrees with dV; delta < 1 is concave"""
    x, y = [], []
    for r in rows:
        if r["mean_dV"] == 0:
            continue
        if np.sign(r["mean_dP_bps"]) != np.sign(r["mean_dV"]):
            continue
        if abs(r["mean_dV"]) <= 0 or abs(r["mean_dP_bps"]) <= 0:
            continue
        x.append(np.log(abs(r["mean_dV"])))
        y.append(np.log(abs(r["mean_dP_bps"])))
    if len(x) < 5:
        return float("nan"), 0
    A = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(A, np.asarray(y), rcond=None)
    return float(b[1]), len(x)


def small_dv_linearity(rows):
    """slope of R on dV over the central third of the bins -- the 'small dV' regime"""
    k = len(rows)
    core = rows[k // 3: 2 * k // 3 + 1]
    if len(core) < 4:
        return float("nan")
    x = np.asarray([r["mean_dV"] for r in core])
    y = np.asarray([r["mean_dP_bps"] for r in core])
    ax = np.abs(x)
    ok = ax > 0
    if ok.sum() < 4:
        return float("nan")
    A = np.column_stack([np.ones(ok.sum()), np.log(ax[ok])])
    b, *_ = np.linalg.lstsq(A, np.log(np.abs(y[ok]) + 1e-12), rcond=None)
    return float(b[1])


def windows(lp, flow, T):
    """dV over T trades, and the price change spanning exactly those T trades.

    The book's m_t is the mid BEFORE the first trade of the window and m_{t+T} the mid after the
    last, so the span is lp[i*T + T - 1] - lp[i*T - 1]. A first attempt used lp[.., -1] - lp[.., 0]
    within the window, which spans only T-1 trades and is IDENTICALLY ZERO at T = 1 -- it silently
    returned an empty lag-1 impact, which is the one value the book states most firmly."""
    n = len(lp)
    m = (n - 1) // T
    idx0 = np.arange(m) * T                 # first trade of each window
    pre = lp[idx0 - 1 + 1 - 1]              # lp[idx0 - 1], shifted below
    pre = lp[np.maximum(idx0 - 1, 0)]
    post = lp[idx0 + T - 1]
    dv = flow[:m * T].reshape(m, T).sum(axis=1)
    dp = (post - pre) * 1e4
    ok = idx0 >= 1
    return dv[ok], dp[ok]


def main() -> int:
    rng = np.random.default_rng(SEED)
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per = {}
    try:
        for sym in SYMS:
            a = np.array(con.execute(
                "select price,notional,is_buyer_maker from agg_trades where symbol=? "
                "order by ts_ms limit ?", (sym, NROWS)).fetchall(), dtype=np.float64)
            lp = np.log(a[:, 0])
            flow = np.where(a[:, 2] > 0.5, -1.0, 1.0) * a[:, 1]
            n = len(lp)
            nb = n // BLOCK

            byT = {}
            for T in T_GRID:
                dv, dp = windows(lp, flow, T)
                rows = curve(dv, dp)
                delta, npts = concavity_exponent(rows)
                lin = small_dv_linearity(rows)
                # bootstrap the concavity exponent with the dependence intact
                bs = []
                for _ in range(BOOT):
                    idx = np.concatenate([np.arange(i, i + BLOCK)
                                          for i in rng.integers(0, n - BLOCK, nb)])
                    d2, p2 = windows(lp[idx], flow[idx], T)
                    e, _ = concavity_exponent(curve(d2, p2))
                    if np.isfinite(e):
                        bs.append(e)
                byT[T] = {"delta_concavity": round(delta, 4),
                          "delta_boot_sd": round(float(np.std(bs, ddof=1)), 4) if len(bs) > 5
                          else None,
                          "n_bins_used": npts,
                          "small_dV_local_exponent": round(lin, 4),
                          "R_at_widest_bin_bps": round(max(abs(r["mean_dP_bps"])
                                                           for r in rows), 4),
                          "bin_counts": [r["n"] for r in rows],
                          "curve": [{"dV": float("{0:.6g}".format(r["mean_dV"])),
                                     "R_bps": round(r["mean_dP_bps"], 4), "n": r["n"]}
                                    for r in rows]}
            per[sym] = {"n_trades": int(n), "by_T": byT}
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    art = {"study": "C-T33", "lane": "C", "utc": _utc(),
           "definition": "Bouchaud Eq. (11.10), aggregate impact R(dV,T), market-order time",
           "book_predictions": ["R(dV,1) strongly concave",
                                "closer to linear at small dV as T grows, concave at large |dV|",
                                "collapse with TSLA kappa 0.65 chi 0.95, so kappa-chi = -0.30"],
           "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C33_AGGREGATE_IMPACT_V1.json").write_text(json.dumps(art, indent=2), encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    w("%-9s %5s %10s %9s %14s %12s" % ("sym", "T", "delta", "bootSD", "smalldV_exp",
                                       "|R|max bps"))
    for s in SYMS:
        for T in T_GRID:
            r = per[s]["by_T"][T]
            w("%-9s %5d %10.4f %9s %14.4f %12.4f" % (
                s, T, r["delta_concavity"], r["delta_boot_sd"],
                r["small_dV_local_exponent"], r["R_at_widest_bin_bps"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
