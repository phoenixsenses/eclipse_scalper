r"""LANE C, round 39 -- apply C-T38's lesson to this lane's own SURVIVING claims, unprompted.

Three headlines have now been retracted (C-T29's price super-diffusion, C-T34's efficiency
failure, C-T32's stated reason), and the first two fell to one defect: a fit range that crossed a
scaling-regime boundary. The right response is not to wait for a fourth challenge but to point
the same instrument at what is still standing.

Two claims of mine have never been swept:

  (1) C-T28's DECOMPOSITION -- "the entire excess of chi above 1/2 is SIGN memory; size memory
      alone contributes nothing" (share 0.968-1.003 across 12 cells). It was computed at a single
      fit range, T = 20..1000, which is exactly the range C-T38 showed straddles regimes. The
      conclusion may still hold, because the sign-shuffle collapses chi to a KNOWN value (1/2)
      rather than to a fitted one -- but "may still hold" is not a measurement.

  (2) C-T33's COLLAPSE -- Bouchaud's TSLA chi = 0.95 reproducing here as 0.92 / 1.06 / 0.96. That
      is this lane's ONLY clean positive replication against the literature, and it was fitted
      over one T range, T = 5..100. If chi = 0.95 is a compromise across a drifting surface
      rather than a property of a regime, the replication is worth less than I claimed.

Both are checked here at their own scales rather than at one chosen range. A claim that survives
its own author's sweep is worth more than one that has never been swept.
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
BANDS = ((4, 40), (40, 400), (400, 4000), (20, 1000))     # last one is C-T28's own range
REPS = 12
SEED = 20260827


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def chi_band(x, lo, hi, npts=10):
    n = len(x)
    Ts = np.unique(np.round(np.geomspace(lo, hi, npts)).astype(int))
    T, S = [], []
    for t in Ts:
        m = n // t
        if m < 200:
            continue
        T.append(float(t))
        S.append(float(np.std(x[:m * t].reshape(m, t).sum(axis=1), ddof=1)))
    if len(T) < 4:
        return float("nan")
    lt, ls = np.log(T), np.log(S)
    A = np.column_stack([np.ones(len(lt)), lt])
    b, *_ = np.linalg.lstsq(A, ls, rcond=None)
    return float(b[1])


def main() -> int:
    rng = np.random.default_rng(SEED)
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    part1 = {}
    try:
        for sym in SYMS:
            a = np.array(con.execute(
                "select notional,is_buyer_maker from agg_trades where symbol=? "
                "order by ts_ms limit ?", (sym, NROWS)).fetchall(), dtype=np.float64)
            size = a[:, 0]
            sign = np.where(a[:, 1] > 0.5, -1.0, 1.0)
            sv = sign * size
            n = len(sv)
            d = {}
            for lo, hi in BANDS:
                obs = chi_band(sv, lo, hi)
                sh = float(np.mean([chi_band(sign[rng.permutation(n)] * size, lo, hi)
                                    for _ in range(REPS)]))
                sz = float(np.mean([chi_band(sign * size[rng.permutation(n)], lo, hi)
                                    for _ in range(REPS)]))
                exc = obs - 0.5
                d["{0}-{1}".format(lo, hi)] = {
                    "observed_chi": round(obs, 4),
                    "sign_shuffled": round(sh, 4),
                    "size_shuffled": round(sz, 4),
                    "excess_above_half": round(exc, 4),
                    "share_from_sign_memory": (round(1.0 - (sh - 0.5) / exc, 3)
                                               if abs(exc) > 1e-6 else None),
                    "share_from_size_memory": (round(1.0 - (sz - 0.5) / exc, 3)
                                               if abs(exc) > 1e-6 else None)}
            part1[sym] = d
            sys.stderr.write("part1 {0} done\n".format(sym))
    finally:
        con.close()

    # ---- part 2: is C-T33's collapse stable across T sub-ranges?
    src = json.loads((OUT / "C33_AGGREGATE_IMPACT_V1.json").read_text(encoding="utf-8"))
    SUBS = [(5, 10, 20), (10, 20, 50), (20, 50, 100), (5, 10, 20, 50, 100)]

    def collapse_err(cur, S, R1, kap, chi, Ts):
        xs, ys = [], []
        for T in Ts:
            cc = cur[str(T)]["curve"]
            dvv = np.array([c["dV"] for c in cc], float)
            rr = np.array([c["R_bps"] for c in cc], float)
            xs.append(dvv / (S * T ** chi))
            ys.append(rr / (R1 * T ** kap))
        lo = max(x.min() for x in xs)
        hi = min(x.max() for x in xs)
        if not np.isfinite(lo) or hi <= lo:
            return np.inf
        g = np.linspace(lo, hi, 25)
        M = []
        for x, y in zip(xs, ys):
            o = np.argsort(x)
            M.append(np.interp(g, x[o], y[o]))
        M = np.array(M)
        sc = np.abs(M).mean()
        return float(M.std(axis=0).mean() / sc) if sc > 0 else np.inf

    part2 = {}
    for sym in SYMS:
        cur = src["per_symbol"][sym]["by_T"]
        c1 = cur["1"]["curve"]
        S = float(np.mean([abs(c["dV"]) for c in c1]))
        R1 = float(np.mean([abs(c["R_bps"]) for c in c1]))
        out = {}
        for Ts in SUBS:
            best = (np.inf, None, None)
            for kap in np.arange(0.20, 1.401, 0.02):
                for chi in np.arange(0.20, 1.601, 0.02):
                    er = collapse_err(cur, S, R1, kap, chi, Ts)
                    if er < best[0]:
                        best = (er, float(kap), float(chi))
            er, kap, chi = best
            out["T_" + "_".join(map(str, Ts))] = {
                "kappa": round(kap, 3), "chi": round(chi, 3),
                "kappa_minus_chi": round(kap - chi, 3), "rel_err": round(er, 4),
                "at_grid_edge": bool(abs(kap - 0.20) < 1e-9 or abs(kap - 1.40) < 1e-9
                                     or abs(chi - 0.20) < 1e-9 or abs(chi - 1.60) < 1e-9)}
        vals = [v["chi"] for v in out.values()]
        out["chi_spread_across_subranges"] = round(max(vals) - min(vals), 3)
        out["chi_stable_within_0_10"] = bool(max(vals) - min(vals) <= 0.10)
        part2[sym] = out

    art = {"study": "C-T39", "lane": "C", "utc": _utc(),
           "why": ("three headlines fell to a fit range crossing a regime boundary; these are the "
                   "surviving claims of the same lane, swept before anyone asks"),
           "part1_decomposition_by_band": part1,
           "part2_collapse_by_subrange": part2}
    (OUT / "C39_OWN_SURVIVORS_SWEPT_V1.json").write_text(json.dumps(art, indent=2),
                                                         encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    w("PART 1 -- C-T28's decomposition, band by band")
    w("%-9s %-12s %9s %11s %11s %11s %11s" % ("sym", "band", "chi", "signShuf", "sizeShuf",
                                              "signShare", "sizeShare"))
    for s in SYMS:
        for b, v in part1[s].items():
            w("%-9s %-12s %9.4f %11.4f %11.4f %11s %11s" % (
                s, b, v["observed_chi"], v["sign_shuffled"], v["size_shuffled"],
                v["share_from_sign_memory"], v["share_from_size_memory"]))
    w("")
    w("PART 2 -- C-T33's collapse, sub-range by sub-range (book TSLA: kappa 0.65, chi 0.95)")
    w("%-9s %-22s %8s %8s %10s %9s" % ("sym", "sub-range", "kappa", "chi", "kappa-chi", "rel err"))
    for s in SYMS:
        for k, v in part2[s].items():
            if not isinstance(v, dict):
                continue
            w("%-9s %-22s %8.3f %8.3f %10.3f %9.4f%s" % (
                s, k, v["kappa"], v["chi"], v["kappa_minus_chi"], v["rel_err"],
                "  <-- EDGE" if v["at_grid_edge"] else ""))
        w("%-9s %-22s chi spread %.3f  stable within 0.10: %s" % (
            s, "", part2[s]["chi_spread_across_subranges"],
            part2[s]["chi_stable_within_0_10"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
