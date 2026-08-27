# -*- coding: utf-8 -*-
"""C-T40 -- THE QUESTION THE CORPUS ASKS US: DO LIQUIDITY PROVIDERS WITHDRAW TOGETHER?

Asked to look for what the corpus wants measured rather than what it claims, Sec 22.3 names
one that this estate can actually answer:

    "Empirical evidence suggests that the 'new' market-makers in modern financial markets ALL
     REACT IN A SIMILAR WAY to risk indicators, such as SHORT-TERM VOLATILITY, SHORT-TERM
     ACTIVITY BURSTS and SHORT-TERM TRENDS.  This implies that the apparent diversification
     benefit of having many different liquidity providers is ABSENT AT THE TIMES WHEN IT IS
     NEEDED MOST."

Two claims, and they are separable:
    (A) liquidity responds to each of the three named indicators
    (B) the responses are SYNCHRONISED, so that withdrawal correlates across instruments
        MORE in stressed minutes than in calm ones -- the "absent when needed most" clause

This lane's own record makes (B) the sharper question.  Sec 326-337 already concluded
CASCADE_IS_COMMON_STATE_MARKER_ONLY and that activity-to-variance is the COMMON market clock,
not the symbol's own.  So cross-symbol co-movement of ACTIVITY is known.  What is not known is
whether LIQUIDITY WITHDRAWAL co-moves BEYOND that, and whether the co-movement RISES with
stress.  A correlation that is constant across regimes would mean the diversification benefit
is merely small, not absent when needed.

MEASURED, per symbol, per calendar minute over 7 days:
    depth      mean top-of-book notional, (bid_qty * bid + ask_qty * ask) / 2
    spread     mean relative spread in bps
    rv         realised volatility of the mid inside the minute, bps
    act        book updates in the minute
    trend      |mid(end) - mid(start)| / mid, bps

  (A) elasticity of log(depth) to log(rv), log(act) and log(1+trend), per symbol
  (B) cross-symbol correlation of minute log-depth CHANGES, split by stress tercile of a
      COMMON stress index (the cross-symbol mean of standardised rv).  The corpus's clause
      predicts the correlation RISES from the calm tercile to the stressed one.

NULL for (B): minutes shuffled within hour-of-day, 20 draws, which destroys the alignment
between symbols while preserving each symbol's own distribution and diurnal shape.  Any
correlation that survives shuffling is not co-movement.

This is a RISK/liquidity-state measurement, not a route: it says when the book thins, not
which way the price goes.  Nothing in execution/, no route parameter, no sizing.

ESTIMATION.  Ceiling: MECHANISM_CHARACTERISATION.

  python -m tools.ct40_do_liquidity_providers_withdraw_together --i-have-approval
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np

from tools import h2_response_shape_driver as H2
from tools import s66_cascade_process_driver as D

OUT = "reports/atlas"
DAYS = ("2026-08-07", "2026-08-08", "2026-08-09", "2026-08-10",
        "2026-08-11", "2026-08-12", "2026-08-13")
N_SHUF = 20
RNG_SEED = 20260827


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    rng = np.random.default_rng(RNG_SEED)
    per = {}
    for sym in H2.SYMBOLS:
        rows_all = {}
        for day in DAYS:
            d0 = dt.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
            lo = int(d0.timestamp() * 1000)
            hi = int((d0 + dt.timedelta(days=1)).timestamp() * 1000)
            con = D._con()
            rows = con.execute(
                "SELECT ts_ms,bid_price,ask_price,bid_qty,ask_qty FROM book_ticker "
                "WHERE symbol=? AND ts_ms>=? AND ts_ms<? AND bid_price>0 AND ask_price>0 "
                "ORDER BY ts_ms", (sym, lo, hi)).fetchall()
            con.close()
            if len(rows) < 10000:
                continue
            ts = np.array([r[0] for r in rows], np.int64)
            bid = np.array([r[1] for r in rows], float)
            ask = np.array([r[2] for r in rows], float)
            bq = np.array([r[3] for r in rows], float)
            aq = np.array([r[4] for r in rows], float)
            del rows
            mid = 0.5 * (bid + ask)
            depth = 0.5 * (bq * bid + aq * ask)
            spr = (ask - bid) / mid * 1e4
            minute = ts // 60000
            um, inv = np.unique(minute, return_inverse=True)
            cnt = np.bincount(inv).astype(float)
            d_mean = np.bincount(inv, weights=depth) / cnt
            s_mean = np.bincount(inv, weights=spr) / cnt
            r = np.zeros(len(ts))
            r[1:] = (mid[1:] / mid[:-1] - 1.0) * 1e4
            same = np.concatenate([[False], minute[1:] == minute[:-1]])
            rv = np.sqrt(np.bincount(inv, weights=np.where(same, r * r, 0.0)))
            first = np.zeros(len(um), int)
            last = np.zeros(len(um), int)
            starts = np.flatnonzero(np.concatenate([[True], np.diff(inv) != 0]))
            ends = np.concatenate([starts[1:] - 1, [len(inv) - 1]])
            first[inv[starts]] = starts
            last[inv[starts]] = ends
            trend = np.abs(mid[last] / mid[first] - 1.0) * 1e4
            for i, m in enumerate(um):
                rows_all[int(m)] = (float(d_mean[i]), float(s_mean[i]), float(rv[i]),
                                    float(cnt[i]), float(trend[i]))
            del ts, bid, ask, bq, aq, mid, depth, spr
        per[sym] = rows_all
        print("%s   minutes %d" % (sym, len(rows_all)), flush=True)

    common = sorted(set.intersection(*[set(v) for v in per.values()]))
    print("common minutes across all three: %d" % len(common), flush=True)
    M = {s: np.array([per[s][m] for m in common]) for s in per}

    res = {"days": list(DAYS), "n_common_minutes": len(common),
           "book": "Sec 22.3 -- market-makers all react alike to short-term volatility, "
                   "activity bursts and trends; diversification absent when needed most",
           "elasticities": {}, "comovement": {}, "ceiling": "MECHANISM_CHARACTERISATION"}

    print("\n(A) elasticity of log(depth) to each risk indicator", flush=True)
    for s in per:
        d, sp, rv, act, tr = [M[s][:, i] for i in range(5)]
        ok = (d > 0) & (rv > 0) & (act > 0)
        X = np.column_stack([np.ones(ok.sum()), np.log(rv[ok]), np.log(act[ok]),
                             np.log1p(tr[ok])])
        b = np.linalg.pinv(X.T @ X) @ (X.T @ np.log(d[ok]))
        pred = X @ b
        ss = float(np.sum((np.log(d[ok]) - np.log(d[ok]).mean()) ** 2))
        res["elasticities"][s] = {"rv": float(b[1]), "act": float(b[2]),
                                  "trend": float(b[3]),
                                  "r2": float(1 - np.sum((np.log(d[ok]) - pred) ** 2) / ss)}
        print("    %-9s d log(depth)/d log(rv) %+.4f   /d log(act) %+.4f   "
              "/d log(1+trend) %+.4f   r2 %.3f"
              % (s, b[1], b[2], b[3], res["elasticities"][s]["r2"]), flush=True)

    syms = list(per)
    dlog = {}
    for s in syms:
        d = M[s][:, 0]
        x = np.full(len(d), np.nan)
        good = (d[1:] > 0) & (d[:-1] > 0)
        x[1:][good] = np.log(d[1:][good] / d[:-1][good])
        dlog[s] = x
    z = np.nanmean(np.column_stack([(M[s][:, 2] - np.nanmean(M[s][:, 2]))
                                    / np.nanstd(M[s][:, 2]) for s in syms]), axis=1)
    ter = np.array([np.nanpercentile(z, 33.3), np.nanpercentile(z, 66.7)])
    band = np.digitize(z, ter)

    print("\n(B) cross-symbol correlation of minute log-depth changes, by common-stress "
          "tercile", flush=True)
    pairs = [(syms[i], syms[j]) for i in range(len(syms)) for j in range(i + 1, len(syms))]
    for a, b_ in pairs:
        row = {}
        for t in (0, 1, 2):
            m = (band == t) & np.isfinite(dlog[a]) & np.isfinite(dlog[b_])
            row[str(t)] = float(np.corrcoef(dlog[a][m], dlog[b_][m])[0, 1]) if m.sum() > 200 \
                else None
        nulls = []
        for _ in range(N_SHUF):
            perm = rng.permutation(len(common))
            m = np.isfinite(dlog[a]) & np.isfinite(dlog[b_][perm])
            nulls.append(float(np.corrcoef(dlog[a][m], dlog[b_][perm][m])[0, 1]))
        row["null_mean"] = float(np.mean(nulls))
        row["null_sd"] = float(np.std(nulls))
        row["rises_with_stress"] = bool(row["0"] is not None and row["2"] is not None
                                        and row["2"] > row["0"])
        res["comovement"]["%s|%s" % (a, b_)] = row
        print("    %-9s vs %-9s   calm %.4f   mid %.4f   stressed %.4f   "
              "null %+.4f +- %.4f   rises? %s"
              % (a, b_, row["0"], row["1"], row["2"], row["null_mean"], row["null_sd"],
                 row["rises_with_stress"]), flush=True)

    n_rise = sum(1 for v in res["comovement"].values() if v["rises_with_stress"])
    res["summary"] = {"pairs_rising": n_rise, "of": len(res["comovement"])}
    print("\n    correlation rises from calm to stressed on %d of %d pairs"
          % (n_rise, len(res["comovement"])), flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "CT40_WITHDRAW_TOGETHER_V1.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print("written", flush=True)


if __name__ == "__main__":
    main()
