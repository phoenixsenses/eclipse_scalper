r"""LANE C, round 32 -- a threat to C-T29's own headline, and the test that decides it.

C-T29 reported that liquidity providers cancel 57.7% / 74.9% / 102.3% of the order flow's long
memory, reading the gap between the flow exponent chi and the price exponent H as cancellation.
C-T30 refuted the tick explanation for the spread and left one lead: a SHARE effect. This round
takes that lead seriously enough to point it at C-T29.

THE THREAT. Suppose impact is perfectly PERMANENT -- nothing is cancelled at all -- and the price
is simply

    d_t = G . (eps_t v_t) + noise_t ,      noise iid, no temporal structure whatsoever

Then the variance of the price over T trades is

    Var(sum_T d) = G^2 . Var(sum_T eps v)  +  sigma^2 . T
                 ~ G^2 . c . T^(2 chi)     +  sigma^2 . T

which is a SUM OF TWO POWERS, not one. Fitted over a finite window of T it returns an EFFECTIVE
exponent lying between chi and 1/2, set by the RATIO of the two terms -- that is, by the share of
price variance the flow explains. So a measured H below chi does not by itself demonstrate that
anything was cancelled. Zero cancellation produces exactly the same signature.

THE TEST. Build that zero-cancellation world out of the REAL data, so that the flow's long memory,
the real volumes, the real impact coefficient and the real noise level are all preserved, and only
the cancellation is removed by construction:

    G, sigma  from regressing the real d_t on the real (eps_t v_t)
    d_sim     = G . (eps_t v_t) + iid draws matched to the real residual variance

Run the identical estimator on d_sim. Then:

    H_sim ~= H_obs   -> the "cancellation" reading is WRONG. The gap is a two-term crossover and
                        C-T29's headline number measures the flow's SHARE of price variance, not
                        the counter-balancing role of liquidity providers.
    H_sim >  H_obs   -> genuine cancellation exists, and H_sim - H_obs is how much of the gap it
                        actually accounts for. The rest is crossover.

Both outcomes are informative and one of them costs this lane its previous headline. The residual
is drawn iid rather than resampled in blocks precisely BECAUSE iid is the no-cancellation
assumption: any block structure would smuggle back the decay this test is trying to remove.

A second reading falls out for free: the share of price variance explained by contemporaneous
signed flow, per T, which is the blend weight itself.
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
WINDOW_T = (20, 50, 100, 200, 500, 1000)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEED = 20260827

# C-T29's published readings, for the confrontation
CT29 = {"BTCUSDT": {"H": 0.6175, "chi": 0.7776, "cancelled": 0.577},
        "ETHUSDT": {"H": 0.5512, "chi": 0.7043, "cancelled": 0.749},
        "SOLUSDT": {"H": 0.4911, "chi": 0.8918, "cancelled": 1.023}}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def slope(T, S):
    x = np.log(np.asarray(T, float))
    y = np.log(np.asarray(S, float))
    A = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(b[1])


def hurst_and_share(d, flow):
    """H of the price, and the share of price variance explained by contemporaneous flow, per T"""
    n = len(d)
    T, S, share = [], [], []
    for t in WINDOW_T:
        m = n // t
        if m < 200:
            continue
        sd_ = d[:m * t].reshape(m, t).sum(axis=1)
        sf = flow[:m * t].reshape(m, t).sum(axis=1)
        T.append(float(t))
        S.append(float(np.std(sd_, ddof=1)))
        r = float(np.corrcoef(sd_, sf)[0, 1])
        share.append(round(r * r, 4))
    return slope(T, S), T, share


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
            d = np.empty_like(lp)
            d[0] = 0.0
            d[1:] = np.diff(lp)
            eps = np.where(a[:, 2] > 0.5, -1.0, 1.0)
            flow = eps * a[:, 1]
            n = len(d)

            # calibrate the permanent-impact world on the real data
            G = float(np.dot(flow, d) / np.dot(flow, flow))
            resid = d - G * flow
            sigma = float(np.std(resid, ddof=1))
            r2_trade = float(np.corrcoef(d, flow)[0, 1] ** 2)

            H_obs, Tg, share_obs = hurst_and_share(d, flow)

            sims = []
            for _ in range(REPS):
                d_sim = G * flow + rng.normal(0.0, sigma, n)
                sims.append(hurst_and_share(d_sim, flow)[0])
            H_sim = float(np.mean(sims))
            sd_sim = float(np.std(sims, ddof=1))

            chi = CT29[sym]["chi"]
            canc_obs = 1.0 - (H_obs - 0.5) / (chi - 0.5)
            canc_sim = 1.0 - (H_sim - 0.5) / (chi - 0.5)
            per[sym] = {
                "G_impact_per_unit_flow": G,
                "sigma_residual": sigma,
                "r2_at_trade_level": round(r2_trade, 6),
                "H_observed": round(H_obs, 4),
                "H_zero_cancellation_world": round(H_sim, 4),
                "H_sim_sd": round(sd_sim, 4),
                "chi": chi,
                "cancellation_reported_by_c_t29": round(canc_obs, 4),
                "cancellation_a_zero_cancellation_world_would_show": round(canc_sim, 4),
                "gap_explained_by_crossover_alone": (
                    round(canc_sim / canc_obs, 3) if canc_obs != 0 else None),
                "genuine_residual_cancellation": round(canc_obs - canc_sim, 4),
                "T_grid": [int(t) for t in Tg],
                "flow_share_of_price_variance_by_T": share_obs,
            }
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    art = {"study": "C-T32", "lane": "C", "utc": _utc(), "reps": REPS, "seed": SEED,
           "threat": ("Var(sum_T d) = G^2 c T^(2chi) + sigma^2 T is a sum of two powers; fitted "
                      "over a finite T window it returns an effective exponent between chi and "
                      "1/2 set by the flow's share of price variance. Zero cancellation produces "
                      "the same signature C-T29 read as cancellation."),
           "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C32_IS_CANCELLATION_REAL_V1.json").write_text(json.dumps(art, indent=2),
                                                          encoding="utf-8")
    enc = sys.stdout.encoding or "utf-8"
    sys.stdout.write(json.dumps(per, indent=2).encode(enc, "replace").decode(enc, "replace")
                     + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
