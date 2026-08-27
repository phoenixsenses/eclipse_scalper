r"""LANE C, round 44 -- three challenges from other lanes, all of them run.

The shared log has moved a long way since this lane last read it: lane A is at A-S63, and a lane D
now exists. Three of their `to C` messages bear directly on claims this lane has published, and
all three are runnable here. Per the operator's standing instruction, all three are run rather
than one being chosen.

CHALLENGE 1 -- A-S62. "the saturation lag is now measured on this estate's own data: 40-60
minutes, and it SHORTENS with event size. That is a direct constraint on the propagator you are
fitting -- G(l) must saturate on that timescale here, and its saturation time is SIZE-DEPENDENT,
which the standard form does not carry."

  Correct: G(l) ~ l^-beta has no saturation time and no size dependence. This lane fitted that
  form in trade time (C-T34, C-T35) and never asked either question. Both are checked: where does
  the measured response saturate, in minutes, and does that lag shorten with trade size?

CHALLENGE 2 -- D-E3. "false protectivity (ABG 6.6), declining hazard ratio by distance-to-barrier
(10.3.2), and crossover-by-frailty-selection (6.5.2) all produce a relative effect that DECLINES
and can cross below one, with NO change at the individual level... If any of your amplitude or
exponent ratios decline with horizon, that is now three textbook nulls deep before it is a
finding."

  This lane has exactly one such curve: C-T42's latency ladder, where the edge falls 63% (BTC) and
  89% (ETH) over 50 trades of delay. Frailty selection would produce that shape with no individual
  decay at all -- if the events carrying the d=0 edge are a fast-resolving subset, the aggregate
  declines while most events are flat. The event SET is identical at every delay here, so
  composition cannot change; but heterogeneity within the set can still generate the shape. The
  test is to split events by their own d=0 outcome and watch each stratum decay separately.

CHALLENGE 3 -- D-E4. "the three symbols' liquidation-episode arrivals co-fire at 6.2x chance
within +-1 minute... if any of your cross-symbol amplitude comparisons treat BTC/ETH/SOL as
independent draws, the effective count is closer to 1 than to 3 at short horizons."

  This lane has published "3 of 3", "12 of 12 cells" and "6 of 6 cases" repeatedly, and every one
  of those counts treats the symbols as independent. The effective count is measured here on this
  lane's OWN quantities -- the block-level edge and the block-level sign-memory exponent -- rather
  than inherited from lane D's episode arrivals, and the published counts are restated.
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
LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
NBLOCK = 20
T0, DELAYS = 50, (0, 1, 5, 10, 20, 50)
SEED = 20260827


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def xcorr(a, b, L):
    n = len(a)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.conj(np.fft.rfft(a, m)) * np.fft.rfft(b, m), m)[:L + 1]
    return c / (n - np.arange(L + 1))


def response(lp, eps, lags):
    """R(l) = <(m_{t+l} - m_t) eps_t> in bps, by FFT"""
    r = np.empty_like(lp)
    r[0] = 0.0
    r[1:] = np.diff(lp) * 1e4
    S = xcorr(eps, r, max(lags))
    cum = np.cumsum(S)
    return {l: float(cum[l - 1]) for l in lags if l - 1 < len(cum)}


def saturation_lag(R, frac=0.90):
    """first lag at which R reaches `frac` of its maximum over the measured range"""
    ks = sorted(R)
    v = [R[k] for k in ks]
    mx = max(v)
    if mx <= 0:
        return None
    for k, x in zip(ks, v):
        if x >= frac * mx:
            return k
    return None


def main() -> int:
    rng = np.random.default_rng(SEED)
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    ch1, ch2, blocks = {}, {}, {}
    try:
        for sym in SYMS:
            a = np.array(con.execute(
                "select ts_ms,price,notional,is_buyer_maker from agg_trades where symbol=? "
                "order by ts_ms limit ?", (sym, NROWS)).fetchall(), dtype=np.float64)
            ts, px, vol = a[:, 0], a[:, 1], a[:, 2]
            lp = np.log(px)
            eps = np.where(a[:, 3] > 0.5, -1.0, 1.0)
            n = len(lp)
            span_min = (ts[-1] - ts[0]) / 60000.0
            tpm = n / span_min                        # trades per minute

            # --- CHALLENGE 1: saturation lag, and its size dependence
            R_all = response(lp, eps, LAGS)
            sat_all = saturation_lag(R_all)
            big = vol >= np.percentile(vol, 90)
            sml = vol <= np.percentile(vol, 50)
            R_big = response(lp, np.where(big, eps, 0.0), LAGS)
            R_sml = response(lp, np.where(sml, eps, 0.0), LAGS)
            s_big, s_sml = saturation_lag(R_big), saturation_lag(R_sml)
            ch1[sym] = {
                "trades_per_minute": round(tpm, 2),
                "span_minutes": round(span_min, 1),
                "A_S62_lag_40_60_min_in_trades": [int(40 * tpm), int(60 * tpm)],
                "saturation_lag_trades_all": sat_all,
                "saturation_lag_minutes_all": (round(sat_all / tpm, 1) if sat_all else None),
                "saturation_lag_trades_large": s_big,
                "saturation_lag_minutes_large": (round(s_big / tpm, 1) if s_big else None),
                "saturation_lag_trades_small": s_sml,
                "saturation_lag_minutes_small": (round(s_sml / tpm, 1) if s_sml else None),
                "large_saturates_faster": (bool(s_big < s_sml) if (s_big and s_sml) else None),
                "R_curve_bps": {str(k): round(v, 5) for k, v in R_all.items()},
            }

            # --- CHALLENGE 2: is the latency decay a within-stratum decay or heterogeneity?
            m = (n - 1) // T0
            i0 = np.arange(1, m - 1) * T0
            s = np.sign((eps * vol)[:m * T0].reshape(m, T0).sum(axis=1))[1:len(i0) + 1]
            keep = s != 0
            i0k, sk = i0[keep], s[keep]
            g = {}
            for d in DELAYS:
                e_ = np.clip(i0k + T0 - 1 + d, 0, n - 1)
                x_ = np.clip(e_ + T0, 0, n - 1)
                g[d] = (lp[x_] - lp[e_]) * 1e4 * sk
            q = np.percentile(g[0], [25, 50, 75])
            strata = {}
            for lab, lo, hi in (("q1_worst", -np.inf, q[0]), ("q2", q[0], q[1]),
                                ("q3", q[1], q[2]), ("q4_best", q[2], np.inf)):
                mm = (g[0] > lo) & (g[0] <= hi)
                prof = {str(d): round(float(g[d][mm].mean()), 4) for d in DELAYS}
                base = prof["0"]
                strata[lab] = {"n": int(mm.sum()), "profile": prof,
                               "retained_at_d50": (round(prof["50"] / base, 3)
                                                   if base != 0 else None)}
            ch2[sym] = {"aggregate_profile": {str(d): round(float(g[d].mean()), 4)
                                              for d in DELAYS},
                        "aggregate_retained_at_d50": round(float(g[50].mean() / g[0].mean()), 3),
                        "by_d0_stratum": strata,
                        "note": ("the event SET is identical at every delay, so composition "
                                 "cannot change; only within-set heterogeneity could produce "
                                 "the shape without individual decay")}

            # --- CHALLENGE 3: block-level series for the cross-symbol dependence
            bm = n // NBLOCK
            be, bc = [], []
            for b in range(NBLOCK):
                sl = slice(b * bm, (b + 1) * bm)
                lpb, epsb, volb = lp[sl], eps[sl], vol[sl]
                mm = (len(lpb) - 1) // T0
                ii = np.arange(1, mm - 1) * T0
                ss = np.sign((epsb * volb)[:mm * T0].reshape(mm, T0).sum(axis=1))[1:len(ii) + 1]
                en = np.clip(ii + T0 - 1, 0, len(lpb) - 1)
                ex = np.clip(en + T0, 0, len(lpb) - 1)
                gg = (lpb[ex] - lpb[en]) * 1e4 * ss
                be.append(float(gg[ss != 0].mean()))
                Ts = np.unique(np.round(np.geomspace(20, 1000, 8)).astype(int))
                T, S = [], []
                for t in Ts:
                    k = len(epsb) // t
                    if k < 100:
                        continue
                    T.append(float(t))
                    S.append(float(np.std(epsb[:k * t].reshape(k, t).sum(axis=1), ddof=1)))
                A = np.column_stack([np.ones(len(T)), np.log(T)])
                bb, *_ = np.linalg.lstsq(A, np.log(S), rcond=None)
                bc.append(float(bb[1]))
            blocks[sym] = {"edge": be, "chi": bc}
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    # effective number of independent symbols, on THIS lane's own quantities
    ch3 = {}
    for q in ("edge", "chi"):
        M = np.array([blocks[s][q] for s in SYMS])
        C = np.corrcoef(M)
        off = [C[i, j] for i in range(3) for j in range(i + 1, 3)]
        rbar = float(np.mean(off))
        n_eff_kish = 3.0 / (1.0 + 2.0 * rbar) if (1 + 2 * rbar) > 0 else None
        ev = np.linalg.eigvalsh(C)
        n_eff_ent = float(np.exp(-np.sum((ev / ev.sum()) * np.log(ev / ev.sum() + 1e-300))))
        ch3[q] = {"pairwise_correlations": {"BTC-ETH": round(float(C[0, 1]), 4),
                                            "BTC-SOL": round(float(C[0, 2]), 4),
                                            "ETH-SOL": round(float(C[1, 2]), 4)},
                  "mean_r": round(rbar, 4),
                  "n_eff_kish": round(n_eff_kish, 3) if n_eff_kish else None,
                  "n_eff_entropy": round(n_eff_ent, 3),
                  "blocks": NBLOCK}

    art = {"study": "C-T44", "lane": "C", "utc": _utc(),
           "challenge_1_from": "A-S62 (saturation lag 40-60 min, size-dependent)",
           "challenge_2_from": "D-E3 (a declining relative effect has three textbook nulls)",
           "challenge_3_from": "D-E4 (symbols co-fire; effective count nearer 1 than 3)",
           "ch1_saturation": ch1, "ch2_latency_strata": ch2,
           "ch3_effective_symbol_count": ch3}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C44_CROSS_LANE_CHALLENGES_V1.json").write_text(json.dumps(art, indent=2),
                                                           encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    w("CHALLENGE 1 (A-S62) -- where does the response saturate, and is the lag size-dependent?")
    w("%-9s %8s %16s %14s %14s %14s %10s" % ("sym", "tr/min", "A-S62 40-60min", "sat all",
                                             "sat LARGE", "sat SMALL", "large<small"))
    for s in SYMS:
        c = ch1[s]
        w("%-9s %8.1f %16s %14s %14s %14s %10s" % (
            s, c["trades_per_minute"], c["A_S62_lag_40_60_min_in_trades"],
            "{0} ({1}m)".format(c["saturation_lag_trades_all"],
                                c["saturation_lag_minutes_all"]),
            "{0} ({1}m)".format(c["saturation_lag_trades_large"],
                                c["saturation_lag_minutes_large"]),
            "{0} ({1}m)".format(c["saturation_lag_trades_small"],
                                c["saturation_lag_minutes_small"]),
            c["large_saturates_faster"]))
    w("")
    w("CHALLENGE 2 (D-E3) -- latency decay: uniform within strata, or heterogeneity?")
    for s in SYMS:
        c = ch2[s]
        w("  %-9s aggregate retained at d=50: %s" % (s, c["aggregate_retained_at_d50"]))
        for lab, v in c["by_d0_stratum"].items():
            w("      %-9s n=%-6d profile %s   retained %s" % (
                lab, v["n"], " ".join("%s:%s" % (k, x) for k, x in v["profile"].items()),
                v["retained_at_d50"]))
    w("")
    w("CHALLENGE 3 (D-E4) -- effective number of independent symbols, on this lane's quantities")
    for q, v in ch3.items():
        w("  %-6s pairwise %s  mean r %s  n_eff(Kish) %s  n_eff(entropy) %s" % (
            q, v["pairwise_correlations"], v["mean_r"], v["n_eff_kish"], v["n_eff_entropy"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
