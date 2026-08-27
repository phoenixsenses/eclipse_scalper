# -*- coding: utf-8 -*-
"""D-E11 -- P2 and P3 of the frozen preregistration.

P2  CIF_1 and mu_tau against a CALIBRATED NULL.
    The null is the one the corpus names for this object: a DRIFTLESS walk with the episode's own
    PRE-ANCHOR volatility.  ABG chapter 10 gives the first-passage time of a Wiener process; with
    the drift set to zero, what remains is pure barrier geometry.  So the question P2 asks is the
    only one that matters: IS 18.10 MINUTES MORE THAN GEOMETRY ALONE WOULD GIVE?
    Volatility is estimated on [t0-60m, t0) -- strictly BEFORE the anchor, so calibrating the null
    consumes no outcome.

P3  The size contrast, reported TWICE and labelled.
    The prereg V2 restricts a hazard ratio to DESCRIPTIVE ONLY under Hernan & Robins Technical
    Point 8.1 -- conditioning on survival to t is conditioning on a collider, so the ratio is
    biased even under randomisation and its SIGN can invert with elapsed time.  H&R's own remedy in
    that passage is that RISK-type quantities stay unbiased.  So the same contrast is reported both
    ways: the cumulative `mu_tau` difference (usable) and the hazard-ratio reading (descriptive).

Reuses the evaluator's frozen constants and its `alive_spell` verbatim, so P2/P3 cannot drift from
what D-E10 executed.  The prereg hash is checked at startup by the import.

Usage:  python tools/d_e11_p2_p3_v1.py
"""
from __future__ import annotations

import collections
import json
import math
import os
import sqlite3
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.d_e8_evaluator_v1 import (  # noqa: E402
    CUTOFF_MS, DB, FLOOR_PRIMARY, K_BPS, SLIP_TOLERANCE_MS, TAU_MIN,
    aalen_johansen, alive_spell, assert_spec_unchanged, episodes, marks, mu_tau)

OUT = os.path.join(ROOT, "reports", "atlas", "D_E11_P2_P3_V1.json")
REPS = 200
PRE_VOL_MIN = 60
SEED = 20260827


def collect(floor, k_bps):
    """Observed rows PLUS the per-episode pre-anchor volatility the null needs."""
    tau_ms = int(TAU_MIN * 60000)
    eps = episodes(floor)
    cn = sqlite3.connect("file:%s?mode=ro" % DB, uri=True, timeout=300)
    cn.execute("PRAGMA query_only=ON")
    rows = []
    for s, v in eps.items():
        ms, px = marks(cn, s)
        t0s = np.array([x[0] for x in v], np.int64)
        nxt = np.append(t0s[1:], np.int64(1 << 62))
        for (t0, d, q), nt in zip(v, nxt):
            i0 = np.searchsorted(ms, t0, side="right") - 1
            if i0 < 1:
                continue
            j = np.searchsorted(ms, t0 + tau_ms, side="left")
            if j >= len(ms) or (ms[j] - (t0 + tau_ms)) > SLIP_TOLERANCE_MS:
                continue
            # PRE-ANCHOR volatility, strictly before t0 -- calibrating the null reads no outcome
            a = np.searchsorted(ms, t0 - PRE_VOL_MIN * 60000, side="left")
            pre = px[a:i0]
            if len(pre) < 60:
                continue
            lr = np.diff(np.log(pre[pre > 0]))
            sig = float(np.std(lr)) if len(lr) > 10 else 0.0
            # POST-anchor volatility too, as a HARDER null.  Pre-anchor vol understates the
            # cascade's own volatility, which would make the driftless null cross the barrier
            # LESS often than it should and flatter me.  Volatility is not the outcome here --
            # the outcome is the SIGNED path -- so calibrating a symmetric null on it is legitimate
            # and it is the conservative direction.
            post = px[i0:j + 1]
            plr = np.diff(np.log(post[post > 0]))
            sig_post = float(np.std(plr)) if len(plr) > 10 else sig
            if not np.isfinite(sig) or sig <= 0:
                continue
            cause, tt = alive_spell(ms, px, t0, float(px[i0 - 1]), d, k_bps, tau_ms)
            if nt - t0 < tau_ms and (cause == "ADMINISTRATIVE" or tt > nt - t0):
                cause, tt = "INTERRUPTED", int(nt - t0)
            rows.append({"cause": cause, "t_ms": float(tt), "sym": s,
                         # t0 EXPOSED D-E35, additively.  `stratum` carries the day only, and a
                         # cross-symbol look-back window needs the anchor itself.  This adds a key
                         # and changes NOTHING that is computed: verified by pinning n=628,
                         # mu_tau=18.104100 and a hash of (sym, cause, t_ms) across the change.
                         "t0_ms": int(t0),
                         "stratum": "%s|%d" % (s, t0 // 86400000),
                         "sigma_1s": sig, "sigma_1s_post": sig_post, "qv": float(q),
                         "next_ms": float(min(nt - t0, tau_ms + 1)), "d": d})
    cn.close()
    return rows


def simulate_once(rows, k_bps, rng, vol_key="sigma_1s"):
    """One driftless world.  Same barrier, same tau, same interruption times."""
    tau_ms = int(TAU_MIN * 60000)
    n_steps = tau_ms // 1000
    sig = np.array([r[vol_key] for r in rows], float)[:, None]
    z = rng.standard_normal((len(rows), n_steps))
    logp = np.cumsum(sig * z, axis=1)                     # driftless, per-episode vol
    r_bps = (np.exp(logp) - 1.0) * 1e4                    # d is absorbed: the walk is symmetric
    alive = r_bps >= k_bps
    t, c = [], []
    for i, row in enumerate(rows):
        a = alive[i]
        if not a.any():
            t.append(0.0); c.append("NEVER_ALIVE"); continue
        i0 = int(np.argmax(a))
        after = np.flatnonzero(~a[i0:])
        if len(after) == 0:
            tt, cc = float(tau_ms), "ADMINISTRATIVE"
        else:
            tt, cc = float((i0 + after[0] + 1) * 1000), "EDGE_GONE"
        if row["next_ms"] < tau_ms and (cc == "ADMINISTRATIVE" or tt > row["next_ms"]):
            tt, cc = row["next_ms"], "INTERRUPTED"
        t.append(tt); c.append(cc)
    return np.array(t), np.array(c)


def p2(rows, k_bps, vol_key="sigma_1s"):
    tau_ms = TAU_MIN * 60000.0
    t = np.array([r["t_ms"] for r in rows], float)
    c = np.array([r["cause"] for r in rows])
    g, p00, cif1, _ = aalen_johansen(t, c, tau_ms)
    obs_mu, obs_cif = mu_tau(g, p00), float(cif1[-1])
    rng = np.random.default_rng(SEED)
    mus, cifs, nas = [], [], []
    for _ in range(REPS):
        st, sc = simulate_once(rows, k_bps, rng, vol_key)
        gg, pp, cc1, _ = aalen_johansen(st, sc, tau_ms)
        mus.append(mu_tau(gg, pp)); cifs.append(float(cc1[-1]))
        nas.append(float((sc == "NEVER_ALIVE").mean()))
    mus, cifs = np.array(mus), np.array(cifs)
    return {"test": "P2_vs_driftless_null", "reps": REPS, "vol_source": vol_key,
            "null": "driftless walk at each episode's own PRE-ANCHOR 60m volatility, same barrier, "
                    "same tau, same interruption times",
            "observed_mu_tau_min": round(obs_mu, 4),
            "null_mu_tau_mean": round(float(mus.mean()), 4),
            "null_mu_tau_sd": round(float(mus.std(ddof=1)), 4),
            "null_mu_tau_p05_p95": [round(float(np.percentile(mus, 5)), 4),
                                    round(float(np.percentile(mus, 95)), 4)],
            "z_mu_tau": round(float((obs_mu - mus.mean()) / mus.std(ddof=1)), 2),
            "observed_CIF1_tau": round(obs_cif, 4),
            "null_CIF1_mean": round(float(cifs.mean()), 4),
            "null_CIF1_sd": round(float(cifs.std(ddof=1)), 4),
            "z_CIF1": round(float((obs_cif - cifs.mean()) / cifs.std(ddof=1)), 2),
            "observed_never_alive": round(float((c == "NEVER_ALIVE").mean()), 4),
            "null_never_alive_mean": round(float(np.mean(nas)), 4)}


def p3(rows):
    """The size contrast, twice: cumulative (usable) and hazard-ratio (descriptive only)."""
    tau_ms = TAU_MIN * 60000.0
    qv = np.array([r["qv"] for r in rows], float)
    lq = np.log(qv)
    cut = np.percentile(lq, [33.333, 66.667])
    lab = np.digitize(lq, cut)
    t = np.array([r["t_ms"] for r in rows], float)
    c = np.array([r["cause"] for r in rows])
    strata = np.array([r["stratum"] for r in rows])
    rng = np.random.default_rng(SEED)
    out = {}
    for g_ in (0, 1, 2):
        m = lab == g_
        gg, pp, cc1, _ = aalen_johansen(t[m], c[m], tau_ms)
        uniq = np.unique(strata[m])
        bs = []
        for _ in range(400):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.flatnonzero(strata[m] == u) for u in pick])
            g2, p2_, _, _ = aalen_johansen(t[m][idx], c[m][idx], tau_ms)
            bs.append(mu_tau(g2, p2_))
        out["tercile_%d" % g_] = {
            "n": int(m.sum()), "median_q_usd": round(float(np.median(qv[m])), 0),
            "mu_tau_min": round(mu_tau(gg, pp), 4),
            "ci95": [round(float(np.percentile(bs, 2.5)), 4),
                     round(float(np.percentile(bs, 97.5)), 4)],
            "CIF1_tau": round(float(cc1[-1]), 4),
            "never_alive": round(float((c[m] == "NEVER_ALIVE").mean()), 4)}
    # descriptive hazard-ratio reading, early vs late halves of [0, tau]
    mid = tau_ms / 2.0
    hr = {}
    for g_ in (0, 2):
        m = (lab == g_) & (c == "EDGE_GONE")
        early = float(((t[m] <= mid)).sum())
        late = float(((t[m] > mid)).sum())
        hr["tercile_%d" % g_] = {"early_events": early, "late_events": late,
                                 "late_over_early": round(late / early, 4) if early else None}
    return {"test": "P3_size_contrast",
            "cumulative_usable": out,
            "hazard_ratio_DESCRIPTIVE_ONLY": hr,
            "why_descriptive": "H&R Technical Point 8.1: conditioning on survival to t is "
                               "conditioning on a collider; the ratio is biased even under "
                               "randomisation and its sign can invert with elapsed time.  In the "
                               "same passage RISK-type quantities stay unbiased, which is why the "
                               "cumulative block above is the usable one."}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    h = assert_spec_unchanged()
    print("D-E11  P2/P3   prereg sha256 %s VERIFIED" % h[:16])
    rows = collect(FLOOR_PRIMARY, K_BPS)
    print("  rows with a usable pre-anchor volatility: %d" % len(rows))
    res = {"prereg_sha256": h, "n_rows": len(rows),
           "P2": p2(rows, K_BPS),
           "P2_harder_null_post_anchor_vol": p2(rows, K_BPS, "sigma_1s_post"),
           "P3": p3(rows)}
    a = res["P2"]
    print("\nP2   observed mu_tau %.2f min   null %.2f +/- %.2f  p05-p95 %s   z %+.2f"
          % (a["observed_mu_tau_min"], a["null_mu_tau_mean"], a["null_mu_tau_sd"],
             a["null_mu_tau_p05_p95"], a["z_mu_tau"]))
    print("     observed CIF_1(tau) %.4f   null %.4f +/- %.4f   z %+.2f"
          % (a["observed_CIF1_tau"], a["null_CIF1_mean"], a["null_CIF1_sd"], a["z_CIF1"]))
    print("     never_alive observed %.4f   null %.4f"
          % (a["observed_never_alive"], a["null_never_alive_mean"]))
    print("\nP3   mu_tau by log(Q/ADV) tercile  (cumulative = usable)")
    for k, v in res["P3"]["cumulative_usable"].items():
        print("     %-10s n=%-4d medQ $%-12.0f mu_tau %.2f  CI %s  never_alive %.3f"
              % (k, v["n"], v["median_q_usd"], v["mu_tau_min"], v["ci95"], v["never_alive"]))
    print("     hazard-ratio reading (DESCRIPTIVE ONLY): %s"
          % {k: v["late_over_early"] for k, v in res["P3"]["hazard_ratio_DESCRIPTIVE_ONLY"].items()})
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
