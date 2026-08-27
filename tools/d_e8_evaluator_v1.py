# -*- coding: utf-8 -*-
"""D-E8 EVALUATOR V1 -- executes the frozen preregistration, and refuses to deviate from it.

The spec is `reports/atlas/D_E8_EDGE_LIFETIME_PREREGISTRATION_V2.md`, sha256
e7968ac4e933610e281b15709da3245b6d76662b2e963d1d9e30722a47332c4c.  This file HASHES THAT DOCUMENT
AT STARTUP and refuses to run if it has changed.  Section 9 of the prereg lists what execution may
not change; every one of those is a module constant here and none is a command-line flag.

  --spec       print the frozen constants and the support counts.  READS NO OUTCOME.
  --selftest   run the evaluator against synthetic worlds with KNOWN truth.  READS NO DATA.
  --estimate   run it.  REFUSES unless --selftest passes first, in the same process.

The selftest is not decoration.  An estimator that has never been run against a world whose answer
is known is a guard that has not been run against its own case (LANE_MIND_PROTOCOL_V1), and this
lane has already published one of those and had to repair it.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import pickle
import sqlite3
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------- FROZEN (prereg section 9)
PREREG = os.path.join(ROOT, "reports", "atlas",
                      "D_E8_EDGE_LIFETIME_PREREGISTRATION_V2.md")
PREREG_SHA256 = "e7968ac4e933610e281b15709da3245b6d76662b2e963d1d9e30722a47332c4c"

TAU_MIN = 60.0                       # section 5, fixed by rule
K_BPS = 10.0                         # section 3, canonical BINANCE_BASE round-trip taker
K_SENSITIVITIES = (4.0, 0.0)         # section 3, declared; no other k may be added
FLOOR_PRIMARY = 50_000.0             # section 1, the reproducible population
FLOOR_SECONDARY = 0.0                # section 1, sensitivity, carries D-E7's defect
SLIP_TOLERANCE_MS = 60_000           # section 4, cause SLIP_DROPPED
MIN_GAP_MS = 900_000                 # the episode rule
CUTOFF_MS = 1787270400000
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
SAMPLE = os.path.join(ROOT, "data", "pve_01_v1", "_s97_extended.pkl")
DB = os.path.join(ROOT, "data", "microstructure_02.db")
BOOT = 1000
CAUSES = ("EDGE_GONE", "INTERRUPTED", "ADMINISTRATIVE", "SLIP_DROPPED")
OUT = os.path.join(ROOT, "reports", "atlas", "D_E8_RESULTS_V1.json")


def assert_spec_unchanged():
    h = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
    if h != PREREG_SHA256:
        raise SystemExit("REFUSED: the preregistration has changed.\n  expected %s\n  found    %s\n"
                         "  A changed spec is a NEW prereg with a NEW hash (section 9)."
                         % (PREREG_SHA256, h))
    return h


# ---------------------------------------------------------------- the estimator
def alive_spell(path_ms, path_px, t0_ms, p_ref, d, k_bps, tau_ms):
    """Prereg section 3.  Returns (cause, time_ms).

    ALIVE(t) <=> d * (P(t)/p_ref - 1) * 1e4 >= k.  T_1 is the end of the FIRST alive spell.
    """
    m = (path_ms >= t0_ms) & (path_ms <= t0_ms + tau_ms)
    ts, px = path_ms[m], path_px[m]
    if len(ts) < 2:
        return "ADMINISTRATIVE", tau_ms
    r = d * (px / p_ref - 1.0) * 1e4
    alive = r >= k_bps
    if not alive.any():
        return "NEVER_ALIVE", 0
    i0 = int(np.argmax(alive))                      # first True
    after = np.flatnonzero(~alive[i0:])
    if len(after) == 0:
        return "ADMINISTRATIVE", tau_ms             # still alive at tau
    return "EDGE_GONE", int(ts[i0 + after[0]] - t0_ms)


def aalen_johansen(times_ms, causes, tau_ms, grid=241):
    """P00, CIF_1, CIF_2 on a fixed grid.  Cause 3 censors; causes 1 and 2 compete.

    SLIP_DROPPED rows must be removed by the caller and counted -- they are not censoring.
    """
    g = np.linspace(0.0, tau_ms, grid)
    t = np.asarray(times_ms, float)
    c = np.asarray(causes)
    n = len(t)
    p00 = np.ones(grid)
    cif1 = np.zeros(grid)
    cif2 = np.zeros(grid)
    s = 1.0
    c1 = c2 = 0.0
    order = np.argsort(t)
    ti, ci = t[order], c[order]
    j = 0
    for gi in range(grid):
        while j < n and ti[j] <= g[gi]:
            risk = n - j
            if risk > 0 and ci[j] in ("EDGE_GONE", "INTERRUPTED"):
                inc = s / risk
                if ci[j] == "EDGE_GONE":
                    c1 += inc
                else:
                    c2 += inc
                s *= (1.0 - 1.0 / risk)
            elif risk > 0 and ci[j] == "NEVER_ALIVE":
                inc = s / risk
                c1 += inc
                s *= (1.0 - 1.0 / risk)
            j += 1
        p00[gi], cif1[gi], cif2[gi] = s, c1, c2
    return g, p00, cif1, cif2


def mu_tau(g, p00):
    """Prereg section 5: integral_0^tau P00(u) du, in MINUTES."""
    integ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integ(p00, g) / 60000.0)


# ---------------------------------------------------------------- selftest
def _synth(n, kind, tau_ms, rng):
    """Worlds with a KNOWN answer."""
    t, c = [], []
    for _ in range(n):
        if kind == "always_alive":
            t.append(tau_ms); c.append("ADMINISTRATIVE")
        elif kind == "never_alive":
            t.append(0); c.append("NEVER_ALIVE")
        elif kind == "half_at_half":
            t.append(tau_ms / 2); c.append("EDGE_GONE")
        elif kind == "exp_rate":
            x = rng.exponential(tau_ms / 2.0)
            if x >= tau_ms:
                t.append(tau_ms); c.append("ADMINISTRATIVE")
            else:
                t.append(x); c.append("EDGE_GONE")
    return np.array(t, float), np.array(c)


def selftest(verbose=True):
    tau = TAU_MIN * 60000.0
    rng = np.random.default_rng(20260827)
    checks = []

    def chk(name, got, want, tol):
        ok = abs(got - want) <= tol
        checks.append({"check": name, "got": round(got, 4), "expected": round(want, 4),
                       "tol": tol, "pass": bool(ok)})
        return ok

    # 1. everything survives to tau  ->  mu_tau = tau
    t, c = _synth(500, "always_alive", tau, rng)
    g, p00, c1, c2 = aalen_johansen(t, c, tau)
    chk("always_alive: mu_tau == tau", mu_tau(g, p00), TAU_MIN, 0.01)
    chk("always_alive: CIF_1(tau) == 0", float(c1[-1]), 0.0, 1e-9)

    # 2. nothing is ever alive  ->  mu_tau = 0, CIF_1 = 1
    t, c = _synth(500, "never_alive", tau, rng)
    g, p00, c1, c2 = aalen_johansen(t, c, tau)
    chk("never_alive: mu_tau == 0", mu_tau(g, p00), 0.0, 0.01)
    chk("never_alive: CIF_1(tau) == 1", float(c1[-1]), 1.0, 1e-6)

    # 3. all fail at tau/2  ->  mu_tau = tau/2
    t, c = _synth(500, "half_at_half", tau, rng)
    g, p00, c1, c2 = aalen_johansen(t, c, tau)
    chk("half_at_half: mu_tau == tau/2", mu_tau(g, p00), TAU_MIN / 2.0, 0.5)

    # 4. exponential with mean tau/2, administratively censored at tau.
    #    E[min(T,tau)] = mean*(1-exp(-tau/mean)) = (tau/2)*(1-e^-2)
    t, c = _synth(20000, "exp_rate", tau, rng)
    g, p00, c1, c2 = aalen_johansen(t, c, tau)
    want = (TAU_MIN / 2.0) * (1.0 - math.exp(-2.0))
    chk("exponential: mu_tau == mean*(1-exp(-tau/mean))", mu_tau(g, p00), want, 0.5)

    # 5. the alive-spell rule itself, on a hand-built path
    ms = np.arange(0, 3_600_001, 1000, dtype=np.int64)
    px = np.full(len(ms), 100.0)
    px[600:1200] = 100.3                      # +30 bps for 10 minutes from t+600s
    cause, tt = alive_spell(ms, px, 0, 100.0, +1.0, K_BPS, tau)
    chk("alive_spell finds the end of the first spell (minutes)",
        tt / 60000.0, 1200 / 60.0, 0.05)
    checks.append({"check": "alive_spell cause is EDGE_GONE", "got": cause,
                   "expected": "EDGE_GONE", "tol": 0, "pass": cause == "EDGE_GONE"})

    # 6. a path that never clears k must be NEVER_ALIVE, not EDGE_GONE
    px2 = np.full(len(ms), 100.05)            # +5 bps, below k = 10
    cause2, tt2 = alive_spell(ms, px2, 0, 100.0, +1.0, K_BPS, tau)
    checks.append({"check": "below k is NEVER_ALIVE not EDGE_GONE", "got": cause2,
                   "expected": "NEVER_ALIVE", "tol": 0, "pass": cause2 == "NEVER_ALIVE"})

    # 7. direction is honoured: the same path with d = -1 must NOT be alive
    cause3, _ = alive_spell(ms, px, 0, 100.0, -1.0, K_BPS, tau)
    checks.append({"check": "d = -1 inverts the sign", "got": cause3,
                   "expected": "NEVER_ALIVE", "tol": 0, "pass": cause3 == "NEVER_ALIVE"})

    ok = all(c["pass"] for c in checks)
    if verbose:
        for c in checks:
            print("  [%s] %-52s got %-12s expected %s"
                  % ("PASS" if c["pass"] else "FAIL", c["check"], c["got"], c["expected"]))
        print("  selftest: %d/%d" % (sum(c["pass"] for c in checks), len(checks)))
    return ok, checks


# ---------------------------------------------------------------- data
def episodes(floor):
    d = pickle.loads(open(SAMPLE, "rb").read())
    rows = d["rows"] if isinstance(d, dict) and "rows" in d else d
    out = collections.defaultdict(list)
    for r in rows:
        if float(r["q"]) >= floor:
            out[r["sym"]].append((int(r["t0"]), float(r["d"]), float(r["q"])))
    return {s: sorted(v) for s, v in out.items()}


def support(floor):
    eps = episodes(floor)
    days = set()
    for s, v in eps.items():
        for t0, _, _ in v:
            days.add((s, t0 // 86400000))
    return {"floor_usd": floor,
            "n_episodes": {s: len(v) for s, v in sorted(eps.items())},
            "total": sum(len(v) for v in eps.values()),
            "symbol_day_strata": len(days)}


def marks(cn, sym):
    r = cn.execute("SELECT ts_ms,mark_price FROM mark_prices WHERE symbol=? AND ts_ms<? "
                   "ORDER BY ts_ms", (sym, CUTOFF_MS)).fetchall()
    return (np.array([x[0] for x in r], np.int64),
            np.array([x[1] for x in r], float))


def build(floor, k_bps):
    """One row per episode: (cause, time_ms, symbol, day).  Reads mark prices."""
    tau_ms = int(TAU_MIN * 60000)
    eps = episodes(floor)
    cn = sqlite3.connect("file:%s?mode=ro" % DB, uri=True, timeout=300)
    cn.execute("PRAGMA query_only=ON")
    rows, dropped = [], collections.Counter()
    for s, v in eps.items():
        ms, px = marks(cn, s)
        t0s = np.array([x[0] for x in v], np.int64)
        nxt = np.append(t0s[1:], np.int64(1 << 62))
        for (t0, d, q), nt in zip(v, nxt):
            i0 = np.searchsorted(ms, t0, side="right") - 1
            if i0 < 1:
                dropped["no_reference"] += 1
                continue
            # SLIP gate, prereg section 4 cause 4
            j = np.searchsorted(ms, t0 + tau_ms, side="left")
            if j >= len(ms) or (ms[j] - (t0 + tau_ms)) > SLIP_TOLERANCE_MS:
                dropped["SLIP_DROPPED"] += 1
                continue
            cause, tt = alive_spell(ms, px, t0, float(px[i0 - 1]), d, k_bps, tau_ms)
            # cause 2 INTERRUPTED: the next same-symbol episode arrives first
            if nt - t0 < tau_ms and (cause == "ADMINISTRATIVE" or tt > nt - t0):
                cause, tt = "INTERRUPTED", int(nt - t0)
            rows.append({"cause": cause, "t_ms": float(tt), "sym": s,
                         "stratum": "%s|%d" % (s, t0 // 86400000)})
    cn.close()
    return rows, dict(dropped)


def estimate(rows, label):
    tau_ms = TAU_MIN * 60000.0
    t = np.array([r["t_ms"] for r in rows], float)
    c = np.array([r["cause"] for r in rows])
    g, p00, cif1, cif2 = aalen_johansen(t, c, tau_ms)
    m = mu_tau(g, p00)
    strata = np.array([r["stratum"] for r in rows])
    uniq = np.unique(strata)
    rng = np.random.default_rng(20260827)
    bs = []
    for _ in range(BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(strata == u) for u in pick])
        gg, pp, _, _ = aalen_johansen(t[idx], c[idx], tau_ms)
        bs.append(mu_tau(gg, pp))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"label": label, "n": len(rows),
            "cause_counts": {k: int(v) for k, v in collections.Counter(c).items()},
            "never_alive_share": round(float((c == "NEVER_ALIVE").mean()), 4),
            "mu_tau_minutes": round(m, 4),
            "ci95_symbol_day_cluster": [round(float(lo), 4), round(float(hi), 4)],
            "n_strata": int(len(uniq)),
            "P00_at_tau": round(float(p00[-1]), 4),
            "CIF_edge_gone_at_tau": round(float(cif1[-1]), 4),
            "CIF_interrupted_at_tau": round(float(cif2[-1]), 4),
            "defective_distribution": bool(p00[-1] > 0.05)}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    a = ap.parse_args()
    h = assert_spec_unchanged()
    print("D-E8 EVALUATOR V1   prereg sha256 %s  VERIFIED" % h[:16])

    if a.spec or not any([a.selftest, a.estimate]):
        print(json.dumps({"frozen": {"tau_min": TAU_MIN, "k_bps": K_BPS,
                                     "k_sensitivities": list(K_SENSITIVITIES),
                                     "floor_primary": FLOOR_PRIMARY,
                                     "floor_secondary": FLOOR_SECONDARY,
                                     "slip_tolerance_ms": SLIP_TOLERANCE_MS,
                                     "causes": list(CAUSES)},
                          "support": [support(FLOOR_PRIMARY), support(FLOOR_SECONDARY)]},
                         indent=1))
        if not (a.selftest or a.estimate):
            return 0

    if a.selftest or a.estimate:
        print("\nSELFTEST -- synthetic worlds with known truth, no data read:")
        ok, checks = selftest()
        if not ok:
            raise SystemExit("REFUSED: selftest failed.  The evaluator does not estimate until it "
                             "reproduces worlds whose answer is known.")
        if not a.estimate:
            return 0

    print("\nESTIMATE -- reading mark prices for the first time in lane D.")
    res = {"prereg_sha256": h, "selftest": "PASS", "arms": []}
    for floor, tag in ((FLOOR_PRIMARY, "PRIMARY"), (FLOOR_SECONDARY, "SENSITIVITY_floor0")):
        for k in (K_BPS,) + (K_SENSITIVITIES if floor == FLOOR_PRIMARY else ()):
            rows, dropped = build(floor, k)
            r = estimate(rows, "%s floor=$%d k=%.1fbps" % (tag, floor, k))
            r["dropped"] = dropped
            res["arms"].append(r)
            print("  %-38s n=%-5d mu_tau=%.2f min  CI [%.2f, %.2f]  never_alive=%.1f%%  P00(tau)=%.3f"
                  % (r["label"], r["n"], r["mu_tau_minutes"],
                     r["ci95_symbol_day_cluster"][0], r["ci95_symbol_day_cluster"][1],
                     100 * r["never_alive_share"], r["P00_at_tau"]))
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1))
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
