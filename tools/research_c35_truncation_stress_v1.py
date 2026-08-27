r"""LANE C, round 35 -- the one thing that could overturn C-T34, tested.

C-T34 concluded that on BTC and ETH the bare propagator does not appreciably decay over lags
4-128, so impact there is near-permanent and the book's efficiency condition fails. The book
itself names the danger in the same paragraph that recommends the method:

    "in practice, l must be smaller than the maximum lag L that allows a reasonable estimation of
     R and C, beyond which it becomes difficult to separate the signal from noise"

That cuts both ways. Truncating the Toeplitz system at L = 128 means the inversion can only see
decay that happens INSIDE 128 trades. If the true propagator decays over thousands of trades --
which is entirely plausible for a market whose sign memory runs to lag 2000 -- then a slow decay
would present as "flat" and I would have called it near-permanent.

TWO TESTS, IN THE ORDER THAT MATTERS.

  1. INSTRUMENT, ADVERSARIALLY. C-T34's recovery check built the synthetic propagator on exactly
     the same lag range it then inverted, so it could not detect this failure mode by
     construction. Here the synthetic G decays over a range MUCH LONGER than L, and the question
     is whether an inversion truncated at L still returns the right beta. If it does not, the
     recovery check in C-T34 was answering an easier question than the one that mattered.

  2. REAL DATA, SWEPT IN L. Invert at L = 64, 128, 256, 512, 1024 and watch beta. Stable in L
     means C-T34's reading survives; growing with L means "near-permanent" was truncation.

S(l) and C(l) are computed by FFT cross-correlation so that L = 1024 is affordable, and the FFT
is checked against the direct O(nL) loop at small lags before it is used for anything.
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
L_SWEEP = (64, 128, 256, 512, 1024)
SEED = 20260827


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def xcorr(a, b, L):
    """sum_t a[t] b[t+l] / (n - l) for l = 0..L, by FFT"""
    n = len(a)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    fa = np.fft.rfft(a, m)
    fb = np.fft.rfft(b, m)
    c = np.fft.irfft(np.conj(fa) * fb, m)[:L + 1]
    return c / (n - np.arange(L + 1))


def xcorr_direct(a, b, L):
    return np.array([float(np.dot(a[:len(a) - l], b[l:]) / (len(a) - l)) for l in range(L + 1)])


def invert(S, C):
    L = len(S)
    M = np.empty((L, L))
    idx = np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
    M[:] = C[idx]
    cond = float(np.linalg.cond(M))
    K, *_ = np.linalg.lstsq(M, S, rcond=None)
    return np.concatenate([[0.0], np.cumsum(K)]), cond


def beta_of(G, lo, hi):
    l = np.arange(lo, hi + 1)
    g = G[lo:hi + 1]
    ok = g > 0
    if ok.sum() < 8:
        return float("nan")
    x, y = np.log(l[ok]), np.log(g[ok])
    A = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(-b[1])


def run_inversion(r, eps, L):
    S = xcorr(eps, r, L - 1)
    v = float(np.mean((eps - eps.mean()) ** 2))
    C = xcorr(eps - eps.mean(), eps - eps.mean(), L) / v
    G, cond = invert(S, C)
    return G, cond


def main() -> int:
    rng = np.random.default_rng(SEED)
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per = {}
    fft_check = None
    try:
        for sym in SYMS:
            a = np.array(con.execute(
                "select price,is_buyer_maker from agg_trades where symbol=? "
                "order by ts_ms limit ?", (sym, NROWS)).fetchall(), dtype=np.float64)
            lp = np.log(a[:, 0])
            eps = np.where(a[:, 1] > 0.5, -1.0, 1.0)
            r = np.empty_like(lp)
            r[0] = 0.0
            r[1:] = np.diff(lp) * 1e4
            n = len(r)

            if fft_check is None:
                f = xcorr(eps, r, 20)
                d = xcorr_direct(eps, r, 20)
                fft_check = {"max_abs_diff": float(np.max(np.abs(f - d))),
                             "scale": float(np.max(np.abs(d))),
                             "symbol": sym}

            # ---- TEST 1: can a truncated inversion see a LONG decay?
            adversarial = {}
            for true_beta in (0.10, 0.25, 0.40):
                LONG = 8192
                Gt = np.concatenate([[0.0], (np.arange(1, LONG + 1.0)) ** (-true_beta)])
                Kt = np.diff(Gt)
                r_sim = np.convolve(eps, Kt)[:n] + rng.normal(0.0, float(np.std(r)), n)
                got = {}
                for L in L_SWEEP:
                    G, _ = run_inversion(r_sim, eps, L)
                    got[L] = round(beta_of(G, 4, int(L * 0.75)), 4)
                adversarial["true_{0}".format(true_beta)] = {
                    "true": true_beta, "recovered_by_L": got,
                    "worst_error": round(max(abs(v - true_beta) for v in got.values()
                                             if np.isfinite(v)), 4)}

            # ---- TEST 2: real data, swept in L
            real = {}
            for L in L_SWEEP:
                G, cond = run_inversion(r, eps, L)
                real[L] = {"beta": round(beta_of(G, 4, int(L * 0.75)), 4),
                           "cond": round(cond, 1),
                           "G_at_1": round(float(G[1]), 5),
                           "G_at_L_over_2": round(float(G[L // 2]), 5),
                           "G_ratio_L2_over_4": (round(float(G[L // 2] / G[4]), 3)
                                                 if G[4] != 0 else None)}
            per[sym] = {"adversarial_recovery": adversarial, "real_by_L": real}
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    art = {"study": "C-T35", "lane": "C", "utc": _utc(),
           "threat": ("truncating the Toeplitz system at L means the inversion can only see decay "
                      "inside L trades; C-T34's recovery check built its synthetic propagator on "
                      "the same range it inverted, so it could not detect this by construction"),
           "fft_validation": fft_check, "L_sweep": list(L_SWEEP), "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C35_TRUNCATION_STRESS_V1.json").write_text(json.dumps(art, indent=2), encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    w("FFT vs direct cross-correlation: max abs diff {0:.3e} on a scale of {1:.3e}".format(
        fft_check["max_abs_diff"], fft_check["scale"]))
    w("")
    w("TEST 1 -- can an inversion truncated at L recover a decay that runs to lag 8192?")
    w("%-9s %8s %10s %10s %10s %10s %10s" % ("sym", "true", "L=64", "L=128", "L=256", "L=512",
                                             "L=1024"))
    for s in SYMS:
        for k, v in per[s]["adversarial_recovery"].items():
            g = v["recovered_by_L"]
            w("%-9s %8.2f %10.4f %10.4f %10.4f %10.4f %10.4f" % (
                s, v["true"], g[64], g[128], g[256], g[512], g[1024]))
    w("")
    w("TEST 2 -- real data, beta swept in L")
    w("%-9s %10s %10s %10s %10s %10s" % ("sym", "L=64", "L=128", "L=256", "L=512", "L=1024"))
    for s in SYMS:
        rr = per[s]["real_by_L"]
        w("%-9s %10.4f %10.4f %10.4f %10.4f %10.4f" % (
            s, rr[64]["beta"], rr[128]["beta"], rr[256]["beta"], rr[512]["beta"],
            rr[1024]["beta"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
