r"""LANE C, round 28 part 2 -- what carries the one exponent that survived its null.

Part 1 calibrated C-T23's whole exponent table against a null in which impact is real, the
marginals are real, and only the exponents are trivial. One column survived overwhelmingly:

    chi   observed 0.6498 / 0.6817 / 0.5902     null 0.4997 / 0.5002 / 0.4997 (sd ~0.004)
                                                z    +42.2  / +49.2  / +23.3

chi is the scaling of sd(sum of signed notional over T trades). Under the joint shuffle it lands
on 0.5 to three decimals and the Gaussian control agrees, so the machinery is sound and the
excess is temporal dependence in SIGNED FLOW. That is the only thing in the table that is
unambiguously real.

But "signed flow has memory" is two claims wedged together, and the exponent alone cannot
separate them:

    sv_i = s_i * |v_i|        s_i = sign (buy/sell), |v_i| = size

Either the SIGNS are long-range correlated (Bouchaud's order-splitting story, C(l) ~ l^-gamma),
or the SIZES cluster in time, or both. Two partial shuffles separate them, because each destroys
exactly one and keeps the other in place:

    SIGN-SHUFFLE      permute s_i, keep |v_i| where it is   -> kills sign memory only
    SIZE-SHUFFLE      permute |v_i|, keep s_i where it is   -> kills size memory only

If chi collapses to 0.5 under the sign-shuffle and survives under the size-shuffle, the memory
is in the signs and the long-memory reading is earned. If the reverse, C-T23's chi is a volume-
clustering statistic that was being read as an order-flow one.

WHY THIS MATTERS BEYOND THE TABLE. C-T24 declared gamma NOT IDENTIFIABLE from the direct LMF
fit. For a long-memory sign process with C(l) ~ l^-gamma the partial sums obey
Var ~ T^(2-gamma), i.e. chi = 1 - gamma/2, so

    gamma = 2 (1 - chi)

is an identity -- and chi is measured here at z = 23 to 49 against an exact null. That route to
gamma is only legitimate if the sign-shuffle shows the memory is in the signs, which is what
this script tests rather than assumes.
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
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
SEED = 20260827

# C-T24's direct LMF fit, for the cross-check
CT24_GAMMA_LMF = {"BTCUSDT": 0.7746, "ETHUSDT": 0.7892, "SOLUSDT": 0.2092}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def chi_of(sv):
    """C-T23's chi: log-log slope of sd(sum of signed notional over T trades) against T"""
    n = len(sv)
    Tv, Cv = [], []
    for T in WINDOW_T:
        m = n // T
        if m < 200:
            continue
        dv = sv[:m * T].reshape(m, T).sum(axis=1)
        Tv.append(float(T))
        Cv.append(float(np.std(dv, ddof=1)))
    x, y = np.log(np.asarray(Tv)), np.log(np.asarray(Cv))
    A = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(b[1])


def load(con, sym):
    rows = con.execute("select notional,is_buyer_maker from agg_trades "
                       "where symbol=? order by ts_ms limit ?", (sym, NROWS)).fetchall()
    a = np.array(rows, dtype=np.float64)
    size, bm = a[:, 0], a[:, 1]
    sign = np.where(bm > 0.5, -1.0, 1.0)
    return sign, size


def stat(vals, obs):
    v = np.asarray(vals, float)
    sd = float(v.std(ddof=1))
    return {"mean": round(float(v.mean()), 4), "sd": round(sd, 4),
            "z_of_observed": round((obs - float(v.mean())) / sd, 2) if sd > 0 else None,
            "reps": int(len(v))}


def main() -> int:
    rng = np.random.default_rng(SEED)
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per = {}
    try:
        for sym in SYMS:
            sign, size = load(con, sym)
            n = len(sign)
            obs = chi_of(sign * size)
            sign_sh, size_sh, both_sh = [], [], []
            for _ in range(REPS):
                sign_sh.append(chi_of(sign[rng.permutation(n)] * size))
                size_sh.append(chi_of(sign * size[rng.permutation(n)]))
                both_sh.append(chi_of((sign * size)[rng.permutation(n)]))
            d = {"observed_chi": round(obs, 4),
                 "sign_shuffle": stat(sign_sh, obs),
                 "size_shuffle": stat(size_sh, obs),
                 "joint_shuffle": stat(both_sh, obs)}
            # how much of the excess above 0.5 each shuffle removes
            exc = obs - 0.5
            d["excess_above_half"] = round(exc, 4)
            d["excess_surviving_sign_shuffle"] = round(d["sign_shuffle"]["mean"] - 0.5, 4)
            d["excess_surviving_size_shuffle"] = round(d["size_shuffle"]["mean"] - 0.5, 4)
            d["share_of_excess_from_sign_memory"] = (
                round(1.0 - (d["sign_shuffle"]["mean"] - 0.5) / exc, 3) if exc != 0 else None)
            d["share_of_excess_from_size_memory"] = (
                round(1.0 - (d["size_shuffle"]["mean"] - 0.5) / exc, 3) if exc != 0 else None)
            # the identity route to gamma, reported but NOT asserted until the shuffles decide
            d["gamma_from_identity_2_times_1_minus_chi"] = round(2.0 * (1.0 - obs), 4)
            d["gamma_c_t24_direct_lmf_fit"] = CT24_GAMMA_LMF[sym]
            per[sym] = d
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    art = {"study": "C-T28-part2", "lane": "C", "utc": _utc(), "reps": REPS, "seed": SEED,
           "question": ("chi survived part 1's null at z = 23 to 49. Is its excess above 0.5 "
                        "sign memory (order splitting) or size memory (volume clustering)?"),
           "identity": "for C(l) ~ l^-gamma, Var(sum_T) ~ T^(2-gamma), so chi = 1 - gamma/2",
           "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C28B_FLOW_MEMORY_DECOMPOSITION_V1.json").write_text(
        json.dumps(art, indent=2), encoding="utf-8")
    enc = sys.stdout.encoding or "utf-8"
    sys.stdout.write(json.dumps(per, indent=2).encode(enc, "replace").decode(enc, "replace")
                     + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
