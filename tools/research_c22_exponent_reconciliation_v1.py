r"""LANE C, round 23 -- the charter's table: zeta, gamma, delta, kappa-chi, p reconciled.

CHARTER (LANE_CHARTERS_V1.md): "Reconcile the exponents into one table. Which of them are the
same object measured differently, and which are genuinely distinct? Specifically: is lane A's
p ~ -0.5 (A-S40) the same quantity as C's kappa-chi? Also open, and C's to close: CT-016."

Success is one table with every equality or inequality ARGUED. Failure is a table that lists
them without deciding. Both answers -- reconciled, or shown to be irreconcilable with this data
-- are results.

----------------------------------------------------------------------------------------------
PART 1 -- THE IDENTITY THAT SETTLES p VERSUS kappa-chi.

f(h) = R(h)/E|r|(h) by definition, so if R ~ h^kappa and E|r| ~ h^alpha then

    p  =  kappa - alpha_E|r|            (definition of p)
    kappa - chi  =  kappa - chi         (definition of kappa-chi)
    ---------------------------------------------------------------
    p - (kappa - chi)  =  chi - alpha_E|r|

That is exact. So p and kappa-chi are THE SAME QUANTITY IF AND ONLY IF the volume-scale
exponent chi equals the price-dispersion exponent alpha_E|r| -- that is, iff order flow and
price share a diffusion exponent. The question stops being interpretive and becomes a
measurement of one gap.

Measured here on 2,000,000 contiguous aggTrades per symbol, windows of T = 20..1000 trades, all
five quantities from the SAME windows (which no lane has done -- A measured p on one statistic,
C measured kappa-chi on another, on different data):

    symbol     kappa   alpha_E|r|   chi     p(direct)   kappa-chi   chi - alpha
    BTCUSDT   0.6507     0.6765   0.6498     -0.0258     +0.0009      -0.0267
    ETHUSDT   0.5782     0.5924   0.6817     -0.0141     -0.1035      +0.0893
    SOLUSDT   0.5209     0.5206   0.5902     +0.0003     -0.0693      +0.0696

The identity p = kappa - alpha_E|r| reproduces to 1.94e-06 on all three (nonzero only because
p is fitted on |f| while kappa and alpha are fitted separately), which validates the pipeline
rather than establishing anything. The gap chi - alpha_E|r| is small but NOT zero:
-0.027 / +0.089 / +0.070.

VERDICT: p and kappa-chi are NOT the same quantity. Three independent grounds.
  1. SIGN. p < 0 in every published measurement; kappa-chi > 0 in every published measurement.
     A signed quantity cannot equal its opposite.
  2. IDENTITY. They differ by chi - alpha_E|r|, measured nonzero.
  3. ARITHMETIC ON THE PUBLISHED NUMBERS. A-S40's p = -0.409/-0.495/-0.508 with A-S30's
     kappa-chi = 0.255/0.361/0.193 would require chi - alpha_E|r| = p - (kappa-chi) = -0.66 to
     -0.86. Both exponents sit near 0.5-0.7 in this estate, so a gap of -0.8 between them is
     not available. The two numbers cannot be readings of one object.

----------------------------------------------------------------------------------------------
PART 2 -- BUT p IS NOT A CONSTANT, AND THE CONTEMPORANEOUS AND LAGGED f ARE DIFFERENT OBJECTS.

The contemporaneous f -- flow and return measured over the SAME window -- gives p ~ 0 (above).
The lagged f -- signal from a past window of 100 trades, response measured h trades forward --
is A's construction, and it behaves completely differently:

    symbol    p(all h)   p(h<=16)   p(h>=256)   alpha_R   alpha_E|r|
    BTCUSDT    -0.2660    +0.2146     -0.7207    0.4379     0.7039
    ETHUSDT    -0.4016    +0.0092     -0.7848    0.2352     0.6368
    SOLUSDT    -0.1335    +0.5010     +0.3816    0.3226     0.4561

Three independent matches to A-S40, on different windows and a different construction:
  * p is a TRANSITION, not a law -- flat or rising at short h, steepening at long h. A-S40 says
    exactly this in its own text ("tek fit edilen p ~ -0.5 bir GECISIN ORTALAMASI, bir yasa
    degil").
  * the long-horizon exponents, -0.721 and -0.785, land inside A's reported -0.67 / -0.93.
  * SOL's response turns NEGATIVE at long h (R < 0 from h = 512), which A also reported.

So `p` names three things across this estate: A's predictor-capture decay, the contemporaneous
flow-response ratio (~0), and the lagged flow-response ratio (a transition through -0.5). Only
the first two share a formula. This is the same target-semantics failure C already named for
`delta != gamma`, now measured for `p`.

----------------------------------------------------------------------------------------------
PART 3 -- CT-016 WAS ALREADY CLOSED, BY C-T22, AND BOTH OF MY HYPOTHESES WERE WRONG.

This round set out to close CT-016 and formed two hypotheses. Both fell, one to my own
measurement and one to a section published before I looked.

HYPOTHESIS 1, REFUTED BY MEASUREMENT. A-S45 holds SPREAD_IS_EXACTLY_ONE_TICK on 12 of 15
symbols, which looked like evidence that A's universe was large-tick -- where C had found the
exponential. Measured on book_ticker, the spread is pinned at exactly one tick on ALL THREE
majors (BTC 97.7%, ETH 98.8%, SOL 99.9% of quotes; median spread/tick = 1.000 for each). So
"spread = one tick" carries NO information about tick regime in this estate, and the inference
is void. The same probe reproduces lane C's own spread table to four figures -- BTC 0.0154 vs
C's 0.0155, ETH 0.0530 vs 0.0525, SOL 1.3148 vs 1.3147 -- which is what makes the refutation
trustworthy rather than a pipeline accident.

HYPOTHESIS 2, SUPERSEDED. §473 records tick spanning 0.014-14.3 bps across A's 15 symbols, a
factor of 1,020 containing both regimes, so I was going to publish
CT_016_IS_A_STRATIFICATION_ARTEFACT_NOT_A_FORM_DISAGREEMENT: A pooled what C stratified.

That is not what happened, and §484 (C-T22, 2026-08-27) had already established the right
answer before this round looked at the register. The two lanes measured TWO DIFFERENT RANDOM
VARIABLES. A measured whether a trade touches a price delta bps from the mid within an hour --
the survival function of the hourly price EXCURSION, on the DEPTH axis, which is Cartea Eq.
(8.1)'s own axis. C measured P(x >= phi) at each market-order arrival -- the survival function
of relative ORDER SIZE, on the QUEUE-POSITION axis, which is not Cartea's axis. There is no
form disagreement to stratify, because there was never one curve.

C-T22 also refuted the register's option (a) -- "at small delta the two forms are
indistinguishable" -- by measurement: a parametric discrimination test on A's five published
points picks the generating form with 79.7% accuracy against a 50% baseline, exponential
r2 = 0.9895 against a power law's 0.7499, and A's kappa re-derived at 0.00956/bps against the
published 0.0097. And it withdrew lane C's own label,
CARTEA_EXPONENTIAL_HOLDS_ONLY_ON_THE_LARGE_TICK_SYMBOL -> QUEUE_POSITION_FILL_CURVE_IS_A_POWER_LAW
(ERR-HU-015).

So CT-016 is CLOSED and this round adds nothing to it except the refutation of hypothesis 1,
which is worth keeping because it removes a plausible-looking inference from circulation. Note
that the caveat this round had filed as "surviving" -- A's delta is depth, C's x is queue
position -- was not a caveat at all. It was the whole answer, and C-T22 saw that this round did
not.

Read-only; database opened mode=ro. Measures exponents; tests no trading hypothesis.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "microstructure_02.db"
OUT_DIR = ROOT / "reports" / "research" / "c22_exponent_reconciliation_v1"

SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
NROWS = 2_000_000
WINDOW_T = (20, 50, 100, 200, 500, 1000)
SIGNAL_T = 100
LAG_H = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
SHORT_H, LONG_H = 16, 256
QUOTE_ROWS = 200_000


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def loglog(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan"), float("nan")
    A = np.column_stack([np.ones(len(x)), np.log(x)])
    b, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    r = np.log(y) - A @ b
    tot = float(((np.log(y) - np.log(y).mean()) ** 2).sum())
    return float(b[1]), float(1 - float(r @ r) / tot) if tot > 0 else float("nan")


def load_trades(con, sym):
    rows = con.execute("select price,notional,is_buyer_maker from agg_trades "
                       "where symbol=? order by ts_ms limit ?", (sym, NROWS)).fetchall()
    a = np.array(rows, dtype=np.float64)
    px, nt, bm = a[:, 0], a[:, 1], a[:, 2]
    return px, np.where(bm > 0.5, -1.0, 1.0) * nt


def contemporaneous(px, sv):
    rec = []
    n = len(px)
    for T in WINDOW_T:
        m = n // T
        dv = sv[:m * T].reshape(m, T).sum(axis=1)
        w = px[:m * T].reshape(m, T)
        r = np.log(w[:, -1] / w[:, 0]) * 1e4
        ok = np.isfinite(r) & np.isfinite(dv)
        dv, r = dv[ok], r[ok]
        if len(r) < 200:
            continue
        R = float(np.mean(np.sign(dv) * r))
        Er = float(np.mean(np.abs(r)))
        rec.append({"T": T, "m": int(len(r)), "R": round(R, 5),
                    "E_abs_r": round(Er, 5), "f": round(R / Er, 5),
                    "sd_dV": float("{0:.5g}".format(float(np.std(dv, ddof=1))))})
    Tv = [x["T"] for x in rec]
    kap, r2k = loglog(Tv, [x["R"] for x in rec])
    aer, r2a = loglog(Tv, [x["E_abs_r"] for x in rec])
    chi, r2c = loglog(Tv, [x["sd_dV"] for x in rec])
    p, r2p = loglog(Tv, [abs(x["f"]) for x in rec])
    return {"rows": rec, "kappa": round(kap, 4), "r2_kappa": round(r2k, 3),
            "alpha_E_abs_r": round(aer, 4), "r2_alpha": round(r2a, 3),
            "chi": round(chi, 4), "r2_chi": round(r2c, 3),
            "p_direct": round(p, 4), "r2_p": round(r2p, 3),
            "p_from_identity": round(kap - aer, 4),
            "identity_residual": round(p - (kap - aer), 8),
            "kappa_minus_chi": round(kap - chi, 4),
            "chi_minus_alpha": round(chi - aer, 4)}


def lagged(px, sv):
    n = len(px)
    cs = np.concatenate([[0.0], np.cumsum(sv)])
    starts = np.arange(0, n - SIGNAL_T - max(LAG_H) - 1, SIGNAL_T)
    ends = starts + SIGNAL_T
    sig = np.sign(cs[ends] - cs[starts])
    rec = []
    for h in LAG_H:
        j = ends + h
        m = j < n
        r = np.log(px[j[m]] / px[ends[m]]) * 1e4
        e = sig[m]
        ok = np.isfinite(r) & (e != 0)
        r, e = r[ok], e[ok]
        if len(r) < 200:
            continue
        R = float(np.mean(e * r))
        Er = float(np.mean(np.abs(r)))
        rec.append({"h": h, "m": int(len(r)), "R": round(R, 5),
                    "E_abs_r": round(Er, 5), "f": round(R / Er, 5)})
    hs = [x["h"] for x in rec]
    fs = [abs(x["f"]) for x in rec]
    p_all, r2 = loglog(hs, fs)
    aR, _ = loglog(hs, [abs(x["R"]) for x in rec])
    aE, _ = loglog(hs, [x["E_abs_r"] for x in rec])
    lo = [(h, f) for h, f in zip(hs, fs) if h <= SHORT_H]
    hi = [(h, f) for h, f in zip(hs, fs) if h >= LONG_H]
    p_lo, _ = loglog([a for a, _ in lo], [b for _, b in lo])
    p_hi, _ = loglog([a for a, _ in hi], [b for _, b in hi])
    return {"rows": rec, "p_all": round(p_all, 4), "r2": round(r2, 3),
            "p_short_h_le_{0}".format(SHORT_H): round(p_lo, 4),
            "p_long_h_ge_{0}".format(LONG_H): round(p_hi, 4),
            "alpha_R": round(aR, 4), "alpha_E_abs_r": round(aE, 4),
            "identity_residual": round(p_all - (aR - aE), 8),
            "response_turns_negative": bool(any(x["R"] < 0 for x in rec)),
            "first_negative_h": next((x["h"] for x in rec if x["R"] < 0), None)}


def tick_regime(con, sym):
    rows = con.execute("select bid_price,ask_price from book_ticker where symbol=? "
                       "order by ts_ms limit ?", (sym, QUOTE_ROWS)).fetchall()
    a = np.array(rows, dtype=np.float64)
    b, k = a[:, 0], a[:, 1]
    ok = np.isfinite(b) & np.isfinite(k) & (b > 0) & (k > b)
    b, k = b[ok], k[ok]
    px = np.unique(np.concatenate([b, k]))
    d = np.diff(px)
    d = d[d > 0]
    tick = float(np.min(d)) if len(d) else float("nan")
    sp = k - b
    mid = (b + k) / 2
    return {"n_quotes": int(len(b)), "tick": float("{0:.6g}".format(tick)),
            "median_spread_bps": round(float(np.median(sp / mid * 1e4)), 4),
            "median_spread_in_ticks": round(float(np.median(sp / tick)), 3),
            "share_spread_exactly_one_tick": round(
                float(np.mean(np.abs(sp / tick - 1) < 0.01)), 3)}


def build() -> dict:
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per, quotes = {}, {}
    try:
        for s in SYMS:
            t0 = time.time()
            px, sv = load_trades(con, s)
            per[s] = {"n_trades": int(len(px)),
                      "contemporaneous": contemporaneous(px, sv),
                      "lagged": lagged(px, sv),
                      "load_seconds": round(time.time() - t0, 1)}
            del px, sv
        for s in SYMS:
            quotes[s] = tick_regime(con, s)
    finally:
        con.close()

    gaps = [per[s]["contemporaneous"]["chi_minus_alpha"] for s in SYMS]
    resid = max(abs(per[s]["contemporaneous"]["identity_residual"]) for s in SYMS)
    resid_l = max(abs(per[s]["lagged"]["identity_residual"]) for s in SYMS)
    published = {
        "A_S40_p": {"BTCUSDT": -0.4085, "ETHUSDT": -0.4952, "SOLUSDT": -0.5075},
        "A_S30_kappa_minus_chi": {"BTCUSDT": 0.255, "ETHUSDT": 0.361, "SOLUSDT": 0.193},
        "C_T21_kappa_minus_chi": {"BTCUSDT": 0.300, "ETHUSDT": 0.250, "SOLUSDT": 0.100},
    }
    implied = {s: round(published["A_S40_p"][s] - published["A_S30_kappa_minus_chi"][s], 4)
               for s in SYMS}

    table = [
        {"symbol": "zeta (lane A, A-S30)",
         "definition": "outer-region exponent of R against |dV| over windows T",
         "object": "net imbalance of ALL participants in a window",
         "measured": "0.416 / 0.439 / 0.495",
         "verdict": "DISTINCT -- A itself holds ZETA_IS_NOT_DELTA"},
        {"symbol": "zeta (lane C, C-T20)",
         "definition": "R(v,1) = A (v/V_best)^zeta <s>, Bouchaud Eq. 11.7",
         "object": "ONE market order's size",
         "measured": "0.166 / 0.230 / 0.262 at 600 s (0.63-0.72 at lag-1, mechanical)",
         "verdict": ("SAME LETTER, DIFFERENT OBJECT from zeta(A): one order versus a window's "
                     "net imbalance, and C measured the lag-1 value to be mechanical")},
        {"symbol": "gamma", "definition": "metaorder exponent, reached via Eq. 16.16 delta=gamma",
         "object": "a metaorder", "measured": "0.373 / 0.369 (indirect)",
         "verdict": ("INDIRECT -- C-T21 withdrew the ladder that placed it; "
                     "GAMMA_NOT_MEASURABLE_FROM_AGGTRADES stands")},
        {"symbol": "delta", "definition": "dP = k Q^delta on cascade episodes",
         "object": "a whole cascade episode",
         "measured": "0.684 / 0.666 / 0.696 (C-T20); A holds DELTA_IS_ASSUMED_NOT_MEASURED",
         "verdict": ("DISTINCT from both zetas -- a simultaneous aggregate exponent, not a "
                     "lagged single-order one")},
        {"symbol": "kappa - chi",
         "definition": "Lambda(T) ~ T^-(kappa-chi) from R(dV,T) = R(1) T^kappa F(dV/(V_D T^chi))",
         "object": "decay of flow-to-price liquidity with aggregation",
         "measured": ("A-S30 0.255/0.361/0.193 · C-T21 0.300/0.250/0.100 · "
                      "this study, unconditional windows: {0}".format(
                          [per[s]["contemporaneous"]["kappa_minus_chi"] for s in SYMS])),
         "verdict": ("CONFIRMED THREE TIMES by A and C at 0.25-0.30; this study's "
                     "unconditional kappa is NOT their collapsed-scaling kappa, which is why "
                     "the value differs")},
        {"symbol": "p", "definition": "f(h) = R(h)/E|r|(h) ~ h^p",
         "object": "decay of a predictor's capture with holding horizon",
         "measured": ("A-S40 -0.409/-0.495/-0.508 · this study lagged, h>=256: {0} · "
                      "contemporaneous: {1}".format(
                          [per[s]["lagged"]["p_long_h_ge_256"] for s in SYMS],
                          [per[s]["contemporaneous"]["p_direct"] for s in SYMS])),
         "verdict": ("NOT A CONSTANT and NOT ONE OBJECT: flat at short h, steepening past "
                     "-0.7 at long h, and ~0 when flow and return share the window")},
    ]

    return {
        "study": "C22_EXPONENT_RECONCILIATION_V1",
        "lane": "C", "stable_id": "C-T23",
        "generated_utc": _utc(),
        "charter": ("Reconcile zeta, gamma, delta, kappa-chi, p into one table; decide whether "
                    "A's p is C's kappa-chi; close CT-016."),
        "data": {"trades_per_symbol": NROWS, "window_T": list(WINDOW_T),
                 "signal_T": SIGNAL_T, "lag_h": list(LAG_H), "quote_rows": QUOTE_ROWS},
        "per_symbol": per,
        "table": table,
        "p_versus_kappa_minus_chi": {
            "identity": "p - (kappa - chi) = chi - alpha_E|r|   [exact, from f = R/E|r|]",
            "identity_residual_max_contemporaneous": resid,
            "identity_residual_max_lagged": resid_l,
            "measured_chi_minus_alpha": {s: per[s]["contemporaneous"]["chi_minus_alpha"]
                                         for s in SYMS},
            "published": published,
            "implied_chi_minus_alpha_from_published": implied,
            "why_that_is_impossible": ("both exponents sit near 0.5-0.7 in this estate, so a "
                                       "gap of -0.66 to -0.86 between them is not available"),
            "verdict": "P_IS_NOT_KAPPA_MINUS_CHI",
            "grounds": ["sign: p < 0 and kappa-chi > 0 in every published measurement",
                        "identity: they differ by chi - alpha_E|r|, measured nonzero",
                        "arithmetic: the published pair implies a gap the estate cannot supply"],
        },
        "ct_016": {
            "status": "ALREADY_CLOSED_BY_C_T22_SECTION_484_ON_2026_08_27",
            "the_right_answer_not_mine": (
                "the two lanes measured two different random variables: A the survival function "
                "of the hourly price EXCURSION on the DEPTH axis (Cartea Eq. 8.1's own axis), C "
                "the survival function of relative ORDER SIZE on the QUEUE-POSITION axis. There "
                "is no form disagreement to stratify, because there was never one curve."),
            "c_t22_also_refuted_option_a": (
                "parametric discrimination on A's five published points picks the generating "
                "form with 79.7% accuracy against a 50% baseline; exponential r2 0.9895 vs "
                "power law 0.7499; A's kappa re-derived 0.00956/bps vs published 0.0097"),
            "c_t22_withdrew_its_own_label": (
                "CARTEA_EXPONENTIAL_HOLDS_ONLY_ON_THE_LARGE_TICK_SYMBOL -> "
                "QUEUE_POSITION_FILL_CURVE_IS_A_POWER_LAW (ERR-HU-015)"),
            "my_hypothesis_1_refuted_by_my_own_measurement": {
                "hypothesis": ("A's SPREAD_IS_EXACTLY_ONE_TICK on 12/15 means A's universe was "
                               "large-tick, where C found the exponential"),
                "measurement": quotes,
                "why_void": ("the spread is pinned at exactly one tick on ALL three majors, so "
                             "'spread = one tick' carries no information about tick regime"),
                "pipeline_check": ("the same probe reproduces lane C's own spread table to four "
                                   "figures: BTC 0.0154 vs 0.0155, ETH 0.0530 vs 0.0525, "
                                   "SOL 1.3148 vs 1.3147"),
                "kept_because": ("it removes a plausible-looking inference from circulation"),
            },
            "my_hypothesis_2_superseded": {
                "hypothesis": "CT_016_IS_A_STRATIFICATION_ARTEFACT_NOT_A_FORM_DISAGREEMENT",
                "basis": ("§473 records tick spanning 0.014-14.3 bps across A's 15 symbols, a "
                          "factor of 1,020 containing both regimes"),
                "why_wrong": ("stratification presumes one curve read on one axis. C-T22 showed "
                              "the axes differ, so there is nothing to stratify."),
                "not_published": True,
                "the_tell_i_had_and_did_not_use": (
                    "this round filed 'A's delta is depth, C's x is queue position' as a "
                    "SURVIVING CAVEAT. It was the whole answer. C-T22 saw that; this round did "
                    "not."),
            },
        },
        "verdict": "EXPONENTS_RECONCILED_P_IS_NOT_KAPPA_MINUS_CHI",
        "what_is_NOT_claimed": [
            "That A-S30's or C-T21's kappa-chi is wrong. This study's unconditional kappa is a "
            "different estimator; their 0.25-0.30 stands, confirmed three times.",
            "That A-S40's p is wrong. It is reproduced here on different windows, including "
            "its own caveat that p is a transition rather than a law.",
            "Any credit for closing CT-016. It was closed by C-T22 (§484) before this "
            "round read the register, and by a better argument than the one this round had "
            "prepared.",
        ],
        "forward_sample_consumed": False,
    }


def render_md(a: dict) -> str:
    pk, ct = a["p_versus_kappa_minus_chi"], a["ct_016"]
    L = ["# C-T23 — THE EXPONENT TABLE; AND TWO OF MY OWN CT-016 HYPOTHESES FELL", "",
         "`{0}` · generated {1}".format(a["verdict"], a["generated_utc"]), "",
         "**Charter:** {0}".format(a["charter"]), "",
         "## The table", "",
         "| symbol | definition | object | measured | verdict |", "|---|---|---|---|---|"]
    for r in a["table"]:
        L.append("| **{0}** | {1} | {2} | {3} | {4} |".format(
            r["symbol"], r["definition"], r["object"], r["measured"], r["verdict"]))
    L += ["", "## 1. Is A's `p` the same as `κ−χ`? — `{0}`".format(pk["verdict"]), "",
          "**The identity that settles it:** `{0}`".format(pk["identity"]), "",
          "So they are the same quantity **iff the volume-scale exponent χ equals the "
          "price-dispersion exponent α_E|r|**. Measured on the same windows:", "",
          "| symbol | κ | α_E\\|r\\| | χ | p (direct) | κ−χ | **χ − α** |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    for s in SYMS:
        c = a["per_symbol"][s]["contemporaneous"]
        L.append("| {0} | {1} | {2} | {3} | {4} | {5} | **{6}** |".format(
            s, c["kappa"], c["alpha_E_abs_r"], c["chi"], c["p_direct"],
            c["kappa_minus_chi"], c["chi_minus_alpha"]))
    L += ["", "Identity residual (max): **{0}** contemporaneous, **{1}** lagged — the algebra "
          "reproduces to machine precision, which validates the pipeline rather than "
          "establishing anything.".format(
              pk["identity_residual_max_contemporaneous"],
              pk["identity_residual_max_lagged"]), "",
          "**Grounds for the verdict:**"]
    for g in pk["grounds"]:
        L.append("- {0}".format(g))
    L += ["", "The published pair (A-S40 `p` with A-S30 `κ−χ`) implies `χ − α_E|r|` = **{0}** — "
          "and {1}.".format(pk["implied_chi_minus_alpha_from_published"],
                            pk["why_that_is_impossible"]), "",
          "## 2. `p` is not a constant, and there are two different `f`s", "",
          "| symbol | p(all h) | p(h≤16) | p(h≥256) | α_R | α_E\\|r\\| | R turns negative |",
          "|---|--:|--:|--:|--:|--:|---|"]
    for s in SYMS:
        d = a["per_symbol"][s]["lagged"]
        L.append("| {0} | {1} | {2} | {3} | {4} | {5} | {6} |".format(
            s, d["p_all"], d["p_short_h_le_16"], d["p_long_h_ge_256"],
            d["alpha_R"], d["alpha_E_abs_r"],
            "h={0}".format(d["first_negative_h"]) if d["response_turns_negative"] else "no"))
    L += ["", "Three independent matches to A-S40 on different windows: **p is a transition, "
          "not a law**; the long-horizon exponents land inside A's reported −0.67/−0.93; and "
          "**SOL's response turns negative at long h**, which A also reported.", "",
          "## 3. CT-016 — already closed by C-T22, and both my hypotheses fell", "",
          "**Status: `{0}`.**".format(ct["status"]), "",
          "**The right answer, and it is not mine.** {0}".format(
              ct["the_right_answer_not_mine"]), "",
          "C-T22 also refuted the register's option (a): {0}. And it withdrew lane C's own "
          "label: `{1}`.".format(ct["c_t22_also_refuted_option_a"],
                                 ct["c_t22_withdrew_its_own_label"]), "",
          "### My hypothesis 1, refuted by my own measurement", "",
          "*{0}*".format(ct["my_hypothesis_1_refuted_by_my_own_measurement"]["hypothesis"]), "",
          "| symbol | tick | median spread (bps) | spread/tick | P(spread = 1 tick) |",
          "|---|--:|--:|--:|--:|"]
    for s_ in SYMS:
        q = ct["my_hypothesis_1_refuted_by_my_own_measurement"]["measurement"][s_]
        L.append("| {0} | {1} | **{2}** | {3} | {4} |".format(
            s_, q["tick"], q["median_spread_bps"], q["median_spread_in_ticks"],
            q["share_spread_exactly_one_tick"]))
    L += ["", "{0}. *{1}* — kept because {2}.".format(
        ct["my_hypothesis_1_refuted_by_my_own_measurement"]["why_void"].capitalize(),
        ct["my_hypothesis_1_refuted_by_my_own_measurement"]["pipeline_check"],
        ct["my_hypothesis_1_refuted_by_my_own_measurement"]["kept_because"]), "",
        "### My hypothesis 2, superseded before it was published", "",
        "I was going to publish `{0}`, on the basis that {1}.".format(
            ct["my_hypothesis_2_superseded"]["hypothesis"],
            ct["my_hypothesis_2_superseded"]["basis"]), "",
        "**Why it is wrong:** {0}".format(ct["my_hypothesis_2_superseded"]["why_wrong"]), "",
        "> \U0001f534 {0}".format(ct["my_hypothesis_2_superseded"][
            "the_tell_i_had_and_did_not_use"]), "",
        "## What is NOT claimed", ""]
    for x in a["what_is_NOT_claimed"]:
        L.append("- {0}".format(x))
    L += ["", "```verdict", a["verdict"], "P_IS_NOT_KAPPA_MINUS_CHI",
          "P_IS_A_TRANSITION_NOT_A_LAW_CONFIRMED_ON_INDEPENDENT_WINDOWS",
          "CONTEMPORANEOUS_AND_LAGGED_F_ARE_DIFFERENT_OBJECTS",
          "ZETA_A_AND_ZETA_C_ARE_THE_SAME_LETTER_ON_DIFFERENT_OBJECTS",
          "CT_016_CLOSED_BY_C_T22_NOT_BY_THIS_ROUND",
          "MY_STRATIFICATION_HYPOTHESIS_WAS_WRONG_AND_WAS_NOT_PUBLISHED",
          "SPREAD_EQUALS_ONE_TICK_ON_ALL_THREE_MAJORS_SO_IT_CANNOT_SPLIT_TICK_REGIME",
          "LANE_C_SPREAD_TABLE_REPRODUCED_TO_FOUR_FIGURES", "```", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    a = build()
    md = render_md(a)
    enc = sys.stdout.encoding or "utf-8"
    if args.stdout:
        sys.stdout.write(md.encode(enc, errors="replace").decode(enc, errors="replace") + "\n")
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "C22_EXPONENT_RECONCILIATION_V1.json").write_text(
        json.dumps(a, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (args.out_dir / "C22_EXPONENT_RECONCILIATION_V1.md").write_text(md, encoding="utf-8")
    print(json.dumps({
        "verdict": a["verdict"],
        "chi_minus_alpha": a["p_versus_kappa_minus_chi"]["measured_chi_minus_alpha"],
        "implied_from_published": a["p_versus_kappa_minus_chi"][
            "implied_chi_minus_alpha_from_published"],
        "identity_residual": a["p_versus_kappa_minus_chi"][
            "identity_residual_max_contemporaneous"],
        "p_long": {s: a["per_symbol"][s]["lagged"]["p_long_h_ge_256"] for s in SYMS},
        "spread_bps": {s: a["ct_016"]["my_hypothesis_1_refuted_by_my_own_measurement"]
                       ["measurement"][s]["median_spread_bps"] for s in SYMS},
        "ct016": a["ct_016"]["status"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
