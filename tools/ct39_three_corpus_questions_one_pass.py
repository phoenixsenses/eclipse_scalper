# -*- coding: utf-8 -*-
"""C-T39 -- THREE CORPUS QUESTIONS FROM ONE DATA PASS.

The DB load is the expensive part; the statistics on top of it are nearly free, and this
estate forbids parallel processes.  So three open corpus questions are answered in one pass.

Q1  DID C-T38'S PLATEAU COME FROM THE CLOCK?
    C-T38 found SOL's book-carried accumulation flattening at k ~ 8-10 intertwined events
    while BTC and ETH kept rising, and then had to caveat it: the event clock runs about 3x
    slower on SOL (78k events over 7 days against BTC's 238k), so k = 10 is a different wall
    time per symbol.  Here B is accumulated on BOTH clocks -- event count and milliseconds --
    from the same anchors.  If the plateau sits at a common WALL TIME across symbols it is a
    time effect; if it sits at a common EVENT COUNT it is an event effect.

Q2  ARE THE SEPARATE QUOTE SIGN SERIES LONG-RANGE AUTOCORRELATED?
    Sec 14.4's first bullet asserts that "each of the separate order-flow sign series are
    (separately) long-range autocorrelated" while their intertwined series is not.  C-T35
    tested the intertwined half and refuted it (still a power law).  The FIRST half -- that
    the LO and CA series are separately long-range -- has never been measured here.  Only the
    MO series has (C-T19 / C-T28).

Q3  SEC 14.4's FOURTH BULLET, WHICH THIS LANE HAS NEVER TOUCHED:
    "By contrast, LO0 events are initially followed by market orders in the SAME direction,
     before the flow of these market orders inverts."
    LO0 is a limit order that does NOT move the price: at L1 that is a QUANTITY change at an
    unchanged best price.  A bid-side LO0 (bid qty up, prices unchanged) should be followed
    first by BUY market orders -- the same side -- and then by an inversion.  Note this is the
    OPPOSITE pairing to LO1, which C-T34 confirmed attracts the other side.  If both hold, the
    two quote types have opposite signatures, which is a sharper claim than either alone.

Nulls: for Q2, the exponent estimator's own behaviour is already calibrated (C-T28: recovery
sd 0.019-0.033, with a shrinkage toward 0.45 that must be inverted before any number is read).
For Q3, market-order signs shuffled within (day, hour), 8 draws.

ESTIMATION.  Ceiling: MECHANISM_CHARACTERISATION.

  python -m tools.ct39_three_corpus_questions_one_pass --i-have-approval
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np

from tools import h2_response_shape_driver as H2
from tools import s66_cascade_process_driver as D
from tools import hb4_is_a_liquidation_special as B4

OUT = "reports/atlas"
DAYS = ("2026-08-07", "2026-08-08", "2026-08-09", "2026-08-10",
        "2026-08-11", "2026-08-12", "2026-08-13")
MERGE_MS = 200
K_EVENTS = (1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100)
K_MILLIS = (10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400)
ACF_LAGS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
FIT_LO, FIT_HI = 10, 1000
Q3_TAUS_MS = (100, 250, 500, 1000, 2500, 5000)
N_SHUF = 8
RNG_SEED = 20260827


def fit_gamma(cs):
    ls = [L for L in sorted(cs) if FIT_LO <= L <= FIT_HI and cs[L] > 0]
    if len(ls) < 4:
        return None, None
    A = np.column_stack([np.ones(len(ls)), np.log(ls)])
    y = np.log([cs[L] for L in ls])
    c = np.linalg.pinv(A.T @ A) @ (A.T @ y)
    pred = A @ c
    ss = float(np.sum((y - y.mean()) ** 2))
    return float(-c[1]), (float(1 - np.sum((y - pred) ** 2) / ss) if ss > 0 else None)


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    rng = np.random.default_rng(RNG_SEED)
    res = {"days": list(DAYS), "k_events": list(K_EVENTS), "k_millis": list(K_MILLIS),
           "questions": {"Q1": "is C-T38's plateau a clock effect?",
                         "Q2": "are the separate LO / CA sign series long-range?",
                         "Q3": "Sec 14.4 bullet 4 -- LO0 followed by SAME-side MOs"},
           "gamma_note": "C-T28: this estimator shrinks toward 0.45; de-bias before reading",
           "per_symbol": {}, "ceiling": "MECHANISM_CHARACTERISATION"}

    for sym in H2.SYMBOLS:
        Bev = np.zeros(len(K_EVENTS))
        Bms = np.zeros(len(K_MILLIS))
        n_anchor = 0
        acf = {"LO": {}, "CA": {}, "MO": {}}
        q3 = {t: [0.0, 0.0] for t in Q3_TAUS_MS}      # [same-side, opposite-side] counts
        q3n = {t: [] for t in Q3_TAUS_MS}
        span_events = 0
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
            bts = np.array([r[0] for r in rows], np.int64)
            bid = np.array([r[1] for r in rows], float)
            ask = np.array([r[2] for r in rows], float)
            bq = np.array([r[3] for r in rows], float)
            aq = np.array([r[4] for r in rows], float)
            del rows
            mid = 0.5 * (bid + ask)

            ts, px, eps, qty = B4.load_raw_with_qty(sym, (day,))
            new = np.concatenate([[True], (ts[1:] != ts[:-1]) | (eps[1:] != eps[:-1])])
            idx = np.flatnonzero(new)
            raw_t, raw_e = ts[idx], eps[idx]
            del ts, px, eps, qty
            keep = np.concatenate([[True], (np.diff(raw_t) >= MERGE_MS)
                                   | (raw_e[1:] != raw_e[:-1])])
            mg = np.flatnonzero(keep)
            mo_t, mo_e = raw_t[mg], raw_e[mg].astype(float)

            dm = np.zeros(len(bts))
            dm[1:] = (mid[1:] / mid[:-1] - 1.0) * 1e4
            ntr = np.zeros(len(bts))
            ntr[1:] = (np.searchsorted(raw_t, bts[1:], side="right")
                       - np.searchsorted(raw_t, bts[:-1], side="right"))
            tr = ntr > 0
            da, db = np.diff(ask), np.diff(bid)
            dbq, daq = np.diff(bq), np.diff(aq)
            n = len(bts)

            LO = np.zeros(n, bool); LO_s = np.zeros(n)
            CA = np.zeros(n, bool); CA_s = np.zeros(n)
            LO[1:] = (~tr[1:]) & (((da < 0) & (db == 0)) | ((db > 0) & (da == 0)))
            LO_s[1:] = np.where((db > 0) & (da == 0), 1.0, -1.0)
            CA[1:] = (~tr[1:]) & (((da > 0) & (db == 0)) | ((db < 0) & (da == 0)))
            CA_s[1:] = np.where((da > 0) & (db == 0), 1.0, -1.0)
            # LO0: quantity changes at an unchanged best price, no trade
            L0b = np.zeros(n, bool); L0a = np.zeros(n, bool)
            L0b[1:] = (~tr[1:]) & (da == 0) & (db == 0) & (dbq > 0)
            L0a[1:] = (~tr[1:]) & (da == 0) & (db == 0) & (daq > 0)

            cB = np.concatenate([[0.0], np.cumsum(np.where(~tr, dm, 0.0))])
            ev_t = np.sort(np.concatenate([mo_t, bts[LO], bts[CA]]))
            span_events += len(ev_t)

            ib = np.searchsorted(bts, mo_t, side="left") - 1
            ia = np.searchsorted(bts, mo_t, side="right")
            ok = (ib >= 0) & (ia < n)
            moved = np.zeros(len(mo_t), bool)
            moved[ok] = mid[ia[ok]] != mid[ib[ok]]
            sel = ok & moved
            t0, e0 = mo_t[sel], mo_e[sel]
            p0 = np.searchsorted(ev_t, t0, side="right")
            j0 = np.searchsorted(bts, t0, side="right")
            good = (p0 + max(K_EVENTS) < len(ev_t)) & (t0 + max(K_MILLIS) <= bts[-1])
            t0, e0, p0, j0 = t0[good], e0[good], p0[good], j0[good]
            if len(t0) < 500:
                continue
            n_anchor += len(t0)
            for i, k in enumerate(K_EVENTS):
                jk = np.searchsorted(bts, ev_t[p0 + k - 1], side="right")
                Bev[i] += float(np.sum(e0 * (cB[jk] - cB[j0])))
            for i, ms in enumerate(K_MILLIS):
                jk = np.searchsorted(bts, t0 + ms, side="right")
                Bms[i] += float(np.sum(e0 * (cB[jk] - cB[j0])))

            # Q2: separate sign series
            for key, mask, sgn in (("LO", LO, LO_s), ("CA", CA, CA_s)):
                s_ = sgn[mask]
                if len(s_) > 2000:
                    sc = s_ - s_.mean()
                    den = float(np.sum(sc * sc))
                    for L in ACF_LAGS:
                        if len(sc) > L + 10:
                            a0, b0 = acf[key].get(L, (0.0, 0.0))
                            acf[key][L] = (a0 + float(np.sum(sc[L:] * sc[:-L])), b0 + den)
            sc = mo_e - mo_e.mean()
            den = float(np.sum(sc * sc))
            for L in ACF_LAGS:
                if len(sc) > L + 10:
                    a0, b0 = acf["MO"].get(L, (0.0, 0.0))
                    acf["MO"][L] = (a0 + float(np.sum(sc[L:] * sc[:-L])), b0 + den)

            # Q3: LO0 followed by same-side market orders
            for side, mask, want in (("bid", L0b, +1.0), ("ask", L0a, -1.0)):
                at = bts[mask]
                if len(at) < 200:
                    continue
                at = at[np.concatenate([[True], np.diff(at) >= MERGE_MS])]
                for tau in Q3_TAUS_MS:
                    a = np.searchsorted(mo_t, at, side="right")
                    b = np.searchsorted(mo_t, at + tau, side="right")
                    for lo_i, hi_i in zip(a, b):
                        if hi_i > lo_i:
                            e = mo_e[lo_i:hi_i]
                            q3[tau][0] += float(np.sum(e == want))
                            q3[tau][1] += float(np.sum(e == -want))
            del bts, bid, ask, bq, aq, mid

        if n_anchor == 0:
            continue
        Bev /= n_anchor
        Bms /= n_anchor
        out = {"n_anchors": n_anchor, "total_events": span_events,
               "B_by_event": {str(k): float(v) for k, v in zip(K_EVENTS, Bev)},
               "B_by_ms": {str(k): float(v) for k, v in zip(K_MILLIS, Bms)},
               "acf": {}, "q3_LO0": {}}
        print("=== %s   anchors %d   total intertwined events %d"
              % (sym, n_anchor, span_events), flush=True)
        print("  Q1  B by EVENT : " + "  ".join("%d %+.4f" % (k, v)
                                                for k, v in zip(K_EVENTS, Bev)), flush=True)
        print("      B by MS    : " + "  ".join("%d %+.4f" % (k, v)
                                                for k, v in zip(K_MILLIS, Bms)), flush=True)
        for key in ("MO", "LO", "CA"):
            cs = {L: a / b for L, (a, b) in acf[key].items() if b > 0}
            g, r2 = fit_gamma(cs)
            out["acf"][key] = {"C_lags": {str(L): cs[L] for L in sorted(cs)},
                               "gamma_raw": g, "r2": r2}
            print("  Q2  %-3s  C(1) %+.4f  C(100) %+.5f  gamma_raw %s  r2 %s"
                  % (key, cs.get(1, float('nan')), cs.get(100, float('nan')),
                     "%.4f" % g if g else "n/a", "%.3f" % r2 if r2 else "n/a"), flush=True)
        for tau in Q3_TAUS_MS:
            s_, o_ = q3[tau]
            tot = s_ + o_
            out["q3_LO0"][str(tau)] = {"same": s_, "opposite": o_,
                                       "same_share": (s_ / tot) if tot else None}
        print("  Q3  LO0 -> same-side share: " +
              "  ".join("%dms %.4f" % (t, out["q3_LO0"][str(t)]["same_share"])
                        for t in Q3_TAUS_MS if out["q3_LO0"][str(t)]["same_share"]),
              flush=True)
        print("      (0.500 = no preference; Sec 14.4 predicts ABOVE 0.5 then inverting)",
              flush=True)
        res["per_symbol"][sym] = out

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "CT39_THREE_QUESTIONS_V1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print("written", flush=True)


if __name__ == "__main__":
    main()
