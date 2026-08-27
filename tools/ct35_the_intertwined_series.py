# -*- coding: utf-8 -*-
"""C-T35 -- THE TIT-FOR-TAT DANCE: IS THE INTERTWINED EVENT SERIES SHORT-MEMORY?

The fine-balance route to diffusivity is closed for this estate: C-T29 rejected the composite
{kappa-chi = beta} AND {beta = (1-gamma)/2} on 2 of 3 symbols, and C-T30 could not separate
the legs because the model's "news" term is both dominant (76-83 percent of D) and
autocorrelated.  Sec 14.4 offers a DIFFERENT mechanism for the same thing, and it needs
neither G(l) nor beta:

    "When considering all market order arrivals, limit order arrivals and cancellations
     together (i.e. mixing between the different event types), autocorrelations of eps_t decay
     EXPONENTIALLY, which indicates that this series is short-range autocorrelated.
     Therefore, although each of the separate order-flow sign series are (separately)
     long-range autocorrelated, THEIR INTERTWINED SERIES IS NOT.  This is the mechanism of the
     'tit-for-tat' dance that makes prices diffusive."

So diffusivity may not require a fine balance between beta and gamma at all.  It may require
only that the COMBINED series alternate.  That is testable with quantities this lane has
already calibrated: C-T28 showed the gamma pipeline recovers a known exponent with sd
0.019-0.033 once its shrinkage is inverted.

CONSTRUCTION.  Every event is signed by the direction it pushes the mid, which is what makes
the series "intertwined" in the tit-for-tat sense -- a buy market order pushes up, the refill
that follows pushes down:

    AM_buy   +1     buy market order that moved the mid
    AM_sell  -1
    AL_bid   +1     bid rises, ask unchanged, no trade   (spread narrows from below)
    AL_ask   -1     ask falls, bid unchanged, no trade
    CX_bid   -1     bid falls, ask unchanged, no trade   (the bid queue gives way)
    CX_ask   +1     ask rises, bid unchanged, no trade

PREREGISTERED PREDICTION (Sec 14.4):
    MO-only series      power law, gamma about 0.37   (C-T19 / C-T28, already established)
    INTERTWINED series  EXPONENTIAL decay -- the power-law fit should be clearly worse, and
                        C(l) should reach the noise floor within a few tens of events

DISCRIMINATION IS ITSELF CALIBRATED, the CT-016 way: synthetic series with a KNOWN
exponential decay and with a KNOWN power law are pushed through the identical fit comparison,
and the fraction of times it picks the generating form is reported.  A form test that cannot
tell the two apart at these lengths decides nothing, and this lane has already published one
verdict that had to be withdrawn for exactly that reason.

De-fragmented at 200 ms (H-T5).  ESTIMATION.  Ceiling: MECHANISM_CHARACTERISATION.

  python -m tools.ct35_the_intertwined_series --i-have-approval
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
LAGS = tuple([1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 350, 600, 1000])
FIT_LO, FIT_HI = 5, 200
N_SIM = 120
RNG_SEED = 20260827
GAMMA_MO_CT19 = {"BTCUSDT": 0.407, "ETHUSDT": 0.379, "SOLUSDT": 0.411}


def acf_sums(x, lags):
    xc = x - x.mean()
    den = float(np.sum(xc * xc))
    return {L: (float(np.sum(xc[L:] * xc[:-L])), den) for L in lags if len(xc) > L + 10}


def fits(cs):
    ls = [L for L in sorted(cs) if FIT_LO <= L <= FIT_HI and cs[L] > 0]
    if len(ls) < 5:
        return None
    y = np.log([cs[L] for L in ls])
    Ap = np.column_stack([np.ones(len(ls)), np.log(ls)])
    Ae = np.column_stack([np.ones(len(ls)), np.array(ls, float)])
    cp = np.linalg.pinv(Ap.T @ Ap) @ (Ap.T @ y)
    ce = np.linalg.pinv(Ae.T @ Ae) @ (Ae.T @ y)
    ss = float(np.sum((y - y.mean()) ** 2))
    r2p = 1 - float(np.sum((y - Ap @ cp) ** 2)) / ss if ss > 0 else None
    r2e = 1 - float(np.sum((y - Ae @ ce) ** 2)) / ss if ss > 0 else None
    return {"gamma_power": float(-cp[1]), "r2_power": r2p,
            "decay_exp": float(-ce[1]), "r2_exp": r2e,
            "winner": ("power" if (r2p or 0) > (r2e or 0) else "exponential"),
            "n_lags": len(ls)}


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    rng = np.random.default_rng(RNG_SEED)
    res = {"days": list(DAYS), "lags": list(LAGS), "fit_range": [FIT_LO, FIT_HI],
           "book": "Sec 14.4: the intertwined series is SHORT-range autocorrelated",
           "gamma_MO_only_CT19": GAMMA_MO_CT19,
           "per_symbol": {}, "ceiling": "MECHANISM_CHARACTERISATION"}

    # --- is the form test able to discriminate at these lengths? -------------
    def synth(kind, n, lam=0.05, gam=0.4):
        L = np.array(sorted(LAGS), float)
        c = np.exp(-lam * L) if kind == "exponential" else L ** (-gam)
        c = c * 0.25
        noise = rng.standard_normal(len(L)) / np.sqrt(n)
        return {int(l): float(max(v + z, 1e-9)) for l, v, z in zip(L, c, noise)}

    hits = {"exponential": 0, "power": 0}
    for kind in ("exponential", "power"):
        for _ in range(N_SIM):
            f = fits(synth(kind, 1_500_000))
            if f and f["winner"] == kind:
                hits[kind] += 1
    disc = (hits["exponential"] + hits["power"]) / (2 * N_SIM)
    res["form_test_discrimination"] = {"exponential": hits["exponential"] / N_SIM,
                                       "power": hits["power"] / N_SIM, "overall": disc}
    print("FORM TEST DISCRIMINATION: exponential %.3f  power %.3f  overall %.3f  (chance .5)"
          % (hits["exponential"] / N_SIM, hits["power"] / N_SIM, disc), flush=True)

    for sym in H2.SYMBOLS:
        acc = {"MO_only": {}, "intertwined": {}}
        counts = {}
        for day in DAYS:
            d0 = dt.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
            lo = int(d0.timestamp() * 1000)
            hi = int((d0 + dt.timedelta(days=1)).timestamp() * 1000)
            con = D._con()
            rows = con.execute(
                "SELECT ts_ms,bid_price,ask_price FROM book_ticker WHERE symbol=? "
                "AND ts_ms>=? AND ts_ms<? AND bid_price>0 AND ask_price>0 ORDER BY ts_ms",
                (sym, lo, hi)).fetchall()
            con.close()
            if len(rows) < 10000:
                continue
            bts = np.array([r[0] for r in rows], np.int64)
            bid = np.array([r[1] for r in rows], float)
            ask = np.array([r[2] for r in rows], float)
            del rows
            mid = 0.5 * (bid + ask)
            ts, px, eps, qty = B4.load_raw_with_qty(sym, (day,))
            new = np.concatenate([[True], (ts[1:] != ts[:-1]) | (eps[1:] != eps[:-1])])
            idx = np.flatnonzero(new)
            ots0, oeps0 = ts[idx], eps[idx]
            del ts, px, eps, qty
            keep = np.concatenate([[True], (np.diff(ots0) >= 200)
                                   | (oeps0[1:] != oeps0[:-1])])
            j = np.flatnonzero(keep)
            ots, oeps = ots0[j], oeps0[j].astype(float)

            ib = np.searchsorted(bts, ots, side="left") - 1
            ia = np.searchsorted(bts, ots, side="right")
            ok = (ib >= 0) & (ia < len(bts))
            moved = np.zeros(len(ots), bool)
            moved[ok] = mid[ia[ok]] != mid[ib[ok]]
            mo_t, mo_s = ots[ok & moved], oeps[ok & moved]

            cnt = (np.searchsorted(ots, bts[1:], side="right")
                   - np.searchsorted(ots, bts[:-1], side="right"))
            no_tr = cnt == 0
            da, db = np.diff(ask), np.diff(bid)
            bt = bts[1:]
            ev_t, ev_s = [mo_t], [mo_s]
            for mask, sign, name in ((no_tr & (db > 0) & (da == 0), +1.0, "AL_bid"),
                                     (no_tr & (da < 0) & (db == 0), -1.0, "AL_ask"),
                                     (no_tr & (db < 0) & (da == 0), -1.0, "CX_bid"),
                                     (no_tr & (da > 0) & (db == 0), +1.0, "CX_ask")):
                t = bt[mask]
                if len(t):
                    t = t[np.concatenate([[True], np.diff(t) >= 200])]
                ev_t.append(t)
                ev_s.append(np.full(len(t), sign))
                counts[name] = counts.get(name, 0) + len(t)
            counts["MO"] = counts.get("MO", 0) + len(mo_t)

            T = np.concatenate(ev_t)
            S = np.concatenate(ev_s)
            o = np.argsort(T, kind="stable")
            inter = S[o]

            for key, series in (("MO_only", mo_s), ("intertwined", inter)):
                for L, (nu, de) in acf_sums(series, LAGS).items():
                    a0, b0 = acc[key].get(L, (0.0, 0.0))
                    acc[key][L] = (a0 + nu, b0 + de)
            del bts, bid, ask, mid

        if not acc["MO_only"]:
            continue
        out = {"counts": counts, "series": {}}
        print("=== %s   " % sym + "  ".join("%s %d" % (k, v)
                                            for k, v in sorted(counts.items())), flush=True)
        for key in ("MO_only", "intertwined"):
            cs = {L: nu / de for L, (nu, de) in acc[key].items() if de > 0}
            f = fits(cs)
            out["series"][key] = {"C_lags": {str(L): cs[L] for L in sorted(cs)}, "fit": f}
            print("    %-12s C(1) %+.4f  C(20) %+.5f  C(200) %+.5f" %
                  (key, cs.get(1, float('nan')), cs.get(20, float('nan')),
                   cs.get(200, float('nan'))), flush=True)
            if f:
                print("                 power gamma %.4f r2 %.4f | exponential rate %.5f "
                      "r2 %.4f  => %s"
                      % (f["gamma_power"], f["r2_power"], f["decay_exp"], f["r2_exp"],
                         f["winner"].upper()), flush=True)
        mo, it = out["series"]["MO_only"]["fit"], out["series"]["intertwined"]["fit"]
        out["prediction_holds"] = bool(mo and it and mo["winner"] == "power"
                                       and it["winner"] == "exponential")
        print("    Sec 14.4 prediction (MO power, intertwined exponential): %s"
              % out["prediction_holds"], flush=True)
        res["per_symbol"][sym] = out

    n_ok = sum(1 for v in res["per_symbol"].values() if v.get("prediction_holds"))
    res["summary"] = {"holds_on": n_ok, "of": len(res["per_symbol"]),
                      "readable_only_if_discrimination_high": disc}
    print("SUMMARY  %d of %d   (form test discrimination %.3f)"
          % (n_ok, len(res["per_symbol"]), disc), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "CT35_INTERTWINED_V1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print("written", flush=True)


if __name__ == "__main__":
    main()
