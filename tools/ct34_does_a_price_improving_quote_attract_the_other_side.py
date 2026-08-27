# -*- coding: utf-8 -*-
"""C-T34 -- DOES A PRICE-IMPROVING QUOTE ATTRACT THE OPPOSITE SIDE?  H-T6 WAS DIRECTION-BLIND.

C-T33 found R(-1) < 0 on all three symbols: just before a buy market order, the price has
gone DOWN.  Sign autocorrelation predicts the opposite -- a buy follows a buy, which pushed
the price up -- so the measured sign is inverted, and that needs a mechanism.

Sec 14.4 states one:

    "Newly posted price-improving limit orders LO1 ATTRACT market orders.  More precisely, LO1
     events rapidly trigger a strong OPPOSITE flow of MO1 orders and, for large-tick stocks,
     of MO0 orders as well."

A price-improving SELL limit order lowers the ask, which lowers the mid, and then draws BUY
market orders.  That is exactly R(-1) < 0 for buys.

H-T6 measured the AL -> AM pair and found essentially nothing on the small-tick symbols:
0.97 / 1.17 (BTC at 0.5 s / 1 s), 0.87 / 1.06 (ETH), with only SOL elevated at 1.40 / 1.49.
But H-T6's classes were DIRECTION-BLIND: AL pooled bid-improving and ask-improving events, AM
pooled buys and sells.  The corpus's claim is about the CROSS pairing, and pooling both sides
cancels a cross effect by construction.  So H-T6's null on this cell is not evidence against
the mechanism -- it is a measurement that could not see it.  This run separates the sides.

    AL_ask   ask falls AND bid unchanged, no trade -> spread narrows from the ask side
    AL_bid   bid rises AND ask unchanged, no trade -> spread narrows from the bid side
             (defining these by one side alone also catches the whole book shifting,
              which is not a limit-order event; that error made the first pass void)
    AM_buy   buy market order that moved the mid
    AM_sell  sell market order that moved the mid

PREREGISTERED PREDICTION (Sec 14.4):
    AL_ask -> AM_buy   >   AL_ask -> AM_sell
    AL_bid -> AM_sell  >   AL_bid -> AM_buy
i.e. the CROSS ratio exceeds the SAME ratio.  Equality refutes the mechanism as the
explanation for R(-1) < 0 and leaves that anomaly open.

Null: the second series is shuffled within 60 s bins, which preserves rate and diurnal shape
and destroys only the pairing -- H-T6's own null, reused unchanged.  De-fragmented at 200 ms
(H-T5), and every lag reported is above that merge window.

ESTIMATION.  Ceiling: MECHANISM_CHARACTERISATION.

  python -m tools.ct34_does_a_price_improving_quote_attract_the_other_side --i-have-approval
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
from tools import ht4_book_and_trade_excitation as T4

OUT = "reports/atlas"
DAYS = ("2026-08-07", "2026-08-08", "2026-08-09", "2026-08-10",
        "2026-08-11", "2026-08-12", "2026-08-13")
MERGE_MS = 200
TAUS_S = (0.5, 1.0, 5.0, 30.0)
SHUFFLE_S = 60.0
N_SIM = 12
RNG_SEED = 20260827
HT6_AL_TO_AM = {"BTCUSDT": (0.97, 1.17), "ETHUSDT": (0.87, 1.06), "SOLUSDT": (1.40, 1.49)}

PAIRS = (("AL_ask", "AM_buy", "CROSS"), ("AL_ask", "AM_sell", "SAME"),
         ("AL_bid", "AM_sell", "CROSS"), ("AL_bid", "AM_buy", "SAME"))


def merge(ms, w=MERGE_MS):
    if len(ms) == 0:
        return ms
    return ms[np.concatenate([[True], np.diff(ms) >= w])]


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    rng = np.random.default_rng(RNG_SEED)
    res = {"days": list(DAYS), "merge_ms": MERGE_MS, "taus_s": list(TAUS_S),
           "book": "Sec 14.4: LO1 events rapidly trigger a strong OPPOSITE flow of MO1",
           "ht6_direction_blind_AL_to_AM": HT6_AL_TO_AM,
           "prediction": "CROSS > SAME at the SHORT lag (Sec 14.4 says RAPIDLY trigger); the long-lag comparison is not part of the claim",
           "per_symbol": {}, "ceiling": "MECHANISM_CHARACTERISATION"}

    for sym in H2.SYMBOLS:
        agg = {p: {t: [] for t in TAUS_S} for p in PAIRS}
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
            ots, oeps = ts[idx], eps[idx]
            del ts, px, eps, qty

            ib = np.searchsorted(bts, ots, side="left") - 1
            ia = np.searchsorted(bts, ots, side="right")
            ok = (ib >= 0) & (ia < len(bts))
            moved = np.zeros(len(ots), bool)
            moved[ok] = mid[ia[ok]] != mid[ib[ok]]
            AM_buy = ots[ok & moved & (oeps > 0)]
            AM_sell = ots[ok & moved & (oeps < 0)]

            cnt = (np.searchsorted(ots, bts[1:], side="right")
                   - np.searchsorted(ots, bts[:-1], side="right"))
            no_tr = cnt == 0
            da, db = np.diff(ask), np.diff(bid)
            # a price-improving limit order NARROWS the spread from one side only.
            # "ask fell" alone also catches the whole book shifting down, which is not a
            # limit-order event -- that contamination is what made the first pass
            # unreadable and disagree with H-T6.
            AL_ask = bts[1:][no_tr & (da < 0) & (db == 0)]
            AL_bid = bts[1:][no_tr & (db > 0) & (da == 0)]

            S = {"AL_ask": merge(AL_ask), "AL_bid": merge(AL_bid),
                 "AM_buy": merge(AM_buy), "AM_sell": merge(AM_sell)}
            for k, v in S.items():
                counts[k] = counts.get(k, 0) + len(v)
            T0 = float(bts[0])
            Ssec = {k: np.sort((v - T0) / 1000.0) for k, v in S.items()}
            span = float(max((v[-1] for v in Ssec.values() if len(v)), default=1.0) + 1.0)

            for (a, b, kind) in PAIRS:
                A, B = Ssec[a], Ssec[b]
                if len(A) < 200 or len(B) < 200:
                    continue
                obs = T4.pair_count(A, B, TAUS_S)
                acc = np.zeros((N_SIM, len(TAUS_S)))
                for i in range(N_SIM):
                    bb = np.floor(B / SHUFFLE_S)
                    u = np.clip((bb + rng.random(len(B))) * SHUFFLE_S, 0.0, span - 1e-6)
                    u.sort()
                    acc[i] = T4.pair_count(A, u, TAUS_S)
                m = acc.mean(0)
                for j, t in enumerate(TAUS_S):
                    if m[j] > 0:
                        agg[(a, b, kind)][t].append(float(obs[j] / m[j]))
            del bts, bid, ask, mid

        if not counts:
            continue
        out = {"counts": counts, "ratios": {}}
        print("=== %s   " % sym + "  ".join("%s %d" % (k, v) for k, v in counts.items()),
              flush=True)
        for (a, b, kind) in PAIRS:
            r = {str(t): (float(np.mean(agg[(a, b, kind)][t]))
                          if agg[(a, b, kind)][t] else None) for t in TAUS_S}
            out["ratios"]["%s->%s" % (a, b)] = {"kind": kind, "by_tau": r}
            print("    %-9s -> %-8s [%-5s]  " % (a, b, kind) +
                  "  ".join("%ss %s" % (t, "%.3f" % r[str(t)] if r[str(t)] else "n/a")
                            for t in TAUS_S), flush=True)
        cr, sa = {}, {}
        for t in TAUS_S:
            c = [out["ratios"]["%s->%s" % (a, b)]["by_tau"][str(t)]
                 for (a, b, k) in PAIRS if k == "CROSS"]
            s_ = [out["ratios"]["%s->%s" % (a, b)]["by_tau"][str(t)]
                  for (a, b, k) in PAIRS if k == "SAME"]
            c = [v for v in c if v is not None]
            s_ = [v for v in s_ if v is not None]
            if c and s_:
                cr[str(t)] = float(np.mean(c))
                sa[str(t)] = float(np.mean(s_))
        out["CROSS_mean"] = cr
        out["SAME_mean"] = sa
        out["cross_minus_same"] = {t: cr[t] - sa[t] for t in cr}
        out["prediction_holds"] = bool(cr and all(cr[t] > sa[t] for t in cr))
        print("    CROSS mean " + "  ".join("%ss %.3f" % (t, cr[str(t)]) for t in TAUS_S
                                            if str(t) in cr), flush=True)
        print("    SAME  mean " + "  ".join("%ss %.3f" % (t, sa[str(t)]) for t in TAUS_S
                                            if str(t) in sa), flush=True)
        print("    CROSS > SAME at every tau? %s   (H-T6 direction-blind AL->AM was %.2f/%.2f)"
              % (out["prediction_holds"], HT6_AL_TO_AM[sym][0], HT6_AL_TO_AM[sym][1]),
              flush=True)
        res["per_symbol"][sym] = out

    n_ok = sum(1 for v in res["per_symbol"].values() if v.get("prediction_holds"))
    res["summary"] = {"holds_on": n_ok, "of": len(res["per_symbol"])}
    print("SUMMARY  CROSS > SAME on %d of %d symbols" % (n_ok, len(res["per_symbol"])),
          flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "CT34_QUOTE_ATTRACTS_V1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print("written", flush=True)


if __name__ == "__main__":
    main()
