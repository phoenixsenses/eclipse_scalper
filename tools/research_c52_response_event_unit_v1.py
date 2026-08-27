r"""LANE C, round 52 -- re-measure R(l) on the unit C-T51 proved is the right one.

C-T51 established that on the small-tick symbols 10.75% (BTC) and 13.81% (ETH) of market orders
walk the book at median depths of 7 and 3 ticks, so CONSECUTIVE AGGTRADES OF ONE ORDER SHARE A
DIRECTION. Measuring anything per aggTrade there partly measures the inside of one order: the
reversion probability moved from 0.3072 to 0.7214 on BTC when the unit was corrected, and the size
of the correction tracked the walk rate exactly.

C-T29's response function R(l) = <(m_{t+l} - m_t) eps_t> was measured per aggTrade. Its saturation
value -- 0.487 / 0.494 / 0.060 bps -- is the number this lane has quoted as the whole directional
content of one trade, and it is the number the estate's economics rests beside. C-T51 makes it a
debt rather than a result, and `--who response function event unit` returns nothing in either
language, so nobody has paid it.

IT IS ALSO NOW SOMEONE ELSE'S INPUT. The other lane-C session's C-KULLIYAT-T49 read Bouchaud
Eq. (17.3), E[G] ~ T v0 E[theta] ( E[s]/2 + w - R_inf ), and its bracket contains R_inf. Whatever
that bracket is worth, it is worth it at the CORRECT R_inf.

WHAT CHANGES AND WHY IT MIGHT NOT. Two effects pull in opposite directions and neither is obvious
in advance:
  - a walking order contributes several same-signed aggTrades, which INFLATES the measured response
    at small lags because the lag counter advances inside one order;
  - collapsing the walk into one event makes each event larger, which can RAISE the per-event
    response even as the per-aggTrade one falls.
So the direction is not predictable from the walk rate alone, and the prediction stated here is
only that the two differ MOST on BTC and LEAST on SOL, in proportion to the walk rate -- the same
signature that independently confirmed C-T51's diagnosis.

DECLARED THRESHOLD, AND SWEPT: the run definition is consecutive aggTrades with identical ts_ms
AND identical side. The sensitivity of the whole result to that choice is reported by also
collapsing on ts_ms ALONE, which merges opposite-side orders in the same millisecond and is the
looser boundary.

SAMPLE BY ARTIFACT PATH: data/microstructure_02.db :: agg_trades, first 2,000,000 rows by ts_ms
per symbol -- the same population as C-T29.
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
LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
SEED = 20260827

CT29_R_INF = {"BTCUSDT": 0.4873, "ETHUSDT": 0.4940, "SOLUSDT": 0.0605}
CT50_WALK_RATE = {"BTCUSDT": 0.10752, "ETHUSDT": 0.13814, "SOLUSDT": 0.00297}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def xcorr(a, b, L):
    n = len(a)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.conj(np.fft.rfft(a, m)) * np.fft.rfft(b, m), m)[:L + 1]
    return c / (n - np.arange(L + 1))


def response(lp, eps, lags):
    """R(l) = <(m_{t+l} - m_t) eps_t> in bps, cumulated from the lagged sign-return correlation"""
    r = np.empty_like(lp)
    r[0] = 0.0
    r[1:] = np.diff(lp) * 1e4
    S = xcorr(eps, r, max(lags))
    cum = np.cumsum(S)
    return {l: float(cum[l - 1]) for l in lags if l - 1 < len(cum)}


def collapse(ts, px, bm, by_side=True):
    """one row per market-order event: its last price and its side"""
    change = np.empty(len(ts), dtype=bool)
    change[0] = True
    if by_side:
        change[1:] = (ts[1:] != ts[:-1]) | (bm[1:] != bm[:-1])
    else:
        change[1:] = ts[1:] != ts[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:] - 1, len(ts) - 1)
    return np.log(px[ends]), np.where(bm[starts] > 0.5, -1.0, 1.0)


def main() -> int:
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per = {}
    try:
        for sym in SYMS:
            a = np.array(con.execute(
                "select ts_ms,price,is_buyer_maker from agg_trades where symbol=? "
                "order by ts_ms limit ?", (sym, NROWS)).fetchall(), dtype=np.float64)
            ts, px, bm = a[:, 0], a[:, 1], a[:, 2]

            lp_raw = np.log(px)
            eps_raw = np.where(bm > 0.5, -1.0, 1.0)
            R_raw = response(lp_raw, eps_raw, LAGS)

            lp_ev, eps_ev = collapse(ts, px, bm, by_side=True)
            R_ev = response(lp_ev, eps_ev, LAGS)

            lp_ms, eps_ms = collapse(ts, px, bm, by_side=False)
            R_ms = response(lp_ms, eps_ms, LAGS)

            def sat(R):
                ks = [k for k in sorted(R) if k >= 256]
                return round(float(np.mean([R[k] for k in ks])), 4) if ks else None

            s_raw, s_ev, s_ms = sat(R_raw), sat(R_ev), sat(R_ms)
            per[sym] = {
                "n_aggtrades": int(len(px)), "n_events": int(len(lp_ev)),
                "walk_rate_C_T50": CT50_WALK_RATE[sym],
                "R_inf_per_aggtrade": s_raw,
                "R_inf_per_event": s_ev,
                "R_inf_per_millisecond_looser": s_ms,
                "C_T29_published": CT29_R_INF[sym],
                "ratio_event_over_aggtrade": (round(s_ev / s_raw, 3) if s_raw else None),
                "shift_bps": (round(s_ev - s_raw, 4) if (s_ev is not None and s_raw is not None)
                              else None),
                "R1_per_aggtrade": round(R_raw.get(1, float("nan")), 5),
                "R1_per_event": round(R_ev.get(1, float("nan")), 5),
                "curve_per_aggtrade": {str(k): round(v, 5) for k, v in R_raw.items()},
                "curve_per_event": {str(k): round(v, 5) for k, v in R_ev.items()},
            }
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    shifts = {s: abs(per[s]["shift_bps"]) for s in SYMS if per[s]["shift_bps"] is not None}
    order_pred = sorted(SYMS, key=lambda s: -CT50_WALK_RATE[s])
    order_obs = sorted(shifts, key=lambda s: -shifts[s])
    art = {"study": "C-T52", "lane": "C", "utc": _utc(),
           "debt_from": "C-T51 proved the aggTrade unit measures inside walking orders",
           "also_an_input_to": ("the other lane-C session's C-KULLIYAT-T49 read Eq. (17.3), whose "
                                "bracket contains R_inf"),
           "sample": ("data/microstructure_02.db :: agg_trades, first 2,000,000 rows by ts_ms "
                      "per symbol"),
           "threshold_declared_and_swept": ("event = consecutive aggTrades with identical ts_ms "
                                            "AND side; looser variant collapses on ts_ms alone"),
           "prediction": ("the two units differ MOST on BTC and LEAST on SOL, in proportion to "
                          "the walk rate"),
           "predicted_order_by_walk_rate": order_pred,
           "observed_order_by_shift": order_obs,
           "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C52_RESPONSE_EVENT_UNIT_V1.json").write_text(json.dumps(art, indent=2),
                                                         encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    w("%-9s %10s %12s %14s %13s %14s %10s %9s" % (
        "sym", "walk %", "n events", "R_inf aggTr", "R_inf EVENT", "R_inf ms-only",
        "ratio", "shift"))
    for s in SYMS:
        p = per[s]
        w("%-9s %9.3f%% %12d %14s %13s %14s %10s %9s" % (
            s, 100 * p["walk_rate_C_T50"], p["n_events"], p["R_inf_per_aggtrade"],
            p["R_inf_per_event"], p["R_inf_per_millisecond_looser"],
            p["ratio_event_over_aggtrade"], p["shift_bps"]))
    w("")
    w("R(1), the immediate impact, per unit:")
    for s in SYMS:
        p = per[s]
        w("   %-9s per aggTrade %-10s per event %s" % (s, p["R1_per_aggtrade"],
                                                       p["R1_per_event"]))
    w("")
    w("predicted order of |shift| by walk rate: {0}".format(order_pred))
    w("observed  order of |shift|            : {0}".format(order_obs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
