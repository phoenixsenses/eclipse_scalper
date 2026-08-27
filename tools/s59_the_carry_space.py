# -*- coding: utf-8 -*-
"""S59/S60 -- the dimension this estate never opened, and the identity that decides it.

THE EMPTY SPACE
---------------
Twelve consecutive studies (A-S49..A-S58) measured the DENOMINATOR: impact, timing risk,
the pot, the ledger, the floor, adverse selection.  Every numerator came from one object --
liquidation events on three symbols.  The estate's entire search is for PRICE PREDICTION,
conducted (A-S57) in a regime where the fee is an order of magnitude larger than the
microstructure it is meant to sit beside.

CARRY is a different object.  Funding is a CASH FLOW, settled every eight hours.  It does
not need to beat a spread and it requires no forecast.  And in Harris's frame it is the
continuous form of the same thing the liquidation feed shows in bursts: the crowded side
paying the other side for a reason that is not profit-seeking.

AND THE DATA IS ON DISK, UNUSED
-------------------------------
    data/funding_history.db
      funding_rates       20,218 rows   3 symbols   2020-01-01 -> 2026-05-12
      futures_klines_1h  161,129 rows   3 symbols   same span
      spot_klines_1h     161,862 rows   3 symbols   same span     <- SPOT, so basis exists
      open_interest_hist   1,500 rows   3 symbols   2026-04-21 -> 05-12

**6.4 YEARS.**  This estate's central constraint is that the sample is exhausted: 201 days
gives t = 0.65 and roughly five years pooled would be needed.  Five years is on the disk,
in a dimension nobody searched.

WHAT THIS STUDY IS, AND IS NOT
------------------------------
It is NOT a hypothesis hunt on a fresh sample.  Doing that would be §200's error committed
somewhere new, and the fresh sample is the one asset that must not be spent carelessly.

It is TWO IDENTITY QUESTIONS, both descriptive, neither a test:

  S59  MAGNITUDE.  How large is funding, unconditionally, against the 4.0 bps cost floor
       A-S57 established?  A cash flow that is smaller than the cost of collecting it is
       not a dimension, it is a rounding error, and that has to be settled first.

  S60  THE NO-ARBITRAGE IDENTITY.  A perpetual's funding exists to pull its price to spot.
       If funding is fully offset by the subsequent move in the basis, the receiver earns
       nothing and there is no transfer -- only an accounting convention.  If it is NOT
       fully offset, a structural transfer exists and 6.4 years is enough to characterise
       it.  This is a regression of one observable on another, not a strategy.
"""

import io
import json
import math
import sqlite3

DB = "file:data/funding_history.db?mode=ro"
OUT = "reports/research/h2_response_shape_v1/S59_THE_CARRY_SPACE_V1.json"
COST_FLOOR_BPS = 4.0        # A-S57: maker round trip, BINANCE_BASE
PER_YEAR = 3 * 365          # funding settles every 8 hours


def load(sym):
    c = sqlite3.connect(DB, uri=True)
    fr = c.execute("SELECT funding_time_ms, funding_rate, mark_price_at_settlement "
                   "FROM funding_rates WHERE symbol=? ORDER BY funding_time_ms",
                   (sym,)).fetchall()
    fut = dict((int(t), float(p)) for t, p in c.execute(
        "SELECT open_time_ms, close FROM futures_klines_1h WHERE symbol=?", (sym,)))
    spt = dict((int(t), float(p)) for t, p in c.execute(
        "SELECT open_time_ms, close FROM spot_klines_1h WHERE symbol=?", (sym,)))
    c.close()
    return fr, fut, spt


def basis_bps(fut, spt, ms):
    """(perp - spot)/spot in bps at the hour containing ms.  Both legs required."""
    h = (ms // 3600000) * 3600000
    f, s = fut.get(h), spt.get(h)
    if f and s and s > 0:
        return 1e4 * (f - s) / s
    return None


def stats(v):
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else float("nan")
    s = sorted(v)
    return {"n": n, "mean": m, "sd": sd, "se": sd / math.sqrt(n),
            "p05": s[int(0.05 * n)], "p50": s[n // 2], "p95": s[int(0.95 * n)]}


def ols(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    if sxx <= 0:
        return None
    b = sxy / sxx
    a = my - b * mx
    res = [yy - (a + b * xx) for xx, yy in zip(x, y)]
    ssr = sum(r * r for r in res)
    sst = sum((yy - my) ** 2 for yy in y)
    se = math.sqrt(ssr / (n - 2) / sxx)
    return {"beta": b, "alpha": a, "se": se, "t": b / se if se else float("nan"),
            "r2": 1 - ssr / sst if sst > 0 else float("nan"), "n": n}


def main():
    res = {}
    print("S59 -- MAGNITUDE.  Is funding larger than the cost of collecting it?")
    print("  cost floor %.1f bps round trip (A-S57, maker, BINANCE_BASE)" % COST_FLOOR_BPS)
    print("  %-9s %8s %11s %11s %11s %13s %12s"
          % ("symbol", "n pays", "mean bps", "sd bps", "p05 / p95", "annualised %", "vs floor"))
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        fr, fut, spt = load(sym)
        rates = [1e4 * float(r[1]) for r in fr]          # funding rate in bps per 8h
        st = stats(rates)
        ann = st["mean"] * PER_YEAR / 100.0              # % per year
        print("  %-9s %8s %11.4f %11.4f %5.2f /%5.2f %13.2f %12.2fx"
              % (sym, format(st["n"], ","), st["mean"], st["sd"], st["p05"], st["p95"],
                 ann, abs(st["mean"]) / COST_FLOOR_BPS))
        res[sym] = {"funding_bps_per_8h": st, "annualised_pct": ann}
    print()
    print("  A single funding payment is a FRACTION of one basis point on average.  One")
    print("  round trip at the floor costs 4.0 bps -- about %d payments' worth."
          % int(COST_FLOOR_BPS / max(1e-9, abs(res['BTCUSDT']['funding_bps_per_8h']['mean']))))
    print("  So carry is not a trade you put on and take off; it is a POSITION you hold,")
    print("  and its unit of account is the year, not the round trip.")

    print()
    print("S60 -- THE IDENTITY.  Is funding offset by the subsequent move in the basis?")
    print("  A perpetual's funding exists to pull its price toward spot.  If the receiver")
    print("  of funding then watches the basis move against them by the same amount, the")
    print("  transfer is an accounting convention and nothing more.")
    print("  regression:  d_basis(t -> t+8h)  on  funding(t)      both in bps")
    print("  full offset would be beta = -1.  no offset, beta = 0.")
    print("  %-9s %9s %11s %10s %9s %11s"
          % ("symbol", "n", "beta", "se", "t", "R2"))
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        fr, fut, spt = load(sym)
        xs, ys, carry = [], [], []
        for (t0, r0, _m0), (t1, _r1, _m1) in zip(fr, fr[1:]):
            b0 = basis_bps(fut, spt, int(t0))
            b1 = basis_bps(fut, spt, int(t1))
            if b0 is None or b1 is None:
                continue
            f = 1e4 * float(r0)
            xs.append(f)
            ys.append(b1 - b0)
            # what a funding RECEIVER earns over the interval: the funding, plus the
            # basis move in their favour (they are short the perp when funding is
            # positive, so a falling basis is a gain)
            carry.append(abs(f) - (b1 - b0) * (1.0 if f > 0 else -1.0))
        o = ols(xs, ys)
        st = stats(carry)
        print("  %-9s %9s %11.4f %10.4f %9.2f %11.4f"
              % (sym, format(o["n"], ","), o["beta"], o["se"], o["t"], o["r2"]))
        res[sym]["offset_regression"] = o
        res[sym]["receiver_net_bps_per_8h"] = st
    print()
    print("  AND WHAT THE RECEIVER ACTUALLY NETS, per 8-hour interval, in bps")
    print("  (funding collected, minus the basis move against them)")
    print("  %-9s %9s %11s %10s %9s %13s"
          % ("symbol", "n", "mean bps", "se", "t", "annualised %"))
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        st = res[sym]["receiver_net_bps_per_8h"]
        t = st["mean"] / st["se"] if st["se"] else float("nan")
        print("  %-9s %9s %11.4f %10.4f %9.2f %13.2f"
              % (sym, format(st["n"], ","), st["mean"], st["se"], t,
                 st["mean"] * PER_YEAR / 100.0))
    print()
    print("  THE t-STATISTICS HERE ARE NOT INFERENCE.  Consecutive 8-hour intervals are")
    print("  not independent and no clustering is applied; they are reported so the")
    print("  magnitude can be read against its own noise, nothing more.  The identity")
    print("  question -- is beta -1 or 0 -- is what this study answers.")

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"study": "S59_S60_CARRY", "cost_floor_bps": COST_FLOOR_BPS,
         "per_year": PER_YEAR, "symbols": res}, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()


def s61():
    """S61 -- THE QUESTION THAT DECIDES WHETHER S60's NUMBER IS ATTAINABLE.

    Receiving funding means being SHORT the perpetual.  Over 2020-2026 these three went
    up a great deal.  A number like "15% a year to the receiver" is only real if the
    directional exposure is hedged, and hedging it needs the SPOT leg -- which this estate
    does not have in its infrastructure.

    So: what did each leg actually earn?
      naked short perp   = funding collected  +  price P&L of the short
      hedged (cash-carry)= funding collected  +  basis convergence only
    """
    import sqlite3
    print()
    print("S61 -- IS S60's NUMBER ATTAINABLE WITHOUT A SPOT LEG?")
    print("  receiving funding = being SHORT the perpetual.  what did that cost?")
    print("  %-9s %8s %13s %14s %14s %14s"
          % ("symbol", "years", "funding %/yr", "perp move %/yr", "NAKED SHORT", "HEDGED"))
    out = {}
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        fr, fut, spt = load(sym)
        t0, t1 = int(fr[0][0]), int(fr[-1][0])
        yrs = (t1 - t0) / 86400000.0 / 365.25
        # funding collected by a constant short, in log terms
        fund = sum(float(r[1]) for r in fr)
        hs = sorted(fut)
        p0, p1 = fut[hs[0]], fut[hs[-1]]
        perp = math.log(p1 / p0)
        # basis at both ends -- the hedged position keeps only the basis convergence
        b0 = basis_bps(fut, spt, hs[0])
        b1 = basis_bps(fut, spt, hs[-1])
        naked = fund - perp                       # short: funding in, price move against
        hedged = fund - ((b1 - b0) / 1e4 if (b0 is not None and b1 is not None) else 0.0)
        print("  %-9s %8.2f %12.2f%% %13.2f%% %13.2f%% %13.2f%%"
              % (sym, yrs, 100 * fund / yrs, 100 * perp / yrs,
                 100 * naked / yrs, 100 * hedged / yrs))
        out[sym] = {"years": yrs, "funding_total": fund, "perp_logret": perp,
                    "naked_short_per_year_pct": 100 * naked / yrs,
                    "hedged_per_year_pct": 100 * hedged / yrs,
                    "basis_start_bps": b0, "basis_end_bps": b1}
    print()
    print("  NAKED SHORT is what this estate could actually put on today.  HEDGED needs a")
    print("  spot leg -- capital on both sides, custody, and infrastructure that does not")
    print("  exist here.  The gap between the two columns IS the reason the funding")
    print("  transfer is paid at all: it compensates for carrying the direction.")
    return out
