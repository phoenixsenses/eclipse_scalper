# -*- coding: utf-8 -*-
"""S54 -- the shape of the price path around a forced liquidation.

WHAT S53 LEFT OPEN
------------------
A-S53 established that the 123.7 bps continuation cannot be the liquidation's own impact --
the largest liquidation ever recorded is 160x to 558x too small under the square-root law.
It left three explanations unseparated: SELECTION, COMMON CAUSE, and WINDOW CONTAMINATION,
and assigned them to whoever owns §311/§315's construction.

**One of the three is testable without touching that construction.**  If the liquidation is
a marker of a move already in progress, then the price has ALREADY moved in the direction
of the forced flow before the liquidation prints.  That is measurable directly from
one-minute bars and liquidation timestamps.

THE CORPUS PICTURE
------------------
Bouchaud TQP Figure 12.1, "Average shape of the impact path":

    "Over the course of its execution, a buy metaorder pushes the price up, until it
     reaches a peak impact.  Upon completion, the buying pressure stops and the price
     reverts abruptly.  Some impact is however still observable long after the metaorder
     execution is completed, and sometimes persists permanently."

That is the shape of a CAUSE: flat before, rising during, reverting after.  A MARKER has a
different shape: the move is already there when the event prints.  The two are
distinguishable by looking at what happens BEFORE t=0, which the impact literature
routinely plots and this estate has never plotted for its own liquidations.

ORIENTATION
-----------
`side` is the side of the liquidation ORDER.  A long being liquidated generates a forced
SELL.  So the path is oriented by the direction of the forced FLOW: +1 for BUY, -1 for
SELL.  An oriented path that is positive before t=0 means the price had already moved the
way the forced flow will push it -- the event is downstream of the move, not upstream.

WHAT THIS IS NOT
----------------
Not a test and not a trading rule.  Events overlap heavily (one every ~2 minutes on BTC),
so the path is a conditional MEAN reported descriptively.  Day-clustered standard errors
are given for the two headline numbers only, and no p-value is computed anywhere.
"""

import io
import json
import math
import sqlite3

LIQ = "file:data/microstructure_02.db?mode=ro"
PANEL = "file:data/xsec_klines_ext.db?mode=ro"
CUT = 1787270400000
OUT = "reports/research/h2_response_shape_v1/S54_THE_PATH_V1.json"

SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
K = 60                                   # minutes either side
MIN_MS = 60000


def closes(sym, lo, hi):
    c = sqlite3.connect(PANEL, uri=True)
    rows = c.execute(
        "SELECT open_time/%d, close FROM klines WHERE symbol=? AND open_time>=? "
        "AND open_time<?" % MIN_MS, (sym, lo, hi)).fetchall()
    c.close()
    return {int(b): float(p) for b, p in rows if p and p > 0}


def events(sym, lo, hi):
    c = sqlite3.connect(LIQ, uri=True)
    rows = c.execute(
        "SELECT ts_ms, side, notional FROM liquidations WHERE symbol=? AND ts_ms>=? "
        "AND ts_ms<? AND notional>0", (sym, lo, hi)).fetchall()
    c.close()
    return [(int(t // MIN_MS), (1.0 if s == "BUY" else -1.0), float(n)) for t, s, n in rows]


def path(px, evs):
    """Oriented cumulative log return relative to t=0, averaged over events.

    An event contributes to offset k only if BOTH its own bar and bar k are present, so a
    gap drops that offset rather than silently carrying the price across it."""
    acc = [0.0] * (2 * K + 1)
    cnt = [0] * (2 * K + 1)
    per_day = {}
    used = 0
    for b, sgn, _n in evs:
        p0 = px.get(b)
        if not p0:
            continue
        used += 1
        d = b // 1440
        for i, k in enumerate(range(-K, K + 1)):
            p = px.get(b + k)
            if not p:
                continue
            v = sgn * math.log(p / p0) * 1e4
            acc[i] += v
            cnt[i] += 1
            if k in (-K, K):
                per_day.setdefault((d, k), []).append(v)
    mean = [(acc[i] / cnt[i]) if cnt[i] else float("nan") for i in range(2 * K + 1)]
    return mean, cnt, per_day, used


def day_se(per_day, k):
    """Cluster by calendar day -- the coarsest unit these overlapping events allow."""
    vals = [sum(v) / len(v) for (d, kk), v in per_day.items() if kk == k]
    n = len(vals)
    if n < 3:
        return float("nan"), n
    m = sum(vals) / n
    var = sum((x - m) ** 2 for x in vals) / (n - 1)
    return math.sqrt(var / n), n


def main():
    c = sqlite3.connect(LIQ, uri=True)
    llo, lhi = c.execute("SELECT MIN(ts_ms), MAX(ts_ms) FROM liquidations WHERE ts_ms<?",
                         (CUT,)).fetchone()
    c.close()
    p = sqlite3.connect(PANEL, uri=True)
    plo, phi = p.execute("SELECT MIN(open_time), MAX(open_time) FROM klines WHERE symbol=?",
                         ("BTCUSDT",)).fetchone()
    p.close()
    lo, hi = max(llo, plo), min(lhi, phi)

    print("THE SHAPE OF THE PRICE PATH AROUND A FORCED LIQUIDATION")
    print("  oriented by the direction of the forced FLOW (+1 BUY, -1 SELL)")
    print("  positive BEFORE t=0 means the price had ALREADY moved the way the forced")
    print("  flow will push it -- the event is downstream of the move.")
    print("  common window: liquidations and prices both present.")

    res = {}
    for s in SYMS:
        px = closes(s, lo, hi)
        ev = events(s, lo, hi)
        if not px or not ev:
            continue
        big = sorted(ev, key=lambda x: -x[2])[:max(1, len(ev) // 100)]
        for tag, E in (("all", ev), ("p99 largest", big)):
            mean, cnt, pd, used = path(px, E)
            pre = mean[0]                       # t-60 relative to t0, oriented
            post = mean[-1]                     # t+60
            se_pre, nd = day_se(pd, -K)
            se_post, _ = day_se(pd, K)
            print()
            print("  %-9s %-12s events used %s of %s   days %d"
                  % (s, tag, format(used, ","), format(len(E), ","), nd))
            print("    %-26s %10s %10s" % ("", "bps", "day-clustered SE"))
            print("    %-26s %10.2f %10.2f" % ("t-60 -> t0  (PRE move)", -pre, se_pre))
            print("    %-26s %10.2f %10.2f" % ("t0 -> t+60  (POST move)", post, se_post))
            if abs(post) > 1e-9:
                print("    %-26s %10.2f" % ("PRE / POST", -pre / post))
            print("    path (bps, oriented): " + " ".join(
                "%s%.1f" % ("t%+d=" % k if k in (-60, -30, -10, -1, 0, 1, 10, 30, 60) else "", mean[i])
                for i, k in enumerate(range(-K, K + 1))
                if k in (-60, -30, -10, -1, 0, 1, 10, 30, 60)))
            res.setdefault(s, {})[tag] = {
                "events_used": used, "events_total": len(E), "days": nd,
                "pre_bps": -pre, "post_bps": post, "se_pre": se_pre, "se_post": se_post,
                "path": mean, "counts": cnt}

    print()
    print("READING IT AGAINST TQP FIGURE 12.1")
    print("  A CAUSE looks like: flat before, rising during, partial reversion after.")
    print("  A MARKER looks like: the move is already there when the event prints.")
    print("  The pre/post ratio above is the discriminator, and it needs no reference to")
    print("  §311/§315's construction -- only bars and timestamps.")

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"study": "S54_THE_PATH", "window_ms": [lo, hi], "K_minutes": K,
         "symbols": res}, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
