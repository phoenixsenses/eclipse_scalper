# -*- coding: utf-8 -*-
"""S65 -- is A-S54's 40-60 minute lag a property of IMPACT, or of the EPISODE process?

THE QUESTION, ASKED BY LANE C
-----------------------------
C-T44, to A:

    "your size-dependence prediction is confirmed at single-trade scale on BTC and ETH, at a
     clean factor of two.  But the saturation TIMES are two orders apart -- 0.2-0.6 minutes
     for one trade against your 40-60 for an episode -- so the constraint does not bind
     G(l).  If your 40-60 minutes is meant as a property of impact rather than of the
     episode's own duration, that gap is worth one measurement on your side."

And lane D, D-E4, supplies the instrument: same-symbol liquidations arrive as a dead-time
Poisson process with a 58-71 minute mean gap, so P(another inside w) = 1 - exp(-lambda(w-900s)).
A-S64 applied it: at no size floor the 50-minute window is 39-45% contaminated and the
60-minute one is 47.5%.

THE TEST
--------
Split A-S54's events by whether ANOTHER same-symbol liquidation lands inside [t0, t0+60m]:

    CLEAN          nothing else arrives          if the build-up is IMPACT it is still there
    CONTAMINATED   at least one more arrives     if it is the EPISODE process it lives here

WHAT THIS IS NOT, STATED BEFORE THE NUMBERS
-------------------------------------------
The split conditions on the FUTURE arrival process.  It is not conditioning on the outcome,
but arrivals and returns are dependent -- liquidations follow moves -- so a quiet window is
also a low-volatility window, and the arms will differ in |r| for that reason alone.  This
is therefore a DECOMPOSITION, not a causal test, and E|r| is reported per arm so the
selection is visible rather than hidden.

And D-E4's second point applies: the independence unit is not the symbol.  The three
symbols co-fire at 4.5-6.2x chance within +-1 minute, so a window with no SAME-symbol
liquidation can still sit inside a market-wide cascade.  "CLEAN" here means same-symbol
clean, and that is a weaker condition than it sounds.
"""

import io
import json
import math
import sqlite3

LIQ = "file:data/microstructure_02.db?mode=ro"
PANEL = "file:data/xsec_klines_ext.db?mode=ro"
CUT = 1787270400000
OUT = "reports/research/h2_response_shape_v1/S65_IMPACT_OR_EPISODE_V1.json"
SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
K = 60
MIN_MS = 60000
FLOORS = (0.0, 500000.0)          # D-E2: contamination is floor-conditional; both published


def closes(sym, lo, hi):
    c = sqlite3.connect(PANEL, uri=True)
    r = c.execute("SELECT open_time/%d, close FROM klines WHERE symbol=? AND open_time>=? "
                  "AND open_time<?" % MIN_MS, (sym, lo, hi)).fetchall()
    c.close()
    return {int(b): float(p) for b, p in r if p and p > 0}


def events(sym, lo, hi):
    c = sqlite3.connect(LIQ, uri=True)
    r = c.execute("SELECT ts_ms, side, notional FROM liquidations WHERE symbol=? AND ts_ms>=? "
                  "AND ts_ms<? AND notional>0 ORDER BY ts_ms", (sym, lo, hi)).fetchall()
    c.close()
    return [(int(t), (1.0 if s == "BUY" else -1.0), float(n)) for t, s, n in r]


def path_of(px, evs):
    """Oriented cumulative log return from t0, and E|r| at +60, averaged over events."""
    acc = [0.0] * (2 * K + 1)
    cnt = [0] * (2 * K + 1)
    absr = []
    for b, sgn, _n in evs:
        p0 = px.get(b)
        if not p0:
            continue
        for i, k in enumerate(range(-K, K + 1)):
            p = px.get(b + k)
            if not p:
                continue
            acc[i] += sgn * math.log(p / p0) * 1e4
            cnt[i] += 1
            if k == K:
                absr.append(abs(math.log(p / p0)) * 1e4)
    m = [(acc[i] / cnt[i]) if cnt[i] else float("nan") for i in range(2 * K + 1)]
    return m, cnt, (sum(absr) / len(absr) if absr else float("nan")), len(absr)


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

    print("IS THE 40-60 MINUTE LAG IMPACT, OR THE EPISODE PROCESS?  (C-T44's question)")
    print("  CLEAN = no other SAME-SYMBOL liquidation inside [t0, t0+60m]")
    print("  the split conditions on the future ARRIVAL process, not on the outcome --")
    print("  but arrivals and returns are dependent, so E|r| is reported per arm.")
    print("  D-E4: the independence unit is not the symbol; same-symbol clean is weaker")
    print("  than it sounds (the three co-fire at 4.5-6.2x chance within +-1 minute).")

    res = {}
    for floor in FLOORS:
        print()
        print("  ===== SIZE FLOOR $%s =====" % format(int(floor), ","))
        for s in SYMS:
            px = closes(s, lo, hi)
            ev = [e for e in events(s, lo, hi) if e[2] >= floor]
            if not px or len(ev) < 100:
                continue
            ts = [e[0] for e in ev]
            clean, dirty = [], []
            j = 0
            for i, e in enumerate(ev):
                t0 = e[0]
                # is there another same-symbol event in (t0, t0+60m]?
                k = i + 1
                nxt = ts[k] if k < len(ts) else None
                (clean if (nxt is None or nxt - t0 > K * MIN_MS) else dirty).append(
                    (t0 // MIN_MS, e[1], e[2]))
            mc, cc, ac, nc = path_of(px, clean)
            md, cd, ad, nd = path_of(px, dirty)
            share = 100.0 * len(dirty) / max(1, len(ev))
            print()
            print("  %s   events %s   contaminated %.1f%%  (D-E4 predicts %s)"
                  % (s, format(len(ev), ","), share,
                     "47.5%" if floor == 0 else "12.3%"))
            print("    %-14s %8s %9s %9s %9s %9s %11s"
                  % ("arm", "n", "t+10", "t+20", "t+30", "t+60", "E|r| t+60"))
            for lab, m, cnt, a, n in (("CLEAN", mc, cc, ac, nc),
                                      ("CONTAMINATED", md, cd, ad, nd)):
                if n < 30:
                    print("    %-14s %8s  insufficient" % (lab, format(n, ",")))
                    continue
                print("    %-14s %8s %9.2f %9.2f %9.2f %9.2f %11.1f"
                      % (lab, format(n, ","), m[K + 10], m[K + 20], m[K + 30], m[K + 60], a))
            if nc >= 30 and nd >= 30:
                print("    CLEAN builds %+.2f bps from t+10 to t+60; CONTAMINATED %+.2f"
                      % (mc[K + 60] - mc[K + 10], md[K + 60] - md[K + 10]))
                res["%s|%.0f" % (s, floor)] = {
                    "n_clean": nc, "n_dirty": nd, "contaminated_pct": share,
                    "clean_path": mc, "dirty_path": md,
                    "clean_abs_r": ac, "dirty_abs_r": ad,
                    "clean_build_10_60": mc[K + 60] - mc[K + 10],
                    "dirty_build_10_60": md[K + 60] - md[K + 10]}

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"study": "S65_IMPACT_OR_EPISODE", "K_minutes": K, "floors": list(FLOORS),
         "cells": res}, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
