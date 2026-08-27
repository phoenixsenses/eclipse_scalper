# -*- coding: utf-8 -*-
"""S58 -- adverse selection by queue position: the one measurement that could overturn S57.

WHAT RESTS ON THIS
------------------
A-S57 computed the cost floor as "maker fee + impact, everything else under 0.03 bps".  One
of the terms it set aside is ADVERSE SELECTION, and it did not measure it -- it inherited
`~0` from CLAUDE.md §206.  That is the same failure shape A-S53 caught in itself: a number
taken from another study and carried into a conclusion without a check.

And the corpus says the inherited value cannot be a single number.  TQP §21.4:

    "limit orders with HIGH PRIORITY should benefit from short-term mean-reversion (or
     'bid-ask bounce'), while limit orders with LOW PRIORITY will suffer from the adverse
     selection of sweeping market orders"

    on execution, the passive trader's "gain or loss ... comes from the balance between
    ADVERSE SELECTION and the value of s/2 + w"

So adverse selection is a FUNCTION OF QUEUE POSITION, and if it is materially positive for
any reachable position, S57's floor rises and the room it measured (0.91-2.02 bps) shrinks
or vanishes.

THE DATA
--------
`data/book_ticker_2024_v1` -- DL-003 + DL-004.  The only place in this estate with OBSERVED
quotes AND the queue: `best_bid_qty` / `best_ask_qty`, plus matching aggTrades.  15 symbols,
one day, spanning a ~1,000x range in relative tick size.

THE MEASUREMENT
---------------
`is_buyer_maker = true` means the aggressor SOLD into the bid, so resting BUY orders were
filled.  That is exactly the passive-buy fill event of TQP's event (v).

Queue position is not observable per order, but the FRACTION OF THE QUEUE CONSUMED is, and
it bounds which positions were reached:

    ratio = (volume of the market order) / (best_bid_qty just before it)

    ratio small  -> only the front of the queue filled  -> HIGH priority
    ratio >= 1   -> the level was swept                 -> even LOW priority filled

Adverse selection cost for the passive buyer, over horizon D:

    cost_bps = (mid_at_fill - mid_at_fill_plus_D) / mid_at_fill * 1e4

positive = the price moved against the passive buyer = a cost.  TQP predicts this rises
with `ratio`.  That is a directional prediction made before looking.

Consecutive aggTrades sharing a transact_time and side are ONE market order (the same rule
A-S21 used); pricing or sizing them separately would split one order into many.
"""

import io
import json
import math
import sys
import zipfile

DIR = "data/book_ticker_2024_v1"
OUT = "reports/research/h2_response_shape_v1/S58_ADVERSE_SELECTION_V1.json"
GRID_MS = 100
HORIZONS_S = (1, 10, 60)
BUCKETS = ((0.0, 0.05), (0.05, 0.20), (0.20, 0.50), (0.50, 1.00), (1.00, 1e9))
BLAB = ("<5%", "5-20%", "20-50%", "50-100%", "SWEPT >=100%")


def rows(sym, kind):
    """Yield raw BYTE fields.  Decoding 18M lines to str costs more than the arithmetic
    does; float() and int() both accept bytes directly."""
    z = zipfile.ZipFile("%s/%s-%s-2024-03-28.zip" % (DIR, sym, kind))
    with z.open(z.namelist()[0]) as fh:
        fh.readline()
        for line in fh:
            yield line.split(b",")


def run(sym):
    # ---- pass 1: the mid grid, and the book state stream merged with trades
    tr = rows(sym, "aggTrades")
    bk = rows(sym, "bookTicker")

    mid_grid = {}

    ev = []                      # (t_ms, ratio, mid_at_fill)
    pend_t = None
    pend_qty = 0.0
    pend_bidq = pend_mid = None

    b = next(bk, None)
    bt = int(b[5]) if b else None
    cur_slot = None
    last = None                  # the most recent book row, kept UNPARSED

    def flush():
        if pend_t is not None and pend_bidq and pend_bidq > 0:
            ev.append((pend_t, pend_qty / pend_bidq, pend_mid))

    for t in tr:
        tt = int(t[5])
        # advance the book to just before this trade.  Only two things happen per row:
        # the timestamp comparison, and -- when a 100 ms slot closes -- one mid.  The
        # prices themselves are parsed lazily, at slot boundaries and at trades only.
        while b is not None and bt <= tt:
            sl = bt // GRID_MS
            if sl != cur_slot:
                if last is not None:
                    mid_grid[cur_slot] = 0.5 * (float(last[1]) + float(last[3]))
                cur_slot = sl
            last = b
            b = next(bk, None)
            bt = int(b[5]) if b else None
        if last is None:
            continue
        if not t[6].startswith(b"true"):   # is_buyer_maker; bid-side fills only
            if pend_t is not None:
                flush()
                pend_t = None
                pend_qty = 0.0
            continue
        q = float(t[2])
        if pend_t == tt:
            pend_qty += q
        else:
            if pend_t is not None:
                flush()
            pend_t, pend_qty = tt, q
            pend_bidq = float(last[2])
            pend_mid = 0.5 * (float(last[1]) + float(last[3]))
    if pend_t is not None:
        flush()

    # ---- forward mids
    keys = sorted(mid_grid)
    if not keys or not ev:
        return None

    def mid_at(ms):
        k = ms // GRID_MS
        # walk forward at most 50 slots (5 s) to find a quote
        for j in range(k, k + 50):
            if j in mid_grid:
                return mid_grid[j]
        return None

    # tick from the finest non-zero gap between consecutive slot mids -- the per-row
    # estimate was removed with the per-row parse, and this is the same quantity.
    ks = sorted(mid_grid)
    # The naive "smallest non-zero gap" is WRONG on a high-priced instrument: at BTC's
    # ~7e4 the float64 mid carries ~1e-11 of rounding residue, and the minimum picks that
    # residue rather than the tick.  It returned 2.9e-11 = 0.000 bps on BTCUSDT.  Gaps
    # below a relative epsilon are float noise and are dropped.
    eps = 1e-9
    _px = mid_grid[ks[len(ks) // 2]]
    gaps = sorted(g for g in (abs(mid_grid[b_] - mid_grid[a_])
                              for a_, b_ in zip(ks, ks[1:])) if g > _px * eps)
    tick = 2.0 * gaps[0] if gaps else float("nan")
    px = mid_grid[keys[len(keys) // 2]]
    res = {"symbol": sym, "n_fills": len(ev), "tick": tick,
           "rel_tick_bps": 1e4 * tick / px, "mid_ref": px, "buckets": {}}

    for lo, hi, lab in ((a, b_, c) for (a, b_), c in zip(BUCKETS, BLAB)):
        sel = [e for e in ev if lo <= e[1] < hi]
        row = {"n": len(sel)}
        for H in HORIZONS_S:
            vals = []
            for t0, _r, m0 in sel:
                m1 = mid_at(t0 + H * 1000)
                if m1 and m0:
                    vals.append((m0 - m1) / m0 * 1e4)
            if len(vals) >= 30:
                mu = sum(vals) / len(vals)
                sd = math.sqrt(sum((x - mu) ** 2 for x in vals) / (len(vals) - 1))
                row["%ds" % H] = {"mean_bps": mu, "se": sd / math.sqrt(len(vals)),
                                  "n": len(vals)}
        res["buckets"][lab] = row
    return res


def main():
    syms = sys.argv[1:] or ["BTCUSDT"]
    out = {}
    if io is not None:
        try:
            out = json.load(io.open(OUT, encoding="utf-8")).get("symbols", {})
        except Exception:
            out = {}
    for s in syms:
        r = run(s)
        if not r:
            print("  %s: no usable fills" % s)
            continue
        out[s] = r
        print()
        print("  %s   %s bid-side fills   tick %.8g = %.3f bps of mid"
              % (s, format(r["n_fills"], ","), r["tick"], r["rel_tick_bps"]))
        print("    %-14s %9s %11s %11s %11s"
              % ("queue consumed", "n", "1s", "10s", "60s"))
        for lab in BLAB:
            b = r["buckets"].get(lab, {})
            cells = []
            for H in HORIZONS_S:
                c = b.get("%ds" % H)
                cells.append("%7.3f+-%-3.2f" % (c["mean_bps"], c["se"]) if c else "      -    ")
            print("    %-14s %9s %s" % (lab, format(b.get("n", 0), ","), " ".join(cells)))
        print("    positive = the mid moved AGAINST the passive buyer = adverse selection")
        io.open(OUT, "w", encoding="utf-8").write(json.dumps(
            {"study": "S58_ADVERSE_SELECTION", "grid_ms": GRID_MS,
             "horizons_s": list(HORIZONS_S), "symbols": out}, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
