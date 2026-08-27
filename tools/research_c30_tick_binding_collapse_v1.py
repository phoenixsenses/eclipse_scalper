r"""LANE C, round 30 -- is C-T29's cancellation spread a TICK ordering? Tested within symbols.

C-T29 measured how much of the order flow's long memory the price refuses to inherit:

    BTC  57.7%   ETH  74.9%   SOL  102.3%      (cancellation of the memory exponent)

and that ordering is also the tick ordering: SOL is the large-tick symbol (C-T26 measured its
spread at 11.62% of cost against 0.15% for BTC). With three symbols that is one ordering of
three points and it decides nothing -- picking the story that fits three points is the exact
move section 200 forbids.

The corpus supplies the mechanism and it is mechanical, not behavioural. Bouchaud Sec. 7.5: when
a queue race ends at the best quote, with probability rho_0 a limit order immediately refills the
vacated level and THE MID-PRICE REVERTS. Chapter 17 treats market-making for small-tick (17.2)
and large-tick (17.3) stocks separately for the same reason. So the prediction is directional and
was made before this measurement: the more binding the tick, the more mechanical reversion, the
more of the flow's memory is cancelled, and the lower H.

THE DESIGN. Tick-binding is not a property of a symbol, it is a property of a period: it is the
tick measured against how far the price actually moves, and that varies inside a symbol as
volatility and price level move. So the test does not need more symbols.

    k := E|price change| / tick,  in ticks per trade, measured per block of 50,000 trades

k near 1 is the binding regime; k large means the tick is irrelevant. Each symbol contributes ~40
blocks, so the ordering is tested THREE TIMES independently, within symbol, with the symbol
identity held fixed -- and then the cross-symbol points are checked for whether they fall on the
same curve. A collapse across symbols is a much stronger claim than three ordered points.

WHAT WOULD REFUTE IT. If H is unrelated to k within symbols, the cross-symbol ordering is a
coincidence of three points and C-T29's spread needs another explanation. If the within-symbol
slope has the OPPOSITE sign to the cross-symbol ordering, that is Simpson's paradox and the
cross-symbol reading is actively wrong.

The tick is MEASURED, not assumed: the smallest positive price difference observed in the symbol.
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
BLOCK = 50_000
WINDOW_T = (20, 50, 100, 200, 500)
SEED = 20260827


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan"), float("nan"), 0
    A = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ b
    n = len(x)
    s2 = float(resid @ resid) / max(n - 2, 1)
    sxx = float(((x - x.mean()) ** 2).sum())
    se = float(np.sqrt(s2 / sxx)) if sxx > 0 else float("nan")
    return float(b[1]), se, n


def loglog_slope(T, S):
    x = np.log(np.asarray(T, float))
    y = np.log(np.asarray(S, float))
    A = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(b[1])


def hurst(x):
    T, S = [], []
    for t in WINDOW_T:
        m = len(x) // t
        if m < 80:
            continue
        T.append(float(t))
        S.append(float(np.std(x[:m * t].reshape(m, t).sum(axis=1), ddof=1)))
    if len(T) < 3:
        return float("nan")
    return loglog_slope(T, S)


def measure_tick(price):
    """smallest positive price difference actually observed -- measured, not assumed"""
    d = np.abs(np.diff(np.unique(price)))
    d = d[d > 0]
    if len(d) == 0:
        return float("nan")
    # the mode of the small differences is the tick; the minimum alone is float-noise sensitive
    q = float(np.percentile(d, 1))
    return float(np.min(d[d >= q * 0.5])) if np.any(d >= q * 0.5) else float(np.min(d))


def main() -> int:
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per, pooled = {}, []
    try:
        for sym in SYMS:
            a = np.array(con.execute(
                "select price,is_buyer_maker from agg_trades where symbol=? "
                "order by ts_ms limit ?", (sym, NROWS)).fetchall(), dtype=np.float64)
            px = a[:, 0]
            eps = np.where(a[:, 1] > 0.5, -1.0, 1.0)
            tick = measure_tick(px)
            lp = np.log(px)
            d = np.empty_like(lp)
            d[0] = 0.0
            d[1:] = np.diff(lp)

            blocks = []
            nb = len(px) // BLOCK
            for i in range(nb):
                sl_ = slice(i * BLOCK, (i + 1) * BLOCK)
                p, dd, ee = px[sl_], d[sl_], eps[sl_]
                dp = np.abs(np.diff(p))
                k = float(np.mean(dp) / tick)              # ticks per trade
                zero = float(np.mean(dp == 0.0))           # fraction of unchanged prices
                H = hurst(dd)
                chi = hurst(ee)
                if not (np.isfinite(H) and np.isfinite(chi)) or chi <= 0.5:
                    continue
                blocks.append({"block": i, "k_ticks_per_trade": round(k, 4),
                               "zero_return_fraction": round(zero, 4),
                               "H": round(H, 4), "chi": round(chi, 4),
                               "cancelled": round(1.0 - (H - 0.5) / (chi - 0.5), 4),
                               "mean_price": round(float(p.mean()), 4),
                               "relative_tick_bps": round(tick / float(p.mean()) * 1e4, 4)})
            lk = [np.log(b["k_ticks_per_trade"]) for b in blocks]
            b_H, se_H, n = slope(lk, [b["H"] for b in blocks])
            b_C, se_C, _ = slope(lk, [b["cancelled"] for b in blocks])
            per[sym] = {
                "tick_measured": tick,
                "n_blocks": n,
                "k_range": [min(b["k_ticks_per_trade"] for b in blocks),
                            max(b["k_ticks_per_trade"] for b in blocks)],
                "relative_tick_bps_median": round(float(np.median(
                    [b["relative_tick_bps"] for b in blocks])), 4),
                "H_median": round(float(np.median([b["H"] for b in blocks])), 4),
                "cancelled_median": round(float(np.median(
                    [b["cancelled"] for b in blocks])), 4),
                "slope_H_on_log_k": round(b_H, 4), "se": round(se_H, 4),
                "t_H": round(b_H / se_H, 2) if se_H and np.isfinite(se_H) else None,
                "slope_cancelled_on_log_k": round(b_C, 4), "se_c": round(se_C, 4),
                "t_cancelled": round(b_C / se_C, 2) if se_C and np.isfinite(se_C) else None,
                "blocks": blocks,
            }
            for b in blocks:
                pooled.append((sym, b["k_ticks_per_trade"], b["H"], b["cancelled"]))
            sys.stderr.write("{0} done: {1} blocks, tick={2}\n".format(sym, n, tick))
    finally:
        con.close()

    # cross-symbol collapse: do the three clouds lie on one curve in k?
    lk = [np.log(r[1]) for r in pooled]
    b_all, se_all, n_all = slope(lk, [r[2] for r in pooled])
    b_allc, se_allc, _ = slope(lk, [r[3] for r in pooled])
    signs = [per[s]["slope_H_on_log_k"] for s in SYMS]
    art = {
        "study": "C-T30", "lane": "C", "utc": _utc(), "block_trades": BLOCK,
        "prediction": ("Bouchaud Sec. 7.5 -- a refill at a vacated best quote reverts the mid, so "
                       "the more binding the tick the more mechanical reversion, the lower H and "
                       "the higher the cancellation. Directional and stated before measurement."),
        "per_symbol": {s: {k: v for k, v in per[s].items() if k != "blocks"} for s in SYMS},
        "within_symbol_slopes_H_on_log_k": {s: per[s]["slope_H_on_log_k"] for s in SYMS},
        "within_symbol_t": {s: per[s]["t_H"] for s in SYMS},
        "all_three_same_sign": bool(len(set(np.sign(signs))) == 1),
        "pooled": {"n": n_all, "slope_H_on_log_k": round(b_all, 4),
                   "t": round(b_all / se_all, 2) if se_all else None,
                   "slope_cancelled_on_log_k": round(b_allc, 4),
                   "t_cancelled": round(b_allc / se_allc, 2) if se_allc else None},
        "reading_rule": ("within-symbol and pooled slopes agreeing in sign is a collapse; "
                         "disagreeing is Simpson's paradox and the cross-symbol reading fails"),
        "blocks": {s: per[s]["blocks"] for s in SYMS},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C30_TICK_BINDING_COLLAPSE_V1.json").write_text(json.dumps(art, indent=2),
                                                           encoding="utf-8")
    enc = sys.stdout.encoding or "utf-8"
    brief = {k: v for k, v in art.items() if k not in ("blocks",)}
    sys.stdout.write(json.dumps(brief, indent=2).encode(enc, "replace").decode(enc, "replace")
                     + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
