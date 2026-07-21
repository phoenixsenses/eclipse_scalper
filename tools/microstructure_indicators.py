"""
microstructure_indicators.py — academic microstructure indicators from L1 book + trades.

Beyond basic TA (price-derived), these measure ORDER-FLOW and LIQUIDITY microstructure:
  microprice, Order-Flow-Imbalance (OFI, Cont et al.), CVD + slope, Kyle's lambda (price impact),
  Amihud illiquidity, realized volatility, effective spread, trade intensity.

FRAME (honest, per SYSTEM_STATE 151-158): these are RESEARCH/CHARACTERIZATION tools. The historical
directional edge is exhausted; any "this predicts" claim must be FORWARD-validated, not backtested on
burned data. Read-only (mode=ro), causal (only data <= as_of), bounded windows.
"""
from __future__ import annotations
import sqlite3, math

SYMBOL = "ETHUSDT"


def _trades(cur, ts, win_ms, symbol=SYMBOL):
    lo = ts - win_ms
    return cur.execute("SELECT ts_ms,price,notional,is_buyer_maker FROM agg_trades "
                       "WHERE symbol=? AND ts_ms BETWEEN ? AND ? ORDER BY ts_ms",
                       (symbol, lo, ts)).fetchall()


def _book(cur, ts, win_ms, symbol=SYMBOL):
    lo = ts - win_ms
    return cur.execute("SELECT ts_ms,bid_price,bid_qty,ask_price,ask_qty,mid_price,spread_pct,book_imbalance "
                       "FROM book_ticker WHERE symbol=? AND ts_ms BETWEEN ? AND ? ORDER BY ts_ms",
                       (symbol, lo, ts)).fetchall()


def compute_microstructure(conn: sqlite3.Connection, as_of_ms: int, window_min: int = 15,
                           symbol: str = SYMBOL) -> dict:
    cur = conn.cursor()
    win = window_min * 60_000
    tr = _trades(cur, as_of_ms, win, symbol)
    bk = _book(cur, as_of_ms, win, symbol)
    out: dict = {"window_min": window_min, "n_trades": len(tr), "n_book": len(bk), "ind": {}}
    I = out["ind"]

    # ---- MICROPRICE (imbalance-weighted fair value) + deviation from mid ----
    if bk:
        _, bp, bq, ap, aq, mid, spr, imb = bk[-1]
        if bq is not None and aq is not None and (bq + aq) > 0:
            micro = (bp * aq + ap * bq) / (bq + aq)   # heavier side pulls price
            I["microprice"] = micro
            I["microprice_dev_bps"] = (micro - mid) / mid * 1e4 if mid else None
        I["spread_pct_now"] = spr
        I["book_imbalance_now"] = imb

    # ---- CVD (cumulative volume delta) + slope over window ----
    if tr:
        cvd = 0.0; series = []
        for _, _, notional, ibm in tr:
            cvd += (-(notional or 0.0)) if ibm == 1 else (notional or 0.0)  # buy(+) - sell(-)
            series.append(cvd)
        I["cvd_window"] = cvd
        # slope: (last - first) normalized by n (bps of $ per trade, descriptive)
        I["cvd_slope"] = (series[-1] - series[0]) / max(len(series), 1)
        # aggressive volumes
        asell = sum((n or 0.0) for _, _, n, ibm in tr if ibm == 1)
        abuy = sum((n or 0.0) for _, _, n, ibm in tr if ibm == 0)
        I["agg_buy_notional"] = abuy
        I["agg_sell_notional"] = asell
        I["trade_intensity_per_min"] = len(tr) / window_min

    # ---- OFI (Order Flow Imbalance, Cont-Kukanov-Stoikov, L1) ----
    if len(bk) >= 2:
        ofi = 0.0
        for k in range(1, len(bk)):
            _, bp, bq, ap, aq, *_ = bk[k]
            _, pbp, pbq, pap, paq, *_ = bk[k - 1]
            if None in (bp, bq, ap, aq, pbp, pbq, pap, paq):
                continue
            e = 0.0
            e += bq if bp > pbp else (-pbq if bp < pbp else (bq - pbq))     # bid side
            e -= aq if ap < pap else (-paq if ap > pap else (aq - paq))     # ask side
            ofi += e
        I["ofi_window"] = ofi

    # ---- Kyle's lambda (price impact per signed $ volume) & Amihud illiquidity ----
    if tr and len(tr) >= 5:
        p0 = tr[0][1]; p1 = tr[-1][1]
        ret_bps = (p1 - p0) / p0 * 1e4 if p0 else None
        signed_vol = (I.get("agg_buy_notional", 0) - I.get("agg_sell_notional", 0))
        tot_vol = (I.get("agg_buy_notional", 0) + I.get("agg_sell_notional", 0))
        I["kyle_lambda_bps_per_Mnotional"] = (ret_bps / (signed_vol / 1e6)) if (ret_bps is not None and abs(signed_vol) > 1e-9) else None
        I["amihud_illiq"] = (abs(ret_bps) / (tot_vol / 1e6)) if (ret_bps is not None and tot_vol > 1e-9) else None

    # ---- Realized volatility (from book mid ticks, sampled) ----
    if len(bk) >= 3:
        mids = [r[5] for r in bk if r[5]]
        if len(mids) >= 3:
            rets = [(mids[i] - mids[i - 1]) / mids[i - 1] for i in range(1, len(mids)) if mids[i - 1]]
            if rets:
                rv = math.sqrt(sum(x * x for x in rets)) * 1e4  # bps, window realized vol
                I["realized_vol_bps"] = rv

    # ---- Effective spread proxy (mean top-of-book spread) ----
    if bk:
        sprs = [r[6] for r in bk if r[6] is not None]
        if sprs:
            I["mean_spread_pct"] = sum(sprs) / len(sprs)

    return out


if __name__ == "__main__":
    import json
    conn = sqlite3.connect("file:data/microstructure.db?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    mx = conn.execute("SELECT MAX(ts_ms) FROM book_ticker WHERE symbol=?", (SYMBOL,)).fetchone()[0]
    out = compute_microstructure(conn, int(mx), window_min=15)
    conn.close()
    print(json.dumps(out, indent=2, default=str))
