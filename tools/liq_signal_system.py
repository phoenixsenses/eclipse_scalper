"""
liq_signal_system.py — comprehensive TIP/decision-support readout for a liquidation event.

Culmination of the microstructure fingerprint research (SYSTEM_STATE 151-158+). This is a TIP system,
NOT an alpha edge: it produces honest PROBABILISTIC hints + situational awareness for a liq event,
combining every indicator we built. Nothing here claims a fee-surviving edge (none was found); it
tilts odds and characterizes state — decision support.

Findings baked in (all descriptive, forward-unvalidated):
  - flush(SELL-liq)->bounce / squeeze(BUY-liq)->reversal base rate ~58%
  - exhaustion (decel_flow<0.9): modest bounce tilt (~62% on broad sample; 79% on n=14, likely noisy)
  - continuation-risk: if selling CONTINUES after T0 -> disaster (reactive, not predictive)
  - BTC-ETH: ~0.85 contemporaneous coupling, NO clean lead-lag -> BTC is context, not a leader
Read-only, causal.
"""
from __future__ import annotations
import sqlite3, sys, datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.liq_indicator_library import compute_indicators  # noqa: E402
from tools.microstructure_indicators import compute_microstructure  # noqa: E402

SYM = "ETHUSDT"


def _session(hour_utc: int) -> str:
    """Trading session by UTC hour. Regime finding (SYSTEM_STATE §161, in-sample n=280):
    the SELL-flush->bounce signal RANK-ORDERS materially better in ASIA/EU than US hours."""
    if hour_utc < 7:
        return "ASIA"
    if hour_utc < 13:
        return "EU"
    return "US"


def _session_context(ev_ts):
    """Session-conditional situational tilt. IN-SAMPLE regime discovery, NOT a tradeable edge:
    drop->bounce rank-ordering AUC ~0.68 in ASIA/EU vs ~0.50 in US (n=280) — but even the strong
    session is net-of-fee NEGATIVE (Asia+EU 30m = -9bps). So this only tilts *reliability of
    ordering*, never promises profit. Month-inconsistent (Apr 0.63 / Jun 0.50 / Jul 0.67) =>
    fragile => the liq_tip_forward ledger must confirm it forward before it is trusted at all."""
    h = dt.datetime.fromtimestamp(ev_ts / 1000, dt.timezone.utc).hour
    sess = _session(h)
    if sess in ("ASIA", "EU"):
        read = f"{sess} (bounce ordering historically more reliable, AUC~0.68 in-sample; NOT fee-tradeable, forward-unconfirmed)"
    else:
        read = "US (bounce ordering historically random, AUC~0.50; treat flush as directionless)"
    return {"session": sess, "hour_utc": h, "read": read}


def _mk(cur, sym, ts):
    r = cur.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (sym, ts)).fetchone()
    return r[0] if r else None


def _aggvol(cur, ibm, a, b, sym=SYM):
    return cur.execute(f"SELECT SUM(notional) FROM agg_trades WHERE symbol=? AND is_buyer_maker={ibm} AND ts_ms BETWEEN ? AND ?", (sym, a, b)).fetchone()[0] or 0.0


def continuation_check(conn, ev_ts):
    """REACTIVE risk assessment, valid ~60s AFTER the event. Developed & validated (SELL-flush n=129):
    the -447bps disasters occur ONLY when heavy selling continues in the first 60s; if buying returns
    fast the tail caps ~-130. This is the early-warning continuation detector (60s, earlier than 2min)."""
    cur = conn.cursor()
    r = cur.execute("SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),"
                    "SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END) FROM agg_trades "
                    "WHERE symbol=? AND ts_ms BETWEEN ? AND ?", (SYM, ev_ts, ev_ts + 60_000)).fetchone()
    cf = (r[0] or 0) - (r[1] or 0)   # CVD first 60s (buy - sell)
    if cf > 0:
        return {"cvd_first60s_$M": round(cf / 1e6, 2), "risk": "LOW (buying returned; tail historically caps ~-130)"}
    if cf > -1e6:
        return {"cvd_first60s_$M": round(cf / 1e6, 2), "risk": "MODERATE (mild selling)"}
    return {"cvd_first60s_$M": round(cf / 1e6, 2), "risk": "HIGH — heavy selling continues; ALL -447 disasters live here → CUT"}


def _trend_context(cur, ev_ts, e0):
    """Regime context (partly explains the liq-size bounce reversal): flushes in a strong 4h downtrend
    have higher continuation risk (big liqs cluster there; -83 vs -51bps median). T0-observable version
    of continuation risk. Weak but real."""
    e4 = None
    r = cur.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (SYM, ev_ts - 4 * 3_600_000)).fetchone()
    if r:
        e4 = r[0]
    t4 = (e0 - e4) / e4 * 1e4 if (e0 and e4) else None
    if t4 is None:
        return {"trend4h_bps": None, "read": "unknown"}
    read = "STRONG-DOWNTREND (continuation risk HIGHER)" if t4 < -100 else ("mild-down" if t4 < 0 else "up/range (bounce more reliable)")
    return {"trend4h_bps": round(t4, 1), "read": read}


def signal_readout(conn, ev_ts):
    cur = conn.cursor()
    # identity
    row = cur.execute("SELECT SUM(CASE WHEN side='SELL' THEN notional ELSE 0 END),"
                      "SUM(CASE WHEN side='BUY' THEN notional ELSE 0 END),MAX(notional),COUNT(*) "
                      "FROM liquidations WHERE symbol=? AND ts_ms BETWEEN ? AND ?",
                      (SYM, ev_ts - 300_000, ev_ts + 300_000)).fetchone()
    lsell, lbuy, lmax, lc = (row[0] or 0, row[1] or 0, row[2] or 0, row[3] or 0)
    is_flush = lsell >= lbuy
    kind = "LONG-FLUSH" if is_flush else "SHORT-SQUEEZE"
    bias = "LONG (bounce)" if is_flush else "SHORT (reversal)"
    dom = max(lsell, lbuy)
    ibm = 1 if is_flush else 0  # exhausting side: sells for flush, buys for squeeze

    # bounce-tilt = FORCED-PRINT FOOTPRINT (robust across >=200K, survived where exhaustion collapsed).
    # huge single forced print => violent capitulation => ~65% bounce vs ~57% for small-print (8pp tilt).
    maxprint = cur.execute(f"SELECT MAX(notional) FROM agg_trades WHERE symbol=? AND is_buyer_maker={ibm} AND ts_ms BETWEEN ? AND ?", (SYM, ev_ts - 60_000, ev_ts)).fetchone()[0] or 0.0
    big_print = maxprint > 1_000_000  # ~median at >=500K; >$1M forced print = whale/violent
    tilt = "modest-favorable (~65%, huge forced print)" if big_print else "slightly-below-base (~57%, retail-size prints)"
    # exhaustion also computed but DOWNGRADED (tested: measurement-dependent noise, collapses on broad sample)
    sr = _aggvol(cur, ibm, ev_ts - 30_000, ev_ts); sp = _aggvol(cur, ibm, ev_ts - 90_000, ev_ts - 30_000)
    decel = sr / (sp / 2) if sp > 0 else None

    # BTC context (same-window move, coupling not lead)
    e5 = _mk(cur, SYM, ev_ts - 300_000); e0 = _mk(cur, SYM, ev_ts)
    b5 = _mk(cur, "BTCUSDT", ev_ts - 300_000); b0 = _mk(cur, "BTCUSDT", ev_ts)
    eth_move = (e0 - e5) / e5 * 1e4 if e5 and e0 else None
    btc_move = (b0 - b5) / b5 * 1e4 if b5 and b0 else None
    btc_ctx = "with-BTC (coupled)" if (eth_move and btc_move and eth_move * btc_move > 0) else "ETH-idiosyncratic (BTC diverging)"

    # indicators
    ind = compute_indicators(conn, ev_ts).values
    mic = compute_microstructure(conn, ev_ts, window_min=5)["ind"]

    return {
        "utc_ms": ev_ts, "identity": kind, "trade_bias": bias,
        "size_$M": round(dom / 1e6, 2), "max_print_$M": round(lmax / 1e6, 2), "liqs_10m": lc,
        "pre_5m_move_bps": round(eth_move, 1) if eth_move is not None else None,
        "TIP_bounce_tilt": {"max_forced_print_$M": round(maxprint / 1e6, 2), "big_print": big_print, "tilt": tilt,
                            "decel_flow_NOISY": round(decel, 2) if decel else None},
        "TIP_btc_context": {"btc_5m_bps": round(btc_move, 1) if btc_move is not None else None, "read": btc_ctx},
        "TIP_session_regime": _session_context(ev_ts),
        "TIP_regime_context": _trend_context(cur, ev_ts, e0),
        "TIP_positioning": {"funding_pctile_14d": ind.get("funding_pctile_14d"), "oi_$B": round((ind.get("open_interest_usd") or 0) / 1e9, 2)},
        "TIP_microstructure": {"OFI": round(mic.get("ofi_window") or 0), "CVD_$M": round((mic.get("cvd_window") or 0) / 1e6, 2),
                               "microprice_dev_bps": round(mic.get("microprice_dev_bps") or 0, 3),
                               "kyle_lambda": round(mic.get("kyle_lambda_bps_per_Mnotional") or 0, 2) if mic.get("kyle_lambda_bps_per_Mnotional") else None},
        "TIP_continuation_risk": continuation_check(conn, ev_ts),  # reactive; valid ~60s after event
        "HONEST_NOTE": "TIP only, not an edge. Odds tilts are weak + forward-unvalidated. book-refill tested & DROPPED (no separation).",
    }


if __name__ == "__main__":
    import json
    conn = sqlite3.connect("file:data/microstructure.db?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    last = conn.execute("SELECT ts_ms FROM liquidations WHERE symbol=? AND notional>=200000 ORDER BY ts_ms DESC LIMIT 1", (SYM,)).fetchone()
    out = signal_readout(conn, last[0])
    conn.close()
    print(json.dumps(out, indent=2, default=str))
