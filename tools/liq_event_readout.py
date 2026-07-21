"""
liq_event_readout.py — human-readable BEFORE / DURING / AFTER readout of a liquidation event.

PURPOSE (operator): while a liq is happening, understand WHAT is going on — the full arc:
  BEFORE (the move that causes it) -> DURING (the liq firing) -> AFTER (the outcome).
This is UNDERSTANDING / situational awareness, NOT prediction. Established this session
(SYSTEM_STATE 151-158): a liq is the LAGGING EFFECT of a move already underway; the "after" is
descriptive only (vol clusters, but direction is ~coin-flip). So the readout tells you the STORY,
honestly, without pretending to forecast the next move.

Read-only (mode=ro). Reuses tools/liq_indicator_library for the multi-stream fingerprint.
"""
from __future__ import annotations
import sqlite3, datetime as dt, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.liq_indicator_library import compute_indicators, SYMBOL  # noqa: E402

PRE = [("-1h", 3_600_000), ("-30m", 1_800_000), ("-15m", 900_000), ("-5m", 300_000)]
POST = [("+5m", 300_000), ("+15m", 900_000), ("+30m", 1_800_000), ("+1h", 3_600_000), ("+6h", 6 * 3_600_000)]


def _mark(cur, ts):
    r = cur.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? "
                    "ORDER BY ts_ms DESC LIMIT 1", (SYMBOL, ts)).fetchone()
    return r[0] if r else None


def readout(conn, ev_ts, out=print):
    cur = conn.cursor()
    row = cur.execute("SELECT SUM(CASE WHEN side='SELL' THEN notional ELSE 0 END),"
                      "SUM(CASE WHEN side='BUY' THEN notional ELSE 0 END),MAX(notional),COUNT(*) "
                      "FROM liquidations WHERE symbol=? AND ts_ms BETWEEN ? AND ?",
                      (SYMBOL, ev_ts - 300_000, ev_ts + 300_000)).fetchone()
    lsell, lbuy, lmax, lc = (row[0] or 0, row[1] or 0, row[2] or 0, row[3] or 0)
    kind = "LONG-FLUSH (SELL-liq, longs wiped)" if lsell > lbuy else "SHORT-SQUEEZE (BUY-liq, shorts wiped)"
    dom = max(lsell, lbuy)
    p0 = _mark(cur, ev_ts)
    ind = compute_indicators(conn, ev_ts).values

    when = dt.datetime.fromtimestamp(ev_ts / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
    out(f"=== LIQ EVENT {when} UTC — {kind} ===")
    out(f"  SIZE: ${dom/1e6:.1f}M dominant, max-print ${lmax/1e6:.1f}M, {lc} liqs in +-5m")

    out("  BEFORE (the CAUSE — the move is already happening; liq follows it):")
    for lbl, ms in PRE:
        pp = _mark(cur, ev_ts - ms)
        out(f"    {lbl:>4}: {(p0-pp)/pp*1e4 if p0 and pp else 0:+6.0f} bps")

    out("  DURING (the liq firing — flow/book/positioning state):")
    out(f"    flow sell-imb 60s = {ind.get('flow_sell_imbalance_60s')}")
    out(f"    book_imbalance = {ind.get('book_imbalance')}  spread% = {ind.get('spread_pct')}  "
        f"bid_depth = ${(ind.get('bid_depth_usd') or 0)/1e3:.0f}K")
    out(f"    funding = {ind.get('funding_rate')} (14d pctile {ind.get('funding_pctile_14d')})  "
        f"OI = ${(ind.get('open_interest_usd') or 0)/1e9:.2f}B  vol_decile = {ind.get('vol_decile')}")

    out("  AFTER (DESCRIPTIVE outcome — NOT a prediction; direction is ~coin-flip):")
    for lbl, ms in POST:
        pa = _mark(cur, ev_ts + ms)
        out(f"    {lbl:>4}: {(pa-p0)/p0*1e4 if pa and p0 else 0:+6.0f} bps")


if __name__ == "__main__":
    conn = sqlite3.connect("file:data/microstructure.db?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    last = conn.execute("SELECT ts_ms FROM liquidations WHERE symbol=? AND notional>=200000 "
                        "ORDER BY ts_ms DESC LIMIT 1", (SYMBOL,)).fetchone()
    if last:
        readout(conn, last[0])
    conn.close()
