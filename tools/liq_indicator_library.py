"""
liq_indicator_library.py — point-in-time multi-stream INDICATOR library (ETHUSDT).

WHAT THIS IS (honest framing):
  A reusable, read-only, freshness-aware indicator/characterization layer computed from the
  microstructure DB streams. It produces a point-in-time indicator vector at any timestamp, so it
  works both LIVE (pass "now") and HISTORICALLY (pass any past ts). It is TOOLING for:
    (a) live monitoring / discretionary situational awareness, and
    (b) a frozen feature foundation for any FUTURE forward-validated test.

WHAT THIS IS NOT:
  It is NOT a backtested alpha and it does NOT rediscover a liquidation edge. Six convergent studies
  (SYSTEM_STATE 151-156) established that this ETH-perp microstructure has no clean fee-surviving,
  forward-exploitable liq edge; combining these indicators over the same burned data will not change
  that. Any predictive claim built on these indicators MUST be pre-registered and forward-validated.

DESIGN PRINCIPLES:
  - READ-ONLY DB (mode=ro, query_only=1). Bounded, indexed ts_ms queries only. No writes, no full scans.
  - EVERY indicator carries a freshness/validity flag. Lesson from 156: an indicator built on a stale
    feed (esp. spot -> basis) is an ARTIFACT. If the source is stale, the indicator is marked invalid
    rather than silently wrong.
  - Point-in-time / causal: only data with ts <= as_of is used. No lookahead.
"""
from __future__ import annotations
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Any

from ami.storage.union_reader import history_floor_ms

SYMBOL = "ETHUSDT"
# freshness ceilings (ms): beyond these the source is considered stale -> indicator invalid
FRESH_MARK_MS = 30_000        # mark ~6s cadence
FRESH_SPOT_MS = 20_000        # spot: honor 156 -> only ~60s live regime is usable; require <=20s
FRESH_BOOK_MS = 10_000        # book ~sub-second when live
FRESH_FUND_MS = 15 * 60_000   # funding updates slowly
FRESH_OI_MS = 5 * 60_000      # OI poller ~60s public endpoint

# How far BACK this module reaches. Named rather than inlined so that an audit of
# "how much history does a standing role need?" can find it: section 265 measured that
# question by grepping each role's own *_LOOKBACK_MS constants, and missed this one
# entirely because it lived two imports away inside an expression. The answer it
# produced (7 days) was wrong; the true figure is this constant (14 days).
FUNDING_PCTILE_WINDOW_MS = 14 * 86_400_000


def _one(cur, sql, args=()):
    r = cur.execute(sql, args).fetchone()
    return r if r else None


def _last_before(cur, table, col, ts, extra=""):
    """Most recent row of `table` with ts_ms <= ts (indexed). Returns (ts_ms, value) or None."""
    row = _one(cur,
               f"SELECT ts_ms,{col} FROM {table} "
               f"WHERE symbol=? AND ts_ms<=? {extra} ORDER BY ts_ms DESC LIMIT 1",
               (SYMBOL, ts))
    return (row[0], row[1]) if row else None


@dataclass
class Indicators:
    as_of_ms: int
    symbol: str = SYMBOL
    values: dict[str, Any] = field(default_factory=dict)
    fresh: dict[str, bool] = field(default_factory=dict)   # per-source validity
    notes: list[str] = field(default_factory=list)

    def set(self, name, value, source_ok=True):
        self.values[name] = value
        self.fresh[name] = bool(source_ok) and value is not None

    def as_dict(self):
        return asdict(self)


def _mark_ret_bps(cur, ts, back_ms):
    now = _last_before(cur, "mark_prices", "mark_price", ts)
    then = _last_before(cur, "mark_prices", "mark_price", ts - back_ms)
    if not now or not then or not then[1]:
        return None
    return (now[1] - then[1]) / then[1] * 1e4


def compute_indicators(conn: sqlite3.Connection, as_of_ms: int, symbol: str = SYMBOL) -> Indicators:
    """Compute the point-in-time indicator vector at `as_of_ms` (causal, read-only)."""
    cur = conn.cursor()
    ind = Indicators(as_of_ms=as_of_ms, symbol=symbol)

    # ---- PRICE / MARK ----
    m = _last_before(cur, "mark_prices", "mark_price", as_of_ms)
    mark_fresh = bool(m) and (as_of_ms - m[0] <= FRESH_MARK_MS)
    ind.set("mark_price", m[1] if m else None, mark_fresh)
    for lbl, ms in (("ret_1m_bps", 60_000), ("ret_15m_bps", 900_000),
                    ("ret_1h_bps", 3_600_000), ("ret_4h_bps", 4 * 3_600_000)):
        ind.set(lbl, _mark_ret_bps(cur, as_of_ms, ms), mark_fresh)

    # ---- VOL (precomputed) ----
    v = _one(cur, "SELECT ts_ms,rv_5m,vol_decile,high_vol_alert FROM vol_state "
                  "WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (symbol, as_of_ms))
    ind.set("rv_5m", v[1] if v else None, bool(v))
    ind.set("vol_decile", v[2] if v else None, bool(v))
    ind.set("high_vol_alert", v[3] if v else None, bool(v))

    # ---- FUNDING (level + extremity via trailing distribution) ----
    f = _last_before(cur, "mark_prices", "funding_rate", as_of_ms)
    fund_fresh = bool(f) and (as_of_ms - f[0] <= FRESH_FUND_MS)
    ind.set("funding_rate", f[1] if f else None, fund_fresh)
    # extremity: percentile of current funding within trailing 14d sample (bounded query)
    #
    # The window must be BACKED BY DATA, not merely requested. SQL does not error when a
    # range predicate reaches before the earliest stored row - it quietly returns the
    # shorter sample, so after the frozen segment is deleted this becomes a 12-day
    # percentile still labelled 14d. That is the same class of defect as the stale-feed
    # artifact this module was built around (see the header): an indicator computed on a
    # source that cannot support it is invalid, not approximately right.
    if f and f[1] is not None:
        lo = as_of_ms - FUNDING_PCTILE_WINDOW_MS
        floor = history_floor_ms(conn, "mark_prices")
        if floor is not None and floor <= lo:
            row = _one(cur, "SELECT AVG(funding_rate), "
                            "AVG(CASE WHEN funding_rate<=? THEN 1.0 ELSE 0.0 END) "
                            "FROM mark_prices WHERE symbol=? AND ts_ms BETWEEN ? AND ? "
                            "AND funding_rate IS NOT NULL", (f[1], symbol, lo, as_of_ms))
            ind.set("funding_pctile_14d", row[1] if row else None, fund_fresh)
        else:
            # marked invalid rather than raising: these run in unsupervised 30s loops,
            # and section 189 hardened them precisely so a missing segment cannot kill them
            ind.set("funding_pctile_14d", None, False)
    else:
        ind.set("funding_pctile_14d", None, False)

    # ---- OPEN INTEREST (level + change) ----
    oi = _last_before(cur, "open_interest", "open_interest_usd", as_of_ms)
    oi_fresh = bool(oi) and (as_of_ms - oi[0] <= FRESH_OI_MS)
    oi_prev = _last_before(cur, "open_interest", "open_interest_usd", as_of_ms - 3_600_000)
    ind.set("open_interest_usd", oi[1] if oi else None, oi_fresh)
    ind.set("oi_chg_1h_pct",
            ((oi[1] - oi_prev[1]) / oi_prev[1] * 100.0) if (oi and oi_prev and oi_prev[1]) else None,
            oi_fresh and bool(oi_prev))

    # ---- BASIS (mark - spot) WITH MANDATORY SPOT-FRESHNESS GATE (lesson: SYSTEM_STATE 156) ----
    s = _last_before(cur, "spot_prices", "spot_price", as_of_ms)
    spot_age = (as_of_ms - s[0]) if s else None
    spot_fresh = bool(s) and spot_age is not None and spot_age <= FRESH_SPOT_MS
    if m and s and s[1] and spot_fresh and mark_fresh:
        ind.set("basis_bps", (m[1] - s[1]) / s[1] * 1e4, True)
    else:
        ind.set("basis_bps", None, False)
        ind.notes.append(f"basis invalid: spot_age_ms={spot_age} (need<= {FRESH_SPOT_MS}); "
                         f"per 156 stale spot manufactures fake basis")
    ind.values["spot_age_ms"] = spot_age

    # ---- BOOK (top-of-book state; Apr-11+) ----
    b = _one(cur, "SELECT ts_ms,spread_pct,book_imbalance,bid_depth_usd FROM book_ticker "
                  "WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (symbol, as_of_ms))
    book_fresh = bool(b) and (as_of_ms - b[0] <= FRESH_BOOK_MS)
    ind.set("spread_pct", b[1] if b else None, book_fresh)
    ind.set("book_imbalance", b[2] if b else None, book_fresh)
    ind.set("bid_depth_usd", b[3] if b else None, book_fresh)

    # ---- FLOW (agg_trades, last 60s window; bounded) ----
    lo = as_of_ms - 60_000
    row = _one(cur, "SELECT "
                    "SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END), "  # aggressive SELL
                    "SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END), "  # aggressive BUY
                    "COUNT(*) FROM agg_trades WHERE symbol=? AND ts_ms BETWEEN ? AND ?",
                    (symbol, lo, as_of_ms))
    asell, abuy, ntr = (row[0] or 0.0, row[1] or 0.0, row[2] or 0) if row else (0.0, 0.0, 0)
    tot = asell + abuy
    ind.set("asell_notional_60s", asell, True)
    ind.set("flow_sell_imbalance_60s", ((asell - abuy) / tot) if tot else None, tot > 0)
    ind.set("trade_count_60s", ntr, True)

    # ---- LIQUIDATIONS (activity in last 5m; side imbalance) ----
    lo5 = as_of_ms - 300_000
    row = _one(cur, "SELECT "
                    "SUM(CASE WHEN side='SELL' THEN notional ELSE 0 END), "
                    "SUM(CASE WHEN side='BUY' THEN notional ELSE 0 END), "
                    "MAX(notional), COUNT(*) FROM liquidations "
                    "WHERE symbol=? AND ts_ms BETWEEN ? AND ?", (symbol, lo5, as_of_ms))
    lsell, lbuy, lmax, lcnt = (row[0] or 0.0, row[1] or 0.0, row[2] or 0.0, row[3] or 0) if row else (0.0, 0.0, 0.0, 0)
    ind.set("liq_sell_notional_5m", lsell, True)
    ind.set("liq_buy_notional_5m", lbuy, True)
    ind.set("liq_max_notional_5m", lmax, True)
    ind.set("liq_count_5m", lcnt, True)

    return ind


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIC TA INDICATORS (price-derived) + STANDARD frozen thresholds.
# HONEST NOTE: these are functions of price. Price is directionally-efficient here
# (SYSTEM_STATE 157). A VWAP+EMA+delta system was already built on this data and died
# at standard fees (Scalper Stack, Jun 2026, ~+4bps gross). These are TOOLING/monitoring,
# not a rediscovered edge. THRESHOLDS BELOW ARE STANDARD CONVENTIONS, FROZEN — never tune
# them on burned data (that is the graveyard trap); any signal must be forward-validated.
# ─────────────────────────────────────────────────────────────────────────────

def _minute_bars(cur, as_of_ms, lookback_min, symbol=SYMBOL):
    """Causal 1-min close series over [as_of-lookback, as_of] (last mark per minute)."""
    lo = as_of_ms - lookback_min * 60_000
    rows = cur.execute("SELECT ts_ms,mark_price FROM mark_prices "
                       "WHERE symbol=? AND ts_ms BETWEEN ? AND ? ORDER BY ts_ms",
                       (symbol, lo, as_of_ms)).fetchall()
    bars = {}
    for ts, p in rows:
        bars[ts // 60_000] = p
    return [bars[k] for k in sorted(bars)]  # chronological closes


def _ema(series, period):
    if len(series) < period:
        return None
    k = 2.0 / (period + 1)
    e = series[0]
    for x in series[1:]:
        e = x * k + e * (1 - k)
    return e


def _rsi(series, period=14):
    if len(series) < period + 1:
        return None
    gains = losses = 0.0
    for i in range(-period, 0):
        d = series[i] - series[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - 100.0 / (1 + rs)


def _macd(series, fast=12, slow=26, signal=9):
    if len(series) < slow + signal:
        return None, None
    def ema_series(s, p):
        k = 2.0 / (p + 1); e = s[0]; out = [e]
        for x in s[1:]:
            e = x * k + e * (1 - k); out.append(e)
        return out
    macd_line = [f - s for f, s in zip(ema_series(series, fast), ema_series(series, slow))]
    sig = ema_series(macd_line, signal)
    return macd_line[-1], macd_line[-1] - sig[-1]  # macd, histogram


def _boll(series, period=20, k=2.0):
    if len(series) < period:
        return None
    win = series[-period:]
    mid = sum(win) / period
    var = sum((x - mid) ** 2 for x in win) / period
    sd = var ** 0.5
    return mid, mid + k * sd, mid - k * sd


def compute_ta_indicators(conn: sqlite3.Connection, as_of_ms: int, symbol: str = SYMBOL) -> dict:
    """Classic TA on the causal 1-min mark series, with STANDARD frozen-threshold signals."""
    cur = conn.cursor()
    s = _minute_bars(cur, as_of_ms, lookback_min=300, symbol=symbol)  # 5h of 1-min closes
    out = {"n_bars": len(s), "signals": {}, "values": {}}
    if len(s) < 30:
        out["note"] = "insufficient bars"
        return out
    price = s[-1]
    ema9, ema21, ema50 = _ema(s, 9), _ema(s, 21), _ema(s, 50)
    rsi = _rsi(s, 14)
    macd, hist = _macd(s)
    boll = _boll(s, 20, 2.0)
    out["values"] = {"price": price, "ema9": ema9, "ema21": ema21, "ema50": ema50,
                     "rsi14": rsi, "macd": macd, "macd_hist": hist,
                     "boll_mid": boll[0] if boll else None,
                     "boll_up": boll[1] if boll else None,
                     "boll_dn": boll[2] if boll else None}
    # STANDARD FROZEN thresholds (conventions — do NOT tune on burned data):
    sg = out["signals"]
    if rsi is not None:
        sg["rsi_oversold_30"] = rsi < 30      # convention
        sg["rsi_overbought_70"] = rsi > 70    # convention
    if ema9 and ema21:
        sg["ema9_over_ema21"] = ema9 > ema21  # short-trend up
    if ema21 and ema50:
        sg["ema21_over_ema50"] = ema21 > ema50
    if hist is not None:
        sg["macd_bull"] = hist > 0
    if boll:
        sg["above_boll_up_2sd"] = price > boll[1]   # 2σ convention
        sg["below_boll_dn_2sd"] = price < boll[2]
    return out


if __name__ == "__main__":
    import json
    conn = sqlite3.connect("file:data/microstructure.db?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    mx = conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol=?", (SYMBOL,)).fetchone()[0]
    out = compute_indicators(conn, int(mx))
    ta = compute_ta_indicators(conn, int(mx))
    conn.close()
    print(json.dumps({"microstructure": out.as_dict(), "ta": ta}, indent=2, default=str))
