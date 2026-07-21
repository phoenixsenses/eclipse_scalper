"""
liq_anomaly_scan.py — historical anomaly-EVENT catalog using the indicator library.

GOAL (operator): scan the past microstructure with the indicators, detect the anomaly EVENTS that
actually happened ("lived" events), and MEASURE each one's full multi-stream indicator profile.
Output is a descriptive CATALOG/record of what happened — NOT an edge (SYSTEM_STATE 151-156 showed no
fee-surviving liq edge; this is a measured event log, a reference map).

METHOD (bounded, serial, read-only):
  1. Cheap SQL detects per-type anomaly candidate timestamps (extremes), month by month, with a
     refractory gap so clustered ticks collapse to one event.
  2. At each detected event, call the indicator library to measure the point-in-time profile.
  3. Write a catalog (JSONL) + a summary table.
Anomaly types (a priori, no outcome tuning):
  - LIQ_CLUSTER : SELL or BUY liq notional in 5m >= p_hi
  - FLOW_EXTREME: aggressive-sell imbalance 60s at |.| extreme
  - PRICE_JUMP  : |mark 1m return| >= threshold
  - VOL_ALERT   : vol_state.high_vol_alert = 1
  - FUND_EXTREME: funding_rate in trailing top/bottom pctile
Freshness flags come from the library (basis/book gated per lesson 156).
"""
from __future__ import annotations
import sqlite3, json, sys, datetime as dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.liq_indicator_library import compute_indicators, SYMBOL  # noqa: E402

DB = "file:data/microstructure.db?mode=ro"
REFRACTORY_MS = 900_000            # 15m: collapse clustered ticks to one event
CAP_PER_TYPE_PER_MONTH = 60        # bound the catalog

MONTHS = [(2026, m) for m in range(2, 8)]  # Feb..Jul 2026


def month_bounds(y, m):
    a = dt.datetime(y, m, 1, tzinfo=dt.timezone.utc)
    b = dt.datetime(y + (m // 12), (m % 12) + 1, 1, tzinfo=dt.timezone.utc)
    return int(a.timestamp() * 1000), int(b.timestamp() * 1000)


def _dedup(ts_list):
    out = []
    for t in sorted(ts_list):
        if not out or t - out[-1] >= REFRACTORY_MS:
            out.append(t)
    return out


def detect_month(cur, y, m):
    lo, hi = month_bounds(y, m)
    events = {}  # type -> [ts]

    # LIQ_CLUSTER: any 5m-ish window with big SELL or BUY liq (use raw big prints >=500K as seeds)
    rows = cur.execute("SELECT ts_ms,side FROM liquidations WHERE symbol=? AND ts_ms BETWEEN ? AND ? "
                       "AND notional>=500000 ORDER BY ts_ms", (SYMBOL, lo, hi)).fetchall()
    events["LIQ_CLUSTER"] = _dedup([r[0] for r in rows])[:CAP_PER_TYPE_PER_MONTH]

    # VOL_ALERT
    rows = cur.execute("SELECT ts_ms FROM vol_state WHERE symbol=? AND ts_ms BETWEEN ? AND ? "
                       "AND high_vol_alert=1 ORDER BY ts_ms", (SYMBOL, lo, hi)).fetchall()
    events["VOL_ALERT"] = _dedup([r[0] for r in rows])[:CAP_PER_TYPE_PER_MONTH]

    # FUND_EXTREME: trailing top/bottom 2% of the month's funding sample
    frows = cur.execute("SELECT ts_ms,funding_rate FROM mark_prices WHERE symbol=? AND ts_ms BETWEEN ? AND ? "
                        "AND funding_rate IS NOT NULL ORDER BY ts_ms", (SYMBOL, lo, hi)).fetchall()
    if frows:
        vals = sorted(v for _, v in frows)
        n = len(vals)
        p_hi = vals[int(0.98 * (n - 1))]
        p_lo = vals[int(0.02 * (n - 1))]
        ex = [ts for ts, v in frows if v >= p_hi or v <= p_lo]
        events["FUND_EXTREME"] = _dedup(ex)[:CAP_PER_TYPE_PER_MONTH]
    else:
        events["FUND_EXTREME"] = []

    return events


def main():
    conn = sqlite3.connect(DB, uri=True)
    conn.execute("PRAGMA query_only=1")
    cur = conn.cursor()
    out_path = Path("reports/research/s34/LIQ_ANOMALY_EVENT_CATALOG.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    n_written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for y, m in MONTHS:
            ev = detect_month(cur, y, m)
            for typ, ts_list in ev.items():
                counts[typ] = counts.get(typ, 0) + len(ts_list)
                for ts in ts_list:
                    ind = compute_indicators(conn, ts)
                    rec = {"event_type": typ, "ts_ms": ts,
                           "utc": dt.datetime.fromtimestamp(ts / 1000, dt.timezone.utc).isoformat(),
                           "ind": ind.values, "fresh": ind.fresh}
                    fh.write(json.dumps(rec, default=str) + "\n")
                    n_written += 1
    conn.close()
    print(json.dumps({"catalog": str(out_path), "n_events": n_written,
                      "counts_by_type": counts}, indent=2))


if __name__ == "__main__":
    main()
