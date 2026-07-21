"""S34 Feature Factory Phase 2 — multi-symbol / multi-side expansion.

Appends ETHUSDT SELL, SOLUSDT BUY/SELL, BTCUSDT BUY/SELL to the existing
s34_feature_factory.db using non-destructive UPSERT.

Existing ETH BUY rows are preserved unless --symbols includes ETHUSDT with
--sides BUY and --rebuild is passed.

Usage:
  python tools/research_s34_feature_factory_phase2_multi.py
  python tools/research_s34_feature_factory_phase2_multi.py --symbols SOLUSDT --sides SELL
  python tools/research_s34_feature_factory_phase2_multi.py --rebuild
  python tools/research_s34_feature_factory_phase2_multi.py --dry-run
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DB_URI = f"file:{(ROOT / 'data' / 'microstructure.db').as_posix()}?mode=ro"
FEATURE_DB_PATH = ROOT / "data" / "s34_feature_factory.db"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_FEATURE_FACTORY_PHASE2_MULTI.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_FEATURE_FACTORY_PHASE2_MULTI.md"

MAX_HORIZON_SEC = 3600
FEE_BPS = 8.0
BUCKET_SEC = 300
MIN_GAP_SEC = 900


# ---------------------------------------------------------------------------
# Route catalogue: (route_id, direction, entry_delay_sec, tp_bps, sl_bps, be_bps)
# LONG routes: SL=40, BE=30  (matches BUY pre-reg standard)
# SHORT routes: SL=40, BE=40 (matches SELL pre-reg standard)
# ---------------------------------------------------------------------------

def _r(rid: str, direction: str, delay: int, tp: float, sl: float, be: float) -> dict[str, Any]:
    return {"route_id": rid, "direction": direction, "entry_delay_sec": delay,
            "tp_bps": tp, "sl_bps": sl, "be_bps": be}


ROUTES: dict[tuple[str, str], list[dict]] = {
    ("ETHUSDT", "BUY"): [  # already in DB — included for --rebuild only
        _r("LONG_DELAY0_TP60",         "LONG",  0,  60.0, 40.0, 30.0),
        _r("LONG_DELAY60_TP120",       "LONG",  60, 120.0, 40.0, 30.0),
        _r("SHORT_DELAY0_TP40_CONTROL","SHORT", 0,  40.0, 40.0, 30.0),
    ],
    ("ETHUSDT", "SELL"): [
        _r("SHORT_DELAY0_TP60",        "SHORT", 0,  60.0, 40.0, 40.0),
        _r("SHORT_DELAY0_TP80",        "SHORT", 0,  80.0, 40.0, 40.0),
        _r("LONG_DELAY0_TP40_CONTROL", "LONG",  0,  40.0, 40.0, 30.0),
    ],
    ("SOLUSDT", "BUY"): [
        _r("LONG_DELAY0_TP60",         "LONG",  0, 60.0, 40.0, 30.0),
        _r("SHORT_DELAY0_TP40_CONTROL","SHORT", 0, 40.0, 40.0, 40.0),
    ],
    ("SOLUSDT", "SELL"): [
        _r("SHORT_DELAY0_TP60",        "SHORT", 0, 60.0, 40.0, 40.0),
        _r("SHORT_DELAY0_TP40",        "SHORT", 0, 40.0, 40.0, 40.0),
        _r("LONG_DELAY0_TP40_CONTROL", "LONG",  0, 40.0, 40.0, 30.0),
    ],
    ("BTCUSDT", "BUY"): [
        _r("LONG_DELAY0_TP60",         "LONG",  0, 60.0, 40.0, 30.0),
        _r("SHORT_DELAY0_TP40_CONTROL","SHORT", 0, 40.0, 40.0, 40.0),
    ],
    ("BTCUSDT", "SELL"): [
        _r("SHORT_DELAY0_TP40",        "SHORT", 0, 40.0, 40.0, 40.0),
        _r("SHORT_DELAY0_TP60",        "SHORT", 0, 60.0, 40.0, 40.0),
        _r("LONG_DELAY0_TP40_CONTROL", "LONG",  0, 40.0, 40.0, 30.0),
    ],
}

# Minimum cluster threshold per symbol-side (calculator filters up via cluster_notional)
THRESHOLDS: dict[tuple[str, str], float] = {
    ("ETHUSDT", "BUY"):  200_000.0,
    ("ETHUSDT", "SELL"): 500_000.0,
    ("SOLUSDT", "BUY"):  100_000.0,
    ("SOLUSDT", "SELL"): 100_000.0,
    ("BTCUSDT", "BUY"):  1_000_000.0,
    ("BTCUSDT", "SELL"): 1_000_000.0,
}

# Default combos to run (excludes ETH BUY which is already done)
DEFAULT_COMBOS: list[tuple[str, str]] = [
    ("ETHUSDT", "SELL"),
    ("SOLUSDT", "BUY"),
    ("SOLUSDT", "SELL"),
    ("BTCUSDT", "BUY"),
    ("BTCUSDT", "SELL"),
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def iso(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return dt.datetime.fromtimestamp(ts_ms / 1000, dt.timezone.utc).isoformat()


def day_start_ms(ts_ms: int) -> int:
    d = dt.datetime.fromtimestamp(ts_ms / 1000, dt.timezone.utc).date()
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000)


def shape_label(duration_sec: float, max_share_pct: float) -> str:
    if max_share_pct >= 80.0:
        return "single_dominant_80pct"
    if duration_sec >= 120.0:
        return "stretched_120s"
    return "distributed_mid_duration"


# ---------------------------------------------------------------------------
# In-memory mark-price index for fast lookups
# ---------------------------------------------------------------------------

class MarkIndex:
    """Sorted (ts_ms, price) pairs for O(log N) lookups."""

    def __init__(self, rows: list[tuple[int, float]]) -> None:
        self._ts: list[int] = [int(r[0]) for r in rows]
        self._px: list[float] = [float(r[1]) for r in rows]

    def at_or_after(self, ts_ms: int) -> tuple[int, float] | None:
        idx = bisect.bisect_left(self._ts, ts_ms)
        if idx >= len(self._ts):
            return None
        return self._ts[idx], self._px[idx]

    def at_or_before(self, ts_ms: int) -> tuple[int, float] | None:
        idx = bisect.bisect_right(self._ts, ts_ms) - 1
        if idx < 0:
            return None
        return self._ts[idx], self._px[idx]

    def slice_range(self, start_ms: int, end_ms: int) -> list[tuple[int, float]]:
        lo = bisect.bisect_left(self._ts, start_ms)
        hi = bisect.bisect_right(self._ts, end_ms)
        return list(zip(self._ts[lo:hi], self._px[lo:hi]))

    def high_low(self, start_ms: int, end_ms: int) -> tuple[float | None, float | None]:
        rows = self.slice_range(start_ms, end_ms)
        if not rows:
            return None, None
        prices = [r[1] for r in rows]
        return max(prices), min(prices)

    def ret_bps(self, start_ms: int, end_ms: int) -> float | None:
        a = self.at_or_after(start_ms)
        b = self.at_or_after(end_ms)
        if not a or not b or not a[1]:
            return None
        return (b[1] - a[1]) / a[1] * 10_000.0


class LiqIndex:
    """Sorted (ts_ms, notional) pairs for O(log N) sum-to-date lookups."""

    def __init__(self, rows: list[tuple[int, float]]) -> None:
        self._ts: list[int] = [int(r[0]) for r in rows]
        self._notional: list[float] = [float(r[1]) for r in rows]
        # prefix sum for fast range sum
        self._prefix: list[float] = []
        total = 0.0
        for n in self._notional:
            total += n
            self._prefix.append(total)

    def sum_range(self, start_ms: int, end_ms: int) -> float:
        lo = bisect.bisect_left(self._ts, start_ms)
        hi = bisect.bisect_right(self._ts, end_ms)
        if lo >= hi:
            return 0.0
        total_hi = self._prefix[hi - 1]
        total_lo = self._prefix[lo - 1] if lo > 0 else 0.0
        return total_hi - total_lo


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_marks(src: sqlite3.Connection, symbol: str) -> MarkIndex:
    print(f"  Loading mark_prices for {symbol}...", flush=True)
    rows = src.execute(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? ORDER BY ts_ms",
        (symbol,),
    ).fetchall()
    print(f"    {len(rows):,} rows", flush=True)
    return MarkIndex(rows)


def load_liqs(src: sqlite3.Connection, symbol: str, side: str) -> LiqIndex:
    rows = src.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol=? AND side=? ORDER BY ts_ms",
        (symbol, side),
    ).fetchall()
    return LiqIndex(rows)


def agg_count_range(src: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> int:
    row = src.execute(
        "SELECT COUNT(*) FROM agg_trades WHERE symbol=? AND ts_ms>=? AND ts_ms<=?",
        (symbol, start_ms, end_ms),
    ).fetchone()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# Cluster extraction
# ---------------------------------------------------------------------------

def extract_clusters(
    src: sqlite3.Connection,
    symbol: str,
    liq_side: str,
    threshold: float,
    bucket_sec: int = BUCKET_SEC,
    min_gap_sec: int = MIN_GAP_SEC,
) -> list[dict[str, Any]]:
    rows = src.execute(
        """
        SELECT CAST(ts_ms / ? AS INTEGER) AS bucket,
               MIN(ts_ms) AS first_ts_ms,
               MAX(ts_ms) AS last_ts_ms,
               COUNT(*) AS liq_count,
               SUM(notional) AS total_notional,
               MAX(notional) AS max_notional,
               MAX(price) AS max_price,
               MIN(price) AS min_price
        FROM liquidations
        WHERE symbol=? AND side=?
        GROUP BY bucket
        HAVING SUM(notional) >= ?
        ORDER BY first_ts_ms ASC
        """,
        (bucket_sec * 1000, symbol, liq_side, threshold),
    ).fetchall()

    events: list[dict[str, Any]] = []
    last_kept_ms = -(10 ** 18)
    prev_candidate_ms: int | None = None
    prev_kept_ms: int | None = None

    for bucket, first_ts, last_ts, count, total, max_notional, max_price, min_price in rows:
        first_ts = int(first_ts)
        if first_ts - last_kept_ms < min_gap_sec * 1000:
            prev_candidate_ms = first_ts
            continue
        duration_sec = max(1.0, (int(last_ts) - first_ts) / 1000.0)
        max_share_pct = (float(max_notional) / float(total) * 100.0) if total else 0.0
        inter_candidate = None if prev_candidate_ms is None else (first_ts - prev_candidate_ms) / 1000.0
        inter_kept = None if prev_kept_ms is None else (first_ts - prev_kept_ms) / 1000.0
        events.append({
            "event_id": f"{symbol}_{liq_side}_{int(bucket)}",
            "symbol": symbol,
            "liq_side": liq_side,
            "bucket": int(bucket),
            "event_ts_ms": first_ts,
            "event_utc": iso(first_ts),
            "cluster_window_sec": bucket_sec,
            "cluster_start_ts_ms": first_ts,
            "cluster_end_ts_ms": int(last_ts),
            "cluster_duration_sec": duration_sec,
            "cluster_count": int(count),
            "cluster_notional": float(total),
            "cluster_max_notional": float(max_notional),
            "cluster_max_price": float(max_price),
            "cluster_min_price": float(min_price),
            "cluster_intensity_notional_per_sec": float(total) / duration_sec,
            "inter_candidate_gap_sec": inter_candidate,
            "inter_kept_gap_sec": inter_kept,
            # geometry fields (inline)
            "cluster_liq_count": int(count),
            "max_single_liq_share": max_share_pct,
            "intensity_per_sec": float(total) / duration_sec,
            "inter_cluster_gap_sec": inter_kept,
            "shape_label": shape_label(duration_sec, max_share_pct),
        })
        last_kept_ms = first_ts
        prev_candidate_ms = first_ts
        prev_kept_ms = first_ts

    return events


# ---------------------------------------------------------------------------
# Feature enrichment
# ---------------------------------------------------------------------------

def enrich_event(
    event: dict[str, Any],
    marks_own: MarkIndex,
    marks_btc: MarkIndex,
    marks_eth: MarkIndex,
    marks_sol: MarkIndex,
    liqs_buy: LiqIndex,
    liqs_sell: LiqIndex,
    src: sqlite3.Connection,
    symbol: str,
) -> dict[str, Any]:
    ts = event["event_ts_ms"]
    row = dict(event)

    # Momentum returns
    row["symbol_pre_1m_bps"]  = marks_own.ret_bps(ts - 60_000,  ts)
    row["symbol_pre_5m_bps"]  = marks_own.ret_bps(ts - 300_000, ts)
    row["symbol_pre_15m_bps"] = marks_own.ret_bps(ts - 900_000, ts)
    row["btc_pre_1m_bps"]     = marks_btc.ret_bps(ts - 60_000,  ts)
    row["btc_pre_5m_bps"]     = marks_btc.ret_bps(ts - 300_000, ts)
    row["btc_pre_15m_bps"]    = marks_btc.ret_bps(ts - 900_000, ts)
    row["eth_pre_15m_bps"]    = marks_eth.ret_bps(ts - 900_000, ts)
    row["sol_pre_15m_bps"]    = marks_sol.ret_bps(ts - 900_000, ts)

    # Day context (no-lookahead)
    d_start = day_start_ms(ts)
    open_row = marks_own.at_or_after(d_start)
    cur_row  = marks_own.at_or_before(ts)
    if open_row and cur_row and open_row[1]:
        row["day_trend_bps"] = (cur_row[1] - open_row[1]) / open_row[1] * 10_000.0
        high, low = marks_own.high_low(d_start, ts)
        row["day_range_bps"] = (high - low) / low * 10_000.0 if (high and low) else None
    else:
        row["day_trend_bps"] = None
        row["day_range_bps"] = None

    row["day_buy_liq_notional"]  = liqs_buy.sum_range(d_start, ts)
    row["day_sell_liq_notional"] = liqs_sell.sum_range(d_start, ts)
    row["day_agg_count"]         = agg_count_range(src, symbol, d_start, ts)

    return row


# ---------------------------------------------------------------------------
# Route simulation
# ---------------------------------------------------------------------------

def simulate_route(
    marks: MarkIndex,
    event: dict[str, Any],
    route: dict[str, Any],
) -> dict[str, Any] | None:
    direction  = route["direction"]
    delay_sec  = int(route["entry_delay_sec"])
    tp_bps     = float(route["tp_bps"])
    sl_bps     = float(route["sl_bps"])
    be_bps     = float(route["be_bps"])

    entry_target = event["event_ts_ms"] + delay_sec * 1000
    entry_row = marks.at_or_after(entry_target)
    if not entry_row:
        return None
    entry_ts, entry_price = entry_row

    path = marks.slice_range(entry_ts, entry_ts + MAX_HORIZON_SEC * 1000)
    if not path:
        return None

    be_active = False
    be_ts: int | None = None
    mfe = -1e9
    mae =  1e9
    time_to_mfe = 0.0
    tp_touch = sl_touch = False
    exit_reason = "TIME"
    exit_ts, exit_price = int(path[-1][0]), float(path[-1][1])

    for ts_ms, price in path:
        ts_ms = int(ts_ms)
        price = float(price)
        ret = ((price - entry_price) if direction == "LONG" else (entry_price - price)) / entry_price * 10_000.0
        if ret > mfe:
            mfe = ret
            time_to_mfe = (ts_ms - entry_ts) / 1000.0
        if ret < mae:
            mae = ret
        if ret >= tp_bps:
            tp_touch = True
        if ret <= -sl_bps:
            sl_touch = True
        if not be_active and ret >= be_bps:
            be_active = True
            be_ts = ts_ms
        if ret >= tp_bps:
            exit_reason, exit_ts, exit_price = "TP", ts_ms, price
            break
        if ret <= -sl_bps:
            exit_reason, exit_ts, exit_price = "SL", ts_ms, price
            break
        if be_active and ret <= 0:
            exit_reason, exit_ts, exit_price = "BE", ts_ms, price
            break

    gross = ((exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)) / entry_price * 10_000.0

    return {
        "event_id":        event["event_id"],
        "route_id":        route["route_id"],
        "direction":       direction,
        "entry_delay_sec": delay_sec,
        "tp_bps":          tp_bps,
        "sl_bps":          sl_bps,
        "be_bps":          be_bps,
        "max_horizon_sec": MAX_HORIZON_SEC,
        "entry_ts_ms":     entry_ts,
        "entry_utc":       iso(entry_ts),
        "entry_price":     entry_price,
        "exit_ts_ms":      exit_ts,
        "exit_utc":        iso(exit_ts),
        "exit_price":      exit_price,
        "exit_reason":     exit_reason,
        "gross_bps":       gross,
        "fee_bps":         FEE_BPS,
        "net_bps":         gross - FEE_BPS,
        "mfe_bps":         mfe,
        "mae_bps":         mae,
        "time_to_mfe_sec": time_to_mfe,
        "tp_touch":        int(tp_touch),
        "sl_touch":        int(sl_touch),
        "be_hit":          int(be_ts is not None),
        "be_ts_ms":        be_ts,
    }


# ---------------------------------------------------------------------------
# DB operations
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "event_id", "symbol", "liq_side", "bucket", "event_ts_ms", "event_utc",
    "cluster_window_sec", "cluster_start_ts_ms", "cluster_end_ts_ms",
    "cluster_duration_sec", "cluster_count", "cluster_notional",
    "cluster_max_notional", "cluster_max_price", "cluster_min_price",
    "cluster_intensity_notional_per_sec", "inter_candidate_gap_sec", "inter_kept_gap_sec",
    "symbol_pre_1m_bps", "symbol_pre_5m_bps", "symbol_pre_15m_bps",
    "btc_pre_1m_bps", "btc_pre_5m_bps", "btc_pre_15m_bps",
    "eth_pre_15m_bps", "sol_pre_15m_bps",
    "day_trend_bps", "day_range_bps", "day_buy_liq_notional", "day_sell_liq_notional", "day_agg_count",
    "cluster_liq_count", "max_single_liq_share", "intensity_per_sec",
    "inter_cluster_gap_sec", "shape_label",
]

LABEL_COLS = [
    "event_id", "route_id", "direction", "entry_delay_sec", "tp_bps", "sl_bps", "be_bps",
    "max_horizon_sec", "entry_ts_ms", "entry_utc", "entry_price",
    "exit_ts_ms", "exit_utc", "exit_price", "exit_reason",
    "gross_bps", "fee_bps", "net_bps", "mfe_bps", "mae_bps", "time_to_mfe_sec",
    "tp_touch", "sl_touch", "be_hit", "be_ts_ms",
]


def ensure_db_columns(fdb: sqlite3.Connection) -> None:
    existing = {r[1] for r in fdb.execute("PRAGMA table_info(liq_event_features)").fetchall()}
    extras = {
        "cluster_liq_count":    "INTEGER",
        "max_single_liq_share": "REAL",
        "intensity_per_sec":    "REAL",
        "inter_cluster_gap_sec":"REAL",
        "shape_label":          "TEXT",
    }
    for col, typ in extras.items():
        if col not in existing:
            fdb.execute(f"ALTER TABLE liq_event_features ADD COLUMN {col} {typ}")
    fdb.commit()


def upsert_features(fdb: sqlite3.Connection, features: list[dict]) -> int:
    placeholders = ",".join("?" for _ in FEATURE_COLS)
    fdb.executemany(
        f"INSERT OR REPLACE INTO liq_event_features ({','.join(FEATURE_COLS)}) VALUES ({placeholders})",
        [[row.get(c) for c in FEATURE_COLS] for row in features],
    )
    return len(features)


def upsert_labels(fdb: sqlite3.Connection, labels: list[dict]) -> int:
    placeholders = ",".join("?" for _ in LABEL_COLS)
    fdb.executemany(
        f"INSERT OR REPLACE INTO liq_event_outcome_labels ({','.join(LABEL_COLS)}) VALUES ({placeholders})",
        [[row.get(c) for c in LABEL_COLS] for row in labels],
    )
    return len(labels)


def delete_combo(fdb: sqlite3.Connection, symbol: str, side: str) -> None:
    fdb.execute(
        "DELETE FROM liq_event_outcome_labels WHERE event_id IN "
        "(SELECT event_id FROM liq_event_features WHERE symbol=? AND liq_side=?)",
        (symbol, side),
    )
    fdb.execute(
        "DELETE FROM liq_event_features WHERE symbol=? AND liq_side=?",
        (symbol, side),
    )


def route_summary(labels: list[dict]) -> list[dict]:
    out = []
    for route_id in sorted({row["route_id"] for row in labels}):
        rows = [row for row in labels if row["route_id"] == route_id]
        nets = [float(row["net_bps"]) for row in rows]
        if not nets:
            continue
        median_val = sorted(nets)[len(nets) // 2]
        out.append({
            "route_id": route_id,
            "n": len(nets),
            "median_net_bps": median_val,
            "mean_net_bps": sum(nets) / len(nets),
            "wr": sum(v > 0 for v in nets) / len(nets),
            "tp": sum(r["exit_reason"] == "TP" for r in rows),
            "be": sum(r["exit_reason"] == "BE" for r in rows),
            "sl": sum(r["exit_reason"] == "SL" for r in rows),
            "time": sum(r["exit_reason"] == "TIME" for r in rows),
        })
    return out


# ---------------------------------------------------------------------------
# Per-combo runner
# ---------------------------------------------------------------------------

def run_combo(
    src: sqlite3.Connection,
    fdb: sqlite3.Connection,
    symbol: str,
    liq_side: str,
    marks_cache: dict[str, MarkIndex],
    dry_run: bool,
    rebuild: bool,
) -> dict[str, Any]:
    t0 = time.time()
    threshold = THRESHOLDS[(symbol, liq_side)]
    routes = ROUTES[(symbol, liq_side)]
    print(f"\n--- {symbol} {liq_side}  threshold={threshold:,.0f}  routes={[r['route_id'] for r in routes]} ---", flush=True)

    # Load mark prices (reuse from cache)
    for sym in (symbol, "BTCUSDT", "ETHUSDT", "SOLUSDT"):
        if sym not in marks_cache:
            marks_cache[sym] = load_marks(src, sym)

    marks_own  = marks_cache[symbol]
    marks_btc  = marks_cache["BTCUSDT"]
    marks_eth  = marks_cache["ETHUSDT"]
    marks_sol  = marks_cache["SOLUSDT"]

    # Load liquidations for the event symbol (both sides for day context)
    print(f"  Loading liquidations for {symbol}...", flush=True)
    liqs_buy  = load_liqs(src, symbol, "BUY")
    liqs_sell = load_liqs(src, symbol, "SELL")
    print(f"    BUY={len(liqs_buy._ts):,}  SELL={len(liqs_sell._ts):,}", flush=True)

    # Extract clusters
    events = extract_clusters(src, symbol, liq_side, threshold)
    print(f"  Clusters extracted: {len(events)}", flush=True)
    if not events:
        return {"symbol": symbol, "liq_side": liq_side, "events": 0, "labels": 0,
                "routes": [], "min_utc": None, "max_utc": None, "elapsed_sec": time.time() - t0}

    # Enrich features
    print(f"  Enriching features...", flush=True)
    features = []
    for i, ev in enumerate(events):
        enriched = enrich_event(ev, marks_own, marks_btc, marks_eth, marks_sol,
                                liqs_buy, liqs_sell, src, symbol)
        features.append(enriched)
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(events)} enriched", flush=True)

    # Simulate routes
    print(f"  Simulating {len(routes)} routes × {len(features)} events...", flush=True)
    labels = []
    no_fill = 0
    for ev in features:
        for route in routes:
            label = simulate_route(marks_own, ev, route)
            if label:
                labels.append(label)
            else:
                no_fill += 1
    print(f"  Labels: {len(labels)}  no-fill: {no_fill}", flush=True)

    if dry_run:
        print(f"  [DRY RUN] would upsert {len(features)} features, {len(labels)} labels", flush=True)
    else:
        if rebuild:
            delete_combo(fdb, symbol, liq_side)
            print(f"  Deleted existing rows for {symbol} {liq_side}", flush=True)
        upsert_features(fdb, features)
        upsert_labels(fdb, labels)
        fdb.commit()
        print(f"  Upserted to DB.", flush=True)

    summaries = route_summary(labels)
    min_utc = features[0]["event_utc"] if features else None
    max_utc = features[-1]["event_utc"] if features else None
    return {
        "symbol": symbol,
        "liq_side": liq_side,
        "threshold": threshold,
        "events": len(features),
        "labels": len(labels),
        "no_fill": no_fill,
        "routes": summaries,
        "min_utc": min_utc,
        "max_utc": max_utc,
        "elapsed_sec": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_report(results: list[dict], db_size_bytes: int) -> None:
    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "results": results,
        "db_size_bytes": db_size_bytes,
        "db_size_mb": round(db_size_bytes / 1_048_576, 2),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# S34 Feature Factory Phase 2 — Multi-Symbol Expansion",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "Appends ETH SELL, SOL BUY, SOL SELL, BTC BUY, BTC SELL events to `data/s34_feature_factory.db`.",
        "Uses UPSERT — existing ETH BUY rows are not modified.",
        "",
        f"DB size after: {payload['db_size_mb']:.1f} MB",
        "",
        "## Results by Symbol-Side",
        "",
    ]

    for r in results:
        lines.append(f"### {r['symbol']} {r['liq_side']}")
        lines.append(f"- Threshold: {r['threshold']:,.0f}")
        lines.append(f"- Events: {r['events']}")
        lines.append(f"- Labels: {r['labels']}  no-fill: {r.get('no_fill', 0)}")
        lines.append(f"- Date range: {(r['min_utc'] or '-')[:10]} → {(r['max_utc'] or '-')[:10]}")
        lines.append(f"- Runtime: {r['elapsed_sec']}s")
        lines.append("")
        if r.get("routes"):
            lines.append("| Route | N | Median | WR | TP | SL | BE | TIME |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for s in r["routes"]:
                lines.append(
                    f"| {s['route_id']} | {s['n']} | {s['median_net_bps']:+.1f} | "
                    f"{s['wr']*100:.0f}% | {s['tp']} | {s['sl']} | {s['be']} | {s['time']} |"
                )
            lines.append("")

    lines += [
        "## Verification",
        "",
        "```sql",
        "SELECT symbol, liq_side, COUNT(*) FROM liq_event_features GROUP BY symbol, liq_side;",
        "```",
        "",
        "## Note",
        "",
        "All features in `liq_event_features` are signal-time only (no lookahead).",
        "Route outcomes live exclusively in `liq_event_outcome_labels`.",
        "_Read-only research DB expansion. No runner, config, or pre-reg changes made._",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S34 Feature Factory Phase 2 — multi-symbol expansion")
    p.add_argument("--symbols", default=",".join(sorted({s for s, _ in DEFAULT_COMBOS})),
                   help="Comma-separated symbols to process (default: ETHUSDT,SOLUSDT,BTCUSDT)")
    p.add_argument("--sides", default="BUY,SELL",
                   help="Comma-separated sides to process (default: BUY,SELL)")
    p.add_argument("--rebuild", action="store_true",
                   help="Delete existing rows for selected combos before inserting")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute everything but do not write to DB")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    sides   = [s.strip().upper() for s in args.sides.split(",") if s.strip()]

    combos = [(sym, side) for sym in symbols for side in sides if (sym, side) in ROUTES]
    if not combos:
        print("No valid combos. Available:", list(ROUTES.keys()))
        return

    # Exclude ETH BUY unless explicitly requested with --rebuild
    # (ETH BUY data already present — re-running is safe but unnecessary by default)
    if ("ETHUSDT", "BUY") in combos and not args.rebuild:
        print("Skipping ETHUSDT BUY (already in DB). Pass --rebuild to regenerate it.")
        combos = [c for c in combos if c != ("ETHUSDT", "BUY")]
    if not combos:
        print("Nothing to do.")
        return

    print(f"Combos to process: {combos}")
    print(f"Rebuild: {args.rebuild}  Dry-run: {args.dry_run}")
    print()

    src = sqlite3.connect(SOURCE_DB_URI, uri=True, timeout=30)
    src.execute("PRAGMA query_only=1")
    fdb = None if args.dry_run else sqlite3.connect(FEATURE_DB_PATH)
    if fdb:
        fdb.execute("PRAGMA journal_mode=wal")
        ensure_db_columns(fdb)

    marks_cache: dict[str, MarkIndex] = {}
    results = []
    t_total = time.time()

    for symbol, side in combos:
        result = run_combo(src, fdb, symbol, side, marks_cache, args.dry_run, args.rebuild)
        results.append(result)
        print(f"  Done: {symbol} {side} — {result['events']} events / {result['labels']} labels / {result['elapsed_sec']}s", flush=True)

    src.close()
    if fdb:
        fdb.close()

    db_size = FEATURE_DB_PATH.stat().st_size if FEATURE_DB_PATH.exists() else 0
    elapsed = round(time.time() - t_total, 1)
    print(f"\nTotal elapsed: {elapsed}s")
    print(f"DB size: {db_size / 1_048_576:.1f} MB")

    if not args.dry_run:
        write_report(results, db_size)
        print(f"\nJSON: {OUT_JSON}")
        print(f"MD  : {OUT_MD}")

    # Verification query
    if not args.dry_run and FEATURE_DB_PATH.exists():
        vdb = sqlite3.connect(FEATURE_DB_PATH)
        print("\nVerification — liq_event_features coverage:")
        for row in vdb.execute(
            "SELECT symbol, liq_side, COUNT(*), MIN(event_utc), MAX(event_utc) "
            "FROM liq_event_features GROUP BY symbol, liq_side ORDER BY symbol, liq_side"
        ).fetchall():
            print(f"  {row[0]:10s} {row[1]:4s}  N={row[2]:5d}  {str(row[3])[:10]} to {str(row[4])[:10]}")
        vdb.close()


if __name__ == "__main__":
    main()
