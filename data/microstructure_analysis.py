# data/microstructure_analysis.py — Microstructure Event Detectors
#
# Reads from the SQLite database populated by microstructure_collector.py
# and detects actionable events:
#   A. Liquidation cascades
#   B. Aggression imbalance
#   C. Large trade clusters
#
# Usage:
#   python -m data.microstructure_analysis --db-path data/microstructure.db
#   python -m data.microstructure_analysis --db-path data/microstructure.db --last-hours 4

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LiquidationCascade:
    ts_ms: int
    symbol: str
    direction: str      # "LONG_LIQ" (sell-side liqs) or "SHORT_LIQ" (buy-side liqs)
    count: int
    total_notional: float
    window_sec: int
    severity: str       # "SMALL", "MEDIUM", "LARGE"

    @property
    def timestamp(self) -> str:
        return datetime.fromtimestamp(self.ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class AggressionImbalance:
    ts_ms: int
    symbol: str
    direction: str      # "BUY_DOMINANT" or "SELL_DOMINANT"
    ratio: float        # buy_vol / sell_vol
    buy_volume: float
    sell_volume: float
    window_sec: int

    @property
    def timestamp(self) -> str:
        return datetime.fromtimestamp(self.ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class LargeTradeCluster:
    ts_ms: int
    symbol: str
    direction: str      # "BUY" or "SELL"
    count: int
    total_notional: float
    avg_size_percentile: float
    window_sec: int

    @property
    def timestamp(self) -> str:
        return datetime.fromtimestamp(self.ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ═══════════════════════════════════════════════════════════════════════
# DETECTORS
# ═══════════════════════════════════════════════════════════════════════

class LiquidationCascadeDetector:
    """Detect clusters of same-direction liquidations within time windows."""

    def __init__(self, windows_sec: List[int] = None, min_count: int = 3):
        self.windows_sec = windows_sec or [10, 30, 60]
        self.min_count = min_count

    def detect(self, conn: sqlite3.Connection, symbol: str,
               start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> List[LiquidationCascade]:
        """Detect liquidation cascades from the database."""
        query = "SELECT ts_ms, side, notional FROM liquidations WHERE symbol = ?"
        params: list = [symbol]
        if start_ms:
            query += " AND ts_ms >= ?"
            params.append(start_ms)
        if end_ms:
            query += " AND ts_ms <= ?"
            params.append(end_ms)
        query += " ORDER BY ts_ms"

        rows = conn.execute(query, params).fetchall()
        if len(rows) < self.min_count:
            return []

        df = pd.DataFrame(rows, columns=["ts_ms", "side", "notional"])
        cascades = []

        for window_sec in self.windows_sec:
            window_ms = window_sec * 1000
            for side in ["SELL", "BUY"]:
                side_df = df[df["side"] == side].reset_index(drop=True)
                if len(side_df) < self.min_count:
                    continue

                # Sliding window: for each liquidation, count how many same-direction
                # liquidations occur within the next window_ms
                i = 0
                while i < len(side_df):
                    anchor_ts = int(side_df.iloc[i]["ts_ms"])
                    window_end = anchor_ts + window_ms

                    # Count events in window
                    mask = (side_df["ts_ms"] >= anchor_ts) & (side_df["ts_ms"] <= window_end)
                    cluster = side_df[mask]

                    if len(cluster) >= self.min_count:
                        total_notional = float(cluster["notional"].sum())

                        if total_notional >= 1_000_000:
                            severity = "LARGE"
                        elif total_notional >= 100_000:
                            severity = "MEDIUM"
                        else:
                            severity = "SMALL"

                        direction = "LONG_LIQ" if side == "SELL" else "SHORT_LIQ"
                        cascades.append(LiquidationCascade(
                            ts_ms=anchor_ts, symbol=symbol, direction=direction,
                            count=len(cluster), total_notional=total_notional,
                            window_sec=window_sec, severity=severity,
                        ))
                        # Skip past this cluster to avoid overlapping detections
                        i = int(cluster.index[-1]) + 1
                    else:
                        i += 1

        # Sort by timestamp
        cascades.sort(key=lambda c: c.ts_ms)
        return cascades


class AggressionImbalanceDetector:
    """Detect periods where taker-buy or taker-sell volume is dominant."""

    def __init__(self, windows_sec: List[int] = None, imbalance_threshold: float = 2.0):
        self.windows_sec = windows_sec or [60, 300, 900]  # 1min, 5min, 15min
        self.imbalance_threshold = imbalance_threshold

    def detect(self, conn: sqlite3.Connection, symbol: str,
               start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> List[AggressionImbalance]:
        """Detect aggression imbalances from agg_trades."""
        query = "SELECT ts_ms, notional, is_buyer_maker FROM agg_trades WHERE symbol = ?"
        params: list = [symbol]
        if start_ms:
            query += " AND ts_ms >= ?"
            params.append(start_ms)
        if end_ms:
            query += " AND ts_ms <= ?"
            params.append(end_ms)
        query += " ORDER BY ts_ms"

        rows = conn.execute(query, params).fetchall()
        if not rows:
            return []

        df = pd.DataFrame(rows, columns=["ts_ms", "notional", "is_buyer_maker"])
        # is_buyer_maker=0 means buyer is taker (aggressive buy)
        # is_buyer_maker=1 means seller is taker (aggressive sell)
        df["buy_vol"] = df["notional"] * (1 - df["is_buyer_maker"])
        df["sell_vol"] = df["notional"] * df["is_buyer_maker"]

        imbalances = []

        for window_sec in self.windows_sec:
            window_ms = window_sec * 1000
            # Sample at regular intervals (every window_sec/2 for overlap)
            step_ms = window_ms // 2
            if len(df) == 0:
                continue

            min_ts = int(df["ts_ms"].min())
            max_ts = int(df["ts_ms"].max())

            ts = min_ts
            while ts <= max_ts:
                mask = (df["ts_ms"] >= ts) & (df["ts_ms"] < ts + window_ms)
                window_df = df[mask]

                if len(window_df) < 10:  # need minimum trades
                    ts += step_ms
                    continue

                buy_vol = float(window_df["buy_vol"].sum())
                sell_vol = float(window_df["sell_vol"].sum())

                if sell_vol > 0:
                    ratio = buy_vol / sell_vol
                elif buy_vol > 0:
                    ratio = float("inf")
                else:
                    ts += step_ms
                    continue

                if ratio >= self.imbalance_threshold:
                    imbalances.append(AggressionImbalance(
                        ts_ms=ts + window_ms // 2,  # center of window
                        symbol=symbol, direction="BUY_DOMINANT",
                        ratio=ratio, buy_volume=buy_vol, sell_volume=sell_vol,
                        window_sec=window_sec,
                    ))
                elif ratio <= 1.0 / self.imbalance_threshold:
                    imbalances.append(AggressionImbalance(
                        ts_ms=ts + window_ms // 2,
                        symbol=symbol, direction="SELL_DOMINANT",
                        ratio=ratio, buy_volume=buy_vol, sell_volume=sell_vol,
                        window_sec=window_sec,
                    ))

                ts += step_ms

        imbalances.sort(key=lambda x: x.ts_ms)
        return imbalances


class LargeTradeDetector:
    """Detect individual large trades and clusters of large same-direction trades."""

    def __init__(self, percentile: float = 99.0, cluster_window_sec: int = 60,
                 min_cluster_count: int = 3, lookback_trades: int = 10000):
        self.percentile = percentile
        self.cluster_window_sec = cluster_window_sec
        self.min_cluster_count = min_cluster_count
        self.lookback_trades = lookback_trades

    def detect(self, conn: sqlite3.Connection, symbol: str,
               start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> List[LargeTradeCluster]:
        """Detect clusters of large trades."""
        query = "SELECT ts_ms, notional, is_buyer_maker FROM agg_trades WHERE symbol = ?"
        params: list = [symbol]
        if start_ms:
            query += " AND ts_ms >= ?"
            params.append(start_ms)
        if end_ms:
            query += " AND ts_ms <= ?"
            params.append(end_ms)
        query += " ORDER BY ts_ms"

        rows = conn.execute(query, params).fetchall()
        if len(rows) < 100:
            return []

        df = pd.DataFrame(rows, columns=["ts_ms", "notional", "is_buyer_maker"])

        # Compute rolling percentile threshold using a rolling window
        # For efficiency, compute threshold on the full dataset first
        threshold = float(np.percentile(df["notional"].values, self.percentile))

        # Flag large trades
        large = df[df["notional"] >= threshold].copy()
        if len(large) < self.min_cluster_count:
            return []

        # Direction: is_buyer_maker=0 -> aggressive BUY, is_buyer_maker=1 -> aggressive SELL
        large["direction"] = large["is_buyer_maker"].map({0: "BUY", 1: "SELL"})

        clusters = []
        window_ms = self.cluster_window_sec * 1000

        for direction in ["BUY", "SELL"]:
            dir_trades = large[large["direction"] == direction].reset_index(drop=True)
            if len(dir_trades) < self.min_cluster_count:
                continue

            i = 0
            while i < len(dir_trades):
                anchor_ts = int(dir_trades.iloc[i]["ts_ms"])
                window_end = anchor_ts + window_ms

                mask = (dir_trades["ts_ms"] >= anchor_ts) & (dir_trades["ts_ms"] <= window_end)
                cluster = dir_trades[mask]

                if len(cluster) >= self.min_cluster_count:
                    total_notional = float(cluster["notional"].sum())
                    # Compute average percentile rank of trades in cluster
                    avg_pctile = float(np.mean([
                        float(np.searchsorted(
                            np.sort(df["notional"].values), n
                        )) / len(df) * 100
                        for n in cluster["notional"].values
                    ]))

                    clusters.append(LargeTradeCluster(
                        ts_ms=anchor_ts, symbol=symbol, direction=direction,
                        count=len(cluster), total_notional=total_notional,
                        avg_size_percentile=avg_pctile, window_sec=self.cluster_window_sec,
                    ))
                    i = int(cluster.index[-1]) + 1
                else:
                    i += 1

        clusters.sort(key=lambda c: c.ts_ms)
        return clusters


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════

def print_db_summary(conn: sqlite3.Connection):
    """Print summary of data in the database."""
    print(f"\n{'=' * 80}")
    print(f"DATABASE SUMMARY")
    print(f"{'=' * 80}")

    for table in ["agg_trades", "mark_prices", "liquidations"]:
        row = conn.execute(f"SELECT COUNT(*), MIN(ts_ms), MAX(ts_ms) FROM {table}").fetchone()
        count, min_ts, max_ts = row
        if count > 0 and min_ts and max_ts:
            start = datetime.fromtimestamp(min_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            end = datetime.fromtimestamp(max_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            hours = (max_ts - min_ts) / 3_600_000
            print(f"  {table:<15}: {count:>12,} rows  |  {start} to {end}  ({hours:.1f}h)")
        else:
            print(f"  {table:<15}: {count:>12,} rows")

    # Per-symbol breakdown for trades
    rows = conn.execute(
        "SELECT symbol, COUNT(*), SUM(notional) FROM agg_trades GROUP BY symbol"
    ).fetchall()
    if rows:
        print(f"\n  Trade volume by symbol:")
        for sym, count, total_notional in rows:
            print(f"    {sym}: {count:,} trades, ${total_notional:,.0f} notional")

    # Liquidation summary
    rows = conn.execute(
        "SELECT symbol, side, COUNT(*), SUM(notional) FROM liquidations GROUP BY symbol, side"
    ).fetchall()
    if rows:
        print(f"\n  Liquidations by symbol/side:")
        for sym, side, count, total_notional in rows:
            print(f"    {sym} {side}: {count:,} events, ${total_notional:,.0f} notional")

    print(f"{'=' * 80}")


def run_analysis(db_path: str, symbol: str, last_hours: Optional[float] = None):
    """Run all detectors on a symbol and print results."""
    conn = sqlite3.connect(db_path)

    start_ms = None
    if last_hours:
        import time
        start_ms = int((time.time() - last_hours * 3600) * 1000)

    print(f"\n{'=' * 90}")
    print(f"MICROSTRUCTURE ANALYSIS: {symbol}")
    if last_hours:
        print(f"  (Last {last_hours:.0f} hours)")
    print(f"{'=' * 90}")

    # A. Liquidation cascades
    print(f"\n  --- LIQUIDATION CASCADES ---")
    liq_detector = LiquidationCascadeDetector(windows_sec=[10, 30, 60], min_count=3)
    cascades = liq_detector.detect(conn, symbol, start_ms=start_ms)
    if cascades:
        for c in cascades[:50]:  # limit output
            print(f"    {c.timestamp}  {c.direction:<12}  "
                  f"{c.count} liqs in {c.window_sec}s  "
                  f"${c.total_notional:>12,.0f}  [{c.severity}]")
        print(f"  Total cascades: {len(cascades)}")
    else:
        print(f"    No cascades detected")

    # B. Aggression imbalance
    print(f"\n  --- AGGRESSION IMBALANCE ---")
    agg_detector = AggressionImbalanceDetector(windows_sec=[60, 300, 900], imbalance_threshold=2.0)
    imbalances = agg_detector.detect(conn, symbol, start_ms=start_ms)
    if imbalances:
        for imb in imbalances[:50]:
            print(f"    {imb.timestamp}  {imb.direction:<16}  "
                  f"ratio={imb.ratio:>5.1f}  "
                  f"window={imb.window_sec}s  "
                  f"buy=${imb.buy_volume:>10,.0f}  sell=${imb.sell_volume:>10,.0f}")
        print(f"  Total imbalances: {len(imbalances)}")

        # Breakdown by window
        for ws in [60, 300, 900]:
            ws_imbs = [i for i in imbalances if i.window_sec == ws]
            buy_dom = sum(1 for i in ws_imbs if i.direction == "BUY_DOMINANT")
            sell_dom = len(ws_imbs) - buy_dom
            print(f"    {ws}s window: {len(ws_imbs)} events (buy-dom: {buy_dom}, sell-dom: {sell_dom})")
    else:
        print(f"    No significant imbalances detected")

    # C. Large trade clusters
    print(f"\n  --- LARGE TRADE CLUSTERS ---")
    large_detector = LargeTradeDetector(percentile=99.0, cluster_window_sec=60, min_cluster_count=3)
    clusters = large_detector.detect(conn, symbol, start_ms=start_ms)
    if clusters:
        for c in clusters[:50]:
            print(f"    {c.timestamp}  {c.direction:<6}  "
                  f"{c.count} trades in {c.window_sec}s  "
                  f"${c.total_notional:>12,.0f}  "
                  f"avg pctile={c.avg_size_percentile:.1f}%")
        print(f"  Total clusters: {len(clusters)}")
        buy_clusters = sum(1 for c in clusters if c.direction == "BUY")
        sell_clusters = len(clusters) - buy_clusters
        print(f"    BUY clusters: {buy_clusters}, SELL clusters: {sell_clusters}")
    else:
        print(f"    No large trade clusters detected")

    conn.close()

    return {
        "cascades": cascades,
        "imbalances": imbalances,
        "clusters": clusters,
    }


def main():
    parser = argparse.ArgumentParser(description="Microstructure Event Analysis")
    parser.add_argument("--db-path", type=str, default="data/microstructure.db",
                        help="SQLite database path")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT",
                        help="Comma-separated symbols to analyze")
    parser.add_argument("--last-hours", type=float, default=None,
                        help="Only analyze last N hours (default: all data)")
    args = parser.parse_args()

    db_path = args.db_path
    if not Path(db_path).exists():
        print(f"ERROR: Database not found at {db_path}")
        print(f"Run the collector first: python -m data.microstructure_collector")
        return

    conn = sqlite3.connect(db_path)
    print_db_summary(conn)
    conn.close()

    symbols = [s.strip() for s in args.symbols.split(",")]
    for symbol in symbols:
        run_analysis(db_path, symbol, last_hours=args.last_hours)


if __name__ == "__main__":
    main()
