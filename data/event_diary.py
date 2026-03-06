# data/event_diary.py — Automated Event Diary for Microstructure Analysis
#
# Runs alongside the collector. Every 60 seconds, scans the SQLite DB for
# "interesting moments" and logs them with price snapshots to a CSV.
#
# NOT a signal generator — just a structured observation log for later study.
#
# Events detected:
#   1. Price move >0.5% in last 5 minutes
#   2. Liquidation cluster: 3+ same direction in 60s
#   3. Aggression imbalance: ratio >2.5:1 over last 5 minutes
#   4. Large trade: single trade >$500k notional
#
# For each event, snapshots price at 0/+5/+15/+30/+60 minutes.
# The forward prices are filled in retroactively on subsequent scans.
#
# Usage:
#   python -m data.event_diary --db-path data/microstructure.db
#   python -m data.event_diary --db-path data/microstructure.db --csv-path data/event_diary.csv

from __future__ import annotations

import argparse
import csv
import os
import signal
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.stdout.reconfigure(encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════
# CSV SCHEMA
# ═══════════════════════════════════════════════════════════════════════

CSV_HEADERS = [
    "event_id",         # unique: {type}_{symbol}_{ts_ms}
    "ts_ms",            # event timestamp (ms)
    "timestamp",        # human readable UTC
    "symbol",
    "event_type",       # PRICE_MOVE, LIQ_CASCADE, AGG_IMBALANCE, LARGE_TRADE
    "direction",        # UP/DOWN, LONG_LIQ/SHORT_LIQ, BUY/SELL
    "magnitude",        # % move, notional, ratio — depends on type
    "detail",           # extra info (e.g. "3 liqs in 60s, $1.2M")
    "price_at_event",   # mark price at event time
    "price_5m",         # price 5 minutes after (filled retroactively)
    "price_15m",        # price 15 minutes after
    "price_30m",        # price 30 minutes after
    "price_60m",        # price 60 minutes after
    "move_5m_pct",      # % change at +5m
    "move_15m_pct",     # % change at +15m
    "move_30m_pct",     # % change at +30m
    "move_60m_pct",     # % change at +60m
    "liq_count_5m",     # liquidation count in surrounding 5 min window
    "agg_ratio_5m",     # aggression ratio (buy/sell) at event time over 5 min
]


class EventDiary:
    """Automated event observation diary."""

    FORWARD_HORIZONS_MS = {
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "60m": 60 * 60 * 1000,
    }

    def __init__(self, db_path: str, csv_path: str, scan_interval: float = 60.0,
                 symbols: List[str] = None):
        self.db_path = db_path
        self.csv_path = csv_path
        self.scan_interval = scan_interval
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]

        self._running = False
        self._seen_events: Set[str] = set()  # event_ids already logged
        self._pending_fills: Dict[str, dict] = {}  # events needing forward price fills

        # Load existing events from CSV to avoid duplicates
        self._load_existing()

    def _load_existing(self):
        """Load existing event IDs and pending fills from CSV."""
        if not Path(self.csv_path).exists():
            return

        try:
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    eid = row.get("event_id", "")
                    self._seen_events.add(eid)
                    # Check if forward prices still need filling
                    if not row.get("price_60m"):
                        self._pending_fills[eid] = row
        except Exception as e:
            print(f"  [WARN] Could not load existing diary: {e}")

    def _ensure_csv(self):
        """Create CSV with headers if it doesn't exist."""
        if not Path(self.csv_path).exists():
            Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)

    def _append_event(self, event: dict):
        """Append a single event row to the CSV."""
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([event.get(h, "") for h in CSV_HEADERS])

    def _rewrite_csv(self, all_events: List[dict]):
        """Rewrite entire CSV with updated forward prices."""
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
            for event in all_events:
                writer.writerow([event.get(h, "") for h in CSV_HEADERS])

    def run(self):
        """Main scan loop."""
        self._running = True
        self._ensure_csv()

        print(f"\n{'=' * 70}")
        print(f"EVENT DIARY")
        print(f"{'=' * 70}")
        print(f"  DB: {self.db_path}")
        print(f"  CSV: {self.csv_path}")
        print(f"  Symbols: {self.symbols}")
        print(f"  Scan interval: {self.scan_interval}s")
        print(f"  Loaded {len(self._seen_events)} existing events, "
              f"{len(self._pending_fills)} pending fills")
        print(f"{'=' * 70}")
        print(f"  Running... Press Ctrl+C to stop.\n", flush=True)

        while self._running:
            try:
                self._scan()
            except Exception as e:
                print(f"  [ERROR] Scan failed: {e}", flush=True)

            # Sleep in small increments so we can respond to stop signal
            for _ in range(int(self.scan_interval)):
                if not self._running:
                    break
                time.sleep(1)

    def _scan(self):
        """Run one scan cycle."""
        conn = sqlite3.connect(self.db_path)
        now_ms = int(time.time() * 1000)
        new_events = []

        for symbol in self.symbols:
            # 1. Price moves
            new_events.extend(self._detect_price_moves(conn, symbol, now_ms))
            # 2. Liquidation clusters
            new_events.extend(self._detect_liq_clusters(conn, symbol, now_ms))
            # 3. Aggression imbalance
            new_events.extend(self._detect_agg_imbalance(conn, symbol, now_ms))
            # 4. Large trades
            new_events.extend(self._detect_large_trades(conn, symbol, now_ms))

        # Add context to new events
        for event in new_events:
            eid = event["event_id"]
            if eid in self._seen_events:
                continue

            # Get surrounding context
            event.update(self._get_context(conn, event["symbol"], int(event["ts_ms"])))
            self._append_event(event)
            self._seen_events.add(eid)
            self._pending_fills[eid] = event

            ts_str = event.get("timestamp", "")
            print(f"  [{ts_str}] {event['symbol']} {event['event_type']} "
                  f"{event['direction']} mag={event['magnitude']} "
                  f"price={event.get('price_at_event', '?')}", flush=True)

        # Fill forward prices for pending events
        filled = self._fill_forward_prices(conn, now_ms)
        if filled > 0:
            # Rewrite CSV with updated fills
            self._rewrite_all(conn)

        conn.close()

    def _detect_price_moves(self, conn: sqlite3.Connection, symbol: str,
                             now_ms: int) -> List[dict]:
        """Detect >0.5% price moves in last 5 minutes."""
        window_ms = 5 * 60 * 1000
        start = now_ms - window_ms
        # Only look at the last scan interval to avoid duplicates
        lookback = now_ms - int(self.scan_interval * 1000) - window_ms

        rows = conn.execute(
            "SELECT ts_ms, mark_price FROM mark_prices "
            "WHERE symbol = ? AND ts_ms >= ? AND ts_ms <= ? ORDER BY ts_ms",
            (symbol, start, now_ms)
        ).fetchall()

        if len(rows) < 10:
            return []

        events = []
        first_price = rows[0][1]
        last_price = rows[-1][1]
        move_pct = (last_price - first_price) / first_price * 100

        if abs(move_pct) >= 0.5:
            ts_ms = rows[-1][0]
            eid = f"PRICE_MOVE_{symbol}_{ts_ms // 60000}"  # dedupe per minute
            if eid not in self._seen_events:
                direction = "UP" if move_pct > 0 else "DOWN"
                events.append({
                    "event_id": eid,
                    "ts_ms": ts_ms,
                    "timestamp": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol,
                    "event_type": "PRICE_MOVE",
                    "direction": direction,
                    "magnitude": f"{move_pct:+.2f}%",
                    "detail": f"{move_pct:+.2f}% in 5min ({first_price:.2f} -> {last_price:.2f})",
                    "price_at_event": f"{last_price:.2f}",
                })

        return events

    def _detect_liq_clusters(self, conn: sqlite3.Connection, symbol: str,
                              now_ms: int) -> List[dict]:
        """Detect 3+ same-direction liquidations within 60 seconds."""
        # Look at liquidations in the last scan interval + buffer
        lookback = now_ms - int(self.scan_interval * 1000 * 2)

        rows = conn.execute(
            "SELECT ts_ms, side, notional FROM liquidations "
            "WHERE symbol = ? AND ts_ms >= ? ORDER BY ts_ms",
            (symbol, lookback)
        ).fetchall()

        if len(rows) < 3:
            return []

        events = []
        window_ms = 60 * 1000

        for side in ["SELL", "BUY"]:
            side_rows = [(ts, notional) for ts, s, notional in rows if s == side]
            if len(side_rows) < 3:
                continue

            i = 0
            while i < len(side_rows):
                anchor_ts = side_rows[i][0]
                cluster = [(ts, n) for ts, n in side_rows if anchor_ts <= ts <= anchor_ts + window_ms]

                if len(cluster) >= 3:
                    total_notional = sum(n for _, n in cluster)
                    direction = "LONG_LIQ" if side == "SELL" else "SHORT_LIQ"
                    eid = f"LIQ_CASCADE_{symbol}_{anchor_ts // 60000}"

                    if eid not in self._seen_events:
                        # Get price at event time
                        price_row = conn.execute(
                            "SELECT mark_price FROM mark_prices "
                            "WHERE symbol = ? AND ts_ms <= ? ORDER BY ts_ms DESC LIMIT 1",
                            (symbol, anchor_ts)
                        ).fetchone()
                        price = price_row[0] if price_row else 0

                        events.append({
                            "event_id": eid,
                            "ts_ms": anchor_ts,
                            "timestamp": datetime.fromtimestamp(anchor_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                            "symbol": symbol,
                            "event_type": "LIQ_CASCADE",
                            "direction": direction,
                            "magnitude": f"${total_notional:,.0f}",
                            "detail": f"{len(cluster)} liqs in 60s, ${total_notional:,.0f} notional",
                            "price_at_event": f"{price:.2f}",
                        })

                    # Skip past cluster
                    i += len(cluster)
                else:
                    i += 1

        return events

    def _detect_agg_imbalance(self, conn: sqlite3.Connection, symbol: str,
                               now_ms: int) -> List[dict]:
        """Detect aggression imbalance >2.5:1 over last 5 minutes."""
        window_ms = 5 * 60 * 1000
        start = now_ms - window_ms

        rows = conn.execute(
            "SELECT notional, is_buyer_maker FROM agg_trades "
            "WHERE symbol = ? AND ts_ms >= ? AND ts_ms <= ?",
            (symbol, start, now_ms)
        ).fetchall()

        if len(rows) < 50:
            return []

        buy_vol = sum(n for n, m in rows if m == 0)
        sell_vol = sum(n for n, m in rows if m == 1)

        if sell_vol == 0 or buy_vol == 0:
            return []

        ratio = buy_vol / sell_vol
        threshold = 2.5

        events = []

        if ratio >= threshold or ratio <= 1.0 / threshold:
            eid = f"AGG_IMBALANCE_{symbol}_{now_ms // 60000}"
            if eid not in self._seen_events:
                direction = "BUY_DOMINANT" if ratio >= threshold else "SELL_DOMINANT"

                price_row = conn.execute(
                    "SELECT mark_price FROM mark_prices "
                    "WHERE symbol = ? AND ts_ms <= ? ORDER BY ts_ms DESC LIMIT 1",
                    (symbol, now_ms)
                ).fetchone()
                price = price_row[0] if price_row else 0

                events.append({
                    "event_id": eid,
                    "ts_ms": now_ms,
                    "timestamp": datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol,
                    "event_type": "AGG_IMBALANCE",
                    "direction": direction,
                    "magnitude": f"{ratio:.1f}:1",
                    "detail": f"buy=${buy_vol:,.0f} sell=${sell_vol:,.0f} ratio={ratio:.2f}",
                    "price_at_event": f"{price:.2f}",
                })

        return events

    def _detect_large_trades(self, conn: sqlite3.Connection, symbol: str,
                              now_ms: int) -> List[dict]:
        """Detect individual trades >$500k notional."""
        lookback = now_ms - int(self.scan_interval * 1000 * 2)

        rows = conn.execute(
            "SELECT ts_ms, price, quantity, notional, is_buyer_maker FROM agg_trades "
            "WHERE symbol = ? AND ts_ms >= ? AND notional >= 500000 ORDER BY ts_ms",
            (symbol, lookback)
        ).fetchall()

        events = []
        for ts, price, qty, notional, is_buyer_maker in rows:
            direction = "BUY" if is_buyer_maker == 0 else "SELL"
            eid = f"LARGE_TRADE_{symbol}_{ts}"

            if eid not in self._seen_events:
                events.append({
                    "event_id": eid,
                    "ts_ms": ts,
                    "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol,
                    "event_type": "LARGE_TRADE",
                    "direction": direction,
                    "magnitude": f"${notional:,.0f}",
                    "detail": f"{qty:.4f} @ {price:.2f} = ${notional:,.0f}",
                    "price_at_event": f"{price:.2f}",
                })

        return events

    def _get_context(self, conn: sqlite3.Connection, symbol: str, ts_ms: int) -> dict:
        """Get surrounding context for an event."""
        context = {}

        # Liquidation count in surrounding 5 min
        window = 5 * 60 * 1000
        row = conn.execute(
            "SELECT COUNT(*) FROM liquidations "
            "WHERE symbol = ? AND ts_ms >= ? AND ts_ms <= ?",
            (symbol, ts_ms - window, ts_ms + window)
        ).fetchone()
        context["liq_count_5m"] = row[0] if row else 0

        # Aggression ratio over last 5 min
        rows = conn.execute(
            "SELECT notional, is_buyer_maker FROM agg_trades "
            "WHERE symbol = ? AND ts_ms >= ? AND ts_ms <= ?",
            (symbol, ts_ms - window, ts_ms)
        ).fetchall()

        if rows:
            buy_vol = sum(n for n, m in rows if m == 0)
            sell_vol = sum(n for n, m in rows if m == 1)
            if sell_vol > 0:
                context["agg_ratio_5m"] = f"{buy_vol / sell_vol:.2f}"
            elif buy_vol > 0:
                context["agg_ratio_5m"] = "inf"
            else:
                context["agg_ratio_5m"] = "1.00"
        else:
            context["agg_ratio_5m"] = ""

        return context

    def _fill_forward_prices(self, conn: sqlite3.Connection, now_ms: int) -> int:
        """Fill in forward price snapshots for pending events."""
        filled_count = 0

        to_remove = []
        for eid, event in self._pending_fills.items():
            event_ts = int(event["ts_ms"])
            symbol = event["symbol"]
            all_filled = True

            for label, offset_ms in self.FORWARD_HORIZONS_MS.items():
                price_key = f"price_{label}"
                move_key = f"move_{label}_pct"

                if event.get(price_key):
                    continue  # already filled

                target_ts = event_ts + offset_ms
                if target_ts > now_ms:
                    all_filled = False
                    continue  # not enough time has passed

                # Find closest mark price to target time
                row = conn.execute(
                    "SELECT mark_price FROM mark_prices "
                    "WHERE symbol = ? AND ts_ms >= ? AND ts_ms <= ? "
                    "ORDER BY ABS(ts_ms - ?) LIMIT 1",
                    (symbol, target_ts - 5000, target_ts + 5000, target_ts)
                ).fetchone()

                if row:
                    future_price = row[0]
                    event[price_key] = f"{future_price:.2f}"

                    price_at_event = float(event.get("price_at_event", 0))
                    if price_at_event > 0:
                        move = (future_price - price_at_event) / price_at_event * 100
                        event[move_key] = f"{move:+.3f}"

                    filled_count += 1
                else:
                    all_filled = False

            if all_filled:
                to_remove.append(eid)

        for eid in to_remove:
            del self._pending_fills[eid]

        return filled_count

    def _rewrite_all(self, conn: sqlite3.Connection):
        """Rewrite CSV with all events including updated fills."""
        if not Path(self.csv_path).exists():
            return

        # Read all existing events
        all_events = []
        try:
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    eid = row.get("event_id", "")
                    # Use pending_fills version if available (has updated prices)
                    if eid in self._pending_fills:
                        all_events.append(self._pending_fills[eid])
                    else:
                        all_events.append(row)
        except Exception:
            return

        self._rewrite_csv(all_events)

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Microstructure Event Diary")
    parser.add_argument("--db-path", type=str, default="data/microstructure.db",
                        help="SQLite database path (default: data/microstructure.db)")
    parser.add_argument("--csv-path", type=str, default="data/event_diary.csv",
                        help="Output CSV path (default: data/event_diary.csv)")
    parser.add_argument("--scan-interval", type=float, default=60.0,
                        help="Seconds between scans (default: 60)")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT",
                        help="Comma-separated symbols (default: BTCUSDT,ETHUSDT)")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    diary = EventDiary(
        db_path=args.db_path,
        csv_path=args.csv_path,
        scan_interval=args.scan_interval,
        symbols=symbols,
    )

    def signal_handler(sig, frame):
        print("\n  Stopping diary...", flush=True)
        diary.stop()

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        diary.run()
    except KeyboardInterrupt:
        pass

    print("  Diary stopped.")


if __name__ == "__main__":
    main()
