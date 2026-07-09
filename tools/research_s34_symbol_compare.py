import datetime as dt
import json
import math
import sqlite3
import statistics
from pathlib import Path

from ami.storage import production as PR
from ami.storage import research_reader as RR

SOURCE_DB = "file:data/microstructure.db?mode=ro"
SOURCE_DB_PATH = "data/microstructure.db"
OUT_JSON = Path("reports/research/s34/S34_SYMBOL_COMPARISON_BUY_200K.json")
OUT_MD = Path("reports/research/s34/S34_SYMBOL_COMPARISON_BUY_200K.md")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
LIQ_SIDE = "BUY"
CLUSTER_THRESHOLD = 200_000.0
BUCKET_SEC = 300
MIN_GAP_SEC = 900
MAX_HORIZON_SEC = 3600
FEE_BPS = 8.0

ROUTES = [
    {"route_id": "LONG_DELAY0_TP60", "direction": "LONG", "entry_delay_sec": 0, "tp_bps": 60.0},
    {"route_id": "LONG_DELAY60_TP120", "direction": "LONG", "entry_delay_sec": 60, "tp_bps": 120.0},
    {"route_id": "SHORT_DELAY0_TP40_CONTROL", "direction": "SHORT", "entry_delay_sec": 0, "tp_bps": 40.0},
]


def iso(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return dt.datetime.fromtimestamp(ts_ms / 1000, dt.timezone.utc).isoformat()


def day_start_ms(ts_ms: int) -> int:
    day = dt.datetime.fromtimestamp(ts_ms / 1000, dt.timezone.utc).date()
    return int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp() * 1000)


def mark_at(con: sqlite3.Connection, symbol: str, ts_ms: int, before: bool = False):
    # OUT-OF-SCOPE for RANGE-READ V4: ASOF-style point lookup (ORDER BY
    # ts_ms ASC/DESC LIMIT 1) -- belongs to the ASOF track's
    # lookup_latest_at_or_before, not the range-read helper this gate
    # migrates. Left on direct SQL deliberately.
    op = "<=" if before else ">="
    order = "desc" if before else "asc"
    return con.execute(
        f"""
        select ts_ms, mark_price
        from mark_prices
        where symbol=? and ts_ms {op} ?
        order by ts_ms {order}
        limit 1
        """,
        (symbol, ts_ms),
    ).fetchone()


def ret_bps(con: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int):
    start = mark_at(con, symbol, start_ms, before=False)
    end = mark_at(con, symbol, end_ms, before=True)
    if not start or not end or not start[1]:
        return None
    return (float(end[1]) - float(start[1])) / float(start[1]) * 10000.0


def _day_high_low(con: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int):
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `_day_high_low_v2` (BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-
    CONSUMER-MIGRATION-V4). No longer called by `day_so_far`. Returns
    (max, min) mark_price over the inclusive window; (None, None) for an
    empty window (SQL MAX/MIN over no rows)."""
    return con.execute(
        """
        select max(mark_price), min(mark_price)
        from mark_prices
        where symbol=? and ts_ms>=? and ts_ms<=?
        """,
        (symbol, start_ms, end_ms),
    ).fetchone()


def _day_high_low_v2(root, symbol: str, start_ms: int, end_ms: int, source_db_path=None):
    """Reader-backed replacement for `_day_high_low`, via `plan_read`/
    `execute_read`. `symbol` is a genuine runtime parameter. Inclusive
    upper bound reproduced with `end_ms+1` (exact for integer ts_ms).
    `mark_price` is non-nullable, but None values are skipped defensively
    for exact SQL MAX/MIN parity anyway; an empty window yields
    (None, None), matching SQL's MAX/MIN over no rows."""
    plan = RR.plan_read(root, table="mark_prices", symbol=symbol, start_ms=int(start_ms), end_ms=int(end_ms) + 1)
    result = RR.execute_read(plan, columns=("mark_price",), source_db_path=source_db_path)
    hi = lo = None
    for (mp,) in result.iter_rows():
        if mp is None:
            continue
        v = float(mp)
        hi = v if hi is None or v > hi else hi
        lo = v if lo is None or v < lo else lo
    return (hi, lo)


def _day_agg_count(con: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> int:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `_day_agg_count_v2` (same gate). COUNT(*) of agg_trades over the
    inclusive window."""
    return con.execute(
        """
        select count(*)
        from agg_trades
        where symbol=? and ts_ms>=? and ts_ms<=?
        """,
        (symbol, start_ms, end_ms),
    ).fetchone()[0]


def _day_agg_count_v2(root, symbol: str, start_ms: int, end_ms: int, source_db_path=None) -> int:
    """Reader-backed replacement for `_day_agg_count`, via `plan_read`/
    `execute_read`. `symbol` is a genuine runtime parameter. Inclusive
    upper bound reproduced with `end_ms+1`; the streamed row count equals
    COUNT(*)."""
    plan = RR.plan_read(root, table="agg_trades", symbol=symbol, start_ms=int(start_ms), end_ms=int(end_ms) + 1)
    result = RR.execute_read(plan, columns=("ts_ms",), source_db_path=source_db_path)
    return sum(1 for _ in result.iter_rows())


def day_so_far(con: sqlite3.Connection, symbol: str, ts_ms: int, root, source_db_path=None) -> dict:
    start_ms = day_start_ms(ts_ms)
    open_row = mark_at(con, symbol, start_ms, before=False)
    cur_row = mark_at(con, symbol, ts_ms, before=True)
    if not open_row or not cur_row or not open_row[1]:
        return {}
    # Migrated to the reader (allowlisted mark_prices / agg_trades):
    high, low = _day_high_low_v2(root, symbol, start_ms, ts_ms, source_db_path=source_db_path)
    # OUT-OF-ALLOWLIST (liquidations has no archive partition / reader
    # support): left on direct SQL deliberately.
    buy_liq = con.execute(
        """
        select coalesce(sum(notional), 0)
        from liquidations
        where symbol=? and side='BUY' and ts_ms>=? and ts_ms<=?
        """,
        (symbol, start_ms, ts_ms),
    ).fetchone()[0]
    agg_count = _day_agg_count_v2(root, symbol, start_ms, ts_ms, source_db_path=source_db_path)
    return {
        "day_trend_bps": (float(cur_row[1]) - float(open_row[1])) / float(open_row[1]) * 10000.0,
        "day_range_bps": (float(high) - float(low)) / float(low) * 10000.0 if high and low else None,
        "day_buy_liq_notional": float(buy_liq or 0.0),
        "day_agg_count": int(agg_count or 0),
    }


def load_clusters(con: sqlite3.Connection, symbol: str, root, source_db_path=None) -> list[dict]:
    # OUT-OF-SCOPE for RANGE-READ V4: `liquidations` is an out-of-allowlist
    # table (no archive partition / reader support). Left on direct SQL.
    rows = con.execute(
        """
        select cast(ts_ms / ? as integer) as bucket,
               min(ts_ms) as first_ts_ms,
               max(ts_ms) as last_ts_ms,
               count(*) as liq_count,
               sum(notional) as cluster_notional,
               max(notional) as max_notional,
               max(price) as max_price,
               min(price) as min_price
        from liquidations
        where symbol=? and side=?
        group by bucket
        having sum(notional)>=?
        order by first_ts_ms asc
        """,
        (BUCKET_SEC * 1000, symbol, LIQ_SIDE, CLUSTER_THRESHOLD),
    ).fetchall()
    events = []
    last_signal_ms = -10**18
    previous_candidate_ts = None
    previous_kept_ts = None
    for row in rows:
        bucket, first_ts, last_ts, count, total, max_notional, max_price, min_price = row
        first_ts = int(first_ts)
        if first_ts - last_signal_ms < MIN_GAP_SEC * 1000:
            previous_candidate_ts = first_ts
            continue
        duration_sec = max(1.0, (int(last_ts) - first_ts) / 1000.0)
        event = {
            "event_id": f"{symbol}_{LIQ_SIDE}_{int(bucket)}",
            "symbol": symbol,
            "liq_side": LIQ_SIDE,
            "bucket": int(bucket),
            "event_ts_ms": first_ts,
            "event_utc": iso(first_ts),
            "cluster_start_ts_ms": first_ts,
            "cluster_end_ts_ms": int(last_ts),
            "cluster_duration_sec": duration_sec,
            "cluster_count": int(count or 0),
            "cluster_notional": float(total or 0.0),
            "cluster_max_notional": float(max_notional or 0.0),
            "cluster_max_price": float(max_price or 0.0),
            "cluster_min_price": float(min_price or 0.0),
            "cluster_intensity_notional_per_sec": float(total or 0.0) / duration_sec,
            "inter_candidate_gap_sec": None if previous_candidate_ts is None else (first_ts - previous_candidate_ts) / 1000.0,
            "inter_kept_gap_sec": None if previous_kept_ts is None else (first_ts - previous_kept_ts) / 1000.0,
            "symbol_pre_15m_bps": ret_bps(con, symbol, first_ts - 900_000, first_ts),
            "btc_pre_15m_bps": ret_bps(con, "BTCUSDT", first_ts - 900_000, first_ts),
        }
        event.update(day_so_far(con, symbol, first_ts, root, source_db_path=source_db_path))
        events.append(event)
        last_signal_ms = first_ts
        previous_candidate_ts = first_ts
        previous_kept_ts = first_ts
    return events


def _horizon_marks(con: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> list[tuple]:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `_horizon_marks_v2` (BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-
    CONSUMER-MIGRATION-V4). No longer called by `simulate_route`."""
    return con.execute(
        """
        select ts_ms, mark_price
        from mark_prices
        where symbol=? and ts_ms>=? and ts_ms<=?
        order by ts_ms
        """,
        (symbol, start_ms, end_ms),
    ).fetchall()


def _horizon_marks_v2(root, symbol: str, start_ms: int, end_ms: int, source_db_path=None) -> list[tuple]:
    """Reader-backed replacement for `_horizon_marks`, via `plan_read`/
    `execute_read`. `symbol` is a genuine runtime parameter (SYMBOLS:
    BTCUSDT/ETHUSDT/SOLUSDT). Inclusive upper bound reproduced with
    `end_ms+1` (exact for integer ts_ms). Streams in canonical
    `(ts_ms ASC, id ASC)` order -- a refinement of the oracle's
    `ORDER BY ts_ms` that yields an identical ts_ms sequence."""
    plan = RR.plan_read(root, table="mark_prices", symbol=symbol, start_ms=int(start_ms), end_ms=int(end_ms) + 1)
    result = RR.execute_read(plan, columns=("ts_ms", "mark_price"), source_db_path=source_db_path)
    return list(result.iter_rows())


def simulate_route(con: sqlite3.Connection, event: dict, route: dict, root, source_db_path=None) -> dict | None:
    direction = str(route["direction"])
    entry_delay_sec = int(route["entry_delay_sec"])
    tp_bps = float(route["tp_bps"])
    sl_bps = 40.0
    be_bps = 30.0
    entry_target_ms = int(event["event_ts_ms"]) + entry_delay_sec * 1000
    entry = mark_at(con, event["symbol"], entry_target_ms, before=False)
    if not entry:
        return None
    entry_ts, entry_price = int(entry[0]), float(entry[1])
    marks = _horizon_marks_v2(root, event["symbol"], entry_ts, entry_ts + MAX_HORIZON_SEC * 1000, source_db_path=source_db_path)
    if not marks:
        return None
    be_active = False
    mfe = -1e9
    mae = 1e9
    time_to_mfe = 0.0
    exit_reason = "TIME"
    exit_ts, exit_price = int(marks[-1][0]), float(marks[-1][1])
    for ts_ms, price in marks:
        ts_ms = int(ts_ms)
        price = float(price)
        ret = (price - entry_price) / entry_price * 10000.0
        if direction == "SHORT":
            ret = -ret
        if ret > mfe:
            mfe = ret
            time_to_mfe = (ts_ms - entry_ts) / 1000.0
        if ret < mae:
            mae = ret
        if not be_active and ret >= be_bps:
            be_active = True
        if ret >= tp_bps:
            exit_reason = "TP"
            exit_ts, exit_price = ts_ms, price
            break
        if ret <= -sl_bps:
            exit_reason = "SL"
            exit_ts, exit_price = ts_ms, price
            break
        if be_active and ret <= 0:
            exit_reason = "BE"
            exit_ts, exit_price = ts_ms, price
            break
    gross_bps = (exit_price - entry_price) / entry_price * 10000.0
    if direction == "SHORT":
        gross_bps = -gross_bps
    return {
        "route_id": route["route_id"],
        "direction": direction,
        "entry_delay_sec": entry_delay_sec,
        "tp_bps": tp_bps,
        "entry_ts_ms": entry_ts,
        "entry_price": entry_price,
        "exit_ts_ms": exit_ts,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_bps": gross_bps,
        "net_bps": gross_bps - FEE_BPS,
        "mfe_bps": mfe,
        "mae_bps": mae,
        "time_to_mfe_sec": time_to_mfe,
    }


def median(vals: list[float]) -> float:
    return statistics.median(vals) if vals else 0.0


def summarize(events: list[dict], labels: list[dict], route_id: str, predicate=None) -> dict:
    filtered = []
    for event, label in zip(events, labels):
        if label["route_id"] != route_id:
            continue
        if predicate is not None and not predicate(event):
            continue
        filtered.append((event, label))
    vals = [float(label["net_bps"]) for _, label in filtered]
    days = sorted({dt.datetime.fromtimestamp(int(event["event_ts_ms"]) / 1000, dt.timezone.utc).date().isoformat() for event, _ in filtered})
    exits = {k: sum(1 for _, label in filtered if label["exit_reason"] == k) for k in ["TP", "BE", "SL", "TIME"]}
    return {
        "n": len(vals),
        "days": len(days),
        "mean": sum(vals) / len(vals) if vals else 0.0,
        "median": median(vals),
        "cum": sum(vals),
        "wr": sum(1 for v in vals if v > 0) / len(vals) if vals else 0.0,
        "exits": exits,
        "mean_mfe": sum(float(label["mfe_bps"]) for _, label in filtered) / len(filtered) if filtered else 0.0,
        "mean_mae": sum(float(label["mae_bps"]) for _, label in filtered) / len(filtered) if filtered else 0.0,
    }


def main() -> None:
    # `con` stays for the still-direct ASOF (mark_at/ret_bps) and
    # out-of-allowlist (liquidations) reads; the mark_prices/agg_trades
    # range reads moved to the reader (via `root`/SOURCE_DB_PATH).
    con = sqlite3.connect(SOURCE_DB, uri=True)
    root, _ = PR.resolve_production_root()
    payload = {"generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "symbols": {}}
    for symbol in SYMBOLS:
        events = load_clusters(con, symbol, root, source_db_path=SOURCE_DB_PATH)
        labels = []
        for event in events:
            for route in ROUTES:
                label = simulate_route(con, event, route, root, source_db_path=SOURCE_DB_PATH)
                if label:
                    labels.append({"event_id": event["event_id"], **label})
        route_summaries = {}
        for route in ROUTES:
            route_id = route["route_id"]
            route_summaries[route_id] = {
                "base": summarize(events, [label for label in labels if label["route_id"] == route_id], route_id),
                "cluster_500k_daytrend0": summarize(
                    events,
                    [label for label in labels if label["route_id"] == route_id],
                    route_id,
                    lambda event: float(event.get("cluster_notional") or 0.0) >= 500_000.0
                    and float(event.get("day_trend_bps") or -1e9) >= 0.0,
                ),
            }
        payload["symbols"][symbol] = {
            "event_count": len(events),
            "first_event_utc": events[0]["event_utc"] if events else None,
            "last_event_utc": events[-1]["event_utc"] if events else None,
            "median_cluster_notional": median([event["cluster_notional"] for event in events]),
            "median_cluster_count": median([event["cluster_count"] for event in events]),
            "median_duration_sec": median([event["cluster_duration_sec"] for event in events]),
            "route_summaries": route_summaries,
        }
    con.close()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S34 Symbol Comparison - BUY 200K",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "Scope: BTCUSDT / ETHUSDT / SOLUSDT BUY liquidation clusters >= 200K, 300s bucket, 900s minimum gap.",
        "",
        "Costs are the simplified Phase-1 model: net = gross - 8 bps. This is descriptive research, not a runner change.",
        "",
        "## Coverage",
        "",
        "| Symbol | Events | First | Last | Median Notional | Median Count | Median Duration sec |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for symbol, row in payload["symbols"].items():
        lines.append(
            f"| {symbol} | {row['event_count']} | {row['first_event_utc']} | {row['last_event_utc']} | "
            f"{row['median_cluster_notional']:,.0f} | {row['median_cluster_count']:.0f} | {row['median_duration_sec']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Base Route Comparison",
            "",
            "| Symbol | Route | N | Days | Mean | Median | Cum | WR | TP/BE/SL/TIME | Mean MFE | Mean MAE |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for symbol, row in payload["symbols"].items():
        for route in ROUTES:
            s = row["route_summaries"][route["route_id"]]["base"]
            exits = s["exits"]
            lines.append(
                f"| {symbol} | {route['route_id']} | {s['n']} | {s['days']} | {s['mean']:+.2f} | {s['median']:+.2f} | "
                f"{s['cum']:+.2f} | {s['wr']*100:.1f}% | {exits['TP']}/{exits['BE']}/{exits['SL']}/{exits['TIME']} | "
                f"{s['mean_mfe']:+.2f} | {s['mean_mae']:+.2f} |"
            )
    lines.extend(
        [
            "",
            "## 500K + Day-Trend >= 0 Slice",
            "",
            "| Symbol | Route | N | Days | Mean | Median | Cum | WR | TP/BE/SL/TIME |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for symbol, row in payload["symbols"].items():
        for route in ROUTES:
            s = row["route_summaries"][route["route_id"]]["cluster_500k_daytrend0"]
            exits = s["exits"]
            lines.append(
                f"| {symbol} | {route['route_id']} | {s['n']} | {s['days']} | {s['mean']:+.2f} | {s['median']:+.2f} | "
                f"{s['cum']:+.2f} | {s['wr']*100:.1f}% | {exits['TP']}/{exits['BE']}/{exits['SL']}/{exits['TIME']} |"
            )
    lines.extend(
        [
            "",
            "## Observations",
            "",
            "- This report compares the same mechanical BUY-liq cluster setup across symbols.",
            "- It does not account for historical bid/ask real-fill parity by symbol.",
            "- It does not promote, kill, or modify any runner rule.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(json.dumps(payload["symbols"], indent=2)[:4000])


if __name__ == "__main__":
    main()
