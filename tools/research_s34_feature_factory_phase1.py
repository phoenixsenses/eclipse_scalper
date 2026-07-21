import datetime as dt
import json
import sqlite3
from pathlib import Path


SOURCE_DB = "file:data/microstructure.db?mode=ro"
OUT_DB = Path("data/s34_feature_factory.db")
OUT_JSON = Path("reports/research/s34/S34_FEATURE_FACTORY_PHASE1_ETH_BUY_200K.json")
OUT_MD = Path("reports/research/s34/S34_FEATURE_FACTORY_PHASE1_ETH_BUY_200K.md")

SYMBOL = "ETHUSDT"
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
    d = dt.datetime.fromtimestamp(ts_ms / 1000, dt.timezone.utc).date()
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000)


def mark_at(con: sqlite3.Connection, symbol: str, ts_ms: int, before: bool = False):
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
    end = mark_at(con, symbol, end_ms, before=False)
    if not start or not end or not start[1]:
        return None
    return (end[1] - start[1]) / start[1] * 10000.0


def day_so_far(con: sqlite3.Connection, symbol: str, ts_ms: int):
    start_ms = day_start_ms(ts_ms)
    open_row = mark_at(con, symbol, start_ms, before=False)
    cur_row = mark_at(con, symbol, ts_ms, before=True)
    if not open_row or not cur_row or not open_row[1]:
        return {}
    high, low = con.execute(
        """
        select max(mark_price), min(mark_price)
        from mark_prices
        where symbol=? and ts_ms>=? and ts_ms<=?
        """,
        (symbol, start_ms, ts_ms),
    ).fetchone()
    buy_liq = con.execute(
        """
        select coalesce(sum(notional), 0)
        from liquidations
        where symbol=? and side='BUY' and ts_ms>=? and ts_ms<=?
        """,
        (symbol, start_ms, ts_ms),
    ).fetchone()[0]
    sell_liq = con.execute(
        """
        select coalesce(sum(notional), 0)
        from liquidations
        where symbol=? and side='SELL' and ts_ms>=? and ts_ms<=?
        """,
        (symbol, start_ms, ts_ms),
    ).fetchone()[0]
    agg_count = con.execute(
        """
        select count(*)
        from agg_trades
        where symbol=? and ts_ms>=? and ts_ms<=?
        """,
        (symbol, start_ms, ts_ms),
    ).fetchone()[0]
    return {
        "day_trend_bps": (cur_row[1] - open_row[1]) / open_row[1] * 10000.0,
        "day_range_bps": (high - low) / low * 10000.0 if high and low else None,
        "day_buy_liq_notional": float(buy_liq or 0.0),
        "day_sell_liq_notional": float(sell_liq or 0.0),
        "day_agg_count": int(agg_count or 0),
    }


def load_clusters(con: sqlite3.Connection):
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
        (BUCKET_SEC * 1000, SYMBOL, LIQ_SIDE, CLUSTER_THRESHOLD),
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
        events.append(
            {
                "event_id": f"{SYMBOL}_{LIQ_SIDE}_{int(bucket)}",
                "symbol": SYMBOL,
                "liq_side": LIQ_SIDE,
                "bucket": int(bucket),
                "event_ts_ms": first_ts,
                "event_utc": iso(first_ts),
                "cluster_window_sec": BUCKET_SEC,
                "cluster_start_ts_ms": first_ts,
                "cluster_end_ts_ms": int(last_ts),
                "cluster_duration_sec": duration_sec,
                "cluster_count": int(count or 0),
                "cluster_notional": float(total or 0.0),
                "cluster_max_notional": float(max_notional or 0.0),
                "cluster_max_price": float(max_price or 0.0),
                "cluster_min_price": float(min_price or 0.0),
                "cluster_intensity_notional_per_sec": float(total or 0.0) / duration_sec,
                "inter_candidate_gap_sec": None
                if previous_candidate_ts is None
                else (first_ts - previous_candidate_ts) / 1000.0,
                "inter_kept_gap_sec": None
                if previous_kept_ts is None
                else (first_ts - previous_kept_ts) / 1000.0,
            }
        )
        last_signal_ms = first_ts
        previous_candidate_ts = first_ts
        previous_kept_ts = first_ts
    return events


def enrich_features(con: sqlite3.Connection, events: list[dict]):
    enriched = []
    for event in events:
        ts = event["event_ts_ms"]
        row = dict(event)
        row.update(
            {
                "symbol_pre_1m_bps": ret_bps(con, SYMBOL, ts - 60_000, ts),
                "symbol_pre_5m_bps": ret_bps(con, SYMBOL, ts - 300_000, ts),
                "symbol_pre_15m_bps": ret_bps(con, SYMBOL, ts - 900_000, ts),
                "btc_pre_1m_bps": ret_bps(con, "BTCUSDT", ts - 60_000, ts),
                "btc_pre_5m_bps": ret_bps(con, "BTCUSDT", ts - 300_000, ts),
                "btc_pre_15m_bps": ret_bps(con, "BTCUSDT", ts - 900_000, ts),
                "eth_pre_15m_bps": ret_bps(con, "ETHUSDT", ts - 900_000, ts),
                "sol_pre_15m_bps": ret_bps(con, "SOLUSDT", ts - 900_000, ts),
            }
        )
        row.update(day_so_far(con, SYMBOL, ts))
        enriched.append(row)
    return enriched


def simulate_route(con: sqlite3.Connection, event: dict, route: dict):
    direction = route["direction"]
    entry_delay_sec = int(route["entry_delay_sec"])
    tp_bps = float(route["tp_bps"])
    sl_bps = 40.0
    be_bps = 30.0
    entry_target_ms = event["event_ts_ms"] + entry_delay_sec * 1000
    entry = mark_at(con, event["symbol"], entry_target_ms, before=False)
    if not entry:
        return None
    entry_ts, entry_price = int(entry[0]), float(entry[1])
    rows = con.execute(
        """
        select ts_ms, mark_price
        from mark_prices
        where symbol=? and ts_ms>=? and ts_ms<=?
        order by ts_ms
        """,
        (event["symbol"], entry_ts, entry_ts + MAX_HORIZON_SEC * 1000),
    ).fetchall()
    if not rows:
        return None

    be_active = False
    be_ts = None
    mfe = -1e9
    mae = 1e9
    time_to_mfe = 0.0
    tp_touch = False
    sl_touch = False
    exit_reason = "TIME"
    exit_ts, exit_price = int(rows[-1][0]), float(rows[-1][1])

    for ts_ms, price in rows:
        ts_ms = int(ts_ms)
        price = float(price)
        if direction == "LONG":
            ret = (price - entry_price) / entry_price * 10000.0
        else:
            ret = (entry_price - price) / entry_price * 10000.0
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
            exit_reason = "TP"
            exit_ts = ts_ms
            exit_price = price
            break
        if ret <= -sl_bps:
            exit_reason = "SL"
            exit_ts = ts_ms
            exit_price = price
            break
        if be_active and ret <= 0:
            exit_reason = "BE"
            exit_ts = ts_ms
            exit_price = price
            break

    if direction == "LONG":
        gross = (exit_price - entry_price) / entry_price * 10000.0
    else:
        gross = (entry_price - exit_price) / entry_price * 10000.0

    return {
        "event_id": event["event_id"],
        "route_id": route["route_id"],
        "direction": direction,
        "entry_delay_sec": entry_delay_sec,
        "tp_bps": tp_bps,
        "sl_bps": sl_bps,
        "be_bps": be_bps,
        "max_horizon_sec": MAX_HORIZON_SEC,
        "entry_ts_ms": entry_ts,
        "entry_utc": iso(entry_ts),
        "entry_price": entry_price,
        "exit_ts_ms": exit_ts,
        "exit_utc": iso(exit_ts),
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_bps": gross,
        "fee_bps": FEE_BPS,
        "net_bps": gross - FEE_BPS,
        "mfe_bps": mfe,
        "mae_bps": mae,
        "time_to_mfe_sec": time_to_mfe,
        "tp_touch": int(tp_touch),
        "sl_touch": int(sl_touch),
        "be_hit": int(be_ts is not None),
        "be_ts_ms": be_ts,
    }


def write_db(features: list[dict], labels: list[dict]):
    OUT_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(OUT_DB)
    con.execute("pragma journal_mode=wal")
    con.executescript(
        """
        drop table if exists liq_event_features;
        drop table if exists liq_event_outcome_labels;
        drop table if exists feature_factory_metadata;

        create table liq_event_features (
            event_id text primary key,
            symbol text not null,
            liq_side text not null,
            bucket integer not null,
            event_ts_ms integer not null,
            event_utc text not null,
            cluster_window_sec integer not null,
            cluster_start_ts_ms integer not null,
            cluster_end_ts_ms integer not null,
            cluster_duration_sec real not null,
            cluster_count integer not null,
            cluster_notional real not null,
            cluster_max_notional real not null,
            cluster_max_price real not null,
            cluster_min_price real not null,
            cluster_intensity_notional_per_sec real not null,
            inter_candidate_gap_sec real,
            inter_kept_gap_sec real,
            symbol_pre_1m_bps real,
            symbol_pre_5m_bps real,
            symbol_pre_15m_bps real,
            btc_pre_1m_bps real,
            btc_pre_5m_bps real,
            btc_pre_15m_bps real,
            eth_pre_15m_bps real,
            sol_pre_15m_bps real,
            day_trend_bps real,
            day_range_bps real,
            day_buy_liq_notional real,
            day_sell_liq_notional real,
            day_agg_count integer
        );

        create table liq_event_outcome_labels (
            event_id text not null,
            route_id text not null,
            direction text not null,
            entry_delay_sec integer not null,
            tp_bps real not null,
            sl_bps real not null,
            be_bps real not null,
            max_horizon_sec integer not null,
            entry_ts_ms integer not null,
            entry_utc text not null,
            entry_price real not null,
            exit_ts_ms integer not null,
            exit_utc text not null,
            exit_price real not null,
            exit_reason text not null,
            gross_bps real not null,
            fee_bps real not null,
            net_bps real not null,
            mfe_bps real not null,
            mae_bps real not null,
            time_to_mfe_sec real not null,
            tp_touch integer not null,
            sl_touch integer not null,
            be_hit integer not null,
            be_ts_ms integer,
            primary key(event_id, route_id),
            foreign key(event_id) references liq_event_features(event_id)
        );

        create table feature_factory_metadata (
            key text primary key,
            value text not null
        );

        create index idx_features_symbol_ts on liq_event_features(symbol, event_ts_ms);
        create index idx_labels_route on liq_event_outcome_labels(route_id, net_bps);
        """
    )
    feature_cols = list(features[0].keys()) if features else []
    label_cols = list(labels[0].keys()) if labels else []
    if features:
        placeholders = ",".join("?" for _ in feature_cols)
        con.executemany(
            f"insert into liq_event_features ({','.join(feature_cols)}) values ({placeholders})",
            [[row.get(col) for col in feature_cols] for row in features],
        )
    if labels:
        placeholders = ",".join("?" for _ in label_cols)
        con.executemany(
            f"insert into liq_event_outcome_labels ({','.join(label_cols)}) values ({placeholders})",
            [[row.get(col) for col in label_cols] for row in labels],
        )
    metadata = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_db": "data/microstructure.db",
        "scope": "ETHUSDT BUY liquidation clusters >=200K, bucket 300s, min_gap 900s",
        "lookahead_guard": "liq_event_features contains only signal-time features; future/path outcomes live only in liq_event_outcome_labels",
        "routes": json.dumps(ROUTES, sort_keys=True),
    }
    con.executemany(
        "insert into feature_factory_metadata(key,value) values (?,?)",
        list(metadata.items()),
    )
    con.commit()
    con.close()


def summarize_labels(labels: list[dict]):
    out = []
    for route in ROUTES:
        route_labels = [row for row in labels if row["route_id"] == route["route_id"]]
        vals = [row["net_bps"] for row in route_labels]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        median = vals_sorted[len(vals_sorted) // 2] if len(vals_sorted) % 2 else (vals_sorted[len(vals_sorted)//2 - 1] + vals_sorted[len(vals_sorted)//2]) / 2
        out.append(
            {
                "route_id": route["route_id"],
                "n": len(vals),
                "mean_net_bps": sum(vals) / len(vals),
                "median_net_bps": median,
                "cum_net_bps": sum(vals),
                "wr": sum(v > 0 for v in vals) / len(vals),
                "tp": sum(row["exit_reason"] == "TP" for row in route_labels),
                "be": sum(row["exit_reason"] == "BE" for row in route_labels),
                "sl": sum(row["exit_reason"] == "SL" for row in route_labels),
                "time": sum(row["exit_reason"] == "TIME" for row in route_labels),
                "mean_mfe_bps": sum(row["mfe_bps"] for row in route_labels) / len(route_labels),
                "mean_mae_bps": sum(row["mae_bps"] for row in route_labels) / len(route_labels),
            }
        )
    return out


def write_reports(features: list[dict], labels: list[dict], summaries: list[dict]):
    payload = {"features": features, "summaries": summaries, "routes": ROUTES, "out_db": str(OUT_DB)}
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S34 Feature Factory Phase 1 - ETH BUY 200K",
        "",
        f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
        "Scope: ETHUSDT BUY liquidation clusters >= 200K, 300s bucket, 900s minimum gap.",
        "",
        "Output DB: `data/s34_feature_factory.db`",
        "",
        "## Lookahead Boundary",
        "",
        "- `liq_event_features`: signal-time/no-lookahead features only.",
        "- `liq_event_outcome_labels`: future path labels and route outcomes only.",
        "- Wait/confirmation returns are not stored in the feature table in Phase 1. They must be modeled through route `entry_delay_sec` or added later to a separate delayed-feature table.",
        "",
        "## Extraction Summary",
        "",
        f"- Feature rows: `{len(features)}`",
        f"- Outcome label rows: `{len(labels)}`",
        f"- Anchor routes: `{len(ROUTES)}`",
        "",
        "## Anchor Route Results",
        "",
        "| Route | N | Mean Net | Median Net | Cum Net | WR | TP | BE | SL | TIME | Mean MFE | Mean MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['route_id']} | {row['n']} | {row['mean_net_bps']:+.2f} | "
            f"{row['median_net_bps']:+.2f} | {row['cum_net_bps']:+.2f} | "
            f"{row['wr']*100:.1f}% | {row['tp']} | {row['be']} | {row['sl']} | {row['time']} | "
            f"{row['mean_mfe_bps']:+.2f} | {row['mean_mae_bps']:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Phase 1 Acceptance",
            "",
            "- Separate feature and label tables created.",
            "- Source `microstructure.db` read-only.",
            "- Feature table has no future path columns.",
            "- Only three anchor routes computed, avoiding the full combinatorial route explosion.",
            "",
            "## Read",
            "",
            "This is infrastructure, not a new trading decision. Use this DB as the base for a query layer. Do not promote a new paper variant from Phase 1 without outlier/day-spread checks and live-fill confirmation.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    src = sqlite3.connect(SOURCE_DB, uri=True, timeout=10)
    src.execute("pragma query_only=1")
    events = load_clusters(src)
    features = enrich_features(src, events)
    labels = []
    for event in features:
        for route in ROUTES:
            label = simulate_route(src, event, route)
            if label:
                labels.append(label)
    src.close()

    write_db(features, labels)
    summaries = summarize_labels(labels)
    write_reports(features, labels, summaries)

    print(OUT_MD)
    print(json.dumps({"feature_rows": len(features), "label_rows": len(labels), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
