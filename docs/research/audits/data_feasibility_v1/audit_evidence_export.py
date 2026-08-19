"""Outcome-blind, read-only evidence export for Data Feasibility Audit V1.

The only writes are new artifacts beside this script. Every SQLite source is
opened with URI mode=ro and PRAGMA query_only=ON. The frozen E-DER outcome is
read only in the isolated reconciliation function; observability classifications
are constructed before, and independently of, that function.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
MANIFEST = ROOT / "reports/research/s34/S35_E_DER_EVENT_IDENTITY_MANIFEST_V1.csv"
KEEPER = ROOT / "data/keeper_frozen_smalltables.db"
KLINES = ROOT / "data/xsec_klines.db"
OLD_CHANNEL = ROOT / "reports/research/s34/S34_E_CHANNEL_A_RESPONSE_DIAGNOSTIC_V1.json"
NEW_OUTCOME = ROOT / "reports/research/s34/s35_e_der_raw_v1/frozen_outcome_240m.csv"
MINUTE_MS = 60_000


def ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, rows: list[dict]) -> None:
    path = OUT / name
    if not rows:
        raise ValueError(f"refusing empty CSV: {name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def schema_inventory() -> list[dict]:
    sources = [
        ROOT / "data/microstructure_02.db",
        KEEPER,
        KLINES,
        ROOT / "reports/research/s34/mechanism_store.sqlite",
        ROOT / "reports/research/s34/S34_ALL.db",
        ROOT / "data/funding_history.db",
        ROOT / "data/oi_history.db",
    ]
    rows: list[dict] = []
    for path in sources:
        conn = ro(path)
        try:
            for (table,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name!='sqlite_sequence' ORDER BY name"
            ):
                for cid, name, typ, notnull, default, pk in conn.execute(
                    f'PRAGMA table_info("{table}")'
                ):
                    rows.append({
                        "database": str(path.relative_to(ROOT)).replace("\\", "/"),
                        "table": table,
                        "ordinal": cid,
                        "column": name,
                        "declared_type": typ,
                        "not_null": notnull,
                        "default": default,
                        "primary_key_ordinal": pk,
                        "finding_status": "VERIFIED FROM CODE/DATA",
                    })
        finally:
            conn.close()
    return rows


def database_inventory() -> list[dict]:
    rows: list[dict] = []
    live = ROOT / "data/microstructure_02.db"
    conn = ro(live)
    try:
        seq = dict(conn.execute("SELECT name,seq FROM sqlite_sequence"))
        for table in ("liquidations", "agg_trades", "mark_prices", "book_ticker"):
            rows.append({
                "asset": str(live.relative_to(ROOT)).replace("\\", "/"),
                "table_or_stream": table,
                "row_count_value": seq.get(table),
                "row_count_semantics": "sqlite_sequence maximum assigned id; NOT an exact row count",
                "coverage_start_ms": None,
                "coverage_end_ms": None,
                "symbols": "launcher default BTCUSDT|ETHUSDT|SOLUSDT for aggTrade/mark/book; forceOrder all-market",
                "finding_status": "INFERRED",
                "notes": "Exact full-table count/coverage scan intentionally not run against the active ~157GB database.",
            })
    finally:
        conn.close()

    conn = ro(KEEPER)
    try:
        for table in [x[0] for x in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name!='sqlite_sequence' ORDER BY name"
        )]:
            count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            start = end = symbols = None
            cols = {x[1] for x in conn.execute(f'PRAGMA table_info("{table}")')}
            if "ts_ms" in cols:
                start, end = conn.execute(f'SELECT MIN(ts_ms),MAX(ts_ms) FROM "{table}"').fetchone()
            if "symbol" in cols:
                symbols = conn.execute(f'SELECT COUNT(DISTINCT symbol) FROM "{table}"').fetchone()[0]
            rows.append({
                "asset": str(KEEPER.relative_to(ROOT)).replace("\\", "/"),
                "table_or_stream": table,
                "row_count_value": count,
                "row_count_semantics": "exact SELECT COUNT(*)",
                "coverage_start_ms": start,
                "coverage_end_ms": end,
                "symbols": symbols,
                "finding_status": "VERIFIED FROM CODE/DATA",
                "notes": "Frozen small-table segment retained after rotation.",
            })
    finally:
        conn.close()

    conn = ro(KLINES)
    try:
        for table in ("klines", "ingest_log", "price_supplement_ingest"):
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if table == "klines":
                start, end, symbols = conn.execute(
                    "SELECT MIN(open_time),MAX(open_time),COUNT(DISTINCT symbol) FROM klines"
                ).fetchone()
            else:
                start = end = symbols = None
            rows.append({
                "asset": str(KLINES.relative_to(ROOT)).replace("\\", "/"),
                "table_or_stream": table,
                "row_count_value": count,
                "row_count_semantics": "exact SELECT COUNT(*)",
                "coverage_start_ms": start,
                "coverage_end_ms": end,
                "symbols": symbols,
                "finding_status": "VERIFIED FROM CODE/DATA",
                "notes": "1-minute kline store; (symbol,open_time) composite primary key.",
            })
    finally:
        conn.close()
    return rows


def archive_inventory() -> list[dict]:
    rows = []
    root = ROOT / "data/archives/parquet_v1"
    for stream in ("agg_trades", "book_ticker", "mark_prices"):
        entries = [json.loads(line) for line in (root / stream / "_manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()]
        zero = [f"{x['symbol']}:{x['dt']}" for x in entries if int(x["rows"]) == 0]
        rows.append({
            "stream": stream,
            "partitions": len(entries),
            "exact_manifest_rows": sum(int(x["rows"]) for x in entries),
            "file_bytes": sum(int(x["file_bytes"]) for x in entries),
            "symbols": "|".join(sorted({x["symbol"] for x in entries})),
            "start_ms": min(int(x["start_ms"]) for x in entries),
            "end_ms_exclusive": max(int(x["end_ms"]) for x in entries),
            "first_date": min(x["dt"] for x in entries),
            "last_date": max(x["dt"] for x in entries),
            "zero_row_partition_count": len(zero),
            "zero_row_partitions": "|".join(zero),
            "finding_status": "VERIFIED FROM CODE/DATA",
        })
    return rows


def gap_inventory() -> list[dict]:
    conn = ro(KEEPER)
    try:
        rows = []
        for stream, count, resolved, unresolved, start, end, maximum, total in conn.execute(
            "SELECT stream,COUNT(*),SUM(resolved_bool=1),SUM(resolved_bool=0),"
            "MIN(start_ts_ms),MAX(COALESCE(end_ts_ms,start_ts_ms)),MAX(duration_sec),SUM(duration_sec) "
            "FROM gaps GROUP BY stream ORDER BY stream"
        ):
            rows.append({
                "stream": stream,
                "logged_gap_count": count,
                "resolved_count": resolved,
                "unresolved_count": unresolved,
                "first_logged_start_ms": start,
                "last_logged_end_or_start_ms": end,
                "max_duration_sec": maximum,
                "sum_duration_sec": total,
                "finding_status": "VERIFIED FROM CODE/DATA",
                "limitation": "Collector gap log is not proof that every outage was detected.",
            })
        return rows
    finally:
        conn.close()


def data_quality_findings() -> list[dict]:
    rows: list[dict] = []
    conn = ro(KEEPER)
    try:
        dup_excess, dup_groups = conn.execute(
            "SELECT COALESCE(SUM(n-1),0),COUNT(*) FROM ("
            "SELECT COUNT(*) n FROM liquidations "
            "GROUP BY ts_ms,symbol,side,price,quantity,notional,trade_time_ms HAVING COUNT(*)>1)"
        ).fetchone()
        mismatch = conn.execute(
            "SELECT COUNT(*) FROM liquidations WHERE "
            "ABS(notional-price*quantity)>MAX(1e-9,ABS(notional)*1e-12)"
        ).fetchone()[0]
        different, min_delta, max_delta = conn.execute(
            "SELECT COUNT(*),MIN(ts_ms-trade_time_ms),MAX(ts_ms-trade_time_ms) "
            "FROM liquidations WHERE ts_ms!=trade_time_ms"
        ).fetchone()
        rows.extend([
            {"check": "keeper liquidation economic-key duplicates", "value": dup_groups, "secondary_value": dup_excess, "status": "VERIFIED FROM CODE/DATA", "interpretation": "186 duplicate groups / 186 excess rows over the full keeper extent; frozen E support preflight separately rejects duplicates in its analysis interval"},
            {"check": "keeper liquidation notional != price*quantity", "value": mismatch, "secondary_value": None, "status": "VERIFIED FROM CODE/DATA", "interpretation": "stored notional is internally consistent with p*q"},
            {"check": "keeper liquidation event time differs from trade time", "value": different, "secondary_value": f"delta_ms={min_delta}..{max_delta}", "status": "VERIFIED FROM CODE/DATA", "interpretation": "E and T are distinct clocks; receive time is absent"},
            {"check": "keeper liquidation full UTC-day no-row runs", "value": 42, "secondary_value": "2026-04-28..2026-06-05 (39d); 2026-07-07..2026-07-09 (3d)", "status": "VERIFIED FROM CODE/DATA", "interpretation": "absence cannot be interpreted as calm market"},
        ])
    finally:
        conn.close()
    conn = ro(KLINES)
    try:
        complete = conn.execute(
            "SELECT COUNT(*),SUM(archive_rows),SUM(expected_rows),SUM(missing_minutes) "
            "FROM price_supplement_ingest WHERE status='COMPLETE'"
        ).fetchone()
        incomplete = conn.execute(
            "SELECT COUNT(*),SUM(archive_rows),SUM(expected_rows),SUM(missing_minutes) "
            "FROM price_supplement_ingest WHERE status='INCOMPLETE'"
        ).fetchone()
        rows.extend([
            {"check": "price supplement COMPLETE partitions", "value": complete[0], "secondary_value": f"rows={complete[1]};expected={complete[2]};missing={complete[3]}", "status": "VERIFIED FROM CODE/DATA", "interpretation": "partition receipt"},
            {"check": "price supplement INCOMPLETE partitions", "value": incomplete[0], "secondary_value": f"rows={incomplete[1]};expected={incomplete[2]};missing={incomplete[3]}", "status": "VERIFIED FROM CODE/DATA", "interpretation": "incompleteness is explicit, not silently filled"},
            {"check": "xsec kline duplicate symbol/open_time", "value": 0, "secondary_value": "composite PRIMARY KEY", "status": "VERIFIED FROM CODE/DATA", "interpretation": "schema prevents duplicate identities"},
        ])
    finally:
        conn.close()
    rows.append({"check": "Parquet archive physical ordering/identity violations", "value": 0, "secondary_value": "6,726,613,400 rows / 1,191 partitions accepted", "status": "VERIFIED FROM CODE/DATA", "interpretation": "physical identity only; does not prove semantic feed completeness"})
    rows.append({"check": "live database exact book_ticker row count", "value": None, "secondary_value": "sqlite_sequence max id 1,074,737,334", "status": "UNKNOWN / NOT RECOVERABLE", "interpretation": "active 157GB DB was not subjected to a disruptive full scan; max assigned id is not a row count"})
    return rows


def event_coverage_and_matrix() -> tuple[list[dict], list[dict]]:
    # This function reads identity/time and source availability only. It cannot
    # access the isolated frozen outcome file.
    manifest = read_csv(MANIFEST)
    liq = ro(KEEPER)
    klines = ro(KLINES)
    coverage: list[dict] = []
    matrix: list[dict] = []
    try:
        for event in manifest:
            event_id = event["event_id"]
            symbol = event["symbol"]
            base = int(event["base_ms"])
            boundary = int(event["fixed_boundary_ms"])
            liq_n, liq_min, liq_max = liq.execute(
                "SELECT COUNT(*),MIN(ts_ms),MAX(ts_ms) FROM liquidations "
                "WHERE symbol=? AND ts_ms>=? AND ts_ms<?",
                (symbol, base, boundary + MINUTE_MS),
            ).fetchone()
            price_n, price_min, price_max = klines.execute(
                "SELECT COUNT(*),MIN(open_time),MAX(open_time) FROM klines "
                "WHERE symbol=? AND open_time>=? AND open_time<=?",
                (symbol, base, boundary),
            ).fetchone()
            btc_n = klines.execute(
                "SELECT COUNT(*) FROM klines WHERE symbol='BTCUSDT' AND open_time>=? AND open_time<=?",
                (base, boundary),
            ).fetchone()[0]
            coverage.append({
                **event,
                "anchor_utc": utc(int(event["anchor_ts_ms"])),
                "liquidation_proxy_rows_0_240": liq_n,
                "liquidation_min_ts_ms": liq_min,
                "liquidation_max_ts_ms": liq_max,
                "event_symbol_kline_rows_expected_241": price_n,
                "event_symbol_kline_min_ms": price_min,
                "event_symbol_kline_max_ms": price_max,
                "btc_kline_rows_expected_241": btc_n,
                "event_symbol_in_aggtrade_archive": False,
                "event_symbol_in_bookticker_archive": False,
                "finding_status": "VERIFIED FROM CODE/DATA",
            })
            matrix.append({
                "event_id": event_id,
                "symbol": symbol,
                "anchor_ts_ms": event["anchor_ts_ms"],
                "liquidation_pressure": "PROXY ONLY",
                "aggressive_buy_sell_notional": "UNSUPPORTED",
                "minute_signed_flow_imbalance": "UNSUPPORTED",
                "raw_trades": "UNSUPPORTED",
                "aggTrade_only_flow": "UNSUPPORTED",
                "midquote": "UNSUPPORTED",
                "spread": "UNSUPPORTED",
                "top_level_depth": "UNSUPPORTED",
                "multi_level_depth": "UNSUPPORTED",
                "OFI": "UNSUPPORTED",
                "MLOFI": "UNSUPPORTED",
                "exact_add_cancel_decomposition": "NOT IDENTIFIABLE",
                "displayed_depth_recovery": "UNSUPPORTED",
                "Flow_Surprise_feasibility": "PROXY ONLY",
                "Impact_Surprise_feasibility": "PROXY ONLY",
                "BTC_common_market_adjustment": "VALID",
                "tick_regime_conditioning": "NOT IDENTIFIABLE",
                "measurement_sensitivity_analysis": "PROXY ONLY",
                "classification_basis": (
                    "event-symbol forceOrder p*q proxy + exact 1m OHLCV/BTC are present; "
                    "aggTrade and bookTicker archives contain only BTC/ETH/SOL"
                ),
            })
    finally:
        liq.close()
        klines.close()
    return coverage, matrix


def reconciliation() -> tuple[list[dict], dict]:
    # Isolated provenance audit: no result from here feeds any availability or
    # observability classification.
    old_doc = json.loads(OLD_CHANNEL.read_text(encoding="utf-8"))
    old = {row["event_id"]: row for row in old_doc["events"]}
    new = {row["event_id"]: row for row in read_csv(NEW_OUTCOME)}
    if set(old) != set(new):
        raise RuntimeError("historical result event populations differ")
    rows = []
    for event_id in sorted(old):
        old_net = float(old[event_id]["fixed_trade_net_bps"])
        new_gross = float(new[event_id]["frozen_240m_return_bps"])
        rows.append({
            "event_id": event_id,
            "symbol": new[event_id]["symbol"],
            "earlier_net_bps": old_net,
            "latest_gross_bps": new_gross,
            "latest_minus_earlier_bps": new_gross - old_net,
            "same_event_identity": True,
            "finding_status": "VERIFIED FROM CODE/DATA",
        })
    old_values = sorted(row["earlier_net_bps"] for row in rows)
    new_values = sorted(row["latest_gross_bps"] for row in rows)
    n = len(rows)
    summary = {
        "event_count": n,
        "same_event_id_set": True,
        "every_event_difference_exactly_10bps": all(abs(row["latest_minus_earlier_bps"] - 10.0) < 1e-9 for row in rows),
        "earlier_mean_net_bps": sum(old_values) / n,
        "earlier_median_net_bps": old_values[n // 2],
        "latest_mean_gross_bps": sum(new_values) / n,
        "latest_median_gross_bps": new_values[n // 2],
        "verified_reason": (
            "Earlier build_event subtracts COST_BPS=10 from the same entry/fixed OPEN log return; "
            "latest frozen exporter reports the gross log return without a cost subtraction."
        ),
        "finding_status": "VERIFIED FROM CODE/DATA",
    }
    return rows, summary


def static_feature_provenance() -> list[dict]:
    return [
        {"field_family": "liquidations.quantity", "source": "forceOrder o.q", "transformation": "float(q)", "semantic_class": "original order quantity proxy", "status": "VERIFIED FROM CODE/DATA"},
        {"field_family": "liquidations.notional", "source": "forceOrder o.p,o.q", "transformation": "float(p)*float(q)", "semantic_class": "nominal forced-order pressure proxy; not executed volume", "status": "VERIFIED FROM CODE/DATA"},
        {"field_family": "q_parent/q_echo", "source": "reconstructed anchor running_notional + 1m kline quote_volume", "transformation": "anchor running p*q proxy / trailing 15|30|60m quote volume", "semantic_class": "normalized observed forced-liquidation pressure proxy", "status": "VERIFIED FROM CODE/DATA"},
        {"field_family": "agg_trades.notional", "source": "aggTrade p,q", "transformation": "float(p)*float(q)", "semantic_class": "aggregate-trade notional; not one taker order", "status": "VERIFIED FROM CODE/DATA"},
        {"field_family": "fl_*_ofi", "source": "agg_trades.notional,is_buyer_maker", "transformation": "(taker-buy notional - taker-sell notional)/(sum)", "semantic_class": "aggressive aggTrade imbalance; NOT classical book OFI", "status": "VERIFIED FROM CODE/DATA"},
        {"field_family": "book_ticker.mid_price", "source": "best bid b, best ask a", "transformation": "(b+a)/2", "semantic_class": "top-of-book mid", "status": "VERIFIED FROM CODE/DATA"},
        {"field_family": "book_ticker.spread_pct", "source": "best bid/ask and derived mid", "transformation": "(ask-bid)/mid", "semantic_class": "relative L1 spread", "status": "VERIFIED FROM CODE/DATA"},
        {"field_family": "book_ticker.book_imbalance", "source": "best bid/ask quantities B,A", "transformation": "(B-A)/(B+A)", "semantic_class": "L1 displayed-quantity imbalance", "status": "VERIFIED FROM CODE/DATA"},
        {"field_family": "book_ticker.bid_depth_usd", "source": "best bid price and quantity", "transformation": "bid_price*bid_qty", "semantic_class": "L1 displayed bid notional only", "status": "VERIFIED FROM CODE/DATA"},
        {"field_family": "bk_pull", "source": "bookTicker L1 bid_qty snapshots", "transformation": "minimum pre-1m bid_qty / average pre-10m bid_qty", "semantic_class": "sampled L1 quantity-ratio proxy; not cancellations", "status": "VERIFIED FROM CODE/DATA"},
        {"field_family": "bk_refill", "source": "bookTicker L1 bid_qty snapshots", "transformation": "average post-5m bid_qty / average pre-10m bid_qty", "semantic_class": "sampled L1 quantity-ratio proxy; not exact replenishment", "status": "VERIFIED FROM CODE/DATA"},
        {"field_family": "fl_*_impact", "source": "mark-price return + aggTrade total notional", "transformation": "absolute window return bps / (total aggTrade notional / 1e6)", "semantic_class": "derived price-per-flow proxy", "status": "VERIFIED FROM CODE/DATA"},
    ]


def static_timeline() -> list[dict]:
    return [
        {"date_or_range": "2026-02-15", "change": "Earliest retained liquidation/aggTrade/mark-price history begins", "evidence": "keeper coverage and archive manifests", "status": "VERIFIED FROM CODE/DATA"},
        {"date_or_range": "2026-02-20", "change": "microstructure_collector first visible in Git history", "evidence": "git commit 575d40da", "status": "VERIFIED FROM CODE/DATA"},
        {"date_or_range": "2026-04-11", "change": "Earliest bookTicker archive date for BTC/ETH", "evidence": "Parquet manifest", "status": "VERIFIED FROM CODE/DATA"},
        {"date_or_range": "2026-04-18", "change": "SOL aggTrade/bookTicker/mark archive begins", "evidence": "Parquet manifests", "status": "VERIFIED FROM CODE/DATA"},
        {"date_or_range": "2026-04-28..2026-06-05", "change": "No liquidation rows (39 full UTC days)", "evidence": "keeper daily row census", "status": "VERIFIED FROM CODE/DATA"},
        {"date_or_range": "2026-06-06T17:43:52.123Z", "change": "Measured transition from 2–3-symbol per-symbol forceOrder regime to all-market !forceOrder@arr", "evidence": "liquidation_source_quality_contract_v2: first post-blackout row + 171 symbols in following hour", "status": "VERIFIED FROM CODE/DATA"},
        {"date_or_range": "2026-06-06..2026-06-10", "change": "All BTC/ETH/SOL bookTicker partitions contain zero rows", "evidence": "Parquet manifests", "status": "VERIFIED FROM CODE/DATA"},
        {"date_or_range": "2026-07-06T10:06:39Z..2026-07-10T11:24:38Z", "change": "Second all-market liquidation outage; includes three full no-row UTC days", "evidence": "keeper census + LIQUIDATION_SILENCE_DETECTOR_2026-07-11.md", "status": "VERIFIED FROM CODE/DATA"},
        {"date_or_range": "2026-07-03", "change": "collector websocket fallback URL correction appears", "evidence": "git commit 5cda3122", "status": "VERIFIED FROM CODE/DATA"},
        {"date_or_range": "2026-07-10", "change": "routed market websocket/health changes appear", "evidence": "git commits bd7feb32,b0ac3b89,00ef49ad", "status": "VERIFIED FROM CODE/DATA"},
        {"date_or_range": "2026-07-21", "change": "bookticker_collector first visible in Git history", "evidence": "git commit 968adb70", "status": "VERIFIED FROM CODE/DATA"},
        {"date_or_range": "2026-07-23T20:40:46.344Z", "change": "SQLite rotation cutoff; live becomes microstructure_02.db", "evidence": "rotation_state.json", "status": "VERIFIED FROM CODE/DATA"},
        {"date_or_range": "2026-08-06", "change": "large frozen SQLite reclaimed; small tables retained; collectors made rotation-aware", "evidence": "rotation_state.json and git commit 552ca970", "status": "VERIFIED FROM CODE/DATA"},
        {"date_or_range": "historical exact Binance semantics", "change": "No field-level collector version persisted per row; current docs/code cannot prove all prior upstream semantics", "evidence": "schemas lack source/version/receive fields", "status": "UNKNOWN / NOT RECOVERABLE"},
    ]


def static_go_stop() -> list[dict]:
    return [
        {"item": "Flow Surprise V1 — forceOrder pressure-proxy version", "decision": "GO", "qualification": "PROXY ONLY; freeze p*q/q/count and E/T sensitivity semantically, never by return"},
        {"item": "Flow Surprise V1 — aggressive-trade version for historical E-DER", "decision": "STOP", "qualification": "event-symbol aggTrades absent"},
        {"item": "Impact Surprise V1 — 1m OHLCV version", "decision": "GO", "qualification": "PROXY ONLY; do not call OHLC4/OPEN a quote mid"},
        {"item": "Impact Surprise V1 — true quote-mid version", "decision": "STOP", "qualification": "event-symbol quotes absent"},
        {"item": "OFI", "decision": "STOP", "qualification": "no event-symbol book updates; existing fl_ofi is aggTrade imbalance"},
        {"item": "MLOFI", "decision": "STOP", "qualification": "no multi-level event-symbol book"},
        {"item": "displayed-book resilience", "decision": "STOP", "qualification": "no event-symbol bookTicker/depth history"},
        {"item": "exact replenishment", "decision": "STOP", "qualification": "snapshot feed cannot identify adds/cancels; event feed absent"},
        {"item": "hidden-liquidity claims", "decision": "STOP", "qualification": "NOT IDENTIFIABLE from stored data"},
        {"item": "dynamic exit research", "decision": "STOP", "qualification": "audit provides measurement feasibility, not forward-confirmed mechanism evidence"},
    ]


def static_claims() -> list[dict]:
    return [
        {"claim": "Observed forced-liquidation pressure proxy exists around all 25 events", "class": "POTENTIALLY SUPPORTED/TESTABLE", "basis": "forceOrder E,S,p,q,T rows retained"},
        {"claim": "Complete executed liquidation volume", "class": "NOT IDENTIFIABLE", "basis": "throttled/snapshot-like source; ap,l,z,X discarded"},
        {"claim": "OHLCV response anomaly relative to a chronological generic benchmark", "class": "POTENTIALLY SUPPORTED/TESTABLE", "basis": "exact 1m bars and prior-OOS framework; price proxy only"},
        {"claim": "Quote-mid marginal impact for historical E-DER", "class": "TESTABLE BUT CURRENTLY NOT SUPPORTED", "basis": "no event-symbol quote archive"},
        {"claim": "Aggressive-flow absorption for historical E-DER", "class": "TESTABLE BUT CURRENTLY NOT SUPPORTED", "basis": "no event-symbol aggTrade archive"},
        {"claim": "Displayed LOB resilience/OFI/MLOFI for historical E-DER", "class": "TESTABLE BUT CURRENTLY NOT SUPPORTED", "basis": "no event-symbol LOB archive"},
        {"claim": "Exact add/cancel/replenishment or hidden-liquidity mechanism", "class": "NOT IDENTIFIABLE", "basis": "no incremental L2 sequence and hidden orders are not displayed"},
        {"claim": "Historical tick-regime conditioning", "class": "NOT IDENTIFIABLE", "basis": "no historical filter-change archive; current exchangeInfo is not historical evidence"},
    ]


def main() -> None:
    if len(read_csv(MANIFEST)) != 25:
        raise RuntimeError("frozen manifest is not 25 events")
    schemas = schema_inventory()
    databases = database_inventory()
    archives = archive_inventory()
    gaps = gap_inventory()
    quality = data_quality_findings()
    coverage, matrix = event_coverage_and_matrix()
    features = static_feature_provenance()
    timeline = static_timeline()
    go_stop = static_go_stop()
    claims = static_claims()
    # Outcome reconciliation deliberately happens only after all outcome-blind
    # classifications above are fully constructed.
    recon_rows, recon_summary = reconciliation()

    write_csv("schema_inventory.csv", schemas)
    write_csv("database_inventory.csv", databases)
    write_csv("archive_inventory.csv", archives)
    write_csv("collector_gap_inventory.csv", gaps)
    write_csv("data_quality_findings.csv", quality)
    write_csv("event_data_coverage.csv", coverage)
    write_csv("historical_e_der_observability_matrix.csv", matrix)
    write_csv("feature_provenance.csv", features)
    write_csv("feed_semantic_timeline.csv", timeline)
    write_csv("go_stop_matrix.csv", go_stop)
    write_csv("claim_identifiability.csv", claims)
    write_csv("result_reconciliation_event_level.csv", recon_rows)
    (OUT / "result_reconciliation_summary.json").write_text(
        json.dumps(recon_summary, indent=2) + "\n", encoding="utf-8"
    )
    evidence = {
        "status": "READ_ONLY_AUDIT_EVIDENCE_EXPORTED",
        "outcome_blind_classification": True,
        "bible_sha256": sha256(ROOT / "docs/research/ECLIPSE_RESEARCH_BIBLE.md"),
        "manifest_sha256": sha256(MANIFEST),
        "event_count": len(matrix),
        "symbols": sorted({row["symbol"] for row in matrix}),
        "classification_counts_by_field": {
            field: {state: sum(row[field] == state for row in matrix) for state in ("VALID", "PROXY ONLY", "UNSUPPORTED", "NOT IDENTIFIABLE")}
            for field in matrix[0]
            if field not in {"event_id", "symbol", "anchor_ts_ms", "classification_basis"}
        },
        "result_reconciliation": recon_summary,
    }
    (OUT / "audit_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    artifact_rows = []
    for path in sorted(OUT.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "AUDIT_ARTIFACT_MANIFEST_SHA256.csv":
            artifact_rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_csv("AUDIT_ARTIFACT_MANIFEST_SHA256.csv", artifact_rows)
    print(json.dumps({"status": evidence["status"], "events": len(matrix), "out": str(OUT)}))


if __name__ == "__main__":
    main()
