"""As-of-lookup consumer migration parity, pilot #2 (BATCH-STORAGE-
ROTATION-RETENTION-ASOF-LOOKUP-CONSUMER-MIGRATION-V2).

`tools/research_s34_btc_microtrend_eth_quality.py`'s two identical
inline direct-SQL `ORDER BY ts_ms DESC LIMIT 1` queries (BTC mark price
10s-before-entry and at-entry) were extracted into
`btc_mark_price_before_or_at` (kept unchanged as the parity oracle) and
migrated to `btc_mark_price_before_or_at_v2` (reader-backed, used by
`main()` instead).

`mark_prices` has NO archived partition for BTCUSDT (only ETHUSDT/
2026-05 is archived) -- real production usage of this consumer always
resolves SQLITE_ONLY. This is confirmed directly against the real
database below, and the archive-only / hybrid scenarios this real data
cannot reach are instead covered with the same small synthetic
fixtures used by test_ami_storage_research_reader_lookup.py, so the
migrated function's correctness isn't limited to the one path real
BTCUSDT data happens to exercise.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3

import pytest

from ami.storage import production as PR
from ami.storage import research_reader as RR
from ami.storage import source_access as SRC
from ami.storage.registry import get_table_spec
from tools.research_s34_btc_microtrend_eth_quality import (
    btc_mark_price_before_or_at, btc_mark_price_before_or_at_v2)

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)
SPEC = get_table_spec("mark_prices")

pytestmark = pytest.mark.skipif(
    not os.path.exists(REAL_SOURCE_DB), reason="real source database not present in this checkout")


def _root():
    root, _source = PR.resolve_production_root()
    return root


def _old(ts_ms):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return btc_mark_price_before_or_at(conn, ts_ms)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


# ---------------------------------------------------------------------------
# Production smoke: BTCUSDT has no archive partition -> always SQLITE_ONLY
# ---------------------------------------------------------------------------

def test_btcusdt_has_no_archive_partition_in_the_real_catalog():
    entries = RR._root_catalog_entries(_root())
    btc_entries = [e for e in entries if e["symbol"] == "BTCUSDT"]
    assert btc_entries == []


def test_production_smoke_sqlite_only_parity():
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='BTCUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()

    plan = RR.plan_read(_root(), table="mark_prices", symbol="BTCUSDT", start_ms=max_ts - 1000, end_ms=max_ts + 1)
    assert plan.mode == "SQLITE_ONLY"

    for ts_ms in (max_ts, max_ts - 10_000):
        old = _old(ts_ms)
        new = btc_mark_price_before_or_at_v2(_root(), ts_ms, source_db_path=REAL_SOURCE_DB)
        assert old is not None
        assert new == pytest.approx(old, rel=1e-9)


def test_production_smoke_repeated_run_determinism():
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='BTCUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    root = _root()
    results = [btc_mark_price_before_or_at_v2(root, max_ts, source_db_path=REAL_SOURCE_DB) for _ in range(3)]
    assert results[0] == results[1] == results[2]


def test_production_smoke_source_mutation_zero():
    root = _root()
    idx_path = os.path.join(root, PR.ROOT_INDEX_NAME)
    before = os.path.getmtime(idx_path)
    btc_mark_price_before_or_at_v2(root, 1700000000000, source_db_path=REAL_SOURCE_DB)
    after = os.path.getmtime(idx_path)
    assert before == after


# ---------------------------------------------------------------------------
# Synthetic fixtures for archive-only / hybrid / trust-failure scenarios
# real BTCUSDT data cannot reach (mirrors test_ami_storage_research_reader_
# lookup.py's fixture style).
# ---------------------------------------------------------------------------

def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _write_parquet(path, rows, spec):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ami.storage.archive import build_pyarrow_schema
    schema = build_pyarrow_schema(spec)
    arrays = []
    for i, col in enumerate(spec.preserved_columns):
        pa_type = schema.field(col).type
        values = [r[i] for r in rows]
        arrays.append(pa.array(values, type=pa_type))
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), path, compression="zstd")


def _mark_prices_rows(start_id, start_ms, n, symbol="BTCUSDT", step_ms=10_000):
    return [(start_id + i, start_ms + i * step_ms, symbol, 60000.0 + i * 10, None, None) for i in range(n)]


def _build_archive_partition(root, *, symbol, partition_start_ms, partition_end_ms, rows,
                              corrupt_manifest=False):
    rel = os.path.join("table=mark_prices", "venue=BINANCE_USDM_PERP", "market_segment=PERPETUAL_FUTURES",
                        f"symbol={symbol}", "year=2026", "month=04", "version=v1")
    final_dir = os.path.join(root, rel)
    os.makedirs(final_dir, exist_ok=True)
    shard_file = "part-00000.parquet"
    shard_path = os.path.join(final_dir, shard_file)
    _write_parquet(shard_path, rows, SPEC)

    partition_id = f"test-mark_prices-{symbol}-2026-04"
    manifest = {
        "source_table": "mark_prices", "symbol": symbol, "venue": SPEC.venue, "market_segment": SPEC.market_segment,
        "row_count": len(rows), "shards": None,
        "ordered_scientific_content_hash": "0" * 64 if corrupt_manifest else "irrelevant-for-lookup-tests",
        "partition_id": partition_id, "parquet_path": os.path.join(rel, shard_file),
        "production_status": "PRODUCTION_VERIFIED", "purge_authorization": "PROHIBITED",
        "source_watermark_field": "id", "source_watermark_value": rows[-1][0],
        "parquet_sha256": _sha256_file(shard_path),
    }
    manifest_path = os.path.join(final_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with open(os.path.join(final_dir, "_SUCCESS"), "w", encoding="utf-8") as f:
        f.write(partition_id + "\n")

    manifest_sha = _sha256_file(manifest_path)
    catalog_entry = {
        "archive_identity": partition_id, "archive_relative_path": rel,
        "source_table": "mark_prices", "symbol": symbol, "venue": SPEC.venue, "market_segment": SPEC.market_segment,
        "partition_start_ms": partition_start_ms, "partition_end_ms": partition_end_ms,
        "row_count": len(rows), "manifest_sha256": "f" * 64 if corrupt_manifest else manifest_sha,
        "authorization_receipt_sha256": None,
        "production_status": "PRODUCTION_VERIFIED", "purge_authorization": "PROHIBITED",
        "scientific_content_hash": manifest["ordered_scientific_content_hash"],
    }
    with open(os.path.join(final_dir, "catalog_entry.json"), "w", encoding="utf-8") as f:
        json.dump(catalog_entry, f)
    return catalog_entry


def _write_catalog_index(root, entries):
    index = {"catalog_contract_version": "v1", "entry_count": len(entries), "entries": entries}
    with open(os.path.join(root, "catalog_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)


MARK_PRICES_DDL = """CREATE TABLE mark_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL,
    mark_price REAL NOT NULL, funding_rate REAL, next_funding_time_ms INTEGER)"""


def _sqlite_mark_prices(path, rows):
    conn = sqlite3.connect(path)
    conn.execute(MARK_PRICES_DDL)
    conn.execute("CREATE INDEX idx_mp_symbol_ts ON mark_prices(symbol, ts_ms)")
    conn.executemany(
        "INSERT INTO mark_prices (id, ts_ms, symbol, mark_price, funding_rate, next_funding_time_ms) "
        "VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


APR = 1775001600000
MAY = 1777593600000


def test_archive_only_lookup_parity_synthetic(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    rows = _mark_prices_rows(1, APR, 20)
    entry = _build_archive_partition(root, symbol="BTCUSDT", partition_start_ms=APR, partition_end_ms=MAY, rows=rows)
    _write_catalog_index(root, [entry])

    plan = RR.plan_read(root, table="mark_prices", symbol="BTCUSDT", start_ms=APR + 90_000, end_ms=APR + 90_001)
    assert plan.mode == "ARCHIVE_ONLY"

    result = btc_mark_price_before_or_at_v2(root, APR + 95_000)  # 95s -> row id 10 (ts APR+90000)
    assert result == 60090.0
    lookup = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="BTCUSDT", ts_ms=APR + 95_000,
                                           columns=("mark_price",))
    assert lookup.provenance["source_type"] == "ARCHIVE_ONLY"
    assert lookup.provenance["archive_identity"] == entry["archive_identity"]


def test_hybrid_boundary_lookup_parity_synthetic(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    archive_rows = _mark_prices_rows(1, APR, 10)
    entry = _build_archive_partition(root, symbol="BTCUSDT", partition_start_ms=APR, partition_end_ms=MAY,
                                      rows=archive_rows)
    _write_catalog_index(root, [entry])
    db_path = str(tmp_path / "source.sqlite")
    _sqlite_mark_prices(db_path, [])  # live tail is empty -- forces fallback into archive

    query_ts = MAY + 60_000  # naturally SQLite's turf (after archive ends), but SQLite is empty
    result = btc_mark_price_before_or_at_v2(root, query_ts, source_db_path=db_path)
    assert result == 60090.0  # falls back to the archive's last row (id=10)
    lookup = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="BTCUSDT", ts_ms=query_ts,
                                           columns=("mark_price",), source_db_path=db_path)
    assert lookup.provenance["source_type"] == "HYBRID"
    assert lookup.provenance["result_source"] == "ARCHIVE"


def test_exact_timestamp_hit_and_between_timestamps_fallback_synthetic(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    rows = _mark_prices_rows(1, MAY, 10, step_ms=10_000)
    _sqlite_mark_prices(db_path, rows)

    exact_old = None
    conn = sqlite3.connect(db_path)
    exact_old = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (MAY + 50_000,)).fetchone()[0]
    conn.close()
    exact_new = btc_mark_price_before_or_at_v2(root, MAY + 50_000, source_db_path=db_path)
    assert exact_new == exact_old == 60050.0  # row id=6 (0-indexed i=5) -> 60000+50

    between_new = btc_mark_price_before_or_at_v2(root, MAY + 55_000, source_db_path=db_path)
    assert between_new == 60050.0  # still row id=6; next row (7) is at +60s


def test_no_prior_row_returns_none_synthetic(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    _sqlite_mark_prices(db_path, _mark_prices_rows(1, MAY, 5))
    result = btc_mark_price_before_or_at_v2(root, MAY - 3600_000, source_db_path=db_path)
    assert result is None


def test_tie_break_parity_with_direct_sql_synthetic(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    tied_ts = MAY + 10_000
    rows = [(1, MAY, "BTCUSDT", 60000.0, None, None),
            (2, tied_ts, "BTCUSDT", 60010.0, None, None),
            (3, tied_ts, "BTCUSDT", 60020.0, None, None)]
    _sqlite_mark_prices(db_path, rows)

    conn = sqlite3.connect(db_path)
    oracle = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms<=? "
        "ORDER BY ts_ms DESC, id DESC LIMIT 1", (tied_ts,)).fetchone()[0]
    conn.close()

    new = btc_mark_price_before_or_at_v2(root, tied_ts, source_db_path=db_path)
    assert new == oracle == 60020.0  # higher id wins the tie


def test_selected_column_projection_matches_direct_query_shape(tmp_path):
    """The migrated function only ever needs `mark_price` -- confirms the
    reader-backed path returns exactly that projected column, matching
    the direct-SQL oracle's `SELECT mark_price` shape."""
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    _sqlite_mark_prices(db_path, _mark_prices_rows(1, MAY, 5))
    lookup = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="BTCUSDT", ts_ms=MAY + 20_000,
                                           columns=("mark_price",), source_db_path=db_path)
    assert lookup.columns == ("mark_price",)
    assert len(lookup.row) == 1


def test_provenance_correctness_synthetic(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    _write_catalog_index(root, [])
    db_path = str(tmp_path / "source.sqlite")
    _sqlite_mark_prices(db_path, _mark_prices_rows(1, MAY, 5))
    lookup = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="BTCUSDT", ts_ms=MAY + 20_000,
                                           columns=("mark_price",), source_db_path=db_path)
    prov = lookup.provenance
    assert prov["source_type"] == "SQLITE_ONLY"
    assert prov["result_source"] == "SQLITE"
    assert prov["query_ts_ms"] == MAY + 20_000
    assert prov["inclusive"] is True
    assert prov["columns"] == ["mark_price"]
    assert prov["result_ts_ms"] is not None
    assert prov["archive_identity"] is None
    assert prov["ordering"] == "(ts_ms DESC, id DESC)"


def test_trust_failure_fails_closed_synthetic(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    rows = _mark_prices_rows(1, APR, 10)
    entry = _build_archive_partition(root, symbol="BTCUSDT", partition_start_ms=APR, partition_end_ms=MAY,
                                      rows=rows, corrupt_manifest=True)
    _write_catalog_index(root, [entry])
    with pytest.raises(RR.ArchiveTrustError):
        btc_mark_price_before_or_at_v2(root, APR + 90_000)
