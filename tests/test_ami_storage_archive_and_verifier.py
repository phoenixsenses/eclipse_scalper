"""Focused tests: ami.storage.archive (exporter/schema/manifest) +
ami.storage.verifier. Uses small synthetic fixtures; the live-data
reproduction is covered separately in test_ami_storage_acceptance.py.
"""
from __future__ import annotations

import datetime as dt
import sqlite3

import pytest

from ami.storage import archive as A
from ami.storage import verifier as V
from ami.storage.partition import build_partition_identity
from ami.storage.registry import get_table_spec

NOW = dt.datetime(2026, 7, 7, 18, 0, 0, tzinfo=dt.timezone.utc)


def _fixture_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE mark_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL,
        mark_price REAL NOT NULL, funding_rate REAL, next_funding_time_ms INTEGER)""")
    may_start = 1777593600000
    rows = [(may_start + i * 60000, "ETHUSDT", 3000.0 + i, 0.0001 if i % 2 == 0 else None,
              1777600000000 if i % 2 == 0 else None) for i in range(10)]
    conn.executemany("INSERT INTO mark_prices (ts_ms,symbol,mark_price,funding_rate,next_funding_time_ms) "
                     "VALUES (?,?,?,?,?)", rows)
    conn.commit()
    return conn


def _fixture_partition(watermark=10):
    return build_partition_identity(table="mark_prices", symbol="ETHUSDT", utc_year=2026,
                                     utc_month=5, source_watermark_value=watermark, now=NOW)


# ---------------------------------------------------------------------------
# Row hashing / fetch
# ---------------------------------------------------------------------------

def test_canonical_row_hash_deterministic():
    rows = [(1, "a"), (2, "b")]
    assert A.canonical_row_hash(rows) == A.canonical_row_hash(rows)


def test_fetch_partition_rows_excludes_above_watermark():
    conn = _fixture_conn()
    spec = get_table_spec("mark_prices")
    partition = _fixture_partition(watermark=5)
    rows = A.fetch_partition_rows(conn, spec, partition)
    assert all(r[0] <= 5 for r in rows)
    assert len(rows) == 5
    conn.close()


def test_fetch_partition_rows_stable_ordering():
    conn = _fixture_conn()
    spec = get_table_spec("mark_prices")
    partition = _fixture_partition(watermark=10)
    rows = A.fetch_partition_rows(conn, spec, partition)
    ids = [r[0] for r in rows]
    assert ids == sorted(ids)
    conn.close()


# ---------------------------------------------------------------------------
# Export (requires pyarrow -- available in this environment, confirmed
# during the disposable dry-run batch)
# ---------------------------------------------------------------------------

def test_export_partition_writes_and_publishes(tmp_path):
    conn = _fixture_conn()
    partition = _fixture_partition(watermark=10)
    output_root = str(tmp_path / "out")
    result = A.export_partition(conn, "mark_prices", partition, output_root,
                                allowed_roots=(output_root,), max_output_bytes=10 * 1024 * 1024)
    assert result["row_count"] == 10
    assert result["final_path"].endswith(".parquet")
    import os
    assert os.path.exists(result["final_path"])
    assert not os.path.exists(result["final_path"] + ".partial")
    conn.close()


def test_export_partition_rejects_non_disposable_root(tmp_path):
    conn = _fixture_conn()
    partition = _fixture_partition(watermark=10)
    with pytest.raises(A.ProductionPathRejected):
        A.export_partition(conn, "mark_prices", partition, "C:/production/archive",
                           allowed_roots=(str(tmp_path / "out"),), max_output_bytes=10 * 1024 * 1024)
    conn.close()


def test_export_partition_direct_read_matches_source(tmp_path):
    import pyarrow.parquet as pq
    conn = _fixture_conn()
    spec = get_table_spec("mark_prices")
    partition = _fixture_partition(watermark=10)
    output_root = str(tmp_path / "out")
    result = A.export_partition(conn, "mark_prices", partition, output_root,
                                allowed_roots=(output_root,), max_output_bytes=10 * 1024 * 1024)
    source_rows = A.fetch_partition_rows(conn, spec, partition)
    table = pq.read_table(result["final_path"])
    d = table.to_pydict()
    n = len(d["id"])
    parquet_rows = [tuple(d[c][i] for c in spec.preserved_columns) for i in range(n)]
    assert A.canonical_row_hash(parquet_rows) == A.canonical_row_hash(source_rows)
    conn.close()


def test_export_nullable_columns_preserved(tmp_path):
    import pyarrow.parquet as pq
    conn = _fixture_conn()
    partition = _fixture_partition(watermark=10)
    output_root = str(tmp_path / "out")
    result = A.export_partition(conn, "mark_prices", partition, output_root,
                                allowed_roots=(output_root,), max_output_bytes=10 * 1024 * 1024)
    table = pq.read_table(result["final_path"])
    funding = table.to_pydict()["funding_rate"]
    assert None in funding  # odd-index rows had funding_rate=None
    conn.close()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def test_manifest_hardcodes_disposable_and_prohibited():
    spec = get_table_spec("mark_prices")
    partition = _fixture_partition()
    m = A.build_manifest(spec=spec, partition=partition, row_count=10, scientific_hash="abc",
                         parquet_path="x.parquet", parquet_size=100, parquet_sha256="def",
                         source_schema_hash="ghi", parquet_schema_hash="jkl", unresolved_gap_count=0,
                         export_cutoff="2026-01-01T00:00:00Z", publication_timestamp="2026-01-01T00:00:01Z",
                         verification_status="PASS", dry_run_identity="TEST")
    assert m["production_status"] == "DISPOSABLE_NOT_PRODUCTION"
    assert m["purge_authorization"] == "PROHIBITED"


def test_manifest_field_count_at_least_36():
    spec = get_table_spec("mark_prices")
    partition = _fixture_partition()
    m = A.build_manifest(spec=spec, partition=partition, row_count=10, scientific_hash="abc",
                         parquet_path="x.parquet", parquet_size=100, parquet_sha256="def",
                         source_schema_hash="ghi", parquet_schema_hash="jkl", unresolved_gap_count=0,
                         export_cutoff="2026-01-01T00:00:00Z", publication_timestamp="2026-01-01T00:00:01Z",
                         verification_status="PASS", dry_run_identity="TEST")
    assert len(m) >= 36


def test_manifest_gap_status_reflects_unresolved_count():
    spec = get_table_spec("mark_prices")
    partition = _fixture_partition()
    m_clean = A.build_manifest(spec=spec, partition=partition, row_count=10, scientific_hash="abc",
                               parquet_path="x", parquet_size=1, parquet_sha256="d", source_schema_hash="g",
                               parquet_schema_hash="j", unresolved_gap_count=0, export_cutoff="x",
                               publication_timestamp="y", verification_status="PASS", dry_run_identity="T")
    m_gapped = A.build_manifest(spec=spec, partition=partition, row_count=10, scientific_hash="abc",
                                parquet_path="x", parquet_size=1, parquet_sha256="d", source_schema_hash="g",
                                parquet_schema_hash="j", unresolved_gap_count=3, export_cutoff="x",
                                publication_timestamp="y", verification_status="PASS", dry_run_identity="T")
    assert m_clean["source_gap_status"] == "NO_UNRESOLVED_GAPS"
    assert m_gapped["source_gap_status"] == "GAPS_PRESENT"
    assert m_gapped["unresolved_gap_count"] == 3


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

def test_verify_structural_pass():
    r = V.verify_structural(parquet_readable=True, schema_matches=True, compression="ZSTD",
                            expected_compression="ZSTD", extra_columns=(), missing_columns=())
    assert r.is_verified


def test_verify_structural_fails_wrong_compression():
    r = V.verify_structural(parquet_readable=True, schema_matches=True, compression="GZIP",
                            expected_compression="ZSTD", extra_columns=(), missing_columns=())
    assert r.state == V.FAILED_SCHEMA
    assert not r.is_verified


def test_verify_structural_unreadable_fails_immediately():
    r = V.verify_structural(parquet_readable=False, schema_matches=True, compression="ZSTD",
                            expected_compression="ZSTD", extra_columns=(), missing_columns=())
    assert r.state == V.FAILED_SCHEMA


def test_verify_accounting_row_count_mismatch():
    r = V.verify_accounting(row_count=9, expected_row_count=10, min_id=1, max_id=9,
                            expected_min_id=1, expected_max_id=10, duplicate_count=0,
                            null_count_mismatches=(), watermark_value=10)
    assert r.state == V.FAILED_ACCOUNTING


def test_verify_accounting_duplicate_fails():
    r = V.verify_accounting(row_count=10, expected_row_count=10, min_id=1, max_id=10,
                            expected_min_id=1, expected_max_id=10, duplicate_count=1,
                            null_count_mismatches=(), watermark_value=10)
    assert r.state == V.FAILED_ACCOUNTING


def test_verify_accounting_watermark_exceeded_fails():
    r = V.verify_accounting(row_count=10, expected_row_count=10, min_id=1, max_id=11,
                            expected_min_id=1, expected_max_id=11, duplicate_count=0,
                            null_count_mismatches=(), watermark_value=10)
    assert r.state == V.FAILED_ACCOUNTING


def test_verify_scientific_parity_pass():
    r = V.verify_scientific_parity(source_hash="a", parquet_hash="a", mismatch_count=0)
    assert r.is_verified


def test_verify_scientific_parity_fails_on_hash_mismatch():
    r = V.verify_scientific_parity(source_hash="a", parquet_hash="b", mismatch_count=0)
    assert r.state == V.FAILED_CONTENT_PARITY


def test_verify_checksum_pass_and_fail():
    assert V.verify_checksum(expected_sha256="a", actual_sha256="a").is_verified
    assert V.verify_checksum(expected_sha256="a", actual_sha256="b").state == V.FAILED_CHECKSUM


def test_verify_manifest_all_fields():
    m = {"parquet_sha256": "a", "ordered_scientific_content_hash": "b", "partition_id": "c",
         "production_status": "DISPOSABLE_NOT_PRODUCTION", "purge_authorization": "PROHIBITED"}
    r = V.verify_manifest(manifest=m, expected_parquet_sha256="a", expected_scientific_hash="b",
                          expected_partition_id="c")
    assert r.is_verified


def test_verify_manifest_rejects_production_status():
    m = {"parquet_sha256": "a", "ordered_scientific_content_hash": "b", "partition_id": "c",
         "production_status": "PRODUCTION_ACTIVE", "purge_authorization": "PROHIBITED"}
    r = V.verify_manifest(manifest=m, expected_parquet_sha256="a", expected_scientific_hash="b",
                          expected_partition_id="c")
    assert r.state == V.FAILED_MANIFEST


def test_verify_manifest_rejects_non_prohibited_purge():
    m = {"parquet_sha256": "a", "ordered_scientific_content_hash": "b", "partition_id": "c",
         "production_status": "DISPOSABLE_NOT_PRODUCTION", "purge_authorization": "AUTHORIZED"}
    r = V.verify_manifest(manifest=m, expected_parquet_sha256="a", expected_scientific_hash="b",
                          expected_partition_id="c")
    assert r.state == V.FAILED_MANIFEST


def test_verify_full_short_circuits_on_first_failure():
    ok = V.VerificationResult(V.VERIFIED_DISPOSABLE, ())
    bad = V.VerificationResult(V.FAILED_CHECKSUM, ("x",))
    combined = V.verify_full(ok, bad, ok)
    assert combined.state == V.FAILED_CHECKSUM


def test_verify_full_all_pass():
    ok = V.VerificationResult(V.VERIFIED_DISPOSABLE, ())
    combined = V.verify_full(ok, ok, ok)
    assert combined.is_verified


def test_no_failed_state_equals_verified():
    for state in V.FAILED_STATES:
        assert state != V.VERIFIED_DISPOSABLE
    assert len(V.FAILED_STATES) == len(V.VERIFICATION_STATES) - 1
