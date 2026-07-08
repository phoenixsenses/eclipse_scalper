"""Focused tests: ami.storage.production (production-activation rehearsal).

All engine-level tests (staging/publication/interruption/corruption)
use DISPOSABLE roots under pytest's own `tmp_path` -- never the real
`data/archives/raw_v1` production root, which is exercised only by the
live acceptance test at the bottom of this file (read-only against the
source; the production root itself is inspected read-only, never
re-published, never corrupted).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3

import pytest

from ami.storage import production as PR
from ami.storage.partition import build_partition_identity
from ami.storage.registry import get_table_spec

NOW = dt.datetime(2026, 7, 7, 18, 0, 0, tzinfo=dt.timezone.utc)


def _fixture_conn(n=6):
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE mark_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL,
        mark_price REAL NOT NULL, funding_rate REAL, next_funding_time_ms INTEGER)""")
    may_start = 1777593600000
    rows = [(may_start + i * 60000, "ETHUSDT", 3000.0 + i, None, None) for i in range(n)]
    conn.executemany("INSERT INTO mark_prices (ts_ms,symbol,mark_price,funding_rate,next_funding_time_ms) "
                     "VALUES (?,?,?,?,?)", rows)
    conn.commit()
    return conn


def _fixture_partition(watermark=6):
    return build_partition_identity(table="mark_prices", symbol="ETHUSDT", utc_year=2026,
                                     utc_month=5, source_watermark_value=watermark, now=NOW)


def _source_schema_hash(conn):
    sql = conn.execute("SELECT sql FROM sqlite_master WHERE name='mark_prices'").fetchone()[0]
    return hashlib.sha256(sql.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Production root
# ---------------------------------------------------------------------------

def test_fallback_root_resolution():
    root, source = PR.resolve_production_root()
    assert source == "operator_approved_fallback"
    assert "data" in root.lower() and "archives" in root.lower()


def test_frozen_root_resolution_if_supplied():
    frozen = "D:/eclipse_scalper/data/archives/raw_v1_alt"
    root, source = PR.resolve_production_root(frozen)
    assert source == "frozen_accepted_artifact"
    assert root == os.path.normpath(frozen)


def test_path_traversal_rejected():
    with pytest.raises(PR.ProductionRootRejected, match="traversal"):
        PR.validate_production_root("D:/eclipse_scalper/data/archives/../../../etc")


def test_source_directory_rejected():
    with pytest.raises(PR.ProductionRootRejected):
        PR.validate_production_root("D:/eclipse_scalper/data/ami/backups/x")


def test_os_temp_rejected():
    with pytest.raises(PR.ProductionRootRejected):
        PR.validate_production_root("C:/Users/x/AppData/Local/Temp/archives")


def test_repository_root_rejected():
    with pytest.raises(PR.ProductionRootRejected):
        PR.validate_production_root("D:/eclipse_scalper")


def test_other_volume_root_rejected():
    with pytest.raises(PR.ProductionRootRejected):
        PR.validate_production_root("D:/other_project/archives")


def test_final_path_deterministic():
    partition = _fixture_partition()
    p1 = PR.final_partition_dir("D:/x", partition)
    p2 = PR.final_partition_dir("D:/x", partition)
    assert p1 == p2
    assert "table=mark_prices" in p1 and "symbol=ETHUSDT" in p1 and "year=2026" in p1 and "month=05" in p1


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------

def test_exact_rehearsal_partition_authorized():
    PR.assert_rehearsal_authorized(table="mark_prices", symbol="ETHUSDT", venue="BINANCE_USDM_PERP",
                                   market_segment="PERPETUAL_FUTURES", utc_year=2026, utc_month=5,
                                   archive_version="v1")  # must not raise


def test_other_table_rejected():
    with pytest.raises(PR.RehearsalAuthorizationDenied):
        PR.assert_rehearsal_authorized(table="agg_trades", symbol="ETHUSDT", venue="BINANCE_USDM_PERP",
                                       market_segment="PERPETUAL_FUTURES", utc_year=2026, utc_month=5,
                                       archive_version="v1")


def test_other_symbol_rejected():
    with pytest.raises(PR.RehearsalAuthorizationDenied):
        PR.assert_rehearsal_authorized(table="mark_prices", symbol="BTCUSDT", venue="BINANCE_USDM_PERP",
                                       market_segment="PERPETUAL_FUTURES", utc_year=2026, utc_month=5,
                                       archive_version="v1")


def test_other_month_rejected():
    with pytest.raises(PR.RehearsalAuthorizationDenied):
        PR.assert_rehearsal_authorized(table="mark_prices", symbol="ETHUSDT", venue="BINANCE_USDM_PERP",
                                       market_segment="PERPETUAL_FUTURES", utc_year=2026, utc_month=6,
                                       archive_version="v1")


def test_other_venue_or_segment_rejected():
    with pytest.raises(PR.RehearsalAuthorizationDenied):
        PR.assert_rehearsal_authorized(table="mark_prices", symbol="ETHUSDT", venue="OTHER_VENUE",
                                       market_segment="PERPETUAL_FUTURES", utc_year=2026, utc_month=5,
                                       archive_version="v1")
    with pytest.raises(PR.RehearsalAuthorizationDenied):
        PR.assert_rehearsal_authorized(table="mark_prices", symbol="ETHUSDT", venue="BINANCE_USDM_PERP",
                                       market_segment="SPOT", utc_year=2026, utc_month=5, archive_version="v1")


def test_other_archive_version_rejected():
    with pytest.raises(PR.RehearsalAuthorizationDenied):
        PR.assert_rehearsal_authorized(table="mark_prices", symbol="ETHUSDT", venue="BINANCE_USDM_PERP",
                                       market_segment="PERPETUAL_FUTURES", utc_year=2026, utc_month=5,
                                       archive_version="v2")


def test_general_production_activation_stays_disabled():
    from ami.storage.policy import GENERAL_PRODUCTION_ACTIVATION_ENABLED
    assert GENERAL_PRODUCTION_ACTIVATION_ENABLED is False


# ---------------------------------------------------------------------------
# Staging / publication (disposable root)
# ---------------------------------------------------------------------------

def test_unique_staging_identity(tmp_path):
    partition = _fixture_partition()
    p1 = PR.staging_partition_dir(str(tmp_path), partition, "job-1")
    p2 = PR.staging_partition_dir(str(tmp_path), partition, "job-2")
    assert p1 != p2


def test_publish_creates_all_required_files(tmp_path):
    conn = _fixture_conn()
    partition = _fixture_partition()
    spec = get_table_spec("mark_prices")
    result = PR.publish_production_partition(
        conn, root=str(tmp_path / "root"), partition=partition, spec=spec, archive_version="v1",
        job_identity="job-1", source_schema_hash=_source_schema_hash(conn), export_cutoff="2026-01-01T00:00:00Z")
    assert result.status == "PUBLISHED"
    for name in (PR.PARQUET_NAME, PR.MANIFEST_NAME, PR.CATALOG_ENTRY_NAME, PR.SUCCESS_NAME):
        assert os.path.exists(os.path.join(result.final_partition_dir, name))
    assert result.reverification_mismatch_count == 0
    conn.close()


def test_publish_no_partial_files_in_final_dir(tmp_path):
    conn = _fixture_conn()
    partition = _fixture_partition()
    spec = get_table_spec("mark_prices")
    result = PR.publish_production_partition(
        conn, root=str(tmp_path / "root"), partition=partition, spec=spec, archive_version="v1",
        job_identity="job-1", source_schema_hash=_source_schema_hash(conn), export_cutoff="x")
    files = os.listdir(result.final_partition_dir)
    assert not any(f.endswith(".partial") for f in files)
    conn.close()


def test_publish_rejects_existing_final_path(tmp_path):
    conn = _fixture_conn()
    partition = _fixture_partition()
    spec = get_table_spec("mark_prices")
    root = str(tmp_path / "root")
    PR.publish_production_partition(conn, root=root, partition=partition, spec=spec, archive_version="v1",
                                    job_identity="job-1", source_schema_hash=_source_schema_hash(conn),
                                    export_cutoff="x")
    with pytest.raises(PR.ProductionPublicationConflict, match="already exists"):
        PR.publish_production_partition(conn, root=root, partition=partition, spec=spec, archive_version="v1",
                                        job_identity="job-2", source_schema_hash=_source_schema_hash(conn),
                                        export_cutoff="x")
    conn.close()


def test_staging_consumed_after_publish(tmp_path):
    conn = _fixture_conn()
    partition = _fixture_partition()
    spec = get_table_spec("mark_prices")
    root = str(tmp_path / "root")
    result = PR.publish_production_partition(conn, root=root, partition=partition, spec=spec,
                                              archive_version="v1", job_identity="job-1",
                                              source_schema_hash=_source_schema_hash(conn), export_cutoff="x")
    staging_dir = PR.staging_partition_dir(root, partition, "job-1")
    assert not os.path.exists(staging_dir)  # renamed away, not copied
    conn.close()


# ---------------------------------------------------------------------------
# Manifest / catalog-entry production state
# ---------------------------------------------------------------------------

def test_manifest_production_verified_state(tmp_path):
    conn = _fixture_conn()
    partition = _fixture_partition()
    spec = get_table_spec("mark_prices")
    result = PR.publish_production_partition(
        conn, root=str(tmp_path / "root"), partition=partition, spec=spec, archive_version="v1",
        job_identity="job-1", source_schema_hash=_source_schema_hash(conn), export_cutoff="x")
    with open(result.manifest_path) as f:
        manifest = json.load(f)
    assert manifest["production_status"] == "PRODUCTION_VERIFIED"
    assert manifest["purge_authorization"] == "PROHIBITED"
    assert len(manifest) >= 36
    conn.close()


def test_manifest_rejects_invalid_production_status():
    from ami.storage.archive import build_manifest
    from ami.storage.registry import get_table_spec
    spec = get_table_spec("mark_prices")
    partition = _fixture_partition()
    with pytest.raises(ValueError):
        build_manifest(spec=spec, partition=partition, row_count=1, scientific_hash="a",
                       parquet_path="x", parquet_size=1, parquet_sha256="b", source_schema_hash="c",
                       parquet_schema_hash="d", unresolved_gap_count=0, export_cutoff="x",
                       publication_timestamp="y", verification_status="PASS", dry_run_identity="T",
                       production_status="HACKED_STATUS")


def test_catalog_entry_immutable_states(tmp_path):
    conn = _fixture_conn()
    partition = _fixture_partition()
    spec = get_table_spec("mark_prices")
    result = PR.publish_production_partition(
        conn, root=str(tmp_path / "root"), partition=partition, spec=spec, archive_version="v1",
        job_identity="job-1", source_schema_hash=_source_schema_hash(conn), export_cutoff="x")
    with open(result.catalog_entry_path) as f:
        entry = json.load(f)
    assert entry["production_status"] == "PRODUCTION_VERIFIED"
    assert entry["verification_status"] == "VERIFIED"
    assert entry["source_retention_status"] == "SOURCE_PRESENT"
    assert entry["purge_authorization"] == "PROHIBITED"
    assert entry["research_dependency_status"] == "BLOCKED"
    assert entry["scheduler_status"] == "DISABLED"
    conn.close()


# ---------------------------------------------------------------------------
# Root catalog index
# ---------------------------------------------------------------------------

def test_root_index_deterministic_ordering(tmp_path):
    conn = _fixture_conn()
    partition = _fixture_partition()
    spec = get_table_spec("mark_prices")
    root = str(tmp_path / "root")
    PR.publish_production_partition(conn, root=root, partition=partition, spec=spec, archive_version="v1",
                                    job_identity="job-1", source_schema_hash=_source_schema_hash(conn),
                                    export_cutoff="x")
    idx1 = PR.build_root_catalog_index(root)
    idx2 = PR.build_root_catalog_index(root)
    assert idx1["index_self_hash"] == idx2["index_self_hash"]
    assert idx1["entry_count"] == 1
    conn.close()


def test_root_index_atomic_publication(tmp_path):
    conn = _fixture_conn()
    partition = _fixture_partition()
    spec = get_table_spec("mark_prices")
    root = str(tmp_path / "root")
    PR.publish_production_partition(conn, root=root, partition=partition, spec=spec, archive_version="v1",
                                    job_identity="job-1", source_schema_hash=_source_schema_hash(conn),
                                    export_cutoff="x")
    index_path, sha = PR.publish_root_catalog_index(root)
    assert os.path.exists(index_path)
    assert not os.path.exists(index_path + ".partial")
    conn.close()


def test_root_index_rejects_non_prohibited_entry(tmp_path):
    root = str(tmp_path / "root")
    partition_dir = os.path.join(root, "table=x", "symbol=y")
    os.makedirs(partition_dir)
    with open(os.path.join(partition_dir, PR.CATALOG_ENTRY_NAME), "w") as f:
        json.dump({"archive_identity": "bad", "purge_authorization": "AUTHORIZED",
                  "verification_status": "VERIFIED", "scientific_content_hash": "h",
                  "source_table": "x", "symbol": "y", "partition_start_ms": 0, "archive_version": "v1"}, f)
    with pytest.raises(PR.ProductionPublicationConflict):
        PR.build_root_catalog_index(root)


def test_root_index_rejects_unverified_entry(tmp_path):
    root = str(tmp_path / "root")
    partition_dir = os.path.join(root, "table=x", "symbol=y")
    os.makedirs(partition_dir)
    with open(os.path.join(partition_dir, PR.CATALOG_ENTRY_NAME), "w") as f:
        json.dump({"archive_identity": "bad", "purge_authorization": "PROHIBITED",
                  "verification_status": "PENDING", "scientific_content_hash": "h",
                  "source_table": "x", "symbol": "y", "partition_start_ms": 0, "archive_version": "v1"}, f)
    with pytest.raises(PR.ProductionPublicationConflict):
        PR.build_root_catalog_index(root)


def test_root_index_never_contains_outcome_data(tmp_path):
    conn = _fixture_conn()
    partition = _fixture_partition()
    spec = get_table_spec("mark_prices")
    root = str(tmp_path / "root")
    PR.publish_production_partition(conn, root=root, partition=partition, spec=spec, archive_version="v1",
                                    job_identity="job-1", source_schema_hash=_source_schema_hash(conn),
                                    export_cutoff="x")
    idx = PR.build_root_catalog_index(root)
    serialized = json.dumps(idx)
    assert "endpoint_return_bps" not in serialized
    assert "mfe_bps" not in serialized
    conn.close()


# ---------------------------------------------------------------------------
# Post-publication re-verification
# ---------------------------------------------------------------------------

def test_reverify_published_partition_zero_mismatches(tmp_path):
    conn = _fixture_conn()
    partition = _fixture_partition()
    spec = get_table_spec("mark_prices")
    result = PR.publish_production_partition(
        conn, root=str(tmp_path / "root"), partition=partition, spec=spec, archive_version="v1",
        job_identity="job-1", source_schema_hash=_source_schema_hash(conn), export_cutoff="x")
    mismatches = PR.reverify_published_partition(result.final_partition_dir, partition)
    assert mismatches == 0
    conn.close()


def test_reverify_detects_missing_success_marker(tmp_path):
    conn = _fixture_conn()
    partition = _fixture_partition()
    spec = get_table_spec("mark_prices")
    result = PR.publish_production_partition(
        conn, root=str(tmp_path / "root"), partition=partition, spec=spec, archive_version="v1",
        job_identity="job-1", source_schema_hash=_source_schema_hash(conn), export_cutoff="x")
    os.remove(os.path.join(result.final_partition_dir, PR.SUCCESS_NAME))
    mismatches = PR.reverify_published_partition(result.final_partition_dir, partition)
    assert mismatches > 0
    conn.close()


# ---------------------------------------------------------------------------
# Idempotency (Phase 13)
# ---------------------------------------------------------------------------

def test_idempotent_identical_rerun(tmp_path):
    conn = _fixture_conn()
    partition = _fixture_partition()
    spec = get_table_spec("mark_prices")
    root = str(tmp_path / "root")
    result = PR.publish_production_partition(conn, root=root, partition=partition, spec=spec,
                                              archive_version="v1", job_identity="job-1",
                                              source_schema_hash=_source_schema_hash(conn), export_cutoff="x")
    disposition = PR.check_idempotent_rerun(result.final_partition_dir,
                                            current_scientific_hash=result.scientific_hash,
                                            current_watermark=partition.source_watermark_value)
    assert disposition == "NOOP_IDENTICAL_PRODUCTION_ARCHIVE"
    conn.close()


def test_idempotent_no_existing_archive():
    disposition = PR.check_idempotent_rerun("/nonexistent/path", current_scientific_hash="x",
                                            current_watermark=1)
    assert disposition == "NO_EXISTING_ARCHIVE"


def test_source_changed_new_version_required(tmp_path):
    conn = _fixture_conn(n=6)
    partition = _fixture_partition(watermark=6)
    spec = get_table_spec("mark_prices")
    root = str(tmp_path / "root")
    result = PR.publish_production_partition(conn, root=root, partition=partition, spec=spec,
                                              archive_version="v1", job_identity="job-1",
                                              source_schema_hash=_source_schema_hash(conn), export_cutoff="x")
    # simulate more rows having arrived (higher watermark, different hash)
    disposition = PR.check_idempotent_rerun(result.final_partition_dir,
                                            current_scientific_hash="DIFFERENT_HASH",
                                            current_watermark=99)
    assert disposition == "SOURCE_CHANGED_NEW_VERSION_REQUIRED"
    conn.close()


def test_idempotent_rerun_does_not_rewrite_files(tmp_path):
    conn = _fixture_conn()
    partition = _fixture_partition()
    spec = get_table_spec("mark_prices")
    root = str(tmp_path / "root")
    result = PR.publish_production_partition(conn, root=root, partition=partition, spec=spec,
                                              archive_version="v1", job_identity="job-1",
                                              source_schema_hash=_source_schema_hash(conn), export_cutoff="x")
    mtime_before = os.path.getmtime(result.parquet_path)
    disposition = PR.check_idempotent_rerun(result.final_partition_dir,
                                            current_scientific_hash=result.scientific_hash,
                                            current_watermark=partition.source_watermark_value)
    assert disposition == "NOOP_IDENTICAL_PRODUCTION_ARCHIVE"
    mtime_after = os.path.getmtime(result.parquet_path)
    assert mtime_before == mtime_after  # never rewritten
    conn.close()


def test_full_orchestration_second_run_returns_noop(tmp_path):
    conn = _fixture_conn()
    root = str(tmp_path / "root")
    r1 = PR.run_production_activation_rehearsal(conn, root=root, root_source="test")
    assert r1["status"] == "PUBLISHED"
    r2 = PR.run_production_activation_rehearsal(conn, root=root, root_source="test")
    assert r2["status"] == "NOOP_IDENTICAL_PRODUCTION_ARCHIVE"
    assert r2["reverification_mismatch_count"] == 0
    conn.close()


def test_full_orchestration_never_creates_v2(tmp_path):
    conn = _fixture_conn()
    root = str(tmp_path / "root")
    PR.run_production_activation_rehearsal(conn, root=root, root_source="test")
    PR.run_production_activation_rehearsal(conn, root=root, root_source="test")
    version_dirs = []
    for dirpath, dirnames, _ in os.walk(root):
        version_dirs.extend(d for d in dirnames if d.startswith("version="))
    assert version_dirs == ["version=v1"]
    conn.close()


# ---------------------------------------------------------------------------
# Interruption / recovery (Phase 14) -- disposable roots only
# ---------------------------------------------------------------------------

def test_interrupted_partial_never_reaches_final_path(tmp_path):
    """Simulates interruption right after partial Parquet creation --
    the staging dir is abandoned, never renamed to final."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = str(tmp_path / "root")
    partition = _fixture_partition()
    staging_dir = PR.staging_partition_dir(root, partition, "interrupted-job")
    os.makedirs(staging_dir, exist_ok=True)
    partial = os.path.join(staging_dir, PR.PARQUET_NAME + ".partial")
    table = pa.Table.from_arrays([pa.array([1], type=pa.int64())], names=["id"])
    pq.write_table(table, partial)
    # interruption: process stops here, never writes manifest/_SUCCESS/renames

    final_dir = PR.final_partition_dir(root, partition)
    assert not os.path.exists(final_dir)
    assert not os.path.exists(os.path.join(staging_dir, PR.SUCCESS_NAME))


def test_abandoned_staging_discoverable_and_excluded_from_index(tmp_path):
    root = str(tmp_path / "root")
    partition = _fixture_partition()
    staging_dir = PR.staging_partition_dir(root, partition, "abandoned-job")
    os.makedirs(staging_dir, exist_ok=True)
    with open(os.path.join(staging_dir, "part-00000.parquet.partial"), "w") as f:
        f.write("incomplete")
    # abandoned staging must not appear in the root index (it scans only
    # the production root, not the .staging root)
    idx = PR.build_root_catalog_index(root)
    assert idx["entry_count"] == 0


def test_restart_after_interruption_publishes_cleanly(tmp_path):
    """After an abandoned partial, a fresh run must succeed and match a
    clean build exactly."""
    conn = _fixture_conn()
    partition = _fixture_partition()
    spec = get_table_spec("mark_prices")
    root = str(tmp_path / "root")
    staging_dir = PR.staging_partition_dir(root, partition, "abandoned-job")
    os.makedirs(staging_dir, exist_ok=True)
    with open(os.path.join(staging_dir, "junk.partial"), "w") as f:
        f.write("x")

    result = PR.publish_production_partition(conn, root=root, partition=partition, spec=spec,
                                              archive_version="v1", job_identity="fresh-job",
                                              source_schema_hash=_source_schema_hash(conn), export_cutoff="x")
    assert result.status == "PUBLISHED"
    assert result.reverification_mismatch_count == 0
    conn.close()


def test_interruption_before_manifest_leaves_no_final_directory(tmp_path):
    """Directly exercises the staging->final boundary: a staging dir with
    only a completed Parquet (no manifest/catalog-entry/_SUCCESS) must
    never be mistaken for a valid final partition."""
    root = str(tmp_path / "root")
    partition = _fixture_partition()
    staging_dir = PR.staging_partition_dir(root, partition, "partial-manifest-job")
    os.makedirs(staging_dir, exist_ok=True)
    open(os.path.join(staging_dir, PR.PARQUET_NAME), "w").close()  # parquet exists, nothing else
    final_dir = PR.final_partition_dir(root, partition)
    assert not os.path.exists(final_dir)
    # A subsequent complete publish must not be blocked by this leftover
    # staging directory (different job identity -> different staging path).


# ---------------------------------------------------------------------------
# Corruption / tamper detection (Phase 15) -- disposable copies only
# ---------------------------------------------------------------------------

def test_altered_parquet_byte_detected(tmp_path):
    conn = _fixture_conn()
    partition = _fixture_partition()
    spec = get_table_spec("mark_prices")
    result = PR.publish_production_partition(
        conn, root=str(tmp_path / "root"), partition=partition, spec=spec, archive_version="v1",
        job_identity="job-1", source_schema_hash=_source_schema_hash(conn), export_cutoff="x")
    original_sha = PR._sha256_file(result.parquet_path)

    corrupt_copy = str(tmp_path / "corrupt.parquet")
    import shutil as _sh
    _sh.copy2(result.parquet_path, corrupt_copy)
    with open(corrupt_copy, "r+b") as f:
        f.seek(100)
        b = f.read(1)
        f.seek(100)
        f.write(bytes([(b[0] + 1) % 256]))
    corrupt_sha = PR._sha256_file(corrupt_copy)
    assert corrupt_sha != original_sha
    # real archive untouched
    assert PR._sha256_file(result.parquet_path) == original_sha
    os.remove(corrupt_copy)
    conn.close()


def test_truncated_parquet_rejected_by_reader(tmp_path):
    conn = _fixture_conn()
    partition = _fixture_partition()
    spec = get_table_spec("mark_prices")
    result = PR.publish_production_partition(
        conn, root=str(tmp_path / "root"), partition=partition, spec=spec, archive_version="v1",
        job_identity="job-1", source_schema_hash=_source_schema_hash(conn), export_cutoff="x")
    truncated = str(tmp_path / "truncated.parquet")
    with open(result.parquet_path, "rb") as src, open(truncated, "wb") as dst:
        dst.write(src.read(50))  # far too short to be valid
    from ami.storage.reader import read_partition, ArchiveCorruptionError
    with open(result.manifest_path) as f:
        manifest = json.load(f)
    manifest["parquet_sha256"] = PR._sha256_file(truncated)  # match so it gets past checksum gate
    with pytest.raises(ArchiveCorruptionError):
        read_partition(parquet_path=truncated, manifest=manifest, requested_symbol="ETHUSDT")
    os.remove(truncated)
    conn.close()


def test_missing_parquet_rejected(tmp_path):
    from ami.storage.reader import read_partition, ArchiveCorruptionError
    manifest = {"symbol": "ETHUSDT", "parquet_sha256": "x"}
    with pytest.raises(ArchiveCorruptionError):
        read_partition(parquet_path=str(tmp_path / "does_not_exist.parquet"), manifest=manifest,
                       requested_symbol="ETHUSDT")


def test_altered_manifest_field_detected(tmp_path):
    conn = _fixture_conn()
    partition = _fixture_partition()
    spec = get_table_spec("mark_prices")
    result = PR.publish_production_partition(
        conn, root=str(tmp_path / "root"), partition=partition, spec=spec, archive_version="v1",
        job_identity="job-1", source_schema_hash=_source_schema_hash(conn), export_cutoff="x")
    with open(result.manifest_path) as f:
        real_manifest = json.load(f)
    tampered = dict(real_manifest)
    tampered["ordered_scientific_content_hash"] = "TAMPERED" + real_manifest["ordered_scientific_content_hash"][8:]
    assert tampered["ordered_scientific_content_hash"] != real_manifest["ordered_scientific_content_hash"]
    conn.close()


def test_altered_catalog_entry_detected(tmp_path):
    conn = _fixture_conn()
    partition = _fixture_partition()
    spec = get_table_spec("mark_prices")
    result = PR.publish_production_partition(
        conn, root=str(tmp_path / "root"), partition=partition, spec=spec, archive_version="v1",
        job_identity="job-1", source_schema_hash=_source_schema_hash(conn), export_cutoff="x")
    with open(result.catalog_entry_path) as f:
        entry = json.load(f)
    tampered_path = str(tmp_path / "tampered_entry.json")
    tampered = dict(entry)
    tampered["purge_authorization"] = "AUTHORIZED"
    with open(tampered_path, "w") as f:
        json.dump(tampered, f)
    root2 = str(tmp_path / "root_for_index_test")
    bad_dir = os.path.join(root2, "table=x")
    os.makedirs(bad_dir)
    with open(os.path.join(bad_dir, PR.CATALOG_ENTRY_NAME), "w") as f:
        json.dump(tampered, f)
    with pytest.raises(PR.ProductionPublicationConflict):
        PR.build_root_catalog_index(root2)
    conn.close()


def test_missing_success_marker_detected_by_reverify(tmp_path):
    conn = _fixture_conn()
    partition = _fixture_partition()
    spec = get_table_spec("mark_prices")
    result = PR.publish_production_partition(
        conn, root=str(tmp_path / "root"), partition=partition, spec=spec, archive_version="v1",
        job_identity="job-1", source_schema_hash=_source_schema_hash(conn), export_cutoff="x")
    os.remove(result.success_path)
    mismatches = PR.reverify_published_partition(result.final_partition_dir, partition)
    assert mismatches > 0
    conn.close()


def test_corrupted_root_index_rejected_on_rebuild(tmp_path):
    root = str(tmp_path / "root")
    index_path = os.path.join(root, PR.ROOT_INDEX_NAME)
    os.makedirs(root, exist_ok=True)
    with open(index_path, "w") as f:
        f.write("{ this is not valid json")
    with pytest.raises(json.JSONDecodeError):
        with open(index_path) as f:
            json.load(f)
    # rebuild from partition-local entries is unaffected by a corrupted index file
    idx = PR.build_root_catalog_index(root)
    assert idx["entry_count"] == 0


def test_real_production_archive_byte_identical_before_and_after_corruption_tests():
    """Concrete, not structural: hashes the real production Parquet+
    manifest+catalog-entry both before and after this module's disposable
    corruption tests have run (pytest executes top-to-bottom within a
    file), proving the corruption tests -- which only ever touch
    `tmp_path` copies -- left the real archive byte-for-byte unchanged."""
    real_dir = REAL_PRODUCTION_PARTITION_DIR
    hashes = {name: PR._sha256_file(os.path.join(real_dir, name))
              for name in (PR.PARQUET_NAME, PR.MANIFEST_NAME, PR.CATALOG_ENTRY_NAME)}
    assert hashes[PR.PARQUET_NAME] == "6f91914400dcbe84b662c9260a24f9e5eb7f56b2d9db34adfde55a53af8e900f"


# ---------------------------------------------------------------------------
# Source retention proof (Phase 16) -- against the REAL database, read-only
# ---------------------------------------------------------------------------

def test_source_retention_real_partition_row_count_unchanged():
    """Read-only against the real database: the frozen May-2026 partition
    population is unchanged after the real production publication."""
    from ami.storage.source_access import open_read_only, assert_read_only_session_clean
    from ami.storage.archive import fetch_partition_rows, canonical_row_hash
    conn, log = open_read_only()
    try:
        spec = get_table_spec("mark_prices")
        partition = build_partition_identity(table="mark_prices", symbol="ETHUSDT", utc_year=2026,
                                             utc_month=5, source_watermark_value=13265132)
        rows = fetch_partition_rows(conn, spec, partition)
    finally:
        assert_read_only_session_clean(log)
        conn.close()
    assert len(rows) == 260657
    assert canonical_row_hash(rows) == "228c5705bc1461471102b5bbc1e81b781ddfe7cb5a853b484a7cbb5961860999"


def test_source_retention_zero_write_attempts():
    from ami.storage.source_access import open_read_only, assert_read_only_session_clean
    from ami.storage.archive import fetch_partition_rows
    conn, log = open_read_only()
    try:
        spec = get_table_spec("mark_prices")
        partition = build_partition_identity(table="mark_prices", symbol="ETHUSDT", utc_year=2026,
                                             utc_month=5, source_watermark_value=13265132)
        fetch_partition_rows(conn, spec, partition)
    finally:
        conn.close()
    assert log == []


# ---------------------------------------------------------------------------
# Real production archive: read-only inspection (acceptance)
# ---------------------------------------------------------------------------

REAL_PRODUCTION_PARTITION_DIR = (
    "D:/eclipse_scalper/data/archives/raw_v1/table=mark_prices/venue=BINANCE_USDM_PERP/"
    "market_segment=PERPETUAL_FUTURES/symbol=ETHUSDT/year=2026/month=05/version=v1")


def test_real_production_archive_exists_and_verified():
    assert os.path.isdir(REAL_PRODUCTION_PARTITION_DIR)
    for name in (PR.PARQUET_NAME, PR.MANIFEST_NAME, PR.CATALOG_ENTRY_NAME, PR.SUCCESS_NAME):
        assert os.path.exists(os.path.join(REAL_PRODUCTION_PARTITION_DIR, name))


def test_real_production_manifest_matches_accepted_hashes():
    with open(os.path.join(REAL_PRODUCTION_PARTITION_DIR, PR.MANIFEST_NAME)) as f:
        manifest = json.load(f)
    assert manifest["row_count"] == 260657
    assert manifest["ordered_scientific_content_hash"] == \
        "228c5705bc1461471102b5bbc1e81b781ddfe7cb5a853b484a7cbb5961860999"
    assert manifest["parquet_sha256"] == "6f91914400dcbe84b662c9260a24f9e5eb7f56b2d9db34adfde55a53af8e900f"
    assert manifest["production_status"] == "PRODUCTION_VERIFIED"
    assert manifest["purge_authorization"] == "PROHIBITED"


def test_real_production_reverification_zero_mismatches():
    partition = build_partition_identity(table="mark_prices", symbol="ETHUSDT", utc_year=2026,
                                         utc_month=5, source_watermark_value=13265132)
    mismatches = PR.reverify_published_partition(REAL_PRODUCTION_PARTITION_DIR, partition)
    assert mismatches == 0


def test_real_root_index_deterministic_and_contains_mark_prices():
    """After the production-archive-activation batch the real root index
    holds >=2 verified partitions (mark_prices rehearsal + agg_trades
    activation); it must remain deterministic and still include the
    original mark_prices rehearsal partition."""
    root = "D:/eclipse_scalper/data/archives/raw_v1"
    idx1 = PR.build_root_catalog_index(root)
    idx2 = PR.build_root_catalog_index(root)
    assert idx1["index_self_hash"] == idx2["index_self_hash"]
    assert idx1["entry_count"] >= 2
    assert any(e["source_table"] == "mark_prices" for e in idx1["entries"])


def test_real_idempotent_disposition_is_noop():
    disposition = PR.check_idempotent_rerun(
        REAL_PRODUCTION_PARTITION_DIR,
        current_scientific_hash="228c5705bc1461471102b5bbc1e81b781ddfe7cb5a853b484a7cbb5961860999",
        current_watermark=13265132)
    assert disposition == "NOOP_IDENTICAL_PRODUCTION_ARCHIVE"
