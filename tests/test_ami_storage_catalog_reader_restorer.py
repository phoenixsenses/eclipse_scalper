"""Focused tests: ami.storage.catalog + ami.storage.reader +
ami.storage.restorer.
"""
from __future__ import annotations

import hashlib
import os

import pytest

from ami.storage import catalog as C
from ami.storage import reader as RD
from ami.storage import restorer as RS
from ami.storage.registry import get_table_spec
from ami.storage.verifier import VERIFIED_DISPOSABLE, FAILED_CHECKSUM


def _entry(partition_id="p1", archive_path=None, manifest_path=None, disposable_root=None,
          content_hash="h1", state=VERIFIED_DISPOSABLE, purge="PROHIBITED"):
    return C.CatalogEntry(
        partition_id=partition_id, table="mark_prices", symbol="ETHUSDT", utc_year=2026, utc_month=5,
        archive_path=archive_path or os.path.join(disposable_root, "a.parquet"),
        manifest_path=manifest_path or os.path.join(disposable_root, "m.json"),
        verification_state=state, scientific_content_hash=content_hash, parquet_sha256="s1",
        source_watermark_value=100, unresolved_gap_count=0, repair_status="NOT_APPLICABLE",
        production_status="DISPOSABLE_NOT_PRODUCTION", purge_authorization=purge)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

def test_catalog_registers_verified_entry(tmp_path):
    cat = C.DisposableArchiveCatalog(str(tmp_path))
    e = _entry(disposable_root=str(tmp_path))
    result = cat.register(e)
    assert result.partition_id == "p1"
    assert cat.get("p1") is not None


def test_catalog_rejects_unverified_registration(tmp_path):
    cat = C.DisposableArchiveCatalog(str(tmp_path))
    e = _entry(disposable_root=str(tmp_path), state=FAILED_CHECKSUM)
    with pytest.raises(C.CatalogUnverifiedRegistrationError):
        cat.register(e)


def test_catalog_rejects_non_prohibited_purge(tmp_path):
    cat = C.DisposableArchiveCatalog(str(tmp_path))
    e = _entry(disposable_root=str(tmp_path), purge="AUTHORIZED")
    with pytest.raises(C.CatalogConflictError):
        cat.register(e)


def test_catalog_rejects_path_escape(tmp_path):
    cat = C.DisposableArchiveCatalog(str(tmp_path / "sub"))
    e = _entry(archive_path=str(tmp_path / "outside" / "a.parquet"),
              manifest_path=str(tmp_path / "outside" / "m.json"))
    with pytest.raises(C.CatalogPathEscapeError):
        cat.register(e)


def test_catalog_rejects_production_path(tmp_path):
    prod = str(tmp_path / "production")
    cat = C.DisposableArchiveCatalog(str(tmp_path), production_roots_to_reject=(prod,))
    e = _entry(archive_path=os.path.join(prod, "a.parquet"), manifest_path=os.path.join(prod, "m.json"))
    with pytest.raises(C.CatalogProductionPathError):
        cat.register(e)


def test_catalog_idempotent_reregistration_same_content(tmp_path):
    cat = C.DisposableArchiveCatalog(str(tmp_path))
    e = _entry(disposable_root=str(tmp_path))
    cat.register(e)
    result2 = cat.register(_entry(disposable_root=str(tmp_path)))  # identical content hash
    assert result2.partition_id == "p1"
    assert len(cat.all_entries()) == 1


def test_catalog_rejects_conflicting_identity(tmp_path):
    cat = C.DisposableArchiveCatalog(str(tmp_path))
    cat.register(_entry(disposable_root=str(tmp_path), content_hash="h1"))
    with pytest.raises(C.CatalogConflictError):
        cat.register(_entry(disposable_root=str(tmp_path), content_hash="h2_DIFFERENT"))


def test_catalog_verified_entry_immutable_via_new_version(tmp_path):
    cat = C.DisposableArchiveCatalog(str(tmp_path))
    cat.register(_entry(partition_id="p1", disposable_root=str(tmp_path), content_hash="h1"))
    new_entry = _entry(partition_id="p2", disposable_root=str(tmp_path), content_hash="h2")
    cat.register_new_version("p1", new_entry)
    assert cat.get("p1").scientific_content_hash == "h1"  # untouched
    assert cat.get("p2").scientific_content_hash == "h2"


def test_catalog_history_preserved(tmp_path):
    cat = C.DisposableArchiveCatalog(str(tmp_path))
    cat.register(_entry(partition_id="p1", disposable_root=str(tmp_path)))
    assert len(cat.history("p1")) == 1


def test_catalog_new_version_requires_distinct_id(tmp_path):
    cat = C.DisposableArchiveCatalog(str(tmp_path))
    cat.register(_entry(partition_id="p1", disposable_root=str(tmp_path)))
    with pytest.raises(C.CatalogConflictError):
        cat.register_new_version("p1", _entry(partition_id="p1", disposable_root=str(tmp_path), content_hash="h2"))


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def _write_tiny_parquet(path, rows=((1, "ETHUSDT"),)):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_arrays(
        [pa.array([r[0] for r in rows], type=pa.int64()), pa.array([r[1] for r in rows], type=pa.string())],
        schema=pa.schema([pa.field("id", pa.int64()), pa.field("symbol", pa.string())]))
    pq.write_table(table, path, compression="zstd")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def test_reader_requires_manifest(tmp_path):
    p = str(tmp_path / "a.parquet")
    _write_tiny_parquet(p)
    with pytest.raises(RD.ManifestRequiredError):
        RD.read_partition(parquet_path=p, manifest={}, requested_symbol="ETHUSDT")


def test_reader_rejects_symbol_mismatch(tmp_path):
    p = str(tmp_path / "a.parquet")
    _write_tiny_parquet(p)
    manifest = {"symbol": "BTCUSDT", "parquet_sha256": _sha(p)}
    with pytest.raises(RD.PartitionMismatchError):
        RD.read_partition(parquet_path=p, manifest=manifest, requested_symbol="ETHUSDT")


def test_reader_rejects_checksum_mismatch(tmp_path):
    p = str(tmp_path / "a.parquet")
    _write_tiny_parquet(p)
    manifest = {"symbol": "ETHUSDT", "parquet_sha256": "WRONG_HASH"}
    with pytest.raises(RD.ArchiveCorruptionError):
        RD.read_partition(parquet_path=p, manifest=manifest, requested_symbol="ETHUSDT")


def test_reader_succeeds_with_matching_manifest(tmp_path):
    p = str(tmp_path / "a.parquet")
    _write_tiny_parquet(p, rows=((1, "ETHUSDT"), (2, "ETHUSDT")))
    manifest = {"symbol": "ETHUSDT", "parquet_sha256": _sha(p), "partition_id": "px",
                "verification_status": "PASS"}
    result = RD.read_partition(parquet_path=p, manifest=manifest, requested_symbol="ETHUSDT")
    assert result.row_count == 2
    assert result.partition_id == "px"


def test_reader_rejects_missing_file(tmp_path):
    manifest = {"symbol": "ETHUSDT", "parquet_sha256": "x"}
    with pytest.raises(RD.ArchiveCorruptionError):
        RD.read_partition(parquet_path=str(tmp_path / "missing.parquet"), manifest=manifest,
                          requested_symbol="ETHUSDT")


def test_reader_column_projection(tmp_path):
    p = str(tmp_path / "a.parquet")
    _write_tiny_parquet(p, rows=((1, "ETHUSDT"),))
    manifest = {"symbol": "ETHUSDT", "parquet_sha256": _sha(p), "partition_id": "px",
                "verification_status": "PASS"}
    result = RD.read_partition(parquet_path=p, manifest=manifest, requested_symbol="ETHUSDT", columns=("id",))
    assert result.rows == [(1,)]


# ---------------------------------------------------------------------------
# Restorer
# ---------------------------------------------------------------------------

def test_restorer_rejects_destination_outside_approved_roots(tmp_path):
    spec = get_table_spec("mark_prices")
    rows = [(1, 1777593600001, "ETHUSDT", 3000.0, None, None)]
    h = RS._canonical_row_hash(rows)
    with pytest.raises(RS.RestoreDestinationRejected):
        RS.restore_slice(destination_path=str(tmp_path / "not_approved" / "x.sqlite"), spec=spec,
                         rows=rows, manifest={"ordered_scientific_content_hash": h},
                         expected_scientific_hash=h)


def test_restorer_succeeds_in_runtime_temp_style_path(tmp_path):
    spec = get_table_spec("mark_prices")
    rows = [(1, 1777593600001, "ETHUSDT", 3000.0, None, None),
            (2, 1777593600002, "ETHUSDT", 3001.0, 0.0001, 1777600000000)]
    h = RS._canonical_row_hash(rows)
    dest = str(tmp_path / ".runtime_temp" / "x.sqlite")
    result = RS.restore_slice(destination_path=dest, spec=spec, rows=rows,
                              manifest={"ordered_scientific_content_hash": h}, expected_scientific_hash=h)
    assert result.row_count == 2
    assert result.scientific_content_hash == h
    assert os.path.exists(dest)


def test_restorer_rejects_manifest_mismatch(tmp_path):
    spec = get_table_spec("mark_prices")
    rows = [(1, 1777593600001, "ETHUSDT", 3000.0, None, None)]
    dest = str(tmp_path / ".runtime_temp" / "x.sqlite")
    with pytest.raises(RS.RestoreManifestMismatchError):
        RS.restore_slice(destination_path=dest, spec=spec, rows=rows,
                         manifest={"ordered_scientific_content_hash": "WRONG"},
                         expected_scientific_hash="WRONG")


def test_restorer_refuses_overwrite_nonempty(tmp_path):
    spec = get_table_spec("mark_prices")
    rows = [(1, 1777593600001, "ETHUSDT", 3000.0, None, None)]
    h = RS._canonical_row_hash(rows)
    dest = str(tmp_path / ".runtime_temp" / "x.sqlite")
    RS.restore_slice(destination_path=dest, spec=spec, rows=rows,
                     manifest={"ordered_scientific_content_hash": h}, expected_scientific_hash=h)
    with pytest.raises(RS.RestoreDestinationRejected):
        RS.restore_slice(destination_path=dest, spec=spec, rows=rows,
                         manifest={"ordered_scientific_content_hash": h}, expected_scientific_hash=h)


def test_restorer_cleanup_removes_own_output(tmp_path):
    spec = get_table_spec("mark_prices")
    rows = [(1, 1777593600001, "ETHUSDT", 3000.0, None, None)]
    h = RS._canonical_row_hash(rows)
    dest = str(tmp_path / ".runtime_temp" / "x.sqlite")
    RS.restore_slice(destination_path=dest, spec=spec, rows=rows,
                     manifest={"ordered_scientific_content_hash": h}, expected_scientific_hash=h)
    assert RS.cleanup_restored_slice(dest) is True
    assert not os.path.exists(dest)


def test_restorer_cleanup_rejects_path_outside_temp_roots(tmp_path):
    with pytest.raises(RS.RestoreDestinationRejected):
        RS.cleanup_restored_slice(str(tmp_path / "not_temp" / "x.sqlite"))
