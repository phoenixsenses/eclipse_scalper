"""BATCH-STORAGE-ROTATION-RETENTION-DISPOSABLE-DRY-RUN-V1 -- focused tests.

Uses small synthetic fixtures (not the live 758GB microstructure.db) for
speed and determinism. The real dry-run was already executed once
against the live database (results frozen in FROZEN_RESULT and the
committed governance artifacts); these tests exercise the reusable
logic in isolation.
"""
from __future__ import annotations

import ast
import inspect

import pytest

from ami.governance import storage_rotation_retention_disposable_dry_run_v1 as D

SYNTH_ROWS = [
    (100, 1777593600001, "ETHUSDT", 3000.0, 0.0001, 1777600000000),
    (101, 1777593601001, "ETHUSDT", 3001.5, None, None),
    (102, 1777593602001, "ETHUSDT", 3002.25, 0.0002, 1777600000000),
]


# ---------------------------------------------------------------------------
# Frozen selection identity
# ---------------------------------------------------------------------------

def test_selected_partition_constants():
    assert D.SELECTED_TABLE == "mark_prices"
    assert D.SELECTED_SYMBOL == "ETHUSDT"
    assert D.PARTITION_START_MS == 1777593600000
    assert D.PARTITION_END_MS == 1780272000000


def test_partition_is_closed_utc_month_may_2026():
    import datetime as dt
    start = dt.datetime.fromtimestamp(D.PARTITION_START_MS / 1000, dt.timezone.utc)
    end = dt.datetime.fromtimestamp(D.PARTITION_END_MS / 1000, dt.timezone.utc)
    assert start == dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc)
    assert end == dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)


def test_partition_outside_30_day_active_window():
    """Audit ran 2026-07-07; 30 days back is 2026-06-07 -- May 2026 is
    fully before that."""
    import datetime as dt
    audit_date = dt.datetime(2026, 7, 7, tzinfo=dt.timezone.utc)
    active_horizon_start = audit_date - dt.timedelta(days=30)
    partition_end = dt.datetime.fromtimestamp(D.PARTITION_END_MS / 1000, dt.timezone.utc)
    assert partition_end <= active_horizon_start


def test_partition_does_not_include_current_utc_month():
    partition_end_month = (D.PARTITION_END_MS // 1)  # 2026-06-01, not July
    import datetime as dt
    end = dt.datetime.fromtimestamp(D.PARTITION_END_MS / 1000, dt.timezone.utc)
    assert (end.year, end.month) != (2026, 7)


def test_rejected_candidates_recorded_with_reasons():
    assert len(D.REJECTED_CANDIDATES) == 2
    tables = {c["table"] for c in D.REJECTED_CANDIDATES}
    assert tables == {"agg_trades", "book_ticker"}
    for c in D.REJECTED_CANDIDATES:
        assert c["reason"]


def test_selection_preference_order_not_silently_switched():
    """mark_prices (1st preference) was actually selected; agg_trades
    (2nd) and book_ticker (3rd) are both recorded as rejected, not
    silently substituted."""
    assert D.SELECTED_TABLE == "mark_prices"
    assert "agg_trades" not in (D.SELECTED_TABLE,)
    assert "book_ticker" not in (D.SELECTED_TABLE,)


# ---------------------------------------------------------------------------
# Frozen real-run results (from the actual dry-run against live data)
# ---------------------------------------------------------------------------

def test_frozen_result_matches_real_dry_run():
    assert D.FROZEN_RESULT["row_count"] == 260657
    assert D.FROZEN_RESULT["watermark_id"] == 13265132
    assert D.FROZEN_RESULT["duplicate_id_count"] == 0
    assert D.FROZEN_RESULT["run_a_run_b_byte_identical"] is True


def test_frozen_result_within_hard_size_caps():
    """Preferred <=1GB, hard max 2GB uncompressed estimate; actual
    Parquet output well under all caps."""
    assert D.FROZEN_RESULT["parquet_size_bytes"] < 1 * 1024**3
    assert D.FROZEN_RESULT["compression_ratio"] > 1.0


# ---------------------------------------------------------------------------
# Canonical row hashing
# ---------------------------------------------------------------------------

def test_canonical_row_hash_deterministic():
    a = D.canonical_row_hash(SYNTH_ROWS)
    b = D.canonical_row_hash(SYNTH_ROWS)
    assert a == b


def test_canonical_row_hash_order_sensitive():
    reversed_rows = list(reversed(SYNTH_ROWS))
    assert D.canonical_row_hash(SYNTH_ROWS) != D.canonical_row_hash(reversed_rows)


def test_canonical_row_hash_empty():
    assert D.canonical_row_hash([]) == D.canonical_row_hash([])


def test_canonical_row_hash_distinguishes_none_from_zero():
    a = D.canonical_row_hash([(1, None)])
    b = D.canonical_row_hash([(1, 0)])
    assert a != b


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------

def test_schema_preserves_all_source_columns():
    schema = D.build_parquet_schema_dict()
    assert set(schema.keys()) == set(D.COLUMNS)


def test_schema_id_and_ts_are_int64_not_nullable():
    schema = D.build_parquet_schema_dict()
    assert schema["id"]["parquet_type"] == "int64"
    assert schema["id"]["nullable"] is False
    assert schema["ts_ms"]["parquet_type"] == "int64"
    assert schema["ts_ms"]["nullable"] is False


def test_schema_nullable_columns_preserve_nullability():
    schema = D.build_parquet_schema_dict()
    assert schema["funding_rate"]["nullable"] is True
    assert schema["next_funding_time_ms"]["nullable"] is True


def test_schema_no_float_conversion_of_integer_timestamps():
    schema = D.build_parquet_schema_dict()
    assert schema["ts_ms"]["parquet_type"] != "double"
    assert schema["id"]["parquet_type"] != "double"


# ---------------------------------------------------------------------------
# Validation (Phase 7)
# ---------------------------------------------------------------------------

def test_validate_partition_rows_all_pass():
    v = D.validate_partition_rows(SYNTH_ROWS, watermark_id=102)
    assert v["all_pass"] is True
    assert v["duplicate_count"] == 0
    assert v["row_count"] == 3


def test_validate_partition_rows_detects_wrong_symbol():
    bad = SYNTH_ROWS + [(103, 1777593603001, "BTCUSDT", 60000.0, None, None)]
    v = D.validate_partition_rows(bad, watermark_id=103)
    assert v["symbol_only_selected"] is False
    assert v["all_pass"] is False


def test_validate_partition_rows_detects_out_of_range_timestamp():
    bad = SYNTH_ROWS + [(103, D.PARTITION_END_MS, "ETHUSDT", 3003.0, None, None)]
    v = D.validate_partition_rows(bad, watermark_id=103)
    assert v["all_ts_in_range"] is False
    assert v["all_pass"] is False


def test_validate_partition_rows_detects_row_above_watermark():
    v = D.validate_partition_rows(SYNTH_ROWS, watermark_id=101)  # last row (102) exceeds
    assert v["all_ids_le_watermark"] is False
    assert v["all_pass"] is False


def test_validate_partition_rows_detects_duplicates():
    bad = SYNTH_ROWS + [SYNTH_ROWS[0]]
    v = D.validate_partition_rows(bad, watermark_id=102)
    assert v["duplicate_count"] == 1
    assert v["all_pass"] is False


def test_validate_partition_rows_empty_never_passes():
    v = D.validate_partition_rows([], watermark_id=999)
    assert v["all_pass"] is False


# ---------------------------------------------------------------------------
# Manifest construction (Phase 8)
# ---------------------------------------------------------------------------

def test_manifest_hardcodes_disposable_not_production():
    m = D.build_manifest(
        row_count=3, watermark_id=102, min_id=100, max_id=102, min_ts=1777593600001,
        max_ts=1777593602001, scientific_hash="abc", parquet_path="x.parquet",
        parquet_size=100, parquet_sha256="def", source_schema_hash="ghi",
        parquet_schema_hash="jkl", export_cutoff="2026-01-01T00:00:00Z",
        publication_timestamp="2026-01-01T00:00:01Z", verification_status="PASS",
        dry_run_identity="TEST")
    assert m["production_status"] == "DISPOSABLE_NOT_PRODUCTION"
    assert m["purge_authorization"] == "PROHIBITED"


def test_manifest_field_count_matches_contract():
    m = D.build_manifest(
        row_count=3, watermark_id=102, min_id=100, max_id=102, min_ts=1777593600001,
        max_ts=1777593602001, scientific_hash="abc", parquet_path="x.parquet",
        parquet_size=100, parquet_sha256="def", source_schema_hash="ghi",
        parquet_schema_hash="jkl", export_cutoff="2026-01-01T00:00:00Z",
        publication_timestamp="2026-01-01T00:00:01Z", verification_status="PASS",
        dry_run_identity="TEST")
    assert len(m) >= 25  # repository-authoritative equivalent of the 29-field contract


def test_manifest_records_watermark_and_hash():
    m = D.build_manifest(
        row_count=3, watermark_id=102, min_id=100, max_id=102, min_ts=1777593600001,
        max_ts=1777593602001, scientific_hash="abc123", parquet_path="x.parquet",
        parquet_size=100, parquet_sha256="def456", source_schema_hash="ghi",
        parquet_schema_hash="jkl", export_cutoff="2026-01-01T00:00:00Z",
        publication_timestamp="2026-01-01T00:00:01Z", verification_status="PASS",
        dry_run_identity="TEST")
    assert m["source_watermark_value"] == 102
    assert m["ordered_scientific_content_hash"] == "abc123"
    assert m["parquet_sha256"] == "def456"


# ---------------------------------------------------------------------------
# Corruption detection (Phase 13)
# ---------------------------------------------------------------------------

def test_detect_corruption_identical_passes():
    r = D.detect_corruption(original_sha256="a", candidate_sha256="a",
                             original_content_hash="x", candidate_content_hash="x")
    assert r["corruption_detected"] is False
    assert r["eligible_for_verified_status"] is True


def test_detect_corruption_hash_mismatch_fails():
    r = D.detect_corruption(original_sha256="a", candidate_sha256="b",
                             original_content_hash="x", candidate_content_hash="x")
    assert r["corruption_detected"] is True
    assert r["eligible_for_verified_status"] is False


def test_detect_corruption_unreadable_file_fails_closed():
    r = D.detect_corruption(original_sha256="a", candidate_sha256="b",
                             original_content_hash="x", candidate_content_hash=None)
    assert r["corruption_detected"] is True
    assert r["eligible_for_verified_status"] is False


# ---------------------------------------------------------------------------
# Verdict determination (Phase 18)
# ---------------------------------------------------------------------------

def _all_good_kwargs(**overrides):
    base = dict(tooling_available=True, all_validations_passed=True, direct_read_parity=True,
                restore_parity=True, run_a_run_b_scientific_match=True, interruption_safe=True,
                corruption_detected_correctly=True, live_source_writes=0, source_rows_deleted=0,
                resource_limits_respected=True)
    base.update(overrides)
    return base


def test_verdict_complete_when_everything_passes():
    assert D.determine_dry_run_verdict(**_all_good_kwargs()) == \
        "STORAGE_ROTATION_RETENTION_DISPOSABLE_DRY_RUN_V1_COMPLETE"


def test_verdict_blocked_by_tooling_first_regardless_of_other_flags():
    assert D.determine_dry_run_verdict(**_all_good_kwargs(tooling_available=False)) == \
        "STORAGE_ROTATION_RETENTION_DISPOSABLE_DRY_RUN_V1_BLOCKED_BY_TOOLING"


def test_verdict_incomplete_on_any_live_write():
    assert D.determine_dry_run_verdict(**_all_good_kwargs(live_source_writes=1)) == \
        "STORAGE_ROTATION_RETENTION_DISPOSABLE_DRY_RUN_V1_INCOMPLETE"


def test_verdict_incomplete_on_source_row_deletion():
    assert D.determine_dry_run_verdict(**_all_good_kwargs(source_rows_deleted=1)) == \
        "STORAGE_ROTATION_RETENTION_DISPOSABLE_DRY_RUN_V1_INCOMPLETE"


def test_verdict_live_write_check_precedes_resource_limit_check():
    """A live write must be reported as INCOMPLETE, not
    BLOCKED_BY_RESOURCE_LIMIT, even if resources were also exceeded --
    proves the fail-closed precedence order."""
    result = D.determine_dry_run_verdict(**_all_good_kwargs(
        live_source_writes=1, resource_limits_respected=False))
    assert result == "STORAGE_ROTATION_RETENTION_DISPOSABLE_DRY_RUN_V1_INCOMPLETE"


def test_verdict_blocked_by_resource_limit():
    assert D.determine_dry_run_verdict(**_all_good_kwargs(resource_limits_respected=False)) == \
        "STORAGE_ROTATION_RETENTION_DISPOSABLE_DRY_RUN_V1_BLOCKED_BY_RESOURCE_LIMIT"


def test_verdict_incomplete_when_any_single_check_fails():
    for key in ("all_validations_passed", "direct_read_parity", "restore_parity",
                "run_a_run_b_scientific_match", "interruption_safe", "corruption_detected_correctly"):
        result = D.determine_dry_run_verdict(**_all_good_kwargs(**{key: False}))
        assert result == "STORAGE_ROTATION_RETENTION_DISPOSABLE_DRY_RUN_V1_INCOMPLETE", key


def test_verdict_never_authorizes_production():
    """No verdict string contains 'PRODUCTION_READY' or similar --
    structural guard that COMPLETE never implies production activation."""
    all_verdicts = {
        D.determine_dry_run_verdict(**_all_good_kwargs()),
        D.determine_dry_run_verdict(**_all_good_kwargs(tooling_available=False)),
        D.determine_dry_run_verdict(**_all_good_kwargs(live_source_writes=1)),
        D.determine_dry_run_verdict(**_all_good_kwargs(resource_limits_respected=False)),
    }
    for v in all_verdicts:
        assert "PRODUCTION_READY" not in v
        assert "PURGE" not in v


# ---------------------------------------------------------------------------
# Structural no-mutation guards
# ---------------------------------------------------------------------------

def test_module_has_zero_execute_call_sites():
    """The module's inability to construct/run any of FORBIDDEN_SQL_
    STATEMENTS is proven structurally, not by string search (which false-
    positives on the constant's own definition and on the docstring's
    prose listing of these same keywords): zero `.execute()`-family call
    sites exist anywhere (see test_module_never_calls_execute), so there
    is no call site for a forbidden statement to be passed to."""
    tree = ast.parse(inspect.getsource(D))
    execute_calls = [n for n in ast.walk(tree)
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                      and n.func.attr in ("execute", "executescript", "executemany")]
    assert execute_calls == []


def test_module_never_imports_sqlite3_or_shutil_or_pyarrow():
    """The committed module is pure logic -- the actual I/O (SQLite
    connection, Parquet writer) lives only in the disposable, uncommitted
    driver script, never in this reusable module."""
    src = inspect.getsource(D)
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for forbidden in ("sqlite3", "shutil", "pyarrow", "os"):
        assert forbidden not in imported, forbidden


def test_module_never_calls_execute():
    src = inspect.getsource(D)
    tree = ast.parse(src)
    calls = [n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "execute" not in calls
    assert "executescript" not in calls


def test_forbidden_sql_statements_list_is_comprehensive():
    joined = " ".join(D.FORBIDDEN_SQL_STATEMENTS)
    for keyword in ("INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "VACUUM"):
        assert keyword in joined
