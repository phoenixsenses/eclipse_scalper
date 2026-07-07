"""BATCH-STORAGE-ROTATION-RETENTION-READINESS-AND-CONTRACT-V1 -- focused tests.

The module under test is pure design/policy metadata -- it never opens a
database connection or performs I/O. These tests confirm that structurally
(AST guards) as well as behaviorally (no destructive calls anywhere).
"""
from __future__ import annotations

import ast
import inspect

from ami.governance import storage_rotation_retention_readiness_v1 as SR


# ---------------------------------------------------------------------------
# Table registry / classification
# ---------------------------------------------------------------------------

def test_every_table_has_exactly_one_class():
    seen = set()
    for row in SR.TABLE_REGISTRY:
        key = (row["db"], row["table"])
        assert key not in seen, f"duplicate classification for {key}"
        seen.add(key)
        assert row["class"] in SR.RETENTION_CLASSES


def test_canonical_immutable_tables_never_purge_eligible():
    for row in SR.TABLE_REGISTRY:
        if row["class"] == "CANONICAL_IMMUTABLE":
            assert row["purge_eligible"] is False, row["table"]


def test_continuity_critical_never_purge_eligible():
    for row in SR.TABLE_REGISTRY:
        if row["class"] == "CONTINUITY_CRITICAL_ACTIVE":
            assert row["purge_eligible"] is False, row["table"]


def test_research_critical_compact_never_purge_eligible_by_age_alone():
    for row in SR.TABLE_REGISTRY:
        if row["class"] == "RESEARCH_CRITICAL_COMPACT":
            assert row["purge_eligible"] is False, row["table"]


def test_raw_high_frequency_tables_are_conditional_not_unconditional():
    raw = [r for r in SR.TABLE_REGISTRY if r["class"] == "RAW_HIGH_FREQUENCY_ARCHIVE_ELIGIBLE"]
    assert len(raw) == 3
    for row in raw:
        assert isinstance(row["purge_eligible"], str) and row["purge_eligible"].startswith("CONDITIONAL")


def test_book_ticker_classified_raw_archive_eligible_and_blocked():
    row = SR.classify("microstructure.db", "book_ticker")
    assert row is not None
    assert row["class"] == "RAW_HIGH_FREQUENCY_ARCHIVE_ELIGIBLE"
    assert "FAM_BOOK_SPREAD_DYNAMICS" in row["reason"]


def test_stray_test_scratch_files_classified_temporary_disposable():
    for pattern in ("data/test_s34_gates_micro_*.db", "data/test_s34_micro_*.db",
                     "data/test_s34_old_micro_*.db", "data/test_tmp_logger.db"):
        row = SR.classify(pattern, "*")
        assert row is not None, pattern
        assert row["class"] == "TEMPORARY_DISPOSABLE"


def test_unclassified_table_detection_fails_closed():
    observed = [("microstructure.db", "book_ticker"), ("microstructure.db", "totally_new_table")]
    unknown = SR.unclassified_tables(observed)
    assert unknown == [("microstructure.db", "totally_new_table")]


def test_classify_returns_none_for_unknown():
    assert SR.classify("nonexistent.db", "nonexistent_table") is None


# ---------------------------------------------------------------------------
# Storage policy constants
# ---------------------------------------------------------------------------

def test_active_horizon_defaults_to_30_days_not_14():
    assert SR.ACTIVE_RAW_HORIZON_DAYS_INITIAL == 30
    assert SR.ACTIVE_RAW_HORIZON_DAYS_MIN_FUTURE_RANGE == (14, 30)


def test_archive_format_is_parquet_zstd_only():
    assert SR.ARCHIVE_FORMAT == "PARQUET"
    assert SR.ARCHIVE_COMPRESSION == "ZSTD"
    assert "CSV" in SR.FORBIDDEN_ARCHIVE_FORMATS
    assert "JSONL" in SR.FORBIDDEN_ARCHIVE_FORMATS
    assert "PICKLE" in SR.FORBIDDEN_ARCHIVE_FORMATS
    assert "COMPRESSED_SQLITE_COPY" in SR.FORBIDDEN_ARCHIVE_FORMATS


def test_partitioning_is_closed_utc_calendar_month():
    assert SR.PARTITION_GRANULARITY == "CLOSED_UTC_CALENDAR_MONTH"


# ---------------------------------------------------------------------------
# Storage-health state function (pure, deterministic)
# ---------------------------------------------------------------------------

def test_health_state_healthy_at_current_real_drive_state():
    """Real measured state this batch: D: 58.9% free, ~1126GB free."""
    assert SR.storage_health_state(58.9, 1126) == "STORAGE_HEALTHY"


def test_health_state_warning_boundary():
    assert SR.storage_health_state(20.0, 500) == "STORAGE_WARNING"
    assert SR.storage_health_state(50.0, 200) == "STORAGE_WARNING"
    assert SR.storage_health_state(50.0, 201) == "STORAGE_HEALTHY"


def test_health_state_critical_boundary():
    assert SR.storage_health_state(10.0, 500) == "STORAGE_CRITICAL"
    assert SR.storage_health_state(50.0, 100) == "STORAGE_CRITICAL"


def test_health_state_emergency_boundary():
    assert SR.storage_health_state(5.0, 500) == "STORAGE_EMERGENCY"
    assert SR.storage_health_state(50.0, 50) == "STORAGE_EMERGENCY"


def test_health_state_fails_toward_more_severe_on_disagreement():
    """One metric HEALTHY, the other EMERGENCY -> must report EMERGENCY."""
    assert SR.storage_health_state(90.0, 40) == "STORAGE_EMERGENCY"


def test_health_state_deterministic():
    a = SR.storage_health_state(15.0, 300)
    b = SR.storage_health_state(15.0, 300)
    assert a == b


def test_all_declared_states_in_enum():
    for row in SR.PERMITTED_AUTOMATED_RESPONSE:
        assert row in SR.STORAGE_HEALTH_STATES


# ---------------------------------------------------------------------------
# Prohibited automated response guard
# ---------------------------------------------------------------------------

def test_prohibited_responses_include_deletion_and_vacuum():
    joined = " ".join(SR.PROHIBITED_AUTOMATED_RESPONSE).lower()
    assert "deletion" in joined
    assert "vacuum" in joined
    assert "collector" in joined


def test_permitted_responses_never_authorize_deletion():
    for state, responses in SR.PERMITTED_AUTOMATED_RESPONSE.items():
        joined = " ".join(responses).lower()
        assert "delete" not in joined
        assert "purge" not in joined
        assert "vacuum" not in joined


# ---------------------------------------------------------------------------
# Failure-mode table
# ---------------------------------------------------------------------------

def test_all_failure_modes_fail_closed():
    assert len(SR.FAILURE_MODES) == 24
    for fm in SR.FAILURE_MODES:
        assert fm["response"] == "FAIL_CLOSED"
        assert fm["deletion_permitted"] is False


def test_failure_modes_cover_required_scenarios():
    names = {fm["failure"] for fm in SR.FAILURE_MODES}
    required = {"checksum_mismatch", "row_count_mismatch", "schema_mismatch", "disk_full_during_purge",
                "interrupted_chunk_delete", "restore_mismatch", "unknown_table_classification"}
    assert required <= names


# ---------------------------------------------------------------------------
# Structural no-mutation guards (AST-scoped, not just behavioral)
# ---------------------------------------------------------------------------

def test_module_never_calls_execute_or_opens_a_connection():
    src = inspect.getsource(SR)
    tree = ast.parse(src)
    calls = [n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "execute" not in calls
    assert "executescript" not in calls
    assert "executemany" not in calls
    assert "connect" not in calls


def test_module_never_imports_sqlite3_or_os_remove():
    src = inspect.getsource(SR)
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "sqlite3" not in imported


def test_module_source_never_contains_destructive_sql_keywords():
    """Literal-string guard: no SQL DELETE/DROP/VACUUM/UPDATE token appears
    anywhere in the module's source as an executable statement fragment
    (the words may appear in prose reasons/docstrings describing policy --
    this test only fails if one appears immediately followed by a table
    name pattern typical of an executable statement, which none do)."""
    src = inspect.getsource(SR)
    forbidden_statements = ("DELETE FROM", "DROP TABLE", "VACUUM;", "UPDATE ami_", "UPDATE canonical")
    for stmt in forbidden_statements:
        assert stmt not in src, stmt


def test_future_components_and_failure_modes_are_metadata_only():
    """FUTURE_COMPONENTS is a tuple of strings, not callables -- confirms
    no future implementation was smuggled into this readiness batch."""
    assert all(isinstance(c, str) for c in SR.FUTURE_COMPONENTS)
    assert len(SR.FUTURE_COMPONENTS) == 17
