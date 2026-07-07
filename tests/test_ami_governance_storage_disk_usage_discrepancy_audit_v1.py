"""BATCH-STORAGE-DISK-USAGE-DISCREPANCY-AUDIT-V1 -- focused tests.

The module under test is pure arithmetic/classification -- it never
touches the filesystem or a database. Proven both behaviorally and
structurally (AST guards).
"""
from __future__ import annotations

import ast
import inspect

from ami.governance import storage_disk_usage_discrepancy_audit_v1 as A

REAL_ITEMS = {
    "eclipse_scalper": 798716636064,
    "RECYCLE.BIN": 24897597,
    "Android": 4392694272,
    "c": 369368022,
    "chess97": 24238538,
    "chess97_pytest_tmp": 373,
    "commerce_intelligence": 37083394,
    "eclipse_pentest_platform": 669296655,
    "eclipse_scalper_scratch_pytest_tmp": 0,
    "flutter": 3209422033,
    "lockscreen_rpg": 2131574427,
    "psi97": 767135747,
    "Riot Games": 39389216003,
    "Rise of Kingdoms Game": 5474242471,
    "Steam": 18327552078,
    "SteamLibrary": 389812654,
    "tmp": 5304190,
    "migration_log.txt": 3025600,
}
REAL_USED_BYTES = 874833305600


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def test_bytes_to_units_gb_vs_gib():
    result = A.bytes_to_units(1_000_000_000)
    assert result["gb_decimal"] == 1.0
    assert abs(result["gib_binary"] - 0.9313225746154785) < 1e-9


def test_bytes_to_units_2tb_drive():
    result = A.bytes_to_units(2000381014016)
    assert abs(result["gb_decimal"] - 2000.381014016) < 1e-6
    assert abs(result["gib_binary"] - 1862.9999961853027) < 1e-6


# ---------------------------------------------------------------------------
# Reconciliation (real measured data from this batch)
# ---------------------------------------------------------------------------

def test_reconcile_used_space_real_data_closes_under_threshold():
    rec = A.reconcile_used_space(REAL_USED_BYTES, REAL_ITEMS)
    assert rec["remaining_unexplained_bytes"] == 901805482
    assert rec["remaining_unexplained_pct"] < A.EXPLAINED_THRESHOLD_PCT
    assert rec["meets_explained_threshold"] is True


def test_reconcile_used_space_never_forces_negative_remaining_to_zero():
    """If measured items exceed used_bytes (double-counting), the
    function must report a negative remaining, not silently clamp it --
    that would hide a real accounting bug."""
    rec = A.reconcile_used_space(100, {"a": 60, "b": 60})
    assert rec["remaining_unexplained_bytes"] == -20


def test_reconcile_used_space_100_pct_used_with_no_items():
    rec = A.reconcile_used_space(1000, {})
    assert rec["remaining_unexplained_bytes"] == 1000
    assert rec["remaining_unexplained_pct"] == 100.0
    assert rec["meets_explained_threshold"] is False


def test_reconcile_used_space_items_preserved_verbatim():
    rec = A.reconcile_used_space(REAL_USED_BYTES, REAL_ITEMS)
    assert rec["items"] == REAL_ITEMS
    assert rec["items"] is not REAL_ITEMS  # defensive copy


# ---------------------------------------------------------------------------
# Unit-labeling reconciliation (the actual root cause found this batch)
# ---------------------------------------------------------------------------

def test_unit_labeling_reconciliation_shows_both_interpretations():
    result = A.reconcile_unit_labeling(881, "GB", REAL_USED_BYTES)
    # decimal-GB interpretation: tiny ~6GB delta (real explanation)
    assert abs(result["if_old_was_decimal_gb"]["delta_vs_current_gb"] - 6.1666944) < 1e-4
    # GiB-mislabeled interpretation: larger ~71GB delta
    assert abs(result["if_old_was_gib_mislabeled_as_gb"]["delta_vs_current_gb"] - 71.133241344) < 1e-4


def test_unit_labeling_rejects_invalid_unit():
    import pytest
    with pytest.raises(ValueError):
        A.reconcile_unit_labeling(881, "TB", REAL_USED_BYTES)


def test_unit_labeling_decimal_gb_delta_smaller_than_gib_delta():
    """The whole point of this batch's finding: the decimal-GB
    interpretation produces a far smaller, more plausible delta than
    the GiB-mislabeled-as-GB interpretation."""
    result = A.reconcile_unit_labeling(881, "GB", REAL_USED_BYTES)
    decimal_delta = abs(result["if_old_was_decimal_gb"]["delta_vs_current_gb"])
    gib_delta = abs(result["if_old_was_gib_mislabeled_as_gb"]["delta_vs_current_gb"])
    assert decimal_delta < gib_delta


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_chrome_copy_classification_real_evidence():
    result = A.classify_chrome_copy(
        is_reparse_point=False, referenced_by_active_process=False,
        referenced_by_repo_code=False, days_since_last_write=13)
    assert result == "REPRODUCIBLE_INACTIVE_CLEANUP_CANDIDATE"
    assert result in A.CHROME_COPY_CLASSES


def test_chrome_copy_active_process_wins_over_everything():
    result = A.classify_chrome_copy(
        is_reparse_point=True, referenced_by_active_process=True,
        referenced_by_repo_code=True, days_since_last_write=999)
    assert result == "ACTIVE_REQUIRED"


def test_chrome_copy_recent_write_is_unknown_not_disposable():
    result = A.classify_chrome_copy(
        is_reparse_point=False, referenced_by_active_process=False,
        referenced_by_repo_code=False, days_since_last_write=1)
    assert result == "UNKNOWN_REQUIRES_OPERATOR_REVIEW"


def test_pytest_scratch_classification():
    assert A.classify_pytest_scratch(
        in_authorized_temp_dir=False, referenced_by_active_process=False,
        is_only_copy_of_accepted_evidence=False) == "VERIFIED_DISPOSABLE_CANDIDATE"
    assert A.classify_pytest_scratch(
        in_authorized_temp_dir=False, referenced_by_active_process=True,
        is_only_copy_of_accepted_evidence=False) == "ACTIVE_TEMP"
    assert A.classify_pytest_scratch(
        in_authorized_temp_dir=False, referenced_by_active_process=False,
        is_only_copy_of_accepted_evidence=True) == "RETAINED_ACCEPTED_EVIDENCE"
    assert A.classify_pytest_scratch(
        in_authorized_temp_dir=True, referenced_by_active_process=False,
        is_only_copy_of_accepted_evidence=False) == "ACTIVE_TEMP"


def test_all_classification_enums_are_valid_sets():
    assert len(A.CLEANUP_CLASSES) == 7
    assert len(A.CHROME_COPY_CLASSES) == 5
    assert len(A.PYTEST_SCRATCH_CLASSES) == 4


# ---------------------------------------------------------------------------
# Verdict determination
# ---------------------------------------------------------------------------

def test_verdict_explained_at_real_measured_percentage():
    rec = A.reconcile_used_space(REAL_USED_BYTES, REAL_ITEMS)
    verdict = A.determine_verdict(
        explained_pct=rec["explained_pct"], permission_blocked_material=False,
        measurement_inconsistent=False)
    assert verdict == "STORAGE_DISK_USAGE_DISCREPANCY_EXPLAINED"
    assert verdict in A.VERDICTS


def test_verdict_partially_explained_below_threshold():
    verdict = A.determine_verdict(explained_pct=90.0, permission_blocked_material=False,
                                   measurement_inconsistent=False)
    assert verdict == "STORAGE_DISK_USAGE_DISCREPANCY_PARTIALLY_EXPLAINED"


def test_verdict_blocked_by_permissions_even_at_high_pct():
    """A material permission block must win even if the numeric
    percentage looks good -- fail-closed precedence."""
    verdict = A.determine_verdict(explained_pct=99.9, permission_blocked_material=True,
                                   measurement_inconsistent=False)
    assert verdict == "STORAGE_DISK_USAGE_DISCREPANCY_BLOCKED_BY_PERMISSIONS"


def test_verdict_measurement_inconsistency_wins_over_permission_block():
    verdict = A.determine_verdict(explained_pct=99.9, permission_blocked_material=True,
                                   measurement_inconsistent=True)
    assert verdict == "STORAGE_DISK_USAGE_DISCREPANCY_BLOCKED_BY_MEASUREMENT_INCONSISTENCY"


def test_verdict_exact_threshold_boundary():
    # exactly 98% explained (2% remaining) must pass
    verdict = A.determine_verdict(explained_pct=98.0, permission_blocked_material=False,
                                   measurement_inconsistent=False)
    assert verdict == "STORAGE_DISK_USAGE_DISCREPANCY_EXPLAINED"
    verdict2 = A.determine_verdict(explained_pct=97.999, permission_blocked_material=False,
                                    measurement_inconsistent=False)
    assert verdict2 == "STORAGE_DISK_USAGE_DISCREPANCY_PARTIALLY_EXPLAINED"


# ---------------------------------------------------------------------------
# Next-gate selection
# ---------------------------------------------------------------------------

def test_next_gate_dry_run_when_no_disposable_candidates():
    assert A.next_gate(0) == "BATCH-STORAGE-ROTATION-RETENTION-DISPOSABLE-DRY-RUN-V1"


def test_next_gate_dry_run_below_1gb():
    assert A.next_gate(500_000_000) == "BATCH-STORAGE-ROTATION-RETENTION-DISPOSABLE-DRY-RUN-V1"


def test_next_gate_bounded_cleanup_at_or_above_1gb():
    assert A.next_gate(1_000_000_000) == "BATCH-STORAGE-BOUNDED-CLEANUP-AUTHORIZATION-V1"
    assert A.next_gate(50_000_000_000) == "BATCH-STORAGE-BOUNDED-CLEANUP-AUTHORIZATION-V1"


def test_next_gate_real_estimate_stray_scratch_files_below_1gb():
    """This batch's own measured stray-scratch total (~13MB) is well
    below the 1GB bounded-cleanup trigger."""
    assert A.next_gate(13_008_896) == "BATCH-STORAGE-ROTATION-RETENTION-DISPOSABLE-DRY-RUN-V1"


# ---------------------------------------------------------------------------
# Structural no-mutation guards
# ---------------------------------------------------------------------------

def test_module_never_imports_filesystem_or_db_modules():
    src = inspect.getsource(A)
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for forbidden in ("os", "shutil", "sqlite3", "subprocess", "pathlib"):
        assert forbidden not in imported, forbidden


def test_module_never_calls_execute_or_file_io():
    src = inspect.getsource(A)
    tree = ast.parse(src)
    calls = [n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    for forbidden in ("execute", "remove", "unlink", "rmdir", "rename", "move"):
        assert forbidden not in calls
    names = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "open" not in names
