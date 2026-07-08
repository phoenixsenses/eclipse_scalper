"""Tests for ami/cvd/cvd_source_quality_contract_v1.py -- fail-closed
classification branches, regime-boundary handling, SQL-level status
rejection, append-only immutability. Fully synthetic."""
import sqlite3

import pytest

from ami.cvd import cvd_source_quality_contract_v1 as q


def _mem():
    conn = sqlite3.connect(":memory:")
    q.init_schema(conn)
    return conn


def _classify(**over):
    base = dict(
        missing_minute_count=0, repaired_minute_count=0, coverage_map_available=True,
        cadence_proof_pass=True, duplicate_unresolved=False, regime_ids=["R3"],
        regime_proofs={"R3": True}, proxy_available=True)
    base.update(over)
    return q.classify_window(**base)


def test_happy_path_exact_reconstructable():
    assert _classify() == "EXACT_RECONSTRUCTABLE"


# 20. fail-closed when exact reconstruction is unproven
def test_no_coverage_map_fails_closed():
    assert _classify(coverage_map_available=False) == "SOURCE_COVERAGE_UNRESOLVED"


def test_cadence_proof_unavailable_fails_closed():
    assert _classify(cadence_proof_pass=None) == "SOURCE_COVERAGE_UNRESOLVED"


def test_cadence_fail_degrades_to_proxy_or_gapped():
    assert _classify(cadence_proof_pass=False) == "PROXY_ONLY"
    assert _classify(cadence_proof_pass=False, proxy_available=False) == "SOURCE_GAPPED"


def test_missing_minutes_without_repair_degrade():
    assert _classify(missing_minute_count=3) == "PROXY_ONLY"
    assert _classify(missing_minute_count=3, proxy_available=False) == "SOURCE_GAPPED"


def test_missing_minutes_fully_repaired_is_exact():
    assert _classify(missing_minute_count=3, repaired_minute_count=3) == "EXACT_RECONSTRUCTABLE"


def test_partially_repaired_never_exact():
    assert _classify(missing_minute_count=3, repaired_minute_count=2) == "PROXY_ONLY"


def test_duplicate_unresolved_blocks_exact():
    assert _classify(duplicate_unresolved=True) == "SOURCE_COVERAGE_UNRESOLVED"
    assert _classify(missing_minute_count=1, repaired_minute_count=1,
                     duplicate_unresolved=True) == "SOURCE_COVERAGE_UNRESOLVED"


# 14. source-regime boundary classification
def test_regime_without_proof_never_auto_passes():
    assert _classify(regime_ids=["R2", "R3"],
                     regime_proofs={"R3": True}) == "SOURCE_COVERAGE_UNRESOLVED"
    assert _classify(regime_ids=["R2", "R3"],
                     regime_proofs={"R2": True, "R3": True}) == "EXACT_RECONSTRUCTABLE"


def test_regimes_for_window_spanning_boundary():
    r2_end = 1780767832123
    assert q.regimes_for_window(r2_end - 60_000, r2_end + 60_000) == ["R2", "R3"]
    assert q.regimes_for_window(r2_end, r2_end + 60_000) == ["R3"]
    assert q.regimes_for_window(r2_end - 120_000, r2_end - 60_000) == ["R2"]


def _qrow(status="EXACT_RECONSTRUCTABLE", sig="SIG-A", window="W60"):
    return {
        "signal_id": sig, "independent_cycle_id": "CYC-A", "symbol": "ETHUSDT",
        "signal_birth_ts": 1_000_000, "window_id": window,
        "window_start_ts_ms": 940_000, "window_end_ts_ms": 1_000_000,
        "evidence_layer": "EXACT", "source_regime_ids": '["R0"]', "regime_spanning": 0,
        "legacy_row_count": 10, "repair_row_count": 0, "total_row_count": 10,
        "duplicate_count": 0, "collision_count": 0, "unresolved_match_count": 0,
        "missing_minute_count": 0, "repaired_minute_count": 0,
        "cadence_proof": "{}", "completeness_proof": "{}", "quality_status": status,
        "source_provenance": "test", "data_version_id": "legacy-live-collection",
        "feature_definition_version": "s34-cvd-windowed-taker-flow-v1-birth-truncated",
    }


# 15. invalid quality-status SQL rejection (both via API and raw SQL)
def test_invalid_status_rejected_by_api():
    conn = _mem()
    with pytest.raises(ValueError):
        q.record_window_quality(conn, _qrow(status="BOGUS_STATUS"), assessment_version="a1")


def test_invalid_status_rejected_by_sql_check():
    conn = _mem()
    row = _qrow()
    q.record_window_quality(conn, row, assessment_version="a1")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE ami_cvd_window_quality_v1 SET quality_status='BOGUS_STATUS'")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ami_cvd_window_quality_v1 (quality_id, quality_contract_version,"
            " assessment_version, signal_id, symbol, signal_birth_ts, window_id,"
            " window_start_ts_ms, window_end_ts_ms, evidence_layer, source_regime_ids,"
            " regime_spanning, legacy_row_count, repair_row_count, total_row_count,"
            " duplicate_count, collision_count, unresolved_match_count, missing_minute_count,"
            " repaired_minute_count, cadence_proof, completeness_proof, quality_status,"
            " feature_available_ts_ms, source_provenance, data_version_id,"
            " feature_definition_version, assessed_at_ms)"
            " VALUES ('id2','c','a2','S','ETHUSDT',1000,'W60',0,1000,'EXACT','[]',0,"
            " 0,0,0,0,0,0,0,0,'{}','{}','BOGUS_STATUS',1000,'t','d','f',0)")


def test_append_only_immutable_conflict():
    conn = _mem()
    row = _qrow()
    assert q.record_window_quality(conn, row, assessment_version="a1") == "INSERTED"
    assert q.record_window_quality(conn, row, assessment_version="a1") == "NOOP_IDENTICAL"
    row2 = dict(row)
    row2["quality_status"] = "SOURCE_GAPPED"
    with pytest.raises(q.ImmutableCvdQualityConflict):
        q.record_window_quality(conn, row2, assessment_version="a1")
    # a NEW assessment version may record a different opinion (append, not rewrite)
    assert q.record_window_quality(conn, row2, assessment_version="a2") == "INSERTED"
    assert conn.execute("SELECT COUNT(*) FROM ami_cvd_window_quality_v1").fetchone()[0] == 2


def test_unrepairable_never_auto_assigned_by_classifier():
    # exhaustive-ish sweep over classifier inputs: UNREPAIRABLE must never
    # be produced automatically (reserved for explicit operator assignment)
    for missing in (0, 1, 5):
        for repaired in (0, 1, 5):
            for cad in (True, False, None):
                for cov in (True, False):
                    for dup in (True, False):
                        for proxy in (True, False):
                            st = q.classify_window(
                                missing_minute_count=missing,
                                repaired_minute_count=repaired,
                                coverage_map_available=cov, cadence_proof_pass=cad,
                                duplicate_unresolved=dup, regime_ids=["R1"],
                                regime_proofs={"R1": True}, proxy_available=proxy)
                            assert st in q.STATUSES
                            assert st != "UNREPAIRABLE"
