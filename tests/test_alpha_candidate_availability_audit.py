from pathlib import Path

from tools.alpha_candidate_availability_audit import (
    BLOCKED_BY_AVAILABILITY_UNKNOWN,
    PASS_AVAILABILITY_AUDIT,
    REJECT_ENTRY_PREDICATE_LOOKAHEAD,
    CandidateAvailabilityRecord,
    audit_candidate_availability,
    build_family_audit_records,
    build_family_summary,
    completed_cluster_feature_available_ts,
    compute_threshold_cross_ts,
    write_registry,
)


def test_completed_cluster_aggregate_lookahead_at_cluster_start_entry():
    """Test 1: cluster_start=1000, cluster_end=1200, predicate needs final
    cluster_notional, entry decided at cluster_start (1000) -> rejected."""
    record = CandidateAvailabilityRecord(
        family_id="FIXTURE_FAM",
        candidate_id="FIXTURE_CAND_COMPLETED_AT_START",
        predicate_features=("cluster_notional",),
        event_ts_ms=1000,
        entry_decision_ts_ms=1000,
        feature_available_ts_ms=None,
        feature_available_rule="cluster_start_entry",
    )
    result = audit_candidate_availability(record)
    assert result.disposition == REJECT_ENTRY_PREDICATE_LOOKAHEAD
    assert "cluster_notional" in result.reason


def test_threshold_cross_pass_when_entry_at_or_after_cross():
    """Test 2: cumulative notional crosses 500K at ts=1080; entry decided at 1080 -> pass."""
    events = [(1000, 200_000.0), (1040, 200_000.0), (1080, 150_000.0), (1120, 50_000.0)]
    cross_ts = compute_threshold_cross_ts(events, 500_000.0)
    assert cross_ts == 1080

    record = CandidateAvailabilityRecord(
        family_id="FIXTURE_FAM",
        candidate_id="FIXTURE_CAND_THRESHOLD_CROSS_PASS",
        predicate_features=("running_cluster_notional",),
        event_ts_ms=1000,
        entry_decision_ts_ms=1080,
        feature_available_ts_ms=cross_ts,
        feature_available_rule="threshold_cross",
    )
    result = audit_candidate_availability(record)
    assert result.disposition == PASS_AVAILABILITY_AUDIT


def test_threshold_cross_early_entry_is_rejected():
    """Test 3: same crossing at 1080, but entry decided at 1000 (before the cross) -> rejected."""
    events = [(1000, 200_000.0), (1040, 200_000.0), (1080, 150_000.0), (1120, 50_000.0)]
    cross_ts = compute_threshold_cross_ts(events, 500_000.0)
    assert cross_ts == 1080

    record = CandidateAvailabilityRecord(
        family_id="FIXTURE_FAM",
        candidate_id="FIXTURE_CAND_THRESHOLD_CROSS_EARLY",
        predicate_features=("running_cluster_notional",),
        event_ts_ms=1000,
        entry_decision_ts_ms=1000,
        feature_available_ts_ms=cross_ts,
        feature_available_rule="threshold_cross",
    )
    result = audit_candidate_availability(record)
    assert result.disposition == REJECT_ENTRY_PREDICATE_LOOKAHEAD
    assert "1000" in result.reason and "1080" in result.reason


def test_completed_cluster_aggregate_passes_only_at_or_after_cluster_end():
    """Test 4: entry decided exactly at cluster_end (1200), not cluster_start -> pass."""
    cluster_end_ts_ms = 1200
    feature_available_ts_ms = completed_cluster_feature_available_ts(cluster_end_ts_ms)
    assert feature_available_ts_ms == 1200

    record = CandidateAvailabilityRecord(
        family_id="FIXTURE_FAM",
        candidate_id="FIXTURE_CAND_COMPLETED_AT_END",
        predicate_features=("cluster_notional",),
        event_ts_ms=1000,
        entry_decision_ts_ms=1200,
        feature_available_ts_ms=feature_available_ts_ms,
        feature_available_rule="post_cluster_end_entry",
    )
    result = audit_candidate_availability(record)
    assert result.disposition == PASS_AVAILABILITY_AUDIT


def test_pre_event_feature_pass():
    """Test 5: a pre-event feature known strictly before entry -> pass."""
    record = CandidateAvailabilityRecord(
        family_id="FIXTURE_FAM",
        candidate_id="FIXTURE_CAND_PRE_EVENT",
        predicate_features=("day_trend_bps",),
        event_ts_ms=1000,
        entry_decision_ts_ms=1000,
        feature_available_ts_ms=900,
        feature_available_rule="pre_event_point_in_time",
    )
    result = audit_candidate_availability(record)
    assert result.disposition == PASS_AVAILABILITY_AUDIT


def test_unknown_availability_is_blocked():
    """Test 6: feature_available_ts_ms missing and not otherwise resolvable -> blocked."""
    record = CandidateAvailabilityRecord(
        family_id="FIXTURE_FAM",
        candidate_id="FIXTURE_CAND_UNKNOWN",
        predicate_features=("some_new_undocumented_feature",),
        event_ts_ms=1000,
        entry_decision_ts_ms=1000,
        feature_available_ts_ms=None,
        feature_available_rule="undetermined",
    )
    result = audit_candidate_availability(record)
    assert result.disposition == BLOCKED_BY_AVAILABILITY_UNKNOWN


def test_post_entry_diagnostic_feature_cannot_be_a_predicate():
    """Test 7: time_to_MFE / first_5m_net_bps used as predicates -> rejected with
    an explicit post-entry-diagnostic reason, regardless of timestamps."""
    for feature_name in ("time_to_mfe_s", "first_5m_net_bps"):
        record = CandidateAvailabilityRecord(
            family_id="FIXTURE_FAM",
            candidate_id=f"FIXTURE_CAND_DIAGNOSTIC_{feature_name}",
            predicate_features=(feature_name,),
            event_ts_ms=1000,
            entry_decision_ts_ms=2000,  # even a "safe-looking" late entry does not save it
            feature_available_ts_ms=500,  # even a "safe-looking" early availability does not save it
            feature_available_rule="pre_event_point_in_time",
        )
        result = audit_candidate_availability(record)
        assert result.disposition == REJECT_ENTRY_PREDICATE_LOOKAHEAD
        assert "post-entry" in result.reason
        assert "predicate" in result.reason


def test_current_rejected_family_fixture_fam_eth_buy_liq_continuation():
    """Test 8: the FAM_ETH_BUY_LIQ_CONTINUATION fixture (event_ts_ms ==
    cluster_start_ts_ms, final cluster_notional/cluster_liq_count predicate)
    must audit to REJECT_ENTRY_PREDICATE_LOOKAHEAD for every one of its 5
    candidates, and the family/promotion summary must mark it ineligible."""
    records = build_family_audit_records()
    assert len(records) == 5
    expected_ids = {
        "CAND_ETH_BUY_CONT_500K_DAYTREND_D0_TP40_SL50_BE20",
        "CAND_ETH_BUY_CONT_1M_DAYTREND_D0_TP40_SL50_BE20",
        "CAND_ETH_BUY_CONT_500K_GEOM_COUNT22_D0_TP40_SL50_BE20",
        "CAND_ETH_BUY_CONT_500K_CASCADE_P15_109K_D0_TP40_SL50_BE20",
        "CAND_ETH_BUY_CONT_500K_DAYTREND_GEOM_CASCADE_D0_TP40_SL50_BE20",
    }
    assert {r.candidate_id for r in records} == expected_ids
    for record in records:
        assert record.family_id == "FAM_ETH_BUY_LIQ_CONTINUATION"
        assert record.event_ts_ms == record.entry_decision_ts_ms
        assert record.disposition == REJECT_ENTRY_PREDICATE_LOOKAHEAD

    summary = build_family_summary()
    assert summary["family_disposition"] == REJECT_ENTRY_PREDICATE_LOOKAHEAD
    assert summary["promotion_disposition"] == "REJECT_SPURIOUS_OR_DATA_PATH_SUSPECT"
    assert summary["canonical_alpha_gate_eligible"] is False


def test_write_registry_is_scratch_only_and_makes_no_db_or_archive_reference(tmp_path: Path):
    """Test 9: registry output only touches the paths explicitly passed in
    (here, tmp_path); no sqlite3/DB import and no archive/catalog path
    string appears anywhere in the module source (no production output,
    no archive mutation possible)."""
    out_md = tmp_path / "AUDIT.md"
    out_json = tmp_path / "AUDIT.json"

    payload = write_registry(out_md, out_json)

    assert out_md.exists()
    assert out_json.exists()
    assert payload["family_summary"]["family_disposition"] == REJECT_ENTRY_PREDICATE_LOOKAHEAD

    import tools.alpha_candidate_availability_audit as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "sqlite3" not in source
    assert "data/archives" not in source
    assert "microstructure.db" not in source
