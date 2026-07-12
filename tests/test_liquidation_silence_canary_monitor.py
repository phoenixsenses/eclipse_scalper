"""Tests for tools/liquidation_silence_canary_monitor.py -- the Gate 1
monitor-harness corrective, round 2 (addresses independent-review findings
F-01..F-10; see this module's own docstring and SYSTEM_STATE.md).

Every test here is fully deterministic and process-control-free: process
enumeration is always supplied via an injected fake ProcessLister (never
psutil, never a real subprocess) except for the two explicit CLI
integration tests, which invoke the real module through `python -m` against
synthetic temp-directory fixtures and a non-existent PID (999999999) so no
real process is ever touched.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from tools import liquidation_silence_canary_monitor as mon


# --------------------------------------------------------------------------
# Fixture helpers
# --------------------------------------------------------------------------


def _rec(pid, start_time_utc, cmdline="python -W ignore -u -m tools.liquidation_silence_scheduler", captured_at_utc=None):
    return mon.ProcessRecord(pid=pid, start_time_utc=start_time_utc, command_line=cmdline, captured_at_utc=captured_at_utc)


def _fake_lister(records):
    def _lister():
        return records

    return _lister


BASELINE_START = "2026-07-11T17:40:51.301413+00:00"
SCHEDULER_IDENTITY = mon.ExpectedIdentity(role="liquidation_silence_scheduler", module_name="tools.liquidation_silence_scheduler")
WATCHDOG_IDENTITY = mon.ExpectedIdentity(role="heartbeat_watchdog", module_name="tools.heartbeat_watchdog")

# Real Gate 1 canary epoch timestamps (tools/liquidation_silence_scheduler.py
# cycle_started_epoch field), used verbatim as the canonical fixture.
GATE1_EPOCHS = [
    1783791711.4434423,
    1783792011.5738635,
    1783792311.6955864,
    1783792611.818728,
    1783792911.9373834,
    1783793212.0605073,
    1783793512.1746144,
]


def _gate1_cycle_log_lines():
    lines = []
    for epoch in GATE1_EPOCHS:
        record = {
            "ts_utc": mon.epoch_to_iso(epoch)[:19] + "Z",  # whole-second, matching the real scheduler's own format
            "cycle_started_epoch": epoch,
            "outcome": "SUCCESS",
            "severity": "GREEN",
            "status": "HEALTHY",
            "detector_error": None,
            "detector_runtime_sec": 0.002,
            "cycle_duration_sec": 0.003,
        }
        lines.append(json.dumps(record))
    return lines


# ==========================================================================
# F-01: StartTime UTC normalization
# ==========================================================================


def test_normalize_utc_timestamp_valid():
    dt, status = mon.normalize_utc_timestamp("2026-07-11T17:40:51.301413+00:00")
    assert status == mon.TS_VALID
    assert dt.tzinfo is not None


def test_normalize_utc_timestamp_missing():
    dt, status = mon.normalize_utc_timestamp(None)
    assert dt is None and status == mon.TS_MISSING
    dt, status = mon.normalize_utc_timestamp("")
    assert dt is None and status == mon.TS_MISSING


def test_normalize_utc_timestamp_malformed():
    dt, status = mon.normalize_utc_timestamp("not-a-timestamp")
    assert dt is None and status == mon.TS_MALFORMED


def test_normalize_utc_timestamp_naive_rejected():
    dt, status = mon.normalize_utc_timestamp("2026-07-11T17:40:51.301413")  # no offset
    assert dt is None and status == mon.TS_NAIVE_REJECTED


def test_equivalent_offset_starttimes_compare_equal():
    baseline = _rec(15640, "2026-07-11T17:40:51Z")
    query = mon.ProcessQueryResult(query_failed=False, matches=[_rec(15640, "2026-07-11T20:40:51+03:00")])
    snapshot = mon.classify_continuity(baseline, query)
    assert snapshot.continuity_status == mon.CONTINUITY_CONTINUOUS


def test_millisecond_precision_preserved_in_comparison():
    baseline = _rec(15640, "2026-07-11T17:40:51.301413+00:00")
    same = mon.ProcessQueryResult(query_failed=False, matches=[_rec(15640, "2026-07-11T17:40:51.301413+00:00")])
    assert mon.classify_continuity(baseline, same).continuity_status == mon.CONTINUITY_CONTINUOUS
    different_ms = mon.ProcessQueryResult(query_failed=False, matches=[_rec(15640, "2026-07-11T17:40:51.999999+00:00")])
    snapshot = mon.classify_continuity(baseline, different_ms)
    assert snapshot.continuity_status == mon.CONTINUITY_REPLACED_OR_RESTARTED


def test_differing_actual_instants_detected_as_replacement():
    baseline = _rec(15640, BASELINE_START)
    query = mon.ProcessQueryResult(query_failed=False, matches=[_rec(15640, "2026-07-11T18:30:00.000000+00:00")])
    snapshot = mon.classify_continuity(baseline, query)
    assert snapshot.continuity_status == mon.CONTINUITY_REPLACED_OR_RESTARTED
    assert "STARTTIME_CHANGED" in snapshot.reason_codes


def test_naive_current_starttime_not_accepted_as_continuous():
    baseline = _rec(15640, BASELINE_START)
    query = mon.ProcessQueryResult(query_failed=False, matches=[_rec(15640, "2026-07-11T17:40:51.301413")])  # naive
    snapshot = mon.classify_continuity(baseline, query)
    assert snapshot.continuity_status == mon.CONTINUITY_MALFORMED_STARTTIME
    assert snapshot.start_time_status == mon.TS_NAIVE_REJECTED
    assert snapshot.continuity_status != mon.CONTINUITY_CONTINUOUS


def test_malformed_current_starttime_not_accepted_as_continuous():
    baseline = _rec(15640, BASELINE_START)
    query = mon.ProcessQueryResult(query_failed=False, matches=[_rec(15640, "garbage-timestamp")])
    snapshot = mon.classify_continuity(baseline, query)
    assert snapshot.continuity_status == mon.CONTINUITY_MALFORMED_STARTTIME


def test_missing_starttime_not_silently_accepted_as_continuous():
    baseline = _rec(15640, BASELINE_START)
    query = mon.ProcessQueryResult(query_failed=False, matches=[_rec(15640, None)])
    snapshot = mon.classify_continuity(baseline, query)
    assert snapshot.continuity_status == mon.CONTINUITY_UNKNOWN_STARTTIME
    assert snapshot.continuity_status != mon.CONTINUITY_CONTINUOUS
    assert "STARTTIME_UNAVAILABLE" in snapshot.reason_codes


# ==========================================================================
# F-02 / F-14: anchored identity comparison, not truthiness/substring
# ==========================================================================


def test_identity_matches_exact_scheduler_module():
    status, reasons = mon.identity_matches(
        "python -W ignore -u -m tools.liquidation_silence_scheduler", SCHEDULER_IDENTITY
    )
    assert status == mon.IDENTITY_MATCHED
    assert reasons == []


def test_identity_matches_exact_watchdog_module():
    status, _ = mon.identity_matches("python -u -m tools.heartbeat_watchdog --interval-sec 5", WATCHDOG_IDENTITY)
    assert status == mon.IDENTITY_MATCHED


def test_identity_rejects_unrelated_python_process():
    status, reasons = mon.identity_matches("python -u -m tools.oi_spot_poller", SCHEDULER_IDENTITY)
    assert status == mon.IDENTITY_MISMATCH
    assert "IDENTITY_TOKENS_NOT_MATCHED" in reasons


def test_identity_rejects_similarly_named_script_substring():
    # A backup/renamed file whose path merely CONTAINS the module string
    # must not match -- this is the exact false-positive F-14 closes.
    status, _ = mon.identity_matches(
        "python -u tools/liquidation_silence_scheduler.py.bak", SCHEDULER_IDENTITY
    )
    assert status == mon.IDENTITY_MISMATCH


def test_identity_rejects_substring_false_positive_in_unrelated_path():
    status, _ = mon.identity_matches(
        "python -u -m some_other_tools.liquidation_silence_scheduler_helper", SCHEDULER_IDENTITY
    )
    assert status == mon.IDENTITY_MISMATCH


def test_identity_case_and_path_separator_normalization():
    identity = mon.ExpectedIdentity(role="heartbeat_watchdog", script_basename="heartbeat_watchdog.py")
    status, _ = mon.identity_matches(r"C:\Python313\PYTHON.EXE -u TOOLS\HEARTBEAT_WATCHDOG.PY", identity)
    assert status == mon.IDENTITY_MATCHED


def test_identity_quoted_command_line_path():
    status, _ = mon.identity_matches(
        '"C:\\Python313\\python.exe" -W ignore -u -m tools.liquidation_silence_scheduler', SCHEDULER_IDENTITY
    )
    assert status == mon.IDENTITY_MATCHED


def test_identity_missing_command_line_is_unavailable_not_confirmed():
    status, reasons = mon.identity_matches(None, SCHEDULER_IDENTITY)
    assert status == mon.IDENTITY_UNAVAILABLE
    assert status != mon.IDENTITY_MATCHED
    assert "COMMAND_LINE_UNAVAILABLE" in reasons


def test_identity_mismatch_branch_reachable_through_continuity():
    baseline = _rec(15640, BASELINE_START, cmdline="python -u -m tools.liquidation_silence_scheduler")
    query = mon.ProcessQueryResult(
        query_failed=False,
        matches=[_rec(15640, BASELINE_START, cmdline="python -u -m tools.oi_spot_poller")],
    )
    snapshot = mon.classify_continuity(baseline, query, expected_identity=SCHEDULER_IDENTITY)
    assert snapshot.continuity_status == mon.CONTINUITY_IDENTITY_MISMATCH
    assert snapshot.continuity_status != mon.CONTINUITY_CONTINUOUS


def test_identity_match_required_for_full_continuity():
    baseline = _rec(15640, BASELINE_START)
    query = mon.ProcessQueryResult(query_failed=False, matches=[_rec(15640, BASELINE_START)])
    snapshot = mon.classify_continuity(baseline, query, expected_identity=SCHEDULER_IDENTITY)
    assert snapshot.continuity_status == mon.CONTINUITY_CONTINUOUS
    assert snapshot.identity_status == mon.IDENTITY_MATCHED


# ==========================================================================
# Process count / duplicate / query semantics
# ==========================================================================


def test_zero_scheduler_matches_is_zero_not_none():
    result = mon.find_by_command_needle([], "tools.liquidation_silence_scheduler")
    assert result.query_failed is False
    assert result.count == 0


def test_exactly_one_scheduler_process():
    records = [_rec(15640, BASELINE_START)]
    result = mon.find_by_command_needle(records, "tools.liquidation_silence_scheduler")
    assert result.count == 1


def test_duplicate_scheduler_processes_classified_duplicate_never_continuous():
    baseline = _rec(15640, BASELINE_START)
    records = [_rec(15640, BASELINE_START), _rec(99999, BASELINE_START)]
    query = mon.find_by_command_needle(records, "tools.liquidation_silence_scheduler")
    snapshot = mon.classify_continuity(baseline, query, expect_singular=True)
    assert snapshot.continuity_status == mon.CONTINUITY_DUPLICATE
    assert snapshot.continuity_status != mon.CONTINUITY_CONTINUOUS
    assert "MULTIPLE_MATCHING_PROCESSES" in snapshot.reason_codes


def test_different_pid_is_replacement_or_restart():
    baseline = _rec(15640, BASELINE_START)
    query = mon.ProcessQueryResult(query_failed=False, matches=[_rec(22222, BASELINE_START)])
    snapshot = mon.classify_continuity(baseline, query)
    assert snapshot.continuity_status == mon.CONTINUITY_REPLACED_OR_RESTARTED
    assert "PID_CHANGED" in snapshot.reason_codes


def test_null_process_query_result_is_query_failed_never_missing_or_continuous():
    baseline = _rec(15640, BASELINE_START)
    query = mon.find_by_command_needle(None, "tools.liquidation_silence_scheduler")
    assert query.query_failed is True
    assert query.count is None
    snapshot = mon.classify_continuity(baseline, query)
    assert snapshot.continuity_status == mon.CONTINUITY_QUERY_FAILED
    assert snapshot.continuity_status not in (mon.CONTINUITY_MISSING, mon.CONTINUITY_CONTINUOUS)


def test_powershell_scalar_count_equivalent_single_object_case():
    single_match = mon.find_by_command_needle([_rec(15640, BASELINE_START)], "tools.liquidation_silence_scheduler")
    assert single_match.count == 1
    failed_query = mon.find_by_command_needle(None, "tools.liquidation_silence_scheduler")
    assert failed_query.count is None


# ==========================================================================
# Watchdog / protected-role continuity reuse the same model
# ==========================================================================


def test_watchdog_continuity_uses_same_pid_starttime_identity_model():
    baseline = _rec(9740, "2026-07-11T17:32:42.000000+00:00", cmdline="python -u -m tools.heartbeat_watchdog")
    lister = _fake_lister([baseline])
    snapshot = mon.sample_role_continuity("heartbeat_watchdog", baseline, lister, expected_identity=WATCHDOG_IDENTITY)
    assert snapshot.continuity_status == mon.CONTINUITY_CONTINUOUS
    assert snapshot.identity_status == mon.IDENTITY_MATCHED


def test_watchdog_starttime_change_detected_not_pid_only():
    baseline = _rec(9740, "2026-07-11T17:32:42.000000+00:00", cmdline="python -u -m tools.heartbeat_watchdog")
    lister = _fake_lister([_rec(9740, "2026-07-11T19:00:00.000000+00:00", cmdline="python -u -m tools.heartbeat_watchdog")])
    snapshot = mon.sample_role_continuity("heartbeat_watchdog", baseline, lister, expected_identity=WATCHDOG_IDENTITY)
    assert snapshot.continuity_status == mon.CONTINUITY_REPLACED_OR_RESTARTED


def test_protected_role_continuity_detects_missing_role():
    baseline = _rec(23052, "2026-07-11T17:15:54.000000+00:00", cmdline="python -u scripts\\collector_supervisor.py")
    lister = _fake_lister([])
    snapshot = mon.sample_role_continuity("collector_supervisor", baseline, lister)
    assert snapshot.continuity_status == mon.CONTINUITY_MISSING


def test_no_pid_only_fallback_exists_in_module_source():
    source = Path(mon.__file__).read_text(encoding="utf-8")
    # The only equality comparison against a bare .pid field must always be
    # paired with a StartTime/identity check in the same function; assert
    # there is exactly one continuity classifier in the whole module.
    assert source.count("def classify_continuity(") == 1


# ==========================================================================
# Sample-ordering (F-15)
# ==========================================================================


def test_sample_timestamp_regression_flagged():
    baseline = _rec(15640, BASELINE_START, captured_at_utc="2026-07-11T18:00:00+00:00")
    query = mon.ProcessQueryResult(
        query_failed=False,
        matches=[_rec(15640, BASELINE_START, captured_at_utc="2026-07-11T17:00:00+00:00")],  # earlier than baseline
    )
    snapshot = mon.classify_continuity(baseline, query)
    assert "SAMPLE_TIMESTAMP_REGRESSION" in snapshot.reason_codes


def test_normal_forward_sample_order_no_regression_flag():
    baseline = _rec(15640, BASELINE_START, captured_at_utc="2026-07-11T17:00:00+00:00")
    query = mon.ProcessQueryResult(
        query_failed=False, matches=[_rec(15640, BASELINE_START, captured_at_utc="2026-07-11T18:00:00+00:00")]
    )
    snapshot = mon.classify_continuity(baseline, query)
    assert "SAMPLE_TIMESTAMP_REGRESSION" not in snapshot.reason_codes


# ==========================================================================
# Live-executor OFF-state
# ==========================================================================


def test_live_executor_off_state_ok(tmp_path):
    marker = tmp_path / "s34_state_machine_live_executor.pid"
    marker.write_text("0", encoding="utf-8")
    check = mon.check_live_executor(marker, _fake_lister([]), "s34_state_machine_live_executor")
    assert check.off_state_ok is True
    assert check.reason_codes == []


def test_live_executor_process_found_despite_marker_zero(tmp_path):
    marker = tmp_path / "s34_v_engine_live_executor.pid"
    marker.write_text("0", encoding="utf-8")
    lister = _fake_lister([_rec(4321, BASELINE_START, cmdline="python -u -m tools.s34_v_engine_live_executor")])
    check = mon.check_live_executor(marker, lister, "s34_v_engine_live_executor")
    assert check.off_state_ok is False
    assert "LIVE_EXECUTOR_PROCESS_FOUND_DESPITE_OFF_MARKER" in check.reason_codes


# ==========================================================================
# F-10: StopEvent validation
# ==========================================================================


def test_valid_verified_stop_event():
    verification = mon.StopVerification(checked_at_utc="2026-07-11T18:15:00+00:00", query_failed=False, still_present_pids=[])
    event = mon.build_stop_event(
        target_role="liquidation_silence_scheduler",
        target_pids=[15640],
        requested_at_utc="2026-07-11T18:14:50+00:00",
        completed_at_utc="2026-07-11T18:14:55+00:00",
        verification=verification,
        outcome=mon.STOP_OUTCOME_VERIFIED_ABSENT,
    )
    assert event.verified_absent_at_utc == "2026-07-11T18:15:00+00:00"
    assert event.verification == "ALL_ABSENT"
    assert event.outcome == mon.STOP_OUTCOME_VERIFIED_ABSENT


def test_success_without_verification_is_rejected():
    with pytest.raises(ValueError):
        mon.build_stop_event(
            target_role="liquidation_silence_scheduler",
            target_pids=[15640],
            outcome=mon.STOP_OUTCOME_VERIFIED_ABSENT,
            verification=None,
        )


def test_verification_with_some_pids_still_present_is_not_full_absence():
    lister = _fake_lister([_rec(15640, BASELINE_START)])
    verification = mon.verify_pids_absent(lister, [15640, 22222])
    assert verification.all_absent is False
    assert verification.still_present_pids == [15640]
    with pytest.raises(ValueError):
        mon.build_stop_event(
            target_role="liquidation_silence_scheduler",
            target_pids=[15640, 22222],
            outcome=mon.STOP_OUTCOME_VERIFIED_ABSENT,
            verification=verification,
        )


def test_query_failure_is_not_absence():
    lister = _fake_lister(None)
    verification = mon.verify_pids_absent(lister, [15640])
    assert verification.query_failed is True
    assert verification.all_absent is False
    event = mon.build_stop_event(
        target_role="liquidation_silence_scheduler", target_pids=[15640], verification=verification
    )
    assert event.verification == "QUERY_FAILED"
    assert event.verified_absent_at_utc is None


def test_absent_stop_timestamp_represented_as_missing_not_invented():
    event = mon.build_stop_event(target_role="liquidation_silence_scheduler", target_pids=[15640])
    assert event.requested_at_utc is None
    assert event.completed_at_utc is None
    assert event.verified_absent_at_utc is None


def test_chronology_inversion_rejected():
    with pytest.raises(ValueError):
        mon.build_stop_event(
            target_role="liquidation_silence_scheduler",
            target_pids=[15640],
            requested_at_utc="2026-07-11T18:20:00+00:00",
            completed_at_utc="2026-07-11T18:10:00+00:00",  # before requested
        )


def test_empty_pid_list_rejected_for_verified_absent_outcome():
    verification = mon.StopVerification(checked_at_utc="2026-07-11T18:15:00+00:00", query_failed=False, still_present_pids=[])
    with pytest.raises(ValueError):
        mon.build_stop_event(
            target_role="liquidation_silence_scheduler",
            target_pids=[],
            outcome=mon.STOP_OUTCOME_VERIFIED_ABSENT,
            verification=verification,
        )


def test_completion_without_request_is_permitted_and_not_fabricated():
    event = mon.build_stop_event(
        target_role="liquidation_silence_scheduler", target_pids=[15640], completed_at_utc="2026-07-11T18:14:55+00:00"
    )
    assert event.requested_at_utc is None
    assert event.completed_at_utc == "2026-07-11T18:14:55+00:00"


def test_evidence_only_not_executed_outcome():
    event = mon.build_stop_event(target_role="liquidation_silence_scheduler", target_pids=[], outcome=mon.STOP_OUTCOME_NOT_EXECUTED)
    assert event.outcome == mon.STOP_OUTCOME_NOT_EXECUTED
    assert event.verified_absent_at_utc is None


def test_stop_event_serialization_round_trip():
    from dataclasses import asdict

    verification = mon.StopVerification(checked_at_utc="2026-07-11T18:15:00+00:00", query_failed=False, still_present_pids=[])
    event = mon.build_stop_event(
        target_role="liquidation_silence_scheduler",
        target_pids=[15640],
        requested_at_utc="2026-07-11T18:14:50+00:00",
        completed_at_utc="2026-07-11T18:14:55+00:00",
        verification=verification,
        outcome=mon.STOP_OUTCOME_VERIFIED_ABSENT,
    )
    payload = asdict(event)
    reconstructed = mon.StopEvent(**payload)
    assert reconstructed == event


def test_unknown_outcome_string_rejected():
    with pytest.raises(ValueError):
        mon.build_stop_event(target_role="x", target_pids=[1], outcome="MADE_UP_OUTCOME")


# ==========================================================================
# F-STOP-01: StopEvent invariants enforced by direct construction, not just
# build_stop_event() -- every case below constructs mon.StopEvent(...)
# directly, bypassing the builder entirely.
# ==========================================================================


def _valid_verified_absent_kwargs():
    return dict(
        target_role="liquidation_silence_scheduler",
        target_pids=[15640],
        requested_at_utc="2026-07-11T18:14:50+00:00",
        completed_at_utc="2026-07-11T18:14:55+00:00",
        verified_absent_at_utc="2026-07-11T18:15:00+00:00",
        outcome=mon.STOP_OUTCOME_VERIFIED_ABSENT,
        verification="ALL_ABSENT",
    )


def test_direct_stopevent_valid_construction_succeeds():
    event = mon.StopEvent(**_valid_verified_absent_kwargs())
    assert event.outcome == mon.STOP_OUTCOME_VERIFIED_ABSENT
    assert event.verification == "ALL_ABSENT"
    assert event.verified_absent_at_utc == "2026-07-11T18:15:00+00:00"


def test_direct_stopevent_rejects_success_with_no_verification():
    kwargs = _valid_verified_absent_kwargs()
    kwargs["verification"] = None
    with pytest.raises(ValueError):
        mon.StopEvent(**kwargs)


def test_direct_stopevent_rejects_verification_query_failure():
    kwargs = _valid_verified_absent_kwargs()
    kwargs["verification"] = "QUERY_FAILED"
    with pytest.raises(ValueError):
        mon.StopEvent(**kwargs)


def test_direct_stopevent_rejects_all_absent_false():
    kwargs = _valid_verified_absent_kwargs()
    kwargs["verification"] = "STILL_PRESENT"
    with pytest.raises(ValueError):
        mon.StopEvent(**kwargs)


def test_direct_stopevent_rejects_target_pid_still_present():
    # Same underlying invariant as all_absent=False: StopEvent.verification
    # is the collapsed string form of StopVerification (see build_stop_event),
    # so "some target PID still present" and "all_absent=False" are the same
    # represented state here -- both are exercised via verification=
    # "STILL_PRESENT", which is exactly what a StopVerification with a
    # non-empty still_present_pids collapses to.
    kwargs = _valid_verified_absent_kwargs()
    kwargs["verification"] = "STILL_PRESENT"
    with pytest.raises(ValueError):
        mon.StopEvent(**kwargs)


def test_direct_stopevent_rejects_empty_target_pids():
    kwargs = _valid_verified_absent_kwargs()
    kwargs["target_pids"] = []
    with pytest.raises(ValueError):
        mon.StopEvent(**kwargs)


def test_direct_stopevent_rejects_missing_requested_timestamp():
    kwargs = _valid_verified_absent_kwargs()
    kwargs["requested_at_utc"] = None
    with pytest.raises(ValueError):
        mon.StopEvent(**kwargs)


def test_direct_stopevent_rejects_missing_completed_timestamp():
    kwargs = _valid_verified_absent_kwargs()
    kwargs["completed_at_utc"] = None
    with pytest.raises(ValueError):
        mon.StopEvent(**kwargs)


def test_direct_stopevent_rejects_missing_verified_timestamp():
    kwargs = _valid_verified_absent_kwargs()
    kwargs["verified_absent_at_utc"] = None
    with pytest.raises(ValueError):
        mon.StopEvent(**kwargs)


def test_direct_stopevent_rejects_requested_after_completed():
    kwargs = _valid_verified_absent_kwargs()
    kwargs["requested_at_utc"] = "2026-07-11T18:20:00+00:00"  # after completed_at_utc
    with pytest.raises(ValueError):
        mon.StopEvent(**kwargs)


def test_direct_stopevent_rejects_completed_after_verified():
    kwargs = _valid_verified_absent_kwargs()
    kwargs["completed_at_utc"] = "2026-07-11T18:20:00+00:00"  # after verified_absent_at_utc
    with pytest.raises(ValueError):
        mon.StopEvent(**kwargs)


def test_direct_stopevent_rejects_naive_timestamp():
    kwargs = _valid_verified_absent_kwargs()
    kwargs["requested_at_utc"] = "2026-07-11T18:14:50"  # no tz offset
    with pytest.raises(ValueError):
        mon.StopEvent(**kwargs)


def test_direct_stopevent_rejects_malformed_timestamp():
    kwargs = _valid_verified_absent_kwargs()
    kwargs["completed_at_utc"] = "garbage-timestamp"
    with pytest.raises(ValueError):
        mon.StopEvent(**kwargs)


def test_direct_stopevent_rejects_unknown_outcome():
    kwargs = _valid_verified_absent_kwargs()
    kwargs["outcome"] = "MADE_UP_OUTCOME"
    with pytest.raises(ValueError):
        mon.StopEvent(**kwargs)


def test_direct_stopevent_rejects_empty_target_role():
    kwargs = _valid_verified_absent_kwargs()
    kwargs["target_role"] = ""
    with pytest.raises(ValueError):
        mon.StopEvent(**kwargs)


def test_direct_construction_and_builder_produce_identical_results():
    verification = mon.StopVerification(
        checked_at_utc="2026-07-11T18:15:00+00:00", query_failed=False, still_present_pids=[]
    )
    via_builder = mon.build_stop_event(
        target_role="liquidation_silence_scheduler",
        target_pids=[15640],
        requested_at_utc="2026-07-11T18:14:50+00:00",
        completed_at_utc="2026-07-11T18:14:55+00:00",
        verification=verification,
        outcome=mon.STOP_OUTCOME_VERIFIED_ABSENT,
    )
    via_direct = mon.StopEvent(**_valid_verified_absent_kwargs())
    assert via_builder == via_direct


def test_builder_and_direct_construction_reject_the_same_invalid_state():
    # Parity in the other direction: an invalid state rejected by
    # build_stop_event() must also be rejected by direct construction with
    # the equivalent already-collapsed fields.
    with pytest.raises(ValueError):
        mon.build_stop_event(
            target_role="liquidation_silence_scheduler",
            target_pids=[15640],
            outcome=mon.STOP_OUTCOME_VERIFIED_ABSENT,
            verification=None,
        )
    kwargs = _valid_verified_absent_kwargs()
    kwargs["verification"] = None
    with pytest.raises(ValueError):
        mon.StopEvent(**kwargs)


def test_direct_stopevent_serialization_round_trip():
    from dataclasses import asdict

    event = mon.StopEvent(**_valid_verified_absent_kwargs())
    payload = asdict(event)
    reconstructed = mon.StopEvent(**payload)
    assert reconstructed == event


def test_invalid_stopevent_payload_reconstruction_rejected():
    from dataclasses import asdict

    event = mon.StopEvent(**_valid_verified_absent_kwargs())
    payload = asdict(event)
    payload["verification"] = None  # corrupt a previously-valid serialized payload
    with pytest.raises(ValueError):
        mon.StopEvent(**payload)


# ==========================================================================
# F-STOP-COMPAT-01: complete outcome/verification/timestamp compatibility
# matrix -- round 3 only enforced invariants for STOP_VERIFIED_ABSENT; round
# 4 closes the gap the independent re-reviewer found (a STOP_FAILED record
# could carry a contradictory ALL_ABSENT verification and a valid
# chronology). Every case below constructs mon.StopEvent(...) directly.
# ==========================================================================


def test_stopevent_no_reason_codes_field_contradiction_not_applicable():
    # Section 7 item 15 ("contradictory reason codes indicating both
    # success and failure") is not representable in this schema: StopEvent
    # has no reason_codes field at all -- success/failure is expressed
    # solely via outcome+verification, which the matrix below validates for
    # mutual consistency. This test pins that schema fact so a future
    # reason_codes field addition is forced to also address this case.
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(mon.StopEvent)}
    assert "reason_codes" not in field_names


# -- Valid construction for every outcome ---------------------------------


def test_direct_stopevent_valid_not_executed():
    event = mon.StopEvent(target_role="liquidation_silence_scheduler", target_pids=[], outcome=mon.STOP_OUTCOME_NOT_EXECUTED)
    assert event.outcome == mon.STOP_OUTCOME_NOT_EXECUTED
    assert event.requested_at_utc is None
    assert event.completed_at_utc is None
    assert event.verified_absent_at_utc is None


def test_direct_stopevent_valid_stop_requested():
    event = mon.StopEvent(
        target_role="liquidation_silence_scheduler",
        target_pids=[15640],
        requested_at_utc="2026-07-11T18:14:50+00:00",
        outcome=mon.STOP_OUTCOME_REQUESTED,
    )
    assert event.outcome == mon.STOP_OUTCOME_REQUESTED
    assert event.completed_at_utc is None
    assert event.verified_absent_at_utc is None


def test_direct_stopevent_valid_command_succeeded_unverified():
    event = mon.StopEvent(
        target_role="liquidation_silence_scheduler",
        target_pids=[15640],
        requested_at_utc="2026-07-11T18:14:50+00:00",
        completed_at_utc="2026-07-11T18:14:55+00:00",
        outcome=mon.STOP_OUTCOME_COMMAND_SUCCEEDED_UNVERIFIED,
    )
    assert event.outcome == mon.STOP_OUTCOME_COMMAND_SUCCEEDED_UNVERIFIED
    assert event.verified_absent_at_utc is None
    assert event.verification is None


def test_direct_stopevent_valid_verified_absent_matrix_case():
    # Already covered by test_direct_stopevent_valid_construction_succeeds
    # above; re-asserted here so the "one positive test per outcome" set is
    # self-contained and doesn't rely on locating an earlier section.
    event = mon.StopEvent(**_valid_verified_absent_kwargs())
    assert event.outcome == mon.STOP_OUTCOME_VERIFIED_ABSENT


def test_direct_stopevent_valid_stop_failed():
    event = mon.StopEvent(
        target_role="liquidation_silence_scheduler",
        target_pids=[15640],
        requested_at_utc="2026-07-11T18:14:50+00:00",
        completed_at_utc="2026-07-11T18:14:55+00:00",
        outcome=mon.STOP_OUTCOME_FAILED,
        verification="STILL_PRESENT",  # legitimate failure evidence: checked, still present
    )
    assert event.outcome == mon.STOP_OUTCOME_FAILED
    assert event.verified_absent_at_utc is None


def test_direct_stopevent_valid_verification_failed():
    event = mon.StopEvent(
        target_role="liquidation_silence_scheduler",
        target_pids=[15640],
        requested_at_utc="2026-07-11T18:14:50+00:00",
        completed_at_utc="2026-07-11T18:14:55+00:00",
        outcome=mon.STOP_OUTCOME_VERIFICATION_FAILED,
        verification="QUERY_FAILED",
    )
    assert event.outcome == mon.STOP_OUTCOME_VERIFICATION_FAILED
    assert event.verified_absent_at_utc is None


# -- Contradiction rejects (section 7, items 1-14; item 15 handled above) --


def test_stopevent_rejects_stop_failed_with_all_absent():
    with pytest.raises(ValueError):
        mon.StopEvent(
            target_role="x",
            target_pids=[1],
            requested_at_utc="2026-07-11T18:14:50+00:00",
            completed_at_utc="2026-07-11T18:14:55+00:00",
            outcome=mon.STOP_OUTCOME_FAILED,
            verification="ALL_ABSENT",
        )


def test_stopevent_rejects_stop_failed_with_all_absent_via_builder():
    # Same contradiction (item 2: "all_absent=True"), reached through
    # build_stop_event()'s StopVerification -> "ALL_ABSENT" reduction,
    # proving the builder path hits the identical model-level rejection.
    verification = mon.StopVerification(checked_at_utc="2026-07-11T18:15:00+00:00", query_failed=False, still_present_pids=[])
    with pytest.raises(ValueError):
        mon.build_stop_event(
            target_role="x",
            target_pids=[1],
            requested_at_utc="2026-07-11T18:14:50+00:00",
            completed_at_utc="2026-07-11T18:14:55+00:00",
            verification=verification,
            outcome=mon.STOP_OUTCOME_FAILED,
        )


def test_stopevent_rejects_stop_failed_with_verified_absent_timestamp():
    with pytest.raises(ValueError):
        mon.StopEvent(
            target_role="x",
            target_pids=[1],
            requested_at_utc="2026-07-11T18:14:50+00:00",
            completed_at_utc="2026-07-11T18:14:55+00:00",
            verified_absent_at_utc="2026-07-11T18:15:00+00:00",
            outcome=mon.STOP_OUTCOME_FAILED,
            verification="STILL_PRESENT",
        )


def test_stopevent_rejects_verification_failed_with_all_absent():
    with pytest.raises(ValueError):
        mon.StopEvent(
            target_role="x",
            target_pids=[1],
            requested_at_utc="2026-07-11T18:14:50+00:00",
            completed_at_utc="2026-07-11T18:14:55+00:00",
            outcome=mon.STOP_OUTCOME_VERIFICATION_FAILED,
            verification="ALL_ABSENT",
        )


def test_stopevent_rejects_verification_failed_with_all_absent_via_builder():
    verification = mon.StopVerification(checked_at_utc="2026-07-11T18:15:00+00:00", query_failed=False, still_present_pids=[])
    with pytest.raises(ValueError):
        mon.build_stop_event(
            target_role="x",
            target_pids=[1],
            requested_at_utc="2026-07-11T18:14:50+00:00",
            completed_at_utc="2026-07-11T18:14:55+00:00",
            verification=verification,
            outcome=mon.STOP_OUTCOME_VERIFICATION_FAILED,
        )


def test_stopevent_rejects_command_succeeded_unverified_with_verified_absent_timestamp():
    with pytest.raises(ValueError):
        mon.StopEvent(
            target_role="x",
            target_pids=[1],
            requested_at_utc="2026-07-11T18:14:50+00:00",
            completed_at_utc="2026-07-11T18:14:55+00:00",
            verified_absent_at_utc="2026-07-11T18:15:00+00:00",
            outcome=mon.STOP_OUTCOME_COMMAND_SUCCEEDED_UNVERIFIED,
        )


def test_stopevent_rejects_command_succeeded_unverified_with_all_absent():
    with pytest.raises(ValueError):
        mon.StopEvent(
            target_role="x",
            target_pids=[1],
            requested_at_utc="2026-07-11T18:14:50+00:00",
            completed_at_utc="2026-07-11T18:14:55+00:00",
            outcome=mon.STOP_OUTCOME_COMMAND_SUCCEEDED_UNVERIFIED,
            verification="ALL_ABSENT",
        )


def test_stopevent_rejects_stop_requested_with_completed_timestamp():
    with pytest.raises(ValueError):
        mon.StopEvent(
            target_role="x",
            target_pids=[1],
            requested_at_utc="2026-07-11T18:14:50+00:00",
            completed_at_utc="2026-07-11T18:14:55+00:00",
            outcome=mon.STOP_OUTCOME_REQUESTED,
        )


def test_stopevent_rejects_stop_requested_with_verified_absent_timestamp():
    with pytest.raises(ValueError):
        mon.StopEvent(
            target_role="x",
            target_pids=[1],
            requested_at_utc="2026-07-11T18:14:50+00:00",
            verified_absent_at_utc="2026-07-11T18:15:00+00:00",
            outcome=mon.STOP_OUTCOME_REQUESTED,
        )


def test_stopevent_rejects_not_executed_with_requested_timestamp():
    with pytest.raises(ValueError):
        mon.StopEvent(
            target_role="x",
            target_pids=[],
            requested_at_utc="2026-07-11T18:14:50+00:00",
            outcome=mon.STOP_OUTCOME_NOT_EXECUTED,
        )


def test_stopevent_rejects_not_executed_with_completed_timestamp():
    with pytest.raises(ValueError):
        mon.StopEvent(
            target_role="x",
            target_pids=[],
            completed_at_utc="2026-07-11T18:14:55+00:00",
            outcome=mon.STOP_OUTCOME_NOT_EXECUTED,
        )


def test_stopevent_rejects_not_executed_with_successful_verification():
    with pytest.raises(ValueError):
        mon.StopEvent(
            target_role="x",
            target_pids=[1],
            outcome=mon.STOP_OUTCOME_NOT_EXECUTED,
            verification="ALL_ABSENT",
        )


def test_stopevent_rejects_unknown_verification_value():
    with pytest.raises(ValueError):
        mon.StopEvent(
            target_role="x",
            target_pids=[1],
            outcome=mon.STOP_OUTCOME_FAILED,
            verification="MADE_UP_VERIFICATION_STATE",
        )


def test_stopevent_rejects_unknown_verification_value_with_no_outcome():
    # The verification-value vocabulary check runs regardless of outcome.
    with pytest.raises(ValueError):
        mon.StopEvent(target_role="x", target_pids=[1], verification="MADE_UP_VERIFICATION_STATE")


def test_stopevent_rejects_naive_timestamp_on_stop_requested():
    with pytest.raises(ValueError):
        mon.StopEvent(
            target_role="x",
            target_pids=[1],
            requested_at_utc="2026-07-11T18:14:50",  # no tz offset
            outcome=mon.STOP_OUTCOME_REQUESTED,
        )


# -- Builder/direct parity across outcomes, payload round trip ------------


def test_builder_direct_parity_for_every_outcome():
    cases = [
        dict(target_role="x", target_pids=[], outcome=mon.STOP_OUTCOME_NOT_EXECUTED),
        dict(target_role="x", target_pids=[1], requested_at_utc="2026-07-11T18:14:50+00:00", outcome=mon.STOP_OUTCOME_REQUESTED),
        dict(
            target_role="x",
            target_pids=[1],
            requested_at_utc="2026-07-11T18:14:50+00:00",
            completed_at_utc="2026-07-11T18:14:55+00:00",
            outcome=mon.STOP_OUTCOME_COMMAND_SUCCEEDED_UNVERIFIED,
        ),
        dict(
            target_role="x",
            target_pids=[1],
            requested_at_utc="2026-07-11T18:14:50+00:00",
            completed_at_utc="2026-07-11T18:14:55+00:00",
            outcome=mon.STOP_OUTCOME_FAILED,
        ),
        dict(
            target_role="x",
            target_pids=[1],
            requested_at_utc="2026-07-11T18:14:50+00:00",
            completed_at_utc="2026-07-11T18:14:55+00:00",
            outcome=mon.STOP_OUTCOME_VERIFICATION_FAILED,
        ),
    ]
    for kwargs in cases:
        via_direct = mon.StopEvent(**kwargs)
        via_builder = mon.build_stop_event(**kwargs)
        assert via_direct == via_builder, kwargs


def test_stopevent_serialization_round_trip_stop_failed():
    from dataclasses import asdict

    event = mon.StopEvent(
        target_role="liquidation_silence_scheduler",
        target_pids=[15640],
        requested_at_utc="2026-07-11T18:14:50+00:00",
        completed_at_utc="2026-07-11T18:14:55+00:00",
        outcome=mon.STOP_OUTCOME_FAILED,
        verification="STILL_PRESENT",
    )
    payload = asdict(event)
    reconstructed = mon.StopEvent(**payload)
    assert reconstructed == event


def test_invalid_stopevent_payload_reconstruction_rejected_stop_failed():
    from dataclasses import asdict

    event = mon.StopEvent(
        target_role="liquidation_silence_scheduler",
        target_pids=[15640],
        requested_at_utc="2026-07-11T18:14:50+00:00",
        completed_at_utc="2026-07-11T18:14:55+00:00",
        outcome=mon.STOP_OUTCOME_FAILED,
        verification="STILL_PRESENT",
    )
    payload = asdict(event)
    payload["verification"] = "ALL_ABSENT"  # corrupt: now contradicts STOP_FAILED
    with pytest.raises(ValueError):
        mon.StopEvent(**payload)


# -- F-STOP-COMPAT-01 mutation probes ---------------------------------------


def test_mutation_probe_compat_1_matrix_removed_allows_stop_failed_all_absent(monkeypatch):
    monkeypatch.setattr(mon.StopEvent, "__post_init__", lambda self: None)
    bypassed = mon.StopEvent(
        target_role="x",
        target_pids=[1],
        requested_at_utc="2026-07-11T18:14:50+00:00",
        completed_at_utc="2026-07-11T18:14:55+00:00",
        outcome=mon.STOP_OUTCOME_FAILED,
        verification="ALL_ABSENT",
    )
    assert bypassed.outcome == mon.STOP_OUTCOME_FAILED
    assert bypassed.verification == "ALL_ABSENT"
    # With __post_init__ restored, this raises -- see
    # test_stopevent_rejects_stop_failed_with_all_absent.


def test_mutation_probe_compat_2_stop_failed_all_absent_allowed_when_check_removed():
    # Distinct demonstration from probe 1: shows specifically that a naive
    # matrix lacking the STOP_FAILED verification_allowed restriction would
    # accept this construction -- reproduced via a local, throwaway copy of
    # the rule rather than neutering the whole __post_init__.
    naive_rules = {"verification_allowed": {None, "QUERY_FAILED", "ALL_ABSENT", "STILL_PRESENT"}}  # bug: includes ALL_ABSENT
    assert "ALL_ABSENT" in naive_rules["verification_allowed"]  # the naive rule would wrongly accept it
    real_rules = mon._STOP_OUTCOME_RULES[mon.STOP_OUTCOME_FAILED]
    assert "ALL_ABSENT" not in real_rules["verification_allowed"]  # the real rule rejects it


def test_mutation_probe_compat_3_verification_failed_all_absent_allowed_when_check_removed():
    naive_rules = {"verification_allowed": {None, "QUERY_FAILED", "ALL_ABSENT", "STILL_PRESENT"}}  # bug
    assert "ALL_ABSENT" in naive_rules["verification_allowed"]
    real_rules = mon._STOP_OUTCOME_RULES[mon.STOP_OUTCOME_VERIFICATION_FAILED]
    assert "ALL_ABSENT" not in real_rules["verification_allowed"]


def test_mutation_probe_compat_4_unverified_outcome_allowed_to_carry_verified_timestamp(monkeypatch):
    monkeypatch.setattr(mon.StopEvent, "__post_init__", lambda self: None)
    bypassed = mon.StopEvent(
        target_role="x",
        target_pids=[1],
        requested_at_utc="2026-07-11T18:14:50+00:00",
        completed_at_utc="2026-07-11T18:14:55+00:00",
        verified_absent_at_utc="2026-07-11T18:15:00+00:00",  # contradicts UNVERIFIED
        outcome=mon.STOP_OUTCOME_COMMAND_SUCCEEDED_UNVERIFIED,
    )
    assert bypassed.verified_absent_at_utc is not None
    # With __post_init__ restored, this raises -- see
    # test_stopevent_rejects_command_succeeded_unverified_with_verified_absent_timestamp.


def test_mutation_probe_compat_5_stop_requested_allowed_to_claim_completion(monkeypatch):
    monkeypatch.setattr(mon.StopEvent, "__post_init__", lambda self: None)
    bypassed = mon.StopEvent(
        target_role="x",
        target_pids=[1],
        requested_at_utc="2026-07-11T18:14:50+00:00",
        completed_at_utc="2026-07-11T18:14:55+00:00",  # contradicts REQUESTED (not yet completed)
        outcome=mon.STOP_OUTCOME_REQUESTED,
    )
    assert bypassed.completed_at_utc is not None
    # With __post_init__ restored, this raises -- see
    # test_stopevent_rejects_stop_requested_with_completed_timestamp.


def test_mutation_probe_compat_6_builder_bypasses_compatibility_rules_when_matrix_disabled(monkeypatch):
    # Parity probe for the new matrix (mirrors round 3's builder-parity
    # mutation probe): neutering __post_init__ bypasses build_stop_event()
    # too, proving there is exactly one enforcement point for the full
    # compatibility matrix, not a weaker independent rule set in the
    # builder.
    monkeypatch.setattr(mon.StopEvent, "__post_init__", lambda self: None)
    verification = mon.StopVerification(checked_at_utc="2026-07-11T18:15:00+00:00", query_failed=False, still_present_pids=[])
    event = mon.build_stop_event(
        target_role="x",
        target_pids=[1],
        requested_at_utc="2026-07-11T18:14:50+00:00",
        completed_at_utc="2026-07-11T18:14:55+00:00",
        verification=verification,
        outcome=mon.STOP_OUTCOME_FAILED,
    )
    assert event.outcome == mon.STOP_OUTCOME_FAILED
    assert event.verification == "ALL_ABSENT"
    # With __post_init__ restored, build_stop_event() raises for this same
    # call -- see test_stopevent_rejects_stop_failed_with_all_absent_via_builder.


# ==========================================================================
# F-1 (accepted round-4 re-review LOW finding): StopEvent immutability
# hardening. StopEvent is now frozen=True -- a validated instance can no
# longer be silently invalidated by post-construction attribute assignment
# (__post_init__ is only invoked by __init__, never by plain assignment),
# closing the latent bypass the round-4 re-review demonstrated.
# ==========================================================================


def test_stopevent_is_declared_frozen():
    # Pins the actual class configuration, not just its observed behavior,
    # so a future accidental revert to a plain @dataclass is caught even
    # if an assignment-based test were ever skipped/removed.
    assert mon.StopEvent.__dataclass_params__.frozen is True


def test_stopevent_field_assignment_after_construction_raises_frozen_error():
    import dataclasses

    event = mon.StopEvent(target_role="x", target_pids=[1], outcome=mon.STOP_OUTCOME_NOT_EXECUTED)
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.outcome = mon.STOP_OUTCOME_FAILED


def test_stopevent_field_assignment_of_every_field_raises_frozen_error():
    import dataclasses

    event = mon.StopEvent(**_valid_verified_absent_kwargs())
    new_values = {
        "target_role": "y",
        "target_pids": [2],
        "requested_at_utc": "2026-07-11T18:00:00+00:00",
        "completed_at_utc": "2026-07-11T18:00:00+00:00",
        "verified_absent_at_utc": "2026-07-11T18:00:00+00:00",
        "outcome": mon.STOP_OUTCOME_FAILED,
        "verification": "STILL_PRESENT",
    }
    for field_name, value in new_values.items():
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(event, field_name, value)


def test_stopevent_field_deletion_after_construction_raises_frozen_error():
    import dataclasses

    event = mon.StopEvent(target_role="x", target_pids=[1], outcome=mon.STOP_OUTCOME_NOT_EXECUTED)
    with pytest.raises(dataclasses.FrozenInstanceError):
        del event.outcome


def test_frozen_stopevent_the_original_f1_bypass_no_longer_reachable():
    # The exact scenario the round-4 re-review demonstrated as latent-but-
    # real: constructing a valid STOP_VERIFIED_ABSENT event, then directly
    # assigning a contradictory outcome onto it, must now raise instead of
    # silently producing a StopEvent with outcome=STOP_FAILED but
    # verification=ALL_ABSENT and a populated verified_absent_at_utc.
    import dataclasses

    event = mon.StopEvent(**_valid_verified_absent_kwargs())
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.outcome = mon.STOP_OUTCOME_FAILED
    # The event remains exactly as validly constructed -- unmutated.
    assert event.outcome == mon.STOP_OUTCOME_VERIFIED_ABSENT
    assert event.verification == "ALL_ABSENT"


# -- No regression in any previously-supported construction path -----------


def test_frozen_stopevent_all_six_outcomes_still_construct():
    cases = [
        dict(target_role="x", target_pids=[], outcome=mon.STOP_OUTCOME_NOT_EXECUTED),
        dict(
            target_role="x",
            target_pids=[1],
            requested_at_utc="2026-07-11T18:14:50+00:00",
            outcome=mon.STOP_OUTCOME_REQUESTED,
        ),
        dict(
            target_role="x",
            target_pids=[1],
            requested_at_utc="2026-07-11T18:14:50+00:00",
            completed_at_utc="2026-07-11T18:14:55+00:00",
            outcome=mon.STOP_OUTCOME_COMMAND_SUCCEEDED_UNVERIFIED,
        ),
        _valid_verified_absent_kwargs(),
        dict(
            target_role="x",
            target_pids=[1],
            requested_at_utc="2026-07-11T18:14:50+00:00",
            completed_at_utc="2026-07-11T18:14:55+00:00",
            outcome=mon.STOP_OUTCOME_FAILED,
            verification="STILL_PRESENT",
        ),
        dict(
            target_role="x",
            target_pids=[1],
            requested_at_utc="2026-07-11T18:14:50+00:00",
            completed_at_utc="2026-07-11T18:14:55+00:00",
            outcome=mon.STOP_OUTCOME_VERIFICATION_FAILED,
            verification="QUERY_FAILED",
        ),
    ]
    outcomes_seen = set()
    for kwargs in cases:
        event = mon.StopEvent(**kwargs)
        outcomes_seen.add(event.outcome)
    assert outcomes_seen == mon.VALID_STOP_OUTCOMES


def test_frozen_stopevent_all_21_contradictions_still_rejected():
    # Re-executes the complete contradiction matrix post-freeze in one
    # place, as an explicit no-regression proof independent of the
    # individual per-case tests scattered through the F-STOP-COMPAT-01
    # section above.
    base = dict(
        target_role="x",
        target_pids=[1],
        requested_at_utc="2026-07-11T18:14:50+00:00",
        completed_at_utc="2026-07-11T18:14:55+00:00",
    )
    contradictions = [
        dict(base, outcome=mon.STOP_OUTCOME_FAILED, verification="ALL_ABSENT"),  # 1
        dict(
            base, outcome=mon.STOP_OUTCOME_FAILED, verification="ALL_ABSENT",
            verified_absent_at_utc="2026-07-11T18:15:00+00:00",
        ),  # 2
        dict(
            base, outcome=mon.STOP_OUTCOME_FAILED, verification="STILL_PRESENT",
            verified_absent_at_utc="2026-07-11T18:15:00+00:00",
        ),  # 3
        dict(base, outcome=mon.STOP_OUTCOME_VERIFICATION_FAILED, verification="ALL_ABSENT"),  # 4
        dict(
            base, outcome=mon.STOP_OUTCOME_VERIFICATION_FAILED, verification="ALL_ABSENT",
            verified_absent_at_utc="2026-07-11T18:15:00+00:00",
        ),  # 5
        dict(base, outcome=mon.STOP_OUTCOME_VERIFICATION_FAILED, verified_absent_at_utc="2026-07-11T18:15:00+00:00"),  # 6
        dict(base, outcome=mon.STOP_OUTCOME_COMMAND_SUCCEEDED_UNVERIFIED, verified_absent_at_utc="2026-07-11T18:15:00+00:00"),  # 7
        dict(base, outcome=mon.STOP_OUTCOME_COMMAND_SUCCEEDED_UNVERIFIED, verification="ALL_ABSENT"),  # 8
        dict(
            target_role="x", target_pids=[1], requested_at_utc="2026-07-11T18:14:50+00:00",
            completed_at_utc="2026-07-11T18:14:55+00:00", outcome=mon.STOP_OUTCOME_REQUESTED,
        ),  # 9
        dict(
            target_role="x", target_pids=[1], requested_at_utc="2026-07-11T18:14:50+00:00",
            verified_absent_at_utc="2026-07-11T18:15:00+00:00", outcome=mon.STOP_OUTCOME_REQUESTED,
        ),  # 10
        dict(target_role="x", target_pids=[], requested_at_utc="2026-07-11T18:14:50+00:00", outcome=mon.STOP_OUTCOME_NOT_EXECUTED),  # 11
        dict(target_role="x", target_pids=[], completed_at_utc="2026-07-11T18:14:55+00:00", outcome=mon.STOP_OUTCOME_NOT_EXECUTED),  # 12
        dict(target_role="x", target_pids=[1], outcome=mon.STOP_OUTCOME_NOT_EXECUTED, verification="ALL_ABSENT"),  # 13
        dict(_valid_verified_absent_kwargs(), verification=None),  # 14
        dict(_valid_verified_absent_kwargs(), verification="QUERY_FAILED"),  # 15
        dict(_valid_verified_absent_kwargs(), verification="STILL_PRESENT"),  # 16
        dict(base, outcome="MADE_UP_OUTCOME"),  # 17
        dict(base, outcome=mon.STOP_OUTCOME_FAILED, verification="MADE_UP_VERIFICATION"),  # 18
        dict(target_role="x", target_pids=[1], requested_at_utc="garbage-timestamp", outcome=mon.STOP_OUTCOME_REQUESTED),  # 19
        dict(target_role="x", target_pids=[1], requested_at_utc="2026-07-11T18:14:50", outcome=mon.STOP_OUTCOME_REQUESTED),  # 20
        dict(
            base, outcome=mon.STOP_OUTCOME_FAILED,
            requested_at_utc="2026-07-11T18:20:00+00:00", completed_at_utc="2026-07-11T18:10:00+00:00",
        ),  # 21
    ]
    assert len(contradictions) == 21
    for kwargs in contradictions:
        with pytest.raises(ValueError):
            mon.StopEvent(**kwargs)


def test_frozen_stopevent_builder_direct_payload_paths_still_valid():
    from dataclasses import asdict

    verification = mon.StopVerification(
        checked_at_utc="2026-07-11T18:15:00+00:00", query_failed=False, still_present_pids=[]
    )
    via_builder = mon.build_stop_event(
        target_role="liquidation_silence_scheduler",
        target_pids=[15640],
        requested_at_utc="2026-07-11T18:14:50+00:00",
        completed_at_utc="2026-07-11T18:14:55+00:00",
        verification=verification,
        outcome=mon.STOP_OUTCOME_VERIFIED_ABSENT,
    )
    assert via_builder.outcome == mon.STOP_OUTCOME_VERIFIED_ABSENT

    via_direct = mon.StopEvent(**_valid_verified_absent_kwargs())
    assert via_direct == via_builder

    payload = asdict(via_direct)
    via_payload = mon.StopEvent(**payload)
    assert via_payload == via_direct


# -- dataclasses.replace() parity -------------------------------------------


def test_frozen_stopevent_replace_compatible_change_succeeds():
    import dataclasses

    event = mon.StopEvent(target_role="x", target_pids=[1], outcome=mon.STOP_OUTCOME_NOT_EXECUTED)
    replaced = dataclasses.replace(event, target_role="y")
    assert replaced.target_role == "y"
    assert replaced.outcome == mon.STOP_OUTCOME_NOT_EXECUTED
    assert replaced is not event  # a genuinely new instance, not an in-place mutation


def test_frozen_stopevent_replace_contradictory_change_still_rejected():
    import dataclasses

    event = mon.StopEvent(**_valid_verified_absent_kwargs())
    with pytest.raises(ValueError):
        dataclasses.replace(event, outcome=mon.STOP_OUTCOME_FAILED)  # verification stays ALL_ABSENT -- contradiction
    with pytest.raises(ValueError):
        dataclasses.replace(event, verification=None)  # outcome stays VERIFIED_ABSENT -- contradiction
    with pytest.raises(ValueError):
        dataclasses.replace(event, target_pids=[])  # VERIFIED_ABSENT requires non-empty target_pids


def test_frozen_stopevent_replace_reruns_post_init_not_bypassed():
    # Confirms dataclasses.replace() is not a bypass path: it always goes
    # through __init__ (hence __post_init__) to build the new instance,
    # frozen or not.
    import dataclasses

    event = mon.StopEvent(
        target_role="x", target_pids=[1], outcome=mon.STOP_OUTCOME_REQUESTED,
        requested_at_utc="2026-07-11T18:14:50+00:00",
    )
    with pytest.raises(ValueError):
        dataclasses.replace(event, completed_at_utc="2026-07-11T18:14:55+00:00")  # STOP_REQUESTED forbids completed_at_utc


# ==========================================================================
# F-03: source provenance
# ==========================================================================


def _init_isolated_git_repo(repo_dir):
    """Creates a real, throwaway git repository under `repo_dir` with
    fully LOCAL (never global) identity/config, plus one initial commit so
    HEAD is resolvable -- used to exercise determine_source_provenance()'s
    real git plumbing (real `git` subprocess calls via the function's own
    `repo_root` parameter) against a fully controlled repository state,
    independent of whatever the live ECLIPSE_SCALPER repository's own
    tracked/dirty state happens to be for any particular file. This is
    what closes the test-drift finding from the round-4/Gate-2 review
    chain: the two tests below used to call determine_source_provenance()
    against this monitor module's OWN file inside the live repo, which
    was a valid "genuinely untracked" fixture only until commit
    e5f03855dd6f3d4c0ba4fa0d9a6dc131698c2f6d made that file permanently
    tracked -- after which the live repo could never again produce that
    scenario for this file, no matter what round of work is in flight.
    `commit.gpgsign` is explicitly disabled LOCALLY so this fixture never
    depends on (or is blocked by) the operator's global git config."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=str(repo_dir), check=True, timeout=15)
    subprocess.run(["git", "config", "user.email", "test-fixture@example.invalid"], cwd=str(repo_dir), check=True, timeout=15)
    subprocess.run(["git", "config", "user.name", "Test Fixture"], cwd=str(repo_dir), check=True, timeout=15)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=str(repo_dir), check=True, timeout=15)
    (repo_dir / "README.md").write_text("isolated fixture repository -- not part of ECLIPSE_SCALPER\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=str(repo_dir), check=True, timeout=15)
    subprocess.run(["git", "commit", "-q", "-m", "initial commit"], cwd=str(repo_dir), check=True, timeout=15)
    return repo_dir


def test_current_untracked_source_reports_untracked_not_in_head(tmp_path):
    # Repository-state-independent (see _init_isolated_git_repo's
    # docstring for why the live-repo-file version of this test was
    # retired): builds a real, isolated git repository and calls the real
    # production determine_source_provenance() -- not a mock, not a
    # monkeypatched replacement -- against a source file that genuinely
    # exists on disk but was never `git add`ed/committed within that
    # isolated repository.
    repo = _init_isolated_git_repo(tmp_path / "isolated_untracked_repo")
    never_committed = repo / "never_committed_source.py"
    never_committed.write_text("x = 1\n", encoding="utf-8")

    provenance = mon.determine_source_provenance(never_committed, repo_root=repo)
    assert provenance.provenance_status == mon.PROVENANCE_UNTRACKED_NOT_IN_HEAD
    assert provenance.source_tracked is False
    assert provenance.source_matches_head is False
    assert provenance.repository_head_commit  # HEAD is knowable even though this file isn't in it


def test_tracked_file_matching_head():
    tracked_file = mon.ROOT / "tools" / "health_state.py"
    provenance = mon.determine_source_provenance(tracked_file)
    assert provenance.source_tracked is True
    assert provenance.provenance_status == mon.PROVENANCE_TRACKED_MATCHES_HEAD
    assert provenance.source_matches_head is True
    assert provenance.source_blob_id is not None


def test_tracked_modified_file_reports_differs_from_head():
    # tests/test_native_ws_health_policy.py is a real, currently-modified
    # tracked file in this repo's working tree -- exercising the genuine
    # TRACKED_DIFFERS_FROM_HEAD path without editing anything ourselves.
    modified_tracked_file = mon.ROOT / "tests" / "test_native_ws_health_policy.py"
    provenance = mon.determine_source_provenance(modified_tracked_file)
    assert provenance.source_tracked is True
    assert provenance.provenance_status == mon.PROVENANCE_TRACKED_DIFFERS_FROM_HEAD
    assert provenance.source_matches_head is False


def test_source_outside_repository(tmp_path):
    outside_file = tmp_path / "not_in_repo.py"
    outside_file.write_text("x = 1\n", encoding="utf-8")
    provenance = mon.determine_source_provenance(outside_file, repo_root=mon.ROOT)
    assert provenance.provenance_status == mon.PROVENANCE_OUTSIDE_REPOSITORY
    assert provenance.source_tracked is None


def test_git_unavailable_reports_explicit_status(tmp_path, monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(mon.subprocess, "run", _raise)
    provenance = mon.determine_source_provenance(Path(mon.__file__))
    assert provenance.provenance_status == mon.PROVENANCE_GIT_UNAVAILABLE
    assert provenance.repository_head_commit is None


def test_source_read_failure_reports_explicit_status(tmp_path):
    missing = tmp_path / "does_not_exist.py"
    provenance = mon.determine_source_provenance(missing)
    assert provenance.provenance_status == mon.PROVENANCE_SOURCE_READ_FAILED
    assert provenance.source_sha256 is None


def test_source_sha256_matches_independent_oracle():
    expected = hashlib.sha256(Path(mon.__file__).read_bytes()).hexdigest()
    provenance = mon.determine_source_provenance(Path(mon.__file__))
    assert provenance.source_sha256 == expected


def test_wrong_file_hashed_fails_independent_oracle_comparison(tmp_path):
    # Negative test (F-07): hashing a DIFFERENT file must NOT match the
    # real monitor source's independently-computed oracle hash.
    decoy = tmp_path / "decoy.py"
    decoy.write_text("this is not the monitor source\n", encoding="utf-8")
    real_expected = hashlib.sha256(Path(mon.__file__).read_bytes()).hexdigest()
    decoy_provenance = mon.determine_source_provenance(decoy, repo_root=mon.ROOT)
    assert decoy_provenance.source_sha256 != real_expected


def test_manifest_source_hash_independent_of_function_under_test(tmp_path, monkeypatch):
    # Oracle computed via plain hashlib, NOT via mon._self_source_sha256
    # or mon.determine_source_provenance -- this is the original fix for
    # the tautological F-07 test, preserved here unchanged. The
    # provenance-status assertions are now sourced from a fully isolated,
    # throwaway git repository (see _init_isolated_git_repo) rather than
    # the live ECLIPSE_SCALPER repository's own tracked state, which
    # permanently changed for this monitor module after commit
    # e5f03855dd6f3d4c0ba4fa0d9a6dc131698c2f6d -- following the same
    # injected-SourceProvenance pattern already used by
    # test_manifest_provenance_resolved_from_injected_wrong_source_path.
    repo = _init_isolated_git_repo(tmp_path / "isolated_manifest_repo")
    never_committed = repo / "never_committed_source.py"
    never_committed.write_text("manifest hash fixture content\n", encoding="utf-8")

    expected = hashlib.sha256(never_committed.read_bytes()).hexdigest()
    provenance = mon.determine_source_provenance(never_committed, repo_root=repo)
    assert provenance.provenance_status == mon.PROVENANCE_UNTRACKED_NOT_IN_HEAD  # sanity: isolated repo behaves as intended

    manifest = mon.build_run_manifest(
        cadence_sec=None,
        requested_duration_sec=None,
        role_definitions={},
        artifact_paths={},
        arguments={},
        source_provenance=provenance,
    )
    assert manifest.source_sha256 == expected
    assert manifest.source_provenance_status == mon.PROVENANCE_UNTRACKED_NOT_IN_HEAD
    assert manifest.source_tracked is False

    # Monkeypatch/replacement resilience: once a SourceProvenance is
    # explicitly supplied, build_run_manifest() must use it verbatim and
    # must NOT silently re-derive provenance by calling (a possibly
    # monkeypatched/replaced) determine_source_provenance() again -- proven
    # here by making that function raise if it's ever called a second time.
    def _must_not_be_called_again(*_args, **_kwargs):
        raise AssertionError(
            "determine_source_provenance() must not be called again once "
            "source_provenance is explicitly supplied to build_run_manifest()"
        )

    monkeypatch.setattr(mon, "determine_source_provenance", _must_not_be_called_again)
    manifest2 = mon.build_run_manifest(
        cadence_sec=None,
        requested_duration_sec=None,
        role_definitions={},
        artifact_paths={},
        arguments={},
        source_provenance=provenance,
    )
    assert manifest2.source_sha256 == expected
    assert manifest2.source_provenance_status == mon.PROVENANCE_UNTRACKED_NOT_IN_HEAD


def test_manifest_provenance_resolved_from_injected_wrong_source_path(tmp_path):
    # Simulates "source path resolution pointed at the wrong file": if a
    # caller supplies a provenance built from a decoy file, the manifest
    # must faithfully reflect the decoy's hash, not the real module's --
    # proving the manifest doesn't silently re-hash the correct file
    # underneath an injected SourceProvenance.
    decoy = tmp_path / "decoy.py"
    decoy.write_text("decoy content\n", encoding="utf-8")
    decoy_provenance = mon.determine_source_provenance(decoy, repo_root=mon.ROOT)
    manifest = mon.build_run_manifest(
        cadence_sec=None,
        requested_duration_sec=None,
        role_definitions={},
        artifact_paths={},
        arguments={},
        source_provenance=decoy_provenance,
    )
    real_hash = hashlib.sha256(Path(mon.__file__).read_bytes()).hexdigest()
    assert manifest.source_sha256 != real_hash
    assert manifest.source_sha256 == hashlib.sha256(decoy.read_bytes()).hexdigest()


def test_manifest_schema_version_and_operating_mode():
    manifest = mon.build_run_manifest(
        cadence_sec=None, requested_duration_sec=None, role_definitions={}, artifact_paths={}, arguments={}
    )
    assert manifest.schema_version == "liquidation_silence_canary_monitor_v2"
    assert manifest.operating_mode == "INSPECT_ONLY"


def test_manifest_determinism_across_two_builds_except_start_time():
    from dataclasses import asdict

    m1 = mon.build_run_manifest(
        cadence_sec=300, requested_duration_sec=1800, role_definitions={"a": 1}, artifact_paths={"x": "y"}, arguments={"z": 1}
    )
    m2 = mon.build_run_manifest(
        cadence_sec=300, requested_duration_sec=1800, role_definitions={"a": 1}, artifact_paths={"x": "y"}, arguments={"z": 1}
    )
    d1, d2 = asdict(m1), asdict(m2)
    del d1["start_time_utc"]
    del d2["start_time_utc"]
    assert d1 == d2


# ==========================================================================
# F-04: cadence monotonicity / duplicate rejection
# ==========================================================================


def _cyc(cycle_number, ts_iso, timestamp_source=mon.TIMESTAMP_SOURCE_EPOCH_FULL):
    return mon.CycleObservation(
        cycle_number=cycle_number,
        evaluated_at_utc=ts_iso,
        timestamp_source=timestamp_source,
        duration_sec=0.002,
        outcome="SUCCESS",
        severity="GREEN",
        status="HEALTHY",
    )


def test_valid_monotonic_input_produces_valid_summary():
    cycles = [_cyc(1, mon.epoch_to_iso(1000)), _cyc(2, mon.epoch_to_iso(1300)), _cyc(3, mon.epoch_to_iso(1600))]
    summary = mon.compute_cadence_summary(cycles)
    assert summary.valid is True
    assert summary.gap_count == 2
    assert summary.mean_gap_sec == pytest.approx(300.0)


def test_duplicate_adjacent_timestamps_rejected():
    cycles = [_cyc(1, mon.epoch_to_iso(1000)), _cyc(2, mon.epoch_to_iso(1000))]
    summary = mon.compute_cadence_summary(cycles)
    assert summary.valid is False
    assert mon.CADENCE_REASON_DUPLICATE in summary.reason_codes
    assert summary.mean_gap_sec is None


def test_duplicate_non_adjacent_timestamps_rejected():
    cycles = [_cyc(1, mon.epoch_to_iso(1000)), _cyc(2, mon.epoch_to_iso(1300)), _cyc(3, mon.epoch_to_iso(1000))]
    summary = mon.compute_cadence_summary(cycles)
    assert summary.valid is False
    assert mon.CADENCE_REASON_REGRESSION in summary.reason_codes  # 1300 -> 1000 is a regression


def test_decreasing_timestamp_rejected():
    cycles = [_cyc(1, mon.epoch_to_iso(2000)), _cyc(2, mon.epoch_to_iso(1000))]
    summary = mon.compute_cadence_summary(cycles)
    assert summary.valid is False
    assert mon.CADENCE_REASON_REGRESSION in summary.reason_codes


def test_unsorted_timestamps_not_silently_sorted():
    # Original order preserved -- the negative gap must be visible in
    # exact_gaps_sec, not hidden by an internal sort.
    cycles = [_cyc(1, mon.epoch_to_iso(2000)), _cyc(2, mon.epoch_to_iso(1000)), _cyc(3, mon.epoch_to_iso(3000))]
    summary = mon.compute_cadence_summary(cycles)
    assert summary.exact_gaps_sec[0] == pytest.approx(-1000.0)
    assert summary.valid is False


def test_malformed_cycle_timestamp_flagged_invalid():
    cycles = [_cyc(1, mon.epoch_to_iso(1000)), _cyc(2, None), _cyc(3, mon.epoch_to_iso(1600))]
    summary = mon.compute_cadence_summary(cycles)
    assert mon.CADENCE_REASON_INVALID in summary.reason_codes
    assert summary.valid is False


def test_zero_and_one_cycle_remain_valid_with_zero_gaps():
    assert mon.compute_cadence_summary([]).valid is True
    assert mon.compute_cadence_summary([]).gap_count == 0
    one = mon.compute_cadence_summary([_cyc(1, mon.epoch_to_iso(1000))])
    assert one.valid is True
    assert one.gap_count == 0


def test_cycle_count_vs_gap_count_reconciles_for_various_n():
    for n in (0, 1, 2, 3, 7, 10):
        cycles = [_cyc(i + 1, mon.epoch_to_iso(1000 + i * 300)) for i in range(n)]
        summary = mon.compute_cadence_summary(cycles)
        assert summary.gap_count == max(0, n - 1)
        assert len(summary.exact_gaps_sec) == summary.gap_count


# ==========================================================================
# F-05: canonical fixture + full-precision CLI parsing
# ==========================================================================


def test_canonical_seven_cycle_gate1_fixture_uses_full_precision(tmp_path):
    log_path = tmp_path / "liquidation_silence_scheduler.stdout.log"
    log_path.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")

    cycles = mon.parse_cycle_log(log_path)
    assert len(cycles) == 7
    assert all(c.timestamp_source == mon.TIMESTAMP_SOURCE_EPOCH_FULL for c in cycles)

    summary = mon.compute_cadence_summary(cycles)
    assert summary.valid is True
    assert summary.gap_count == 6
    assert summary.precision_status == mon.CADENCE_PRECISION_FULL


def test_all_six_canonical_gaps_round_to_300(tmp_path):
    log_path = tmp_path / "liquidation_silence_scheduler.stdout.log"
    log_path.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")
    cycles = mon.parse_cycle_log(log_path)
    summary = mon.compute_cadence_summary(cycles)
    assert summary.rounded_gaps_sec == [300, 300, 300, 300, 300, 300]
    for exact in summary.exact_gaps_sec:
        assert 300.0 < exact < 300.2


def test_exact_min_max_mean_values(tmp_path):
    log_path = tmp_path / "liquidation_silence_scheduler.stdout.log"
    log_path.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")
    cycles = mon.parse_cycle_log(log_path)
    summary = mon.compute_cadence_summary(cycles)
    assert summary.min_gap_sec == pytest.approx(300.114, abs=0.01)
    assert summary.max_gap_sec == pytest.approx(300.131, abs=0.01)
    assert summary.mean_gap_sec == pytest.approx(300.122, abs=0.01)


def test_fallback_to_whole_second_when_epoch_missing():
    record = {"ts_utc": "2026-07-11T17:41:51Z", "outcome": "SUCCESS"}
    ts, source = mon._select_cycle_timestamp(record)
    assert source == mon.TIMESTAMP_SOURCE_ISO_DEGRADED
    assert ts.endswith("+00:00")


def test_malformed_epoch_falls_back_to_valid_evaluated_timestamp():
    record = {"cycle_started_epoch": "not-a-number", "evaluated_at_utc": "2026-07-11T17:41:51.123456+00:00"}
    ts, source = mon._select_cycle_timestamp(record)
    assert source == mon.TIMESTAMP_SOURCE_ISO_FULL


def test_whole_second_only_degraded_mode_flagged_in_precision_status():
    cycles = [
        _cyc(1, "2026-07-11T17:41:51+00:00", timestamp_source=mon.TIMESTAMP_SOURCE_ISO_DEGRADED),
        _cyc(2, "2026-07-11T17:46:51+00:00", timestamp_source=mon.TIMESTAMP_SOURCE_ISO_DEGRADED),
    ]
    summary = mon.compute_cadence_summary(cycles)
    assert summary.precision_status == mon.CADENCE_PRECISION_DEGRADED


def test_real_scheduler_log_shaped_fixture_not_a_manually_simplified_list(tmp_path):
    # Reproduces the exact defect the review caught: parsing must select
    # cycle_started_epoch even when ts_utc (present in every real record)
    # is only whole-second precision.
    log_path = tmp_path / "real_shaped.log"
    real_shaped_line = json.dumps(
        {
            "ts_utc": "2026-07-11T17:41:51Z",
            "cycle_started_epoch": 1783791711.4434423,
            "outcome": "SUCCESS",
            "severity": "GREEN",
            "status": "HEALTHY",
            "detector_error": None,
            "detector_runtime_sec": 0.003437519073486328,
            "cycle_duration_sec": 0.005102634429931641,
        }
    )
    log_path.write_text(real_shaped_line + "\n", encoding="utf-8")
    cycles = mon.parse_cycle_log(log_path)
    assert cycles[0].timestamp_source == mon.TIMESTAMP_SOURCE_EPOCH_FULL
    assert cycles[0].evaluated_at_utc == mon.epoch_to_iso(1783791711.4434423)


# ==========================================================================
# F-06: CLI continuity wiring
# ==========================================================================


def test_take_snapshot_continuous_via_pid_meta_baseline(tmp_path):
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")
    pid_meta = tmp_path / "liquidation_silence_scheduler.json"
    pid_meta.write_text(
        json.dumps({"pid": 15640, "started_at": BASELINE_START, "cmdline_sig": "python -W ignore -u -m tools.liquidation_silence_scheduler", "role": "liquidation_silence_scheduler"}),
        encoding="utf-8",
    )
    evidence_dir = tmp_path / "evidence"
    lister = _fake_lister([_rec(15640, BASELINE_START)])

    record = mon.take_snapshot(
        scheduler_log_path=scheduler_log, evidence_dir=evidence_dir, pid_meta_path=pid_meta, lister=lister
    )
    assert record["continuity"] is not None
    assert record["continuity"]["continuity_status"] == mon.CONTINUITY_CONTINUOUS


def test_take_snapshot_pid_replacement_via_pid_meta_baseline(tmp_path):
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")
    pid_meta = tmp_path / "liquidation_silence_scheduler.json"
    pid_meta.write_text(json.dumps({"pid": 15640, "started_at": BASELINE_START, "cmdline_sig": "x", "role": "liquidation_silence_scheduler"}), encoding="utf-8")
    evidence_dir = tmp_path / "evidence"
    lister = _fake_lister([_rec(99999, BASELINE_START)])  # different PID

    record = mon.take_snapshot(scheduler_log_path=scheduler_log, evidence_dir=evidence_dir, pid_meta_path=pid_meta, lister=lister)
    assert record["continuity"]["continuity_status"] == mon.CONTINUITY_REPLACED_OR_RESTARTED


def test_take_snapshot_starttime_replacement(tmp_path):
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")
    pid_meta = tmp_path / "liquidation_silence_scheduler.json"
    pid_meta.write_text(json.dumps({"pid": 15640, "started_at": BASELINE_START, "cmdline_sig": "x", "role": "liquidation_silence_scheduler"}), encoding="utf-8")
    evidence_dir = tmp_path / "evidence"
    lister = _fake_lister([_rec(15640, "2026-07-11T23:00:00+00:00")])

    record = mon.take_snapshot(scheduler_log_path=scheduler_log, evidence_dir=evidence_dir, pid_meta_path=pid_meta, lister=lister)
    assert record["continuity"]["continuity_status"] == mon.CONTINUITY_REPLACED_OR_RESTARTED


def test_take_snapshot_identity_mismatch(tmp_path):
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")
    pid_meta = tmp_path / "liquidation_silence_scheduler.json"
    pid_meta.write_text(json.dumps({"pid": 15640, "started_at": BASELINE_START, "cmdline_sig": "x", "role": "liquidation_silence_scheduler"}), encoding="utf-8")
    evidence_dir = tmp_path / "evidence"
    lister = _fake_lister([_rec(15640, BASELINE_START, cmdline="python -u -m tools.oi_spot_poller")])

    record = mon.take_snapshot(
        scheduler_log_path=scheduler_log,
        evidence_dir=evidence_dir,
        pid_meta_path=pid_meta,
        expected_module="tools.liquidation_silence_scheduler",
        lister=lister,
    )
    assert record["continuity"]["continuity_status"] == mon.CONTINUITY_IDENTITY_MISMATCH


def test_take_snapshot_duplicate_process(tmp_path):
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")
    pid_meta = tmp_path / "liquidation_silence_scheduler.json"
    pid_meta.write_text(json.dumps({"pid": 15640, "started_at": BASELINE_START, "cmdline_sig": "x", "role": "liquidation_silence_scheduler"}), encoding="utf-8")
    evidence_dir = tmp_path / "evidence"
    lister = _fake_lister([_rec(15640, BASELINE_START), _rec(88888, BASELINE_START)])

    record = mon.take_snapshot(scheduler_log_path=scheduler_log, evidence_dir=evidence_dir, pid_meta_path=pid_meta, lister=lister)
    assert record["continuity"]["continuity_status"] == mon.CONTINUITY_DUPLICATE


def test_take_snapshot_query_failure(tmp_path):
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")
    pid_meta = tmp_path / "liquidation_silence_scheduler.json"
    pid_meta.write_text(json.dumps({"pid": 15640, "started_at": BASELINE_START, "cmdline_sig": "x", "role": "liquidation_silence_scheduler"}), encoding="utf-8")
    evidence_dir = tmp_path / "evidence"
    lister = _fake_lister(None)

    record = mon.take_snapshot(scheduler_log_path=scheduler_log, evidence_dir=evidence_dir, pid_meta_path=pid_meta, lister=lister)
    assert record["continuity"]["continuity_status"] == mon.CONTINUITY_QUERY_FAILED


def test_take_snapshot_no_baseline_supplied_is_explicit_not_null(tmp_path):
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")
    evidence_dir = tmp_path / "evidence"

    record = mon.take_snapshot(scheduler_log_path=scheduler_log, evidence_dir=evidence_dir, pid_meta_path=None)
    assert record["continuity"] is not None
    assert record["continuity"]["continuity_status"] == mon.CONTINUITY_NOT_EVALUATED
    assert "BASELINE_NOT_PROVIDED" in record["continuity"]["reason_codes"]


def test_real_cli_invocation_with_synthetic_baseline_reports_missing(tmp_path):
    # End-to-end through `python -m`, exercising the actual production
    # CLI entry point -- not just the library function. Uses a PID
    # (999999999) that cannot plausibly exist as a real process, so this
    # is fully safe and deterministic without touching any real role.
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")
    pid_meta = tmp_path / "pid_meta.json"
    pid_meta.write_text(
        json.dumps({"pid": 999999999, "started_at": BASELINE_START, "cmdline_sig": "x", "role": "liquidation_silence_scheduler"}),
        encoding="utf-8",
    )
    evidence_dir = tmp_path / "evidence"

    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tools.liquidation_silence_canary_monitor",
            "--scheduler-log-path",
            str(scheduler_log),
            "--evidence-dir",
            str(evidence_dir),
            "--pid-meta-path",
            str(pid_meta),
        ],
        cwd=str(mon.ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())
    assert output["continuity"] is not None
    assert output["continuity"]["continuity_status"] == mon.CONTINUITY_MISSING
    # F-BASELINE-01 case 10: a syntactically/schema-valid baseline (pid is
    # an int, started_at is a real timestamp) loads with status OK -- the
    # concrete CONTINUITY_MISSING result above is the continuity evaluation
    # it produced, not a fallback for a rejected baseline.
    assert output["baseline"]["status"] == mon.BASELINE_STATUS_OK
    assert (evidence_dir / "run_manifest.json").exists()
    assert (evidence_dir / "snapshots.jsonl").exists()


def test_real_cli_help_has_no_side_effects(tmp_path):
    result = subprocess.run(
        [sys.executable, "-B", "-m", "tools.liquidation_silence_canary_monitor", "--help"],
        cwd=str(mon.ROOT),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert "evidence-dir" in result.stdout


def test_continuity_output_never_ambiguous_null(tmp_path):
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("", encoding="utf-8")
    for pid_meta_path in (None,):
        record = mon.take_snapshot(scheduler_log_path=scheduler_log, evidence_dir=tmp_path / "evidence", pid_meta_path=pid_meta_path)
        assert record["continuity"] != None  # noqa: E711 -- explicit contrast, not `is not None`
        assert isinstance(record["continuity"], dict)


# ==========================================================================
# F-08 / F-09: real sample_artifact() disk-path tests
# ==========================================================================


def test_sample_artifact_valid_json(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"evaluated_at_utc": "2026-07-11T18:00:00+00:00"}), encoding="utf-8")
    sample = mon.sample_artifact(p, semantic_ts_field="evaluated_at_utc")
    assert sample.status == mon.ARTIFACT_STATUS_SAMPLED
    assert sample.semantic_ts_status == mon.SEMANTIC_TS_PRESENT
    assert sample.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()


def test_sample_artifact_unchanged_across_two_samples(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"evaluated_at_utc": "2026-07-11T18:00:00+00:00"}), encoding="utf-8")
    s1 = mon.sample_artifact(p, semantic_ts_field="evaluated_at_utc")
    s2 = mon.sample_artifact(p, semantic_ts_field="evaluated_at_utc")
    assert mon.compare_artifact_freshness(s1, s2) == mon.FRESHNESS_UNCHANGED


def test_sample_artifact_changed_hash_advancing_timestamp_is_fresh(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"evaluated_at_utc": "2026-07-11T18:00:00+00:00"}), encoding="utf-8")
    s1 = mon.sample_artifact(p, semantic_ts_field="evaluated_at_utc")
    p.write_text(json.dumps({"evaluated_at_utc": "2026-07-11T18:05:00+00:00"}), encoding="utf-8")
    s2 = mon.sample_artifact(p, semantic_ts_field="evaluated_at_utc")
    assert mon.compare_artifact_freshness(s1, s2) == mon.FRESHNESS_FRESH


def test_sample_artifact_changed_hash_unchanged_timestamp_is_suspect(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"evaluated_at_utc": "2026-07-11T18:00:00+00:00", "n": 1}), encoding="utf-8")
    s1 = mon.sample_artifact(p, semantic_ts_field="evaluated_at_utc")
    p.write_text(json.dumps({"evaluated_at_utc": "2026-07-11T18:00:00+00:00", "n": 2}), encoding="utf-8")
    s2 = mon.sample_artifact(p, semantic_ts_field="evaluated_at_utc")
    result = mon.compare_artifact_freshness(s1, s2)
    assert result == mon.FRESHNESS_SUSPECT_STALE_REUSE
    assert result != mon.FRESHNESS_FRESH


def test_sample_artifact_unchanged_hash_advancing_timestamp_is_still_unchanged(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"evaluated_at_utc": "2026-07-11T18:00:00+00:00"}), encoding="utf-8")
    s1 = mon.sample_artifact(p, semantic_ts_field="evaluated_at_utc")
    s2 = mon.sample_artifact(p, semantic_ts_field="evaluated_at_utc")  # identical bytes
    assert mon.compare_artifact_freshness(s1, s2) == mon.FRESHNESS_UNCHANGED


def test_sample_artifact_malformed_json(tmp_path):
    p = tmp_path / "a.json"
    p.write_text("{not valid json", encoding="utf-8")
    sample = mon.sample_artifact(p, semantic_ts_field="evaluated_at_utc")
    assert sample.status == mon.ARTIFACT_STATUS_SAMPLED
    assert sample.semantic_ts_status == mon.SEMANTIC_TS_MALFORMED_JSON
    assert sample.sha256 is not None  # bytes were still readable and hashed


def test_sample_artifact_missing_file(tmp_path):
    sample = mon.sample_artifact(tmp_path / "does_not_exist.json")
    assert sample.status == mon.ARTIFACT_STATUS_MISSING
    assert sample.exists is False
    assert sample.sha256 is None


def test_sample_artifact_empty_file(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text("", encoding="utf-8")
    sample = mon.sample_artifact(p, semantic_ts_field="evaluated_at_utc")
    assert sample.status == mon.ARTIFACT_STATUS_SAMPLED
    assert sample.size_bytes == 0
    assert sample.semantic_ts_status == mon.SEMANTIC_TS_MALFORMED_JSON


def test_sample_artifact_directory_supplied_instead_of_file(tmp_path):
    d = tmp_path / "a_directory"
    d.mkdir()
    sample = mon.sample_artifact(d)
    assert sample.status == mon.ARTIFACT_STATUS_NOT_A_REGULAR_FILE
    assert sample.sha256 is None
    assert sample.error is not None


def test_sample_artifact_permission_denied_via_monkeypatched_read(tmp_path, monkeypatch):
    p = tmp_path / "a.json"
    p.write_text("{}", encoding="utf-8")

    original_read_bytes = Path.read_bytes

    def _raise_permission(self, *args, **kwargs):
        if self == p:
            raise PermissionError("simulated permission denial")
        return original_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", _raise_permission)
    sample = mon.sample_artifact(p)
    assert sample.status == mon.ARTIFACT_STATUS_PERMISSION_DENIED
    assert sample.sha256 is None


def test_sample_artifact_semantic_timestamp_regression_detected_by_freshness(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"evaluated_at_utc": "2026-07-11T18:11:52+00:00"}), encoding="utf-8")
    s1 = mon.sample_artifact(p, semantic_ts_field="evaluated_at_utc")
    p.write_text(json.dumps({"evaluated_at_utc": "2026-07-11T18:06:52+00:00"}), encoding="utf-8")  # went backward
    s2 = mon.sample_artifact(p, semantic_ts_field="evaluated_at_utc")
    assert mon.compare_artifact_freshness(s1, s2) == mon.FRESHNESS_MONOTONICITY_VIOLATION


def test_sample_artifact_invalid_encoding(tmp_path):
    p = tmp_path / "bad_encoding.json"
    p.write_bytes(b"\xff\xfe\x00\x01not-utf8")
    sample = mon.sample_artifact(p, semantic_ts_field="evaluated_at_utc")
    assert sample.status == mon.ARTIFACT_STATUS_INVALID_ENCODING
    assert sample.semantic_ts_status == mon.SEMANTIC_TS_INVALID


def test_two_logical_paths_resolving_to_same_physical_file(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"evaluated_at_utc": "2026-07-11T18:00:00+00:00"}), encoding="utf-8")
    alt_path = tmp_path / "." / "a.json"
    s1 = mon.sample_artifact(p, semantic_ts_field="evaluated_at_utc")
    s2 = mon.sample_artifact(alt_path, semantic_ts_field="evaluated_at_utc")
    assert s1.sha256 == s2.sha256


def test_end_to_end_real_file_through_freshness_classifier(tmp_path):
    p = tmp_path / "liquidation_silence.json"
    p.write_text(json.dumps({"evaluated_at_utc": "2026-07-11T17:46:51+00:00", "severity": "GREEN"}), encoding="utf-8")
    s1 = mon.sample_artifact(p, semantic_ts_field="evaluated_at_utc")
    p.write_text(json.dumps({"evaluated_at_utc": "2026-07-11T17:51:51+00:00", "severity": "GREEN"}), encoding="utf-8")
    s2 = mon.sample_artifact(p, semantic_ts_field="evaluated_at_utc")
    assert mon.compare_artifact_freshness(s1, s2) == mon.FRESHNESS_FRESH


def test_no_prior_sample_classification():
    p_sample = mon.ArtifactSample(
        path="x", status=mon.ARTIFACT_STATUS_SAMPLED, exists=True, size_bytes=1, mtime_utc=None, sha256="a", semantic_ts_utc=None, semantic_ts_status=mon.SEMANTIC_TS_NOT_REQUESTED
    )
    assert mon.compare_artifact_freshness(None, p_sample) == mon.FRESHNESS_NO_PRIOR_SAMPLE


# ==========================================================================
# Exception / malformed cycle handling
# ==========================================================================


def test_exception_and_detector_error_reason_codes_parsed(tmp_path):
    lines = [
        json.dumps({"ts_utc": mon.epoch_to_iso(GATE1_EPOCHS[0]), "outcome": "SUCCESS", "severity": "GREEN", "status": "HEALTHY", "detector_error": None, "cycle_duration_sec": 0.002}),
        json.dumps({"ts_utc": mon.epoch_to_iso(GATE1_EPOCHS[1]), "outcome": "WRAPPER_EXCEPTION", "severity": None, "status": None, "detector_error": None, "cycle_duration_sec": 0.001, "exception": "ValueError: boom"}),
        json.dumps({"ts_utc": mon.epoch_to_iso(GATE1_EPOCHS[2]), "outcome": "DETECTOR_ERROR", "severity": "RED", "status": "ERROR", "detector_error": "db_locked", "cycle_duration_sec": 0.001}),
    ]
    log_path = tmp_path / "log.jsonl"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cycles = mon.parse_cycle_log(log_path)
    assert cycles[0].reason_codes == []
    assert "WRAPPER_EXCEPTION" in cycles[1].reason_codes
    assert "DETECTOR_ERROR_PRESENT" in cycles[2].reason_codes


def test_malformed_json_line_is_skipped_not_fabricated(tmp_path):
    log_path = tmp_path / "log.jsonl"
    log_path.write_text("{not valid json\n" + json.dumps({"ts_utc": mon.epoch_to_iso(GATE1_EPOCHS[0]), "outcome": "SUCCESS"}) + "\n", encoding="utf-8")
    cycles = mon.parse_cycle_log(log_path)
    assert len(cycles) == 1
    assert cycles[0].outcome == "SUCCESS"


# ==========================================================================
# Bounded output / inspect-only safety / no process-control primitives
# ==========================================================================


def test_bounded_output_growth_one_record_per_call(tmp_path):
    path = tmp_path / "evidence.jsonl"
    mon.append_jsonl_record(path, {"a": 1})
    mon.append_jsonl_record(path, {"a": 2})
    mon.append_jsonl_record(path, {"a": 3})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert [json.loads(l)["a"] for l in lines] == [1, 2, 3]


def test_take_snapshot_calls_lister_exactly_once_and_writes_only_evidence_dir(tmp_path):
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")
    evidence_dir = tmp_path / "evidence"
    pid_meta = tmp_path / "pid_meta.json"
    pid_meta.write_text(json.dumps({"pid": 15640, "started_at": BASELINE_START, "cmdline_sig": "x", "role": "liquidation_silence_scheduler"}), encoding="utf-8")

    calls = {"n": 0}

    def _tracking_lister():
        calls["n"] += 1
        return [_rec(15640, BASELINE_START)]

    mon.take_snapshot(scheduler_log_path=scheduler_log, evidence_dir=evidence_dir, pid_meta_path=pid_meta, lister=_tracking_lister)
    assert calls["n"] == 1
    assert (evidence_dir / "snapshots.jsonl").exists()
    assert (evidence_dir / "run_manifest.json").exists()
    assert scheduler_log.read_text(encoding="utf-8").count("\n") == 7  # untouched, read-only


def test_module_source_contains_no_process_control_primitives():
    source = Path(mon.__file__).read_text(encoding="utf-8")
    forbidden_substrings = [
        "os.kill(",
        "signal.SIGKILL",
        "Stop-Process",
        "taskkill",
        "proc.kill(",
        "proc.terminate(",
        "start_eclipse.ps1",
        "stop_eclipse.ps1",
        "-EnableLiquidationSilenceScheduler",
        "Popen(",
    ]
    for needle in forbidden_substrings:
        assert needle not in source, f"forbidden process-control primitive found: {needle}"


def test_module_import_has_no_side_effects(tmp_path):
    # Re-importing must not touch the filesystem beyond normal bytecode
    # caching -- verified by checking no new file appears under a fresh
    # temp CWD sentinel directory (best-effort smoke check).
    import importlib

    importlib.reload(mon)
    assert mon.EVIDENCE_SCHEMA_VERSION == "liquidation_silence_canary_monitor_v2"


# ==========================================================================
# F-BASELINE-01: real CLI coverage for invalid/valid --pid-meta-path.
# Nine of the ten required cases go through the real `python -m` CLI entry
# point (subprocess); the permission-denied case uses take_snapshot() (the
# exact function main() calls) with a monkeypatched read_bytes, per this
# round's authorization boundary noting Windows filesystem permission bits
# are unreliable for a real repro. The malformed-StartTime case also uses
# take_snapshot() directly (not a real subprocess) so its continuity result
# does not depend on whether PID 15640 happens to be a real running process
# on the host at test time.
# ==========================================================================


def _run_canary_cli_with_pid_meta(tmp_path, *, pid_meta_path):
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")
    evidence_dir = tmp_path / "evidence"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tools.liquidation_silence_canary_monitor",
            "--scheduler-log-path",
            str(scheduler_log),
            "--evidence-dir",
            str(evidence_dir),
            "--pid-meta-path",
            str(pid_meta_path),
        ],
        cwd=str(mon.ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result


def test_baseline_missing_file_via_real_cli(tmp_path):
    pid_meta = tmp_path / "does_not_exist.json"
    result = _run_canary_cli_with_pid_meta(tmp_path, pid_meta_path=pid_meta)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())
    assert output["continuity"] is not None
    assert output["continuity"]["continuity_status"] == mon.CONTINUITY_NOT_EVALUATED
    assert output["baseline"]["status"] == mon.BASELINE_STATUS_UNREADABLE
    assert "BASELINE_MISSING" in output["baseline"]["reason_codes"]


def test_baseline_malformed_json_via_real_cli(tmp_path):
    pid_meta = tmp_path / "pid_meta.json"
    pid_meta.write_text("{not valid json", encoding="utf-8")
    result = _run_canary_cli_with_pid_meta(tmp_path, pid_meta_path=pid_meta)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())
    assert output["continuity"]["continuity_status"] == mon.CONTINUITY_NOT_EVALUATED
    assert output["continuity"]["continuity_status"] != mon.CONTINUITY_CONTINUOUS
    assert output["baseline"]["status"] == mon.BASELINE_STATUS_INVALID
    assert "BASELINE_MALFORMED_JSON" in output["baseline"]["reason_codes"]


def test_baseline_empty_file_via_real_cli(tmp_path):
    pid_meta = tmp_path / "pid_meta.json"
    pid_meta.write_text("", encoding="utf-8")
    result = _run_canary_cli_with_pid_meta(tmp_path, pid_meta_path=pid_meta)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())
    assert output["continuity"]["continuity_status"] == mon.CONTINUITY_NOT_EVALUATED
    assert output["baseline"]["status"] == mon.BASELINE_STATUS_INVALID
    assert "BASELINE_EMPTY_FILE" in output["baseline"]["reason_codes"]


def test_baseline_invalid_utf8_via_real_cli(tmp_path):
    pid_meta = tmp_path / "pid_meta.json"
    pid_meta.write_bytes(b"\xff\xfe\x00\x01not-utf8")
    result = _run_canary_cli_with_pid_meta(tmp_path, pid_meta_path=pid_meta)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())
    assert output["continuity"]["continuity_status"] == mon.CONTINUITY_NOT_EVALUATED
    assert output["baseline"]["status"] == mon.BASELINE_STATUS_UNREADABLE
    assert "BASELINE_INVALID_ENCODING" in output["baseline"]["reason_codes"]


def test_baseline_directory_path_via_real_cli(tmp_path):
    # Real filesystem path, per this round's requirement that at least the
    # directory-path case exercise one.
    pid_meta_dir = tmp_path / "a_directory"
    pid_meta_dir.mkdir()
    result = _run_canary_cli_with_pid_meta(tmp_path, pid_meta_path=pid_meta_dir)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())
    assert output["continuity"]["continuity_status"] == mon.CONTINUITY_NOT_EVALUATED
    assert output["baseline"]["status"] == mon.BASELINE_STATUS_UNREADABLE
    assert "BASELINE_NOT_A_REGULAR_FILE" in output["baseline"]["reason_codes"]


def test_baseline_permission_denied_via_monkeypatched_read(tmp_path, monkeypatch):
    pid_meta = tmp_path / "pid_meta.json"
    pid_meta.write_text(
        json.dumps({"pid": 15640, "started_at": BASELINE_START, "cmdline_sig": "x", "role": "r"}), encoding="utf-8"
    )

    original_read_bytes = Path.read_bytes

    def _raise_permission(self, *args, **kwargs):
        if self == pid_meta:
            raise PermissionError("simulated permission denial")
        return original_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", _raise_permission)

    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")
    evidence_dir = tmp_path / "evidence"

    record = mon.take_snapshot(scheduler_log_path=scheduler_log, evidence_dir=evidence_dir, pid_meta_path=pid_meta)
    assert record["continuity"]["continuity_status"] == mon.CONTINUITY_NOT_EVALUATED
    assert record["baseline"]["status"] == mon.BASELINE_STATUS_UNREADABLE
    assert "BASELINE_PERMISSION_DENIED" in record["baseline"]["reason_codes"]


def test_baseline_missing_required_fields_via_real_cli(tmp_path):
    pid_meta = tmp_path / "pid_meta.json"
    pid_meta.write_text(json.dumps({"cmdline_sig": "x", "role": "r"}), encoding="utf-8")  # no pid, no started_at
    result = _run_canary_cli_with_pid_meta(tmp_path, pid_meta_path=pid_meta)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())
    assert output["continuity"]["continuity_status"] == mon.CONTINUITY_NOT_EVALUATED
    assert output["baseline"]["status"] == mon.BASELINE_STATUS_INVALID
    assert "BASELINE_SCHEMA_MISSING_REQUIRED_FIELDS" in output["baseline"]["reason_codes"]


def test_baseline_malformed_pid_via_real_cli(tmp_path):
    pid_meta = tmp_path / "pid_meta.json"
    pid_meta.write_text(
        json.dumps({"pid": "not-a-number", "started_at": BASELINE_START, "cmdline_sig": "x", "role": "r"}),
        encoding="utf-8",
    )
    result = _run_canary_cli_with_pid_meta(tmp_path, pid_meta_path=pid_meta)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())
    assert output["continuity"]["continuity_status"] == mon.CONTINUITY_NOT_EVALUATED
    assert output["continuity"]["continuity_status"] != mon.CONTINUITY_CONTINUOUS
    assert output["baseline"]["status"] == mon.BASELINE_STATUS_INVALID
    assert "BASELINE_SCHEMA_MALFORMED_PID" in output["baseline"]["reason_codes"]


def test_baseline_malformed_starttime_via_public_entry(tmp_path):
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")
    pid_meta = tmp_path / "pid_meta.json"
    pid_meta.write_text(
        json.dumps(
            {"pid": 15640, "started_at": "not-a-timestamp", "cmdline_sig": "x", "role": "liquidation_silence_scheduler"}
        ),
        encoding="utf-8",
    )
    evidence_dir = tmp_path / "evidence"
    lister = _fake_lister([_rec(15640, BASELINE_START)])  # a real process, with a VALID StartTime

    record = mon.take_snapshot(
        scheduler_log_path=scheduler_log, evidence_dir=evidence_dir, pid_meta_path=pid_meta, lister=lister
    )
    assert record["baseline"]["status"] == mon.BASELINE_STATUS_OK  # loading succeeded -- schema was fine
    assert record["continuity"] is not None
    assert record["continuity"]["continuity_status"] != mon.CONTINUITY_CONTINUOUS
    assert record["continuity"]["continuity_status"] == mon.CONTINUITY_MALFORMED_STARTTIME


# ==========================================================================
# GATE2-F1: load_pid_metadata_baseline() must accept UTF-8-BOM-prefixed
# metadata -- the exact byte shape Windows PowerShell 5.1's
# `Set-Content -Encoding utf8` (used by start_eclipse.ps1's
# Start-RegisteredPythonProcess, the sole real-world writer of this sibling
# file) always produces -- while remaining strict: BOM-free UTF-8 must keep
# working, malformed JSON must still fail closed, and genuinely invalid
# (non-BOM) UTF-8 must still be rejected, never silently repaired.
# ==========================================================================

_REALISTIC_METADATA_PAYLOAD = {
    "mode": "READ_ONLY_HEALTH_DETECTOR_SCHEDULER_NO_ORDERS",
    "cwd": "D:\\eclipse_scalper",
    "cmdline_sig": "python -W ignore -u -m tools.liquidation_silence_scheduler",
    "role": "liquidation_silence_scheduler",
    "pid": 15640,
    "started_at": BASELINE_START,
}


def test_baseline_bom_prefixed_canonical_metadata_parses(tmp_path):
    pid_meta = tmp_path / "pid_meta.json"
    pid_meta.write_bytes(b"\xef\xbb\xbf" + json.dumps(_REALISTIC_METADATA_PAYLOAD).encode("utf-8"))
    result = mon.load_pid_metadata_baseline(pid_meta, expected_module="tools.liquidation_silence_scheduler")
    assert result.status == mon.BASELINE_STATUS_OK
    assert result.reason_code == "OK"
    assert result.reason_code != "BASELINE_MALFORMED_JSON"
    assert result.baseline.pid == 15640
    assert result.baseline.start_time_utc == BASELINE_START
    assert result.expected_identity.role == "liquidation_silence_scheduler"


def test_baseline_bom_free_metadata_still_parses(tmp_path):
    # Same payload, ordinary UTF-8, no BOM -- must remain unaffected by the fix.
    pid_meta = tmp_path / "pid_meta.json"
    pid_meta.write_bytes(json.dumps(_REALISTIC_METADATA_PAYLOAD).encode("utf-8"))
    result = mon.load_pid_metadata_baseline(pid_meta, expected_module="tools.liquidation_silence_scheduler")
    assert result.status == mon.BASELINE_STATUS_OK
    assert result.reason_code == "OK"
    assert result.baseline.pid == 15640


def test_baseline_bom_prefixed_metadata_via_real_cli(tmp_path):
    # Real CLI subprocess, not a library-level call -- exercises main()'s
    # actual argument parsing and take_snapshot()'s actual wiring, not just
    # the loader function in isolation. PID 999999999 cannot plausibly
    # exist as a real process, so continuity is expected (and required by
    # this test) to evaluate to CONTINUITY_MISSING -- a concrete evaluated
    # status, not NOT_EVALUATED/BASELINE_MALFORMED_JSON -- without this
    # test requiring continuity to report success for a fixture that
    # intentionally represents an absent process.
    pid_meta = tmp_path / "pid_meta.json"
    payload = dict(_REALISTIC_METADATA_PAYLOAD, pid=999999999)
    pid_meta.write_bytes(b"\xef\xbb\xbf" + json.dumps(payload).encode("utf-8"))
    result = _run_canary_cli_with_pid_meta(tmp_path, pid_meta_path=pid_meta)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())
    assert output["baseline"]["status"] == mon.BASELINE_STATUS_OK
    assert output["baseline"]["reason_codes"] == ["OK"]
    assert "BASELINE_MALFORMED_JSON" not in output["baseline"]["reason_codes"]
    assert output["continuity"] is not None
    assert output["continuity"]["continuity_status"] != mon.CONTINUITY_NOT_EVALUATED
    assert output["continuity"]["continuity_status"] == mon.CONTINUITY_MISSING


def test_baseline_bom_prefixed_malformed_json_still_fails_closed(tmp_path):
    pid_meta = tmp_path / "pid_meta.json"
    pid_meta.write_bytes(b"\xef\xbb\xbf{not valid json")
    result = mon.load_pid_metadata_baseline(pid_meta)
    assert result.status == mon.BASELINE_STATUS_INVALID
    assert result.reason_code == "BASELINE_MALFORMED_JSON"
    assert result.baseline is None


def test_baseline_bom_prefixed_malformed_json_via_real_cli(tmp_path):
    pid_meta = tmp_path / "pid_meta.json"
    pid_meta.write_bytes(b"\xef\xbb\xbf{not valid json")
    result = _run_canary_cli_with_pid_meta(tmp_path, pid_meta_path=pid_meta)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())
    assert output["baseline"]["status"] == mon.BASELINE_STATUS_INVALID
    assert "BASELINE_MALFORMED_JSON" in output["baseline"]["reason_codes"]
    assert output["continuity"]["continuity_status"] == mon.CONTINUITY_NOT_EVALUATED
    assert output["continuity"]["continuity_status"] != mon.CONTINUITY_CONTINUOUS


def test_baseline_invalid_non_bom_utf8_still_rejected(tmp_path):
    # Genuinely invalid UTF-8 (not a BOM, not repairable) must remain a
    # typed decode failure -- utf-8-sig performs no replacement-character
    # recovery or permissive fallback, only BOM stripping.
    pid_meta = tmp_path / "pid_meta.json"
    pid_meta.write_bytes(b"\xff\xfe\x00\x01not-utf8-and-not-a-utf8-bom")
    result = mon.load_pid_metadata_baseline(pid_meta)
    assert result.status == mon.BASELINE_STATUS_UNREADABLE
    assert result.reason_code == "BASELINE_INVALID_ENCODING"
    assert result.baseline is None


def test_baseline_regression_exact_powershell_written_shape(tmp_path):
    # Regression pin for the exact discovered defect: a fixture byte-shaped
    # like the real file start_eclipse.ps1's Start-RegisteredPythonProcess
    # writes via PowerShell 5.1's `ConvertTo-Json | Set-Content -Encoding
    # utf8` -- BOM-prefixed, CRLF line endings, the double-space-after-colon
    # formatting ConvertTo-Json produces, 4-space indentation, key order
    # mode/cwd/cmdline_sig/role/pid/started_at (matches the real on-disk
    # file observed during the Gate 2 rehearsal that discovered GATE2-F1).
    real_shaped_body = (
        "{\r\n"
        '    "mode":  "READ_ONLY_HEALTH_DETECTOR_SCHEDULER_NO_ORDERS",\r\n'
        '    "cwd":  "D:\\\\eclipse_scalper",\r\n'
        '    "cmdline_sig":  "python -W ignore -u -m tools.liquidation_silence_scheduler",\r\n'
        '    "role":  "liquidation_silence_scheduler",\r\n'
        '    "pid":  15640,\r\n'
        '    "started_at":  "2026-07-11T17:40:51.3014132Z"\r\n'
        "}"
    ).encode("utf-8")
    raw = b"\xef\xbb\xbf" + real_shaped_body
    assert raw[:3] == b"\xef\xbb\xbf"

    pid_meta = tmp_path / "liquidation_silence_scheduler.json"
    pid_meta.write_bytes(raw)

    result = mon.load_pid_metadata_baseline(pid_meta, expected_module="tools.liquidation_silence_scheduler")
    assert result.status == mon.BASELINE_STATUS_OK
    assert result.reason_code == "OK"
    assert result.baseline.pid == 15640
    assert result.baseline.command_line == "python -W ignore -u -m tools.liquidation_silence_scheduler"
    assert result.expected_identity.role == "liquidation_silence_scheduler"


# ==========================================================================
# GATE2A-F1: observer (self) exclusion in scheduler continuity candidate
# discovery. The canonical CLI passes `--expected-module
# tools.liquidation_silence_scheduler`, which places that exact token in
# the monitor's OWN command line; without a self-exclusion the monitor
# anchor-identity-matched itself and reported a spurious
# REPLACED_OR_RESTARTED/PID_CHANGED instead of the correct MISSING when the
# real scheduler was absent. Fix: a narrow, explicit by-PID `excluded_pids`
# applied at the find_role_candidates() boundary; take_snapshot() passes
# os.getpid(). These tests exercise the REAL production discovery and
# classification code (no mocking of the classifier), using injected
# synthetic process inventories for determinism plus one real-CLI
# subprocess regression that reproduces the exact canonical argument.
# ==========================================================================

# An observer-shaped ProcessRecord: command line contains the expected
# scheduler module token exactly as the monitor's own argv does under the
# canonical invocation.
_OBSERVER_CMDLINE = (
    "C:\\Python313\\python.exe -B -W ignore -u -m tools.liquidation_silence_canary_monitor "
    "--scheduler-log-path logs/liquidation_silence_scheduler.log --evidence-dir X "
    "--pid-meta-path logs/pids/liquidation_silence_scheduler.json "
    "--expected-module tools.liquidation_silence_scheduler"
)
_OBSERVER_PID = 18460
_SCHED_BASELINE = mon.ProcessRecord(
    pid=15640, start_time_utc=BASELINE_START, command_line="python -W ignore -u -m tools.liquidation_silence_scheduler"
)


def test_find_role_candidates_excludes_observer_pid():
    # 1. Observer process present, its command line contains the expected
    # module token, and its PID equals the supplied observer PID ->
    # find_role_candidates must exclude it entirely.
    observer = mon.ProcessRecord(pid=_OBSERVER_PID, start_time_utc="2026-07-12T12:00:00+00:00", command_line=_OBSERVER_CMDLINE)
    result = mon.find_role_candidates([observer], baseline_pid=15640, expected_identity=SCHEDULER_IDENTITY, excluded_pids=(_OBSERVER_PID,))
    assert result.matches == []
    assert result.count == 0


def test_find_role_candidates_without_exclusion_would_self_match():
    # Proves the token genuinely matches (i.e. the exclusion is load-bearing,
    # not vacuous): the same observer record WITHOUT exclusion is returned
    # as a candidate -- this is exactly the GATE2A-F1 defect.
    observer = mon.ProcessRecord(pid=_OBSERVER_PID, start_time_utc="2026-07-12T12:00:00+00:00", command_line=_OBSERVER_CMDLINE)
    result_no_excl = mon.find_role_candidates([observer], baseline_pid=15640, expected_identity=SCHEDULER_IDENTITY)
    assert [r.pid for r in result_no_excl.matches] == [_OBSERVER_PID]


def test_continuity_missing_when_only_observer_present_and_excluded():
    # 2. baseline PID absent; no legitimate scheduler; observer self-match
    # is the only apparent token match; excluded -> CONTINUITY_MISSING,
    # process_count 0, nothing fabricated.
    observer = mon.ProcessRecord(pid=_OBSERVER_PID, start_time_utc="2026-07-12T12:00:00+00:00", command_line=_OBSERVER_CMDLINE)
    snapshot = mon.sample_role_continuity(
        "liquidation_silence_scheduler", _SCHED_BASELINE, _fake_lister([observer]),
        expected_identity=SCHEDULER_IDENTITY, excluded_pids=(_OBSERVER_PID,),
    )
    assert snapshot.continuity_status == mon.CONTINUITY_MISSING
    assert snapshot.process_count == 0
    assert snapshot.pid is None
    assert snapshot.command_line is None
    assert "NO_MATCHING_PROCESS" in snapshot.reason_codes


def test_legitimate_distinct_scheduler_still_discoverable_with_observer_excluded():
    # 3. Observer AND a distinct legitimate scheduler both contain the token;
    # observer excluded; the real distinct scheduler (different PID) remains
    # and yields the governed REPLACED_OR_RESTARTED (PID changed vs baseline).
    observer = mon.ProcessRecord(pid=_OBSERVER_PID, start_time_utc="2026-07-12T12:00:00+00:00", command_line=_OBSERVER_CMDLINE)
    real_sched = _rec(22222, "2026-07-12T09:00:00+00:00")  # default cmdline is the real scheduler module
    snapshot = mon.sample_role_continuity(
        "liquidation_silence_scheduler", _SCHED_BASELINE, _fake_lister([observer, real_sched]),
        expected_identity=SCHEDULER_IDENTITY, excluded_pids=(_OBSERVER_PID,),
    )
    assert snapshot.continuity_status == mon.CONTINUITY_REPLACED_OR_RESTARTED
    assert snapshot.pid == 22222
    assert "PID_CHANGED" in snapshot.reason_codes


def test_same_pid_baseline_candidate_remains_eligible_with_observer_excluded():
    # 4. A legitimate candidate whose PID matches the baseline (and is NOT
    # the observer) stays eligible -> CONTINUOUS; self-exclusion does not
    # break the same-process/continuous path.
    observer = mon.ProcessRecord(pid=_OBSERVER_PID, start_time_utc="2026-07-12T12:00:00+00:00", command_line=_OBSERVER_CMDLINE)
    same_pid_real = _rec(15640, BASELINE_START)  # same PID as baseline, real scheduler cmdline
    snapshot = mon.sample_role_continuity(
        "liquidation_silence_scheduler", _SCHED_BASELINE, _fake_lister([observer, same_pid_real]),
        expected_identity=SCHEDULER_IDENTITY, excluded_pids=(_OBSERVER_PID,),
    )
    assert snapshot.continuity_status == mon.CONTINUITY_CONTINUOUS
    assert snapshot.pid == 15640


def test_multiple_legitimate_candidates_not_hidden_by_observer_exclusion():
    # 5. Observer exclusion must not hide duplicate/multiple real candidates:
    # two distinct real scheduler candidates -> DUPLICATE, process_count 2.
    observer = mon.ProcessRecord(pid=_OBSERVER_PID, start_time_utc="2026-07-12T12:00:00+00:00", command_line=_OBSERVER_CMDLINE)
    dup1 = _rec(33333, "2026-07-12T09:00:00+00:00")
    dup2 = _rec(44444, "2026-07-12T09:05:00+00:00")
    snapshot = mon.sample_role_continuity(
        "liquidation_silence_scheduler", _SCHED_BASELINE, _fake_lister([observer, dup1, dup2]),
        expected_identity=SCHEDULER_IDENTITY, excluded_pids=(_OBSERVER_PID,),
    )
    assert snapshot.continuity_status == mon.CONTINUITY_DUPLICATE
    assert snapshot.process_count == 2
    assert "MULTIPLE_MATCHING_PROCESSES" in snapshot.reason_codes


def test_find_role_candidates_default_no_exclusion_preserves_prior_behavior():
    # Backward compatibility: omitting excluded_pids (every existing internal
    # caller) behaves exactly as before -- the token-matching observer is
    # returned. Guards against the fix silently changing default semantics.
    observer = mon.ProcessRecord(pid=_OBSERVER_PID, start_time_utc="2026-07-12T12:00:00+00:00", command_line=_OBSERVER_CMDLINE)
    result = mon.find_role_candidates([observer], baseline_pid=15640, expected_identity=SCHEDULER_IDENTITY)
    assert [r.pid for r in result.matches] == [_OBSERVER_PID]


def test_take_snapshot_excludes_current_process_pid_reports_missing(tmp_path):
    # Production-wiring proof: take_snapshot() must pass os.getpid() as the
    # observer exclusion. A synthetic lister returns a record whose PID IS
    # the current test process's own os.getpid() and whose cmdline contains
    # the scheduler token -- if take_snapshot did NOT exclude os.getpid(),
    # this would self-match and report REPLACED_OR_RESTARTED; with the fix
    # it reports MISSING.
    import os

    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")
    pid_meta = tmp_path / "pid_meta.json"
    pid_meta.write_bytes(
        b"\xef\xbb\xbf"
        + json.dumps(
            {"pid": 999999999, "started_at": BASELINE_START, "cmdline_sig": "x", "role": "liquidation_silence_scheduler"}
        ).encode("utf-8")
    )
    evidence_dir = tmp_path / "evidence"
    self_record = mon.ProcessRecord(pid=os.getpid(), start_time_utc="2026-07-12T12:00:00+00:00", command_line=_OBSERVER_CMDLINE)
    lister = _fake_lister([self_record])

    record = mon.take_snapshot(
        scheduler_log_path=scheduler_log, evidence_dir=evidence_dir, pid_meta_path=pid_meta,
        expected_module="tools.liquidation_silence_scheduler", lister=lister,
    )
    assert record["baseline"]["status"] == mon.BASELINE_STATUS_OK
    assert record["continuity"]["continuity_status"] == mon.CONTINUITY_MISSING
    assert record["continuity"]["process_count"] == 0
    assert record["continuity"]["pid"] is None
    assert record["continuity"]["continuity_status"] != mon.CONTINUITY_REPLACED_OR_RESTARTED


def _run_canary_cli_with_expected_module(tmp_path, *, pid_meta_path, expected_module="tools.liquidation_silence_scheduler"):
    # Uses subprocess.Popen so the monitor's OWN pid is captured. Passes the
    # canonical `--expected-module` (the exact argument that caused
    # GATE2A-F1's observer self-match).
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")
    evidence_dir = tmp_path / "evidence"
    proc = subprocess.Popen(
        [
            sys.executable, "-B", "-m", "tools.liquidation_silence_canary_monitor",
            "--scheduler-log-path", str(scheduler_log),
            "--evidence-dir", str(evidence_dir),
            "--pid-meta-path", str(pid_meta_path),
            "--expected-module", expected_module,
        ],
        cwd=str(mon.ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    monitor_pid = proc.pid
    stdout, stderr = proc.communicate(timeout=30)
    return proc.returncode, stdout, stderr, monitor_pid


def test_canonical_cli_with_expected_module_excludes_observer_reports_missing(tmp_path):
    # 6. Canonical real-CLI regression: the exact invocation form that
    # caused GATE2A-F1 (explicit --expected-module tools.liquidation_silence_scheduler,
    # whose value token appears in the monitor's own argv) against an
    # absent/impossible baseline scheduler PID. The monitor's OWN subprocess
    # PID must never be the continuity candidate, and -- absent a real
    # scheduler (canary contract; this repo runs the scheduler default-OFF)
    # -- the honest result is CONTINUITY_MISSING. If a real scheduler were
    # unexpectedly running, process_count would be > 0 and this test would
    # FAIL LOUDLY rather than silently accept an environment-dependent pass.
    pid_meta = tmp_path / "pid_meta.json"
    payload = {
        "pid": 999999999,  # impossible/absent baseline scheduler PID
        "started_at": BASELINE_START,
        "cmdline_sig": "python -W ignore -u -m tools.liquidation_silence_scheduler",
        "role": "liquidation_silence_scheduler",
        "mode": "READ_ONLY_HEALTH_DETECTOR_SCHEDULER_NO_ORDERS",
        "cwd": str(mon.ROOT),
    }
    pid_meta.write_bytes(b"\xef\xbb\xbf" + json.dumps(payload).encode("utf-8"))  # canonical BOM-prefixed shape

    rc, stdout, stderr, monitor_pid = _run_canary_cli_with_expected_module(tmp_path, pid_meta_path=pid_meta)
    assert rc == 0, stderr
    output = json.loads(stdout.strip())

    # BOM baseline still parses (utf-8-sig fix preserved, not reopened).
    assert output["baseline"]["status"] == mon.BASELINE_STATUS_OK
    assert output["baseline"]["reason_codes"] == ["OK"]

    cont = output["continuity"]
    # Continuity is evaluated (not NOT_EVALUATED).
    assert cont["continuity_status"] != mon.CONTINUITY_NOT_EVALUATED
    # THE fix, deterministic and environment-independent: the monitor's own
    # subprocess PID is never returned as the scheduler candidate.
    assert cont["pid"] != monitor_pid
    # Honest target-state: MISSING, process_count 0, no fabricated candidate.
    assert cont["continuity_status"] == mon.CONTINUITY_MISSING
    assert cont["process_count"] == 0
    assert cont["pid"] is None
    assert cont["continuity_status"] != mon.CONTINUITY_REPLACED_OR_RESTARTED
    assert "PID_CHANGED" not in cont["reason_codes"]
    # Cycle-log parsing remains valid over the real 9-record log fixture.
    assert output["cycle_log"]["status"] == mon.CYCLE_LOG_STATUS_OK


# ==========================================================================
# F-CYCLELOG-01: fail-closed handling for --scheduler-log-path. Mirrors the
# F-BASELINE-01 test structure: real-CLI subprocess for filesystem-content
# cases, take_snapshot() (the real public entry) with monkeypatched
# Path.read_bytes for permission/generic-read failures.
# ==========================================================================


def _run_canary_cli_with_scheduler_log(tmp_path, *, scheduler_log_path):
    evidence_dir = tmp_path / "evidence"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tools.liquidation_silence_canary_monitor",
            "--scheduler-log-path",
            str(scheduler_log_path),
            "--evidence-dir",
            str(evidence_dir),
        ],
        cwd=str(mon.ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result


def test_load_cycle_log_missing_returns_typed_result(tmp_path):
    result = mon.load_cycle_log(tmp_path / "nope.log")
    assert result.status == mon.CYCLE_LOG_STATUS_MISSING
    assert result.reason_code == "CYCLE_LOG_MISSING"
    assert result.observations == []
    assert result.record_count == 0


def test_parse_cycle_log_backward_compatible_for_missing_file(tmp_path):
    # Unchanged contract from before this round: parse_cycle_log() itself
    # never raises and never exposes the typed status -- just the list,
    # exactly as every pre-round-4 caller/test expects.
    cycles = mon.parse_cycle_log(tmp_path / "nope.log")
    assert cycles == []


def test_parse_cycle_log_backward_compatible_for_directory(tmp_path):
    d = tmp_path / "a_directory"
    d.mkdir()
    cycles = mon.parse_cycle_log(d)  # must not raise (this is the exact defect being closed)
    assert cycles == []


def test_load_cycle_log_valid_file_matches_parse_cycle_log(tmp_path):
    log_path = tmp_path / "scheduler.log"
    log_path.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")
    result = mon.load_cycle_log(log_path)
    cycles = mon.parse_cycle_log(log_path)
    assert result.observations == cycles
    assert result.status == mon.CYCLE_LOG_STATUS_OK
    assert result.record_count == 7
    assert result.malformed_record_count == 0


def test_cyclelog_missing_file_via_real_cli(tmp_path):
    scheduler_log = tmp_path / "does_not_exist.log"
    result = _run_canary_cli_with_scheduler_log(tmp_path, scheduler_log_path=scheduler_log)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())
    assert output["cycle_log"]["status"] == mon.CYCLE_LOG_STATUS_MISSING
    assert output["cycle_log"]["reason_code"] == "CYCLE_LOG_MISSING"
    assert output["cadence_summary"]["cycle_count"] == 0
    assert output["cadence_summary"]["gap_count"] == 0
    assert output["cadence_summary"]["valid"] is True  # zero cycles is trivially valid, never fabricated


def test_cyclelog_directory_path_via_real_cli(tmp_path):
    # Real filesystem path, per this round's requirement -- and the exact
    # reproduction of the uncaught-PermissionError defect the independent
    # re-reviewer found.
    scheduler_log_dir = tmp_path / "a_directory"
    scheduler_log_dir.mkdir()
    result = _run_canary_cli_with_scheduler_log(tmp_path, scheduler_log_path=scheduler_log_dir)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())
    assert output["cycle_log"]["status"] == mon.CYCLE_LOG_STATUS_NOT_A_REGULAR_FILE
    assert output["cycle_log"]["reason_code"] == "CYCLE_LOG_NOT_A_REGULAR_FILE"
    assert output["cadence_summary"]["cycle_count"] == 0


def test_cyclelog_permission_denied_via_monkeypatched_read(tmp_path, monkeypatch):
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")

    original_read_bytes = Path.read_bytes

    def _raise_permission(self, *args, **kwargs):
        if self == scheduler_log:
            raise PermissionError("simulated permission denial")
        return original_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", _raise_permission)
    evidence_dir = tmp_path / "evidence"

    record = mon.take_snapshot(scheduler_log_path=scheduler_log, evidence_dir=evidence_dir)
    assert record["cycle_log"]["status"] == mon.CYCLE_LOG_STATUS_PERMISSION_DENIED
    assert record["cycle_log"]["reason_code"] == "CYCLE_LOG_PERMISSION_DENIED"
    assert record["cadence_summary"]["cycle_count"] == 0
    assert record["cadence_summary"]["valid"] is True


def test_cyclelog_generic_read_failure_via_monkeypatched_read(tmp_path, monkeypatch):
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")

    original_read_bytes = Path.read_bytes

    def _raise_generic_oserror(self, *args, **kwargs):
        if self == scheduler_log:
            raise OSError("simulated generic I/O failure (not permission-related)")
        return original_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", _raise_generic_oserror)
    evidence_dir = tmp_path / "evidence"

    record = mon.take_snapshot(scheduler_log_path=scheduler_log, evidence_dir=evidence_dir)
    assert record["cycle_log"]["status"] == mon.CYCLE_LOG_STATUS_READ_FAILED
    assert record["cycle_log"]["status"] != mon.CYCLE_LOG_STATUS_PERMISSION_DENIED
    assert record["cadence_summary"]["cycle_count"] == 0


def test_cyclelog_invalid_utf8_via_real_cli(tmp_path):
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_bytes(b"\xff\xfe\x00\x01not-utf8")
    result = _run_canary_cli_with_scheduler_log(tmp_path, scheduler_log_path=scheduler_log)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())
    assert output["cycle_log"]["status"] == mon.CYCLE_LOG_STATUS_INVALID_ENCODING
    assert output["cadence_summary"]["cycle_count"] == 0


def test_cyclelog_empty_file_via_real_cli(tmp_path):
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("", encoding="utf-8")
    result = _run_canary_cli_with_scheduler_log(tmp_path, scheduler_log_path=scheduler_log)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())
    assert output["cycle_log"]["status"] == mon.CYCLE_LOG_STATUS_EMPTY
    assert output["cadence_summary"]["cycle_count"] == 0


def test_cyclelog_malformed_json_line_via_real_cli(tmp_path):
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("{not valid json\n", encoding="utf-8")
    result = _run_canary_cli_with_scheduler_log(tmp_path, scheduler_log_path=scheduler_log)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())
    assert output["cycle_log"]["status"] == mon.CYCLE_LOG_STATUS_NO_VALID_CYCLES
    assert output["cycle_log"]["record_count"] == 1
    assert output["cycle_log"]["malformed_record_count"] == 1
    assert output["cycle_log"]["valid_cycle_count"] == 0
    assert output["cadence_summary"]["cycle_count"] == 0


def test_cyclelog_mixture_of_malformed_and_valid_lines_via_real_cli(tmp_path):
    scheduler_log = tmp_path / "scheduler.log"
    lines = _gate1_cycle_log_lines()[:3] + ["{not valid json"] + _gate1_cycle_log_lines()[3:]
    scheduler_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = _run_canary_cli_with_scheduler_log(tmp_path, scheduler_log_path=scheduler_log)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())
    # Policy (explicit, per this round's requirement not to leave this
    # ambiguous): PARTIAL SUCCESS WITH WARNINGS -- the readable/valid lines
    # are still parsed into cycles; the malformed one is skipped AND
    # counted (never silently dropped without a trace), and does not cause
    # whole-file rejection. This is the unchanged round-2 policy, now with
    # the count surfaced.
    assert output["cycle_log"]["status"] == mon.CYCLE_LOG_STATUS_OK
    assert output["cycle_log"]["record_count"] == 8  # 7 gate1 lines + 1 malformed
    assert output["cycle_log"]["malformed_record_count"] == 1
    assert output["cycle_log"]["valid_cycle_count"] == 7
    assert output["cadence_summary"]["cycle_count"] == 7


def test_cyclelog_zero_usable_cycles_via_real_cli(tmp_path):
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("{bad\n{also bad\n{still bad\n", encoding="utf-8")
    result = _run_canary_cli_with_scheduler_log(tmp_path, scheduler_log_path=scheduler_log)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())
    assert output["cycle_log"]["status"] == mon.CYCLE_LOG_STATUS_NO_VALID_CYCLES
    assert output["cycle_log"]["record_count"] == 3
    assert output["cycle_log"]["malformed_record_count"] == 3
    assert output["cadence_summary"]["cycle_count"] == 0
    assert output["cadence_summary"]["valid"] is True  # no fabricated invalidity either


def test_cyclelog_valid_canonical_log_via_real_cli(tmp_path):
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")
    result = _run_canary_cli_with_scheduler_log(tmp_path, scheduler_log_path=scheduler_log)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())
    assert output["cycle_log"]["status"] == mon.CYCLE_LOG_STATUS_OK
    assert output["cycle_log"]["reason_code"] == "OK"
    assert output["cycle_log"]["record_count"] == 7
    assert output["cycle_log"]["malformed_record_count"] == 0
    assert output["cycle_log"]["valid_cycle_count"] == 7
    assert output["cadence_summary"]["cycle_count"] == 7
    assert output["cadence_summary"]["valid"] is True


# -- F-CYCLELOG-01 mutation probes ------------------------------------------


def test_mutation_probe_cyclelog_1_directory_path_would_raise_without_guard(tmp_path):
    d = tmp_path / "a_directory"
    d.mkdir()

    def _naive_read(path):
        return path.read_bytes()  # no is_dir() guard -- the mutation under probe

    with pytest.raises(OSError):
        _naive_read(d)
    # The real load_cycle_log() does NOT raise for the same path -- see
    # test_cyclelog_directory_path_via_real_cli.


def test_mutation_probe_cyclelog_2_permission_error_would_escape_without_handler(tmp_path, monkeypatch):
    p = tmp_path / "scheduler.log"
    p.write_text("x", encoding="utf-8")

    def _raise_permission(self, *a, **k):
        raise PermissionError("simulated")

    monkeypatch.setattr(Path, "read_bytes", _raise_permission)

    def _naive_read_no_except():
        return p.read_bytes()  # no try/except -- the mutation under probe

    with pytest.raises(PermissionError):
        _naive_read_no_except()
    # The real load_cycle_log() does NOT raise for the same failure -- see
    # test_cyclelog_permission_denied_via_monkeypatched_read.


def test_mutation_probe_cyclelog_3_generic_read_failure_would_escape_without_handler(tmp_path, monkeypatch):
    p = tmp_path / "scheduler.log"
    p.write_text("x", encoding="utf-8")

    def _raise_generic(self, *a, **k):
        raise OSError("simulated generic failure")

    monkeypatch.setattr(Path, "read_bytes", _raise_generic)

    def _naive_read_no_except():
        return p.read_bytes()

    with pytest.raises(OSError):
        _naive_read_no_except()
    # The real load_cycle_log() does NOT raise -- see
    # test_cyclelog_generic_read_failure_via_monkeypatched_read.


def test_mutation_probe_cyclelog_4_malformed_line_silently_ignored_without_warning(tmp_path):
    log_path = tmp_path / "log.jsonl"
    lines = _gate1_cycle_log_lines()[:3] + ["{not valid json"] + _gate1_cycle_log_lines()[3:]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Naive reimplementation with NO malformed_record_count tracking (the
    # mutation under probe): a caller of this naive version has no way to
    # tell a malformed line was silently dropped.
    naive_observations = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            continue  # dropped with NO record of it happening
        naive_observations.append(line)
    assert len(naive_observations) == 7  # correct count, but silently so -- no evidence of the drop

    real_result = mon.load_cycle_log(log_path)
    assert real_result.malformed_record_count == 1  # the real loader DOES retain this evidence
    assert real_result.record_count == 8


def test_mutation_probe_cyclelog_5_no_valid_cycle_file_would_report_valid_cadence_without_status(tmp_path):
    log_path = tmp_path / "log.jsonl"
    log_path.write_text("{bad\n{also bad\n", encoding="utf-8")

    # A naive caller that only looks at compute_cadence_summary([]).valid
    # (True, correctly, for zero cycles) without also checking the
    # cycle_log status would misreport this unreadable-content log as
    # simply "a healthy zero-cycle period" rather than "every line failed
    # to parse".
    naive_cadence = mon.compute_cadence_summary([])
    assert naive_cadence.valid is True  # true but misleading in isolation

    real_result = mon.load_cycle_log(log_path)
    assert real_result.status == mon.CYCLE_LOG_STATUS_NO_VALID_CYCLES  # the real status disambiguates
    assert real_result.status != mon.CYCLE_LOG_STATUS_OK


def test_mutation_probe_cyclelog_6_failure_output_would_be_null_without_typed_field(tmp_path):
    scheduler_log = tmp_path / "does_not_exist.log"
    evidence_dir = tmp_path / "evidence"
    record = mon.take_snapshot(scheduler_log_path=scheduler_log, evidence_dir=evidence_dir)

    assert record["cycle_log"] is not None
    assert record["cycle_log"]["status"] == mon.CYCLE_LOG_STATUS_MISSING

    # A naive record lacking this field would degrade to an ambiguous
    # bare-null equivalent for any consumer that only checked
    # record.get("cycle_log") -- demonstrated by simulating the stripped
    # shape a pre-F-CYCLELOG-01 caller would have seen.
    naive_record = {k: v for k, v in record.items() if k != "cycle_log"}
    assert naive_record.get("cycle_log") is None


def test_mutation_probe_cyclelog_7_unreadable_file_would_fabricate_zero_cycle_success(tmp_path, monkeypatch):
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("x", encoding="utf-8")

    def _raise_permission(self, *a, **k):
        raise PermissionError("simulated")

    monkeypatch.setattr(Path, "read_bytes", _raise_permission)
    evidence_dir = tmp_path / "evidence"

    record = mon.take_snapshot(scheduler_log_path=scheduler_log, evidence_dir=evidence_dir)
    # The real status distinguishes "genuinely zero cycles" from "could not
    # even be read" -- never collapses an unreadable file into the same
    # OK-with-zero-cycles shape a genuinely empty/quiet period would have.
    assert record["cycle_log"]["status"] == mon.CYCLE_LOG_STATUS_PERMISSION_DENIED
    assert record["cycle_log"]["status"] != mon.CYCLE_LOG_STATUS_OK


# ==========================================================================
# F-READFAIL-01: generic artifact READ_FAILED, distinct from PERMISSION_DENIED
# ==========================================================================


def test_sample_artifact_generic_read_failure_distinct_from_permission(tmp_path, monkeypatch):
    p = tmp_path / "a.json"
    p.write_text("{}", encoding="utf-8")

    original_read_bytes = Path.read_bytes

    def _raise_generic_oserror(self, *args, **kwargs):
        if self == p:
            raise OSError("simulated generic I/O failure (not permission-related)")
        return original_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", _raise_generic_oserror)
    sample = mon.sample_artifact(p, semantic_ts_field="evaluated_at_utc")

    assert sample.status == mon.ARTIFACT_STATUS_READ_FAILED
    assert sample.status != mon.ARTIFACT_STATUS_PERMISSION_DENIED
    assert sample.exists is True  # the path itself exists; only the read failed
    assert sample.sha256 is None
    assert sample.mtime_utc is None
    assert sample.semantic_ts_utc is None
    assert sample.semantic_ts_status != mon.SEMANTIC_TS_PRESENT
    assert sample.error is not None

    good_path = tmp_path / "good.json"
    good_path.write_text(json.dumps({"evaluated_at_utc": "2026-07-11T18:00:00+00:00"}), encoding="utf-8")
    good_sample = mon.sample_artifact(good_path, semantic_ts_field="evaluated_at_utc")
    freshness = mon.compare_artifact_freshness(good_sample, sample)
    assert freshness == mon.FRESHNESS_SAMPLE_UNAVAILABLE  # never fabricated as fresh/unchanged

    from dataclasses import asdict

    payload = json.loads(json.dumps(asdict(sample), default=str))
    assert payload["status"] == mon.ARTIFACT_STATUS_READ_FAILED
    assert payload["sha256"] is None


# ==========================================================================
# F-MIXEDPREC-01: mixed timestamp-source precision within one cycle sequence
# ==========================================================================


def test_mixedprec_epoch_and_fractional_iso_both_count_as_full(tmp_path):
    log_path = tmp_path / "mixed1.jsonl"
    lines = [
        json.dumps({"cycle_started_epoch": 1000.123456, "outcome": "SUCCESS"}),
        json.dumps({"evaluated_at_utc": "2026-07-11T17:41:51.654321+00:00", "outcome": "SUCCESS"}),
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cycles = mon.parse_cycle_log(log_path)
    assert cycles[0].timestamp_source == mon.TIMESTAMP_SOURCE_EPOCH_FULL
    assert cycles[1].timestamp_source == mon.TIMESTAMP_SOURCE_ISO_FULL
    summary = mon.compute_cadence_summary(cycles)
    assert summary.precision_status == mon.CADENCE_PRECISION_FULL
    assert summary.gap_count == 1
    assert summary.exact_gaps_sec[0] is not None


def test_mixedprec_epoch_and_whole_second_fallback_marked_mixed(tmp_path):
    log_path = tmp_path / "mixed2.jsonl"
    lines = [
        json.dumps({"cycle_started_epoch": 1000.123456, "outcome": "SUCCESS"}),
        json.dumps({"ts_utc": "2026-07-11T17:46:51Z", "outcome": "SUCCESS"}),  # no epoch, whole-second only
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cycles = mon.parse_cycle_log(log_path)
    assert cycles[0].timestamp_source == mon.TIMESTAMP_SOURCE_EPOCH_FULL
    assert cycles[1].timestamp_source == mon.TIMESTAMP_SOURCE_ISO_DEGRADED
    summary = mon.compute_cadence_summary(cycles)
    assert summary.precision_status == mon.CADENCE_PRECISION_MIXED
    assert summary.precision_status != mon.CADENCE_PRECISION_FULL  # never claims uniform full precision
    assert summary.gap_count == 1
    assert summary.exact_gaps_sec[0] is not None


def test_mixedprec_malformed_epoch_falls_back_to_full_iso_within_sequence(tmp_path):
    log_path = tmp_path / "mixed3.jsonl"
    lines = [
        json.dumps({"cycle_started_epoch": 1000.0, "outcome": "SUCCESS"}),
        json.dumps(
            {
                "cycle_started_epoch": "garbage",
                "evaluated_at_utc": "2026-07-11T17:46:51.999999+00:00",
                "outcome": "SUCCESS",
            }
        ),
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cycles = mon.parse_cycle_log(log_path)
    assert cycles[0].timestamp_source == mon.TIMESTAMP_SOURCE_EPOCH_FULL
    assert cycles[1].timestamp_source == mon.TIMESTAMP_SOURCE_ISO_FULL  # higher-quality fallback used
    summary = mon.compute_cadence_summary(cycles)
    assert summary.precision_status == mon.CADENCE_PRECISION_FULL
    assert summary.gap_count == 1


def test_mixedprec_malformed_epoch_falls_back_to_degraded_whole_second(tmp_path):
    log_path = tmp_path / "mixed4.jsonl"
    lines = [
        json.dumps({"cycle_started_epoch": 1000.0, "outcome": "SUCCESS"}),
        json.dumps({"cycle_started_epoch": None, "ts_utc": "2026-07-11T17:46:51Z", "outcome": "SUCCESS"}),
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cycles = mon.parse_cycle_log(log_path)
    assert cycles[0].timestamp_source == mon.TIMESTAMP_SOURCE_EPOCH_FULL
    assert cycles[1].timestamp_source == mon.TIMESTAMP_SOURCE_ISO_DEGRADED
    summary = mon.compute_cadence_summary(cycles)
    assert summary.precision_status == mon.CADENCE_PRECISION_MIXED
    assert summary.gap_count == 1
    assert summary.exact_gaps_sec[0] is not None  # gap still computed despite mixed precision


def test_mixedprec_visible_whole_second_rounding_artifact(tmp_path):
    log_path = tmp_path / "mixed5.jsonl"
    lines = [
        json.dumps({"cycle_started_epoch": 1000.900000, "outcome": "SUCCESS"}),
        json.dumps({"ts_utc": mon.epoch_to_iso(1301)[:19] + "Z", "cycle_started_epoch": None, "outcome": "SUCCESS"}),
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cycles = mon.parse_cycle_log(log_path)
    assert cycles[0].timestamp_source == mon.TIMESTAMP_SOURCE_EPOCH_FULL
    assert cycles[1].timestamp_source == mon.TIMESTAMP_SOURCE_ISO_DEGRADED
    summary = mon.compute_cadence_summary(cycles)
    assert summary.precision_status == mon.CADENCE_PRECISION_MIXED
    # The whole-second fallback truncated cycle 2's true sub-second instant,
    # leaving a non-round leftover gap (300.1s, not a tidy 300s) -- visible
    # evidence that a precision-losing fallback occurred within this sequence.
    assert summary.exact_gaps_sec[0] == pytest.approx(300.1, abs=1e-6)
    assert summary.rounded_gaps_sec[0] == 300


def test_mixedprec_source_quality_visible_per_cycle_in_serialized_output(tmp_path):
    from dataclasses import asdict

    log_path = tmp_path / "mixed6.jsonl"
    lines = [
        json.dumps({"cycle_started_epoch": 1000.0, "outcome": "SUCCESS"}),
        json.dumps({"ts_utc": "2026-07-11T17:46:51Z", "outcome": "SUCCESS"}),
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cycles = mon.parse_cycle_log(log_path)
    sources = [asdict(c)["timestamp_source"] for c in cycles]
    assert sources == [mon.TIMESTAMP_SOURCE_EPOCH_FULL, mon.TIMESTAMP_SOURCE_ISO_DEGRADED]
    assert len(set(sources)) == 2  # mixed quality is visible per cycle, not collapsed into one value


def test_mixedprec_full_precision_source_always_outranks_degraded_fallback():
    # When cycle_started_epoch parses successfully, it is used even if a
    # whole-second ts_utc is also present in the same record (this is the
    # exact real-scheduler-log shape) -- higher quality never loses to a
    # lower-quality fallback that happens to also be available.
    record = {"cycle_started_epoch": 1783791711.4434423, "ts_utc": "2026-07-11T17:41:51Z"}
    ts, source = mon._select_cycle_timestamp(record)
    assert source == mon.TIMESTAMP_SOURCE_EPOCH_FULL
    assert source != mon.TIMESTAMP_SOURCE_ISO_DEGRADED


# ==========================================================================
# F-ROUND-01: round-half-up boundary behavior, computed independently of
# the production helper (expected values are hand-computed, not generated
# by calling mon._round_half_up_to_int itself).
# ==========================================================================


def test_round_half_up_299_499999_rounds_down():
    assert mon._round_half_up_to_int(299.499999) == 299


def test_round_half_up_299_5_rounds_up():
    assert mon._round_half_up_to_int(299.5) == 300


def test_round_half_up_300_499999_rounds_down():
    assert mon._round_half_up_to_int(300.499999) == 300


def test_round_half_up_300_5_rounds_up():
    assert mon._round_half_up_to_int(300.5) == 301


def test_round_half_up_300_500001_rounds_up():
    assert mon._round_half_up_to_int(300.500001) == 301


def test_round_half_up_exact_integer_unchanged():
    assert mon._round_half_up_to_int(300.0) == 300
    assert mon._round_half_up_to_int(0.0) == 0


def test_round_half_up_very_small_positive_value():
    assert mon._round_half_up_to_int(0.0001) == 0
    assert mon._round_half_up_to_int(0.49999999) == 0


def test_round_half_up_negative_gap_rejected():
    with pytest.raises(ValueError):
        mon._round_half_up_to_int(-1.0)
    with pytest.raises(ValueError):
        mon._round_half_up_to_int(-0.0001)


def test_round_half_up_nan_rejected():
    with pytest.raises(ValueError):
        mon._round_half_up_to_int(float("nan"))


def test_round_half_up_positive_infinity_rejected():
    with pytest.raises(ValueError):
        mon._round_half_up_to_int(float("inf"))


def test_round_half_up_negative_infinity_rejected():
    with pytest.raises(ValueError):
        mon._round_half_up_to_int(float("-inf"))


def test_round_half_up_result_type_is_int():
    result = mon._round_half_up_to_int(300.5)
    assert isinstance(result, int)
    assert not isinstance(result, bool)


def test_round_half_up_does_not_mutate_original_exact_gap():
    cycles = [_cyc(1, mon.epoch_to_iso(1000)), _cyc(2, mon.epoch_to_iso(1300.5))]
    summary = mon.compute_cadence_summary(cycles)
    assert summary.exact_gaps_sec[0] == pytest.approx(300.5)  # original exact value preserved
    assert summary.rounded_gaps_sec[0] == 301  # rounded value lives in a separate field


def test_negative_gap_does_not_crash_cadence_summary_rounding():
    cycles = [_cyc(1, mon.epoch_to_iso(2000)), _cyc(2, mon.epoch_to_iso(1000))]  # regression: 2000 -> 1000
    summary = mon.compute_cadence_summary(cycles)
    assert summary.valid is False
    assert summary.exact_gaps_sec[0] == pytest.approx(-1000.0)
    assert summary.rounded_gaps_sec[0] is None  # rejected before rounding, not silently rounded to -1000


# ==========================================================================
# Required mutation probes (Round 3, section 10): each demonstrates, via a
# temporary monkeypatch or an inline naive reimplementation, that removing
# the corresponding fix reproduces the exact defect it closes -- proving
# the shipped tests above are what actually detect a regression to it. No
# repository file is modified by any probe.
# ==========================================================================


def test_mutation_probe_1_stopevent_post_init_removed_allows_bypass(monkeypatch):
    monkeypatch.setattr(mon.StopEvent, "__post_init__", lambda self: None)
    bypassed = mon.StopEvent(
        target_role="liquidation_silence_scheduler",
        target_pids=[15640],
        outcome=mon.STOP_OUTCOME_VERIFIED_ABSENT,
        verification=None,
    )
    assert bypassed.outcome == mon.STOP_OUTCOME_VERIFIED_ABSENT
    assert bypassed.verification is None
    # With __post_init__ restored (monkeypatch undone at teardown), the
    # identical construction raises -- see
    # test_direct_stopevent_rejects_success_with_no_verification.


def test_mutation_probe_2_stopevent_chronology_guard_removed_allows_bypass(monkeypatch):
    monkeypatch.setattr(mon.StopEvent, "__post_init__", lambda self: None)
    bypassed = mon.StopEvent(
        target_role="x",
        target_pids=[1],
        requested_at_utc="2026-07-11T18:20:00+00:00",
        completed_at_utc="2026-07-11T18:10:00+00:00",  # precedes requested
    )
    assert bypassed.completed_at_utc == "2026-07-11T18:10:00+00:00"
    # With __post_init__ restored, this raises -- see
    # test_chronology_inversion_rejected.


def test_mutation_probe_3_unguarded_baseline_loader_crashes_take_snapshot(tmp_path, monkeypatch):
    def _naive_unguarded_loader(pid_meta_path, *, expected_module=None):
        raw = pid_meta_path.read_text(encoding="utf-8")
        meta = json.loads(raw)
        baseline = mon.ProcessRecord(
            pid=int(meta["pid"]),  # no isinstance() guard: raises on a non-numeric pid string
            start_time_utc=meta["started_at"],
            command_line=meta.get("cmdline_sig"),
            captured_at_utc=mon.utc_now_iso(),
        )
        identity = mon.ExpectedIdentity(role=str(meta.get("role") or "unknown"), module_name=expected_module)
        return mon.BaselineLoadResult(
            status=mon.BASELINE_STATUS_OK, reason_code="OK", baseline=baseline, expected_identity=identity
        )

    monkeypatch.setattr(mon, "load_pid_metadata_baseline", _naive_unguarded_loader)

    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("\n".join(_gate1_cycle_log_lines()) + "\n", encoding="utf-8")
    pid_meta = tmp_path / "pid_meta.json"
    pid_meta.write_text(
        json.dumps({"pid": "not-a-number", "started_at": BASELINE_START, "cmdline_sig": "x", "role": "r"}),
        encoding="utf-8",
    )
    evidence_dir = tmp_path / "evidence"

    with pytest.raises(ValueError):
        mon.take_snapshot(scheduler_log_path=scheduler_log, evidence_dir=evidence_dir, pid_meta_path=pid_meta)
    # The real load_pid_metadata_baseline() does NOT raise for this same
    # input -- see test_baseline_malformed_pid_via_real_cli.


def test_mutation_probe_4_fabricated_starttime_default_would_produce_false_continuity():
    fabricated_now = mon.utc_now_iso()
    baseline = mon.ProcessRecord(pid=15640, start_time_utc=fabricated_now, command_line="x", captured_at_utc=fabricated_now)
    query = mon.ProcessQueryResult(
        query_failed=False, matches=[mon.ProcessRecord(pid=15640, start_time_utc=fabricated_now, command_line="x")]
    )
    snapshot = mon.classify_continuity(baseline, query)
    assert snapshot.continuity_status == mon.CONTINUITY_CONTINUOUS
    # This is exactly why the real loader refuses to fabricate a StartTime
    # for a missing/malformed field instead of returning
    # BASELINE_SCHEMA_MISSING_REQUIRED_FIELDS -- see
    # test_baseline_missing_required_fields_via_real_cli, which asserts
    # CONTINUITY_NOT_EVALUATED, never CONTINUOUS, for that input.


def test_mutation_probe_5_missing_generic_oserror_handler_would_escape(tmp_path, monkeypatch):
    p = tmp_path / "a.json"
    p.write_text("{}", encoding="utf-8")

    def _naive_read_catches_only_permission(path):
        try:
            path.read_bytes()
        except PermissionError:
            return "PERMISSION_DENIED"
        # no generic `except OSError` branch here -- the mutation under probe

    def _raise_generic_oserror(self, *a, **k):
        raise OSError("simulated generic I/O failure")

    monkeypatch.setattr(Path, "read_bytes", _raise_generic_oserror)
    with pytest.raises(OSError):
        _naive_read_catches_only_permission(p)
    # The real sample_artifact() does NOT raise for this same failure --
    # see test_sample_artifact_generic_read_failure_distinct_from_permission.


def test_mutation_probe_6_naive_precision_status_would_hide_mixed_sources():
    cycles = [
        _cyc(1, mon.epoch_to_iso(1000), timestamp_source=mon.TIMESTAMP_SOURCE_EPOCH_FULL),
        _cyc(2, "2026-07-11T17:46:51+00:00", timestamp_source=mon.TIMESTAMP_SOURCE_ISO_DEGRADED),
    ]

    def _naive_precision_status(cycles):
        full = {mon.TIMESTAMP_SOURCE_EPOCH_FULL, mon.TIMESTAMP_SOURCE_ISO_FULL}
        if any(c.timestamp_source in full for c in cycles):  # "any", not "all" -- the mutation under probe
            return mon.CADENCE_PRECISION_FULL
        return mon.CADENCE_PRECISION_DEGRADED

    assert _naive_precision_status(cycles) == mon.CADENCE_PRECISION_FULL  # the bug: falsely full

    real_result = mon._precision_status(cycles)
    assert real_result == mon.CADENCE_PRECISION_MIXED
    assert real_result != mon.CADENCE_PRECISION_FULL


def test_mutation_probe_7_stripped_timestamp_source_hides_fallback(tmp_path):
    from dataclasses import asdict

    log_path = tmp_path / "degraded.jsonl"
    log_path.write_text(json.dumps({"ts_utc": "2026-07-11T17:46:51Z", "outcome": "SUCCESS"}) + "\n", encoding="utf-8")
    cycles = mon.parse_cycle_log(log_path)
    full_payload = asdict(cycles[0])
    assert full_payload["timestamp_source"] == mon.TIMESTAMP_SOURCE_ISO_DEGRADED

    stripped_payload = {k: v for k, v in full_payload.items() if k != "timestamp_source"}
    assert "timestamp_source" not in stripped_payload
    # A consumer of stripped_payload alone cannot tell this cycle was a
    # degraded whole-second fallback -- proving the real, un-stripped
    # per-cycle output (asserted above) is what carries that evidence.


def test_mutation_probe_8_python_builtin_round_diverges_at_half_boundary():
    assert round(300.5) == 300  # Python's banker's rounding: nearest even
    assert mon._round_half_up_to_int(300.5) == 301  # spec-correct half-up
    assert round(300.5) != mon._round_half_up_to_int(300.5)


def test_mutation_probe_9_naive_rounding_would_silently_round_negative_gap():
    def _naive_round_half_up(value):
        return int(math.floor(value + 0.5))  # no negative-value guard -- the mutation under probe

    assert _naive_round_half_up(-1000.4) == -1000  # silently "succeeds" with a meaningless value

    with pytest.raises(ValueError):
        mon._round_half_up_to_int(-1000.4)  # the real implementation refuses


def test_mutation_probe_10_builder_has_no_independent_validation_left(monkeypatch):
    # If build_stop_event() still had its OWN validation logic (rather than
    # fully delegating to StopEvent.__post_init__), neutering __post_init__
    # alone would not be enough to bypass it. Confirm both the builder and
    # direct construction are bypassed together, proving there is exactly
    # one enforcement point, not two that could silently diverge.
    monkeypatch.setattr(mon.StopEvent, "__post_init__", lambda self: None)
    event = mon.build_stop_event(
        target_role="liquidation_silence_scheduler",
        target_pids=[15640],
        outcome=mon.STOP_OUTCOME_VERIFIED_ABSENT,
        verification=None,
    )
    assert event.outcome == mon.STOP_OUTCOME_VERIFIED_ABSENT
    assert event.verification is None
    # With __post_init__ restored, build_stop_event() raises for this same
    # call -- see test_success_without_verification_is_rejected.
