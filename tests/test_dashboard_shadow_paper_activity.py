"""Tests for dashboard/backend/adapters/shadow_paper_activity.py.

Covers: the eight-state bucket classifier (precedence order), the
gap-detector primitive in isolation, and an end-to-end build() over a
fixture repo tree exercising PROCESS_DEAD / healthy / incident-log paths.
No network, no live process, no write to any real repo path -- everything
is rooted at ``tmp_path``.
"""
from __future__ import annotations

import json

from dashboard.backend.adapters import shadow_paper_activity as spa
from dashboard.backend.sources import DashboardContext
from dashboard.backend.view_models import ParseStatus

NOW = 1_800_000_000.0


# --- classify_bucket_state: precedence order --------------------------------

def test_error_wins_over_everything():
    state = spa.classify_bucket_state(
        process_alive=True, error_present=True,
        has_open_position=True, market_data_stale=True,
    )
    assert state == spa.STATE_ERROR


def test_process_dead_wins_over_data():
    state = spa.classify_bucket_state(process_alive=False, has_open_position=True)
    assert state == spa.STATE_PROCESS_DEAD


def test_missing_artifact_is_evaluation_stale_not_data_stale():
    state = spa.classify_bucket_state(
        process_alive=True, artifact_parse_status=ParseStatus.MISSING,
    )
    assert state == spa.STATE_EVALUATION_STALE


def test_malformed_artifact_is_evaluation_stale():
    state = spa.classify_bucket_state(
        process_alive=True, artifact_parse_status=ParseStatus.MALFORMED,
    )
    assert state == spa.STATE_EVALUATION_STALE


def test_stale_artifact_age_is_evaluation_stale():
    state = spa.classify_bucket_state(
        process_alive=True, artifact_age_sec=1000.0, artifact_threshold_sec=300.0,
    )
    assert state == spa.STATE_EVALUATION_STALE


def test_evaluation_stale_wins_over_market_data_stale():
    state = spa.classify_bucket_state(
        process_alive=True, artifact_age_sec=1000.0, artifact_threshold_sec=300.0,
        market_data_stale=True,
    )
    assert state == spa.STATE_EVALUATION_STALE


def test_market_data_stale_reported_when_evaluation_is_fresh():
    state = spa.classify_bucket_state(
        process_alive=True, artifact_age_sec=10.0, artifact_threshold_sec=300.0,
        market_data_stale=True,
    )
    assert state == spa.STATE_DATA_STALE


def test_open_position_wins_over_last_event():
    state = spa.classify_bucket_state(
        process_alive=True, has_open_position=True, last_event_kind="REJECT",
    )
    assert state == spa.STATE_VIRTUAL_POSITION_OPEN


def test_market_data_stale_wins_over_open_position():
    # Documents the precedence choice explicitly: data staleness is treated
    # as more urgent/actionable than "there happens to be an open virtual
    # position" -- the position isn't hidden (callers still surface
    # open_trades/currently_open regardless of top-line state), only the
    # single-state classification defers to DATA_STALE.
    state = spa.classify_bucket_state(
        process_alive=True, market_data_stale=True, has_open_position=True,
    )
    assert state == spa.STATE_DATA_STALE


def test_recent_close_is_virtual_trade_closed():
    state = spa.classify_bucket_state(
        process_alive=True, last_event_kind="CLOSE", last_event_age_sec=30.0,
        recent_close_window_sec=900.0,
    )
    assert state == spa.STATE_VIRTUAL_TRADE_CLOSED


def test_old_close_falls_through_to_no_eligible_signal():
    state = spa.classify_bucket_state(
        process_alive=True, last_event_kind="CLOSE", last_event_age_sec=10_000.0,
        recent_close_window_sec=900.0,
    )
    assert state == spa.STATE_NO_ELIGIBLE_SIGNAL


def test_reject_is_signal_seen_no_entry():
    state = spa.classify_bucket_state(process_alive=True, last_event_kind="REJECT")
    assert state == spa.STATE_SIGNAL_SEEN_NO_ENTRY


def test_healthy_idle_is_no_eligible_signal():
    state = spa.classify_bucket_state(process_alive=True)
    assert state == spa.STATE_NO_ELIGIBLE_SIGNAL


# --- detect_evidence_gaps ---------------------------------------------------

def test_detect_evidence_gaps_finds_internal_gap():
    ts = [1000.0, 1010.0, 5000.0, 5010.0]
    gaps = spa.detect_evidence_gaps(ts, gap_threshold_sec=100.0)
    assert len(gaps) == 1
    assert gaps[0]["gap_start_epoch"] == 1010.0
    assert gaps[0]["gap_end_epoch"] == 5000.0
    assert gaps[0]["ongoing"] is False


def test_detect_evidence_gaps_reports_trailing_ongoing_gap():
    ts = [1000.0, 1010.0]
    gaps = spa.detect_evidence_gaps(ts, gap_threshold_sec=100.0, now=2000.0)
    assert len(gaps) == 1
    assert gaps[0]["ongoing"] is True
    assert gaps[0]["gap_end_epoch"] == 2000.0


def test_detect_evidence_gaps_empty_when_uniform():
    ts = [1000.0, 1010.0, 1020.0, 1030.0]
    gaps = spa.detect_evidence_gaps(ts, gap_threshold_sec=100.0, now=1035.0)
    assert gaps == []


def test_detect_evidence_gaps_handles_empty_input():
    assert spa.detect_evidence_gaps([], gap_threshold_sec=100.0, now=NOW) == []


# --- read_incident_log -------------------------------------------------------

def test_read_incident_log_missing_file_is_honest_not_silent(tmp_path):
    ctx = DashboardContext(repo_root=tmp_path, now=NOW)
    result = spa.read_incident_log(ctx)
    assert result["incidents"] == []
    assert result["parse_status"] == "missing"
    assert "not evidence" in result["note"].lower() or "not" in result["note"].lower()


def test_read_incident_log_reads_real_incidents(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "dashboard_incident_log.json").write_text(
        json.dumps({"incidents": [{"incident_id": "X"}]}), encoding="utf-8",
    )
    ctx = DashboardContext(repo_root=tmp_path, now=NOW)
    result = spa.read_incident_log(ctx)
    assert result["parse_status"] == "ok"
    assert result["incidents"] == [{"incident_id": "X"}]


# --- build(): end-to-end over a fixture repo tree ---------------------------

def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _make_ctx(tmp_path, *, proc_snapshot):
    return DashboardContext(repo_root=tmp_path, now=NOW, proc_snapshot=proc_snapshot)


def test_build_reports_process_dead_when_no_role_matches(tmp_path):
    ctx = _make_ctx(tmp_path, proc_snapshot=[])  # nothing running at all
    vm = spa.build(ctx)
    assert vm.key == "shadow_paper_activity"
    assert vm.fields["state_counts"][spa.STATE_PROCESS_DEAD] >= 1
    assert vm.fields["no_control_actions_available"] is True
    assert vm.severity.value in ("high", "critical")


def test_build_healthy_idle_shadow_paper_runner(tmp_path):
    _write_json(
        tmp_path / "reports" / "research" / "s34" / "S34_SHADOW_PAPER_STATUS.json",
        {
            "updated_at_utc": "2027-01-15T00:00:00+00:00",
            "open_trades": 0, "new_trades": 0, "risk_skipped_trades": 0,
            "total_trades": 10, "closed_trades": 5,
            "regime_config": {"enabled": True}, "risk_config": {}, "fill_model": {},
        },
    )
    fixed_now = spa.parse_iso("2027-01-15T00:00:30+00:00")
    proc_snapshot = [{
        "pid": 111,
        "cmdline": "python -u tools\\s34_shadow_paper_runner.py --loop",
        "create_time": fixed_now - 100.0,
    }]
    ctx = DashboardContext(repo_root=tmp_path, now=fixed_now, proc_snapshot=proc_snapshot)
    vm = spa.build(ctx)
    runner = vm.fields["runners"]["s34_shadow_paper_runner"]
    assert runner["parse_status"] == "ok"
    bucket = [r for r in vm.rows if r["runner"] == "s34_shadow_paper_runner"][0]
    assert bucket["state"] == spa.STATE_NO_ELIGIBLE_SIGNAL


def test_build_never_exposes_a_control_endpoint_or_live_toggle(tmp_path):
    ctx = _make_ctx(tmp_path, proc_snapshot=[])
    vm = spa.build(ctx)
    d = vm.to_dict()
    serialized = json.dumps(d).lower()
    for forbidden in ("enable_live", "confirm-live-orders", "\"action\": \"start\"", "\"action\": \"stop\""):
        assert forbidden not in serialized
    assert vm.fields["no_control_actions_available"] is True


# --- _tail_log_timestamps / _extract_json_field -----------------------------

def test_tail_log_timestamps_parses_leading_iso_lines(tmp_path):
    log = tmp_path / "v02.out.log"
    log.write_text(
        "2026-07-13T20:53:49.948111+00:00 PROTO rows=13 added=0 tick_ms=122\n"
        "2026-07-13T20:56:50.136684+00:00 PROTO rows=13 added=0 tick_ms=124\n",
        encoding="utf-8",
    )
    ts = spa._tail_log_timestamps(log)
    assert len(ts) == 2
    assert ts[1] > ts[0]


def test_tail_log_timestamps_parses_json_line_format(tmp_path):
    log = tmp_path / "shadow_paper.stdout.log"
    log.write_text(
        json.dumps({"updated_at_utc": "2026-07-13T21:05:30.313100+00:00", "total_trades": 1}) + "\n"
        + json.dumps({"updated_at_utc": "2026-07-13T21:06:30.523102+00:00", "total_trades": 2}) + "\n",
        encoding="utf-8",
    )
    ts = spa._tail_log_timestamps(log)
    assert len(ts) == 2


def test_tail_log_timestamps_missing_file_returns_empty_not_error(tmp_path):
    assert spa._tail_log_timestamps(tmp_path / "does_not_exist.log") == []


def test_tail_log_timestamps_skips_garbage_lines(tmp_path):
    log = tmp_path / "mixed.log"
    log.write_text("not a timestamp line\n{}\n2026-07-13T20:53:49+00:00 ok\n", encoding="utf-8")
    ts = spa._tail_log_timestamps(log)
    assert len(ts) == 1


def test_build_detects_a_real_gap_from_uniform_tick_log(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "s34_v_engine_v02_shadow_mirror.out.log").write_text(
        "2026-07-13T20:53:49+00:00 PROTO rows=1 added=0\n"
        "2026-07-15T19:12:56+00:00 PROTO rows=1 added=0\n",  # ~2 day gap
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "runtime" / "s34_v_engine_v02_shadow_mirror_state.json",
        {"updated_at_utc": "2026-07-15T19:15:56+00:00", "interval_sec": 180, "protocol_id": "PROTO"},
    )
    fixed_now = spa.parse_iso("2026-07-15T19:16:00+00:00")
    proc_snapshot = [{"pid": 1, "cmdline": "python -m tools.s34_v_engine_v02_shadow_mirror --loop", "create_time": fixed_now - 60}]
    ctx = DashboardContext(repo_root=tmp_path, now=fixed_now, proc_snapshot=proc_snapshot)
    vm = spa.build(ctx)
    v02 = vm.fields["runners"]["s34_v_engine_v02_shadow_mirror"]
    assert v02["log_data_available"] is True
    assert len(v02["evidence_gaps"]) == 1
    assert v02["evidence_gaps"][0]["duration_sec"] > 24 * 3600


# --- pnl["updated_utc"] phantom-bucket regression ---------------------------

def test_pnl_updated_utc_metadata_key_is_not_treated_as_a_signal_bucket(tmp_path):
    _write_json(
        tmp_path / "reports" / "shadow" / "s34_state_machine_shadow_state.json",
        {
            "updated_utc": "2026-07-15T19:18:30+00:00",
            "positions": {},
            "pnl": {
                "LONG_SILENCE": {"n": 5, "wins": 3, "wr": 0.6, "avg_net": 10.0, "total_net": 50.0},
                "updated_utc": "2026-07-15T19:18:30.599883+00:00",  # metadata, not a bucket
            },
        },
    )
    fixed_now = spa.parse_iso("2026-07-15T19:18:40+00:00")
    proc_snapshot = [{"pid": 2, "cmdline": "python -m tools.s34_realtime_shadow_runner", "create_time": fixed_now - 60}]
    ctx = DashboardContext(repo_root=tmp_path, now=fixed_now, proc_snapshot=proc_snapshot)
    vm = spa.build(ctx)
    sm_buckets = [r for r in vm.rows if r["runner"] == "s34_state_machine_shadow_runner"]
    names = {b["bucket"] for b in sm_buckets}
    assert names == {"LONG_SILENCE"}
    assert "updated_utc" not in names


def test_missing_log_reports_log_data_available_false_not_zero_gaps(tmp_path):
    # No logs/ directory at all -- must not be silently read as "confirmed
    # zero gaps == healthy".
    _write_json(
        tmp_path / "runtime" / "s34_v_engine_v02_shadow_mirror_state.json",
        {"updated_at_utc": "2026-07-15T19:15:56+00:00", "interval_sec": 180, "protocol_id": "PROTO"},
    )
    fixed_now = spa.parse_iso("2026-07-15T19:16:00+00:00")
    proc_snapshot = [{"pid": 1, "cmdline": "python -m tools.s34_v_engine_v02_shadow_mirror --loop", "create_time": fixed_now - 60}]
    ctx = DashboardContext(repo_root=tmp_path, now=fixed_now, proc_snapshot=proc_snapshot)
    vm = spa.build(ctx)
    v02 = vm.fields["runners"]["s34_v_engine_v02_shadow_mirror"]
    assert v02["log_data_available"] is False
    assert v02["evidence_gaps"] == []
    assert "no log data available" in v02["evidence_gap_reason"]


def test_current_gaps_and_historical_incidents_counted_independently(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "s34_v_engine_v02_shadow_mirror.out.log").write_text(
        "2026-07-13T20:53:49+00:00 PROTO rows=1 added=0\n"
        "2026-07-15T19:12:56+00:00 PROTO rows=1 added=0\n",  # one real live gap
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "runtime" / "s34_v_engine_v02_shadow_mirror_state.json",
        {"updated_at_utc": "2026-07-15T19:15:56+00:00", "interval_sec": 180, "protocol_id": "PROTO"},
    )
    _write_json(
        tmp_path / "runtime" / "dashboard_incident_log.json",
        {"incidents": [{"incident_id": "A"}, {"incident_id": "B"}]},  # two historical incidents
    )
    fixed_now = spa.parse_iso("2026-07-15T19:16:00+00:00")
    proc_snapshot = [{"pid": 1, "cmdline": "python -m tools.s34_v_engine_v02_shadow_mirror --loop", "create_time": fixed_now - 60}]
    ctx = DashboardContext(repo_root=tmp_path, now=fixed_now, proc_snapshot=proc_snapshot)
    vm = spa.build(ctx)
    assert vm.fields["current_evidence_gap_count"] == 1
    assert vm.fields["confirmed_historical_incident_count"] == 2
    assert "total_evidence_gap_count" not in vm.fields


def test_market_data_stale_fails_closed_on_adapter_exception(tmp_path, monkeypatch):
    def _boom(_ctx):
        raise RuntimeError("simulated native_ws failure")

    monkeypatch.setattr(spa.native_ws, "build", _boom)
    ctx = DashboardContext(repo_root=tmp_path, now=NOW, proc_snapshot=[])
    assert spa._market_data_stale(ctx) is True


# --- aggregator integration: healthy panel must read as CURRENT -------------

def test_panel_trust_state_is_current_via_aggregator_regardless_of_bucket_health(tmp_path):
    """Regression test for the bug where freshness_threshold_seconds=None made
    this panel permanently render as non-current/degraded even when every
    bucket was perfectly healthy. Goes through the real aggregator path
    (apply_freshness), not spa.build() directly, since that's where the bug
    actually manifested. Deliberately uses an all-processes-dead snapshot
    (severity=HIGH) to prove trust_state (is this reading itself fresh) and
    severity (is what it found healthy) are independent signals -- the fix
    must not require bucket health to produce a trustworthy reading."""
    from dashboard.backend.freshness import apply_freshness
    from dashboard.backend.view_models import NON_CURRENT_TRUST, Severity

    ctx = DashboardContext(repo_root=tmp_path, now=NOW, proc_snapshot=[])
    vm = spa.build(ctx)
    apply_freshness(vm)
    assert vm.trust_state not in NON_CURRENT_TRUST
    assert vm.severity == Severity.HIGH  # unhealthy (all dead) yet still a trustworthy reading
