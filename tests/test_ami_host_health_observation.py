"""BATCH-OPERATOR-HOST-HEALTH-AND-RESTART-READINESS-DASHBOARD-V1 -- focused
tests for the observation collector (ami/host_health/observation.py).

Uses monkeypatch to fully control psutil/PowerShell/file inputs so these
tests are deterministic and independent of the machine they run on.
Confirms fail-closed behavior (missing/broken sensor -> None/UNKNOWN,
never a fabricated "healthy" default) and the structural safety
guarantees over `_POWERSHELL_SCRIPT` (no restart/reboot/shutdown/
registry-write/package-install cmdlet anywhere in it).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ami.host_health import observation as O


# ---------------------------------------------------------------------------
# sustained_value -- pure helper, instantaneous vs sustained
# ---------------------------------------------------------------------------

def test_sustained_value_empty_history_is_none():
    assert O.sustained_value([], window_minutes=15, now_ts=1000.0) is None


def test_sustained_value_single_sample_is_instantaneous_only():
    assert O.sustained_value([(990.0, 50.0)], window_minutes=15, now_ts=1000.0) is None


def test_sustained_value_full_window_returns_min():
    now = 10_000.0
    history = [(now - 14 * 60, 70.0), (now - 10 * 60, 95.0), (now - 1 * 60, 60.0)]
    result = O.sustained_value(history, window_minutes=15, now_ts=now)
    assert result == 60.0


def test_sustained_value_partial_window_coverage_is_none():
    """History exists but doesn't actually reach back to window_start
    yet -- must not claim "sustained" prematurely."""
    now = 10_000.0
    history = [(now - 2 * 60, 90.0), (now - 1 * 60, 91.0)]
    assert O.sustained_value(history, window_minutes=15, now_ts=now) is None


def test_sustained_value_ignores_samples_outside_window():
    now = 10_000.0
    history = [(now - 60 * 60, 5.0), (now - 14 * 60, 70.0), (now - 1 * 60, 80.0)]
    result = O.sustained_value(history, window_minutes=15, now_ts=now)
    assert result == 70.0


# ---------------------------------------------------------------------------
# _run_powershell -- fail-closed
# ---------------------------------------------------------------------------

def test_run_powershell_returns_none_on_nonzero_exit(monkeypatch):
    class FakeProc:
        returncode = 1
        stdout = b""
    monkeypatch.setattr(O.subprocess, "run", lambda *a, **k: FakeProc())
    monkeypatch.setattr(O.sys, "platform", "win32")
    assert O._run_powershell("whatever") is None


def test_run_powershell_returns_none_on_timeout(monkeypatch):
    def raise_timeout(*a, **k):
        raise O.subprocess.TimeoutExpired(cmd="powershell", timeout=5)
    monkeypatch.setattr(O.subprocess, "run", raise_timeout)
    monkeypatch.setattr(O.sys, "platform", "win32")
    assert O._run_powershell("whatever") is None


def test_run_powershell_returns_none_on_bad_json(monkeypatch):
    class FakeProc:
        returncode = 0
        stdout = b"not json{{{"
    monkeypatch.setattr(O.subprocess, "run", lambda *a, **k: FakeProc())
    monkeypatch.setattr(O.sys, "platform", "win32")
    assert O._run_powershell("whatever") is None


def test_run_powershell_parses_valid_json(monkeypatch):
    class FakeProc:
        returncode = 0
        stdout = json.dumps({"a": 1}).encode("utf-8")
    monkeypatch.setattr(O.subprocess, "run", lambda *a, **k: FakeProc())
    monkeypatch.setattr(O.sys, "platform", "win32")
    assert O._run_powershell("whatever") == {"a": 1}


def test_run_powershell_skipped_on_non_windows(monkeypatch):
    monkeypatch.setattr(O.sys, "platform", "linux")
    assert O._run_powershell("whatever") is None


# ---------------------------------------------------------------------------
# _event_bucket_counts
# ---------------------------------------------------------------------------

def test_event_bucket_counts_none_input_is_unknown():
    result = O._event_bucket_counts(None, datetime.now(tz=timezone.utc))
    assert result == {"count_24h": None, "count_7d": None}


def test_event_bucket_counts_empty_list_is_zero_not_unknown():
    result = O._event_bucket_counts([], datetime.now(tz=timezone.utc))
    assert result == {"count_24h": 0, "count_7d": 0}


def test_event_bucket_counts_buckets_by_age():
    now = datetime.now(tz=timezone.utc)
    events = [
        {"time_utc": (now - timedelta(hours=1)).isoformat()},
        {"time_utc": (now - timedelta(hours=48)).isoformat()},
        {"time_utc": (now - timedelta(hours=200)).isoformat()},
    ]
    result = O._event_bucket_counts(events, now)
    assert result["count_24h"] == 1
    assert result["count_7d"] == 2


# ---------------------------------------------------------------------------
# _collect_pids_health -- fail-closed over missing/stale files
# ---------------------------------------------------------------------------

def test_collect_pids_health_missing_files_is_unknown(tmp_path: Path):
    status, detail, hb_age = O._collect_pids_health(tmp_path)
    assert status == "UNKNOWN"
    assert hb_age is None


def test_collect_pids_health_ok_state(tmp_path: Path):
    health_dir = tmp_path / "logs" / "health"
    health_dir.mkdir(parents=True)
    (health_dir / "overall.json").write_text(json.dumps({
        "state": "ok",
        "ts_utc": datetime.now(tz=timezone.utc).isoformat(),
    }))
    status, detail, hb_age = O._collect_pids_health(tmp_path)
    assert status == "HEALTHY"


def test_collect_pids_health_degraded_state(tmp_path: Path):
    health_dir = tmp_path / "logs" / "health"
    health_dir.mkdir(parents=True)
    (health_dir / "overall.json").write_text(json.dumps({
        "state": "degraded",
        "ts_utc": datetime.now(tz=timezone.utc).isoformat(),
    }))
    status, _, _ = O._collect_pids_health(tmp_path)
    assert status == "DEGRADED"


def test_collect_pids_health_halted_state(tmp_path: Path):
    health_dir = tmp_path / "logs" / "health"
    health_dir.mkdir(parents=True)
    (health_dir / "overall.json").write_text(json.dumps({
        "state": "halted",
        "ts_utc": datetime.now(tz=timezone.utc).isoformat(),
    }))
    status, _, _ = O._collect_pids_health(tmp_path)
    assert status == "FAILED"


def test_collect_pids_health_stale_overall_is_unknown(tmp_path: Path):
    health_dir = tmp_path / "logs" / "health"
    health_dir.mkdir(parents=True)
    stale_ts = (datetime.now(tz=timezone.utc) - timedelta(hours=2)).isoformat()
    (health_dir / "overall.json").write_text(json.dumps({"state": "ok", "ts_utc": stale_ts}))
    status, detail, _ = O._collect_pids_health(tmp_path)
    assert status == "UNKNOWN"
    assert detail.get("reason") == "overall_health_stale"


def test_collect_pids_health_malformed_json_does_not_raise(tmp_path: Path):
    health_dir = tmp_path / "logs" / "health"
    health_dir.mkdir(parents=True)
    (health_dir / "overall.json").write_text("{not valid json")
    status, detail, _ = O._collect_pids_health(tmp_path)
    assert status == "UNKNOWN"
    assert "overall_error" in detail


# ---------------------------------------------------------------------------
# _storage_health_state -- delegates to governance classifier, fails closed
# ---------------------------------------------------------------------------

def test_storage_health_state_none_inputs_unknown():
    assert O._storage_health_state(None, None) == "STORAGE_STATE_UNKNOWN"


def test_storage_health_state_zero_total_unknown():
    assert O._storage_health_state(100, 0) == "STORAGE_STATE_UNKNOWN"


def test_storage_health_state_healthy():
    total = 2_000_000_000_000
    free = 1_000_000_000_000
    assert O._storage_health_state(free, total) == "STORAGE_HEALTHY"


def test_storage_health_state_emergency():
    total = 2_000_000_000_000
    free = 10_000_000_000  # ~9.3 GiB free, well under both pct and abs emergency thresholds
    assert O._storage_health_state(free, total) == "STORAGE_EMERGENCY"


# ---------------------------------------------------------------------------
# build_health_inputs -- mapping + staleness detection
# ---------------------------------------------------------------------------

def _sample_observation(**overrides) -> O.HostObservation:
    now = datetime.now(tz=timezone.utc).isoformat()
    fields = dict(
        observation_ts_utc=now, host_name="H", os_identity="win32",
        boot_ts_utc=now, uptime_seconds=3600.0, uptime_human="1h 0m",
        pending_reboot="FALSE", pending_reboot_evidence={},
        ram_total_bytes=100, ram_available_bytes=50, ram_used_pct=50.0,
        commit_limit_kb=100, commit_used_kb=50, commit_used_pct=50.0,
        pagefile_total_bytes=100, pagefile_used_bytes=50, pagefile_used_pct=50.0,
        cpu_pct_snapshot=10.0, cpu_pct_recent_avg=None,
        c_drive_total_bytes=100, c_drive_free_bytes=50,
        d_drive_total_bytes=100, d_drive_free_bytes=50, d_drive_free_gb=50.0,
        d_drive_distance_to_threshold_gb=-750.0,
        microstructure_db_size_bytes=100, microstructure_wal_size_bytes=10,
        storage_health_state="STORAGE_HEALTHY",
        collector_status="HEALTHY", collector_status_detail={}, latest_collector_heartbeat_age_sec=1.0,
        physical_disks=(), ssd_990pro_detected=False, ssd_temp_c=None, ssd_health_state="UNKNOWN",
        recent_unexpected_shutdown_count_24h=0, recent_unexpected_shutdown_count_7d=0,
        recent_app_crash_count_24h=0, recent_disk_ntfs_critical_count_24h=0,
        recent_whea_critical_count_24h=0, recent_oom_event_count_24h=0, event_log_access="OK",
        critical_operation_active=False, critical_operation_evidence=(),
        observation_errors=(), stale_fields=(), unknown_fields=(),
        powershell_available=True,
    )
    fields.update(overrides)
    return O.HostObservation(**fields)


def test_build_health_inputs_basic_mapping():
    obs = _sample_observation()
    inputs = O.build_health_inputs(obs)
    assert inputs.uptime_days == pytest.approx(3600.0 / 86400.0)
    assert inputs.pending_reboot == "FALSE"
    assert inputs.ram_pct_instantaneous == 50.0
    assert inputs.materially_stale is False


def test_build_health_inputs_stale_observation_timestamp():
    old_ts = (datetime.now(tz=timezone.utc) - timedelta(minutes=10)).isoformat()
    obs = _sample_observation(observation_ts_utc=old_ts)
    inputs = O.build_health_inputs(obs)
    assert inputs.materially_stale is True


def test_build_health_inputs_explicit_stale_override():
    obs = _sample_observation()
    inputs = O.build_health_inputs(obs, materially_stale=True)
    assert inputs.materially_stale is True


def test_build_health_inputs_passes_through_sustained_values():
    obs = _sample_observation()
    inputs = O.build_health_inputs(obs, ram_pct_sustained=77.0, commit_pct_sustained=88.0)
    assert inputs.ram_pct_sustained == 77.0
    assert inputs.commit_pct_sustained == 88.0


def test_build_health_inputs_boot_unavailable_when_missing():
    obs = _sample_observation(boot_ts_utc=None, uptime_seconds=None)
    inputs = O.build_health_inputs(obs)
    assert inputs.boot_time_available is False


# ---------------------------------------------------------------------------
# collect_host_observation -- fully monkeypatched, fail-closed integration
# ---------------------------------------------------------------------------

def test_collect_host_observation_fail_closed_when_psutil_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(O, "psutil", None)
    monkeypatch.setattr(O, "_run_powershell", lambda *a, **k: None)
    obs = O.collect_host_observation(repo_root=tmp_path)
    assert obs.boot_ts_utc is None
    assert obs.ram_used_pct is None
    assert obs.commit_used_pct is None
    assert obs.pending_reboot == "UNKNOWN"
    assert "boot_time" in obs.unknown_fields
    assert obs.powershell_available is False


def test_collect_host_observation_never_raises_on_partial_psutil_failure(monkeypatch, tmp_path: Path):
    class BrokenPsutil:
        @staticmethod
        def boot_time():
            raise RuntimeError("boom")

        @staticmethod
        def virtual_memory():
            raise RuntimeError("boom")

        @staticmethod
        def swap_memory():
            raise RuntimeError("boom")

        @staticmethod
        def cpu_percent(interval=0.0):
            raise RuntimeError("boom")

        @staticmethod
        def disk_usage(path):
            raise RuntimeError("boom")

    monkeypatch.setattr(O, "psutil", BrokenPsutil())
    monkeypatch.setattr(O, "_run_powershell", lambda *a, **k: None)
    obs = O.collect_host_observation(repo_root=tmp_path)
    assert obs.boot_ts_utc is None
    assert obs.d_drive_free_gb is None
    assert len(obs.observation_errors) > 0


def test_collect_host_observation_reduces_to_evaluator_unknown_state(monkeypatch, tmp_path: Path):
    """End-to-end: with every sensor unavailable, the evaluator must
    land on HOST_RESTART_UNKNOWN, never a fabricated GREEN."""
    from ami.host_health.evaluator import evaluate_restart_readiness

    monkeypatch.setattr(O, "psutil", None)
    monkeypatch.setattr(O, "_run_powershell", lambda *a, **k: None)
    obs = O.collect_host_observation(repo_root=tmp_path)
    inputs = O.build_health_inputs(obs)
    result = evaluate_restart_readiness(inputs)
    assert result.state == "HOST_RESTART_UNKNOWN"
    assert result.no_automatic_action is True


def test_collect_host_observation_uses_microstructure_db_size(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(O, "psutil", None)
    monkeypatch.setattr(O, "_run_powershell", lambda *a, **k: None)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "microstructure.db").write_bytes(b"x" * 1234)
    obs = O.collect_host_observation(repo_root=tmp_path)
    assert obs.microstructure_db_size_bytes == 1234


# ---------------------------------------------------------------------------
# Structural safety guards over the PowerShell script itself
# ---------------------------------------------------------------------------

_FORBIDDEN_PS_PREFIXES = (
    "Set-", "Remove-", "Stop-", "Restart-", "New-", "Clear-", "Install-",
    "Uninstall-", "Disable-", "Enable-",
)


def test_powershell_script_contains_no_mutating_cmdlets():
    script = O._POWERSHELL_SCRIPT
    for prefix in _FORBIDDEN_PS_PREFIXES:
        assert prefix not in script, f"found forbidden cmdlet prefix {prefix!r} in _POWERSHELL_SCRIPT"


def test_powershell_script_contains_no_shutdown_or_restart_verbs():
    """"shutdown" legitimately appears as part of observing unexpected-
    shutdown *events* (a Get-WinEvent filter, e.g. `unexpected_shutdown`
    the variable name) -- this checks for actually dangerous invocation
    patterns, not the English word."""
    script = O._POWERSHELL_SCRIPT.lower()
    for token in ("shutdown.exe", "shutdown /r", "shutdown -r", "shutdown -s",
                  "stop-computer", "restart-computer"):
        assert token not in script, token


def test_powershell_script_only_uses_readonly_cmdlets():
    script = O._POWERSHELL_SCRIPT
    allowed_verbs = ("Get-", "Test-", "ConvertTo-", "Measure-", "ForEach-", "Where-", "Select-", "Write-")
    # Every '<Word>-<Word>' cmdlet-shaped token must start with an allowed
    # verb. Single-quoted string literals (provider names like
    # 'Microsoft-Windows-WHEA-Logger', 'disk', 'Ntfs') are stripped first
    # so they can't be mistaken for cmdlet calls.
    import re
    without_string_literals = re.sub(r"'[^']*'", "''", script)
    for match in re.finditer(r"\b[A-Z][A-Za-z]+-[A-Z][A-Za-z]+\b", without_string_literals):
        token = match.group(0)
        assert any(token.startswith(v) for v in allowed_verbs), token


def test_powershell_timeout_is_bounded():
    assert 0 < O.POWERSHELL_TIMEOUT_SEC <= 60
