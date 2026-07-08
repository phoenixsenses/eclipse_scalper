"""BATCH-OPERATOR-HOST-HEALTH-AND-RESTART-READINESS-DASHBOARD-V1 -- focused
tests for the pure evaluator (ami/host_health/evaluator.py).

The module under test performs NO I/O -- these tests confirm that both
structurally (AST guards) and behaviorally (deterministic classification
over plain inputs, no subprocess/registry/process calls of any kind).
"""
from __future__ import annotations

import ast
import inspect

from ami.host_health import evaluator as E
from ami.host_health.evaluator import HostHealthInputs, evaluate_restart_readiness


def _base_inputs(**overrides) -> HostHealthInputs:
    """GREEN-baseline inputs; individual tests override just the field(s)
    under test."""
    defaults = dict(
        boot_time_available=True,
        uptime_days=1.0,
        pending_reboot="FALSE",
        memory_observation_available=True,
        ram_pct_instantaneous=40.0,
        ram_pct_sustained=None,
        commit_pct_instantaneous=40.0,
        commit_pct_sustained=None,
        pagefile_pct=30.0,
        ssd_sensor_available=True,
        ssd_temp_c=45.0,
        ssd_temp_sustained_high=False,
        collector_status="HEALTHY",
        repeated_collector_failure=False,
        storage_state="STORAGE_HEALTHY",
        critical_operation_active=False,
        recent_unexpected_shutdown_count_24h=0,
        recent_disk_ntfs_critical_count_24h=0,
        recent_whea_critical_count_24h=0,
        recent_app_crash_count_24h=0,
        recent_oom_event_count_24h=0,
        event_log_access="OK",
        materially_stale=False,
    )
    defaults.update(overrides)
    return HostHealthInputs(**defaults)


# ---------------------------------------------------------------------------
# GREEN
# ---------------------------------------------------------------------------

def test_green_baseline():
    result = evaluate_restart_readiness(_base_inputs())
    assert result.state == "HOST_RESTART_GREEN"
    assert result.no_automatic_action is True
    assert result.deferred is False
    assert "UPTIME_NORMAL" in result.reason_codes


# ---------------------------------------------------------------------------
# Uptime -- never RED by itself
# ---------------------------------------------------------------------------

def test_uptime_below_7_days_is_normal():
    result = evaluate_restart_readiness(_base_inputs(uptime_days=3.0))
    assert "UPTIME_NORMAL" in result.reason_codes
    assert result.state == "HOST_RESTART_GREEN"


def test_uptime_7_to_14_days_is_yellow():
    result = evaluate_restart_readiness(_base_inputs(uptime_days=8.0))
    assert "UPTIME_ELEVATED" in result.reason_codes
    assert result.state == "HOST_RESTART_YELLOW"


def test_uptime_above_14_days_alone_is_yellow_not_red():
    result = evaluate_restart_readiness(_base_inputs(uptime_days=20.0))
    assert "UPTIME_HIGH" in result.reason_codes
    assert result.state == "HOST_RESTART_YELLOW", "uptime alone must never produce RED"


def test_uptime_alone_never_red_even_at_extreme_values():
    result = evaluate_restart_readiness(_base_inputs(uptime_days=365.0))
    assert result.state != "HOST_RESTART_RED"


def test_uptime_14_days_plus_confirmed_pending_reboot_is_red():
    result = evaluate_restart_readiness(_base_inputs(uptime_days=14.0, pending_reboot="TRUE"))
    assert result.state == "HOST_RESTART_RED"
    assert "WINDOWS_REBOOT_PENDING" in result.reason_codes
    assert "UPTIME_HIGH" in result.reason_codes


# ---------------------------------------------------------------------------
# Pending reboot
# ---------------------------------------------------------------------------

def test_pending_reboot_true_alone_is_yellow():
    result = evaluate_restart_readiness(_base_inputs(pending_reboot="TRUE"))
    assert result.state == "HOST_RESTART_YELLOW"
    assert "WINDOWS_REBOOT_PENDING" in result.reason_codes


def test_pending_reboot_false():
    result = evaluate_restart_readiness(_base_inputs(pending_reboot="FALSE"))
    assert "WINDOWS_REBOOT_NOT_PENDING" in result.reason_codes


def test_pending_reboot_unknown_does_not_force_overall_unknown():
    result = evaluate_restart_readiness(_base_inputs(pending_reboot="UNKNOWN"))
    assert result.state != "HOST_RESTART_UNKNOWN"
    assert "WINDOWS_REBOOT_STATE_UNKNOWN" in result.reason_codes


def test_pending_reboot_contradictory_forces_unknown():
    result = evaluate_restart_readiness(_base_inputs(pending_reboot="CONTRADICTORY"))
    assert result.state == "HOST_RESTART_UNKNOWN"
    assert "WINDOWS_REBOOT_STATE_CONTRADICTORY" in result.reason_codes


# ---------------------------------------------------------------------------
# RAM thresholds -- instantaneous vs sustained
# ---------------------------------------------------------------------------

def test_ram_normal_below_80():
    result = evaluate_restart_readiness(_base_inputs(ram_pct_instantaneous=79.9))
    assert "RAM_NORMAL" in result.reason_codes


def test_ram_elevated_80_to_90():
    result = evaluate_restart_readiness(_base_inputs(ram_pct_instantaneous=85.0))
    assert "RAM_PRESSURE_ELEVATED" in result.reason_codes
    assert result.state == "HOST_RESTART_YELLOW"


def test_ram_critical_instantaneous_only_is_yellow_not_red():
    """A single 95% reading with no sustained history must not alone
    produce RED -- a short spike should be displayed, not acted on."""
    result = evaluate_restart_readiness(_base_inputs(ram_pct_instantaneous=95.0, ram_pct_sustained=None))
    assert "RAM_PRESSURE_CRITICAL" in result.reason_codes
    assert result.state == "HOST_RESTART_YELLOW"


def test_ram_critical_sustained_is_red():
    result = evaluate_restart_readiness(_base_inputs(ram_pct_instantaneous=95.0, ram_pct_sustained=93.0))
    assert result.state == "HOST_RESTART_RED"
    assert "RAM_PRESSURE_CRITICAL" in result.reason_codes


def test_ram_boundary_exactly_90_sustained_is_critical():
    result = evaluate_restart_readiness(_base_inputs(ram_pct_instantaneous=90.0, ram_pct_sustained=90.0))
    assert "RAM_PRESSURE_CRITICAL" in result.reason_codes


def test_ram_boundary_exactly_80_is_elevated():
    result = evaluate_restart_readiness(_base_inputs(ram_pct_instantaneous=80.0))
    assert "RAM_PRESSURE_ELEVATED" in result.reason_codes


def test_ram_observation_unavailable_forces_unknown():
    result = evaluate_restart_readiness(_base_inputs(memory_observation_available=False))
    assert result.state == "HOST_RESTART_UNKNOWN"


# ---------------------------------------------------------------------------
# Commit thresholds
# ---------------------------------------------------------------------------

def test_commit_normal_below_85():
    result = evaluate_restart_readiness(_base_inputs(commit_pct_instantaneous=84.9))
    assert "COMMIT_PRESSURE_NORMAL" in result.reason_codes


def test_commit_boundary_85_is_elevated():
    result = evaluate_restart_readiness(_base_inputs(commit_pct_instantaneous=85.0))
    assert "COMMIT_PRESSURE_ELEVATED" in result.reason_codes
    assert result.state == "HOST_RESTART_YELLOW"


def test_commit_boundary_95_sustained_is_critical_red():
    result = evaluate_restart_readiness(_base_inputs(commit_pct_instantaneous=95.0, commit_pct_sustained=95.0))
    assert "COMMIT_PRESSURE_CRITICAL" in result.reason_codes
    assert result.state == "HOST_RESTART_RED"


def test_commit_critical_instantaneous_only_is_yellow():
    result = evaluate_restart_readiness(_base_inputs(commit_pct_instantaneous=97.0, commit_pct_sustained=None))
    assert result.state == "HOST_RESTART_YELLOW"


# ---------------------------------------------------------------------------
# Pagefile thresholds
# ---------------------------------------------------------------------------

def test_pagefile_normal():
    result = evaluate_restart_readiness(_base_inputs(pagefile_pct=50.0))
    assert "PAGEFILE_PRESSURE_NORMAL" in result.reason_codes


def test_pagefile_elevated_boundary():
    result = evaluate_restart_readiness(_base_inputs(pagefile_pct=85.0))
    assert "PAGEFILE_PRESSURE_ELEVATED" in result.reason_codes
    assert result.state == "HOST_RESTART_YELLOW"


def test_pagefile_critical_boundary():
    result = evaluate_restart_readiness(_base_inputs(pagefile_pct=95.0))
    assert "PAGEFILE_PRESSURE_CRITICAL" in result.reason_codes


# ---------------------------------------------------------------------------
# Storage state
# ---------------------------------------------------------------------------

def test_storage_healthy():
    result = evaluate_restart_readiness(_base_inputs(storage_state="STORAGE_HEALTHY"))
    assert result.state == "HOST_RESTART_GREEN"


def test_storage_warning_is_yellow():
    result = evaluate_restart_readiness(_base_inputs(storage_state="STORAGE_WARNING"))
    assert result.state == "HOST_RESTART_YELLOW"


def test_storage_critical_is_red():
    result = evaluate_restart_readiness(_base_inputs(storage_state="STORAGE_CRITICAL"))
    assert result.state == "HOST_RESTART_RED"


def test_storage_emergency_is_red():
    result = evaluate_restart_readiness(_base_inputs(storage_state="STORAGE_EMERGENCY"))
    assert result.state == "HOST_RESTART_RED"


def test_800gb_threshold_constant():
    assert E.D_DRIVE_INTERVENTION_FREE_GB == 800.0


# ---------------------------------------------------------------------------
# SSD temperature
# ---------------------------------------------------------------------------

def test_ssd_temp_green_below_60():
    result = evaluate_restart_readiness(_base_inputs(ssd_temp_c=55.0))
    assert "SSD_TEMPERATURE_NORMAL" in result.reason_codes


def test_ssd_temp_yellow_60_to_70():
    result = evaluate_restart_readiness(_base_inputs(ssd_temp_c=65.0))
    assert "SSD_TEMPERATURE_ELEVATED" in result.reason_codes
    assert result.state == "HOST_RESTART_YELLOW"


def test_ssd_temp_red_sustained_70_plus():
    result = evaluate_restart_readiness(_base_inputs(ssd_temp_c=72.0, ssd_temp_sustained_high=True))
    assert result.state == "HOST_RESTART_RED"
    assert "SSD_TEMPERATURE_CRITICAL" in result.reason_codes


def test_ssd_temp_unsustained_high_is_yellow_not_red():
    result = evaluate_restart_readiness(_base_inputs(ssd_temp_c=72.0, ssd_temp_sustained_high=False))
    assert result.state == "HOST_RESTART_YELLOW"


def test_ssd_sensor_unavailable_does_not_force_red_or_unknown():
    result = evaluate_restart_readiness(_base_inputs(ssd_sensor_available=False, ssd_temp_c=None))
    assert result.state == "HOST_RESTART_GREEN"
    assert "SSD_SENSOR_UNKNOWN" in result.reason_codes
    assert "ssd_temp_c" in result.unknown_fields


# ---------------------------------------------------------------------------
# Collector / critical-process health
# ---------------------------------------------------------------------------

def test_collector_healthy():
    result = evaluate_restart_readiness(_base_inputs(collector_status="HEALTHY"))
    assert "COLLECTORS_HEALTHY" in result.reason_codes
    assert result.state == "HOST_RESTART_GREEN"


def test_collector_degraded_is_yellow():
    result = evaluate_restart_readiness(_base_inputs(collector_status="DEGRADED"))
    assert result.state == "HOST_RESTART_YELLOW"
    assert "COLLECTORS_DEGRADED" in result.reason_codes


def test_collector_failed_first_time_is_yellow():
    result = evaluate_restart_readiness(_base_inputs(collector_status="FAILED", repeated_collector_failure=False))
    assert result.state == "HOST_RESTART_YELLOW"


def test_collector_repeated_failure_is_red():
    result = evaluate_restart_readiness(_base_inputs(collector_status="FAILED", repeated_collector_failure=True))
    assert result.state == "HOST_RESTART_RED"
    assert "COLLECTORS_FAILED" in result.reason_codes


def test_collector_unknown_forces_overall_unknown():
    result = evaluate_restart_readiness(_base_inputs(collector_status="UNKNOWN"))
    assert result.state == "HOST_RESTART_UNKNOWN"


# ---------------------------------------------------------------------------
# Critical operation active -> deferred
# ---------------------------------------------------------------------------

def test_critical_operation_active_defers_yellow():
    result = evaluate_restart_readiness(_base_inputs(uptime_days=8.0, critical_operation_active=True))
    assert result.deferred is True
    assert "CRITICAL_OPERATION_ACTIVE" in result.reason_codes
    assert "RESTART_RECOMMENDED_BUT_DEFER_UNTIL_SAFE_CHECKPOINT" in result.recommended_action


def test_critical_operation_active_does_not_defer_green():
    result = evaluate_restart_readiness(_base_inputs(critical_operation_active=True))
    assert result.state == "HOST_RESTART_GREEN"
    assert result.deferred is False


# ---------------------------------------------------------------------------
# Event-log-derived counts
# ---------------------------------------------------------------------------

def test_recent_unexpected_shutdown_single_is_yellow():
    result = evaluate_restart_readiness(_base_inputs(recent_unexpected_shutdown_count_24h=1))
    assert result.state == "HOST_RESTART_YELLOW"


def test_recent_unexpected_shutdown_repeated_is_red():
    result = evaluate_restart_readiness(_base_inputs(recent_unexpected_shutdown_count_24h=2))
    assert result.state == "HOST_RESTART_RED"


def test_disk_ntfs_critical_event_is_red():
    result = evaluate_restart_readiness(_base_inputs(recent_disk_ntfs_critical_count_24h=1))
    assert result.state == "HOST_RESTART_RED"
    assert "DISK_EVENT_CRITICAL" in result.reason_codes


def test_whea_critical_event_is_red():
    result = evaluate_restart_readiness(_base_inputs(recent_whea_critical_count_24h=1))
    assert result.state == "HOST_RESTART_RED"
    assert "WHEA_EVENT_CRITICAL" in result.reason_codes


def test_repeated_app_crash_is_yellow():
    result = evaluate_restart_readiness(_base_inputs(recent_app_crash_count_24h=3))
    assert result.state == "HOST_RESTART_YELLOW"
    assert "RECENT_APP_CRASHES" in result.reason_codes


def test_single_app_crash_below_threshold_is_not_flagged():
    result = evaluate_restart_readiness(_base_inputs(recent_app_crash_count_24h=1))
    assert "RECENT_APP_CRASHES" not in result.reason_codes


def test_oom_event_is_red():
    result = evaluate_restart_readiness(_base_inputs(recent_oom_event_count_24h=1))
    assert result.state == "HOST_RESTART_RED"
    assert "RESOURCE_EXHAUSTION_EVENTS_CRITICAL" in result.reason_codes


def test_event_log_access_denied_flagged_but_not_forced_unknown():
    result = evaluate_restart_readiness(_base_inputs(event_log_access="ACCESS_DENIED"))
    assert result.state != "HOST_RESTART_UNKNOWN"
    assert "EVENT_LOG_ACCESS_DENIED" in result.reason_codes


# ---------------------------------------------------------------------------
# Fail-closed UNKNOWN gate
# ---------------------------------------------------------------------------

def test_boot_time_unavailable_forces_unknown():
    result = evaluate_restart_readiness(_base_inputs(boot_time_available=False))
    assert result.state == "HOST_RESTART_UNKNOWN"
    assert "boot_time" in result.unknown_fields


def test_materially_stale_forces_unknown():
    result = evaluate_restart_readiness(_base_inputs(materially_stale=True))
    assert result.state == "HOST_RESTART_UNKNOWN"
    assert "OBSERVATION_STALE" in result.reason_codes


def test_unknown_state_never_guesses_a_recommended_restart_action():
    result = evaluate_restart_readiness(_base_inputs(boot_time_available=False))
    assert "restart" not in result.recommended_action.lower() or "do not guess" in result.recommended_action.lower()


def test_unknown_gate_wins_over_red_tier_conditions():
    """Even with several RED-tier conditions present, a core-unknown
    trigger must still win (fail-closed takes priority)."""
    result = evaluate_restart_readiness(_base_inputs(
        boot_time_available=False,
        recent_whea_critical_count_24h=5,
        storage_state="STORAGE_EMERGENCY",
    ))
    assert result.state == "HOST_RESTART_UNKNOWN"


# ---------------------------------------------------------------------------
# Determinism / reason ordering
# ---------------------------------------------------------------------------

def test_deterministic_repeated_calls():
    inputs = _base_inputs(ram_pct_instantaneous=85.0, pending_reboot="TRUE")
    a = evaluate_restart_readiness(inputs)
    b = evaluate_restart_readiness(inputs)
    assert a == b


def test_reason_codes_are_sorted_tuple():
    result = evaluate_restart_readiness(_base_inputs(ram_pct_instantaneous=85.0))
    assert result.reason_codes == tuple(sorted(result.reason_codes))


def test_primary_reason_is_deterministic_among_multiple_red_triggers():
    inputs = _base_inputs(recent_whea_critical_count_24h=1, recent_disk_ntfs_critical_count_24h=1)
    a = evaluate_restart_readiness(inputs)
    b = evaluate_restart_readiness(inputs)
    assert a.primary_reason == b.primary_reason == "WHEA_EVENT_CRITICAL"


def test_all_reason_codes_are_declared():
    result = evaluate_restart_readiness(_base_inputs(ram_pct_instantaneous=95.0, ram_pct_sustained=95.0, pending_reboot="TRUE"))
    for code in result.reason_codes:
        assert code in E.REASON_CODES, code


def test_host_restart_states_frozenset_exact():
    assert E.HOST_RESTART_STATES == {
        "HOST_RESTART_GREEN", "HOST_RESTART_YELLOW", "HOST_RESTART_RED", "HOST_RESTART_UNKNOWN",
    }


def test_no_automatic_action_always_true():
    for inputs in (
        _base_inputs(),
        _base_inputs(storage_state="STORAGE_EMERGENCY"),
        _base_inputs(boot_time_available=False),
    ):
        assert evaluate_restart_readiness(inputs).no_automatic_action is True


# ---------------------------------------------------------------------------
# Structural no-mutation / no-I/O guards
# ---------------------------------------------------------------------------

_FORBIDDEN_CALL_NAMES = {
    "system", "popen", "run", "call", "check_call", "check_output", "Popen",
    "remove", "unlink", "rmdir", "kill", "terminate",
}
_FORBIDDEN_TOKENS = (
    "shutdown.exe", "Restart-Computer", "os.system(", "subprocess.run(",
    "subprocess.Popen(", "winreg.", "socket.socket(", "sqlite3.connect(", "open(",
)


def test_module_never_imports_subprocess_os_socket_or_registry():
    src = inspect.getsource(E)
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {"subprocess", "socket", "winreg", "sqlite3", "urllib", "requests", "os"}
    assert not (imported & forbidden), imported & forbidden


def test_module_source_never_contains_restart_or_shutdown_tokens():
    src = inspect.getsource(E)
    for token in _FORBIDDEN_TOKENS:
        assert token not in src, token


def test_module_defines_no_forbidden_calls():
    src = inspect.getsource(E)
    tree = ast.parse(src)
    call_names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                call_names.add(n.func.id)
            elif isinstance(n.func, ast.Attribute):
                call_names.add(n.func.attr)
    assert not (call_names & _FORBIDDEN_CALL_NAMES), call_names & _FORBIDDEN_CALL_NAMES
