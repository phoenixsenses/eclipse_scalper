"""BATCH-OPERATOR-HOST-HEALTH-AND-RESTART-READINESS-DASHBOARD-V1.

Pure, deterministic restart-readiness state machine. This module performs
NO I/O of any kind -- it never opens a file, a socket, a subprocess, or a
database connection, and it never calls a Windows API. Every function
takes already-collected plain values (see `observation.py` for the
collector) and returns a classification. It never restarts, shuts down,
suspends, or otherwise mutates anything; it only classifies.

State model (exact names, do not rename):
    HOST_RESTART_GREEN    -- no restart needed, continue normal work.
    HOST_RESTART_YELLOW   -- restart advisable in the next safe window
                              (~24h), no automatic action.
    HOST_RESTART_RED      -- controlled restart strongly recommended;
                              confirm no critical operation is active
                              first. No automatic action.
    HOST_RESTART_UNKNOWN  -- required observations are unavailable,
                              contradictory, or stale. Do not guess.

Fail-closed: any of the five core "cannot evaluate" conditions (boot time
unknown, contradictory pending-reboot evidence, memory observation
unavailable, collector health unevaluable, materially stale data) forces
HOST_RESTART_UNKNOWN, taking priority over every other rule. Uptime alone
never produces RED (RED requires uptime>=14d combined with a confirmed
pending reboot, or an independent RED-tier condition).
"""
from __future__ import annotations

from dataclasses import dataclass

HOST_RESTART_STATES = frozenset({
    "HOST_RESTART_GREEN", "HOST_RESTART_YELLOW", "HOST_RESTART_RED", "HOST_RESTART_UNKNOWN",
})

# ---------------------------------------------------------------------------
# Thresholds (Phase 4/5/8 defaults -- documented, overridable via keyword
# arguments to evaluate_restart_readiness, never via hidden global state)
# ---------------------------------------------------------------------------

RAM_ELEVATED_PCT = 80.0
RAM_CRITICAL_PCT = 90.0
COMMIT_ELEVATED_PCT = 85.0
COMMIT_CRITICAL_PCT = 95.0
PAGEFILE_ELEVATED_PCT = 85.0
PAGEFILE_CRITICAL_PCT = 95.0
SUSTAINED_WINDOW_MINUTES = 15

UPTIME_ELEVATED_DAYS = 7.0
UPTIME_HIGH_DAYS = 14.0

SSD_TEMP_ELEVATED_C = 60.0
SSD_TEMP_CRITICAL_C = 70.0

D_DRIVE_INTERVENTION_FREE_GB = 800.0  # distance-to-threshold display only; not a restart trigger by itself

APP_CRASH_YELLOW_THRESHOLD_24H = 3
UNEXPECTED_SHUTDOWN_YELLOW_THRESHOLD_24H = 1
UNEXPECTED_SHUTDOWN_RED_THRESHOLD_24H = 2
OOM_EVENT_RED_THRESHOLD_24H = 1

REASON_CODES = frozenset({
    "UPTIME_NORMAL", "UPTIME_ELEVATED", "UPTIME_HIGH",
    "WINDOWS_REBOOT_PENDING", "WINDOWS_REBOOT_NOT_PENDING", "WINDOWS_REBOOT_STATE_UNKNOWN",
    "WINDOWS_REBOOT_STATE_CONTRADICTORY",
    "RAM_NORMAL", "RAM_PRESSURE_ELEVATED", "RAM_PRESSURE_CRITICAL",
    "COMMIT_PRESSURE_NORMAL", "COMMIT_PRESSURE_ELEVATED", "COMMIT_PRESSURE_CRITICAL",
    "PAGEFILE_PRESSURE_NORMAL", "PAGEFILE_PRESSURE_ELEVATED", "PAGEFILE_PRESSURE_CRITICAL",
    "RECENT_APP_CRASHES", "RECENT_UNEXPECTED_SHUTDOWN",
    "DISK_EVENT_CRITICAL", "WHEA_EVENT_CRITICAL", "RESOURCE_EXHAUSTION_EVENTS_CRITICAL",
    "SSD_TEMPERATURE_NORMAL", "SSD_TEMPERATURE_ELEVATED", "SSD_TEMPERATURE_CRITICAL", "SSD_SENSOR_UNKNOWN",
    "COLLECTORS_HEALTHY", "COLLECTORS_DEGRADED", "COLLECTORS_FAILED", "COLLECTORS_UNKNOWN",
    "CRITICAL_OPERATION_ACTIVE",
    "STORAGE_HEALTHY", "STORAGE_WARNING", "STORAGE_CRITICAL", "STORAGE_EMERGENCY", "STORAGE_STATE_UNKNOWN",
    "OBSERVATION_STALE", "OBSERVATION_INCOMPLETE", "EVENT_LOG_ACCESS_DENIED",
})

_RED = 3
_YELLOW = 2
_GREEN = 1
_NONE = 0

# Deterministic tie-break order: highest-priority reason at a given
# severity is reported as `primary_reason`.
_PRIORITY_ORDER = (
    "WHEA_EVENT_CRITICAL", "DISK_EVENT_CRITICAL", "RESOURCE_EXHAUSTION_EVENTS_CRITICAL",
    "RECENT_UNEXPECTED_SHUTDOWN", "COLLECTORS_FAILED", "RAM_PRESSURE_CRITICAL",
    "COMMIT_PRESSURE_CRITICAL", "PAGEFILE_PRESSURE_CRITICAL", "SSD_TEMPERATURE_CRITICAL",
    "STORAGE_EMERGENCY", "STORAGE_CRITICAL",
    "WINDOWS_REBOOT_PENDING", "UPTIME_HIGH", "UPTIME_ELEVATED",
    "RAM_PRESSURE_ELEVATED", "COMMIT_PRESSURE_ELEVATED", "PAGEFILE_PRESSURE_ELEVATED",
    "SSD_TEMPERATURE_ELEVATED", "COLLECTORS_DEGRADED", "RECENT_APP_CRASHES", "STORAGE_WARNING",
    "OBSERVATION_STALE",
)

_UNKNOWN_PRIORITY_ORDER = (
    "OBSERVATION_INCOMPLETE", "WINDOWS_REBOOT_STATE_CONTRADICTORY", "OBSERVATION_STALE",
)


@dataclass(frozen=True)
class HostHealthInputs:
    """Already-collected, plain-value inputs. See `observation.py` for
    how a real `HostObservation` is reduced to this shape. Kept separate
    from `HostObservation` so this module can stay import-light and pure."""

    boot_time_available: bool
    uptime_days: float | None

    pending_reboot: str  # "TRUE" | "FALSE" | "UNKNOWN" | "CONTRADICTORY"

    memory_observation_available: bool
    ram_pct_instantaneous: float | None
    ram_pct_sustained: float | None  # None if no sustained history yet

    commit_pct_instantaneous: float | None
    commit_pct_sustained: float | None

    pagefile_pct: float | None

    ssd_sensor_available: bool
    ssd_temp_c: float | None
    ssd_temp_sustained_high: bool  # True if the elevated/critical reading has repeated/persisted

    collector_status: str  # "HEALTHY" | "DEGRADED" | "FAILED" | "UNKNOWN"
    repeated_collector_failure: bool

    storage_state: str  # STORAGE_HEALTHY | STORAGE_WARNING | STORAGE_CRITICAL | STORAGE_EMERGENCY | STORAGE_STATE_UNKNOWN

    critical_operation_active: bool

    recent_unexpected_shutdown_count_24h: int
    recent_disk_ntfs_critical_count_24h: int
    recent_whea_critical_count_24h: int
    recent_app_crash_count_24h: int
    recent_oom_event_count_24h: int
    event_log_access: str  # "OK" | "ACCESS_DENIED" | "UNKNOWN"

    materially_stale: bool


@dataclass(frozen=True)
class HostHealthEvaluation:
    state: str
    primary_reason: str
    reason_codes: tuple[str, ...]
    recommended_action: str
    deferred: bool
    unknown_fields: tuple[str, ...]
    no_automatic_action: bool = True


def _pick_primary(reasons: set[str], order: tuple[str, ...]) -> str | None:
    for code in order:
        if code in reasons:
            return code
    return None


def evaluate_restart_readiness(inputs: HostHealthInputs) -> HostHealthEvaluation:
    """Pure classifier. Deterministic: identical `inputs` always produce
    an identical `HostHealthEvaluation` (same state, same primary_reason,
    same reason_codes tuple, same recommended_action)."""

    reasons: set[str] = set()
    unknown_fields: list[str] = []

    # -----------------------------------------------------------------
    # Fail-closed UNKNOWN gate -- checked first, wins over everything.
    # -----------------------------------------------------------------
    unknown_triggers: set[str] = set()
    if not inputs.boot_time_available:
        unknown_triggers.add("OBSERVATION_INCOMPLETE")
        unknown_fields.append("boot_time")
    if inputs.pending_reboot == "CONTRADICTORY":
        unknown_triggers.add("WINDOWS_REBOOT_STATE_CONTRADICTORY")
        unknown_fields.append("pending_reboot")
    if not inputs.memory_observation_available:
        unknown_triggers.add("OBSERVATION_INCOMPLETE")
        unknown_fields.append("memory")
    if inputs.collector_status == "UNKNOWN":
        unknown_triggers.add("OBSERVATION_INCOMPLETE")
        unknown_fields.append("collector_status")
    if inputs.materially_stale:
        unknown_triggers.add("OBSERVATION_STALE")
        unknown_fields.append("materially_stale_observation")

    if unknown_triggers:
        primary = _pick_primary(unknown_triggers, _UNKNOWN_PRIORITY_ORDER) or sorted(unknown_triggers)[0]
        return HostHealthEvaluation(
            state="HOST_RESTART_UNKNOWN",
            primary_reason=primary,
            reason_codes=tuple(sorted(unknown_triggers)),
            recommended_action=(
                "Required observations are unavailable, contradictory, or stale. "
                "Do not guess a restart recommendation -- investigate telemetry first."
            ),
            deferred=False,
            unknown_fields=tuple(dict.fromkeys(unknown_fields)),
        )

    severity = _NONE

    # -----------------------------------------------------------------
    # Uptime -- never RED by itself.
    # -----------------------------------------------------------------
    uptime_days = inputs.uptime_days or 0.0
    if uptime_days >= UPTIME_HIGH_DAYS:
        reasons.add("UPTIME_HIGH")
        severity = max(severity, _YELLOW)
    elif uptime_days >= UPTIME_ELEVATED_DAYS:
        reasons.add("UPTIME_ELEVATED")
        severity = max(severity, _YELLOW)
    else:
        reasons.add("UPTIME_NORMAL")

    # -----------------------------------------------------------------
    # Pending reboot.
    # -----------------------------------------------------------------
    if inputs.pending_reboot == "TRUE":
        reasons.add("WINDOWS_REBOOT_PENDING")
        severity = max(severity, _YELLOW)
        if uptime_days >= UPTIME_HIGH_DAYS:
            severity = max(severity, _RED)
    elif inputs.pending_reboot == "FALSE":
        reasons.add("WINDOWS_REBOOT_NOT_PENDING")
    else:
        reasons.add("WINDOWS_REBOOT_STATE_UNKNOWN")
        unknown_fields.append("pending_reboot")

    # -----------------------------------------------------------------
    # RAM.
    # -----------------------------------------------------------------
    ram_sustained = inputs.ram_pct_sustained is not None
    ram_value = inputs.ram_pct_sustained if ram_sustained else inputs.ram_pct_instantaneous
    if ram_value is None:
        unknown_fields.append("ram_pct")
    elif ram_value >= RAM_CRITICAL_PCT:
        reasons.add("RAM_PRESSURE_CRITICAL")
        severity = max(severity, _RED if ram_sustained else _YELLOW)
    elif ram_value >= RAM_ELEVATED_PCT:
        reasons.add("RAM_PRESSURE_ELEVATED")
        severity = max(severity, _YELLOW)
    else:
        reasons.add("RAM_NORMAL")

    # -----------------------------------------------------------------
    # Commit.
    # -----------------------------------------------------------------
    commit_sustained = inputs.commit_pct_sustained is not None
    commit_value = inputs.commit_pct_sustained if commit_sustained else inputs.commit_pct_instantaneous
    if commit_value is None:
        unknown_fields.append("commit_pct")
    elif commit_value >= COMMIT_CRITICAL_PCT:
        reasons.add("COMMIT_PRESSURE_CRITICAL")
        severity = max(severity, _RED if commit_sustained else _YELLOW)
    elif commit_value >= COMMIT_ELEVATED_PCT:
        reasons.add("COMMIT_PRESSURE_ELEVATED")
        severity = max(severity, _YELLOW)
    else:
        reasons.add("COMMIT_PRESSURE_NORMAL")

    # -----------------------------------------------------------------
    # Pagefile.
    # -----------------------------------------------------------------
    if inputs.pagefile_pct is None:
        unknown_fields.append("pagefile_pct")
    elif inputs.pagefile_pct >= PAGEFILE_CRITICAL_PCT:
        reasons.add("PAGEFILE_PRESSURE_CRITICAL")
        severity = max(severity, _YELLOW)
    elif inputs.pagefile_pct >= PAGEFILE_ELEVATED_PCT:
        reasons.add("PAGEFILE_PRESSURE_ELEVATED")
        severity = max(severity, _YELLOW)
    else:
        reasons.add("PAGEFILE_PRESSURE_NORMAL")

    # -----------------------------------------------------------------
    # SSD temperature -- unavailable sensor never forces RED/UNKNOWN.
    # -----------------------------------------------------------------
    if not inputs.ssd_sensor_available or inputs.ssd_temp_c is None:
        reasons.add("SSD_SENSOR_UNKNOWN")
        unknown_fields.append("ssd_temp_c")
    elif inputs.ssd_temp_c >= SSD_TEMP_CRITICAL_C:
        reasons.add("SSD_TEMPERATURE_CRITICAL")
        severity = max(severity, _RED if inputs.ssd_temp_sustained_high else _YELLOW)
    elif inputs.ssd_temp_c >= SSD_TEMP_ELEVATED_C:
        reasons.add("SSD_TEMPERATURE_ELEVATED")
        severity = max(severity, _YELLOW)
    else:
        reasons.add("SSD_TEMPERATURE_NORMAL")

    # -----------------------------------------------------------------
    # Collector / critical-process health.
    # -----------------------------------------------------------------
    if inputs.collector_status == "FAILED":
        reasons.add("COLLECTORS_FAILED")
        severity = max(severity, _RED if inputs.repeated_collector_failure else _YELLOW)
    elif inputs.collector_status == "DEGRADED":
        reasons.add("COLLECTORS_DEGRADED")
        severity = max(severity, _YELLOW)
    elif inputs.collector_status == "HEALTHY":
        reasons.add("COLLECTORS_HEALTHY")
    else:
        reasons.add("COLLECTORS_UNKNOWN")
        unknown_fields.append("collector_status")

    # -----------------------------------------------------------------
    # Storage state.
    # -----------------------------------------------------------------
    if inputs.storage_state == "STORAGE_EMERGENCY":
        reasons.add("STORAGE_EMERGENCY")
        severity = max(severity, _RED)
    elif inputs.storage_state == "STORAGE_CRITICAL":
        reasons.add("STORAGE_CRITICAL")
        severity = max(severity, _RED)
    elif inputs.storage_state == "STORAGE_WARNING":
        reasons.add("STORAGE_WARNING")
        severity = max(severity, _YELLOW)
    elif inputs.storage_state == "STORAGE_HEALTHY":
        reasons.add("STORAGE_HEALTHY")
    else:
        reasons.add("STORAGE_STATE_UNKNOWN")
        unknown_fields.append("storage_state")

    # -----------------------------------------------------------------
    # Event-log-derived counts.
    # -----------------------------------------------------------------
    if inputs.recent_whea_critical_count_24h > 0:
        reasons.add("WHEA_EVENT_CRITICAL")
        severity = max(severity, _RED)
    if inputs.recent_disk_ntfs_critical_count_24h > 0:
        reasons.add("DISK_EVENT_CRITICAL")
        severity = max(severity, _RED)
    if inputs.recent_oom_event_count_24h >= OOM_EVENT_RED_THRESHOLD_24H:
        reasons.add("RESOURCE_EXHAUSTION_EVENTS_CRITICAL")
        severity = max(severity, _RED)
    if inputs.recent_unexpected_shutdown_count_24h >= UNEXPECTED_SHUTDOWN_RED_THRESHOLD_24H:
        reasons.add("RECENT_UNEXPECTED_SHUTDOWN")
        severity = max(severity, _RED)
    elif inputs.recent_unexpected_shutdown_count_24h >= UNEXPECTED_SHUTDOWN_YELLOW_THRESHOLD_24H:
        reasons.add("RECENT_UNEXPECTED_SHUTDOWN")
        severity = max(severity, _YELLOW)
    if inputs.recent_app_crash_count_24h >= APP_CRASH_YELLOW_THRESHOLD_24H:
        reasons.add("RECENT_APP_CRASHES")
        severity = max(severity, _YELLOW)
    if inputs.event_log_access == "ACCESS_DENIED":
        reasons.add("EVENT_LOG_ACCESS_DENIED")
        unknown_fields.append("event_log")

    # -----------------------------------------------------------------
    # Final state + recommended action.
    # -----------------------------------------------------------------
    if severity >= _RED:
        state = "HOST_RESTART_RED"
        base_action = (
            "Complete or safely stop critical operations, then perform a controlled Windows restart."
        )
        order = _PRIORITY_ORDER
    elif severity >= _YELLOW:
        state = "HOST_RESTART_YELLOW"
        base_action = "Restart in the next safe maintenance window, preferably within 24 hours."
        order = _PRIORITY_ORDER
    else:
        state = "HOST_RESTART_GREEN"
        base_action = "No restart needed. Continue normal work."
        order = _PRIORITY_ORDER

    primary = _pick_primary(reasons, order) or "UPTIME_NORMAL"

    deferred = state in ("HOST_RESTART_YELLOW", "HOST_RESTART_RED") and inputs.critical_operation_active
    if deferred:
        reasons.add("CRITICAL_OPERATION_ACTIVE")
        recommended_action = (
            "RESTART_RECOMMENDED_BUT_DEFER_UNTIL_SAFE_CHECKPOINT: " + base_action +
            " Confirm no critical batch, archive publication, database maintenance, or "
            "collector-sensitive operation is active before proceeding."
        )
    else:
        recommended_action = base_action

    return HostHealthEvaluation(
        state=state,
        primary_reason=primary,
        reason_codes=tuple(sorted(reasons)),
        recommended_action=recommended_action,
        deferred=deferred,
        unknown_fields=tuple(dict.fromkeys(unknown_fields)),
    )
