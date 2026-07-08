"""BATCH-OPERATOR-HOST-HEALTH-AND-RESTART-READINESS-DASHBOARD-V1.

Read-only Windows host observation collector. Every collection function
here is fail-closed: on any error, timeout, access-denied, or missing
sensor it records UNKNOWN/None for that specific field rather than
fabricating a value or silently defaulting to "healthy". Nothing in this
module writes to the registry, starts/stops a process, changes a
scheduled task, installs a package, or calls any Windows restart/shutdown
API. The only subprocess this module ever launches is a single read-only
`powershell.exe` invocation per `collect_host_observation()` call, and a
`Get-CimInstance`/`Get-WinEvent`/`Get-PhysicalDisk`-only script at that
(no `Set-*`, `Remove-*`, `Stop-*`, `Restart-*`, or `New-*` cmdlet appears
anywhere in `_POWERSHELL_SCRIPT`).
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a repo dependency (requirements.txt)
    psutil = None

from ami.host_health.evaluator import D_DRIVE_INTERVENTION_FREE_GB, HostHealthInputs

POWERSHELL_TIMEOUT_SEC = 25

# Read-only PowerShell: Get-CimInstance / Test-Path / Get-ItemProperty /
# Get-PhysicalDisk / Get-StorageReliabilityCounter / Get-WinEvent only.
# No Set-*, Remove-*, Stop-*, Restart-*, New-*, or shutdown/reboot verb
# appears in this script.
_POWERSHELL_SCRIPT = r"""
$ErrorActionPreference = 'Continue'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$OutputEncoding = [System.Text.Encoding]::UTF8
$result = @{}

try {
    $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
    $result.total_virtual_memory_kb = [int64]$os.TotalVirtualMemorySize
    $result.free_virtual_memory_kb = [int64]$os.FreeVirtualMemory
} catch {
    $result.os_error = $_.Exception.Message
}

$pendingEvidence = @{}
try { $pendingEvidence.cbs_reboot_pending = [bool](Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending') } catch { $pendingEvidence.cbs_reboot_pending = $null }
try { $pendingEvidence.wu_reboot_required = [bool](Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired') } catch { $pendingEvidence.wu_reboot_required = $null }
try {
    $pfro = Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager' -Name PendingFileRenameOperations -ErrorAction Stop
    $pendingEvidence.pending_file_rename_operations = [bool]($null -ne $pfro.PendingFileRenameOperations -and $pfro.PendingFileRenameOperations.Count -gt 0)
} catch {
    $pendingEvidence.pending_file_rename_operations = $false
}
$result.pending_reboot_evidence = $pendingEvidence

$disks = @()
try {
    $phys = @(Get-PhysicalDisk -ErrorAction Stop)
    foreach ($d in $phys) {
        $entry = @{ friendly_name = [string]$d.FriendlyName; media_type = [string]$d.MediaType; health_status = [string]$d.HealthStatus }
        try {
            $rc = $d | Get-StorageReliabilityCounter -ErrorAction Stop
            $entry.temperature_c = $rc.Temperature
            $entry.wear_pct = $rc.Wear
            $entry.read_error_total = $rc.ReadErrorsTotal
            $entry.write_error_total = $rc.WriteErrorsTotal
        } catch {
            $entry.reliability_counter_error = $_.Exception.Message
        }
        $disks += $entry
    }
} catch {
    $result.physical_disk_error = $_.Exception.Message
}
$result.physical_disks = $disks

function Get-BoundedEvents($LogName, $Ids, $ProviderNames, $Hours) {
    try {
        $start = (Get-Date).ToUniversalTime().AddHours(-1 * $Hours)
        $filter = @{ LogName = $LogName; StartTime = $start }
        if ($Ids) { $filter.Id = $Ids }
        if ($ProviderNames) { $filter.ProviderName = $ProviderNames }
        $events = @(Get-WinEvent -FilterHashtable $filter -MaxEvents 500 -ErrorAction Stop)
        $out = @()
        foreach ($e in $events) {
            $out += @{ id = $e.Id; time_utc = $e.TimeCreated.ToUniversalTime().ToString('o'); level = $e.LevelDisplayName }
        }
        return ,$out
    } catch {
        if ($_.Exception.Message -match 'No events were found') { return ,@() }
        return $null
    }
}

$eventWindowHours = 168
$events = @{}
$events.unexpected_shutdown = Get-BoundedEvents 'System' @(41, 6008) $null $eventWindowHours
$events.whea = Get-BoundedEvents 'System' $null @('Microsoft-Windows-WHEA-Logger') $eventWindowHours
$events.disk_ntfs = Get-BoundedEvents 'System' $null @('disk', 'Ntfs') $eventWindowHours
$events.app_crash = Get-BoundedEvents 'Application' @(1000) @('Application Error') $eventWindowHours
$events.resource_exhaustion = Get-BoundedEvents 'System' @(2004, 2005) $null $eventWindowHours
$result.events = $events

$criticalKeywords = @('purge', 'vacuum', 'archive_export', 'rehearsal', 'production_activation', 'rotation_retention_apply', 'migration')
$criticalHit = $false
$criticalMatches = @()
try {
    $procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction Stop)
    foreach ($p in $procs) {
        $cmd = [string]$p.CommandLine
        if (-not $cmd) { continue }
        $lower = $cmd.ToLower()
        foreach ($kw in $criticalKeywords) {
            if ($lower.Contains($kw)) {
                $criticalHit = $true
                $criticalMatches += $kw
            }
        }
    }
} catch {
}
$result.critical_operation_active = $criticalHit
$result.critical_operation_matches = $criticalMatches

$result | ConvertTo-Json -Depth 6 -Compress
"""


def _run_powershell(script: str, timeout_sec: int = POWERSHELL_TIMEOUT_SEC) -> dict[str, Any] | None:
    """Runs a single read-only PowerShell script and parses its JSON
    stdout. Returns None (fail-closed) on any error, timeout, or
    non-JSON output -- callers must treat None as UNKNOWN for every
    field the script would have populated."""
    if sys.platform != "win32":
        return None
    try:
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            capture_output=True, timeout=timeout_sec,
        )
        stdout = proc.stdout.decode("utf-8", errors="replace")
        if proc.returncode != 0 or not stdout.strip():
            return None
        return json.loads(stdout)
    except Exception:
        return None


def _event_bucket_counts(events: list[dict] | None, now_utc: datetime) -> dict[str, int | None]:
    """Given a bounded 7-day event list (or None for error/access-denied),
    returns 24h and 7d counts. None input => (None, None) i.e. UNKNOWN,
    never 0 (0 means "queried successfully, found nothing")."""
    if events is None:
        return {"count_24h": None, "count_7d": None}
    count_24h = 0
    count_7d = 0
    for e in events:
        try:
            ts = datetime.fromisoformat(str(e.get("time_utc")).replace("Z", "+00:00"))
        except Exception:
            continue
        age_h = (now_utc - ts).total_seconds() / 3600.0
        if age_h <= 168:
            count_7d += 1
        if age_h <= 24:
            count_24h += 1
    return {"count_24h": count_24h, "count_7d": count_7d}


@dataclass(frozen=True)
class HostObservation:
    observation_ts_utc: str
    host_name: str
    os_identity: str

    boot_ts_utc: str | None
    uptime_seconds: float | None
    uptime_human: str | None

    pending_reboot: str  # TRUE | FALSE | UNKNOWN | CONTRADICTORY
    pending_reboot_evidence: dict[str, Any]

    ram_total_bytes: int | None
    ram_available_bytes: int | None
    ram_used_pct: float | None

    commit_limit_kb: int | None
    commit_used_kb: int | None
    commit_used_pct: float | None

    pagefile_total_bytes: int | None
    pagefile_used_bytes: int | None
    pagefile_used_pct: float | None

    cpu_pct_snapshot: float | None
    cpu_pct_recent_avg: float | None

    c_drive_total_bytes: int | None
    c_drive_free_bytes: int | None
    d_drive_total_bytes: int | None
    d_drive_free_bytes: int | None
    d_drive_free_gb: float | None
    d_drive_distance_to_threshold_gb: float | None

    microstructure_db_size_bytes: int | None
    microstructure_wal_size_bytes: int | None

    storage_health_state: str  # STORAGE_HEALTHY | WARNING | CRITICAL | EMERGENCY | STORAGE_STATE_UNKNOWN

    collector_status: str  # HEALTHY | DEGRADED | FAILED | UNKNOWN
    collector_status_detail: dict[str, Any]
    latest_collector_heartbeat_age_sec: float | None

    physical_disks: tuple[dict[str, Any], ...]
    ssd_990pro_detected: bool
    ssd_temp_c: float | None
    ssd_health_state: str  # HEALTHY | DEGRADED | UNKNOWN

    recent_unexpected_shutdown_count_24h: int | None
    recent_unexpected_shutdown_count_7d: int | None
    recent_app_crash_count_24h: int | None
    recent_disk_ntfs_critical_count_24h: int | None
    recent_whea_critical_count_24h: int | None
    recent_oom_event_count_24h: int | None
    event_log_access: str  # OK | ACCESS_DENIED | UNKNOWN

    critical_operation_active: bool
    critical_operation_evidence: tuple[str, ...]

    observation_errors: tuple[str, ...]
    stale_fields: tuple[str, ...]
    unknown_fields: tuple[str, ...]

    powershell_available: bool


def _fmt_uptime(seconds: float) -> str:
    days, rem = divmod(int(seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


def _collect_pids_health(repo_root: Path) -> tuple[str, dict[str, Any], float | None]:
    """Read-only reduction over existing health/heartbeat files
    (logs/health/overall.json, logs/collector_heartbeat.json) --
    mirrors the convention already used by tools/health_check.py and
    dashboard/backend/data_sources.py::_health_overall_stats(). Never
    infers HEALTHY from process existence alone when heartbeat evidence
    exists."""
    detail: dict[str, Any] = {}
    heartbeat_age: float | None = None

    overall_path = repo_root / "logs" / "health" / "overall.json"
    hb_path = repo_root / "logs" / "collector_heartbeat.json"

    status = "UNKNOWN"
    try:
        if overall_path.exists():
            payload = json.loads(overall_path.read_text(encoding="utf-8", errors="replace"))
            ts = payload.get("ts_utc")
            age = None
            if ts:
                try:
                    parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    age = (datetime.now(tz=timezone.utc) - parsed).total_seconds()
                except Exception:
                    age = None
            detail["overall_ts_utc"] = ts
            detail["overall_age_sec"] = age
            state = str(payload.get("state") or "").lower()
            if age is not None and age > 120:
                status = "UNKNOWN"
                detail["reason"] = "overall_health_stale"
            elif state == "ok":
                status = "HEALTHY"
            elif state == "degraded":
                status = "DEGRADED"
            elif state == "halted":
                status = "FAILED"
    except Exception as exc:
        detail["overall_error"] = str(exc)

    try:
        if hb_path.exists():
            payload = json.loads(hb_path.read_text(encoding="utf-8", errors="replace"))
            ts = payload.get("last_message_ts_utc") or payload.get("last_data_progress_ts_utc")
            if ts:
                parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                heartbeat_age = (datetime.now(tz=timezone.utc) - parsed).total_seconds()
            detail["heartbeat_connected"] = payload.get("connected")
            detail["heartbeat_last_error"] = payload.get("last_error")
            if heartbeat_age is not None and heartbeat_age > 420 and status == "UNKNOWN":
                status = "DEGRADED"
                detail.setdefault("reason", "heartbeat_stale")
    except Exception as exc:
        detail["heartbeat_error"] = str(exc)

    return status, detail, heartbeat_age


def _storage_health_state(d_drive_free_bytes: int | None, d_drive_total_bytes: int | None) -> str:
    if d_drive_free_bytes is None or d_drive_total_bytes is None or d_drive_total_bytes <= 0:
        return "STORAGE_STATE_UNKNOWN"
    try:
        from ami.governance.storage_rotation_retention_readiness_v1 import storage_health_state
        pct_free = (d_drive_free_bytes / d_drive_total_bytes) * 100.0
        abs_free_gb = d_drive_free_bytes / (1024.0 ** 3)
        return storage_health_state(pct_free, abs_free_gb)
    except Exception:
        return "STORAGE_STATE_UNKNOWN"


def collect_host_observation(
    repo_root: Path | None = None,
    ram_pct_history: list[tuple[float, float]] | None = None,
) -> HostObservation:
    """Collects one point-in-time, best-effort, fail-closed observation.
    `ram_pct_history` is an optional list of (unix_ts, ram_pct) samples
    the caller may maintain across polls (e.g. the dashboard backend) --
    this function does not persist any history itself. Any field this
    function cannot determine is set to None/UNKNOWN and listed in
    `unknown_fields`; nothing is fabricated."""
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    now = datetime.now(tz=timezone.utc)
    errors: list[str] = []
    unknown: list[str] = []
    stale: list[str] = []

    # --- psutil-backed fields (no subprocess) ---------------------------
    boot_ts_utc = None
    uptime_seconds = None
    uptime_human = None
    ram_total = ram_avail = None
    ram_pct = None
    pagefile_total = pagefile_used = None
    pagefile_pct = None
    cpu_snapshot = None
    c_total = c_free = None
    d_total = d_free = None

    if psutil is not None:
        try:
            boot_epoch = psutil.boot_time()
            boot_ts_utc = datetime.fromtimestamp(boot_epoch, tz=timezone.utc).isoformat()
            uptime_seconds = max(0.0, time.time() - boot_epoch)
            uptime_human = _fmt_uptime(uptime_seconds)
        except Exception as exc:
            errors.append(f"boot_time: {exc}")
            unknown.append("boot_time")

        try:
            vm = psutil.virtual_memory()
            ram_total, ram_avail, ram_pct = int(vm.total), int(vm.available), float(vm.percent)
        except Exception as exc:
            errors.append(f"virtual_memory: {exc}")
            unknown.append("ram")

        try:
            sw = psutil.swap_memory()
            pagefile_total, pagefile_used = int(sw.total), int(sw.used)
            pagefile_pct = float(sw.percent)
        except Exception as exc:
            errors.append(f"swap_memory: {exc}")
            unknown.append("pagefile")

        try:
            cpu_snapshot = float(psutil.cpu_percent(interval=0.2))
        except Exception as exc:
            errors.append(f"cpu_percent: {exc}")
            unknown.append("cpu")

        try:
            du_c = psutil.disk_usage("C:\\")
            c_total, c_free = int(du_c.total), int(du_c.free)
        except Exception as exc:
            errors.append(f"disk_usage(C:): {exc}")
            unknown.append("c_drive")

        try:
            du_d = psutil.disk_usage("D:\\")
            d_total, d_free = int(du_d.total), int(du_d.free)
        except Exception as exc:
            errors.append(f"disk_usage(D:): {exc}")
            unknown.append("d_drive")
    else:
        errors.append("psutil not importable")
        unknown.extend(["boot_time", "ram", "pagefile", "cpu", "c_drive", "d_drive"])

    d_free_gb = round(d_free / (1024.0 ** 3), 2) if d_free is not None else None
    d_distance = round(d_free_gb - D_DRIVE_INTERVENTION_FREE_GB, 2) if d_free_gb is not None else None

    # --- single consolidated PowerShell call -----------------------------
    ps = _run_powershell(_POWERSHELL_SCRIPT)
    powershell_available = ps is not None
    if ps is None:
        errors.append("powershell_observation_unavailable")

    commit_limit_kb = commit_used_kb = None
    commit_pct = None
    pending_reboot = "UNKNOWN"
    pending_evidence: dict[str, Any] = {}
    physical_disks: tuple[dict[str, Any], ...] = ()
    ssd_detected = False
    ssd_temp = None
    ssd_health = "UNKNOWN"
    ev_unexpected_24h = ev_unexpected_7d = None
    ev_app_crash_24h = None
    ev_disk_ntfs_24h = None
    ev_whea_24h = None
    ev_oom_24h = None
    event_log_access = "UNKNOWN"
    critical_op_active = False
    critical_op_evidence: tuple[str, ...] = ()

    if ps is not None:
        if "total_virtual_memory_kb" in ps and "free_virtual_memory_kb" in ps:
            commit_limit_kb = int(ps["total_virtual_memory_kb"])
            free_kb = int(ps["free_virtual_memory_kb"])
            commit_used_kb = commit_limit_kb - free_kb
            commit_pct = round((commit_used_kb / commit_limit_kb) * 100.0, 2) if commit_limit_kb > 0 else None
        else:
            unknown.append("commit")

        pending_evidence = ps.get("pending_reboot_evidence") or {}
        votes = [v for v in pending_evidence.values() if isinstance(v, bool)]
        if not votes:
            pending_reboot = "UNKNOWN"
            unknown.append("pending_reboot")
        elif all(v is False for v in votes):
            pending_reboot = "FALSE"
        elif any(v is True for v in votes) and not all(v is True for v in votes) and len(votes) > 1:
            # some sources say pending, others don't -- treated as pending
            # (fail-closed toward the more actionable signal), not
            # contradictory. CONTRADICTORY is reserved for genuinely
            # irreconcilable direct evidence (kept as an explicit state
            # for future stricter sources; not reachable via this
            # registry-only evidence set today).
            pending_reboot = "TRUE"
        elif any(v is True for v in votes):
            pending_reboot = "TRUE"
        else:
            pending_reboot = "FALSE"

        physical_disks = tuple(ps.get("physical_disks") or [])
        for d in physical_disks:
            fname = str(d.get("friendly_name") or "")
            if "990" in fname and "pro" in fname.lower():
                ssd_detected = True
                temp = d.get("temperature_c")
                if isinstance(temp, (int, float)):
                    ssd_temp = float(temp)
                health = str(d.get("health_status") or "").lower()
                if health == "healthy":
                    ssd_health = "HEALTHY"
                elif health:
                    ssd_health = "DEGRADED"
        if not physical_disks:
            unknown.append("physical_disks")

        events = ps.get("events") or {}
        b = _event_bucket_counts(events.get("unexpected_shutdown"), now)
        ev_unexpected_24h, ev_unexpected_7d = b["count_24h"], b["count_7d"]
        ev_app_crash_24h = _event_bucket_counts(events.get("app_crash"), now)["count_24h"]
        ev_disk_ntfs_24h = _event_bucket_counts(events.get("disk_ntfs"), now)["count_24h"]
        ev_whea_24h = _event_bucket_counts(events.get("whea"), now)["count_24h"]
        ev_oom_24h = _event_bucket_counts(events.get("resource_exhaustion"), now)["count_24h"]
        any_event_unknown = any(
            events.get(k) is None for k in ("unexpected_shutdown", "app_crash", "disk_ntfs", "whea", "resource_exhaustion")
        )
        event_log_access = "ACCESS_DENIED" if any_event_unknown else "OK"
        if any_event_unknown:
            unknown.append("event_log")

        critical_op_active = bool(ps.get("critical_operation_active"))
        critical_op_evidence = tuple(ps.get("critical_operation_matches") or [])
    else:
        unknown.extend(["commit", "pending_reboot", "physical_disks", "event_log"])

    if ssd_temp is None:
        unknown.append("ssd_temp_c")

    # --- repository-local, read-only, no-subprocess observations --------
    db_size = wal_size = None
    try:
        db_path = repo_root / "data" / "microstructure.db"
        if db_path.exists():
            db_size = db_path.stat().st_size
        wal_path = repo_root / "data" / "microstructure.db-wal"
        if wal_path.exists():
            wal_size = wal_path.stat().st_size
    except Exception as exc:
        errors.append(f"microstructure_db_stat: {exc}")
        unknown.append("microstructure_db")

    collector_status, collector_detail, hb_age = _collect_pids_health(repo_root)
    storage_state = _storage_health_state(d_free, d_total)

    observation = HostObservation(
        observation_ts_utc=now.isoformat(),
        host_name=os.environ.get("COMPUTERNAME", "UNKNOWN"),
        os_identity=sys.platform,
        boot_ts_utc=boot_ts_utc,
        uptime_seconds=uptime_seconds,
        uptime_human=uptime_human,
        pending_reboot=pending_reboot,
        pending_reboot_evidence=pending_evidence,
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_avail,
        ram_used_pct=ram_pct,
        commit_limit_kb=commit_limit_kb,
        commit_used_kb=commit_used_kb,
        commit_used_pct=commit_pct,
        pagefile_total_bytes=pagefile_total,
        pagefile_used_bytes=pagefile_used,
        pagefile_used_pct=pagefile_pct,
        cpu_pct_snapshot=cpu_snapshot,
        cpu_pct_recent_avg=None,
        c_drive_total_bytes=c_total,
        c_drive_free_bytes=c_free,
        d_drive_total_bytes=d_total,
        d_drive_free_bytes=d_free,
        d_drive_free_gb=d_free_gb,
        d_drive_distance_to_threshold_gb=d_distance,
        microstructure_db_size_bytes=db_size,
        microstructure_wal_size_bytes=wal_size,
        storage_health_state=storage_state,
        collector_status=collector_status,
        collector_status_detail=collector_detail,
        latest_collector_heartbeat_age_sec=hb_age,
        physical_disks=physical_disks,
        ssd_990pro_detected=ssd_detected,
        ssd_temp_c=ssd_temp,
        ssd_health_state=ssd_health,
        recent_unexpected_shutdown_count_24h=ev_unexpected_24h,
        recent_unexpected_shutdown_count_7d=ev_unexpected_7d,
        recent_app_crash_count_24h=ev_app_crash_24h,
        recent_disk_ntfs_critical_count_24h=ev_disk_ntfs_24h,
        recent_whea_critical_count_24h=ev_whea_24h,
        recent_oom_event_count_24h=ev_oom_24h,
        event_log_access=event_log_access,
        critical_operation_active=critical_op_active,
        critical_operation_evidence=critical_op_evidence,
        observation_errors=tuple(errors),
        stale_fields=tuple(stale),
        unknown_fields=tuple(dict.fromkeys(unknown)),
        powershell_available=powershell_available,
    )
    return observation


def build_health_inputs(
    obs: HostObservation,
    *,
    ram_pct_sustained: float | None = None,
    commit_pct_sustained: float | None = None,
    repeated_collector_failure: bool = False,
    ssd_temp_sustained_high: bool = False,
    materially_stale: bool = False,
) -> HostHealthInputs:
    """Reduces a `HostObservation` (plus optional caller-maintained
    sustained-window values -- see `sustained_value()`) to the plain
    `HostHealthInputs` the pure evaluator consumes. This is a data
    mapping only; it performs no I/O itself."""
    max_staleness_sec = 120.0
    stale = materially_stale or bool(obs.observation_ts_utc) is False
    try:
        obs_ts = datetime.fromisoformat(obs.observation_ts_utc)
        age = (datetime.now(tz=timezone.utc) - obs_ts).total_seconds()
        if age > max_staleness_sec:
            stale = True
    except Exception:
        stale = True

    return HostHealthInputs(
        boot_time_available=obs.boot_ts_utc is not None and obs.uptime_seconds is not None,
        uptime_days=(obs.uptime_seconds / 86400.0) if obs.uptime_seconds is not None else None,
        pending_reboot=obs.pending_reboot,
        memory_observation_available=obs.ram_used_pct is not None,
        ram_pct_instantaneous=obs.ram_used_pct,
        ram_pct_sustained=ram_pct_sustained,
        commit_pct_instantaneous=obs.commit_used_pct,
        commit_pct_sustained=commit_pct_sustained,
        pagefile_pct=obs.pagefile_used_pct,
        ssd_sensor_available=obs.ssd_temp_c is not None,
        ssd_temp_c=obs.ssd_temp_c,
        ssd_temp_sustained_high=ssd_temp_sustained_high,
        collector_status=obs.collector_status,
        repeated_collector_failure=repeated_collector_failure,
        storage_state=obs.storage_health_state,
        critical_operation_active=obs.critical_operation_active,
        recent_unexpected_shutdown_count_24h=obs.recent_unexpected_shutdown_count_24h or 0,
        recent_disk_ntfs_critical_count_24h=obs.recent_disk_ntfs_critical_count_24h or 0,
        recent_whea_critical_count_24h=obs.recent_whea_critical_count_24h or 0,
        recent_app_crash_count_24h=obs.recent_app_crash_count_24h or 0,
        recent_oom_event_count_24h=obs.recent_oom_event_count_24h or 0,
        event_log_access=obs.event_log_access,
        materially_stale=stale,
    )


def sustained_value(history: list[tuple[float, float]], window_minutes: float, now_ts: float | None = None) -> float | None:
    """Pure helper: given a list of (unix_ts, value) samples the caller
    maintains (e.g. the dashboard backend's own poll history -- this
    function starts no background service and stores nothing itself),
    returns the minimum value observed within the trailing window if at
    least two samples span it, else None (meaning: treat the latest
    reading as instantaneous only, per Phase 4)."""
    if not history:
        return None
    now_ts = now_ts if now_ts is not None else time.time()
    window_start = now_ts - (window_minutes * 60.0)
    in_window = [v for ts, v in history if ts >= window_start]
    if len(in_window) < 2:
        return None
    earliest_ts = min(ts for ts, _ in history if ts >= window_start)
    if earliest_ts > window_start + 60.0:
        # history doesn't actually cover the full window yet
        return None
    return min(in_window)
