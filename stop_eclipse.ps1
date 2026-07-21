param(
    [switch]$Quiet,
    [switch]$LiquidationSilenceSchedulerOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path $PSScriptRoot).Path
Set-Location -LiteralPath $RepoRoot

# Scoped stop: targets ONLY the liquidation-silence scheduler wrapper, never
# any other role. Short-circuits before the full-stack pattern-match/kill
# loop below so this switch can never touch an unrelated Python process.
if ($LiquidationSilenceSchedulerOnly) {
    $liqSchedStopped = @()
    try {
        $rows = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue)
        foreach ($row in $rows) {
            $cmd = [string]$row.CommandLine
            if ($cmd -and $cmd.Contains("tools.liquidation_silence_scheduler")) {
                try {
                    Stop-Process -Id ([int]$row.ProcessId) -Force -ErrorAction SilentlyContinue
                    $liqSchedStopped += [int]$row.ProcessId
                } catch {
                }
            }
        }
    } catch {
    }
    $liqSchedPidFile = Join-Path $RepoRoot "logs\pids\liquidation_silence_scheduler.pid"
    if (Test-Path -LiteralPath $liqSchedPidFile) {
        try { Remove-Item -LiteralPath $liqSchedPidFile -Force -ErrorAction SilentlyContinue } catch {}
    }
    if (-not $Quiet) {
        if ($liqSchedStopped.Count -gt 0) {
            Write-Output ("STOPPED_LIQUIDATION_SILENCE_SCHEDULER pids=" + (($liqSchedStopped | Sort-Object -Unique) -join ","))
        } else {
            Write-Output "STOPPED_LIQUIDATION_SILENCE_SCHEDULER none_running"
        }
    }
    return
}

$patterns = @(
    "scripts\collector_supervisor.py",
    "scripts/collector_supervisor.py",
    "data.microstructure_collector",
    "data.bookticker_collector",
    "data\bookticker_collector.py",
    "data/bookticker_collector.py",
    "data.event_diary",
    "tools.heartbeat_watchdog",
    "tools\s34_shadow_paper_runner.py",
    "tools/s34_shadow_paper_runner.py",
    "tools\s34_live_chart.py",
    "tools/s34_live_chart.py",
    "tools\orderflow_chart.py",
    "tools/orderflow_chart.py",
    "tools\s34_replay.py",
    "tools/s34_replay.py",
    "tools.s34_v_engine_v02_shadow_mirror",
    "tools.collection_watchdog",
    "tools.data_layer_probe",
    "tools.s34_live_order_executor",
    "tools.s34_v_engine_live_executor",
    "tools.s34_state_machine_live_executor",
    "tools.s34_realtime_shadow_runner",
    "data.oi_spot_poller",
    "tools.liquidation_silence_scheduler",
    "tools.s34_leads_monitor_dashboard",
    "tools.research_s34_hold_horizon_forward_ledger"
)

$stopped = @()
try {
    $rows = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue)
    foreach ($row in $rows) {
        $cmd = [string]$row.CommandLine
        if (-not $cmd) { continue }
        $match = $false
        foreach ($p in $patterns) {
            if ($cmd.Contains($p)) {
                $match = $true
                break
            }
        }
        if ($match) {
            try {
                Stop-Process -Id ([int]$row.ProcessId) -Force -ErrorAction SilentlyContinue
                $stopped += [int]$row.ProcessId
            } catch {
            }
        }
    }
} catch {
}

$pidDir = Join-Path $RepoRoot "logs\pids"
if (Test-Path -LiteralPath $pidDir) {
    Get-ChildItem -LiteralPath $pidDir -Filter "*.pid" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in @("collector_supervisor.pid", "heartbeat_watchdog.pid", "microstructure_collector.pid", "bookticker_collector.pid", "event_diary.pid", "s34_shadow_paper_runner.pid", "s34_state_machine_shadow_runner.pid", "s34_live_chart.pid", "s34_v_engine_v02_shadow_mirror.pid", "s34_live_order_executor.pid", "s34_v_engine_live_executor.pid", "s34_state_machine_live_executor.pid", "oi_spot_poller.pid", "liquidation_silence_scheduler.pid") } |
        ForEach-Object {
            try {
                $raw = (Get-Content -LiteralPath $_.FullName -Raw -ErrorAction Stop).Trim()
                if ($raw) {
                    $pidValue = [int]$raw
                    Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
                    $stopped += $pidValue
                }
            } catch {}
        }
    Get-ChildItem -LiteralPath $pidDir -Filter "*.pid" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in @("collector_supervisor.pid", "heartbeat_watchdog.pid", "microstructure_collector.pid", "bookticker_collector.pid", "event_diary.pid", "s34_shadow_paper_runner.pid", "s34_state_machine_shadow_runner.pid", "s34_live_chart.pid", "s34_v_engine_v02_shadow_mirror.pid", "s34_live_order_executor.pid", "s34_v_engine_live_executor.pid", "s34_state_machine_live_executor.pid", "oi_spot_poller.pid", "liquidation_silence_scheduler.pid") } |
        ForEach-Object {
            try { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue } catch {}
        }
}

if (-not $Quiet) {
    if ($stopped.Count -gt 0) {
        Write-Output ("STOPPED pids=" + (($stopped | Sort-Object -Unique) -join ","))
    } else {
        Write-Output "STOPPED none_running"
    }
}
