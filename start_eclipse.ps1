param(
    [string]$Symbols = "BTCUSDT,ETHUSDT,SOLUSDT",
    [int]$StatusWaitSec = 20,
    [switch]$NoCleanStop,
    [switch]$EnableLive,
    [switch]$EnableLiquidationSilenceScheduler
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path $PSScriptRoot).Path
Set-Location -LiteralPath $RepoRoot

$processPath = [Environment]::GetEnvironmentVariable("Path", "Process")
if (-not $processPath) {
    $processPath = [Environment]::GetEnvironmentVariable("PATH", "Process")
}
if ($processPath) {
    [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
    [Environment]::SetEnvironmentVariable("Path", $processPath, "Process")
}

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Pick-Python {
    $local = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    $parent = Join-Path (Split-Path -Parent $RepoRoot) ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $local) { return $local }
    if (Test-Path -LiteralPath $parent) { return $parent }
    return "python"
}

function Test-PidRunning([int]$Pid) {
    if ($Pid -le 0) { return $false }
    try {
        Get-Process -Id $Pid -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Find-PythonProcessByCommandLine([string]$Needle) {
    if (-not $Needle) { return $null }
    try {
        $rows = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction Stop)
        foreach ($row in $rows) {
            $cmd = [string]$row.CommandLine
            if ($cmd -and $cmd.Contains($Needle)) {
                return [int]$row.ProcessId
            }
        }
    } catch {
    }
    return $null
}

function Find-PythonProcessesByCommandLine([string]$Needle) {
    $matches = @()
    if (-not $Needle) { return $matches }
    try {
        $rows = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction Stop)
        foreach ($row in $rows) {
            $cmd = [string]$row.CommandLine
            if ($cmd -and $cmd.Contains($Needle)) {
                $matches += [PSCustomObject]@{
                    ProcessId = [int]$row.ProcessId
                    CommandLine = $cmd
                }
            }
        }
    } catch {
    }
    return $matches
}

function Stop-DuplicatePythonProcesses([string]$Role, [string]$Needle, [Nullable[int]]$PreferredPid) {
    if (-not $Needle) { return $PreferredPid }
    $matches = @(Find-PythonProcessesByCommandLine $Needle | Sort-Object ProcessId)
    if ($matches.Count -le 1) {
        if ($PreferredPid) { return [int]$PreferredPid }
        if ($matches.Count -eq 1) { return [int]$matches[0].ProcessId }
        return $null
    }

    $keepPid = $null
    if ($PreferredPid) {
        foreach ($m in $matches) {
            if ([int]$m.ProcessId -eq [int]$PreferredPid) {
                $keepPid = [int]$PreferredPid
                break
            }
        }
    }
    if (-not $keepPid) {
        # Prefer the newest process id so a fresh start_eclipse run replaces older strays.
        $keepPid = [int]($matches | Sort-Object ProcessId -Descending | Select-Object -First 1).ProcessId
    }

    foreach ($m in $matches) {
        $pidValue = [int]$m.ProcessId
        if ($pidValue -eq $keepPid) { continue }
        try {
            Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
            Write-Output "STOPPED_DUPLICATE ${Role}_pid=$pidValue keep_pid=$keepPid"
        } catch {
        }
    }
    return $keepPid
}

function Stop-PythonProcessesByCommandLine([string]$Role, [string]$Needle) {
    if (-not $Needle) { return }
    $matches = @(Find-PythonProcessesByCommandLine $Needle | Sort-Object ProcessId)
    foreach ($m in $matches) {
        $pidValue = [int]$m.ProcessId
        try {
            Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
            Write-Output "STOPPED_DISABLED ${Role}_pid=$pidValue"
        } catch {
        }
    }
}

function Get-LivePidFromFile([string]$PidFile) {
    if (-not (Test-Path -LiteralPath $PidFile)) { return $null }
    try {
        $raw = (Get-Content -LiteralPath $PidFile -ErrorAction Stop | Select-Object -First 1).Trim()
        if (-not $raw) { return $null }
        $pidValue = [int]$raw
        if (Test-PidRunning $pidValue) { return $pidValue }
    } catch {
    }
    return $null
}

function Start-RegisteredPythonProcess(
    [string]$Role,
    [array]$ProcArgs,
    [string]$StdoutPath,
    [string]$StderrPath,
    [hashtable]$Meta,
    [string]$CommandNeedle = ""
) {
    $pidFile = Join-Path $pidDir "$Role.pid"
    $jsonFile = Join-Path $pidDir "$Role.json"
    $livePid = Get-LivePidFromFile $pidFile
    if (-not $livePid -and $CommandNeedle) {
        $livePid = Find-PythonProcessByCommandLine $CommandNeedle
    }
    if ($CommandNeedle) {
        $livePid = Stop-DuplicatePythonProcesses -Role $Role -Needle $CommandNeedle -PreferredPid $livePid
    }
    if ($livePid) {
        Set-Content -LiteralPath $pidFile -Value ([string]$livePid) -Encoding ascii
        $record = @{
            role = $Role
            pid = [int]$livePid
            cmdline_sig = "$python $($ProcArgs -join ' ')"
            observed_at = [DateTime]::UtcNow.ToString("o")
            cwd = $RepoRoot
            already_running = $true
        }
        foreach ($key in $Meta.Keys) {
            $record[$key] = $Meta[$key]
        }
        $record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $jsonFile -Encoding utf8
        return [PSCustomObject]@{ Id = [int]$livePid; AlreadyRunning = $true }
    }
    $proc = Start-Process `
        -FilePath $python `
        -ArgumentList $ProcArgs `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden `
        -PassThru `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath
    Set-Content -LiteralPath $pidFile -Value ([string]$proc.Id) -Encoding ascii
    $record = @{
        role = $Role
        pid = [int]$proc.Id
        cmdline_sig = "$python $($ProcArgs -join ' ')"
        started_at = [DateTime]::UtcNow.ToString("o")
        cwd = $RepoRoot
    }
    foreach ($key in $Meta.Keys) {
        $record[$key] = $Meta[$key]
    }
    $record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $jsonFile -Encoding utf8
    return [PSCustomObject]@{ Id = [int]$proc.Id; AlreadyRunning = $false }
}

function Write-RoleStatus([string]$Role, $Proc) {
    if ($Proc -and $Proc.Disabled) {
        Write-Output "DISABLED ${Role}_pid=$($Proc.Id)"
        return
    }
    $verb = "STARTED"
    if ($Proc -and $Proc.AlreadyRunning) {
        $verb = "ALREADY_RUNNING"
    }
    Write-Output "$verb ${Role}_pid=$($Proc.Id)"
}

$logs = Join-Path $RepoRoot "logs"
$pidDir = Join-Path $logs "pids"
Ensure-Dir $logs
Ensure-Dir $pidDir

if (-not $NoCleanStop) {
    powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "stop_eclipse.ps1") -Quiet
    Start-Sleep -Seconds 2
}

$python = Pick-Python

$supervisorOut = Join-Path $logs "collector_supervisor.stdout.log"
$supervisorErr = Join-Path $logs "collector_supervisor.stderr.log"
$supervisorArgs = @("-u", "scripts\collector_supervisor.py", "--cwd", $RepoRoot, "--symbols", $Symbols)
$supervisor = Start-RegisteredPythonProcess `
    -Role "collector_supervisor" `
    -ProcArgs $supervisorArgs `
    -StdoutPath $supervisorOut `
    -StderrPath $supervisorErr `
    -Meta @{ symbols = $Symbols } `
    -CommandNeedle "scripts\collector_supervisor.py"

$heartbeatOut = Join-Path $logs "heartbeat_watchdog.stdout.log"
$heartbeatErr = Join-Path $logs "heartbeat_watchdog.stderr.log"
$heartbeatArgs = @("-u", "-m", "tools.heartbeat_watchdog", "--interval-sec", "5", "--max-age-sec", "420", "--expect-bookticker")
$heartbeat = Start-RegisteredPythonProcess `
    -Role "heartbeat_watchdog" `
    -ProcArgs $heartbeatArgs `
    -StdoutPath $heartbeatOut `
    -StderrPath $heartbeatErr `
    -Meta @{} `
    -CommandNeedle "tools.heartbeat_watchdog"

$bookTickerOut = Join-Path $logs "bookticker_collector.stdout.log"
$bookTickerErr = Join-Path $logs "bookticker_collector.stderr.log"
$bookTickerArgs = @("-W", "ignore", "-u", "-m", "data.bookticker_collector", "--symbols", $Symbols, "--db-path", "data/microstructure.db", "--heartbeat-interval", "5")
$bookTicker = Start-RegisteredPythonProcess `
    -Role "bookticker_collector" `
    -ProcArgs $bookTickerArgs `
    -StdoutPath $bookTickerOut `
    -StderrPath $bookTickerErr `
    -Meta @{ symbols = $Symbols } `
    -CommandNeedle "data.bookticker_collector"

$oiSpotOut = Join-Path $logs "oi_spot_poller.stdout.log"
$oiSpotErr = Join-Path $logs "oi_spot_poller.stderr.log"
$oiSpotArgs = @("-W", "ignore", "-u", "-m", "data.oi_spot_poller")
$oiSpot = Start-RegisteredPythonProcess `
    -Role "oi_spot_poller" `
    -ProcArgs $oiSpotArgs `
    -StdoutPath $oiSpotOut `
    -StderrPath $oiSpotErr `
    -Meta @{ mode = "PUBLIC_POLL_60S_OI_SPOT" } `
    -CommandNeedle "data.oi_spot_poller"

# Liquidation / stream ANOMALY MONITOR — read-only situational-awareness tool (NOT an edge; AUC~0.55).
# Writes reports/research/s34/LIQ_ANOMALY_MONITOR.json for the dashboard liq_anomaly panel. DB mode=ro.
$liqAnomOut = Join-Path $logs "liq_anomaly_monitor.stdout.log"
$liqAnomErr = Join-Path $logs "liq_anomaly_monitor.stderr.log"
$liqAnomArgs = @("-W", "ignore", "-u", "-m", "tools.liq_anomaly_monitor", "--interval-sec", "30")
$liqAnom = Start-RegisteredPythonProcess `
    -Role "liq_anomaly_monitor" `
    -ProcArgs $liqAnomArgs `
    -StdoutPath $liqAnomOut `
    -StderrPath $liqAnomErr `
    -Meta @{ mode = "READONLY_ANOMALY_MONITOR_TOOL_NOT_EDGE" } `
    -CommandNeedle "tools.liq_anomaly_monitor"

# Liquidation tip FORWARD-LEDGER — read-only un-burned data accumulation (tip fingerprint + outcome
# backfill per liq event). The only honest way to ever forward-test the weak (AUC~0.55) signal; NOT a
# trader, no orders. Ledger reports/shadow/liq_tip_forward.jsonl. DB mode=ro.
$liqFwdOut = Join-Path $logs "liq_tip_forward.stdout.log"
$liqFwdErr = Join-Path $logs "liq_tip_forward.stderr.log"
$liqFwdArgs = @("-W", "ignore", "-u", "-m", "tools.liq_tip_forward", "--interval-sec", "30")
$liqFwd = Start-RegisteredPythonProcess `
    -Role "liq_tip_forward" `
    -ProcArgs $liqFwdArgs `
    -StdoutPath $liqFwdOut `
    -StderrPath $liqFwdErr `
    -Meta @{ mode = "READONLY_FORWARD_LEDGER_UNBURNED_NOT_EDGE" } `
    -CommandNeedle "tools.liq_tip_forward"

# ECHO FORWARD-LEDGER — frozen-rule (echo_30_90+regime) un-burned forward accumulation for the last
# alpha lead (prereg S34_ECHO_FRESH_GATED_PREREGISTRATION_V1, SYSTEM_STATE §164). Records T0-causal
# qualification (qualified_t0) separately from the discovery rule's T+30m lookahead gate (qualified_full),
# plus a rich indicator snapshot for forward-clean signal development (OD-029). NOT a trader, no orders.
# Ledger reports/shadow/echo_forward_ledger.jsonl. DB mode=ro.
$echoFwdOut = Join-Path $logs "echo_forward_ledger.stdout.log"
$echoFwdErr = Join-Path $logs "echo_forward_ledger.stderr.log"
$echoFwdArgs = @("-W", "ignore", "-u", "-m", "tools.research_s34_echo_forward_ledger", "--interval-sec", "30")
$echoFwd = Start-RegisteredPythonProcess `
    -Role "echo_forward_ledger" `
    -ProcArgs $echoFwdArgs `
    -StdoutPath $echoFwdOut `
    -StderrPath $echoFwdErr `
    -Meta @{ mode = "READONLY_FROZEN_RULE_FORWARD_LEDGER_ECHO_LEAD_NOT_EDGE" } `
    -CommandNeedle "tools.research_s34_echo_forward_ledger"

# HOLD-HORIZON FORWARD-LEDGER — forward paper accumulation of hour17 & echo(causal) outcomes across
# hold horizons 2/4/6/12/24/48h (SYSTEM_STATE §167). Companion to the in-sample sweep
# (research_s34_hold_horizon_sweep, which is BURNED); this is the honest forward test. NOT a trader,
# no orders. Ledger reports/shadow/hold_horizon_forward_ledger.jsonl. DB mode=ro.
$holdFwdOut = Join-Path $logs "hold_horizon_forward_ledger.stdout.log"
$holdFwdErr = Join-Path $logs "hold_horizon_forward_ledger.stderr.log"
$holdFwdArgs = @("-W", "ignore", "-u", "-m", "tools.research_s34_hold_horizon_forward_ledger", "--interval-sec", "30")
$holdFwd = Start-RegisteredPythonProcess `
    -Role "hold_horizon_forward_ledger" `
    -ProcArgs $holdFwdArgs `
    -StdoutPath $holdFwdOut `
    -StderrPath $holdFwdErr `
    -Meta @{ mode = "READONLY_FORWARD_LEDGER_HOLD_HORIZON_SWEEP_NOT_EDGE" } `
    -CommandNeedle "tools.research_s34_hold_horizon_forward_ledger"

# RETIRED by operator (2026-07-21): legacy paper simulator superseded by the dedicated forward
# ledgers (echo_forward_ledger, hold_horizon_forward_ledger) + v02 shadow mirror. It also crash-loops
# whenever an open paper trade falls inside a collector-downtime book_ticker gap (measured-cost fill
# refuses to fabricate a price). Not restarted; a plain start_eclipse run actively stops any stray.
Stop-PythonProcessesByCommandLine -Role "s34_shadow_paper_runner" -Needle "tools\s34_shadow_paper_runner.py"
Set-Content -LiteralPath (Join-Path $pidDir "s34_shadow_paper_runner.pid") -Value "0" -Encoding ascii
$s34Shadow = [PSCustomObject]@{ Id = 0; AlreadyRunning = $false; Disabled = $true }

# RETIRED by operator (2026-07-21): secondary diagnostic live chart (:5050) no longer used;
# canonical dashboard (:8770) + leads monitor (:8771) are the standing surfaces. Not restarted.
Stop-PythonProcessesByCommandLine -Role "s34_live_chart" -Needle "tools\s34_live_chart.py"
Set-Content -LiteralPath (Join-Path $pidDir "s34_live_chart.pid") -Value "0" -Encoding ascii
$s34Chart = [PSCustomObject]@{ Id = 0; AlreadyRunning = $false; Disabled = $true }

$s34V02MirrorInterval = 180
$s34V02MirrorDb = "data/microstructure.db"
$s34V02MirrorCheckpoint = Join-Path $RepoRoot "runtime\s34_v_engine_v02_shadow_mirror_checkpoint.json"
$s34V02MirrorOut = Join-Path $logs "s34_v_engine_v02_shadow_mirror.out.log"
$s34V02MirrorErr = Join-Path $logs "s34_v_engine_v02_shadow_mirror.err.log"
$s34V02MirrorArgs = @("-W", "ignore", "-u", "-m", "tools.s34_v_engine_v02_shadow_mirror", "--loop", "--interval-sec", "$s34V02MirrorInterval")
$s34V02Mirror = Start-RegisteredPythonProcess `
    -Role "s34_v_engine_v02_shadow_mirror" `
    -ProcArgs $s34V02MirrorArgs `
    -StdoutPath $s34V02MirrorOut `
    -StderrPath $s34V02MirrorErr `
    -Meta @{ mode = "SHADOW_ONLY_NO_ORDERS"; interval_sec = $s34V02MirrorInterval; checkpoint = $s34V02MirrorCheckpoint } `
    -CommandNeedle "tools.s34_v_engine_v02_shadow_mirror"

$s34V02MirrorPriority = "n/a"
try {
    $s34V02MirrorProc = Get-Process -Id $s34V02Mirror.Id -ErrorAction Stop
    $s34V02MirrorProc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
    $s34V02MirrorPriority = $s34V02MirrorProc.PriorityClass
} catch {
    $s34V02MirrorPriority = "FAILED_TO_SET: $($_.Exception.Message)"
}
Write-Output (
    "STARTUP_HARDENING s34_v_engine_v02_shadow_mirror pid=$($s34V02Mirror.Id) interval_sec=$s34V02MirrorInterval " +
    "db=$s34V02MirrorDb checkpoint=$s34V02MirrorCheckpoint priority=$s34V02MirrorPriority " +
    "start_time_utc=$([DateTime]::UtcNow.ToString('o'))"
)

$s34StateMachineShadowOut = Join-Path $logs "s34_state_machine_shadow_runner.out.log"
$s34StateMachineShadowErr = Join-Path $logs "s34_state_machine_shadow_runner.err.log"
$s34StateMachineShadowArgs = @("-W", "ignore", "-u", "-m", "tools.s34_realtime_shadow_runner")
$s34StateMachineShadow = Start-RegisteredPythonProcess `
    -Role "s34_state_machine_shadow_runner" `
    -ProcArgs $s34StateMachineShadowArgs `
    -StdoutPath $s34StateMachineShadowOut `
    -StderrPath $s34StateMachineShadowErr `
    -Meta @{ mode = "STATE_MACHINE_SHADOW_NO_ORDERS" } `
    -CommandNeedle "tools.s34_realtime_shadow_runner"

if ($EnableLive) {
    $s34SMLiveOut = Join-Path $logs "s34_state_machine_live_executor.out.log"
    $s34SMLiveErr = Join-Path $logs "s34_state_machine_live_executor.err.log"
    $s34SMliveArgs = @("-W", "ignore", "-u", "-m", "tools.s34_state_machine_live_executor", "--live", "--confirm-live-orders", "--db", "data/microstructure.db")
    $s34SMLive = Start-RegisteredPythonProcess `
        -Role "s34_state_machine_live_executor" `
        -ProcArgs $s34SMliveArgs `
        -StdoutPath $s34SMLiveOut `
        -StderrPath $s34SMLiveErr `
        -Meta @{ mode = "LIVE_STATE_MACHINE" } `
        -CommandNeedle "tools.s34_state_machine_live_executor"
} else {
    Stop-PythonProcessesByCommandLine -Role "s34_state_machine_live_executor" -Needle "tools.s34_state_machine_live_executor"
    Set-Content -LiteralPath (Join-Path $pidDir "s34_state_machine_live_executor.pid") -Value "0" -Encoding ascii
    Set-Content -LiteralPath (Join-Path $pidDir "s34_v_engine_live_executor.pid") -Value "0" -Encoding ascii
    $s34SMLive = [PSCustomObject]@{ Id = 0; AlreadyRunning = $false; Disabled = $true }
}

# Opt-in, disabled by default (per operator authorization -- see
# reports/research/s34/LIQUIDATION_SILENCE_DETECTOR_PERIODIC_SCHEDULING_DESIGN.md
# and its corrective-implementation follow-up). Same shape as the -EnableLive
# gate above: not requesting the flag stops any stray instance and zeroes the
# PID file, so a prior opt-in run never silently lingers across a plain
# start_eclipse.ps1 invocation. Read-only detector wrapper -- no order/
# execution path, mirrors every other role's registered-process bookkeeping.
if ($EnableLiquidationSilenceScheduler) {
    $liqSchedOut = Join-Path $logs "liquidation_silence_scheduler.stdout.log"
    $liqSchedErr = Join-Path $logs "liquidation_silence_scheduler.stderr.log"
    $liqSchedArgs = @("-W", "ignore", "-u", "-m", "tools.liquidation_silence_scheduler")
    $liqSched = Start-RegisteredPythonProcess `
        -Role "liquidation_silence_scheduler" `
        -ProcArgs $liqSchedArgs `
        -StdoutPath $liqSchedOut `
        -StderrPath $liqSchedErr `
        -Meta @{ mode = "READ_ONLY_HEALTH_DETECTOR_SCHEDULER_NO_ORDERS" } `
        -CommandNeedle "tools.liquidation_silence_scheduler"
} else {
    Stop-PythonProcessesByCommandLine -Role "liquidation_silence_scheduler" -Needle "tools.liquidation_silence_scheduler"
    Set-Content -LiteralPath (Join-Path $pidDir "liquidation_silence_scheduler.pid") -Value "0" -Encoding ascii
    $liqSched = [PSCustomObject]@{ Id = 0; AlreadyRunning = $false; Disabled = $true }
}

# RETIRED by operator (2026-07-21): secondary diagnostic orderflow chart (:5051) no longer used. Not restarted.
Stop-PythonProcessesByCommandLine -Role "orderflow_chart" -Needle "tools\orderflow_chart.py"
Set-Content -LiteralPath (Join-Path $pidDir "orderflow_chart.pid") -Value "0" -Encoding ascii
$orderflow = [PSCustomObject]@{ Id = 0; AlreadyRunning = $false; Disabled = $true }

# RETIRED by operator (2026-07-21): secondary diagnostic replay chart (:5052) no longer used. Not restarted.
Stop-PythonProcessesByCommandLine -Role "s34_replay" -Needle "tools\s34_replay.py"
Set-Content -LiteralPath (Join-Path $pidDir "s34_replay.pid") -Value "0" -Encoding ascii
$replay = [PSCustomObject]@{ Id = 0; AlreadyRunning = $false; Disabled = $true }

$s34DashboardOut = Join-Path $logs "s34_canonical_dashboard.stdout.log"
$s34DashboardErr = Join-Path $logs "s34_canonical_dashboard.stderr.log"
$s34DashboardArgs = @("-W", "ignore", "-u", "-m", "tools.s34_cascade_navigation_dashboard", "--serve", "--serve-host", "127.0.0.1", "--serve-port", "8770")
$s34Dashboard = Start-RegisteredPythonProcess `
    -Role "s34_canonical_dashboard" `
    -ProcArgs $s34DashboardArgs `
    -StdoutPath $s34DashboardOut `
    -StderrPath $s34DashboardErr `
    -Meta @{ mode = "READ_ONLY_DASHBOARD_SERVE_GET_HEAD_ONLY" } `
    -CommandNeedle "tools.s34_cascade_navigation_dashboard --serve "

# LEADS MONITOR — secondary/diagnostic read-only surface for the two standing leads
# (echo_30_90+regime CAUSAL, LONG_HOUR17). GET/HEAD only, DB mode=ro, no control. Does NOT
# supersede the canonical :8770 surface. Quarantines outage/gap artifacts (SYSTEM_STATE §141).
$s34LeadsMonOut = Join-Path $logs "s34_leads_monitor_dashboard.stdout.log"
$s34LeadsMonErr = Join-Path $logs "s34_leads_monitor_dashboard.stderr.log"
$s34LeadsMonArgs = @("-W", "ignore", "-u", "-m", "tools.s34_leads_monitor_dashboard", "--serve", "--host", "127.0.0.1", "--serve-port", "8771")
$s34LeadsMon = Start-RegisteredPythonProcess `
    -Role "s34_leads_monitor_dashboard" `
    -ProcArgs $s34LeadsMonArgs `
    -StdoutPath $s34LeadsMonOut `
    -StderrPath $s34LeadsMonErr `
    -Meta @{ mode = "READ_ONLY_LEADS_MONITOR_SERVE_GET_HEAD_ONLY_NOT_CANONICAL" } `
    -CommandNeedle "tools.s34_leads_monitor_dashboard --serve "

Write-RoleStatus "collector_supervisor" $supervisor
Write-RoleStatus "heartbeat_watchdog" $heartbeat
Write-RoleStatus "bookticker_collector" $bookTicker
Write-RoleStatus "oi_spot_poller" $oiSpot
Write-RoleStatus "s34_shadow_paper_runner" $s34Shadow
Write-RoleStatus "s34_live_chart" $s34Chart
Write-RoleStatus "s34_v_engine_v02_shadow_mirror" $s34V02Mirror
Write-RoleStatus "s34_state_machine_shadow_runner" $s34StateMachineShadow
Write-RoleStatus "s34_state_machine_live_executor" $s34SMLive
Write-RoleStatus "liquidation_silence_scheduler" $liqSched
Write-RoleStatus "orderflow_chart" $orderflow
Write-RoleStatus "s34_replay" $replay
Write-RoleStatus "s34_canonical_dashboard" $s34Dashboard
Write-RoleStatus "s34_leads_monitor_dashboard" $s34LeadsMon
Write-RoleStatus "hold_horizon_forward_ledger" $holdFwd
Write-Output "symbols=$Symbols"
Write-Output "logs=$logs"

Start-Sleep -Seconds ([Math]::Max(1, [Math]::Min($StatusWaitSec, 60)))
powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "status_eclipse.ps1")
