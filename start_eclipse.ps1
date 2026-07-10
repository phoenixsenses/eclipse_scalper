param(
    [string]$Symbols = "BTCUSDT,ETHUSDT,SOLUSDT",
    [int]$StatusWaitSec = 20,
    [switch]$NoCleanStop,
    [switch]$EnableLive
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
$heartbeatArgs = @("-u", "-m", "tools.heartbeat_watchdog", "--interval-sec", "30", "--max-age-sec", "420", "--expect-bookticker")
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

$s34ShadowOut = Join-Path $logs "s34_shadow_paper_runner.stdout.log"
$s34ShadowErr = Join-Path $logs "s34_shadow_paper_runner.stderr.log"
$s34ShadowArgs = @(
    "-u", "tools\s34_shadow_paper_runner.py",
    "--loop", "--interval-sec", "60",
    "--regime-filter-enabled",
    "--regime-min-trend-pct", "1.0",
    "--regime-min-range-pct", "2.5",
    "--regime-min-buy-liq-notional", "5000000",
    "--regime-min-agg-trade-count", "250000",
    "--quality-gate-enabled",
    "--quality-gate-min-eclipse", "42.0"
)
$s34Shadow = Start-RegisteredPythonProcess `
    -Role "s34_shadow_paper_runner" `
    -ProcArgs $s34ShadowArgs `
    -StdoutPath $s34ShadowOut `
    -StderrPath $s34ShadowErr `
    -Meta @{ mode = "PAPER_ONLY_NO_ORDERS" } `
    -CommandNeedle "tools\s34_shadow_paper_runner.py"

$s34ChartOut = Join-Path $logs "s34_live_chart.stdout.log"
$s34ChartErr = Join-Path $logs "s34_live_chart.stderr.log"
$s34ChartArgs = @("-u", "tools\s34_live_chart.py", "--host", "127.0.0.1", "--port", "5050", "--no-browser")
$s34Chart = Start-RegisteredPythonProcess `
    -Role "s34_live_chart" `
    -ProcArgs $s34ChartArgs `
    -StdoutPath $s34ChartOut `
    -StderrPath $s34ChartErr `
    -Meta @{ mode = "READ_ONLY_CHART" } `
    -CommandNeedle "tools\s34_live_chart.py"

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

$orderflowOut = Join-Path $logs "orderflow_chart.stdout.log"
$orderflowErr = Join-Path $logs "orderflow_chart.stderr.log"
$orderflowArgs = @("-u", "tools\orderflow_chart.py", "--host", "127.0.0.1", "--port", "5051", "--no-browser")
$orderflow = Start-RegisteredPythonProcess `
    -Role "orderflow_chart" `
    -ProcArgs $orderflowArgs `
    -StdoutPath $orderflowOut `
    -StderrPath $orderflowErr `
    -Meta @{ mode = "READ_ONLY_CHART" } `
    -CommandNeedle "tools\orderflow_chart.py"

$replayOut = Join-Path $logs "s34_replay.stdout.log"
$replayErr = Join-Path $logs "s34_replay.stderr.log"
$replayArgs = @("-u", "tools\s34_replay.py", "--host", "127.0.0.1", "--port", "5052", "--no-browser")
$replay = Start-RegisteredPythonProcess `
    -Role "s34_replay" `
    -ProcArgs $replayArgs `
    -StdoutPath $replayOut `
    -StderrPath $replayErr `
    -Meta @{ mode = "READ_ONLY_CHART" } `
    -CommandNeedle "tools\s34_replay.py"

Write-RoleStatus "collector_supervisor" $supervisor
Write-RoleStatus "heartbeat_watchdog" $heartbeat
Write-RoleStatus "bookticker_collector" $bookTicker
Write-RoleStatus "oi_spot_poller" $oiSpot
Write-RoleStatus "s34_shadow_paper_runner" $s34Shadow
Write-RoleStatus "s34_live_chart" $s34Chart
Write-RoleStatus "s34_v_engine_v02_shadow_mirror" $s34V02Mirror
Write-RoleStatus "s34_state_machine_shadow_runner" $s34StateMachineShadow
Write-RoleStatus "s34_state_machine_live_executor" $s34SMLive
Write-RoleStatus "orderflow_chart" $orderflow
Write-RoleStatus "s34_replay" $replay
Write-Output "symbols=$Symbols"
Write-Output "logs=$logs"

Start-Sleep -Seconds ([Math]::Max(1, [Math]::Min($StatusWaitSec, 60)))
powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "status_eclipse.ps1")
