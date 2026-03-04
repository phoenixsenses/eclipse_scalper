# =============================================================================
# Eclipse Scalper - Paper Trading Startup Script (PowerShell)
#
# Usage:
#   .\scripts\start_paper_trading.ps1
#   .\scripts\start_paper_trading.ps1 -EnvFile .env.paper.dual
#   .\scripts\start_paper_trading.ps1 -EnvFile .env.paper -SkipValidation
#
# What this does:
#   1. Loads .env.paper into the current process environment
#   2. Validates configuration via tools/validate_env.py
#   3. Starts the data collection watchdog in background (PID-based process)
#   4. Starts the scalper via execution.bootstrap (NOT main.py - see note below)
#   5. Cleans up background watchdog process on exit
#
# IMPORTANT - Why not `python main.py`-
#   main.py removes SCALPER_DRY_RUN from the environment unless --dry-run is
#   passed. Using execution.bootstrap directly preserves the SCALPER_DRY_RUN=1
#   set in the .env.paper file, guaranteeing paper mode.
# =============================================================================

param(
    [string]$EnvFile = ".env.paper",
    [switch]$SkipValidation,
    [switch]$NoWatchdog,
    [switch]$ForceRestart
)

$ErrorActionPreference = "Stop"
$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $script:RepoRoot
$watchdogPidFile = Join-Path $script:RepoRoot "logs\\pids\\paper_watchdog.pid"
$watchdogMetaFile = Join-Path $script:RepoRoot "logs\\pids\\paper_watchdog.json"
$watchdogCmdSig = "python -m tools.collection_watchdog"

function Get-WatchdogPythonProcesses {
    $procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue)
    return @($procs | Where-Object {
        $cmd = [string]($_.CommandLine)
        $cmd -match "tools[\\/](collection_watchdog\.py)" -or $cmd -match "(\s|^)-m\s+tools\.collection_watchdog(\s|$)"
    })
}

function Get-WatchdogIdentityStatus {
    param(
        [Parameter(Mandatory = $true)][string]$MetaFile,
        [Parameter(Mandatory = $true)][string]$ExpectedSig
    )
    if (-not (Test-Path $MetaFile)) {
        return [PSCustomObject]@{
            status = "no_registry"
            reason = "registry_missing"
            pid = $null
            process = $null
            meta = $null
            commandLine = $null
        }
    }
    $meta = $null
    try {
        $meta = Get-Content $MetaFile -Raw | ConvertFrom-Json
    } catch {
        return [PSCustomObject]@{
            status = "stale"
            reason = "registry_corrupt"
            pid = $null
            process = $null
            meta = $null
            commandLine = $null
        }
    }
    if ($null -eq $meta) {
        return [PSCustomObject]@{
            status = "stale"
            reason = "registry_empty"
            pid = $null
            process = $null
            meta = $null
            commandLine = $null
        }
    }
    $pidValue = 0
    try { $pidValue = [int]$meta.pid } catch { $pidValue = 0 }
    if ($pidValue -le 0) {
        return [PSCustomObject]@{
            status = "stale"
            reason = "invalid_pid"
            pid = $null
            process = $null
            meta = $meta
            commandLine = $null
        }
    }
    $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
    if ($null -eq $proc) {
        return [PSCustomObject]@{
            status = "stale"
            reason = "pid_not_running"
            pid = $pidValue
            process = $null
            meta = $meta
            commandLine = $null
        }
    }
    $cmdline = $null
    try {
        $cim = Get-CimInstance Win32_Process -Filter ("ProcessId=" + $pidValue) -ErrorAction Stop
        $cmdline = [string]$cim.CommandLine
    } catch {
        $cmdline = $null
    }
    if ([string]::IsNullOrWhiteSpace($cmdline)) {
        # Conservative: process exists but command line not available => assume live and refuse duplicate start.
        return [PSCustomObject]@{
            status = "live_unknown"
            reason = "cmdline_unavailable"
            pid = $pidValue
            process = $proc
            meta = $meta
            commandLine = $null
        }
    }
    $expectedLower = $ExpectedSig.ToLowerInvariant()
    $cmdLower = $cmdline.ToLowerInvariant()
    if ($cmdLower.Contains($expectedLower)) {
        return [PSCustomObject]@{
            status = "live_match"
            reason = "identity_match"
            pid = $pidValue
            process = $proc
            meta = $meta
            commandLine = $cmdline
        }
    }
    return [PSCustomObject]@{
        status = "stale"
        reason = "pid_reuse_signature_mismatch"
        pid = $pidValue
        process = $proc
        meta = $meta
        commandLine = $cmdline
    }
}

# -- Banner --------------------------------------------------------------------
Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  Eclipse Scalper - Paper Trading Mode" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  Repo    : $script:RepoRoot"
Write-Host "  Config  : $EnvFile"
Write-Host "  Date    : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# -- Load .env file into process environment -----------------------------------
$resolvedEnv = $EnvFile
if (-not [System.IO.Path]::IsPathRooted($resolvedEnv)) {
    $resolvedEnv = Join-Path $script:RepoRoot $resolvedEnv
}
if (-not (Test-Path $resolvedEnv)) {
    Write-Error "Config file not found: $resolvedEnv"
    Write-Host "Create it by copying .env.example: Copy-Item .env.example $resolvedEnv" -ForegroundColor Yellow
    exit 1
}

Write-Host "Loading $resolvedEnv ..." -ForegroundColor Gray
Get-Content $resolvedEnv | Where-Object {
    # Skip blank lines and comments
    $_ -match "^\s*[^#\s]" -and $_ -match "="
} | ForEach-Object {
    $parts = $_ -split "=", 2
    $varName = $parts[0].Trim()
    $varValue = $parts[1].Trim()
    # Remove inline comments (text after unquoted #)
    $varValue = $varValue -replace "\s+#.*$", ""
    # Remove surrounding quotes if present
    $varValue = $varValue -replace '^["'']|["'']$', ""
    if ($varName -and $varValue) {
        [System.Environment]::SetEnvironmentVariable($varName, $varValue, "Process")
    }
}

# Verify SCALPER_DRY_RUN=1 immediately
$dryRun = [System.Environment]::GetEnvironmentVariable("SCALPER_DRY_RUN", "Process")
if ($dryRun -ne "1") {
    Write-Host ""
    Write-Host "ERROR: SCALPER_DRY_RUN is not '1' after loading $resolvedEnv" -ForegroundColor Red
    Write-Host "       Current value: '$dryRun'" -ForegroundColor Red
    Write-Host "       This script is for PAPER TRADING only." -ForegroundColor Red
    Write-Host "       Set SCALPER_DRY_RUN=1 in $resolvedEnv" -ForegroundColor Red
    exit 1
}
Write-Host "  Paper mode confirmed: SCALPER_DRY_RUN=1" -ForegroundColor Green

# -- Validate environment ------------------------------------------------------
if (-not $SkipValidation) {
    Write-Host ""
    Write-Host "Running environment validation..." -ForegroundColor Gray
    python -m tools.validate_env --env $resolvedEnv
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Validation failed. Fix the FAIL items above before starting." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "  Validation skipped (-SkipValidation)" -ForegroundColor Yellow
}

# -- Start data collection watchdog (background process) -----------------------
$watchdog = $null
$watchdogStartedHere = $false
if (-not $NoWatchdog) {
    Write-Host ""
    Write-Host "Starting data collection watchdog..." -ForegroundColor Gray
    $status = Get-WatchdogIdentityStatus -MetaFile $watchdogMetaFile -ExpectedSig $watchdogCmdSig
    if ($status.status -eq "live_match" -or $status.status -eq "live_unknown") {
        $watchdog = [PSCustomObject]@{ ProcessId = [int]$status.pid; Source = "registry"; Id = [int]$status.pid }
        Write-Host "  Watchdog already running (pid: $($status.pid), status=$($status.status))" -ForegroundColor Yellow
        if (-not $ForceRestart) {
            Write-Host "  Refusing duplicate start. Use -ForceRestart to restart watchdog safely." -ForegroundColor Yellow
            exit 10
        }
        Write-Host "  ForceRestart requested - stopping existing watchdog pid $($status.pid)" -ForegroundColor Yellow
        Stop-Process -Id ([int]$status.pid) -ErrorAction SilentlyContinue
        Remove-Item $watchdogPidFile,$watchdogMetaFile -ErrorAction SilentlyContinue
        $watchdog = $null
    } elseif ($status.status -eq "stale") {
        Write-Host "  Removed stale watchdog registry (reason=$($status.reason), pid=$($status.pid))" -ForegroundColor DarkGray
        Remove-Item $watchdogPidFile,$watchdogMetaFile -ErrorAction SilentlyContinue
    }
    if ($null -eq $watchdog) {
        $watchdog = Start-Process `
            -FilePath "python" `
            -ArgumentList @("-u", "-m", "tools.collection_watchdog") `
            -WorkingDirectory $script:RepoRoot `
            -WindowStyle Hidden `
            -PassThru
        $watchdogStartedHere = $true
        Write-Host "  Watchdog started (pid: $($watchdog.Id))" -ForegroundColor Green
        Write-Host "  Inspect: Get-Process -Id $($watchdog.Id)" -ForegroundColor DarkGray
        Write-Host "  Stop   : Stop-Process -Id $($watchdog.Id)" -ForegroundColor DarkGray
    }
    $pidDir = Split-Path -Parent $watchdogPidFile
    New-Item -ItemType Directory -Path $pidDir -Force | Out-Null
    $recordPid = if ($watchdog.PSObject.Properties.Name -contains "Id") { $watchdog.Id } else { $watchdog.ProcessId }
    Set-Content -Path $watchdogPidFile -Value "$recordPid" -Encoding ascii
    $record = @{
        role = "paper_watchdog"
        pid = [int]$recordPid
        start_ts_utc = (Get-Date).ToUniversalTime().ToString("s") + "Z"
        cmdline_sig = $watchdogCmdSig
        exe_path = "python"
        parent_pid = $PID
        started_here = [bool]$watchdogStartedHere
        repo_root = $script:RepoRoot
    }
    ($record | ConvertTo-Json -Depth 4) | Set-Content -Path $watchdogMetaFile -Encoding utf8
    Write-Host "  PID file: $watchdogPidFile" -ForegroundColor DarkGray
} else {
    Write-Host "  Watchdog skipped (-NoWatchdog)" -ForegroundColor Yellow
}

# -- Start paper trading -------------------------------------------------------
Write-Host ""
Write-Host "Starting paper trading (CTRL+C to stop)..." -ForegroundColor Cyan
Write-Host ""
$sessionStartUtc = (Get-Date).ToUniversalTime()
$lastShutdown = Join-Path $script:RepoRoot "logs\\last_shutdown.json"
if (Test-Path $lastShutdown) {
    try { Remove-Item $lastShutdown -Force -ErrorAction SilentlyContinue } catch {}
}

try {
    # Use execution.bootstrap directly - NOT python main.py
    # (main.py removes SCALPER_DRY_RUN unless --dry-run is passed)
    python -m execution.bootstrap
    $exitCode = $LASTEXITCODE
} catch {
    Write-Host "Bot exited with exception: $_" -ForegroundColor Red
    $exitCode = 1
} finally {
    Write-Host ""
    Write-Host "=================================================" -ForegroundColor Cyan
    Write-Host "  Eclipse Scalper - Session Ended" -ForegroundColor Cyan
    Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "=================================================" -ForegroundColor Cyan

    # Stop watchdog process
    if ($null -ne $watchdog) {
        $stopPid = if ($watchdog.PSObject.Properties.Name -contains "Id") { $watchdog.Id } else { $watchdog.ProcessId }
        if ($watchdogStartedHere -and $stopPid) {
            Write-Host "Stopping watchdog (pid: $stopPid)..." -ForegroundColor Gray
            Stop-Process -Id $stopPid -ErrorAction SilentlyContinue
            Write-Host "  Watchdog stopped." -ForegroundColor Gray
            Remove-Item $watchdogPidFile,$watchdogMetaFile -ErrorAction SilentlyContinue
        } else {
            Write-Host "Watchdog not started by this session; leaving running." -ForegroundColor DarkGray
        }
    }

    Write-Host ""
    Write-Host "Python exit code: $exitCode" -ForegroundColor Gray
    if (Test-Path $lastShutdown) {
        try {
            $lsw = (Get-Item $lastShutdown).LastWriteTimeUtc
            if ($lsw -ge $sessionStartUtc) {
                Write-Host "Last shutdown diagnostics: $lastShutdown" -ForegroundColor Gray
                Get-Content $lastShutdown -TotalCount 40 | ForEach-Object { Write-Host $_ -ForegroundColor DarkGray }
            } else {
                Write-Host "Last shutdown diagnostics: stale (older than this session)" -ForegroundColor DarkGray
            }
        } catch {
            Write-Host "Last shutdown diagnostics: unreadable" -ForegroundColor DarkGray
        }
    }
    $tradeLogDb = [System.Environment]::GetEnvironmentVariable('ENTRY_TRADE_LOG_DB', 'Process')
    Write-Host "Trade log: $tradeLogDb" -ForegroundColor Gray
    if ($null -ne $watchdog) {
        $finalPid = if ($watchdog.PSObject.Properties.Name -contains "Id") { $watchdog.Id } else { $watchdog.ProcessId }
        Write-Host ('Watchdog process pid: ' + $finalPid) -ForegroundColor DarkGray
        Write-Host ('Manual stop: Stop-Process -Id ' + $finalPid) -ForegroundColor DarkGray
    }
}

exit $exitCode
