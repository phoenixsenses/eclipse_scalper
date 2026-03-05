# Eclipse Scalper Dashboard - one-command launcher
# Starts backend, waits for /api/health, then starts frontend.

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path $PSScriptRoot -Parent
$BackendScript = Join-Path $PSScriptRoot "run_dashboard_backend.ps1"
$FrontendScript = Join-Path $PSScriptRoot "run_dashboard_frontend.ps1"
$RuntimeFile = Join-Path $RepoRoot "runtime\dashboard_backend.json"
$LogsDir = Join-Path $RepoRoot "logs"
$RuntimeDir = Join-Path $RepoRoot "runtime"
$LauncherLock = Join-Path $RuntimeDir "dashboard_launcher.lock"
$BackendOutLog = Join-Path $LogsDir "dashboard_backend_launcher.out.log"
$BackendErrLog = Join-Path $LogsDir "dashboard_backend_launcher.err.log"

$HostName = if ($env:DASHBOARD_HOST) { $env:DASHBOARD_HOST } else { "127.0.0.1" }
$Port = if ($env:DASHBOARD_PORT) { [int]$env:DASHBOARD_PORT } else { 8765 }
$BackendHidden = if ($env:DASHBOARD_BACKEND_HIDDEN) { $env:DASHBOARD_BACKEND_HIDDEN } else { "1" }
$AllowMulti = if ($env:DASHBOARD_ALLOW_MULTI) { $env:DASHBOARD_ALLOW_MULTI } else { "0" }

function Test-PidRunning {
    param([int]$Pid)
    try {
        $null = Get-Process -Id $Pid -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function Resolve-HealthUrl {
    $h = $HostName
    $p = $Port
    if (Test-Path $RuntimeFile) {
        try {
            $runtime = Get-Content $RuntimeFile -Raw | ConvertFrom-Json
            if ($runtime.host) { $h = [string]$runtime.host }
            if ($runtime.port) { $p = [int]$runtime.port }
        } catch {
        }
    }
    return "http://${h}:${p}/api/health"
}

Write-Host "[dashboard] bootstrapping..." -ForegroundColor Cyan
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir | Out-Null
}
if (-not (Test-Path $RuntimeDir)) {
    New-Item -ItemType Directory -Path $RuntimeDir | Out-Null
}

if ((Test-Path $LauncherLock) -and ($AllowMulti -notin @("1", "true", "TRUE", "yes", "YES"))) {
    try {
        $lock = Get-Content $LauncherLock -Raw | ConvertFrom-Json
        $lockPid = [int]($lock.pid)
        if ($lockPid -gt 0 -and (Test-PidRunning -Pid $lockPid)) {
            Write-Host "[dashboard] another launcher is already running (pid=$lockPid). Skipping duplicate start." -ForegroundColor Yellow
            exit 0
        }
    } catch {
    }
}
$lockData = @{ pid = $PID; ts = (Get-Date).ToString("o") } | ConvertTo-Json -Depth 2
Set-Content -Path $LauncherLock -Value $lockData -Encoding UTF8

if (Test-Path $BackendOutLog) { Remove-Item $BackendOutLog -Force -ErrorAction SilentlyContinue }
if (Test-Path $BackendErrLog) { Remove-Item $BackendErrLog -Force -ErrorAction SilentlyContinue }
$healthProbe = Resolve-HealthUrl
$backendAlreadyUp = $false
try {
    $r = Invoke-WebRequest -Uri $healthProbe -UseBasicParsing -TimeoutSec 2
    $backendAlreadyUp = ($r.StatusCode -ge 200 -and $r.StatusCode -lt 300)
} catch {
    $backendAlreadyUp = $false
}

$backendProc = $null
if ($backendAlreadyUp) {
    Write-Host "[dashboard] backend already healthy: $healthProbe" -ForegroundColor Green
} else {
    Write-Host "[dashboard] starting backend..." -ForegroundColor Cyan
    $backendArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$BackendScript`""
    $startParams = @{
        FilePath = "powershell"
        ArgumentList = $backendArgs
        PassThru = $true
        RedirectStandardOutput = $BackendOutLog
        RedirectStandardError = $BackendErrLog
    }
    if ($BackendHidden -in @("1", "true", "TRUE", "yes", "YES")) {
        $startParams["WindowStyle"] = "Hidden"
    }
    $backendProc = Start-Process @startParams
    Write-Host "[dashboard] backend launcher pid=$($backendProc.Id)" -ForegroundColor DarkGray
}

$ready = $false
$maxWaitSec = 45
for ($i = 1; $i -le $maxWaitSec; $i++) {
    if ($backendProc -ne $null) {
        try {
            $procState = Get-Process -Id $backendProc.Id -ErrorAction Stop
        } catch {
            Write-Host "[dashboard] backend process exited before health became ready." -ForegroundColor Red
            break
        }
    }
    $healthUrl = Resolve-HealthUrl
    Write-Host "[dashboard] waiting backend health ($i/${maxWaitSec}) -> $healthUrl" -ForegroundColor DarkGray
    try {
        $resp = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
        if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300) {
            $ready = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $ready) {
    Write-Host "[dashboard] backend health check failed." -ForegroundColor Yellow
    Write-Host "[dashboard] frontend will still start, but API panels may show DOWN." -ForegroundColor Yellow
    if (Test-Path $BackendErrLog) {
        Write-Host "[dashboard] backend stderr tail:" -ForegroundColor Yellow
        Get-Content $BackendErrLog -Tail 30 | ForEach-Object { Write-Host $_ -ForegroundColor DarkYellow }
    }
    if (Test-Path $BackendOutLog) {
        Write-Host "[dashboard] backend stdout tail:" -ForegroundColor Yellow
        Get-Content $BackendOutLog -Tail 30 | ForEach-Object { Write-Host $_ -ForegroundColor DarkYellow }
    }
} else {
    $healthUrl = Resolve-HealthUrl
    Write-Host "[dashboard] backend healthy: $healthUrl" -ForegroundColor Green
}

Write-Host "[dashboard] starting frontend..." -ForegroundColor Cyan
& $FrontendScript

if (Test-Path $LauncherLock) {
    Remove-Item $LauncherLock -Force -ErrorAction SilentlyContinue
}
