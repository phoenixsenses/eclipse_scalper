# Runs focused backend + frontend tests for Live Monitor integration.
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_live_monitor_tests.ps1

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path $PSScriptRoot -Parent
$FrontendDir = Join-Path $RepoRoot "dashboard\frontend"
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }
$StrictMode = if ($env:LIVE_TESTS_STRICT) { $env:LIVE_TESTS_STRICT } else { "0" }
$RuntimeDir = Join-Path $RepoRoot "runtime"
$LogsDir = Join-Path $RepoRoot "logs"
$StatusPath = Join-Path $RuntimeDir "live_monitor_tests_status.json"
$LogPath = Join-Path $LogsDir "live_monitor_tests.log"

if (-not (Test-Path $RuntimeDir)) {
    New-Item -ItemType Directory -Path $RuntimeDir | Out-Null
}
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir | Out-Null
}

function Write-Status {
    param(
        [string]$State,
        [string]$Stage,
        [string]$Message,
        [bool]$BackendOk = $false,
        [bool]$FrontendTypecheckOk = $false,
        [bool]$FrontendSmokeOk = $false,
        [bool]$FrontendSmokeSkipped = $false
    )
    $payload = @{
        ts_utc = (Get-Date).ToUniversalTime().ToString("o")
        state = $State
        stage = $Stage
        message = $Message
        strict_mode = ($StrictMode -in @("1", "true", "TRUE", "yes", "YES"))
        backend_ok = $BackendOk
        frontend_typecheck_ok = $FrontendTypecheckOk
        frontend_smoke_ok = $FrontendSmokeOk
        frontend_smoke_skipped = $FrontendSmokeSkipped
        pid = $PID
        run_command = "powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_live_monitor_tests.ps1"
        log_path = $LogPath
        status_path = $StatusPath
    } | ConvertTo-Json -Depth 6
    Set-Content -Path $StatusPath -Value $payload -Encoding UTF8
    Add-Content -Path $LogPath -Value "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] state=$State stage=$Stage msg=$Message"
}

Write-Host "[live-tests] repo: $RepoRoot" -ForegroundColor Cyan
Set-Location $RepoRoot
Write-Status -State "running" -Stage "init" -Message "starting live monitor test bundle"

Write-Host "[live-tests] py_compile..." -ForegroundColor Cyan
Write-Status -State "running" -Stage "py_compile" -Message "running backend py_compile"
& $PythonExe -m py_compile dashboard/backend/models.py dashboard/backend/data_sources.py dashboard/backend/app.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[live-tests] py_compile failed (exit $LASTEXITCODE)" -ForegroundColor Red
    Write-Status -State "failed" -Stage "py_compile" -Message "py_compile failed exit=$LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "[live-tests] pytest backend live metrics..." -ForegroundColor Cyan
Write-Status -State "running" -Stage "backend_pytest" -Message "running backend pytest live metrics"
& $PythonExe -m pytest -q tests/test_dashboard_live_metrics_api.py --tb=short
if ($LASTEXITCODE -ne 0) {
    Write-Host "[live-tests] backend pytest failed (exit $LASTEXITCODE)" -ForegroundColor Red
    Write-Status -State "failed" -Stage "backend_pytest" -Message "backend pytest failed exit=$LASTEXITCODE"
    exit $LASTEXITCODE
}
Write-Status -State "running" -Stage "backend_done" -Message "backend checks passed" -BackendOk $true

if (-not (Test-Path $FrontendDir)) {
    Write-Host "[live-tests] frontend dir not found: $FrontendDir" -ForegroundColor Red
    Write-Status -State "failed" -Stage "frontend_init" -Message "frontend dir missing: $FrontendDir" -BackendOk $true
    exit 2
}

Set-Location $FrontendDir
if (-not (Test-Path (Join-Path $FrontendDir "node_modules"))) {
    Write-Host "[live-tests] installing frontend deps..." -ForegroundColor Cyan
    Write-Status -State "running" -Stage "frontend_npm_install" -Message "installing frontend deps" -BackendOk $true
    cmd /c npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[live-tests] npm install failed (exit $LASTEXITCODE)" -ForegroundColor Red
        Write-Status -State "failed" -Stage "frontend_npm_install" -Message "npm install failed exit=$LASTEXITCODE" -BackendOk $true
        exit $LASTEXITCODE
    }
}

Write-Host "[live-tests] frontend typecheck..." -ForegroundColor Cyan
Write-Status -State "running" -Stage "frontend_typecheck" -Message "running frontend typecheck" -BackendOk $true
cmd /c npm run typecheck
if ($LASTEXITCODE -ne 0) {
    Write-Host "[live-tests] typecheck failed (exit $LASTEXITCODE)" -ForegroundColor Red
    Write-Status -State "failed" -Stage "frontend_typecheck" -Message "frontend typecheck failed exit=$LASTEXITCODE" -BackendOk $true
    exit $LASTEXITCODE
}
Write-Status -State "running" -Stage "frontend_typecheck_done" -Message "frontend typecheck passed" -BackendOk $true -FrontendTypecheckOk $true

$VitestCmd = Join-Path $FrontendDir "node_modules\.bin\vitest.cmd"
if (-not (Test-Path $VitestCmd)) {
    Write-Host "[live-tests] vitest binary missing; reinstalling frontend deps..." -ForegroundColor Yellow
    Write-Status -State "running" -Stage "frontend_vitest_install" -Message "vitest missing; reinstalling deps" -BackendOk $true -FrontendTypecheckOk $true
    cmd /c npm install
    if ($LASTEXITCODE -ne 0) {
        if ($StrictMode -in @("1", "true", "TRUE", "yes", "YES")) {
            Write-Host "[live-tests] npm install failed in STRICT mode (exit $LASTEXITCODE)" -ForegroundColor Red
            Write-Status -State "failed" -Stage "frontend_vitest_install" -Message "strict mode npm install failed exit=$LASTEXITCODE" -BackendOk $true -FrontendTypecheckOk $true
            exit $LASTEXITCODE
        }
        Write-Host "[live-tests] WARN: npm install failed; skipping frontend smoke (set LIVE_TESTS_STRICT=1 to fail hard)." -ForegroundColor Yellow
        Write-Host "[live-tests] PASS (backend + typecheck only)" -ForegroundColor Green
        Write-Status -State "passed" -Stage "complete" -Message "frontend smoke skipped due to npm install failure" -BackendOk $true -FrontendTypecheckOk $true -FrontendSmokeSkipped $true
        exit 0
    }
}

Write-Host "[live-tests] frontend LiveMonitor smoke..." -ForegroundColor Cyan
Write-Status -State "running" -Stage "frontend_smoke" -Message "running frontend LiveMonitor smoke" -BackendOk $true -FrontendTypecheckOk $true
cmd /c npm run test -- src/pages/__tests__/LiveMonitor.smoke.test.tsx
if ($LASTEXITCODE -ne 0) {
    Write-Host "[live-tests] frontend smoke failed (exit $LASTEXITCODE)" -ForegroundColor Red
    Write-Status -State "failed" -Stage "frontend_smoke" -Message "frontend smoke failed exit=$LASTEXITCODE" -BackendOk $true -FrontendTypecheckOk $true
    exit $LASTEXITCODE
}

Write-Host "[live-tests] ALL PASS" -ForegroundColor Green
Write-Status -State "passed" -Stage "complete" -Message "all checks passed" -BackendOk $true -FrontendTypecheckOk $true -FrontendSmokeOk $true
exit 0
