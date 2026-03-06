# One-command daily execution quality gate runner
# 1) Runs daily execution calibration (+ root-cause)
# 2) Evaluates 7-day canary expansion gate
# 3) Prints final single-line output: CANARY_EXPANSION=GO|HOLD
#
# Usage:
# powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_daily_canary_gate.ps1
# powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_daily_canary_gate.ps1 -Symbol ETHUSDT -Days 14 -WindowDays 7 -MaxTopScore 0.5 -OpenReport

[CmdletBinding()]
param(
    [string]$Symbol = "ETHUSDT",
    [int]$Days = 14,
    [int]$IntervalMs = 100,
    [string]$Physics = "data/derived/physics",
    [string]$ReportDir = "reports/daily",
    [int]$WindowDays = 7,
    [double]$MaxTopScore = 0.5,
    [switch]$SkipDailyCalibration,
    [switch]$OpenReport
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path $PSScriptRoot -Parent
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

$GateJson = Join-Path $RepoRoot "reports\CANARY_EXPANSION_GATE.json"
$GateMd = Join-Path $RepoRoot "reports\CANARY_EXPANSION_GATE.md"

Set-Location $RepoRoot

if (-not $SkipDailyCalibration) {
    Write-Host "[daily-gate] running daily execution calibration..." -ForegroundColor Cyan
    & $PythonExe -m tools.daily_execution_calibration `
        --physics $Physics `
        --symbol $Symbol `
        --interval-ms $IntervalMs `
        --days $Days `
        --run-root-cause 1
    $rcDaily = $LASTEXITCODE
    Write-Host "[daily-gate] daily_execution_calibration rc=$rcDaily" -ForegroundColor DarkGray
    if ($rcDaily -ne 0) {
        Write-Host "CANARY_EXPANSION=HOLD" -ForegroundColor Yellow
        exit $rcDaily
    }
}

Write-Host "[daily-gate] evaluating canary expansion gate..." -ForegroundColor Cyan
& $PythonExe -m tools.evaluate_canary_expansion_gate `
    --report-dir $ReportDir `
    --window-days $WindowDays `
    --max-top-score $MaxTopScore `
    --out-json "reports/CANARY_EXPANSION_GATE.json" `
    --out-md "reports/CANARY_EXPANSION_GATE.md"
$rcGate = $LASTEXITCODE
Write-Host "[daily-gate] evaluate_canary_expansion_gate rc=$rcGate" -ForegroundColor DarkGray

$verdict = "HOLD"
if (Test-Path $GateJson) {
    try {
        $payload = Get-Content $GateJson -Raw | ConvertFrom-Json
        if ([bool]$payload.passed) {
            $verdict = "GO"
        }
    } catch {
    }
}

if ($OpenReport -and (Test-Path $GateMd)) {
    Start-Process $GateMd | Out-Null
}

if ($verdict -eq "GO") {
    Write-Host "CANARY_EXPANSION=GO" -ForegroundColor Green
    exit 0
}

Write-Host "CANARY_EXPANSION=HOLD" -ForegroundColor Yellow
exit 1

