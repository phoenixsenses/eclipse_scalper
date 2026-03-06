# Single-command execution canary launcher wrapper
# - Runs python canary orchestrator
# - Archives key artifacts to reports/canary_runs/<timestamp>/
# - Optionally opens markdown report
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_execution_canary.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_execution_canary.ps1 -Symbol ETHUSDT -MaxCycles 8 -OpenReport

[CmdletBinding()]
param(
    [string]$Symbol = "ETHUSDT",
    [int]$MaxCycles = 5,
    [double]$RefreshSec = 5.0,
    [string]$Db = "data/microstructure.db",
    [string]$RiskPolicy = "",
    [switch]$OpenReport,
    [switch]$NoArchive
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path $PSScriptRoot -Parent
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

$ReportJson = Join-Path $RepoRoot "reports\CANARY_EXECUTION_REPORT.json"
$ReportMd = Join-Path $RepoRoot "reports\CANARY_EXECUTION_REPORT.md"
$NowTag = (Get-Date).ToString("yyyyMMdd_HHmmss")
$ArchiveDir = Join-Path $RepoRoot ("reports\canary_runs\" + $NowTag)

Set-Location $RepoRoot

$argsList = @(
    "-m", "tools.run_execution_canary",
    "--symbol", $Symbol,
    "--max-cycles", "$MaxCycles",
    "--refresh-sec", "$RefreshSec",
    "--db", $Db,
    "--report", "reports/CANARY_EXECUTION_REPORT.json",
    "--report-md", "reports/CANARY_EXECUTION_REPORT.md"
)
if ($RiskPolicy -and $RiskPolicy.Trim() -ne "") {
    $argsList += @("--risk-policy", $RiskPolicy)
}

Write-Host "[canary] starting python launcher..." -ForegroundColor Cyan
& $PythonExe @argsList
$rc = $LASTEXITCODE
Write-Host "[canary] python launcher rc=$rc" -ForegroundColor DarkGray

if (-not $NoArchive) {
    New-Item -ItemType Directory -Path $ArchiveDir -Force | Out-Null
    $toCopy = @(
        "reports\CANARY_EXECUTION_REPORT.json",
        "reports\CANARY_EXECUTION_REPORT.md",
        "reports\EXECUTION_E2E_PIPELINE.json",
        "reports\POST_ROLLOUT_AUDIT.json",
        "reports\POST_ROLLOUT_AUDIT.md",
        "reports\REPLAY_PARITY_REPORT.json",
        "reports\REPLAY_PARITY_REPORT.md",
        "reports\EXECUTION_HEALTH.json",
        "reports\EXECUTION_HEALTH.md",
        "reports\TOXICITY_REPORT.json",
        "reports\TOXICITY_REPORT.md",
        "logs\live_execution_events.jsonl",
        "logs\live_supervisor.json",
        "data\live\status.json"
    )
    foreach ($rel in $toCopy) {
        $src = Join-Path $RepoRoot $rel
        if (Test-Path $src) {
            $dst = Join-Path $ArchiveDir (Split-Path $rel -Leaf)
            Copy-Item -Path $src -Destination $dst -Force
        }
    }
    Write-Host "[canary] archived artifacts -> $ArchiveDir" -ForegroundColor Green
}

if ($OpenReport -and (Test-Path $ReportMd)) {
    Write-Host "[canary] opening report: $ReportMd" -ForegroundColor Cyan
    Start-Process $ReportMd | Out-Null
}

if (Test-Path $ReportJson) {
    try {
        $payload = Get-Content $ReportJson -Raw | ConvertFrom-Json
        $ok = [bool]$payload.overall_ok
        Write-Host ("[canary] overall_ok=" + ($(if ($ok) { "1" } else { "0" }))) -ForegroundColor $(if ($ok) { "Green" } else { "Yellow" })
    } catch {
    }
}

exit $rc

