#requires -version 5.1
<#
.SYNOPSIS
Summarizes shadow-mode event lane gate telemetry for quick review.
#>

param(
    [string]$TelemetryPath = "logs/telemetry.jsonl",
    [string]$Symbol = "ETHUSDT",
    [string]$OutJson = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repoRoot

$venvPythonLocal = Join-Path $repoRoot ".venv\Scripts\python.exe"
$venvPythonParent = Join-Path (Split-Path -Parent $repoRoot) ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path $venvPythonLocal) {
    $venvPythonLocal
} elseif (Test-Path $venvPythonParent) {
    $venvPythonParent
} else {
    "python"
}

$args = @("-m", "tools.review_event_lane_gate_shadow", "--telemetry-path", $TelemetryPath, "--symbol", $Symbol)
if (-not [string]::IsNullOrWhiteSpace($OutJson)) {
    $args += @("--out-json", $OutJson)
}

& $pythonExe @args
