#requires -version 5.1
<#
.SYNOPSIS
Starts paper trading with the event lane gate enabled in shadow mode.

.DESCRIPTION
Thin wrapper around scripts/start_paper_trading.ps1. It only injects the
shadow gate environment so the existing paper startup flow stays unchanged.
#>

param(
    [string]$EnvFile = ".env.paper",
    [switch]$SkipValidation,
    [switch]$SkipPreflight,
    [switch]$NoWatchdog,
    [switch]$ForceRestart,
    [switch]$SmokeOffline,
    [string]$GateDb = "data/microstructure.db",
    [string]$TelemetryPath = "logs/telemetry.jsonl"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repoRoot

$env:ENTRY_EVENT_LANE_GATE_ENABLED = "1"
$env:ENTRY_EVENT_LANE_GATE_SHADOW = "1"
$env:ENTRY_EVENT_LANE_GATE_DB = $GateDb
$env:TELEMETRY_PATH = $TelemetryPath

Write-Host "[event-lane-gate] enabled=1 shadow=1 db=$GateDb telemetry=$TelemetryPath"

$startupScript = Join-Path $PSScriptRoot "start_paper_trading.ps1"
& $startupScript `
    -EnvFile $EnvFile `
    -SkipValidation:$SkipValidation `
    -SkipPreflight:$SkipPreflight `
    -NoWatchdog:$NoWatchdog `
    -ForceRestart:$ForceRestart `
    -SmokeOffline:$SmokeOffline
