param(
    [string]$Symbols = "BTCUSDT,ETHUSDT",
    [int]$WaitSec = 20,
    [int]$FreshSec = 120,
    [switch]$HardRestartCollectorOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot
$WatchdogLog = Join-Path $RepoRoot "logs\data_layer_watchdog.jsonl"

function Append-WatchdogLog([string]$Action, [hashtable]$Fields) {
    try {
        $rec = @{
            ts_utc = [DateTime]::UtcNow.ToString("o")
            action = $Action
        }
        foreach ($k in $Fields.Keys) {
            $rec[$k] = $Fields[$k]
        }
        Add-Content -LiteralPath $WatchdogLog -Value (($rec | ConvertTo-Json -Compress))
    } catch {
    }
}

Write-Output "WATCHDOG probe_start"
python -m tools.data_layer_probe --db data/microstructure.db --csv data/event_diary.csv --symbols $Symbols --wait-sec $WaitSec --fresh-sec $FreshSec
$probeCode = $LASTEXITCODE
Append-WatchdogLog -Action "probe_start" -Fields @{ code = $probeCode; symbols = $Symbols }

if ($probeCode -eq 0) {
    Write-Output "WATCHDOG ok_no_restart"
    Append-WatchdogLog -Action "ok_no_restart" -Fields @{ code = 0; symbols = $Symbols }
    exit 0
}

Write-Output "WATCHDOG connect_test_before_restart"
python -m data.microstructure_collector --symbols $Symbols --connect-test --max-seconds 15
$connectCode = $LASTEXITCODE
Append-WatchdogLog -Action "connect_test" -Fields @{ code = $connectCode; symbols = $Symbols }

$collectorOnly = $false
if ($HardRestartCollectorOnly) {
    $collectorOnly = $true
} elseif ($connectCode -eq 0) {
    # If network connect path passes but data is stale, restart collector writer path first.
    $collectorOnly = $true
}

Write-Output "WATCHDOG stale_restart collector_only=$collectorOnly"
if ($collectorOnly) {
    powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "start_data_layer.ps1") -Symbols $Symbols -ForceRestart -OnlyCollector
    Append-WatchdogLog -Action "restart" -Fields @{ mode = "collector_only"; symbols = $Symbols }
} else {
    powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "start_data_layer.ps1") -Symbols $Symbols -ForceRestart
    Append-WatchdogLog -Action "restart" -Fields @{ mode = "full"; symbols = $Symbols }
}

Write-Output "WATCHDOG probe_after_restart"
python -m tools.data_layer_probe --db data/microstructure.db --csv data/event_diary.csv --symbols $Symbols --wait-sec $WaitSec --fresh-sec $FreshSec
$finalCode = $LASTEXITCODE
Append-WatchdogLog -Action "probe_after_restart" -Fields @{ code = $finalCode; symbols = $Symbols; collector_only = $collectorOnly }

if ($finalCode -eq 0) {
    Write-Output "WATCHDOG final_status=OK"
    Append-WatchdogLog -Action "final_status" -Fields @{ status = "OK"; code = 0; symbols = $Symbols }
    exit 0
}

try {
    $microErr = Join-Path $RepoRoot "logs\microstructure_collector.err.log"
    $microOut = Join-Path $RepoRoot "logs\microstructure_collector.log"
    $hasWinErr = $false
    if (Test-Path -LiteralPath $microErr) {
        $hitErr = Select-String -Path $microErr -Pattern "WinError 5" -Quiet -ErrorAction SilentlyContinue
        if ($hitErr) { $hasWinErr = $true }
    }
    if (-not $hasWinErr -and (Test-Path -LiteralPath $microOut)) {
        $hitOut = Select-String -Path $microOut -Pattern "WinError 5" -Quiet -ErrorAction SilentlyContinue
        if ($hitOut) { $hasWinErr = $true }
    }
    if ($hasWinErr) {
        Write-Output "LIKELY WINDOWS FIREWALL/EDR BLOCKING (NetworkCategory=Public). Run tools/network_diag.ps1 and consider switching network to Private or allowing python.exe."
        Append-WatchdogLog -Action "hint" -Fields @{ type = "winerror5_blocking"; symbols = $Symbols }
    }
} catch {
}

Write-Output "WATCHDOG final_status=FAIL code=$finalCode"
Append-WatchdogLog -Action "final_status" -Fields @{ status = "FAIL"; code = $finalCode; symbols = $Symbols }
exit $finalCode
