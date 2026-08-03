<#
.SYNOPSIS
    Runs the frozen-DB -> Parquet archival export (rotation Phase-3) to completion,
    detached from any editor or agent session.

.DESCRIPTION
    The export is a multi-hour job. Anything that ties it to an interactive session
    dies with that session -- which is exactly what happened on the first attempt
    (killed at 23/305 partitions after ~76 minutes). This script exists so the job
    outlives whatever started it, the same way start_eclipse.ps1 is run outside the
    sandbox for persistence.

    Safe to re-run at any time: the exporter is resumable from its fsynced manifest
    and holds a single-writer lock, so a second launch while one is live is refused
    rather than corrupting the archive.

    Read-only on the source. Writes only under data/archives/parquet_v1 and logs/.
    It never deletes anything from the frozen DB.

.EXAMPLE
    .\scripts\run_frozen_archive.ps1
        Export book_ticker, then run the proof gate. Blocks; shows progress.

.EXAMPLE
    .\scripts\run_frozen_archive.ps1 -Detach
        Same, but in its own window that survives this terminal closing.

.EXAMPLE
    .\scripts\run_frozen_archive.ps1 -StatusOnly
        Cheap progress read from the manifest. Touches nothing.
#>
param(
    [string]$Table = "book_ticker",
    [int]$BatchRows = 100000,
    [long]$ExpectRows = 0,
    [switch]$SkipVerify,
    [switch]$VerifyOnly,
    [switch]$StatusOnly,
    [switch]$Detach
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Row counts from reports/governance/storage/frozen_db_size_census.json (2026-07-30).
# The frozen segment is read-only and has not changed since 2026-07-23, so these are
# the authoritative totals the coverage check must match exactly.
$CensusRows = @{
    "book_ticker" = 5723357020
    "agg_trades"  = 427185688
    "mark_prices" = 24441427
}
if ($ExpectRows -eq 0 -and $CensusRows.ContainsKey($Table)) {
    $ExpectRows = $CensusRows[$Table]
}

if ($StatusOnly) {
    $statusArgs = @("-W", "ignore", "-m", "tools.frozen_db_parquet_export", "--table", $Table, "--status")
    if ($ExpectRows -gt 0) { $statusArgs += @("--expect-rows", "$ExpectRows") }
    & python @statusArgs
    exit $LASTEXITCODE
}

if ($Detach) {
    $childArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath,
                   "-Table", $Table, "-BatchRows", "$BatchRows", "-ExpectRows", "$ExpectRows")
    if ($SkipVerify) { $childArgs += "-SkipVerify" }
    if ($VerifyOnly) { $childArgs += "-VerifyOnly" }
    $child = Start-Process -FilePath "powershell.exe" -ArgumentList $childArgs -PassThru
    Write-Host "DETACHED pid=$($child.Id) table=$Table"
    Write-Host "progress: .\scripts\run_frozen_archive.ps1 -Table $Table -StatusOnly"
    exit 0
}

$logDir = Join-Path $RepoRoot "logs\archive"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $logDir "${Table}_${stamp}.log"

Write-Host "ARCHIVE_RUN table=$Table batch_rows=$BatchRows expect_rows=$ExpectRows"
Write-Host "log=$log"
Write-Host ""

$started = Get-Date

if (-not $VerifyOnly) {
    Write-Host "=== EXPORT ==="
    & python -W ignore -u -m tools.frozen_db_parquet_export --table $Table --batch-rows $BatchRows 2>&1 |
        Tee-Object -FilePath $log -Append
    $exportCode = $LASTEXITCODE
    if ($exportCode -ne 0) {
        Write-Host "EXPORT_FAILED exit=$exportCode -- re-run this script to resume from the manifest"
        exit $exportCode
    }
}

if ($SkipVerify) {
    Write-Host "VERIFY_SKIPPED (the proof gate has NOT been closed for this table)"
    exit 0
}

Write-Host ""
Write-Host "=== PROOF GATE ==="
$verifyArgs = @("-W", "ignore", "-u", "-m", "tools.frozen_db_parquet_export", "--table", $Table, "--verify")
if ($ExpectRows -gt 0) { $verifyArgs += @("--expect-rows", "$ExpectRows") }
& python @verifyArgs 2>&1 | Tee-Object -FilePath $log -Append
$verifyCode = $LASTEXITCODE

$elapsed = (Get-Date) - $started
Write-Host ""
Write-Host ("ARCHIVE_RUN_DONE table={0} verify_exit={1} elapsed={2:hh\:mm\:ss}" -f $Table, $verifyCode, $elapsed)
if ($verifyCode -ne 0) {
    Write-Host "PROOF GATE DID NOT PASS -- nothing may be deleted on the strength of this archive"
}
exit $verifyCode
