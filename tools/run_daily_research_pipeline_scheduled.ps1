$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path $PSScriptRoot -Parent
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }
$LogsDir = Join-Path $RepoRoot "logs"
$RuntimeDir = Join-Path $RepoRoot "runtime"
$LogPath = Join-Path $LogsDir "daily_research_pipeline.log"
$StatusPath = Join-Path $RuntimeDir "daily_research_pipeline_status.json"
$Today = Get-Date -Format "yyyy-MM-dd"

if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir | Out-Null
}
if (-not (Test-Path $RuntimeDir)) {
    New-Item -ItemType Directory -Path $RuntimeDir | Out-Null
}

$startedAt = Get-Date
"[$($startedAt.ToString('o'))] start daily research pipeline date=$Today" | Add-Content -Path $LogPath -Encoding UTF8

Push-Location $RepoRoot
try {
    $output = & $PythonExe -m tools.run_daily_research_pipeline --date $Today 2>&1
    $exitCode = $LASTEXITCODE
    if ($output) {
        $output | ForEach-Object { $_.ToString() } | Add-Content -Path $LogPath -Encoding UTF8
    }
    $finishedAt = Get-Date
    $status = @{
        ok = ($exitCode -eq 0)
        date = $Today
        started_at = $startedAt.ToString("o")
        finished_at = $finishedAt.ToString("o")
        exit_code = $exitCode
        log_path = $LogPath
    }
    $status | ConvertTo-Json -Depth 4 | Set-Content -Path $StatusPath -Encoding UTF8
    "[$($finishedAt.ToString('o'))] finish daily research pipeline exit_code=$exitCode" | Add-Content -Path $LogPath -Encoding UTF8
    if ($exitCode -ne 0) {
        exit $exitCode
    }
} finally {
    Pop-Location
}
