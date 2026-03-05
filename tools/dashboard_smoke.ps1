# Dashboard full smoke helper
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\dashboard_smoke.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\dashboard_smoke.ps1 -StartServices

param(
    [switch]$StartServices,
    [string]$BackendUrl = "http://127.0.0.1:8765",
    [string]$FrontendUrl = "http://localhost:5173",
    [int]$WaitSec = 8
)

$ErrorActionPreference = "Stop"

function Write-Check($ok, $label, $detail = "") {
    if ($ok) {
        Write-Host ("[OK]   {0} {1}" -f $label, $detail) -ForegroundColor Green
    } else {
        Write-Host ("[FAIL] {0} {1}" -f $label, $detail) -ForegroundColor Red
    }
}

function Test-Url($url) {
    try {
        $resp = Invoke-WebRequest -Uri $url -Method GET -TimeoutSec 8 -UseBasicParsing
        return @{ ok = ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300); code = $resp.StatusCode; body = $resp.Content }
    } catch {
        return @{ ok = $false; code = -1; body = $_.Exception.Message }
    }
}

$repoRoot = Split-Path $PSScriptRoot -Parent
$backendRunner = Join-Path $repoRoot "tools\run_dashboard_backend.ps1"
$frontendRunner = Join-Path $repoRoot "tools\run_dashboard_frontend.ps1"

if ($StartServices) {
    Write-Host "[INFO] Starting dashboard backend + frontend in separate terminals..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $backendRunner
    ) | Out-Null
    Start-Process powershell -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $frontendRunner
    ) | Out-Null
    Write-Host ("[INFO] Waiting {0}s for services to boot..." -f $WaitSec) -ForegroundColor Cyan
    Start-Sleep -Seconds $WaitSec
}

Write-Host "=== Dashboard Smoke Check ===" -ForegroundColor Cyan
Write-Host ("Backend:  {0}" -f $BackendUrl)
Write-Host ("Frontend: {0}" -f $FrontendUrl)

$health = Test-Url "$BackendUrl/api/health"
$logs = Test-Url "$BackendUrl/api/logs"
$actions = Test-Url "$BackendUrl/api/debug/actions"
$front = Test-Url $FrontendUrl

Write-Check $health.ok "GET /api/health" ("(code={0})" -f $health.code)
Write-Check $logs.ok "GET /api/logs" ("(code={0})" -f $logs.code)
Write-Check $actions.ok "GET /api/debug/actions" ("(code={0})" -f $actions.code)
Write-Check $front.ok "GET frontend /" ("(code={0})" -f $front.code)

$allOk = $health.ok -and $logs.ok -and $actions.ok -and $front.ok
if ($allOk) {
    Write-Host "[PASS] Dashboard smoke passed." -ForegroundColor Green
    exit 0
}

Write-Host "[FAIL] Dashboard smoke failed. Inspect backend/frontend terminals and retry." -ForegroundColor Red
if (-not $health.ok) { Write-Host ("  health error: {0}" -f $health.body) -ForegroundColor DarkYellow }
if (-not $logs.ok) { Write-Host ("  logs error: {0}" -f $logs.body) -ForegroundColor DarkYellow }
if (-not $actions.ok) { Write-Host ("  debug/actions error: {0}" -f $actions.body) -ForegroundColor DarkYellow }
if (-not $front.ok) { Write-Host ("  frontend error: {0}" -f $front.body) -ForegroundColor DarkYellow }
exit 1
