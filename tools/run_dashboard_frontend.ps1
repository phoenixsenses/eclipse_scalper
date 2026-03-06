# Eclipse Scalper Dashboard - Frontend runner
# Usage: .\tools\run_dashboard_frontend.ps1
# Requires: Node.js >= 18

$RepoRoot = Split-Path $PSScriptRoot -Parent
$FrontendDir = Join-Path $RepoRoot "dashboard\frontend"
$RuntimeFile = Join-Path $RepoRoot "runtime\dashboard_backend.json"

$BackendHost = if ($env:DASHBOARD_HOST) { $env:DASHBOARD_HOST } else { "127.0.0.1" }
$BackendPort = if ($env:DASHBOARD_PORT) { [int]$env:DASHBOARD_PORT } else { 8765 }

if (Test-Path $RuntimeFile) {
    try {
        $runtime = Get-Content $RuntimeFile -Raw | ConvertFrom-Json
        if ($runtime.host) { $BackendHost = [string]$runtime.host }
        if ($runtime.port) { $BackendPort = [int]$runtime.port }
        Write-Host "[dashboard-frontend] Runtime pointer: ${BackendHost}:${BackendPort}" -ForegroundColor DarkGray
    } catch {
        Write-Host "[dashboard-frontend] WARN: runtime pointer parse failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

if (-not (Test-Path (Join-Path $FrontendDir "node_modules"))) {
    Write-Host "[dashboard-frontend] Installing npm dependencies..." -ForegroundColor Cyan
    Set-Location $FrontendDir
    cmd /c npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[dashboard-frontend] npm install failed (exit $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
}

$ProxyTarget = "http://${BackendHost}:${BackendPort}"
[System.Environment]::SetEnvironmentVariable("VITE_PROXY_TARGET", $ProxyTarget, "Process")
Write-Host "[dashboard-frontend] Proxy target: $ProxyTarget" -ForegroundColor DarkGray

Write-Host "[dashboard-frontend] Starting Vite dev server on http://localhost:5173" -ForegroundColor Green
try {
    $backendUp = Get-NetTCPConnection -LocalAddress $BackendHost -LocalPort $BackendPort -State Listen -ErrorAction Stop | Select-Object -First 1
    if (-not $backendUp) {
        Write-Host "[dashboard-frontend] WARNING: backend not listening at ${ProxyTarget}" -ForegroundColor Yellow
        Write-Host "[dashboard-frontend] Start it in another terminal: .\tools\run_dashboard_backend.ps1" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[dashboard-frontend] WARNING: backend not listening at ${ProxyTarget}" -ForegroundColor Yellow
    Write-Host "[dashboard-frontend] Start it in another terminal: .\tools\run_dashboard_backend.ps1" -ForegroundColor Yellow
}

Set-Location $FrontendDir
cmd /c npm run dev
