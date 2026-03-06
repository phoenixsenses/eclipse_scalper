param(
  [ValidateSet("start", "stop", "status")]
  [string]$Action = "start"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
$RuntimeDir = Join-Path $RepoRoot "runtime"
$StackState = Join-Path $RuntimeDir "dashboard_stack.json"
$SupervisorScript = Join-Path $PSScriptRoot "run_dashboard_backend_supervisor.ps1"
$FrontendScript = Join-Path $PSScriptRoot "run_dashboard_frontend.ps1"

function Test-PidRunning([int]$pid) {
  try { Get-Process -Id $pid -ErrorAction Stop | Out-Null; return $true } catch { return $false }
}

function Save-State([hashtable]$state) {
  if (-not (Test-Path $RuntimeDir)) { New-Item -ItemType Directory -Path $RuntimeDir | Out-Null }
  ($state | ConvertTo-Json -Depth 5) | Set-Content -Encoding UTF8 $StackState
}

function Load-State() {
  if (-not (Test-Path $StackState)) { return $null }
  try { return Get-Content $StackState -Raw | ConvertFrom-Json } catch { return $null }
}

function Stop-Stack() {
  $state = Load-State
  if ($null -eq $state) {
    Write-Host "[dashboard-stack] no state file, best-effort cleanup" -ForegroundColor Yellow
    return
  }
  foreach ($k in @("frontend_pid", "supervisor_pid")) {
    $pid = [int]($state.$k)
    if ($pid -gt 0 -and (Test-PidRunning $pid)) {
      Write-Host "[dashboard-stack] stopping $k pid=$pid" -ForegroundColor Yellow
      try { Stop-Process -Id $pid -Force -ErrorAction Stop } catch { }
    }
  }
  Remove-Item $StackState -Force -ErrorAction SilentlyContinue
  Write-Host "[dashboard-stack] stopped" -ForegroundColor Green
}

function Show-Status() {
  $state = Load-State
  if ($null -eq $state) {
    Write-Host "[dashboard-stack] status=down" -ForegroundColor Yellow
    return
  }
  $fp = [int]($state.frontend_pid)
  $sp = [int]($state.supervisor_pid)
  $fr = if ($fp -gt 0) { Test-PidRunning $fp } else { $false }
  $sr = if ($sp -gt 0) { Test-PidRunning $sp } else { $false }
  Write-Host "[dashboard-stack] status frontend=$fr(pid=$fp) supervisor=$sr(pid=$sp)" -ForegroundColor Cyan
  Write-Host "[dashboard-stack] ui=http://localhost:5173" -ForegroundColor DarkGray
}

switch ($Action) {
  "stop" {
    Stop-Stack
    break
  }
  "status" {
    Show-Status
    break
  }
  "start" {
    $existing = Load-State
    if ($null -ne $existing) {
      $fp = [int]($existing.frontend_pid)
      $sp = [int]($existing.supervisor_pid)
      if (($fp -gt 0 -and (Test-PidRunning $fp)) -or ($sp -gt 0 -and (Test-PidRunning $sp))) {
        Write-Host "[dashboard-stack] already running. use -Action stop first." -ForegroundColor Yellow
        Show-Status
        exit 0
      }
    }

    Write-Host "[dashboard-stack] starting supervisor..." -ForegroundColor Cyan
    $supArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$SupervisorScript`""
    $sup = Start-Process -FilePath "powershell" -ArgumentList $supArgs -PassThru -WindowStyle Hidden

    Start-Sleep -Seconds 1

    Write-Host "[dashboard-stack] starting frontend..." -ForegroundColor Cyan
    $frontArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$FrontendScript`""
    $front = Start-Process -FilePath "powershell" -ArgumentList $frontArgs -PassThru

    $state = @{
      ts = (Get-Date).ToString("o")
      supervisor_pid = $sup.Id
      frontend_pid = $front.Id
    }
    Save-State $state

    Write-Host "[dashboard-stack] started supervisor_pid=$($sup.Id) frontend_pid=$($front.Id)" -ForegroundColor Green
    Write-Host "[dashboard-stack] ui=http://localhost:5173" -ForegroundColor Green
    break
  }
}
