# Eclipse Scalper Dashboard - Backend runner
# Usage: .\tools\run_dashboard_backend.ps1
# Compatible with Windows PowerShell 5.1+
# Optional env vars:
#   DASHBOARD_PORT (default 8765)
#   DASHBOARD_HOST (default 127.0.0.1)
#   DASHBOARD_RELOAD (default 0)
#   DASHBOARD_AUTO_PIP (default 0; set 1 to run pip install on startup)

if ($env:DASHBOARD_PORT) {
    $Port = [int]$env:DASHBOARD_PORT
} else {
    $Port = 8765
}

if ($env:DASHBOARD_HOST) {
    $DashHost = $env:DASHBOARD_HOST
} else {
    $DashHost = "127.0.0.1"
}

function Test-BackendHealth {
    param(
        [string]$HostName,
        [int]$PortNum
    )
    $url = "http://${HostName}:${PortNum}/api/health"
    try {
        $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
        return ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300)
    } catch {
        return $false
    }
}

function Test-PortAvailable {
    param(
        [string]$HostName,
        [int]$PortNum
    )
    try {
        $ip = [System.Net.IPAddress]::Parse($HostName)
        $listener = New-Object System.Net.Sockets.TcpListener($ip, $PortNum)
        $listener.Start()
        $listener.Stop()
        return $true
    } catch {
        return $false
    }
}

function Find-FreePort {
    param(
        [string]$HostName,
        [int]$StartPort
    )
    for ($p = $StartPort; $p -lt ($StartPort + 40); $p++) {
        if (Test-PortAvailable -HostName $HostName -PortNum $p) {
            return $p
        }
    }
    return $StartPort
}

$RepoRoot = Split-Path $PSScriptRoot -Parent
$BackendDir = Join-Path $RepoRoot "dashboard\backend"
$ReqFile = Join-Path $BackendDir "requirements.txt"
$VenvDir = Join-Path $RepoRoot ".venv"
$RuntimeDir = Join-Path $RepoRoot "runtime"
$RuntimeFile = Join-Path $RuntimeDir "dashboard_backend.json"
$EnvFile = Join-Path $RepoRoot ".env.paper"
if (-not (Test-Path $EnvFile)) {
    $EnvFile = Join-Path $RepoRoot ".env"
}
$PythonExe = "python"

$VenvActivate = Join-Path $VenvDir "Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    Write-Host "[INFO] Activating virtual environment: $VenvDir" -ForegroundColor Cyan
    & $VenvActivate
    $CandidatePython = Join-Path $VenvDir "Scripts\python.exe"
    if (Test-Path $CandidatePython) {
        $PythonExe = $CandidatePython
    }
} else {
    Write-Host "[INFO] No .venv found, using system Python" -ForegroundColor DarkGray
}

if (Test-Path $ReqFile) {
    $autoPip = $env:DASHBOARD_AUTO_PIP
    if (-not $autoPip) { $autoPip = "0" }
    if ($autoPip -in @("1", "true", "TRUE", "yes", "YES")) {
        Write-Host "[INFO] Installing / verifying dependencies from $ReqFile ..." -ForegroundColor Cyan
        & $PythonExe -m pip install -r $ReqFile --quiet
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] pip install failed (exit $LASTEXITCODE)" -ForegroundColor Red
            exit 1
        }
        Write-Host "[OK] Dependencies ready" -ForegroundColor Green
    } else {
        Write-Host "[INFO] DASHBOARD_AUTO_PIP=0 (skipping pip install)" -ForegroundColor DarkGray
    }
} else {
    Write-Host "[WARN] requirements.txt not found at $ReqFile" -ForegroundColor Yellow
}

if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) { return }
        if ($line -match "^([^=]+)=(.*)$") {
            $k = $Matches[1].Trim()
            $v = $Matches[2].Trim()
            if (($v.StartsWith('"') -and $v.EndsWith('"')) -or
                ($v.StartsWith("'") -and $v.EndsWith("'"))) {
                $v = $v.Substring(1, $v.Length - 2)
            }
            [System.Environment]::SetEnvironmentVariable($k, $v, "Process")
        }
    }
    Write-Host "[OK] Loaded env from $EnvFile" -ForegroundColor Green
} else {
    Write-Host "[INFO] No .env file found at $EnvFile" -ForegroundColor DarkGray
}

# Reuse healthy backend on requested port.
if (Test-BackendHealth -HostName $DashHost -PortNum $Port) {
    Write-Host "[INFO] Healthy backend already running at http://${DashHost}:${Port}. Reusing it." -ForegroundColor Green
    if (-not (Test-Path $RuntimeDir)) {
        New-Item -ItemType Directory -Path $RuntimeDir | Out-Null
    }
    $runtimeReuse = @{
        host = $DashHost
        port = $Port
        pid = $null
        reused = $true
        ts = (Get-Date).ToString("o")
    } | ConvertTo-Json -Depth 4
    Set-Content -Path $RuntimeFile -Value $runtimeReuse -Encoding UTF8
    exit 0
}

# If requested port is unavailable, switch to a free one without requiring admin privileges.
if (-not (Test-PortAvailable -HostName $DashHost -PortNum $Port)) {
    $fallback = Find-FreePort -HostName $DashHost -StartPort ($Port + 1)
    if ($fallback -ne $Port) {
        Write-Host "[WARN] Port ${Port} unavailable. Falling back to ${fallback}." -ForegroundColor Yellow
        $Port = $fallback
    }
}

[System.Environment]::SetEnvironmentVariable("DASHBOARD_PORT", "$Port", "Process")
if (-not (Test-Path $RuntimeDir)) {
    New-Item -ItemType Directory -Path $RuntimeDir | Out-Null
}
$runtime = @{
    host = $DashHost
    port = $Port
    pid = $PID
    reused = $false
    ts = (Get-Date).ToString("o")
} | ConvertTo-Json -Depth 4
Set-Content -Path $RuntimeFile -Value $runtime -Encoding UTF8

Write-Host "[OK] Starting dashboard backend on http://${DashHost}:${Port}" -ForegroundColor Green
Set-Location $RepoRoot
$ReloadEnabled = ($env:DASHBOARD_RELOAD -in @("1", "true", "TRUE", "yes", "YES"))
if ($ReloadEnabled) {
    Write-Host "[INFO] DASHBOARD_RELOAD=1 (dev autoreload ON)" -ForegroundColor DarkGray
    & $PythonExe -m uvicorn dashboard.backend.app:app --host $DashHost --port $Port --reload --log-level info
} else {
    Write-Host "[INFO] DASHBOARD_RELOAD=0 (stable mode, autoreload OFF)" -ForegroundColor DarkGray
    & $PythonExe -m uvicorn dashboard.backend.app:app --host $DashHost --port $Port --log-level info
}
