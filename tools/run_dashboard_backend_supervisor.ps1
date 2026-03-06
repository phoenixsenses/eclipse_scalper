# Dashboard backend supervisor runner
$RepoRoot = Split-Path $PSScriptRoot -Parent
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

Set-Location $RepoRoot
& $PythonExe -m tools.dashboard_backend_supervisor
