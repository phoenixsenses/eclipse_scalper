#requires -version 5.1
<#
.SYNOPSIS
Registers Windows Task Scheduler task for Eclipse Collector Supervisor.
Runs collector_supervisor.py at startup (30s delay), auto-restarts collector + event_diary.
#>

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }
$script = Join-Path $repoRoot "scripts\collector_supervisor.py"
$taskName = "EclipseCollectorSupervisor"

Write-Host "Installing: $taskName"
Write-Host "Python: $python"
Write-Host "Script: $script"
Write-Host "CWD: $repoRoot"

$action = New-ScheduledTaskAction `
    -Execute $python `
    -Argument "`"$script`" --cwd `"$repoRoot`"" `
    -WorkingDirectory $repoRoot

$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT30S"

$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit "00:00:00"

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Force | Out-Null

Write-Host ""
Write-Host "Done. Task '$taskName' registered."
Write-Host "To start now without rebooting:"
Write-Host "  Start-ScheduledTask -TaskName '$taskName'"
Write-Host ""
Write-Host "To verify:"
Write-Host "  Get-ScheduledTask -TaskName '$taskName' | Select-Object TaskName,State"
