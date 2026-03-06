#requires -version 5.1
<#
.SYNOPSIS
Registers a startup scheduled task for Eclipse Scalper supervisor.

.DESCRIPTION
Creates/updates a Windows Task Scheduler entry that runs
`scripts/supervisor.py` at startup after an optional delay.

.PARAMETER TaskName
Task name to register (default: EclipseScalperSupervisor).

.PARAMETER DelaySeconds
Startup delay in seconds (default: 30).

.EXAMPLE
.\scripts\install_task_scheduler.ps1

.EXAMPLE
.\scripts\install_task_scheduler.ps1 -TaskName EclipseScalperSupervisor -DelaySeconds 60

.EXAMPLE
.\scripts\install_task_scheduler.ps1 -WhatIf
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$TaskName = "EclipseScalperSupervisor",
    [ValidateRange(0, 3600)]
    [int]$DelaySeconds = 30
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }
$script = Join-Path $repoRoot "scripts\\supervisor.py"
$arg = "`"$script`" --cwd `"$repoRoot`""

Write-Host "Installing scheduled task: $TaskName"
Write-Host "Repo: $repoRoot"
Write-Host "Python: $python"

$delayIso = "PT${DelaySeconds}S"

$action = New-ScheduledTaskAction -Execute $python -Argument $arg -WorkingDirectory $repoRoot
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = $delayIso
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

$target = "ScheduledTask:$TaskName"
if ($PSCmdlet.ShouldProcess($target, "Register/Update startup supervisor task")) {
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
    Write-Host "Task installed."
    Write-Host "To run now:"
    Write-Host "  Start-ScheduledTask -TaskName `"$TaskName`""
} else {
    Write-Host "WhatIf: skipped task registration for $TaskName"
}
