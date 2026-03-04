param(
    [string]$TaskName = "EclipseScalperSupervisor",
    [int]$DelaySeconds = 30
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = "python"
$script = Join-Path $repoRoot "scripts\\supervisor.py"
$arg = "`"$script`" --cwd `"$repoRoot`""

Write-Host "Installing scheduled task: $TaskName"
Write-Host "Repo: $repoRoot"

$delayIso = "PT${DelaySeconds}S"

$action = New-ScheduledTaskAction -Execute $python -Argument $arg -WorkingDirectory $repoRoot
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = $delayIso
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null

Write-Host "Task installed."
Write-Host "To run now:"
Write-Host "  Start-ScheduledTask -TaskName `"$TaskName`""

