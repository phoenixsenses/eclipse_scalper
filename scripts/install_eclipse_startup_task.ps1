param(
    [string]$TaskName = "EclipseScalperDataStack",
    [string]$Symbols = "BTCUSDT,ETHUSDT,SOLUSDT"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$StartScript = Join-Path $RepoRoot "start_eclipse.ps1"

if (-not (Test-Path -LiteralPath $StartScript)) {
    throw "Missing start script: $StartScript"
}

$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$StartScript`" -Symbols `"$Symbols`""

$Trigger = New-ScheduledTaskTrigger -AtLogOn
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Principal $Principal `
    -Settings $Settings `
    -Force | Out-Null

Write-Output "INSTALLED task=$TaskName"
Write-Output "action=powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$StartScript`" -Symbols `"$Symbols`""
