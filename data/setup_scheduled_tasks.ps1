$ErrorActionPreference = "Stop"

$basePath = "C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper"
$taskPrefix = "EclipseScalper"
$commonSettings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -StartWhenAvailable `
  -RestartCount 3 `
  -RestartInterval (New-TimeSpan -Minutes 1) `
  -MultipleInstances IgnoreNew

$trigger = New-ScheduledTaskTrigger -AtLogOn

# Microstructure collector
$collectorAction = New-ScheduledTaskAction `
  -Execute "$basePath\data\start_collector.bat" `
  -WorkingDirectory $basePath
Register-ScheduledTask `
  -TaskName "$taskPrefix\MicroCollector" `
  -Action $collectorAction `
  -Trigger $trigger `
  -Settings $commonSettings `
  -Description "Binance futures microstructure data collector" `
  -Force | Out-Null

# Event diary
$diaryAction = New-ScheduledTaskAction `
  -Execute "$basePath\data\start_diary.bat" `
  -WorkingDirectory $basePath
Register-ScheduledTask `
  -TaskName "$taskPrefix\EventDiary" `
  -Action $diaryAction `
  -Trigger $trigger `
  -Settings $commonSettings `
  -Description "Microstructure event diary logger" `
  -Force | Out-Null

# Runtime bootstrap singleton (IgnoreNew prevents task-level fan-out)
$bootstrapArgs = @(
  '-NoProfile',
  '-ExecutionPolicy', 'Bypass',
  '-Command',
  "Set-Location '$basePath'; python -m execution.bootstrap 2>&1 | Tee-Object -FilePath 'logs\scheduled_bootstrap.log' -Append"
) -join " "
$bootstrapAction = New-ScheduledTaskAction `
  -Execute "powershell.exe" `
  -Argument $bootstrapArgs `
  -WorkingDirectory $basePath
Register-ScheduledTask `
  -TaskName "$taskPrefix\BootstrapRuntime" `
  -Action $bootstrapAction `
  -Trigger $trigger `
  -Settings $commonSettings `
  -Description "Eclipse Scalper runtime bootstrap (single-instance task)" `
  -Force | Out-Null

Write-Host ""
Write-Host "Scheduled tasks created/updated:" -ForegroundColor Green
Write-Host "  - $taskPrefix\MicroCollector"
Write-Host "  - $taskPrefix\EventDiary"
Write-Host "  - $taskPrefix\BootstrapRuntime (MultipleInstances=IgnoreNew)"
Write-Host ""
Write-Host "Verify:"
Write-Host "  schtasks /Query /TN $taskPrefix\BootstrapRuntime /V /FO LIST"
Write-Host ""
Write-Host "Run now:"
Write-Host "  schtasks /Run /TN $taskPrefix\BootstrapRuntime"
