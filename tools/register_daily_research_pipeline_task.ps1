param(
    [string]$TaskName = "EclipseScalper-DailyResearchPipeline",
    [string]$RunAt = "09:00"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path $PSScriptRoot -Parent
$ScriptPath = Join-Path $PSScriptRoot "run_daily_research_pipeline_scheduled.ps1"
$RuntimeDir = Join-Path $RepoRoot "runtime"
$TaskInfoPath = Join-Path $RuntimeDir "daily_research_pipeline_task.json"

if (-not (Test-Path $ScriptPath)) {
    throw "Missing scheduled runner script: $ScriptPath"
}
if (-not (Test-Path $RuntimeDir)) {
    New-Item -ItemType Directory -Path $RuntimeDir | Out-Null
}

$timeParts = $RunAt.Split(":")
if ($timeParts.Count -ne 2) {
    throw "RunAt must be HH:mm format"
}
$hour = [int]$timeParts[0]
$minute = [int]$timeParts[1]
$triggerTime = Get-Date -Hour $hour -Minute $minute -Second 0

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`""
$trigger = New-ScheduledTaskTrigger -Daily -At $triggerTime
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 4)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null

$task = Get-ScheduledTask -TaskName $TaskName
$taskInfo = Get-ScheduledTaskInfo -TaskName $TaskName
$payload = @{
    task_name = $TaskName
    run_at = $RunAt
    timezone = (Get-TimeZone).Id
    script_path = $ScriptPath
    next_run_time = if ($taskInfo.NextRunTime) { $taskInfo.NextRunTime.ToString("o") } else { $null }
    last_run_time = if ($taskInfo.LastRunTime) { $taskInfo.LastRunTime.ToString("o") } else { $null }
    state = [string]$task.State
}
$payload | ConvertTo-Json -Depth 4 | Set-Content -Path $TaskInfoPath -Encoding UTF8
$payload | ConvertTo-Json -Depth 4
