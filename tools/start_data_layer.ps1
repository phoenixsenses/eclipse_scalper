param(
    [string]$Symbols = "BTCUSDT,ETHUSDT",
    [switch]$ForceRestart,
    [switch]$OnlyCollector,
    [switch]$OnlyDiary
)

$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -Path $Path -ItemType Directory | Out-Null
    }
}

function Write-PidState([string]$PidFile, [string]$MetaFile, [int]$Pid, [string]$ModuleName, [string[]]$Args, [string]$Launcher) {
    Set-Content -LiteralPath $PidFile -Value ([string]$Pid) -Encoding ascii
    $meta = @{
        role = "data_layer"
        pid = [int]$Pid
        module = $ModuleName
        cmdline_sig = "$PythonExe $($Args -join ' ')"
        started_at = [DateTime]::UtcNow.ToString("o")
        launcher = $Launcher
        cwd = $RepoRoot
    }
    ($meta | ConvertTo-Json -Depth 4) | Set-Content -LiteralPath $MetaFile -Encoding utf8
}

function Try-StopPid([string]$PidFile) {
    if (-not (Test-Path -LiteralPath $PidFile)) {
        return
    }
    $raw = (Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if (-not $raw) {
        return
    }
    $targetPid = 0
    if (-not [int]::TryParse($raw.Trim(), [ref]$targetPid)) {
        return
    }
    try {
        Stop-Process -Id $targetPid -Force -ErrorAction SilentlyContinue
    } catch {
    }
}

function Get-LivePidFromFile([string]$PidFile) {
    if (-not (Test-Path -LiteralPath $PidFile)) {
        return $null
    }
    $raw = (Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if (-not $raw) {
        return $null
    }
    $targetPid = 0
    if (-not [int]::TryParse($raw.Trim(), [ref]$targetPid)) {
        return $null
    }
    try {
        $p = Get-Process -Id $targetPid -ErrorAction Stop
        if ($p) { return $targetPid }
    } catch {
    }
    return $null
}

function Get-MatchingPids([string]$ModuleName) {
    $out = @()
    try {
        $rows = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId,CommandLine
        foreach ($row in $rows) {
            $cl = [string]$row.CommandLine
            if ($cl -and $cl.Contains($ModuleName)) {
                $out += [int]$row.ProcessId
            }
        }
    } catch {
        $metaFile = $null
        if ($ModuleName -eq "data.microstructure_collector") {
            $metaFile = $MicroMetaFile
        } elseif ($ModuleName -eq "data.event_diary") {
            $metaFile = $DiaryMetaFile
        }
        if ($metaFile -and (Test-Path -LiteralPath $metaFile)) {
            try {
                $meta = Get-Content -LiteralPath $metaFile -Raw | ConvertFrom-Json
                $pid = [int]($meta.pid)
                $cmd = [string]($meta.cmdline_sig)
                if ($pid -gt 0 -and $cmd -and $cmd.Contains($ModuleName)) {
                    $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
                    if ($proc) {
                        $out += $pid
                    }
                }
            } catch {
            }
        }
    }
    return $out
}

function Stop-MatchingCollectors([bool]$includeCollector, [bool]$includeDiary) {
    try {
        $rows = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Select-Object ProcessId,CommandLine
        foreach ($row in $rows) {
            $cl = [string]$row.CommandLine
            $matchCollector = $includeCollector -and $cl -and $cl.Contains("data.microstructure_collector")
            $matchDiary = $includeDiary -and $cl -and $cl.Contains("data.event_diary")
            if ($matchCollector -or $matchDiary) {
                try {
                    Stop-Process -Id ([int]$row.ProcessId) -Force -ErrorAction SilentlyContinue
                } catch {
                }
            }
        }
    } catch {
        # best-effort only; PID-file based stop already attempted.
    }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$LogsDir = Join-Path $RepoRoot "logs"
$PidDir = Join-Path $LogsDir "pids"
Ensure-Dir $LogsDir
Ensure-Dir $PidDir

$MicroOutLog = Join-Path $LogsDir "microstructure_collector.log"
$MicroErrLog = Join-Path $LogsDir "microstructure_collector.err.log"
$DiaryOutLog = Join-Path $LogsDir "event_diary.log"
$DiaryErrLog = Join-Path $LogsDir "event_diary.err.log"
$RunsLog = Join-Path $LogsDir "data_layer_runs.jsonl"

$MicroPidFile = Join-Path $PidDir "microstructure_collector.pid"
$DiaryPidFile = Join-Path $PidDir "event_diary.pid"
$MicroMetaFile = Join-Path $PidDir "microstructure_collector.json"
$DiaryMetaFile = Join-Path $PidDir "event_diary.json"

$PythonCmd = Get-Command python -ErrorAction Stop
$PythonExe = [string]$PythonCmd.Source

$MicroArgs = @(
    "-u",
    "-m", "data.microstructure_collector",
    "--symbols", $Symbols,
    "--db-path", "data/microstructure.db"
)
$DiaryArgs = @(
    "-u",
    "-m", "data.event_diary",
    "--db-path", "data/microstructure.db",
    "--csv-path", "data/event_diary.csv"
)

$startCollector = $true
$startDiary = $true
if ($OnlyCollector -and $OnlyDiary) {
    throw "OnlyCollector and OnlyDiary cannot both be set."
}
if ($OnlyCollector) {
    $startDiary = $false
}
if ($OnlyDiary) {
    $startCollector = $false
}

if ($ForceRestart) {
    if ($startCollector) {
        Try-StopPid -PidFile $MicroPidFile
    }
    if ($startDiary) {
        Try-StopPid -PidFile $DiaryPidFile
    }
    Stop-MatchingCollectors -includeCollector $startCollector -includeDiary $startDiary
    Start-Sleep -Seconds 1
}

$existingCollectorPids = @()
$existingDiaryPids = @()
if (-not $ForceRestart) {
    $pidFileCollector = Get-LivePidFromFile -PidFile $MicroPidFile
    if ($pidFileCollector) { $existingCollectorPids += [int]$pidFileCollector }
    $pidFileDiary = Get-LivePidFromFile -PidFile $DiaryPidFile
    if ($pidFileDiary) { $existingDiaryPids += [int]$pidFileDiary }
    $existingCollectorPids += Get-MatchingPids -ModuleName "data.microstructure_collector"
    $existingDiaryPids += Get-MatchingPids -ModuleName "data.event_diary"
    $existingCollectorPids = @($existingCollectorPids | Sort-Object -Unique)
    $existingDiaryPids = @($existingDiaryPids | Sort-Object -Unique)
    if ($startCollector -and $existingCollectorPids.Count -gt 0) {
        $startCollector = $false
        Write-PidState -PidFile $MicroPidFile -MetaFile $MicroMetaFile -Pid ([int]$existingCollectorPids[0]) -ModuleName "data.microstructure_collector" -Args @($MicroArgs) -Launcher "start_data_layer.ps1:refresh_existing"
        Write-Output ("ALREADY_RUNNING module=microstructure_collector pids=" + ($existingCollectorPids -join ","))
        Write-Output "micro_stdout=$MicroOutLog"
        Write-Output "micro_stderr=$MicroErrLog"
    }
    if ($startDiary -and $existingDiaryPids.Count -gt 0) {
        $startDiary = $false
        Write-PidState -PidFile $DiaryPidFile -MetaFile $DiaryMetaFile -Pid ([int]$existingDiaryPids[0]) -ModuleName "data.event_diary" -Args @($DiaryArgs) -Launcher "start_data_layer.ps1:refresh_existing"
        Write-Output ("ALREADY_RUNNING module=event_diary pids=" + ($existingDiaryPids -join ","))
        Write-Output "event_diary_stdout=$DiaryOutLog"
        Write-Output "event_diary_stderr=$DiaryErrLog"
    }
}

$pMicro = $null
$pDiary = $null
if ($startCollector) {
    $pMicro = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $MicroArgs `
        -WorkingDirectory $RepoRoot `
        -NoNewWindow:$false `
        -PassThru `
        -RedirectStandardOutput $MicroOutLog `
        -RedirectStandardError $MicroErrLog
    Write-PidState -PidFile $MicroPidFile -MetaFile $MicroMetaFile -Pid ([int]$pMicro.Id) -ModuleName "data.microstructure_collector" -Args @($MicroArgs) -Launcher "start_data_layer.ps1"
}

if ($startDiary) {
    $pDiary = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $DiaryArgs `
        -WorkingDirectory $RepoRoot `
        -NoNewWindow:$false `
        -PassThru `
        -RedirectStandardOutput $DiaryOutLog `
        -RedirectStandardError $DiaryErrLog
    Write-PidState -PidFile $DiaryPidFile -MetaFile $DiaryMetaFile -Pid ([int]$pDiary.Id) -ModuleName "data.event_diary" -Args @($DiaryArgs) -Launcher "start_data_layer.ps1"
}

$tsUtc = [DateTime]::UtcNow.ToString("o")
$microRec = @{
    ts_utc = $tsUtc
    name = "microstructure_collector"
    pid = if ($pMicro) { $pMicro.Id } else { $null }
    cmd = "$PythonExe $($MicroArgs -join ' ')"
    stdout_log = $MicroOutLog
    stderr_log = $MicroErrLog
}
$diaryRec = @{
    ts_utc = $tsUtc
    name = "event_diary"
    pid = if ($pDiary) { $pDiary.Id } else { $null }
    cmd = "$PythonExe $($DiaryArgs -join ' ')"
    stdout_log = $DiaryOutLog
    stderr_log = $DiaryErrLog
}
if ($startCollector) {
    Add-Content -LiteralPath $RunsLog -Value (($microRec | ConvertTo-Json -Compress))
}
if ($startDiary) {
    Add-Content -LiteralPath $RunsLog -Value (($diaryRec | ConvertTo-Json -Compress))
}

Write-Output "STARTED data_layer"
Write-Output "repo_root=$RepoRoot"
Write-Output "python_exe=$PythonExe"
Write-Output "symbols=$Symbols"
if ($startCollector) {
    Write-Output "micro_pid=$($pMicro.Id)"
    Write-Output "micro_stdout=$MicroOutLog"
    Write-Output "micro_stderr=$MicroErrLog"
}
if ($startDiary) {
    Write-Output "event_diary_pid=$($pDiary.Id)"
    Write-Output "event_diary_stdout=$DiaryOutLog"
    Write-Output "event_diary_stderr=$DiaryErrLog"
}
