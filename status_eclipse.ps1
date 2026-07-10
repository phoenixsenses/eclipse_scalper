param(
    [int]$ProbeWaitSec = 5,
    [switch]$DeepProbe
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path $PSScriptRoot).Path
Set-Location -LiteralPath $RepoRoot

function Read-JsonFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try { return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json } catch { return $null }
}

function Has-Process([array]$Rows, [string]$Needle) {
    foreach ($r in $Rows) {
        $cmd = [string]$r.CommandLine
        if ($cmd -and $cmd.Contains($Needle)) { return $true }
    }
    return $false
}

function Read-PidFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try {
        $raw = (Get-Content -LiteralPath $Path -Raw).Trim()
        if (-not $raw) { return $null }
        return [int]$raw
    } catch {
        return $null
    }
}

function Test-PidAlive([int]$ProcessIdValue) {
    if (-not $ProcessIdValue) { return $false }
    try {
        $p = Get-Process -Id $ProcessIdValue -ErrorAction Stop
        return ($null -ne $p)
    } catch {
        return $false
    }
}

function File-AgeSec([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try {
        return [math]::Round(((Get-Date) - (Get-Item -LiteralPath $Path).LastWriteTime).TotalSeconds, 1)
    } catch {
        return $null
    }
}

$rows = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | Select-Object ProcessId,CommandLine)
$collectorSupervisor = Has-Process $rows "scripts\collector_supervisor.py"
if (-not $collectorSupervisor) { $collectorSupervisor = Has-Process $rows "scripts/collector_supervisor.py" }
$collector = Has-Process $rows "data.microstructure_collector"
$bookticker = Has-Process $rows "data.bookticker_collector"
if (-not $bookticker) { $bookticker = Has-Process $rows "data\bookticker_collector.py" }
if (-not $bookticker) { $bookticker = Has-Process $rows "data/bookticker_collector.py" }
$diary = Has-Process $rows "data.event_diary"
$heartbeat = Has-Process $rows "tools.heartbeat_watchdog"
$s34Chart = Has-Process $rows "tools\s34_live_chart.py"
if (-not $s34Chart) { $s34Chart = Has-Process $rows "tools/s34_live_chart.py" }
$s34Shadow = Has-Process $rows "tools\s34_shadow_paper_runner.py"
if (-not $s34Shadow) { $s34Shadow = Has-Process $rows "tools/s34_shadow_paper_runner.py" }
$s34V02Mirror = Has-Process $rows "tools.s34_v_engine_v02_shadow_mirror"
$s34StateMachineLive = Has-Process $rows "tools.s34_state_machine_live_executor"
$s34StateMachineShadow = Has-Process $rows "tools.s34_realtime_shadow_runner"
$collectionWatchdog = Has-Process $rows "tools.collection_watchdog"
$vEngineLive = Has-Process $rows "tools.s34_v_engine_live_executor"
$collectorPid = Read-PidFile (Join-Path $RepoRoot "logs\pids\microstructure_collector.pid")
$booktickerPid = Read-PidFile (Join-Path $RepoRoot "logs\pids\bookticker_collector.pid")
$diaryPid = Read-PidFile (Join-Path $RepoRoot "logs\pids\event_diary.pid")
$heartbeatPid = Read-PidFile (Join-Path $RepoRoot "logs\pids\heartbeat_watchdog.pid")
$supervisorPid = Read-PidFile (Join-Path $RepoRoot "logs\pids\collector_supervisor.pid")
$s34ChartPid = Read-PidFile (Join-Path $RepoRoot "logs\pids\s34_live_chart.pid")
$s34ShadowPid = Read-PidFile (Join-Path $RepoRoot "logs\pids\s34_shadow_paper_runner.pid")
$s34StateMachineShadowPid = Read-PidFile (Join-Path $RepoRoot "logs\pids\s34_state_machine_shadow_runner.pid")
$s34StateMachineLivePid = Read-PidFile (Join-Path $RepoRoot "logs\pids\s34_state_machine_live_executor.pid")
$vEnginePid = Read-PidFile (Join-Path $RepoRoot "logs\pids\s34_v_engine_live_executor.pid")
$v02MirrorPid = Read-PidFile (Join-Path $RepoRoot "logs\pids\s34_v_engine_v02_shadow_mirror.pid")
$vEngineStatePath = Join-Path $RepoRoot "runtime\s34_v_engine_live_state.json"
$vEngineStateAgeSec = File-AgeSec $vEngineStatePath
if (-not $collectorSupervisor -and $supervisorPid) {
    $collectorSupervisor = Test-PidAlive $supervisorPid
}
if (-not $collector -and $collectorPid) {
    $collector = Test-PidAlive $collectorPid
}
if (-not $bookticker -and $booktickerPid) {
    $bookticker = Test-PidAlive $booktickerPid
}
if (-not $diary -and $diaryPid) {
    $diary = Test-PidAlive $diaryPid
}
if (-not $heartbeat -and $heartbeatPid) {
    $heartbeat = Test-PidAlive $heartbeatPid
}
if (-not $s34Chart -and $s34ChartPid) {
    $s34Chart = Test-PidAlive $s34ChartPid
}
if (-not $s34Shadow -and $s34ShadowPid) {
    $s34Shadow = Test-PidAlive $s34ShadowPid
}
if (-not $vEngineLive -and $vEnginePid) {
    $vEngineLive = Test-PidAlive $vEnginePid
}
if (-not $s34StateMachineLive -and $s34StateMachineLivePid) {
    $s34StateMachineLive = Test-PidAlive $s34StateMachineLivePid
}
if (-not $s34StateMachineShadow -and $s34StateMachineShadowPid) {
    $s34StateMachineShadow = Test-PidAlive $s34StateMachineShadowPid
}
if (-not $s34V02Mirror -and $v02MirrorPid) {
    $s34V02Mirror = Test-PidAlive $v02MirrorPid
}
if (-not $vEngineLive -and $vEngineStateAgeSec -ne $null -and $vEngineStateAgeSec -le 30) {
    $vEngineLive = $true
}

$collectorHb = Read-JsonFile (Join-Path $RepoRoot "logs\collector_heartbeat.json")
$collectorHealth = Read-JsonFile (Join-Path $RepoRoot "logs\health\collector.json")
$booktickerHealth = Read-JsonFile (Join-Path $RepoRoot "logs\health\bookticker.json")
$watchdog = Read-JsonFile (Join-Path $RepoRoot "reports\WATCHDOG_STATUS.json")
$s34ShadowStatus = Read-JsonFile (Join-Path $RepoRoot "reports\research\s34\S34_SHADOW_PAPER_STATUS.json")
$vEngineState = Read-JsonFile $vEngineStatePath

Write-Output "ECLIPSE_STATUS"
Write-Output ("collector_supervisor_alive=" + [string]$collectorSupervisor)
Write-Output ("collector_supervisor_pid=" + [string]$supervisorPid)
Write-Output ("collector_alive=" + [string]$collector)
Write-Output ("collector_pid=" + [string]$collectorPid)
Write-Output ("bookticker_collector_alive=" + [string]$bookticker)
Write-Output ("bookticker_collector_pid=" + [string]$booktickerPid)
Write-Output ("event_diary_alive=" + [string]$diary)
Write-Output ("event_diary_pid=" + [string]$diaryPid)
Write-Output ("heartbeat_watchdog_alive=" + [string]$heartbeat)
Write-Output ("heartbeat_watchdog_pid=" + [string]$heartbeatPid)
Write-Output ("s34_live_chart_alive=" + [string]$s34Chart)
Write-Output ("s34_live_chart_pid=" + [string]$s34ChartPid)
Write-Output ("s34_shadow_paper_runner_alive=" + [string]$s34Shadow)
Write-Output ("s34_shadow_paper_runner_pid=" + [string]$s34ShadowPid)
Write-Output ("s34_state_machine_shadow_runner_alive=" + [string]$s34StateMachineShadow)
Write-Output ("s34_state_machine_shadow_runner_pid=" + [string]$s34StateMachineShadowPid)
Write-Output ("s34_v_engine_v02_shadow_mirror_alive=" + [string]$s34V02Mirror)
Write-Output ("s34_v_engine_v02_shadow_mirror_pid=" + [string]$v02MirrorPid)
$v02MirrorState = Read-JsonFile (Join-Path $RepoRoot "runtime\s34_v_engine_v02_shadow_mirror_state.json")
$v02MirrorCheckpointFile = Read-JsonFile (Join-Path $RepoRoot "runtime\s34_v_engine_v02_shadow_mirror_checkpoint.json")
if ($v02MirrorState) {
    $v02MirrorUptimeSec = $null
    if ($v02MirrorState.process_start_utc) {
        try {
            $v02MirrorUptimeSec = [math]::Round(((Get-Date).ToUniversalTime() - [DateTime]::Parse($v02MirrorState.process_start_utc).ToUniversalTime()).TotalSeconds, 1)
        } catch {
            $v02MirrorUptimeSec = $null
        }
    }
    Write-Output ("s34_v_engine_v02_shadow_mirror_uptime_sec=" + [string]$v02MirrorUptimeSec)
    Write-Output ("s34_v_engine_v02_shadow_mirror_priority=" + [string]$v02MirrorState.priority_class)
    Write-Output ("s34_v_engine_v02_shadow_mirror_interval_sec=" + [string]$v02MirrorState.interval_sec)
    Write-Output ("s34_v_engine_v02_shadow_mirror_last_heartbeat_utc=" + [string]$v02MirrorState.updated_at_utc)
    Write-Output ("s34_v_engine_v02_shadow_mirror_last_success_utc=" + [string]$v02MirrorState.last_success_utc)
    Write-Output ("s34_v_engine_v02_shadow_mirror_last_processed_ts_ms=" + [string]$v02MirrorState.last_processed_ts_ms)
    Write-Output ("s34_v_engine_v02_shadow_mirror_last_tick_duration_ms=" + [string]$v02MirrorState.last_tick_duration_ms)
    Write-Output ("s34_v_engine_v02_shadow_mirror_checkpoint_lag_ms=" + [string]$v02MirrorState.tick_stats.checkpoint_lag_ms)
    Write-Output ("s34_v_engine_v02_shadow_mirror_last_error=" + [string]$v02MirrorState.last_error)
}
if ($v02MirrorCheckpointFile) {
    Write-Output ("s34_v_engine_v02_shadow_mirror_checkpoint_closed_before_utc=" + [string]$v02MirrorCheckpointFile.closed_before_ts_ms)
}
Write-Output ("collection_watchdog_alive=" + [string]$collectionWatchdog)
Write-Output ("s34_v_engine_live_executor_alive=" + [string]$vEngineLive)
Write-Output ("s34_v_engine_live_executor_pid=" + [string]$vEnginePid)
Write-Output ("s34_state_machine_live_executor_alive=" + [string]$s34StateMachineLive)
Write-Output ("s34_state_machine_live_executor_pid=" + [string]$s34StateMachineLivePid)
Write-Output ("s34_v_engine_live_state_age_sec=" + [string]$vEngineStateAgeSec)

if ($collectorHb) {
    Write-Output ("collector_connected=" + [string]$collectorHb.connected)
    Write-Output ("collector_last_message_ts_utc=" + [string]$collectorHb.last_message_ts_utc)
    Write-Output ("collector_last_data_progress_ts_utc=" + [string]$collectorHb.last_data_progress_ts_utc)
    Write-Output ("collector_rest_fallback_enabled=" + [string]$collectorHb.rest_fallback_enabled)
    Write-Output ("collector_rest_fallback_active=" + [string]$collectorHb.rest_fallback_active)
    Write-Output ("collector_rest_last_progress_ts_utc=" + [string]$collectorHb.rest_last_progress_ts_utc)
    Write-Output ("collector_rows_written_since_start=" + (($collectorHb.rows_written_since_start | ConvertTo-Json -Compress)))
    Write-Output ("collector_last_error=" + [string]$collectorHb.last_error)
}
if ($collectorHealth) {
    Write-Output ("collector_health_status=" + [string]$collectorHealth.status)
    Write-Output ("collector_health_ts_utc=" + [string]$collectorHealth.ts_utc)
    Write-Output ("collector_transport_connected=" + [string]$collectorHealth.transport_connected)
    Write-Output ("collector_required_streams_progressing=" + [string]$collectorHealth.required_streams_progressing)
    Write-Output ("collector_liquidation_transport_available=" + [string]$collectorHealth.liquidation_transport_available)
}
if ($booktickerHealth) {
    Write-Output ("bookticker_health_status=" + [string]$booktickerHealth.status)
    Write-Output ("bookticker_health_ts_utc=" + [string]$booktickerHealth.ts_utc)
    Write-Output ("bookticker_connected=" + [string]$booktickerHealth.connected)
    Write-Output ("bookticker_last_tick_ts_utc=" + [string]$booktickerHealth.last_tick_ts_utc)
    Write-Output ("bookticker_last_error=" + [string]$booktickerHealth.last_error)
}
if ($watchdog) {
    Write-Output ("watchdog_overall=" + [string]$watchdog.overall)
    Write-Output ("watchdog_issues=" + (($watchdog.issues | ConvertTo-Json -Compress)))
    Write-Output ("collector_native_ws_status=" + [string]$watchdog.native_ws_status)
    Write-Output ("collector_native_ws_reasons=" + (($watchdog.native_ws_reasons | ConvertTo-Json -Compress)))
    Write-Output ("collector_native_ws_message_age_sec=" + [string]$watchdog.native_websocket.message_age_sec)
    Write-Output ("collector_rest_fallback_active_watchdog=" + [string]$watchdog.rest_fallback.active)
    if ($watchdog.source_freshness) {
        Write-Output ("collector_agg_trades_age_sec=" + [string]$watchdog.source_freshness.agg_trades.age_sec)
        Write-Output ("collector_mark_prices_age_sec=" + [string]$watchdog.source_freshness.mark_prices.age_sec)
        Write-Output ("collector_liquidations_age_sec=" + [string]$watchdog.source_freshness.liquidations.age_sec)
    }
}
if ($s34ShadowStatus) {
    Write-Output ("s34_shadow_paper_updated_at_utc=" + [string]$s34ShadowStatus.updated_at_utc)
    Write-Output ("s34_shadow_paper_total_trades=" + [string]$s34ShadowStatus.total_trades)
    Write-Output ("s34_shadow_paper_open_trades=" + [string]$s34ShadowStatus.open_trades)
    Write-Output ("s34_shadow_paper_closed_trades=" + [string]$s34ShadowStatus.closed_trades)
    Write-Output ("s34_shadow_paper_last_new_trades=" + [string]$s34ShadowStatus.new_trades)
}
if ($vEngineState) {
    Write-Output ("s34_v_engine_live_mode=" + [string]$vEngineState.status.mode)
    Write-Output ("s34_v_engine_live_rule=" + [string]$vEngineState.status.rule)
    Write-Output ("s34_v_engine_live_active=" + (($vEngineState.active | ConvertTo-Json -Compress)))
    Write-Output ("s34_v_engine_live_orders_count=" + [string]$vEngineState.orders.Count)
}

if ($DeepProbe) {
    Write-Output "DATA_PROBE"
    python -m tools.data_layer_probe --db data/microstructure.db --csv data/event_diary.csv --symbols BTCUSDT,ETHUSDT,SOLUSDT --wait-sec $ProbeWaitSec --fresh-sec 180 --require-diary
} else {
    Write-Output "DATA_PROBE skipped(use -DeepProbe for slow DB-level verification)"
}
