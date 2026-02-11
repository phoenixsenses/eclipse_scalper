# run-bot-ps2.ps1
# Safer live profile (low leverage + small notional)

# Risk checklist (test/live):
# - Set SCALPER_DRY_RUN=1 for sandbox testing
# - FIRST_LIVE_SAFE=1 and allowlist set
# - FIXED_NOTIONAL_USDT + LEVERAGE sized for account
# - MAX_DAILY_LOSS_PCT / MAX_DRAWDOWN_PCT set
# - Correlation caps (CORR_GROUP_*) sane for symbol universe
# - Validate groups: python tools/corr_group_check.py
# - Review error classes: python tools/telemetry_error_classes.py --since-min 60

# Safety first (set to 1 for dry run)
$env:SCALPER_DRY_RUN = "0"

# Core symbols / signal profile
$env:ACTIVE_SYMBOLS = "DOGEUSDT"
$env:SCALPER_SIGNAL_PROFILE = "micro"
$env:SCALPER_SIGNAL_DIAG = "1"
$env:SCALPER_CONFIDENCE_DIAG = "1"
$env:MIN_CONFIDENCE = "0.30"
$env:ENTRY_MIN_CONFIDENCE = "0.30"

# Fully enhanced signal stack
$env:SCALPER_ENHANCED = "1"
$env:SCALPER_ENH_TREND_1H = "1"
$env:SCALPER_DEBUG_LOOSE = "1"
$env:SCALPER_NEAR_MISS_CONF = "0.20"
$env:SCALPER_VOL_REGIME_GUARD = "0"
$env:SCALPER_TREND_CONFIRM_MODE = "bonus"
$env:SCALPER_VWAP_BASE_DIST_DOGE = "0.0035"
$env:SCALPER_VWAP_ATR_MULT_DOGE = "1.0"
$env:SCALPER_DYN_MOM_MULT_DOGE = "0.55"
$env:SCALPER_DYN_MOM_FLOOR_DOGE = "0.00025"

# Notifications (optional — set these if you want Telegram alerts from the scheduled jobs)
$env:TELEGRAM_TOKEN = ""
$env:TELEGRAM_CHAT_ID = ""

# Quality mode for DOGE (higher quality entries)
$env:SCALPER_QUALITY_MODE = "0"
$env:SCALPER_QUALITY_CONF_MIN = "0.60"

# Strategy guards (test defaults)
$env:SCALPER_DATA_MAX_STALE_SEC = "600"
$env:SCALPER_SESSION_UTC = ""
$env:SCALPER_SESSION_MOM_UTC = ""
$env:SCALPER_SESSION_MOM_MIN = "0.0015"
$env:SCALPER_COOLDOWN_LOSSES = "0"
$env:SCALPER_COOLDOWN_MINUTES = "0"

# Trend confirmation (defaults)
$env:SCALPER_TREND_CONFIRM = "1"
$env:SCALPER_TREND_CONFIRM_MODE = "gate"
$env:SCALPER_TREND_CONFIRM_TF = "1h"
$env:SCALPER_TREND_CONFIRM_FAST = "50"
$env:SCALPER_TREND_CONFIRM_SLOW = "200"

# Telemetry helpers (match `.env.example` defaults so alerts/scripts all read the same file)
$env:TELEMETRY_PATH = "logs/telemetry.jsonl"
$env:TELEMETRY_ALERT_SUMMARY = "1"
$env:TELEMETRY_ANOMALY_STATE = "logs/telemetry_anomaly_state.json"
$env:TELEMETRY_ANOMALY_ACTIONS = "logs/telemetry_anomaly_actions.json"
$env:TELEMETRY_DRIFT_PATH = "logs/telemetry_drift.jsonl"
$env:TELEMETRY_GUARD_HISTORY_EVENTS = "logs/telemetry_guard_history_events.jsonl"
$env:ADAPTIVE_GUARD_STATE = "logs/telemetry_adaptive_guard.json"
$env:RECONCILE_FIRST_GATE_SEVERITY_THRESHOLD = "0.85"
$env:RECONCILE_FIRST_GATE_SEVERITY_STREAK_THRESHOLD = "2"
$env:RELIABILITY_GATE_PATH = "logs/reliability_gate.txt"
$env:RELIABILITY_GATE_MAX_REPLAY_MISMATCH = "0"
$env:RELIABILITY_GATE_MAX_INVALID_TRANSITIONS = "0"
$env:RELIABILITY_GATE_MIN_JOURNAL_COVERAGE = "0.90"
$env:RELIABILITY_GATE_MAX_POSITION_MISMATCH = "1"
$env:RELIABILITY_GATE_MAX_ORPHAN_COUNT = "0"
$env:RELIABILITY_GATE_MAX_COVERAGE_GAP_SECONDS = "0"
$env:RELIABILITY_GATE_MAX_REPLACE_RACE_COUNT = "1"
$env:RELIABILITY_GATE_MAX_EVIDENCE_CONTRADICTION_COUNT = "2"
$env:RUNTIME_RELIABILITY_COUPLING = "1"
$env:BELIEF_RUNTIME_GATE_RECOVER_SEC = "120"
$env:BELIEF_RUNTIME_GATE_WARMUP_NOTIONAL_SCALE = "0.50"
$env:BELIEF_RUNTIME_GATE_WARMUP_LEVERAGE_SCALE = "0.60"
$env:BELIEF_RUNTIME_GATE_CRITICAL_TRIP_THRESHOLD = "3.0"
$env:BELIEF_RUNTIME_GATE_CRITICAL_CLEAR_THRESHOLD = "1.5"
$env:BELIEF_RECONCILE_FIRST_GATE_COUNT_THRESHOLD = "2"
$env:BELIEF_RECONCILE_FIRST_GATE_SEVERITY_THRESHOLD = "0.85"
$env:BELIEF_RECONCILE_FIRST_GATE_STREAK_THRESHOLD = "2"
$env:BELIEF_POST_RED_WARMUP_SEC = "180"
$env:BELIEF_POST_RED_WARMUP_NOTIONAL_SCALE = "0.60"
$env:BELIEF_POST_RED_WARMUP_LEVERAGE_SCALE = "0.70"
$env:BELIEF_POST_RED_WARMUP_MIN_CONF_EXTRA = "0.06"
$env:BELIEF_POST_RED_WARMUP_COOLDOWN_EXTRA_SEC = "20"
$env:ADAPTIVE_GUARD_DURATION_SEC = "900"
$env:ADAPTIVE_GUARD_DRIFT_DURATION_SEC = "900"
$env:ADAPTIVE_GUARD_EXIT_DURATION_SEC = "600"
$env:ADAPTIVE_GUARD_PARTIAL_DELTA = "0.10"
$env:ADAPTIVE_GUARD_PARTIAL_DURATION_SEC = "600"
$env:ADAPTIVE_GUARD_RETRY_DELTA = "0.15"
$env:ADAPTIVE_GUARD_RETRY_DURATION_SEC = "600"
$env:ADAPTIVE_GUARD_TELEMETRY_DURATION_SEC = "600"
$env:ADAPTIVE_GUARD_GUARD_HISTORY_DELTA = "0.10"
$env:ADAPTIVE_GUARD_GUARD_HISTORY_DURATION_SEC = "900"
$env:ADAPTIVE_GUARD_GUARD_HISTORY_LEVERAGE_SCALE = "0.85"
$env:ADAPTIVE_GUARD_GUARD_HISTORY_LEVERAGE_DURATION_SEC = "900"
$env:ADAPTIVE_GUARD_GUARD_HISTORY_NOTIONAL_SCALE = "0.80"
$env:ADAPTIVE_GUARD_GUARD_HISTORY_NOTIONAL_DURATION_SEC = "900"
$env:ADAPTIVE_GUARD_LEVERAGE_SCALE = "1"
$env:ADAPTIVE_GUARD_NOTIONAL_SCALE = "1"
$env:ADAPTIVE_GUARD_QTY_SCALE = "1"
$env:CORE_HEALTH_EQUITY = "100000"

# Strategy audit (test only)
$env:SCALPER_AUDIT = "1"
$env:SCALPER_AUDIT_COOLDOWN_SEC = "5"

# Entry follow-through confirmation (test)
$env:ENTRY_FOLLOW_THROUGH = "1"
$env:ENTRY_FOLLOW_THROUGH_MIN_MOVE_PCT = "0.0005"
$env:ENTRY_CONF_SCALE_ENABLED = "1"
$env:ENTRY_CONF_SCALE_MIN_CONF = "0.40"
$env:ENTRY_CONF_SCALE_MAX_CONF = "0.75"
$env:ENTRY_CONF_SCALE_MIN = "0.60"
$env:ENTRY_CONF_SCALE_MAX = "1.00"

# DOGE-only volatility floor (filters low-ATR chop)
$env:SCALPER_ATR_PCT_MIN_DOGE = "0.0010"

# ETH overrides removed (ETH disabled for now)


# Entry loop (full)
$env:ENTRY_LOOP_MODE = "full"
$env:ENTRY_RESPECT_KILL_SWITCH = "1"

# Sizing + leverage (safer)
$env:FIXED_NOTIONAL_USDT = "6"
$env:FIXED_NOTIONAL_USDT_DOGE = "6"
$env:LEVERAGE = "1"
$env:MARGIN_MODE = "isolated"

# Live safety allowlist
$env:FIRST_LIVE_SAFE = "1"
$env:FIRST_LIVE_SYMBOLS = "DOGEUSDT"
$env:FIRST_LIVE_MAX_NOTIONAL_USDT = "6"

# Correlation group caps
$env:CORR_GROUPS = "MEME:DOGEUSDT,SHIBUSDT,PEPEUSDT;MAJOR:BTCUSDT,ETHUSDT"
$env:CORR_GROUP_MAX_POSITIONS = "1"
$env:CORR_GROUP_MAX_NOTIONAL_USDT = "25"
$env:CORR_GROUP_LIMITS = "MEME=1,MAJOR=2"
$env:CORR_GROUP_NOTIONAL = "MEME=25,MAJOR=100"
$env:CORR_GROUP_SCALE_ENABLED = "1"
$env:CORR_GROUP_SCALE = "0.70"
$env:CORR_GROUP_SCALE_MIN = "0.25"
$env:CORR_GROUP_SCALE_BY_GROUP = ""
$env:CORR_GROUP_EXPOSURE_SCALE_ENABLED = "1"
$env:CORR_GROUP_EXPOSURE_SCALE = "0.75"
$env:CORR_GROUP_EXPOSURE_SCALE_MIN = "0.30"
$env:CORR_GROUP_EXPOSURE_REF_NOTIONAL = "25"
$env:CORR_GROUP_EXPOSURE_SCALE_BY_GROUP = ""
$env:CORR_GROUP_EXPOSURE_SCALE_MIN_BY_GROUP = ""
$env:CORR_GROUP_EXPOSURE_REF_NOTIONAL_BY_GROUP = ""

# Daily risk halt
$env:MAX_DAILY_LOSS_PCT = "0.02"
$env:MAX_DRAWDOWN_PCT = "0.06"
$env:KILL_DAILY_HALT_SEC = "3600"
$env:KILL_DRAWDOWN_HALT_SEC = "3600"
$env:KILL_DAILY_HALT_UNTIL_UTC = "1"

# Router
$env:ROUTER_AUTO_CLIENT_ID = "1"
$env:ROUTER_RETRY_ALERT_TRIES = "4"
$env:ROUTER_RETRY_ALERT_COOLDOWN_SEC = "60"

# Position manager
$env:POSMGR_ENABLED = "1"
$env:POSMGR_TICK_SEC = "2"
$env:POSMGR_ENSURE_STOP = "1"
$env:POSMGR_STOP_CHECK_SEC = "90"
$env:POSMGR_STOP_RESTORE_COOLDOWN_SEC = "300"
$env:POSMGR_OPEN_ORDERS_MIN_INTERVAL_SEC = "12"
$env:POSMGR_BE_STOP_TOLERANCE_PCT = "0.005"

# Exit & trailing (keep defaults unless you want to override)
$env:EXIT_ENABLED = "1"
$env:EXIT_TICK_SEC = "2"
$env:EXIT_MAX_HOLD_SEC = "2400"
$env:EXIT_TIME_COOLDOWN_SEC = "10"
$env:EXIT_STAGNATION_SEC = "900"
$env:EXIT_STAGNATION_ATR = "0.15"
$env:EXIT_MAX_HOLD_SEC_DOGE = "1800"
$env:EXIT_STAGNATION_SEC_DOGE = "600"
$env:EXIT_STAGNATION_ATR_DOGE = "0.12"
$env:EXIT_ATR_SCALE_ENABLED = "1"
$env:EXIT_ATR_SCALE_REF_PCT = "0.003"
$env:EXIT_ATR_SCALE_MIN = "0.6"
$env:EXIT_ATR_SCALE_MAX = "1.6"
$env:TRAILING_ACTIVATION_RR = "0"
$env:TRAILING_REBUILD_DEBOUNCE_SEC = "999999"
$env:DUAL_TRAILING = "0"

# Exit telemetry guard (forces earlier exits when telemetry reports high exposure)
$env:EXIT_TELEMETRY_HIGH_EXPOSURE_USDT = "0"
$env:EXIT_TELEMETRY_FORCE_HOLD_SEC = "0"
$env:EXIT_TELEMETRY_COOLDOWN_MULT = "0.5"
$env:EXIT_TELEMETRY_ALERT_INTERVAL_SEC = "300"
$env:TELEMETRY_GUARD_HISTORY_ACTIONS = "logs/telemetry_guard_history_actions.json"
$env:EXIT_GUARD_HISTORY_HOLD_SCALE = "0.70"
$env:EXIT_GUARD_HISTORY_STAGNATION_SCALE = "0.70"
$env:EXIT_SIGNAL_FEEDBACK_PATH = "logs/signal_exit_feedback.json"
$env:EXIT_SIGNAL_FEEDBACK_MIN_RATIO = "0.25"
$env:EXIT_SIGNAL_FEEDBACK_MIN_COUNT = "3"
$env:EXIT_SIGNAL_FEEDBACK_HOLD_SCALE = "0.70"
$env:EXIT_SIGNAL_FEEDBACK_STAGNATION_SCALE = "0.70"

# Exit tuning (best backtest-aligned)
$env:STOP_ATR_MULT = "0.8"
$env:TP1_RR_MULT = "1.0"
$env:TP2_RR_MULT = "2.0"
$env:TP_ATR_MULT = "1.8"

# Reconcile protection refresh (partial-fill anti-churn)
$env:RECONCILE_STOP_MIN_COVERAGE_RATIO = "0.98"
$env:RECONCILE_STOP_REFRESH_MIN_DELTA_RATIO = "0.10"
$env:RECONCILE_STOP_REFRESH_MIN_DELTA_ABS = "0.0"
$env:RECONCILE_STOP_REFRESH_MAX_INTERVAL_SEC = "45"
$env:RECONCILE_STOP_REFRESH_FORCE_COVERAGE_RATIO = "0.80"
$env:RECONCILE_STOP_REPLACE_RETRIES = "2"

# TP protection refresh (opt-in symmetry with stop coverage)
$env:GUARDIAN_ENSURE_TP = "0"
$env:RECONCILE_TP_MIN_COVERAGE_RATIO = "0.98"
$env:RECONCILE_TP_REFRESH_MIN_DELTA_RATIO = "0.10"
$env:RECONCILE_TP_REFRESH_MIN_DELTA_ABS = "0.0"
$env:RECONCILE_TP_REFRESH_MAX_INTERVAL_SEC = "45"
$env:RECONCILE_TP_REFRESH_FORCE_COVERAGE_RATIO = "0.80"
$env:RECONCILE_TP_REPLACE_RETRIES = "2"
$env:RECONCILE_TP_FALLBACK_PCT = "0.0050"

# ----------------------------
# Preflight + smoke controls
# ----------------------------
$safeProfileMaxLeverage = 1.0
if ($env:SAFE_PROFILE_MAX_LEVERAGE -and $env:SAFE_PROFILE_MAX_LEVERAGE.Trim() -ne "") {
  [void][double]::TryParse($env:SAFE_PROFILE_MAX_LEVERAGE, [ref]$safeProfileMaxLeverage)
}
$runPreflightOnly = ($env:RUN_PREFLIGHT_ONLY -eq "1")
$runSmoke = ($env:RUN_SMOKE -eq "1")

function Ensure-ParentDir([string]$pathValue) {
  if (-not $pathValue) { return }
  $parent = Split-Path -Parent $pathValue
  if (-not $parent) { return }
  New-Item -ItemType Directory -Force -Path $parent | Out-Null
}

function Invoke-Preflight {
  Write-Host "Running safe profile preflight..."
  $errorList = New-Object System.Collections.Generic.List[string]
  $warnList = New-Object System.Collections.Generic.List[string]

  New-Item -ItemType Directory -Force -Path "logs" | Out-Null
  Ensure-ParentDir -pathValue $env:TELEMETRY_PATH
  Ensure-ParentDir -pathValue $env:TELEMETRY_DRIFT_PATH
  Ensure-ParentDir -pathValue $env:TELEMETRY_GUARD_HISTORY_EVENTS
  Ensure-ParentDir -pathValue $env:ADAPTIVE_GUARD_STATE
  Ensure-ParentDir -pathValue $env:RELIABILITY_GATE_PATH
  if ($env:EVENT_JOURNAL_PATH) { Ensure-ParentDir -pathValue $env:EVENT_JOURNAL_PATH }

  if ($env:SCALPER_DRY_RUN -notin @("0","1")) {
    $errorList.Add("SCALPER_DRY_RUN must be 0 or 1. Set `$env:SCALPER_DRY_RUN='1' for testing or '0' for live.")
  }
  if (-not $env:ACTIVE_SYMBOLS -or $env:ACTIVE_SYMBOLS.Trim() -eq "") {
    $errorList.Add("ACTIVE_SYMBOLS cannot be empty.")
  }
  if ((($env:MARGIN_MODE) + "").ToLowerInvariant() -ne "isolated") {
    $errorList.Add("MARGIN_MODE must be 'isolated' for this safe profile.")
  }

  [double]$notional = 0.0
  [double]$lev = 0.0
  [double]$firstLiveCap = 0.0
  [double]$dailyLoss = 0.0
  [double]$drawdown = 0.0
  [void][double]::TryParse(($env:FIXED_NOTIONAL_USDT + ""), [ref]$notional)
  [void][double]::TryParse(($env:LEVERAGE + ""), [ref]$lev)
  [void][double]::TryParse(($env:FIRST_LIVE_MAX_NOTIONAL_USDT + ""), [ref]$firstLiveCap)
  [void][double]::TryParse(($env:MAX_DAILY_LOSS_PCT + ""), [ref]$dailyLoss)
  [void][double]::TryParse(($env:MAX_DRAWDOWN_PCT + ""), [ref]$drawdown)

  if ($notional -le 0) {
    $errorList.Add("FIXED_NOTIONAL_USDT must be > 0.")
  }
  if ($lev -le 0) {
    $errorList.Add("LEVERAGE must be > 0.")
  } elseif ($lev -gt $safeProfileMaxLeverage) {
    $errorList.Add("LEVERAGE=$lev exceeds SAFE_PROFILE_MAX_LEVERAGE=$safeProfileMaxLeverage.")
  }

  if ($env:FIRST_LIVE_SAFE -eq "1") {
    if (-not $env:FIRST_LIVE_SYMBOLS -or $env:FIRST_LIVE_SYMBOLS.Trim() -eq "") {
      $errorList.Add("FIRST_LIVE_SAFE=1 requires FIRST_LIVE_SYMBOLS.")
    }
    if ($firstLiveCap -le 0) {
      $errorList.Add("FIRST_LIVE_SAFE=1 requires FIRST_LIVE_MAX_NOTIONAL_USDT > 0.")
    } elseif ($notional -gt $firstLiveCap) {
      $errorList.Add("FIXED_NOTIONAL_USDT ($notional) must be <= FIRST_LIVE_MAX_NOTIONAL_USDT ($firstLiveCap).")
    }
  }

  if ($dailyLoss -le 0 -or $dailyLoss -gt 0.05) {
    $errorList.Add("MAX_DAILY_LOSS_PCT must be within (0, 0.05]. Current=$dailyLoss.")
  }
  if ($drawdown -le 0 -or $drawdown -gt 0.20) {
    $errorList.Add("MAX_DRAWDOWN_PCT must be within (0, 0.20]. Current=$drawdown.")
  }
  if ($drawdown -gt 0 -and $dailyLoss -gt 0 -and $drawdown -lt $dailyLoss) {
    $errorList.Add("MAX_DRAWDOWN_PCT must be >= MAX_DAILY_LOSS_PCT.")
  }

  $preflightJson = "logs/preflight_check.json"
  python tools/preflight_check.py --max-leverage $safeProfileMaxLeverage --json-out $preflightJson
  if ($LASTEXITCODE -ne 0) {
    $errorList.Add("tools/preflight_check.py failed. Review $preflightJson for details.")
  }

  python tools/corr_group_check.py
  if ($LASTEXITCODE -ne 0) {
    $errorList.Add("tools/corr_group_check.py failed. Fix CORR_GROUP* settings.")
  }

  Write-Host ""
  Write-Host "===== Safe Live Profile Summary ====="
  Write-Host ("Mode: {0}" -f ($(if ($env:SCALPER_DRY_RUN -eq "1") { "DRY-RUN" } else { "LIVE" })))
  Write-Host ("Symbols: {0}" -f $env:ACTIVE_SYMBOLS)
  Write-Host ("Sizing: FIXED_NOTIONAL_USDT={0} LEVERAGE={1} MARGIN_MODE={2}" -f $env:FIXED_NOTIONAL_USDT, $env:LEVERAGE, $env:MARGIN_MODE)
  Write-Host ("First-live-safe: enabled={0} allowlist={1} cap={2}" -f $env:FIRST_LIVE_SAFE, $env:FIRST_LIVE_SYMBOLS, $env:FIRST_LIVE_MAX_NOTIONAL_USDT)
  Write-Host ("Risk limits: MAX_DAILY_LOSS_PCT={0} MAX_DRAWDOWN_PCT={1}" -f $env:MAX_DAILY_LOSS_PCT, $env:MAX_DRAWDOWN_PCT)
  Write-Host ("Correlation caps: groups={0} max_positions={1} max_notional={2}" -f $env:CORR_GROUPS, $env:CORR_GROUP_MAX_POSITIONS, $env:CORR_GROUP_MAX_NOTIONAL_USDT)
  Write-Host ("Reliability gate: coupling={0} mismatch_max={1} invalid_max={2} coverage_min={3}" -f $env:RUNTIME_RELIABILITY_COUPLING, $env:RELIABILITY_GATE_MAX_REPLAY_MISMATCH, $env:RELIABILITY_GATE_MAX_INVALID_TRANSITIONS, $env:RELIABILITY_GATE_MIN_JOURNAL_COVERAGE)
  Write-Host ("Runtime critical trip: threshold={0} clear={1}" -f $env:BELIEF_RUNTIME_GATE_CRITICAL_TRIP_THRESHOLD, $env:BELIEF_RUNTIME_GATE_CRITICAL_CLEAR_THRESHOLD)
  Write-Host ("Reconcile-first gate: count={0} severity={1} streak={2}" -f $env:BELIEF_RECONCILE_FIRST_GATE_COUNT_THRESHOLD, $env:BELIEF_RECONCILE_FIRST_GATE_SEVERITY_THRESHOLD, $env:BELIEF_RECONCILE_FIRST_GATE_STREAK_THRESHOLD)
  Write-Host ("Recovery staging: runtime_warmup_sec={0} post_red_warmup_sec={1}" -f $env:BELIEF_RUNTIME_GATE_RECOVER_SEC, $env:BELIEF_POST_RED_WARMUP_SEC)
  Write-Host "Entry blocks may come from: runtime reliability gate, reconcile-first pressure, per-symbol debt, correlation budget."
  Write-Host "Recovery guidance: clear gate causes (replay/coverage/contradiction), then wait for warmup timers to expire."

  if ($warnList.Count -gt 0) {
    Write-Host ""
    Write-Host "Warnings:"
    foreach ($w in $warnList) { Write-Host (" - {0}" -f $w) }
  }
  if ($errorList.Count -gt 0) {
    Write-Host ""
    Write-Host "Preflight FAILED:"
    foreach ($e in $errorList) { Write-Host (" - {0}" -f $e) }
    return $false
  }

  Write-Host "Preflight PASSED."
  return $true
}

if (-not (Invoke-Preflight)) {
  exit 2
}

if ($runPreflightOnly) {
  Write-Host "RUN_PREFLIGHT_ONLY=1 set, exiting after successful preflight."
  exit 0
}

if ($runSmoke) {
  Write-Host "RUN_SMOKE=1 set, running telemetry smoke assertion..."
  $smokeStatePath = "logs/telemetry_dashboard_notify_state.smoke.json"
  @{
    level = "normal"
    recovery_stage_latest = "GREEN"
    recovery_red_lock_streak = 0
  } | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $smokeStatePath

  python tools/telemetry_smoke_assert.py `
    --state $smokeStatePath `
    --expected-level normal `
    --expected-stage GREEN `
    --expected-red-lock-streak 0
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Smoke assertion failed. Refusing launch."
    exit 3
  }
}

Write-Host "Environment configured. Starting bot..."
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
$env:PYTHONPATH = $scriptDir
if (-not $env:PYTHONPATH -or $env:PYTHONPATH -eq "") {
  $env:PYTHONPATH = $scriptDir
} else {
  $env:PYTHONPATH = "$scriptDir;$env:PYTHONPATH"
}
python -m execution.bootstrap
