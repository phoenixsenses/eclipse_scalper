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
