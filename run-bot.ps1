# run-bot.ps1
# Best-tuned runtime profile (DOGE/ETH quality mode + BTC overrides)

# Risk checklist (live):
# - CONFIRM SCALPER_DRY_RUN=0 is intended
# - FIRST_LIVE_SAFE=1 and allowlist set
# - FIXED_NOTIONAL_USDT + LEVERAGE sized for account
# - MAX_DAILY_LOSS_PCT / MAX_DRAWDOWN_PCT set
# - Correlation caps (CORR_GROUP_*) sane for symbol universe
# - Validate groups: python tools/corr_group_check.py
# - Review error classes: python tools/telemetry_error_classes.py --since-min 60

# Safety first
$env:SCALPER_DRY_RUN = "0"

# Core symbols / signal profile
$env:ACTIVE_SYMBOLS = "DOGEUSDT"
$env:SCALPER_SIGNAL_PROFILE = "micro"
$env:SCALPER_SIGNAL_DIAG = "0"

# Fully enhanced signal stack
$env:SCALPER_ENHANCED = "1"
$env:SCALPER_ENH_TREND_1H = "1"
$env:SCALPER_DEBUG_LOOSE = "0"

# Quality mode for DOGE (higher quality entries)
$env:SCALPER_QUALITY_MODE = "1"
$env:SCALPER_QUALITY_CONF_MIN = "0.60"

# Strategy guards (DOGE tuned)
$env:SCALPER_DATA_MAX_STALE_SEC = "90"
$env:SCALPER_SESSION_UTC = "12-20"
$env:SCALPER_SESSION_MOM_UTC = ""
$env:SCALPER_SESSION_MOM_MIN = "0.0015"
$env:SCALPER_COOLDOWN_LOSSES = "2"
$env:SCALPER_COOLDOWN_MINUTES = "30"

# Trend confirmation (defaults)
$env:SCALPER_TREND_CONFIRM = "1"
$env:SCALPER_TREND_CONFIRM_MODE = "gate"
$env:SCALPER_TREND_CONFIRM_TF = "1h"
$env:SCALPER_TREND_CONFIRM_FAST = "50"
$env:SCALPER_TREND_CONFIRM_SLOW = "200"

# Entry loop (full)
$env:ENTRY_LOOP_MODE = "full"
$env:ENTRY_RESPECT_KILL_SWITCH = "1"

# Sizing + leverage (micro notional)
$env:FIXED_NOTIONAL_USDT = "15"
$env:LEVERAGE = "40"
$env:MARGIN_MODE = "cross"

# Live safety allowlist
$env:FIRST_LIVE_SAFE = "1"
$env:FIRST_LIVE_SYMBOLS = "DOGEUSDT"

# Correlation group caps
$env:CORR_GROUPS = "MEME:DOGEUSDT,SHIBUSDT,PEPEUSDT;MAJOR:BTCUSDT,ETHUSDT"
$env:CORR_GROUP_MAX_POSITIONS = "1"
$env:CORR_GROUP_MAX_NOTIONAL_USDT = "25"
$env:CORR_GROUP_LIMITS = "MEME=1,MAJOR=2"
$env:CORR_GROUP_NOTIONAL = "MEME=25,MAJOR=100"

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
$env:TRAILING_ACTIVATION_RR = "0"
$env:TRAILING_REBUILD_DEBOUNCE_SEC = "999999"
$env:DUAL_TRAILING = "0"
$env:EXIT_STAGNATION_SEC = "900"
$env:EXIT_STAGNATION_ATR = "0.15"

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
