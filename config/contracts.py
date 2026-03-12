# config/contracts.py — Single source of truth for config parameter classification
#
# Three tiers:
#   IMMUTABLE  — structural params, require restart (e.g., LEVERAGE, ACTIVE_SYMBOLS)
#   DANGEROUS  — risk-critical, hot-reloadable with CRITICAL warning + audit
#   SAFE       — freely hot-reloadable (thresholds, timeouts, notifications)
#
# Consumed by: config/hot_reload.py, dashboard, API, documentation generators.

from __future__ import annotations

from typing import FrozenSet


# ── IMMUTABLE: require restart, NEVER hot-reloaded ────────────────────
IMMUTABLE: FrozenSet[str] = frozenset({
    "ACTIVE_SYMBOLS",       # needs data loop restart
    "LEVERAGE",             # affects open positions
    "TIMEFRAME",            # needs data re-fetch
    "TIMEFRAME_5M",
    "TIMEFRAME_15M",
    "KILL_SWITCH_ENABLED",  # safety-critical toggle
    "CONFIG_VERSION",
    "CONFIG_FORGED_DATE",
    "TRADING_HOURS_UTC",    # needs entry loop restart
})


# ── DANGEROUS: hot-reloadable with CRITICAL warning + audit ──────────
# These affect risk models and kill switch behavior.
# Changing mid-trade can cause unexpected risk exposure.
# Controlled by HOT_RELOAD_DANGEROUS_CONFIRM env var (default: allowed with warning).
DANGEROUS: FrozenSet[str] = frozenset({
    "MAX_RISK_PER_TRADE",           # changing mid-position violates risk model
    "MAX_PORTFOLIO_HEAT",           # could allow over-exposure
    "MAX_DAILY_LOSS_PCT",           # reducing mid-day triggers unnecessary kill switch
    "KILL_SWITCH_COOLDOWN_SEC",     # reducing mid-halt shortens safety cooldown
    "KILL_MAX_DATA_STALENESS_SEC",  # loosening masks stale data
    "KILL_MAX_API_ERROR_RATE",      # loosening masks API degradation
})


# ── SAFE: freely hot-reloadable ──────────────────────────────────────
SAFE: FrozenSet[str] = frozenset({
    # Risk limits (non-dangerous)
    "MAX_CONCURRENT_POSITIONS",
    "MIN_RISK_DOLLARS",
    "CORRELATION_HEAT_CAP",
    "SESSION_EQUITY_PEAK_PROTECTION_PCT",
    "VELOCITY_DRAWDOWN_PCT",
    "VELOCITY_MINUTES",
    # Entry thresholds
    "MIN_CONFIDENCE",
    "MIN_CONFIDENCE_HIGH_VOL",
    "ENTRY_MIN_CONFIDENCE",
    "ENTRY_POLL_SEC",
    "ENTRY_PER_SYMBOL_GAP_SEC",
    "ENTRY_LOCAL_COOLDOWN_SEC",
    "ENTRY_ROUTER_RETRIES",
    "MIN_ATR_PCT_FOR_ENTRY",
    # Sizing
    "FIXED_NOTIONAL_USDT",
    "FIXED_QTY",
    "MIN_ENTRY_QTY",
    "MIN_NOTIONAL_USDT",
    "MIN_MARGIN_USDT",
    # Stop/TP
    "STOP_ATR_MULT",
    "MAX_STOP_PCT",
    "BREAKEVEN_BUFFER_ATR_MULT",
    "TP1_RR_MULT",
    "TP2_RR_MULT",
    "TRAILING_ACTIVATION_RR",
    "TRAILING_CALLBACK_RATE",
    "TRAILING_TIGHT_PCT",
    "TRAILING_LOOSE_PCT",
    # Funding filters
    "MAX_FUNDING_LONG",
    "MIN_FUNDING_SHORT",
    # Kill switch (non-dangerous)
    "KILL_MAX_API_ERROR_BURST",
    # Notifications
    "NOTIFY_ON_ENTRY",
    "NOTIFY_ON_EXIT",
    "NOTIFY_ON_BREAKEVEN",
    "NOTIFY_ON_BLACKLIST",
    "ENTRY_NOTIFY",
    # Cooldowns
    "SYMBOL_COOLDOWN_MINUTES",
    "CONSECUTIVE_LOSS_BLACKLIST_COUNT",
    "SYMBOL_BLACKLIST_DURATION_HOURS",
    # Order retries
    "MAX_ORDER_RETRIES",
    "ORDER_RETRY_SLEEP_SEC",
    # Execution
    "SLIPPAGE_MAX_PCT",
    "MIN_FILL_RATIO",
})


# ── Derived sets for convenience ─────────────────────────────────────
HOT_RELOADABLE: FrozenSet[str] = SAFE | DANGEROUS
ALL_CLASSIFIED: FrozenSet[str] = IMMUTABLE | DANGEROUS | SAFE


def classify(param: str) -> str:
    """Return the tier of a config parameter: 'immutable', 'dangerous', 'safe', or 'unknown'."""
    if param in IMMUTABLE:
        return "immutable"
    if param in DANGEROUS:
        return "dangerous"
    if param in SAFE:
        return "safe"
    return "unknown"
