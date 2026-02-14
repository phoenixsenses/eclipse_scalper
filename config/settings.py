# config/settings.py — SCALPER ETERNAL — MICRO CAPITAL ASCENDANT MODE — 2026 v3.3
# Patch vs v3.2:
# - ✅ Adds ENTRY LOOP compatibility keys (ENTRY_* + ACTIVE_SYMBOLS + FIXED_NOTIONAL sizing)
# - ✅ Adds kill-switch / telemetry defaults (safe)
# - ✅ Keeps your production + micro risk logic intact
#
# Why this matters:
# - execution/entry_loop.py reads ENTRY_MIN_CONFIDENCE / ENTRY_POLL_SEC / ACTIVE_SYMBOLS / FIXED_NOTIONAL_USDT
# - Without these, you’ll default to BTCUSDT only + sizing None + silent starvation

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """
    OMNIPOTENT PRODUCTION MODE — 2026 v2 (Baseline)
    Foundation for mid-to-large accounts (equity ≥ $100).
    """

    # === TRANSCENDENT TIMEFRAMES ===
    TIMEFRAME: str = "1m"
    TIMEFRAME_5M: str = "5m"
    TIMEFRAME_15M: str = "15m"

    # === DIVINE RISK & ETERNAL PRESERVATION (PRODUCTION BALANCE) ===
    MAX_RISK_PER_TRADE: float = 0.10
    MAX_PORTFOLIO_HEAT: float = 0.45
    MAX_CONCURRENT_POSITIONS: int = 6
    MIN_RISK_DOLLARS: float = 5.0
    MAX_DAILY_LOSS_PCT: float = 0.15
    SYMBOL_COOLDOWN_MINUTES: int = 20
    CONSECUTIVE_LOSS_BLACKLIST_COUNT: int = 3
    SYMBOL_BLACKLIST_DURATION_HOURS: int = 4
    BLACKLIST_AUTO_RESET_ON_PROFIT: bool = True

    # === OMNIPOTENT MODES ===
    OMNIPOTENT_MODE: bool = True
    ADAPTIVE_RISK_SCALING: bool = True

    # === CORRELATION & VELOCITY GUARDS ===
    CORRELATION_HEAT_CAP: float = 0.30
    CORR_RETURNS_KEY: str = "corr_returns"
    CORR_RETURNS_MAX_LEN: int = 240
    CORR_TAIL_THRESHOLD: float = 0.015
    CORR_WEIGHT_ROLL: float = 0.35
    CORR_WEIGHT_DOWNSIDE: float = 0.35
    CORR_WEIGHT_TAIL: float = 0.30
    CORR_STRESS_ENTER: float = 0.72
    CORR_STRESS_EXIT: float = 0.58
    CORR_TIGHTEN_ENTER: float = 0.48
    CORR_TIGHTEN_EXIT: float = 0.38
    CORR_HYST_UP_SEC: float = 10.0
    CORR_HYST_DOWN_SEC: float = 30.0
    CORR_DEBT_REF_SEC: float = 300.0
    CORR_UNCERTAINTY_UPLIFT_MAX: float = 0.25
    CORR_DEBT_UPLIFT_WEIGHT: float = 0.15
    CORR_HIDDEN_EXPOSURE_UPLIFT_WEIGHT: float = 0.15
    CORR_CONTRADICTION_UPLIFT_WEIGHT: float = 0.10
    CORR_HIDDEN_REF: float = 3.0
    CORR_GROUP_DRIFT_REF: float = 3.0
    CORR_DRIFT_WEIGHT: float = 0.15
    CORR_SAMPLE_REF_PAIRS: float = 6.0
    SESSION_EQUITY_PEAK_PROTECTION_PCT: float = 0.10
    VELOCITY_DRAWDOWN_PCT: float = 0.06
    VELOCITY_MINUTES: int = 6
    MIN_ATR_PCT_FOR_ENTRY: float = 0.007

    # === LEVERAGE & EXECUTION ===
    LEVERAGE: int = 20
    MIN_FILL_RATIO: float = 0.85
    SLIPPAGE_MAX_PCT: float = 0.006

    # === STOP, TAKE PROFIT & TRAILING ===
    STOP_ATR_MULT: float = 1.10
    MAX_STOP_PCT: float = 0.03
    BREAKEVEN_BUFFER_ATR_MULT: float = 0.30
    TP1_RR_MULT: float = 1.00
    TP2_RR_MULT: float = 2.00
    TRAILING_ACTIVATION_RR: float = 1.30
    TRAILING_CALLBACK_RATE: int = 45
    TRAILING_TIGHT_PCT: int = 35
    TRAILING_LOOSE_PCT: int = 75

    # === FUNDING RATE FILTERS ===
    MAX_FUNDING_LONG: float = 0.0006
    MIN_FUNDING_SHORT: float = -0.0004

    # === SIGNAL THRESHOLD ===
    MIN_CONFIDENCE: float = 0.72
    MIN_CONFIDENCE_HIGH_VOL: float = 0.65

    # === ENTRY LOOP COMPATIBILITY (NEW) ===
    # entry_loop.py uses these keys (kept separate from strategy MIN_CONFIDENCE)
    ENTRY_MIN_CONFIDENCE: float = 0.72
    ENTRY_POLL_SEC: float = 1.0
    ENTRY_PER_SYMBOL_GAP_SEC: float = 2.5
    ENTRY_LOCAL_COOLDOWN_SEC: float = 8.0
    ENTRY_RESPECT_KILL_SWITCH: bool = True
    ENTRY_ROUTER_RETRIES: int = 6
    ENTRY_NOTIFY: bool = False

    # Simple sizing fallback used by entry_loop when signal doesn't provide amount:
    # qty = FIXED_NOTIONAL_USDT / price
    FIXED_QTY: float = 0.0
    FIXED_NOTIONAL_USDT: float = 25.0
    MIN_ENTRY_QTY: float = 0.0

    # Symbol universe
    ACTIVE_SYMBOLS: List[str] = field(default_factory=lambda: ["BTCUSDT"])

    # === TRADING HOURS ===
    TRADING_HOURS_UTC: List[int] = field(default_factory=lambda: list(range(24)))

    # === LOGGING & NOTIFICATION ===
    LOGGING_LEVEL: str = "INFO"
    NOTIFY_ON_ENTRY: bool = True
    NOTIFY_ON_EXIT: bool = True
    NOTIFY_ON_BREAKEVEN: bool = True
    NOTIFY_ON_BLACKLIST: bool = True

    # === FLAGS ===
    CONFIDENCE_SCALING: bool = True
    DYNAMIC_TRAILING_FULL: bool = True
    DUAL_TRAILING: bool = True
    MAX_HEAT_POST_ENTRY_ENFORCE: bool = True

    # === MICRO-RELATED EXECUTION MINIMUMS (SAFE DEFAULTS) ===
    MIN_NOTIONAL_USDT: float = 5.0
    MIN_MARGIN_USDT: float = 2.0
    MAX_ORDER_RETRIES: int = 2
    ORDER_RETRY_SLEEP_SEC: float = 0.25

    # === KILL SWITCH DEFAULTS (SAFE) ===
    KILL_SWITCH_ENABLED: bool = True
    KILL_SWITCH_COOLDOWN_SEC: float = 300.0
    KILL_MAX_DATA_STALENESS_SEC: float = 150.0
    KILL_DATA_BOOT_GRACE_SEC: float = 120.0
    KILL_MIN_DATA_SAMPLES_BEFORE_ENFORCE: int = 1
    KILL_MAX_API_ERROR_RATE: float = 0.35
    KILL_MAX_API_ERROR_BURST: int = 12
    KILL_MIN_REQ_WINDOW: int = 10
    KILL_SWITCH_EMERGENCY_FLAT: bool = False  # keep OFF unless you're confident
    KILL_ESCALATE_FLAT_AFTER_TRIPS: int = 0
    KILL_ESCALATE_SHUTDOWN_AFTER_TRIPS: int = 0
    KILL_ESCALATE_WINDOW_SEC: float = 900.0
    KILL_SWITCH_TRIP_HISTORY_MAX: int = 12

    # === TELEMETRY DEFAULTS (SAFE) ===
    TELEMETRY_WRITE_FILE: bool = True
    TELEMETRY_MIRROR_STDOUT: bool = False
    TELEMETRY_RING_MAX: int = 250
    TELEMETRY_PATH: str = ""  # empty => logs/telemetry.jsonl

    # === BELIEF CORRELATION PRESSURE (ENTRY-ONLY CLAMPING) ===
    BELIEF_CORR_WEIGHT: float = 0.30
    BELIEF_CORR_REGIME_STRESS_WEIGHT: float = 0.40
    BELIEF_CORR_REGIME_TIGHTEN_WEIGHT: float = 0.20
    BELIEF_CORR_TAIL_WEIGHT: float = 0.20
    BELIEF_CORR_DOWNSIDE_WEIGHT: float = 0.15
    BELIEF_CORR_UPLIFT_WEIGHT: float = 0.20
    BELIEF_CORR_DRIFT_WEIGHT: float = 0.15
    BELIEF_CORR_HIDDEN_WEIGHT: float = 0.15
    BELIEF_CORR_LOW_CONF_WEIGHT: float = 0.20
    BELIEF_CORR_STRESS_ENTER: float = 0.72
    BELIEF_CORR_TIGHTEN_ENTER: float = 0.48
    BELIEF_CORR_NOTIONAL_SCALE_MIN: float = 0.25
    BELIEF_CORR_LEVERAGE_SCALE_MIN: float = 0.35
    BELIEF_CORR_MIN_CONF_EXTRA_MAX: float = 0.10
    BELIEF_CORR_COOLDOWN_EXTRA_SEC_MAX: float = 15.0
    BELIEF_CORR_HARD_FREEZE_STRESS: bool = False
    BELIEF_CORR_HARD_FREEZE_PRESSURE: float = 0.95
    BELIEF_CORR_ORANGE_PRESSURE: float = 0.88

    # === METADATA ===
    CONFIG_VERSION: str = "omnipotent-production-2026-v2"
    CONFIG_FORGED_DATE: str = "2026-01-07"

    def __post_init__(self):
        # Keep your sanity checks
        if not (0.0 < self.MAX_RISK_PER_TRADE <= 0.50):
            raise ValueError("MAX_RISK_PER_TRADE must be in (0, 0.50].")
        if not (0.0 < self.MAX_PORTFOLIO_HEAT <= 1.00):
            raise ValueError("MAX_PORTFOLIO_HEAT must be in (0, 1.00].")
        if self.MAX_CONCURRENT_POSITIONS < 1:
            raise ValueError("MAX_CONCURRENT_POSITIONS must be >= 1.")
        if self.LEVERAGE < 1:
            raise ValueError("LEVERAGE must be >= 1.")
        if not (0.0 < self.MIN_FILL_RATIO <= 1.0):
            raise ValueError("MIN_FILL_RATIO must be in (0, 1].")
        if not (0.0 < self.MIN_CONFIDENCE <= 1.0):
            raise ValueError("MIN_CONFIDENCE must be in (0, 1].")
        if not (0.0 <= self.MAX_DAILY_LOSS_PCT <= 1.0):
            raise ValueError("MAX_DAILY_LOSS_PCT must be in [0, 1].")
        if not (0.0 < self.CORRELATION_HEAT_CAP <= 1.0):
            raise ValueError("CORRELATION_HEAT_CAP must be in (0, 1].")

        # Keep ENTRY_MIN_CONFIDENCE aligned by default if user didn't override
        try:
            if float(self.ENTRY_MIN_CONFIDENCE or 0.0) <= 0:
                self.ENTRY_MIN_CONFIDENCE = float(self.MIN_CONFIDENCE)
        except Exception:
            pass


@dataclass
class MicroConfig(Config):
    """
    MICRO CAPITAL ASCENDANT MODE — 2026 v3.2
    Recommended when equity < $100.

    v3.2 core fix:
    - Confidence floor lowered to match real signal output distribution
    - Risk reduced to survive exchange minimums + slippage reality
    """

    # === RISK & PRESERVATION OVERRIDES (MICRO-SCALE) ===
    MAX_RISK_PER_TRADE: float = 0.06
    MAX_PORTFOLIO_HEAT: float = 0.15
    MAX_CONCURRENT_POSITIONS: int = 1
    MIN_RISK_DOLLARS: float = 0.0
    MAX_DAILY_LOSS_PCT: float = 0.20
    SYMBOL_COOLDOWN_MINUTES: int = 12

    # === CORRELATION & VELOCITY OVERRIDES (MICRO-SCALE) ===
    CORRELATION_HEAT_CAP: float = 0.12
    SESSION_EQUITY_PEAK_PROTECTION_PCT: float = 0.12
    VELOCITY_DRAWDOWN_PCT: float = 0.07
    VELOCITY_MINUTES: int = 5
    MIN_ATR_PCT_FOR_ENTRY: float = 0.006

    # === EXECUTION OVERRIDES ===
    LEVERAGE: int = 35
    MIN_FILL_RATIO: float = 0.80
    SLIPPAGE_MAX_PCT: float = 0.010

    # === REWARD STRUCTURE OVERRIDES (MICRO) ===
    STOP_ATR_MULT: float = 1.00
    MAX_STOP_PCT: float = 0.035
    BREAKEVEN_BUFFER_ATR_MULT: float = 0.20
    TP1_RR_MULT: float = 1.00
    TP2_RR_MULT: float = 2.20
    TRAILING_ACTIVATION_RR: float = 1.20
    TRAILING_CALLBACK_RATE: int = 40
    TRAILING_TIGHT_PCT: int = 30
    TRAILING_LOOSE_PCT: int = 70

    # === SIGNAL THRESHOLD OVERRIDE (MICRO) ===
    MIN_CONFIDENCE: float = 0.35
    MIN_CONFIDENCE_HIGH_VOL: float = 0.30

    # Mirror into entry-loop gate
    ENTRY_MIN_CONFIDENCE: float = 0.35

    # === MICRO MINIMUMS (MORE REALISTIC FOR $25–$99) ===
    MIN_NOTIONAL_USDT: float = 5.0
    MIN_MARGIN_USDT: float = 0.75
    MAX_ORDER_RETRIES: int = 3
    ORDER_RETRY_SLEEP_SEC: float = 0.35

    # Micro sizing fallback (if signal omits amount)
    FIXED_NOTIONAL_USDT: float = 8.0

    # === METADATA OVERRIDE ===
    CONFIG_VERSION: str = "micro-capital-ascendant-2026-v3"
    CONFIG_FORGED_DATE: str = "2026-01-07"
