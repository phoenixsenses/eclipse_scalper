from __future__ import annotations

from pydantic import BaseModel, Field


class LiveSettings(BaseModel):
    db_path: str = "data/microstructure.db"
    symbol: str = "ETHUSDT"
    interval_ms: int = 100
    lookback_hours: float = 24.0
    refresh_sec: float = 30.0
    mode: str = "taker"
    fee_bps: float = 0.5
    latency_bars: int = 2
    execution_model: str = "simple"  # simple | maker_queue | maker_hazard
    execution_params_path: str = ""
    use_active_artifacts: bool = True
    disable_online_reload: bool = False
    auto_rollback_on_bad_calibration: bool = False
    maker_ttl_bars: int = 10
    out_root: str = "data/live"
    run_root: str = "data/runs/alpha"
    max_trades_per_day: int = 500
    use_regime_experts: bool = False
    experts_path: str = ""
    aligned_regimes_path: str = ""
    enable_risk_engine: bool = False
    risk_policy_path: str = ""
    starting_equity: float = 10_000.0
    exec_engine_unified: bool = False
    exec_event_bus_enabled: bool = False
    exec_runtime_supervisor_enabled: bool = False
    exec_kill_on_contract_violation: bool = False
    supervisor_max_feed_age_sec: float = 120.0
    supervisor_max_order_age_sec: float = 600.0
    supervisor_max_loop_errors: int = 3

    # Drift / health thresholds
    freshness_warn_sec: float = 90.0
    missing_bars_warn_pct: float = 10.0
    spread_jump_warn_frac: float = 0.5
    ofi_shift_warn: float = 1.0
    regime_shift_warn: float = 0.35
    signal_rate_low_warn: float = 0.01
    signal_rate_high_warn: float = 200.0
    # Execution drift thresholds (Sprint 2)
    replay_fill_rate_delta_warn: float = 0.20
    replay_adverse_bps_delta_warn: float = 1.50
    replay_match_rate_low_warn: float = 0.30
    diagnostics_toxicity_warn: float = 2.50
    diagnostics_latency_p95_warn_sec: float = 12.0
