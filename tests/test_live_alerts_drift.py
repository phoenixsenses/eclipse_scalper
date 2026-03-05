from __future__ import annotations

from src.microphys.live.alerts import evaluate_alerts
from src.microphys.live.config import LiveSettings


def test_drift_alerts_trigger() -> None:
    cfg = LiveSettings(
        replay_fill_rate_delta_warn=0.1,
        replay_adverse_bps_delta_warn=0.5,
        replay_match_rate_low_warn=0.5,
        diagnostics_toxicity_warn=1.0,
        diagnostics_latency_p95_warn_sec=1.0,
    )
    status = {
        "replay_fill_rate_delta": 0.2,
        "replay_adverse_bps_delta": 0.8,
        "replay_match_rate_vs_sim": 0.2,
        "diag_toxicity_score": 1.5,
        "diag_latency_fill_delay_sec_p95": 2.0,
        "data_freshness_sec": 0.0,
        "missing_bars_pct_1h": 0.0,
        "spread_jump_frac": 0.0,
        "ofi_shift": 0.0,
        "regime_shift": 0.0,
        "signal_rate_per_hour": 1.0,
    }
    alerts = evaluate_alerts(status, cfg)
    codes = {str(a.get("code")) for a in alerts}
    assert "replay_fill_rate_drift" in codes
    assert "replay_adverse_drift" in codes
    assert "replay_match_rate_low" in codes
    assert "diagnostics_toxicity_high" in codes
    assert "diagnostics_latency_high" in codes

