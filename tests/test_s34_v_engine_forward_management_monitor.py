from tools.s34_v_engine_forward_management_monitor import (
    regime_monitor,
    sizing_monitor,
)


def test_sizing_monitor_flags_env_oversize():
    env = {
        "S34_LIVE_MARGIN_PCT_ETH": "85",
        "S34_LIVE_MAX_LEVERAGE": "40",
        "S34_LIVE_MARGIN_USDT": "30",
    }
    report = sizing_monitor(env, [], [], equity_usdt=35.0, risk_pct=2.0)

    assert report["status"] == "ALERT_OVERSIZE_OPERATOR_ACTION_REQUIRED"
    assert report["action"] == "RECOMMENDATION_ONLY_NO_AUTO_SIZE_CHANGE"
    assert report["max_tail_budget_notional_usdt"] == 11.0
    assert report["planned_live_size_from_env"]["planned_notional_usdt"] == 1190.0


def test_regime_monitor_trips_on_synthetic_forward_degradation():
    rows = [
        {
            "sample_type": "FORWARD_OOS",
            "observation_status": "CLOSED",
            "sim_status": "FILLED",
            "net_bps": -10.0,
        }
        for _ in range(5)
    ]

    report = regime_monitor(rows)

    assert report["status"] == "DATA_INSUFFICIENT"
    assert "PAUSE_ROLLING_5_NEGATIVE" in report["triggers"]
    assert "KILL_FORWARD_SUM_NEGATIVE" in report["triggers"]
