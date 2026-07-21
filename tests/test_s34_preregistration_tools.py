from tools.s34_calibration_check import calibration_check
from tools.s34_holdout_decision import holdout_decision
from tools.s34_quarantine_monitor import quarantine_monitor


def _closed(trade_id: str, net: float, gross: float = 20.0, entry_adv: float = 2.0, exit_adv: float = 1.0):
    fee = 8.0
    spread = gross - entry_adv - exit_adv - fee - net
    return {
        "trade_id": trade_id,
        "status": "CLOSED",
        "signal_ts_ms": int(trade_id[1:]) if trade_id[1:].isdigit() else 0,
        "exit_reason": "TP" if net > 0 else "SL",
        "gross_bps": gross,
        "entry_adverse_bps": entry_adv,
        "exit_adverse_bps": exit_adv,
        "spread_cost_bps": spread,
        "fee_cost_bps": fee,
        "net_bps": net,
        "entry_fill": {"source": "BOOK_TICKER"},
        "exit_fill": {"source": "BOOK_TICKER"},
        "rule": {"name": "ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30"},
        "signal": {"liq_total_notional": 100_000.0},
    }


def test_calibration_k1_and_k2_are_pre_registered():
    positive = [_closed(f"P{i:03d}", net=5.0, gross=20.0, entry_adv=3.0) for i in range(1, 41)]
    result = calibration_check(positive, exclude=set(), n=40)
    assert result["ready"] is True
    assert result["kills"]["K1_mean_net_le_zero"] is False
    assert result["kills"]["K2_median_entry_adverse_ge_mean_abs_gross"] is False

    structurally_bad = [_closed(f"P{i:03d}", net=1.0, gross=5.0, entry_adv=5.0) for i in range(1, 41)]
    result = calibration_check(structurally_bad, exclude=set(), n=40)
    assert result["kills"]["K2_median_entry_adverse_ge_mean_abs_gross"] is True

    negative = [_closed(f"P{i:03d}", net=-1.0, gross=20.0, entry_adv=3.0) for i in range(1, 41)]
    result = calibration_check(negative, exclude=set(), n=40)
    assert result["kills"]["K1_mean_net_le_zero"] is True


def test_quarantine_monitor_trips_only_when_rate_and_correlation_hold():
    trades = []
    for i in range(1, 101):
        q = i > 70
        trades.append(
            {
                "trade_id": f"P{i:03d}",
                "status": "SKIPPED" if q else "CLOSED",
                "risk_gate_reason": "NO_FILL_DATA" if q else "",
                "signal": {"liq_total_notional": 1_000_000.0 if q else 10_000.0 + i},
            }
        )
    result = quarantine_monitor(trades, exclude=set())
    assert result["no_fill_data_rate"] == 0.30
    assert result["K3_triggered"] is True

    calm = []
    for i in range(1, 101):
        calm.append(
            {
                "trade_id": f"P{i:03d}",
                "status": "SKIPPED" if i % 10 == 0 else "CLOSED",
                "risk_gate_reason": "NO_FILL_DATA" if i % 10 == 0 else "",
                "signal": {"liq_total_notional": float(i)},
            }
        )
    result = quarantine_monitor(calm, exclude=set())
    assert result["no_fill_data_rate"] == 0.10
    assert result["K3_triggered"] is False


def test_holdout_refuses_before_n100_and_passes_known_positive_series():
    short = [_closed(f"P{i:03d}", net=10.0) for i in range(1, 100)]
    result = holdout_decision(short, exclude=set(), bootstrap_resamples=200, n_trials=50)
    assert result["decision"] == "INSUFFICIENT_SAMPLE_DO_NOT_RUN_HOLDOUT"

    positive = [_closed(f"P{i:03d}", net=5.0) for i in range(1, 101)]
    result = holdout_decision(positive, exclude=set(), bootstrap_resamples=200, n_trials=50)
    assert result["decision"] == "PASS"
    assert result["economic_significance_pass"] is True
    assert result["statistical_significance_pass"] is True

    negative = [_closed(f"P{i:03d}", net=5.0) for i in range(1, 41)] + [
        _closed(f"P{i:03d}", net=-5.0) for i in range(41, 101)
    ]
    result = holdout_decision(negative, exclude=set(), bootstrap_resamples=200, n_trials=50)
    assert result["decision"] == "FAIL"
    assert result["economic_significance_pass"] is False


def test_validation_scripts_exclude_non_s34_variant():
    trades = [_closed("P001", net=10.0), _closed("P002", net=-10.0)]
    trades[1]["rule"]["name"] = "ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30"

    calibration = calibration_check(trades, exclude=set(), n=1)
    assert calibration["valid_closed_count"] == 1
    assert calibration["net_bps"]["mean"] == 10.0

    decision = holdout_decision(trades * 100, exclude=set(), bootstrap_resamples=100, n_trials=50)
    assert decision["valid_closed_count"] == 100
