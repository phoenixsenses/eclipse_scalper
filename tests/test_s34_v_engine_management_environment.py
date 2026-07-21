from tools.s34_v_engine_execution_management_audit import build_report
from tools.s34_v_engine_management_alerts import apply_dedup, build_alerts
from tools.s34_v_engine_v8_risk_refinement import (
    combined_fail_probability,
    notional_for_budget,
    weighted_bps,
)
from tools.s34_v_engine_v9_kill_forward_review import rule_state, summary
from tools.s34_v_engine_v10_operational_risk_suite import monte_carlo_modes, risk_modes
from tools.s34_v_engine_v11_forward_governance import governance_status, row_quality


def test_management_alert_builder_emits_recommendations_only():
    readout = {
        "tail_aware_sizing_monitor": {
            "status": "ALERT_OVERSIZE_OPERATOR_ACTION_REQUIRED",
            "max_tail_budget_margin_usdt": 0.3,
            "oversize_multiple": 107.8,
            "planned_live_size_from_env": {"planned_margin_usdt": 29.8},
        },
        "atomicity_gap_monitor": {
            "status": "ALERT_ADVERSE_IN_GAP",
            "worst_adverse_bps": -18.5,
        },
        "regime_degradation_monitor": {"status": "DATA_INSUFFICIENT"},
        "explicit_kill_criteria": {"triggered": ["OPERATOR_SIZE_REVIEW_REQUIRED"]},
    }
    audit = {
        "gap_through": {
            "gap_plus_fee_bps": 25.7,
            "current_stop_nominal_bps": 150.0,
            "current_stop_research_max_loss_bps": -175.7,
        },
        "tail_frequency": {"large_loss_rate": 18.7},
        "stop_budget_math": {"planned_stop_loss_pct_equity_research": 59.7},
    }

    v11 = {"operator_governance": {"status": "DECISION_REQUIRED", "oversize_vs_balanced": 73.0}}
    payload = build_alerts(readout, audit, v11=v11)

    assert payload["mode"] == "NOTIFY_ONLY_NO_ACTION"
    assert payload["severity"] == "critical"
    assert {a["code"] for a in payload["alerts"]} >= {
        "S34_OVERSIZE",
        "S34_STOP_GAP_THROUGH",
        "S34_ATOMICITY_GAP",
        "S34_KILL_CRITERIA",
        "S34_OPERATOR_DECISION_REQUIRED",
    }


def test_management_alert_dedup_suppresses_unchanged(tmp_path):
    payload = {
        "generated_at_utc": "2026-06-29T00:00:00+00:00",
        "mode": "NOTIFY_ONLY_NO_ACTION",
        "severity": "critical",
        "alerts": [{"severity": "critical", "code": "S34_OVERSIZE", "message": "same"}],
    }
    state = tmp_path / "alert_state.json"

    first = apply_dedup(dict(payload), state, emit_unchanged=False)
    second = apply_dedup(dict(payload), state, emit_unchanged=False)

    assert first["delivery_status"] == "EMIT_STATE_CHANGED"
    assert second["delivery_status"] == "DEDUP_SUPPRESSED_UNCHANGED"
    assert second["state_changed"] is False


def test_v8_weighted_sizing_between_stop_and_tail():
    p_fail = combined_fail_probability(0.2, 0.1)
    loss_bps = weighted_bps(p_fail, stop_bps=175.7, tail_bps=634.0)
    stop_only = notional_for_budget(35.0, 2.0, 175.7)
    weighted = notional_for_budget(35.0, 2.0, loss_bps)
    tail_only = notional_for_budget(35.0, 2.0, 634.0)

    assert 175.7 < loss_bps < 634.0
    assert tail_only < weighted < stop_only


def test_v9_first_tail_rule_trips_only_on_tail():
    assert rule_state("FIRST_TAIL_PAUSE", [-10.0], 0.0, -10.0) is False
    assert rule_state("FIRST_TAIL_PAUSE", [-101.0], 0.0, -101.0) is True


def test_v9_summary_reports_tail_loss():
    payload = summary([10.0, -120.0, 30.0])

    assert payload["n"] == 3
    assert payload["max_loss_bps"] == -120.0
    assert payload["sum_bps"] == -80.0


def test_v10_risk_modes_include_current_env():
    v8 = {
        "stop_reliability_weighted_sizing": {
            "sizing_rows": [
                {"basis": "tail_only_hard_floor", "max_notional_usdt": 11.0, "max_margin_usdt_at_40x": 0.3, "loss_bps": 634.0},
                {"basis": "conservative_weighted", "max_notional_usdt": 16.3, "max_margin_usdt_at_40x": 0.4, "loss_bps": 428.6},
                {"basis": "stop_only_unreliable_floor", "max_notional_usdt": 39.8, "max_margin_usdt_at_40x": 1.0, "loss_bps": 175.7},
            ]
        }
    }
    mgmt = {"tail_aware_sizing_monitor": {"planned_live_size_from_env": {"planned_notional_usdt": 1190.0}}}

    modes = risk_modes(v8, mgmt)

    assert modes["SURVIVAL"]["notional"] == 11.0
    assert modes["BALANCED"]["notional"] == 16.3
    assert modes["CURRENT_ENV"]["notional"] == 1190.0
    assert modes["BALANCED"]["oversize_vs_env"] > 70


def test_v10_monte_carlo_current_env_riskier_than_balanced():
    stop_rows = [{"baseline_net_bps": 50.0}, {"baseline_net_bps": 100.0}, {"baseline_net_bps": -20.0}]
    modes = {
        "BALANCED": {"notional": 16.3, "loss_bps": 428.6},
        "CURRENT_ENV": {"notional": 1190.0, "loss_bps": 428.6},
    }

    result = monte_carlo_modes(stop_rows, modes, trials=500, seed=1, equity=35.0)

    assert result["horizons"]["30"]["CURRENT_ENV"]["ruin_pct"] > result["horizons"]["30"]["BALANCED"]["ruin_pct"]


def test_v11_row_quality_validates_complete_closed_fill():
    row = {
        "observation_status": "CLOSED",
        "sim_status": "FILLED",
        "net_bps": 10.0,
        "dissipation_observer": [{"tau_sec": 120}],
        "atomicity_gap_observer": {"status": "OBSERVED"},
    }

    quality, reasons = row_quality(row)

    assert quality == "VALID"
    assert reasons == []


def test_v11_governance_requires_decision_when_oversized_without_real_journal():
    v10 = {
        "risk_budget_modes": {
            "CURRENT_ENV": {"notional": 1190.0},
            "BALANCED": {"notional": 16.3},
        }
    }
    journal = [{"event": "decision_template", "decision": "template"}]

    payload = governance_status(v10, journal)

    assert payload["status"] == "DECISION_REQUIRED"
    assert payload["real_decision_rows"] == 0


def test_execution_management_audit_is_read_only_contract():
    report = build_report()

    assert report["mode"] == "READ_ONLY_RESEARCH_NO_LIVE_CHANGE"
    assert report["atomicity_audit"]["exchange_native_stop_market"] is True
    assert report["atomicity_audit"]["finding"] == "NOT_ATOMIC_ENTRY_THEN_STOP_AFTER_FILL_DETECTION"
    assert "Do not change live logic automatically." in report["recommendations"]
