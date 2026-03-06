from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import report_schema_validator as rsv
from tools.micro_edge_smoke import build_json_record


def test_validate_micro_edge_smoke_payload() -> None:
    payload = build_json_record(
        rep={
            "symbol": "ETHUSDT",
            "raw_count": 100,
            "bucket_count": 20,
            "up": 5,
            "down": 3,
            "flat": 2,
            "baseline_hit_rate": 0.6,
            "feature_corr": {"imbalance": 0.1},
            "rules": {"micro_edge_v3_passive_alpha": {"hit_rate": 0.7, "n": 15, "delta_vs_baseline": 0.1}},
            "label_definition": {},
        },
        lookback_min=120,
        bucket_sec=5,
        horizon_sec=30,
        min_rule_n=10,
    )
    assert rsv.infer_schema_name(payload) == "micro_edge_smoke"
    assert rsv.validate_payload(payload, "micro_edge_smoke") == []


def test_validate_micro_edge_forward_payload() -> None:
    payload = {
        "debug": "localtests/forward.jsonl",
        "group_by": ["regime_spread_bin"],
        "discover_frac": 0.6,
        "counts": {
            "total": 10,
            "discovery": 6,
            "validation": 4,
            "selected_discovery": 3,
            "selected_validation": 2,
            "top_groups": 1,
        },
        "thresholds": {"min_n_discovery": 2, "min_n_validation": 2, "min_select_frac": 0.01},
        "discovery": {"n": 3, "avg_net": 0.001, "p90_net": 0.002, "p90_net_negative": False},
        "validation": {"n": 2, "avg_net": 0.0005, "p90_net": 0.001, "p90_net_negative": False},
        "collapse": {"detected": False, "flags": {"p90_sign_flip": False}, "values": {}},
        "liquidation_impact": {
            "discovery": {"available": True, "count": 3, "threshold_q75": 0.4, "active": {"n": 1, "avg_net": 0.0012, "p90_net": 0.0012}, "inactive": {"n": 2, "avg_net": 0.0008, "p90_net": 0.0010}},
            "validation": {"available": False, "count": 0},
        },
        "liquidation_regime_tag_impact": {
            "discovery": {"available": True, "tagged": {"n": 1, "avg_net": 0.0012, "p90_net": 0.0012}, "normal": {"n": 2, "avg_net": 0.0008, "p90_net": 0.0010}},
            "validation": {"available": True, "tagged": {"n": 0, "avg_net": 0.0, "p90_net": 0.0}, "normal": {"n": 2, "avg_net": 0.0005, "p90_net": 0.0010}},
        },
        "run_summary": {
            "version": "1",
            "run_type": "validate_micro_edge_forward",
            "inputs": {"debug": "localtests/forward.jsonl"},
            "metrics": {"total": 10, "collapse_detected": 0},
            "artifacts": {"json": "localtests/forward.json"},
        },
    }
    assert rsv.infer_schema_name(payload) == "validate_micro_edge_forward"
    assert rsv.validate_payload(payload, "validate_micro_edge_forward") == []


def test_validate_liquidation_rule_coverage_payload() -> None:
    payload = {
        "symbol": "ETHUSDT",
        "rule": "high_liq_reversal_regime",
        "bucket_sec": 5,
        "results": [
            {
                "lookback_min": 60,
                "bucket_rows": 100,
                "liq_rows": 10,
                "rule_fire_count": 2,
                "rule_fire_rate": 0.02,
                "rule_given_liq_rate": 0.2,
            }
        ],
        "run_summary": {
            "version": "v1",
            "run_type": "liquidation_rule_coverage",
            "inputs": {"symbol": "ETHUSDT"},
            "metrics": {"windows": 1},
            "artifacts": {"json": "reports/out.json", "md": "reports/out.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "liquidation_rule_coverage"
    assert rsv.validate_payload(payload, "liquidation_rule_coverage") == []


def test_validate_canonical_payload() -> None:
    payload = {
        "status": "pass",
        "run_id": "abc123",
        "source": "data/canonical/canonical_merged.parquet",
        "violations": [],
        "column_stats": {"timestamp": {"timestamp_convertible_nan_ratio": 0.0}},
        "invariant_summary": {"rows": 100, "violations": 0},
        "notes": [],
        "run_summary": {
            "version": "v1",
            "run_type": "validate_canonical",
            "inputs": {"source": "data/canonical/canonical_merged.parquet"},
            "metrics": {"status": "pass", "violation_count": 0, "row_count": 100},
            "artifacts": {"json": "reports/validate_canonical_abc123.json", "md": "reports/validate_canonical_abc123.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "validate_canonical"
    assert rsv.validate_payload(payload, "validate_canonical") == []


def test_validate_passive_pocket_forward_payload() -> None:
    payload = {
        "symbol": "ETHUSDT",
        "horizon_sec": 60,
        "rows_total": 4,
        "pass_count": 2,
        "pass_rate": 0.5,
        "per_combo": [
            {
                "seed": 11,
                "split": 1,
                "filled_n": 40,
                "attempt_fill_rate": 0.5,
                "fail_reason": "ok",
                "pass": True,
            }
        ],
        "per_split": [],
        "failure_attribution_median": {},
        "run_summary": {
            "version": "v1",
            "run_type": "validate_passive_pocket_forward",
            "inputs": {"symbol": "ETHUSDT"},
            "metrics": {"rows_total": 4, "pass_count": 2, "pass_rate": 0.5, "insufficient_fill_rate": 0.25},
            "artifacts": {"json": "reports/PASSIVE_POCKET_FORWARD_VALIDATION.json", "md": "reports/PASSIVE_POCKET_FORWARD_VALIDATION.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "validate_passive_pocket_forward"
    assert rsv.validate_payload(payload, "validate_passive_pocket_forward") == []


def test_validate_summarize_rank_attribution_payload() -> None:
    payload = {
        "source": "reports/PASSIVE_POCKET_RANKING.json",
        "rows_total": 3,
        "top_n": 2,
        "reason_share": {"fees_dominate": 0.66, "mixed": 0.34},
        "gate_high_share": 0.33,
        "next_action": "Next action: fees dominate. Improve fee tier/rebate or increase raw edge.",
        "run_summary": {
            "version": "v1",
            "run_type": "summarize_rank_attribution",
            "inputs": {"source": "reports/PASSIVE_POCKET_RANKING.json", "top_n": 2},
            "metrics": {"rows_total": 3, "top_n": 2, "gate_high_share": 0.33},
            "artifacts": {"json": "reports/rank_attr_summary.json"},
        },
    }
    assert rsv.infer_schema_name(payload) == "summarize_rank_attribution"
    assert rsv.validate_payload(payload, "summarize_rank_attribution") == []


def test_validate_summarize_liq_regime_tag_impact_payload() -> None:
    payload = {
        "source": "reports/forward.json",
        "discovery": {
            "available": True,
            "tagged": {"n": 3, "avg_net": 0.0010, "p90_net": 0.0015},
            "normal": {"n": 7, "avg_net": 0.0004, "p90_net": 0.0008},
            "delta_avg_net": 0.0006,
            "delta_p90_net": 0.0007,
            "sample_warning": False,
        },
        "validation": {
            "available": True,
            "tagged": {"n": 2, "avg_net": -0.0001, "p90_net": 0.0002},
            "normal": {"n": 8, "avg_net": 0.0002, "p90_net": 0.0006},
            "delta_avg_net": -0.0003,
            "delta_p90_net": -0.0004,
            "sample_warning": False,
        },
        "recommendation": "Next action: discovery edge does not survive validation. Keep as annotation, not as a trading gate.",
        "run_summary": {
            "version": "v1",
            "run_type": "summarize_liq_regime_tag_impact",
            "inputs": {"source": "reports/forward.json"},
            "metrics": {"discovery_delta_avg_net": 0.0006, "validation_delta_avg_net": -0.0003},
            "artifacts": {"json": "reports/liq_regime_tag_summary.json"},
        },
    }
    assert rsv.infer_schema_name(payload) == "summarize_liq_regime_tag_impact"
    assert rsv.validate_payload(payload, "summarize_liq_regime_tag_impact") == []


def test_validate_summarize_liq_tag_signal_behavior_payload() -> None:
    payload = {
        "debug": "localtests/debug.jsonl",
        "rule": "high_liq_reversal_regime",
        "overall": {
            "rows_total": 10,
            "tagged": {"n": 2, "avg_net": 0.0009, "p90_net": 0.0012, "break_even_bps_total": 9.0},
            "normal": {"n": 8, "avg_net": 0.0001, "p90_net": 0.0005, "break_even_bps_total": 1.0},
            "delta_avg_net": 0.0008,
            "delta_p90_net": 0.0007,
        },
        "recommendation": "Next action: tagged signals look stronger. Use liquidation regime as a downstream filter candidate.",
        "run_summary": {
            "version": "v1",
            "run_type": "summarize_liq_tag_signal_behavior",
            "inputs": {"debug": "localtests/debug.jsonl", "rule": "high_liq_reversal_regime"},
            "metrics": {"rows_total": 10, "tagged_n": 2, "normal_n": 8, "delta_avg_net": 0.0008, "delta_p90_net": 0.0007},
            "artifacts": {"json": "reports/liq_tag_signal_behavior.json"},
        },
    }
    assert rsv.infer_schema_name(payload) == "summarize_liq_tag_signal_behavior"
    assert rsv.validate_payload(payload, "summarize_liq_tag_signal_behavior") == []


def test_validate_liquidation_regime_alerts_payload() -> None:
    payload = {
        "symbol": "ETHUSDT",
        "rule": "high_liq_reversal_regime",
        "lookback_min": 240,
        "bucket_sec": 5,
        "recent_limit": 20,
        "min_liq_rate": 2.0,
        "summary": {
            "rows_total": 100,
            "tagged_count": 5,
            "tagged_rate": 0.05,
            "recent_alert_count": 2,
            "max_consecutive_tagged": 2,
            "max_liq_rate_recent": 5.0,
            "side_bias_counts": {"LONG": 1, "SHORT": 1},
            "severity_counts": {"high": 1, "medium": 1},
        },
        "alerts": [
            {
                "ts_ms": 1,
                "side_bias": "LONG",
                "severity": "high",
                "liq_rate_per_sec": 5.0,
                "liq_imbalance": 0.8,
                "spread": 0.01,
                "trade_intensity": 10.0,
                "ret_1": -0.002,
            }
        ],
        "run_summary": {
            "version": "v1",
            "run_type": "liquidation_regime_alerts",
            "inputs": {"symbol": "ETHUSDT"},
            "metrics": {"rows_total": 100},
            "artifacts": {"json": "reports/liq_alerts.json", "md": "reports/liq_alerts.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "liquidation_regime_alerts"
    assert rsv.validate_payload(payload, "liquidation_regime_alerts") == []


def test_validate_liquidation_alert_state_payload() -> None:
    payload = {
        "source_json": "reports/LIQUIDATION_REGIME_ALERTS_REAL.json",
        "symbol": "ETHUSDT",
        "rule": "high_liq_reversal_regime",
        "dashboard_summary": "ETHUSDT elevated liquidation regime, bias LONG, 3 recent alerts, freshness fresh.",
        "notification_text": "[liq-regime] symbol=ETHUSDT level=elevated freshness=fresh bias=LONG recent_alerts=3 max_liq_rate=5.2000 action=show_caution",
        "recommended_action": "show_caution",
        "state": {
            "level": "elevated",
            "reasons": ["recent_alert_cluster"],
            "primary_side_bias": "LONG",
            "dominant_severity": "medium",
            "freshness": {"status": "fresh", "age_sec": 4.0, "stale_after_sec": 60},
        },
        "card": {
            "headline": "ETHUSDT liquidation regime elevated",
            "operator_note": "Show on dashboard.",
            "recent_alert_count": 3,
            "tagged_rate": 0.04,
            "max_consecutive_tagged": 2,
            "max_liq_rate_recent": 5.2,
            "primary_side_bias": "LONG",
            "dominant_severity": "medium",
            "latest_alert_ts_ms": 456,
            "freshness_status": "fresh",
            "age_sec": 4.0,
        },
        "summary_snapshot": {
            "rows_total": 50,
            "tagged_count": 2,
            "tagged_rate": 0.04,
            "recent_alert_count": 3,
            "max_consecutive_tagged": 2,
            "max_liq_rate_recent": 5.2,
            "side_bias_counts": {"LONG": 2},
            "severity_counts": {"medium": 2},
        },
        "run_summary": {
            "version": "v1",
            "run_type": "liquidation_alert_state",
            "inputs": {"source_json": "reports/LIQUIDATION_REGIME_ALERTS_REAL.json"},
            "metrics": {"state_level": "elevated", "recommended_action": "show_caution"},
            "artifacts": {"json": "reports/LIQUIDATION_ALERT_STATE.json", "md": "reports/LIQUIDATION_ALERT_STATE.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "liquidation_alert_state"
    assert rsv.validate_payload(payload, "liquidation_alert_state") == []


def test_validate_liquidation_watchlist_payload() -> None:
    payload = {
        "rule": "high_liq_reversal_regime",
        "lookback_min": 240,
        "bucket_sec": 5,
        "recent_limit": 20,
        "min_liq_rate": 0.0,
        "summary": {"symbol_count": 2, "top_n": 2, "state_counts": {"elevated": 1, "quiet": 1}, "top_symbol": "ETHUSDT"},
        "top_summary": {
            "symbol": "ETHUSDT",
            "state_level": "elevated",
            "freshness_status": "fresh",
            "recommended_action": "show_caution",
            "dashboard_summary": "ETH summary",
        },
        "banner": {
            "headline": "Liquidation watchlist top=ETHUSDT level=elevated freshness=fresh action=show_caution",
            "recommended_action": "show_caution",
            "top_symbol": "ETHUSDT",
            "top_state_level": "elevated",
            "top_freshness_status": "fresh",
            "severe_count": 0,
            "elevated_count": 1,
            "quiet_count": 1,
        },
        "rows": [
            {
                "symbol": "ETHUSDT",
                "state_level": "elevated",
                "freshness_status": "fresh",
                "recommended_action": "show_caution",
                "primary_side_bias": "LONG",
                "dominant_severity": "medium",
                "recent_alert_count": 3,
                "max_liq_rate_recent": 5.2,
                "tagged_rate": 0.04,
                "age_sec": 4.0,
                "dashboard_summary": "ETH summary",
                "priority_score": 120.0,
            }
        ],
        "run_summary": {
            "version": "v1",
            "run_type": "liquidation_watchlist",
            "inputs": {"symbols": ["ETHUSDT", "BTCUSDT"]},
            "metrics": {"symbol_count": 2, "elevated_count": 1, "severe_count": 0, "quiet_count": 1},
            "artifacts": {"json": "reports/LIQUIDATION_WATCHLIST.json", "md": "reports/LIQUIDATION_WATCHLIST.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "liquidation_watchlist"
    assert rsv.validate_payload(payload, "liquidation_watchlist") == []


def test_validate_spread_stress_alerts_payload() -> None:
    payload = {
        "symbol": "ETHUSDT",
        "lookback_min": 240,
        "bucket_sec": 5,
        "summary": {
            "rows_total": 100,
            "tagged_count": 5,
            "tagged_rate": 0.05,
            "recent_alert_count": 3,
            "high_count": 1,
            "medium_count": 2,
            "avg_spread_tagged": 0.01,
            "avg_trade_intensity_tagged": 5.0,
        },
        "alerts": [
            {"ts_ms": 1, "severity": "high", "spread": 0.02, "trade_intensity": 1.0, "ret_1": -0.001}
        ],
        "run_summary": {
            "version": "v1",
            "run_type": "spread_stress_alerts",
            "inputs": {"symbol": "ETHUSDT"},
            "metrics": {"rows_total": 100, "tagged_count": 5},
            "artifacts": {"json": "reports/SPREAD_STRESS_ALERTS.json", "md": "reports/SPREAD_STRESS_ALERTS.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "spread_stress_alerts"
    assert rsv.validate_payload(payload, "spread_stress_alerts") == []


def test_validate_spread_stress_state_payload() -> None:
    payload = {
        "source_json": "reports/SPREAD_STRESS_ALERTS_REAL.json",
        "symbol": "ETHUSDT",
        "state": {
            "level": "elevated",
            "reasons": ["recent_spread_stress_cluster"],
            "freshness": {"status": "fresh", "age_sec": 4.0, "stale_after_sec": 60},
        },
        "dashboard_summary": "ETHUSDT elevated spread stress, 4 recent alerts, freshness fresh.",
        "notification_text": "[spread-stress] symbol=ETHUSDT level=elevated freshness=fresh recent_alerts=4 avg_spread=0.000150 action=show_caution",
        "recommended_action": "show_caution",
        "card": {
            "headline": "ETHUSDT spread stress elevated: 4 recent alerts",
            "operator_note": "Show caution for passive execution quality.",
            "recent_alert_count": 4,
            "tagged_rate": 0.05,
            "high_count": 0,
            "medium_count": 4,
            "avg_spread_tagged": 0.00015,
            "avg_trade_intensity_tagged": 500.0,
            "latest_alert_ts_ms": 1,
            "freshness_status": "fresh",
            "age_sec": 4.0,
        },
        "summary_snapshot": {
            "rows_total": 100,
            "tagged_count": 5,
            "tagged_rate": 0.05,
            "recent_alert_count": 4,
            "high_count": 0,
            "medium_count": 4,
            "avg_spread_tagged": 0.00015,
            "avg_trade_intensity_tagged": 500.0,
        },
        "run_summary": {
            "version": "v1",
            "run_type": "spread_stress_state",
            "inputs": {"source_json": "reports/SPREAD_STRESS_ALERTS_REAL.json"},
            "metrics": {"state_level": "elevated", "recommended_action": "show_caution"},
            "artifacts": {"json": "reports/SPREAD_STRESS_STATE.json", "md": "reports/SPREAD_STRESS_STATE.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "spread_stress_state"
    assert rsv.validate_payload(payload, "spread_stress_state") == []


def test_validate_return_shock_alerts_payload() -> None:
    payload = {
        "symbol": "ETHUSDT",
        "lookback_min": 240,
        "bucket_sec": 5,
        "summary": {
            "rows_total": 100,
            "tagged_count": 5,
            "tagged_rate": 0.05,
            "recent_alert_count": 3,
            "high_count": 1,
            "medium_count": 2,
            "avg_abs_ret_1_tagged": 0.0012,
            "avg_trade_intensity_tagged": 450.0,
            "direction_counts": {"UP": 1, "DOWN": 2, "FLAT": 0},
        },
        "alerts": [
            {
                "ts_ms": 1,
                "severity": "high",
                "direction": "DOWN",
                "ret_1": -0.003,
                "abs_ret_1": 0.003,
                "spread": 0.0001,
                "trade_intensity": 500.0,
            }
        ],
        "run_summary": {
            "version": "v1",
            "run_type": "return_shock_alerts",
            "inputs": {"symbol": "ETHUSDT"},
            "metrics": {"rows_total": 100, "tagged_count": 5},
            "artifacts": {"json": "reports/RETURN_SHOCK_ALERTS.json", "md": "reports/RETURN_SHOCK_ALERTS.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "return_shock_alerts"
    assert rsv.validate_payload(payload, "return_shock_alerts") == []


def test_validate_return_shock_state_payload() -> None:
    payload = {
        "source_json": "reports/RETURN_SHOCK_ALERTS_REAL.json",
        "symbol": "ETHUSDT",
        "state": {
            "level": "elevated",
            "reasons": ["recent_return_shock_cluster"],
            "dominant_direction": "DOWN",
            "freshness": {"status": "fresh", "age_sec": 4.0, "stale_after_sec": 60},
        },
        "dashboard_summary": "ETHUSDT elevated return shock, 3 recent alerts, freshness fresh.",
        "notification_text": "[return-shock] symbol=ETHUSDT level=elevated freshness=fresh direction=DOWN recent_alerts=3 avg_abs_ret=0.001200 action=show_caution",
        "recommended_action": "show_caution",
        "card": {
            "headline": "ETHUSDT return shock elevated",
            "operator_note": "Use as event context; do not map directly to trade direction.",
            "recent_alert_count": 3,
            "tagged_rate": 0.05,
            "high_count": 1,
            "medium_count": 2,
            "avg_abs_ret_1_tagged": 0.0012,
            "avg_trade_intensity_tagged": 450.0,
            "dominant_direction": "DOWN",
            "latest_alert_ts_ms": 1,
            "freshness_status": "fresh",
            "age_sec": 4.0,
        },
        "summary_snapshot": {
            "rows_total": 100,
            "tagged_count": 5,
            "tagged_rate": 0.05,
            "recent_alert_count": 3,
            "high_count": 1,
            "medium_count": 2,
            "avg_abs_ret_1_tagged": 0.0012,
            "avg_trade_intensity_tagged": 450.0,
            "direction_counts": {"UP": 1, "DOWN": 2, "FLAT": 0},
        },
        "run_summary": {
            "version": "v1",
            "run_type": "return_shock_state",
            "inputs": {"source_json": "reports/RETURN_SHOCK_ALERTS_REAL.json"},
            "metrics": {"state_level": "elevated", "recommended_action": "show_caution"},
            "artifacts": {"json": "reports/RETURN_SHOCK_STATE.json", "md": "reports/RETURN_SHOCK_STATE.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "return_shock_state"
    assert rsv.validate_payload(payload, "return_shock_state") == []


def test_validate_return_shock_watchlist_payload() -> None:
    payload = {
        "lookback_min": 240,
        "bucket_sec": 5,
        "recent_limit": 20,
        "summary": {"symbol_count": 2, "top_n": 2, "state_counts": {"severe": 1, "elevated": 1}, "top_symbol": "ETHUSDT"},
        "top_summary": {
            "symbol": "ETHUSDT",
            "state_level": "severe",
            "freshness_status": "fresh",
            "recommended_action": "escalate_monitoring",
            "dashboard_summary": "ETH severe return shock",
        },
        "banner": {
            "headline": "Return shock watchlist top=ETHUSDT level=severe freshness=fresh action=escalate_monitoring",
            "recommended_action": "escalate_monitoring",
            "top_symbol": "ETHUSDT",
            "top_state_level": "severe",
            "top_freshness_status": "fresh",
            "severe_count": 1,
            "elevated_count": 1,
            "quiet_count": 0,
        },
        "rows": [
            {
                "symbol": "ETHUSDT",
                "state_level": "severe",
                "freshness_status": "fresh",
                "recommended_action": "escalate_monitoring",
                "dominant_direction": "DOWN",
                "recent_alert_count": 6,
                "high_count": 2,
                "medium_count": 4,
                "avg_abs_ret_1_tagged": 0.0021,
                "avg_trade_intensity_tagged": 400.0,
                "age_sec": 3.0,
                "dashboard_summary": "ETH severe return shock",
                "priority_score": 227.0,
            }
        ],
        "run_summary": {
            "version": "v1",
            "run_type": "return_shock_watchlist",
            "inputs": {"symbols": ["ETHUSDT", "BTCUSDT"]},
            "metrics": {"symbol_count": 2},
            "artifacts": {"json": "reports/RETURN_SHOCK_WATCHLIST.json", "md": "reports/RETURN_SHOCK_WATCHLIST.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "return_shock_watchlist"
    assert rsv.validate_payload(payload, "return_shock_watchlist") == []


def test_validate_volume_vacuum_alerts_payload() -> None:
    payload = {
        "lane": "volume_vacuum",
        "symbol": "ETHUSDT",
        "lookback_min": 240,
        "bucket_sec": 5,
        "summary": {
            "rows_total": 100,
            "tagged_count": 5,
            "tagged_rate": 0.05,
            "recent_alert_count": 3,
            "high_count": 1,
            "medium_count": 2,
            "avg_trade_intensity_tagged": 20.0,
            "avg_spread_tagged": 0.0002,
        },
        "alerts": [
            {"ts_ms": 1, "severity": "high", "trade_intensity": 10.0, "spread": 0.0002, "ret_1": 0.0}
        ],
        "run_summary": {
            "version": "v1",
            "run_type": "volume_vacuum_alerts",
            "inputs": {"symbol": "ETHUSDT"},
            "metrics": {"rows_total": 100, "tagged_count": 5},
            "artifacts": {"json": "reports/VOLUME_VACUUM_ALERTS.json", "md": "reports/VOLUME_VACUUM_ALERTS.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "volume_vacuum_alerts"
    assert rsv.validate_payload(payload, "volume_vacuum_alerts") == []


def test_validate_volume_vacuum_state_payload() -> None:
    payload = {
        "lane": "volume_vacuum",
        "source_json": "reports/VOLUME_VACUUM_ALERTS_REAL.json",
        "symbol": "ETHUSDT",
        "state": {
            "level": "elevated",
            "reasons": ["recent_volume_vacuum_cluster"],
            "freshness": {"status": "fresh", "age_sec": 4.0, "stale_after_sec": 60},
        },
        "dashboard_summary": "ETHUSDT elevated volume vacuum, 3 recent alerts, freshness fresh.",
        "notification_text": "[volume-vacuum] symbol=ETHUSDT level=elevated freshness=fresh recent_alerts=3 avg_trade_intensity=20.00 avg_spread=0.000200 action=monitor_only",
        "recommended_action": "monitor_only",
        "card": {
            "headline": "ETHUSDT volume vacuum elevated",
            "operator_note": "Thin/quiet market context; use as passive execution caution, not directional signal.",
            "recent_alert_count": 3,
            "tagged_rate": 0.05,
            "high_count": 1,
            "medium_count": 2,
            "avg_trade_intensity_tagged": 20.0,
            "avg_spread_tagged": 0.0002,
            "latest_alert_ts_ms": 1,
            "freshness_status": "fresh",
            "age_sec": 4.0,
        },
        "summary_snapshot": {
            "rows_total": 100,
            "tagged_count": 5,
            "tagged_rate": 0.05,
            "recent_alert_count": 3,
            "high_count": 1,
            "medium_count": 2,
            "avg_trade_intensity_tagged": 20.0,
            "avg_spread_tagged": 0.0002,
        },
        "run_summary": {
            "version": "v1",
            "run_type": "volume_vacuum_state",
            "inputs": {"source_json": "reports/VOLUME_VACUUM_ALERTS_REAL.json"},
            "metrics": {"state_level": "elevated", "recommended_action": "monitor_only"},
            "artifacts": {"json": "reports/VOLUME_VACUUM_STATE.json", "md": "reports/VOLUME_VACUUM_STATE.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "volume_vacuum_state"
    assert rsv.validate_payload(payload, "volume_vacuum_state") == []


def test_validate_volume_vacuum_watchlist_payload() -> None:
    payload = {
        "lane": "volume_vacuum",
        "lookback_min": 240,
        "bucket_sec": 5,
        "recent_limit": 20,
        "summary": {"symbol_count": 2, "top_n": 2, "state_counts": {"severe": 1, "elevated": 1}, "top_symbol": "ETHUSDT"},
        "top_summary": {
            "symbol": "ETHUSDT",
            "state_level": "severe",
            "freshness_status": "fresh",
            "recommended_action": "show_caution",
            "dashboard_summary": "ETH severe volume vacuum",
        },
        "banner": {
            "headline": "Volume vacuum watchlist top=ETHUSDT level=severe freshness=fresh action=show_caution",
            "recommended_action": "show_caution",
            "top_symbol": "ETHUSDT",
            "top_state_level": "severe",
            "top_freshness_status": "fresh",
            "severe_count": 1,
            "elevated_count": 1,
            "quiet_count": 0,
        },
        "rows": [
            {
                "symbol": "ETHUSDT",
                "state_level": "severe",
                "freshness_status": "fresh",
                "recommended_action": "show_caution",
                "recent_alert_count": 6,
                "high_count": 2,
                "medium_count": 4,
                "avg_trade_intensity_tagged": 20.0,
                "avg_spread_tagged": 0.00021,
                "age_sec": 3.0,
                "dashboard_summary": "ETH severe volume vacuum",
                "priority_score": 227.0,
            }
        ],
        "run_summary": {
            "version": "v1",
            "run_type": "volume_vacuum_watchlist",
            "inputs": {"symbols": ["ETHUSDT", "BTCUSDT"]},
            "metrics": {"symbol_count": 2, "severe_count": 1, "elevated_count": 1, "quiet_count": 0},
            "artifacts": {"json": "reports/VOLUME_VACUUM_WATCHLIST.json", "md": "reports/VOLUME_VACUUM_WATCHLIST.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "volume_vacuum_watchlist"
    assert rsv.validate_payload(payload, "volume_vacuum_watchlist") == []


def test_validate_volatility_burst_alerts_payload() -> None:
    payload = {
        "lane": "volatility_burst",
        "symbol": "ETHUSDT",
        "lookback_min": 240,
        "bucket_sec": 5,
        "summary": {
            "rows_total": 100,
            "tagged_count": 5,
            "tagged_rate": 0.05,
            "recent_alert_count": 3,
            "high_count": 1,
            "medium_count": 2,
            "avg_abs_ret_1_tagged": 0.001,
            "avg_trade_intensity_tagged": 400.0,
            "direction_counts": {"UP": 2, "DOWN": 1, "FLAT": 0},
        },
        "alerts": [
            {"ts_ms": 1, "severity": "high", "direction": "UP", "ret_1": 0.002, "abs_ret_1": 0.002, "trade_intensity": 500.0, "spread": 0.0002}
        ],
        "run_summary": {
            "version": "v1",
            "run_type": "volatility_burst_alerts",
            "inputs": {"symbol": "ETHUSDT"},
            "metrics": {"rows_total": 100, "tagged_count": 5},
            "artifacts": {"json": "reports/VOLATILITY_BURST_ALERTS.json", "md": "reports/VOLATILITY_BURST_ALERTS.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "volatility_burst_alerts"
    assert rsv.validate_payload(payload, "volatility_burst_alerts") == []


def test_validate_volatility_burst_state_payload() -> None:
    payload = {
        "lane": "volatility_burst",
        "source_json": "reports/VOLATILITY_BURST_ALERTS_REAL.json",
        "symbol": "ETHUSDT",
        "state": {
            "level": "elevated",
            "reasons": ["recent_volatility_burst_cluster"],
            "dominant_direction": "UP",
            "freshness": {"status": "fresh", "age_sec": 4.0, "stale_after_sec": 60},
        },
        "dashboard_summary": "ETHUSDT elevated volatility burst, 3 recent alerts, freshness fresh.",
        "notification_text": "[volatility-burst] symbol=ETHUSDT level=elevated freshness=fresh direction=UP recent_alerts=3 avg_abs_ret=0.001000 action=show_caution",
        "recommended_action": "show_caution",
        "card": {
            "headline": "ETHUSDT volatility burst elevated",
            "operator_note": "Use as expansion context; do not map directly to execution or trade direction.",
            "recent_alert_count": 3,
            "tagged_rate": 0.05,
            "high_count": 1,
            "medium_count": 2,
            "avg_abs_ret_1_tagged": 0.001,
            "avg_trade_intensity_tagged": 400.0,
            "dominant_direction": "UP",
            "latest_alert_ts_ms": 1,
            "freshness_status": "fresh",
            "age_sec": 4.0,
        },
        "summary_snapshot": {
            "rows_total": 100,
            "tagged_count": 5,
            "tagged_rate": 0.05,
            "recent_alert_count": 3,
            "high_count": 1,
            "medium_count": 2,
            "avg_abs_ret_1_tagged": 0.001,
            "avg_trade_intensity_tagged": 400.0,
            "direction_counts": {"UP": 2, "DOWN": 1, "FLAT": 0},
        },
        "run_summary": {
            "version": "v1",
            "run_type": "volatility_burst_state",
            "inputs": {"source_json": "reports/VOLATILITY_BURST_ALERTS_REAL.json"},
            "metrics": {"state_level": "elevated", "recommended_action": "show_caution"},
            "artifacts": {"json": "reports/VOLATILITY_BURST_STATE.json", "md": "reports/VOLATILITY_BURST_STATE.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "volatility_burst_state"
    assert rsv.validate_payload(payload, "volatility_burst_state") == []


def test_validate_volatility_burst_watchlist_payload() -> None:
    payload = {
        "lane": "volatility_burst",
        "lookback_min": 240,
        "bucket_sec": 5,
        "recent_limit": 20,
        "summary": {"symbol_count": 2, "top_n": 2, "state_counts": {"severe": 1, "elevated": 1}, "top_symbol": "ETHUSDT"},
        "top_summary": {
            "symbol": "ETHUSDT",
            "state_level": "severe",
            "freshness_status": "fresh",
            "recommended_action": "escalate_monitoring",
            "dashboard_summary": "ETH severe volatility burst",
        },
        "banner": {
            "headline": "Volatility burst watchlist top=ETHUSDT level=severe freshness=fresh action=escalate_monitoring",
            "recommended_action": "escalate_monitoring",
            "top_symbol": "ETHUSDT",
            "top_state_level": "severe",
            "top_freshness_status": "fresh",
            "severe_count": 1,
            "elevated_count": 1,
            "quiet_count": 0,
        },
        "rows": [
            {
                "symbol": "ETHUSDT",
                "state_level": "severe",
                "freshness_status": "fresh",
                "recommended_action": "escalate_monitoring",
                "dominant_direction": "UP",
                "recent_alert_count": 6,
                "high_count": 2,
                "medium_count": 4,
                "avg_abs_ret_1_tagged": 0.0021,
                "avg_trade_intensity_tagged": 400.0,
                "age_sec": 3.0,
                "dashboard_summary": "ETH severe volatility burst",
                "priority_score": 247.0,
            }
        ],
        "run_summary": {
            "version": "v1",
            "run_type": "volatility_burst_watchlist",
            "inputs": {"symbols": ["ETHUSDT", "BTCUSDT"]},
            "metrics": {"symbol_count": 2, "severe_count": 1, "elevated_count": 1, "quiet_count": 0},
            "artifacts": {"json": "reports/VOLATILITY_BURST_WATCHLIST.json", "md": "reports/VOLATILITY_BURST_WATCHLIST.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "volatility_burst_watchlist"
    assert rsv.validate_payload(payload, "volatility_burst_watchlist") == []


def test_validate_book_proxy_pressure_alerts_payload() -> None:
    payload = {
        "lane": "book_proxy_pressure",
        "symbol": "ETHUSDT",
        "lookback_min": 240,
        "bucket_sec": 5,
        "summary": {
            "rows_total": 100,
            "tagged_count": 5,
            "tagged_rate": 0.05,
            "recent_alert_count": 3,
            "high_count": 1,
            "medium_count": 2,
            "avg_abs_imbalance_tagged": 0.8,
            "avg_trade_intensity_tagged": 400.0,
            "avg_spread_tagged": 0.0008,
            "side_bias_counts": {"LONG": 2, "SHORT": 1, "NEUTRAL": 0},
        },
        "alerts": [
            {"ts_ms": 1, "severity": "high", "side_bias": "LONG", "imbalance": 0.9, "abs_imbalance": 0.9, "trade_intensity": 500.0, "spread": 0.0008, "ret_1": 0.0}
        ],
        "run_summary": {
            "version": "v1",
            "run_type": "book_proxy_pressure_alerts",
            "inputs": {"symbol": "ETHUSDT"},
            "metrics": {"rows_total": 100, "tagged_count": 5},
            "artifacts": {"json": "reports/BOOK_PROXY_PRESSURE_ALERTS.json", "md": "reports/BOOK_PROXY_PRESSURE_ALERTS.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "book_proxy_pressure_alerts"
    assert rsv.validate_payload(payload, "book_proxy_pressure_alerts") == []


def test_validate_book_proxy_pressure_state_payload() -> None:
    payload = {
        "lane": "book_proxy_pressure",
        "source_json": "reports/BOOK_PROXY_PRESSURE_ALERTS_REAL.json",
        "symbol": "ETHUSDT",
        "state": {
            "level": "elevated",
            "reasons": ["recent_proxy_pressure_cluster"],
            "primary_side_bias": "LONG",
            "freshness": {"status": "fresh", "age_sec": 4.0, "stale_after_sec": 60},
        },
        "dashboard_summary": "ETHUSDT elevated book proxy pressure, 3 recent alerts, freshness fresh.",
        "notification_text": "[book-proxy-pressure] symbol=ETHUSDT level=elevated freshness=fresh side_bias=LONG recent_alerts=3 avg_abs_imbalance=0.8000 action=monitor_only",
        "recommended_action": "monitor_only",
        "card": {
            "headline": "ETHUSDT book proxy pressure elevated",
            "operator_note": "Proxy lane built from spread, trade intensity and imbalance; not true order-book depth.",
            "recent_alert_count": 3,
            "tagged_rate": 0.05,
            "high_count": 1,
            "medium_count": 2,
            "avg_abs_imbalance_tagged": 0.8,
            "avg_trade_intensity_tagged": 400.0,
            "avg_spread_tagged": 0.0008,
            "primary_side_bias": "LONG",
            "latest_alert_ts_ms": 1,
            "freshness_status": "fresh",
            "age_sec": 4.0,
        },
        "summary_snapshot": {
            "rows_total": 100,
            "tagged_count": 5,
            "tagged_rate": 0.05,
            "recent_alert_count": 3,
            "high_count": 1,
            "medium_count": 2,
            "avg_abs_imbalance_tagged": 0.8,
            "avg_trade_intensity_tagged": 400.0,
            "avg_spread_tagged": 0.0008,
            "side_bias_counts": {"LONG": 2, "SHORT": 1, "NEUTRAL": 0},
        },
        "run_summary": {
            "version": "v1",
            "run_type": "book_proxy_pressure_state",
            "inputs": {"source_json": "reports/BOOK_PROXY_PRESSURE_ALERTS_REAL.json"},
            "metrics": {"state_level": "elevated", "recommended_action": "monitor_only"},
            "artifacts": {"json": "reports/BOOK_PROXY_PRESSURE_STATE.json", "md": "reports/BOOK_PROXY_PRESSURE_STATE.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "book_proxy_pressure_state"
    assert rsv.validate_payload(payload, "book_proxy_pressure_state") == []


def test_validate_book_proxy_pressure_watchlist_payload() -> None:
    payload = {
        "lane": "book_proxy_pressure",
        "lookback_min": 240,
        "bucket_sec": 5,
        "recent_limit": 20,
        "summary": {"symbol_count": 2, "top_n": 2, "state_counts": {"severe": 1, "elevated": 1}, "top_symbol": "ETHUSDT"},
        "top_summary": {
            "symbol": "ETHUSDT",
            "state_level": "severe",
            "freshness_status": "fresh",
            "recommended_action": "show_caution",
            "dashboard_summary": "ETH severe book proxy pressure",
        },
        "banner": {
            "headline": "Book proxy pressure watchlist top=ETHUSDT level=severe freshness=fresh action=show_caution",
            "recommended_action": "show_caution",
            "top_symbol": "ETHUSDT",
            "top_state_level": "severe",
            "top_freshness_status": "fresh",
            "severe_count": 1,
            "elevated_count": 1,
            "quiet_count": 0,
        },
        "rows": [
            {
                "symbol": "ETHUSDT",
                "state_level": "severe",
                "freshness_status": "fresh",
                "recommended_action": "show_caution",
                "primary_side_bias": "SHORT",
                "recent_alert_count": 6,
                "high_count": 2,
                "medium_count": 4,
                "avg_abs_imbalance_tagged": 0.95,
                "avg_trade_intensity_tagged": 400.0,
                "avg_spread_tagged": 0.0008,
                "age_sec": 3.0,
                "dashboard_summary": "ETH severe book proxy pressure",
                "priority_score": 247.0,
            }
        ],
        "run_summary": {
            "version": "v1",
            "run_type": "book_proxy_pressure_watchlist",
            "inputs": {"symbols": ["ETHUSDT", "BTCUSDT"]},
            "metrics": {"symbol_count": 2, "severe_count": 1, "elevated_count": 1, "quiet_count": 0},
            "artifacts": {"json": "reports/BOOK_PROXY_PRESSURE_WATCHLIST.json", "md": "reports/BOOK_PROXY_PRESSURE_WATCHLIST.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "book_proxy_pressure_watchlist"
    assert rsv.validate_payload(payload, "book_proxy_pressure_watchlist") == []


def test_validate_spread_stress_watchlist_payload() -> None:
    payload = {
        "lookback_min": 240,
        "bucket_sec": 5,
        "recent_limit": 20,
        "summary": {"symbol_count": 2, "top_n": 2, "state_counts": {"severe": 1, "elevated": 1}, "top_symbol": "ETHUSDT"},
        "top_summary": {
            "symbol": "ETHUSDT",
            "state_level": "severe",
            "freshness_status": "fresh",
            "recommended_action": "reduce_passive_aggression",
            "dashboard_summary": "ETH severe spread stress",
        },
        "banner": {
            "headline": "Spread stress watchlist top=ETHUSDT level=severe freshness=fresh action=reduce_passive_aggression",
            "recommended_action": "reduce_passive_aggression",
            "top_symbol": "ETHUSDT",
            "top_state_level": "severe",
            "top_freshness_status": "fresh",
            "severe_count": 1,
            "elevated_count": 1,
            "quiet_count": 0,
        },
        "rows": [
            {
                "symbol": "ETHUSDT",
                "state_level": "severe",
                "freshness_status": "fresh",
                "recommended_action": "reduce_passive_aggression",
                "recent_alert_count": 6,
                "high_count": 2,
                "medium_count": 4,
                "avg_spread_tagged": 0.00021,
                "avg_trade_intensity_tagged": 400.0,
                "age_sec": 3.0,
                "dashboard_summary": "ETH severe spread stress",
                "priority_score": 227.0,
            }
        ],
        "run_summary": {
            "version": "v1",
            "run_type": "spread_stress_watchlist",
            "inputs": {"symbols": ["ETHUSDT", "BTCUSDT"]},
            "metrics": {"symbol_count": 2, "severe_count": 1, "elevated_count": 1, "quiet_count": 0},
            "artifacts": {"json": "reports/SPREAD_STRESS_WATCHLIST.json", "md": "reports/SPREAD_STRESS_WATCHLIST.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "spread_stress_watchlist"
    assert rsv.validate_payload(payload, "spread_stress_watchlist") == []


def test_validate_fill_toxicity_state_payload() -> None:
    payload = {
        "source": "data/live/papertrades_live.parquet",
        "rows": 10,
        "top_side": "buy",
        "state": {"level": "severe", "reasons": ["extreme_toxicity_score"]},
        "dashboard_summary": "fill toxicity severe, top_side buy, toxicity=1.700, adverse=2.200bps.",
        "notification_text": "[fill-toxicity] level=severe top_side=buy toxicity=1.7000 adverse_bps=2.2000 pnl_bps=-0.5000 action=reduce_passive_aggression",
        "recommended_action": "reduce_passive_aggression",
        "card": {
            "headline": "Fill toxicity severe",
            "operator_note": "Escalate monitoring and consider reducing passive aggression.",
            "top_side": "buy",
            "rows": 10,
            "toxicity_score": 1.7,
            "adverse_bps_mean": 2.2,
            "pnl_bps_mean": -0.5,
        },
        "summary_snapshot": {
            "rows": 10,
            "sides": {"buy": {"rows": 5, "adverse_bps_mean": 2.2, "pnl_bps_mean": -0.5, "toxicity_score": 1.7}},
        },
        "run_summary": {
            "version": "v1",
            "run_type": "fill_toxicity_state",
            "inputs": {"source": "data/live/papertrades_live.parquet"},
            "metrics": {"rows": 10, "side_count": 1, "state_level": "severe", "top_side": "buy", "top_toxicity_score": 1.7},
            "artifacts": {"json": "reports/FILL_TOXICITY_STATE.json", "md": "reports/FILL_TOXICITY_STATE.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "fill_toxicity_state"
    assert rsv.validate_payload(payload, "fill_toxicity_state") == []


def test_validate_latency_stress_state_payload() -> None:
    payload = {
        "source": "data/live/papertrades_live.parquet",
        "state": {"level": "severe", "reasons": ["very_high_p95_fill_delay"]},
        "dashboard_summary": "latency stress severe, p95=12.00s, p50=4.50s, fill_rate=18.00%.",
        "notification_text": "[latency-stress] level=severe p95=12.00s p50=4.50s fill_rate=0.1800 corr=-0.3000 action=escalate_monitoring",
        "recommended_action": "escalate_monitoring",
        "card": {
            "headline": "Latency stress severe",
            "operator_note": "Escalate latency monitoring and inspect runtime execution path.",
            "rows": 20,
            "fill_rate": 0.18,
            "latency_fill_delay_sec_p50": 4.5,
            "latency_fill_delay_sec_p95": 12.0,
            "latency_impact_vs_net_corr": -0.3,
        },
        "summary_snapshot": {
            "rows": 20,
            "fill_rate": 0.18,
            "queue_competition_score": 0.5,
            "toxicity_score": 0.8,
            "adverse_selection_bps_mean": 1.2,
            "latency_fill_delay_sec_p50": 4.5,
            "latency_fill_delay_sec_p95": 12.0,
            "latency_impact_vs_net_corr": -0.3,
        },
        "run_summary": {
            "version": "v1",
            "run_type": "latency_stress_state",
            "inputs": {"source": "data/live/papertrades_live.parquet"},
            "metrics": {"rows": 20, "state_level": "severe", "fill_rate": 0.18, "latency_fill_delay_sec_p95": 12.0},
            "artifacts": {"json": "reports/LATENCY_STRESS_STATE.json", "md": "reports/LATENCY_STRESS_STATE.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "latency_stress_state"
    assert rsv.validate_payload(payload, "latency_stress_state") == []


def test_validate_research_event_watchboard_payload() -> None:
    payload = {
        "summary": {"lane_count": 4, "state_counts": {"severe": 1, "quiet": 3}, "top_lane": "liquidation"},
        "top_event": {
            "lane": "liquidation",
            "level": "severe",
            "recommended_action": "escalate_monitoring",
            "headline": "Liq top ETH",
            "detail": "ETH severe liq",
        },
        "banner": {
            "headline": "Liq top ETH",
            "recommended_action": "escalate_monitoring",
            "top_lane": "liquidation",
            "top_level": "severe",
        },
        "lanes": [
            {
                "lane": "liquidation",
                "level": "severe",
                "freshness_status": "fresh",
                "recommended_action": "escalate_monitoring",
                "headline": "Liq top ETH",
                "detail": "ETH severe liq",
                "priority_score": 225.0,
            }
        ],
        "run_summary": {
            "version": "v1",
            "run_type": "research_event_watchboard",
            "inputs": {"symbols": ["ETHUSDT", "BTCUSDT"]},
            "metrics": {"lane_count": 4, "severe_count": 1, "elevated_count": 0, "quiet_count": 3},
            "artifacts": {"json": "reports/RESEARCH_EVENT_WATCHBOARD.json", "md": "reports/RESEARCH_EVENT_WATCHBOARD.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "research_event_watchboard"
    assert rsv.validate_payload(payload, "research_event_watchboard") == []


def test_validate_event_watchboard_trend_payload() -> None:
    payload = {
        "summary": {
            "snapshot_count": 2,
            "start_top_lane": "spread_stress",
            "end_top_lane": "liquidation",
            "delta_priority_score": 100.0,
            "trend": "rising_fast",
        },
        "latest": {
            "index": 1,
            "source": "b.json",
            "top_lane": "liquidation",
            "top_level": "severe",
            "top_recommended_action": "escalate_monitoring",
            "priority_score": 225.0,
        },
        "points": [
            {
                "index": 0,
                "source": "a.json",
                "top_lane": "spread_stress",
                "top_level": "elevated",
                "top_recommended_action": "show_caution",
                "priority_score": 125.0,
            },
            {
                "index": 1,
                "source": "b.json",
                "top_lane": "liquidation",
                "top_level": "severe",
                "top_recommended_action": "escalate_monitoring",
                "priority_score": 225.0,
            },
        ],
        "lane_deltas": [
            {
                "lane": "liquidation",
                "start_priority_score": 0.0,
                "end_priority_score": 225.0,
                "delta_priority_score": 225.0,
                "trend": "rising_fast",
            },
            {
                "lane": "spread_stress",
                "start_priority_score": 125.0,
                "end_priority_score": 50.0,
                "delta_priority_score": -75.0,
                "trend": "falling_fast",
            },
        ],
        "run_summary": {
            "version": "v1",
            "run_type": "event_watchboard_trend",
            "inputs": {"sources": ["a.json", "b.json"]},
            "metrics": {"snapshot_count": 2, "delta_priority_score": 100.0, "trend": "rising_fast"},
            "artifacts": {"json": "reports/RESEARCH_EVENT_WATCHBOARD_TREND.json", "md": "reports/RESEARCH_EVENT_WATCHBOARD_TREND.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "event_watchboard_trend"
    assert rsv.validate_payload(payload, "event_watchboard_trend") == []


def test_validate_event_watchboard_snapshot_append_payload() -> None:
    payload = {
        "history_path": "reports/RESEARCH_EVENT_WATCHBOARD_HISTORY.jsonl",
        "appended": {
            "source": "reports/RESEARCH_EVENT_WATCHBOARD_REAL.json",
            "top_lane": "liquidation",
            "state_counts": {"severe": 2},
            "lanes": [
                {
                    "lane": "liquidation",
                    "priority_score": 225.0,
                    "level": "severe",
                    "freshness_status": "fresh",
                    "recommended_action": "escalate_monitoring",
                }
            ],
            "top_event": {
                "lane": "liquidation",
                "level": "severe",
                "recommended_action": "monitor_only",
                "headline": "Liq top ETH",
            },
            "banner": {
                "headline": "Liq top ETH",
                "recommended_action": "monitor_only",
                "top_lane": "liquidation",
                "top_level": "severe",
            },
            "upstream_run_type": "research_event_watchboard",
        },
        "run_summary": {
            "version": "v1",
            "run_type": "event_watchboard_snapshot_append",
            "inputs": {"source": "reports/RESEARCH_EVENT_WATCHBOARD_REAL.json", "history_path": "reports/RESEARCH_EVENT_WATCHBOARD_HISTORY.jsonl"},
            "metrics": {"top_lane": "liquidation", "severe_count": 2},
            "artifacts": {"json": "reports/RESEARCH_EVENT_WATCHBOARD_SNAPSHOT_APPEND.json", "history_jsonl": "reports/RESEARCH_EVENT_WATCHBOARD_HISTORY.jsonl"},
        },
    }
    assert rsv.infer_schema_name(payload) == "event_watchboard_snapshot_append"
    assert rsv.validate_payload(payload, "event_watchboard_snapshot_append") == []


def test_validate_event_watchboard_trend_from_history_payload() -> None:
    payload = {
        "summary": {
            "snapshot_count": 2,
            "start_top_lane": "spread_stress",
            "end_top_lane": "liquidation",
            "delta_priority_score": 100.0,
            "trend": "rising_fast",
        },
        "latest": {
            "index": 1,
            "source": "b.json",
            "top_lane": "liquidation",
            "top_level": "severe",
            "top_recommended_action": "monitor_only",
            "priority_score": 200.0,
        },
        "points": [
            {
                "index": 0,
                "source": "a.json",
                "top_lane": "spread_stress",
                "top_level": "elevated",
                "top_recommended_action": "show_caution",
                "priority_score": 100.0,
            },
            {
                "index": 1,
                "source": "b.json",
                "top_lane": "liquidation",
                "top_level": "severe",
                "top_recommended_action": "monitor_only",
                "priority_score": 200.0,
            },
        ],
        "lane_deltas": [
            {
                "lane": "liquidation",
                "start_priority_score": 0.0,
                "end_priority_score": 200.0,
                "delta_priority_score": 200.0,
                "trend": "rising_fast",
            }
        ],
        "history": {
            "history_path": "reports/RESEARCH_EVENT_WATCHBOARD_HISTORY.jsonl",
            "last_n": 2,
            "available_rows": 5,
            "used_rows": 2,
        },
        "run_summary": {
            "version": "v1",
            "run_type": "event_watchboard_trend_from_history",
            "inputs": {"history_path": "reports/RESEARCH_EVENT_WATCHBOARD_HISTORY.jsonl", "last_n": 2},
            "metrics": {"available_rows": 5, "used_rows": 2, "delta_priority_score": 100.0, "trend": "rising_fast"},
            "artifacts": {"json": "reports/RESEARCH_EVENT_WATCHBOARD_TREND_FROM_HISTORY.json", "md": "reports/RESEARCH_EVENT_WATCHBOARD_TREND_FROM_HISTORY.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "event_watchboard_trend_from_history"
    assert rsv.validate_payload(payload, "event_watchboard_trend_from_history") == []
def test_validate_artifacts_payload() -> None:
    payload = {
        "ok": True,
        "calibration": {"path": "cal.json", "ok": True, "errors": []},
        "execution": {"path": "exe.json", "ok": True, "errors": []},
        "run_summary": {
            "version": "v1",
            "run_type": "validate_artifacts",
            "inputs": {"calibration": "cal.json", "execution": "exe.json"},
            "metrics": {"ok": True, "calibration_ok": True, "execution_ok": True},
            "artifacts": {"json": "reports/validate_artifacts.json", "report": "reports/validate_artifacts.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "validate_artifacts"
    assert rsv.validate_payload(payload, "validate_artifacts") == []


def test_validate_report_check_payload() -> None:
    payload = {
        "results": [{"path": "reports/x.json", "ok": True, "schema": "validate_canonical", "errors": []}],
        "summary": {"checked": 1, "ok_count": 1, "fail_count": 0},
        "run_summary": {
            "version": "v1",
            "run_type": "report_check",
            "inputs": {"inputs": ["reports/x.json"]},
            "metrics": {"checked": 1, "ok_count": 1, "fail_count": 0},
            "artifacts": {"json": "reports/report_check.json"},
        },
    }
    assert rsv.infer_schema_name(payload) == "report_check"
    assert rsv.validate_payload(payload, "report_check") == []


def test_validate_analyze_cost_breakdown_payload() -> None:
    payload = {
        "tool": "analyze_cost_breakdown",
        "source_json": "reports/RANK_V3_costgrid.json",
        "n_pockets": 1,
        "pockets": [{"symbol": "ETHUSDT", "current_npa_bps": -0.2}],
        "run_summary": {
            "version": "v1",
            "run_type": "analyze_cost_breakdown",
            "inputs": {"source_json": "reports/RANK_V3_costgrid.json"},
            "metrics": {"n_pockets": 1},
            "artifacts": {"json": "reports/COST_BREAKDOWN.json", "md": "reports/COST_BREAKDOWN.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "analyze_cost_breakdown"
    assert rsv.validate_payload(payload, "analyze_cost_breakdown") == []


def test_validate_analyze_fill_timing_payload() -> None:
    payload = {
        "status": "ok",
        "live_parquet": "data/live/papertrades_live.parquet",
        "trade_db": "data/paper_trades.db",
        "bar_sec": 1.0,
        "timeout_candidates": [5.0, 10.0, 30.0],
        "live_summary": {"rows": 2, "recommended_timeout_sec": 10.0},
        "trade_db_summary": {"rows": 1},
        "run_summary": {
            "version": "v1",
            "run_type": "analyze_fill_timing",
            "inputs": {"live_parquet": "data/live/papertrades_live.parquet", "trade_db": "data/paper_trades.db"},
            "metrics": {"live_rows": 2, "trade_db_rows": 1, "recommended_timeout_sec": 10.0},
            "artifacts": {"json": "reports/FILL_TIMING_ANALYSIS.json", "md": "reports/FILL_TIMING_ANALYSIS.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "analyze_fill_timing"
    assert rsv.validate_payload(payload, "analyze_fill_timing") == []


def test_validate_daily_execution_calibration_payload() -> None:
    payload = {
        "ts_utc": "2026-03-05T00:00:00Z",
        "symbol": "ETHUSDT",
        "interval_ms": 100,
        "days": 14,
        "ok": True,
        "steps": [{"cmd": ["python", "-m", "tools.execution_e2e_pipeline"], "rc": 0}],
        "run_summary": {
            "version": "v1",
            "run_type": "daily_execution_calibration",
            "inputs": {"symbol": "ETHUSDT", "interval_ms": 100, "days": 14, "run_root_cause": 0},
            "metrics": {"ok": True, "step_count": 1},
            "artifacts": {"json": "reports/daily/2026-03-05_EXEC_CALIBRATION.json", "md": "reports/daily/2026-03-05_EXEC_CALIBRATION.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "daily_execution_calibration"
    assert rsv.validate_payload(payload, "daily_execution_calibration") == []


def test_validate_execution_diagnostics_payload() -> None:
    payload = {
        "rows": 3,
        "fill_rate": 0.67,
        "queue_competition_score": 0.4,
        "toxicity_score": 1.2,
        "run_summary": {
            "version": "v1",
            "run_type": "execution_diagnostics",
            "inputs": {"source": "data/live/papertrades_live.parquet"},
            "metrics": {"rows": 3, "fill_rate": 0.67},
            "artifacts": {"json": "reports/EXECUTION_HEALTH.json", "md": "reports/EXECUTION_HEALTH.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "execution_diagnostics"
    assert rsv.validate_payload(payload, "execution_diagnostics") == []


def test_validate_preflight_check_payload() -> None:
    payload = {
        "ok": True,
        "failures": [],
        "warnings": ["ACTIVE_SYMBOLS is empty"],
        "checks": {"db_exists": True, "disk_free_gb": 100.0},
        "run_summary": {
            "version": "v1",
            "run_type": "preflight_check",
            "inputs": {"db": "data/microstructure.db", "trade_db": "data/paper_trades.db"},
            "metrics": {"ok": True, "failure_count": 0, "warning_count": 1},
            "artifacts": {"json": "reports/PREFLIGHT_CHECK.json", "md": "reports/PREFLIGHT_CHECK.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "preflight_check"
    assert rsv.validate_payload(payload, "preflight_check") == []


def test_validate_paper_trade_summary_payload() -> None:
    payload = {
        "total_trades": 2,
        "win_rate": 0.5,
        "mean_pnl_bps": -5.0,
        "total_pnl_bps": -10.0,
        "scratch_rate": 0.5,
        "max_drawdown_bps": 20.0,
        "exit_types": {"SCRATCH": 1, "TAKE_PROFIT": 1},
        "daily": [{"date": "2026-03-05", "trades": 2}],
        "rolling7": [{"date": "2026-03-05", "roll7_total_pnl_bps": -10.0}],
        "run_summary": {
            "version": "v1",
            "run_type": "paper_trade_summary",
            "inputs": {"db": "data/paper_trades.db", "days": 7},
            "metrics": {"total_trades": 2, "win_rate": 0.5, "total_pnl_bps": -10.0},
            "artifacts": {"json": "reports/PAPER_TRADE_SUMMARY.json", "md": "reports/PAPER_TRADE_SUMMARY.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "paper_trade_summary"
    assert rsv.validate_payload(payload, "paper_trade_summary") == []


def test_validate_post_rollout_audit_payload() -> None:
    payload = {
        "ts_utc": "2026-03-05T00:00:00Z",
        "flags": {"EXEC_LATENCY_V2": True},
        "checks": {"diag_rows_ok": True},
        "overall_ok": True,
        "run_summary": {
            "version": "v1",
            "run_type": "post_rollout_audit",
            "inputs": {"diag_json": "reports/EXECUTION_HEALTH.json", "tox_json": "reports/TOXICITY_REPORT.json"},
            "metrics": {"overall_ok": True, "check_count": 1},
            "artifacts": {"json": "reports/POST_ROLLOUT_AUDIT.json", "md": "reports/POST_ROLLOUT_AUDIT.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "post_rollout_audit"
    assert rsv.validate_payload(payload, "post_rollout_audit") == []


def test_validate_toxicity_report_payload() -> None:
    payload = {
        "rows": 2,
        "sides": {"buy": {"rows": 1, "toxicity_score": 1.0}},
        "run_summary": {
            "version": "v1",
            "run_type": "toxicity_report",
            "inputs": {"source": "data/live/papertrades_live.parquet"},
            "metrics": {"rows": 2, "side_count": 1},
            "artifacts": {"json": "reports/TOXICITY_REPORT.json", "md": "reports/TOXICITY_REPORT.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "toxicity_report"
    assert rsv.validate_payload(payload, "toxicity_report") == []


def test_validate_replay_parity_report_payload() -> None:
    payload = {
        "sim_count": 3,
        "live_count": 2,
        "matched_count": 1,
        "match_rate_vs_sim": 0.33,
        "matches": [],
        "run_summary": {
            "version": "v1",
            "run_type": "replay_parity_report",
            "inputs": {"sim": "logs/x.jsonl", "live_db": "data/paper_trades.db"},
            "metrics": {"sim_count": 3, "live_count": 2, "matched_count": 1, "match_rate_vs_sim": 0.33},
            "artifacts": {"json": "reports/REPLAY_PARITY_REPORT.json", "md": "reports/REPLAY_PARITY_REPORT.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "replay_parity_report"
    assert rsv.validate_payload(payload, "replay_parity_report") == []


def test_validate_live_fill_drift_root_cause_payload() -> None:
    payload = {
        "ts_utc": "2026-03-05T00:00:00Z",
        "overall_status": "attention",
        "parity_json": "reports/REPLAY_PARITY_REPORT.json",
        "diagnostics_json": "reports/EXECUTION_HEALTH.json",
        "toxicity_json": "reports/TOXICITY_REPORT.json",
        "audit_json": "reports/POST_ROLLOUT_AUDIT.json",
        "causes": [{"name": "Latency Modeling Drift", "score": 1.0, "evidence": [], "actions": []}],
        "pipeline": {},
        "run_summary": {
            "version": "v1",
            "run_type": "live_fill_drift_root_cause",
            "inputs": {"parity_json": "reports/REPLAY_PARITY_REPORT.json", "diag_json": "reports/EXECUTION_HEALTH.json"},
            "metrics": {"overall_status": "attention", "cause_count": 1, "top_score": 1.0},
            "artifacts": {"json": "reports/LIVE_FILL_DRIFT_ROOT_CAUSE.json", "md": "reports/LIVE_FILL_DRIFT_ROOT_CAUSE.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "live_fill_drift_root_cause"
    assert rsv.validate_payload(payload, "live_fill_drift_root_cause") == []


def test_validate_execution_e2e_pipeline_payload() -> None:
    payload = {
        "ok": True,
        "steps": [{"cmd": ["python", "-m", "tools.replay_parity_report"], "rc": 0}],
        "run_summary": {
            "version": "v1",
            "run_type": "execution_e2e_pipeline",
            "inputs": {"sim": "logs/x.jsonl", "live_db": "data/paper_trades.db", "live_parquet": "data/live/papertrades_live.parquet"},
            "metrics": {"ok": True, "step_count": 1},
            "artifacts": {"json": "reports/EXECUTION_E2E_PIPELINE.json"},
        },
    }
    assert rsv.infer_schema_name(payload) == "execution_e2e_pipeline"
    assert rsv.validate_payload(payload, "execution_e2e_pipeline") == []


def test_validate_execution_quality_audit_payload() -> None:
    payload = {
        "status": "ok",
        "timestamp_utc": "2026-03-05T00:00:00Z",
        "input": "data/live/papertrades_live.parquet",
        "rows": 2,
        "overall": {"fill_rate": 0.5},
        "by_side": {"buy": {"rows": 1}},
        "run_summary": {
            "version": "v1",
            "run_type": "execution_quality_audit",
            "inputs": {"in_parquet": "data/live/papertrades_live.parquet", "last_n": 100},
            "metrics": {"status": "ok", "rows": 2, "fill_rate": 0.5},
            "artifacts": {"json": "reports/execution_quality_audit.json", "md": "reports/execution_quality_audit.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "execution_quality_audit"
    assert rsv.validate_payload(payload, "execution_quality_audit") == []


def test_validate_optimize_fill_timeout_payload() -> None:
    payload = {
        "analysis_json": "reports/FILL_TIMING_ANALYSIS.json",
        "env_file": ".env.paper",
        "current": 10,
        "recommended": 15,
        "raw_recommended": 15.0,
        "source": "timeout_eval",
        "reason": "best score",
        "applied": False,
        "run_summary": {
            "version": "v1",
            "run_type": "optimize_fill_timeout",
            "inputs": {"analysis_json": "reports/FILL_TIMING_ANALYSIS.json", "env_file": ".env.paper", "apply": False},
            "metrics": {"recommended": 15, "current": 10, "applied": False},
            "artifacts": {"json": "reports/FILL_TIMEOUT_RECOMMENDATION.json", "md": "reports/FILL_TIMEOUT_RECOMMENDATION.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "optimize_fill_timeout"
    assert rsv.validate_payload(payload, "optimize_fill_timeout") == []


def test_validate_fit_adverse_model_payload() -> None:
    payload = {
        "tool": "fit_adverse_model",
        "generated_utc": "2026-03-05T00:00:00Z",
        "git_hash": "abc1234",
        "inputs": {"db": "data/microstructure.db", "symbols": ["ETHUSDT"]},
        "per_symbol": {"ETHUSDT": {"n_total_buckets": 10}},
        "run_summary": {
            "version": "v1",
            "run_type": "fit_adverse_model",
            "inputs": {"db": "data/microstructure.db", "symbols": ["ETHUSDT"]},
            "metrics": {"symbol_count": 1, "error_count": 0},
            "artifacts": {"json": "reports/ADVERSE_MODEL.json", "md": "reports/ADVERSE_MODEL.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "fit_adverse_model"
    assert rsv.validate_payload(payload, "fit_adverse_model") == []


def test_validate_triage_capacity_payload() -> None:
    payload = {
        "inputs": {"candidates_md": "reports/candidates.md"},
        "gate_config": {"mitigation_profile": "baseline"},
        "rows": [{"symbol": "ETHUSDT", "horizon_sec": 120}],
        "run_summary": {
            "version": "v1",
            "run_type": "triage_capacity",
            "inputs": {"candidates_md": "reports/candidates.md", "db": "data/microstructure.db", "mitigation_profile": "baseline"},
            "metrics": {"candidate_count": 1, "row_count": 1},
            "artifacts": {"json": "reports/TRIAGE_CAPACITY.json"},
        },
    }
    assert rsv.infer_schema_name(payload) == "triage_capacity"
    assert rsv.validate_payload(payload, "triage_capacity") == []


def test_validate_rank_passive_pockets_forward_payload() -> None:
    payload = {
        "count": 1,
        "mitigation_profile": "baseline",
        "gate_config": {},
        "statistical": {},
        "decomposition": [],
        "ranking": [{"symbol": "ETHUSDT"}],
        "run_summary": {
            "version": "v1",
            "run_type": "rank_passive_pockets_forward",
            "inputs": {"candidates_md": "reports/candidates.md", "db": "data/microstructure.db", "rules": ["micro_edge_v3_passive_alpha"], "min_n_frac": 0.0},
            "metrics": {"count": 1, "candidate_count": 1, "survive_fee1_passrate_ge_0_5": 1},
            "artifacts": {"json": "reports/RANK.json", "md": "reports/RANK.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "rank_passive_pockets_forward"
    assert rsv.validate_payload(payload, "rank_passive_pockets_forward") == []


def test_validate_calibrate_capacity_thresholds_payload() -> None:
    payload = {
        "inputs": {"db": "data/microstructure.db"},
        "rows": [{"min_n_frac": 0.0005}],
        "run_summary": {
            "version": "v1",
            "run_type": "calibrate_capacity_thresholds",
            "inputs": {"candidates_md": "reports/candidates.md", "db": "data/microstructure.db", "min_n": 50, "grid": [0.0005]},
            "metrics": {"candidate_count": 1, "row_count": 1},
            "artifacts": {"json": "reports/CAPACITY_THRESHOLD_CALIBRATION.json", "md": "reports/CAPACITY_THRESHOLD_CALIBRATION.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "calibrate_capacity_thresholds"
    assert rsv.validate_payload(payload, "calibrate_capacity_thresholds") == []


def test_validate_evaluate_canary_expansion_gate_payload() -> None:
    payload = {
        "ts_utc": "2026-03-05T00:00:00Z",
        "passed": True,
        "gate": {"days_observed": 7},
        "run_summary": {
            "version": "v1",
            "run_type": "evaluate_canary_expansion_gate",
            "inputs": {"report_dir": "reports/daily", "window_days": 7, "max_top_score": 0.5},
            "metrics": {"passed": True, "days_observed": 7},
            "artifacts": {"json": "reports/CANARY_EXPANSION_GATE.json", "md": "reports/CANARY_EXPANSION_GATE.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "evaluate_canary_expansion_gate"
    assert rsv.validate_payload(payload, "evaluate_canary_expansion_gate") == []


def test_validate_run_full_sweep_payload() -> None:
    payload = {
        "generated_ts": 1234567890,
        "jobs": [{"id": "x", "rc": 0}],
        "run_summary": {
            "version": "v1",
            "run_type": "run_full_sweep",
            "inputs": {"candidates_md": "reports/candidates.md", "symbols": "ETHUSDT", "workers": 1},
            "metrics": {"job_count": 1, "success_count": 1},
            "artifacts": {"json": "runs/day60/manifest.json"},
        },
    }
    assert rsv.infer_schema_name(payload) == "run_full_sweep"
    assert rsv.validate_payload(payload, "run_full_sweep") == []


def test_validate_run_scratch_calibration_payload() -> None:
    payload = {
        "symbol": "ETHUSDT",
        "sell": {"rc": 0},
        "buy": {"rc": 0},
        "run_summary": {
            "version": "v1",
            "run_type": "run_scratch_calibration",
            "inputs": {"symbol": "ETHUSDT", "db": "data/microstructure.db", "exec_model": "passive_realistic"},
            "metrics": {"sell_rc": 0, "buy_rc": 0, "sell_n_final": 25, "buy_n_final": 25},
            "artifacts": {"json": "reports/SCRATCH_CALIBRATION_RUN_SUMMARY.json", "sell_md": "reports/SELL.md", "buy_md": "reports/BUY.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "run_scratch_calibration"
    assert rsv.validate_payload(payload, "run_scratch_calibration") == []


def test_validate_backtest_scratch_payload() -> None:
    payload = {
        "symbol": "ETHUSDT",
        "side": "SELL",
        "baseline": {"n": 5},
        "adverse_sweep": [],
        "trailing_sweep": [],
        "run_summary": {
            "version": "v1",
            "run_type": "backtest_scratch",
            "inputs": {"db": "data/microstructure.db", "symbol": "ETHUSDT", "side": "SELL", "regime": "UP"},
            "metrics": {"baseline_n": 5, "baseline_mean_net": 0.0001},
            "artifacts": {"json": "reports/SCRATCH_ANALYSIS.json", "md": "reports/SCRATCH_ANALYSIS.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "backtest_scratch"
    assert rsv.validate_payload(payload, "backtest_scratch") == []


def test_validate_compare_scratch_live_vs_backtest_payload() -> None:
    payload = {
        "status": "ok",
        "live": {"rows": 10},
        "backtest_sell": {"exists": True},
        "backtest_buy": {"exists": True},
        "run_summary": {
            "version": "v1",
            "run_type": "compare_scratch_live_vs_backtest",
            "inputs": {"trade_db": "data/paper_trades.db", "backtest_sell_json": "reports/sell.json", "backtest_buy_json": "reports/buy.json"},
            "metrics": {"live_rows": 10, "needs_recalibration_sell": False, "needs_recalibration_buy": False},
            "artifacts": {"json": "reports/SCRATCH_LIVE_VS_BACKTEST.json", "md": "reports/SCRATCH_LIVE_VS_BACKTEST.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "compare_scratch_live_vs_backtest"
    assert rsv.validate_payload(payload, "compare_scratch_live_vs_backtest") == []


def test_validate_db_introspect_payload() -> None:
    payload = {
        "db": "data/microstructure.db",
        "tables": [],
        "likely_core_tables": {},
        "run_summary": {
            "version": "v1",
            "run_type": "db_introspect",
            "inputs": {"db": "data/microstructure.db"},
            "metrics": {"table_count": 0},
            "artifacts": {"json": "reports/db_tables.json", "md": "reports/db_schema.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "db_introspect"
    assert rsv.validate_payload(payload, "db_introspect") == []


def test_validate_funding_rate_analysis_payload() -> None:
    payload = {
        "summary": {"trades": 2},
        "sample": [],
        "run_summary": {
            "version": "v1",
            "run_type": "funding_rate_analysis",
            "inputs": {"trades_db": "data/paper_trades.db", "micro_db": "data/microstructure.db", "symbol": "ETHUSDT"},
            "metrics": {"trades": 2, "total_funding_bps": 0.1},
            "artifacts": {"json": "reports/FUNDING_RATE_ANALYSIS.json", "md": "reports/FUNDING_RATE_ANALYSIS.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "funding_rate_analysis"
    assert rsv.validate_payload(payload, "funding_rate_analysis") == []


def test_validate_prototype_ws_vs_db_latency_payload() -> None:
    payload = {
        "status": "ok",
        "symbol": "ETHUSDT",
        "db_path": "data/microstructure.db",
        "collector_heartbeat_path": "logs/collector_heartbeat.json",
        "estimated_ws_bypass_gain_sec": 1.0,
        "run_summary": {
            "version": "v1",
            "run_type": "prototype_ws_vs_db_latency",
            "inputs": {"db": "data/microstructure.db", "symbol": "ETHUSDT", "collector_heartbeat": "logs/collector_heartbeat.json"},
            "metrics": {"collector_connected": True, "db_lag_sec": 1.0},
            "artifacts": {"json": "reports/WS_VS_DB_LATENCY_PROTOTYPE.json", "md": "reports/WS_VS_DB_LATENCY_PROTOTYPE.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "prototype_ws_vs_db_latency"
    assert rsv.validate_payload(payload, "prototype_ws_vs_db_latency") == []


def test_validate_freeze_runtime_profile_payload() -> None:
    payload = {
        "ts_utc": "2026-03-06T00:00:00Z",
        "hash": "abc123",
        "profile": {"SCALPER_DRY_RUN": "1"},
        "run_summary": {
            "version": "v1",
            "run_type": "freeze_runtime_profile",
            "inputs": {"enforce": False, "write_lock": True},
            "metrics": {"profile_key_count": 1, "hash_changed": False},
            "artifacts": {"json": "reports/RUNTIME_PROFILE_LOCK.json", "md": "reports/RUNTIME_PROFILE_LOCK.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "freeze_runtime_profile"
    assert rsv.validate_payload(payload, "freeze_runtime_profile") == []


def test_validate_microstructure_contract_payload() -> None:
    payload = {
        "db": "data/microstructure.db",
        "symbols": ["ETHUSDT", "BTCUSDT"],
        "required_tables": ["agg_trades", "mark_prices", "liquidations"],
        "status": "warn",
        "table_contracts": {
            "agg_trades": {"present": True, "row_count": 1, "timestamp_column": "ts_ms", "symbol_column": "symbol", "required_columns_missing": [], "available_book_fields": []},
            "mark_prices": {"present": True, "row_count": 1, "timestamp_column": "ts_ms", "symbol_column": "symbol", "required_columns_missing": [], "available_book_fields": []},
            "liquidations": {"present": True, "row_count": 1, "timestamp_column": "ts_ms", "symbol_column": "symbol", "required_columns_missing": [], "available_book_fields": []},
        },
        "symbol_coverage": {
            "ETHUSDT": {"agg_trades": True, "mark_prices": True, "liquidations": True},
            "BTCUSDT": {"agg_trades": True, "mark_prices": True, "liquidations": False},
        },
        "feature_capability": {
            "tier": "trade_plus_liq_mark_proxy",
            "mark_only": True,
            "trade_flow": True,
            "trade_plus_liq": True,
            "requires_book": False,
            "book_source_table": "mark_prices",
            "reason": "true_top_of_book_missing_mark_prices_or_proxy_used",
        },
        "warnings": ["true_top_of_book_missing"],
        "failures": [],
        "run_summary": {
            "version": "v1",
            "run_type": "validate_microstructure_contract",
            "inputs": {"db": "data/microstructure.db", "symbols": ["ETHUSDT", "BTCUSDT"], "require_true_book": False},
            "metrics": {"status": "warn", "table_count": 3, "warning_count": 1, "failure_count": 0, "requires_book": False},
            "artifacts": {"json": "reports/MICROSTRUCTURE_CONTRACT.json", "md": "reports/MICROSTRUCTURE_CONTRACT.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "validate_microstructure_contract"
    assert rsv.validate_payload(payload, "validate_microstructure_contract") == []


def test_validate_generate_liq_reversal_candidates_payload() -> None:
    payload = {
        "rule": "high_liq_reversal_regime",
        "regime": "liq_reversal_research",
        "symbols": ["ETHUSDT"],
        "grid": {
            "horizons": [30, 60],
            "min_imbalances": [0.3, 0.5],
            "min_trade_intensities": [200.0, 400.0],
            "max_spreads": [0.00025],
        },
        "count": 8,
        "rows": [
            {
                "symbol": "ETHUSDT",
                "rule": "high_liq_reversal_regime",
                "regime": "liq_reversal_research",
                "horizon_sec": 30,
                "min_imbalance": 0.3,
                "min_trade_intensity": 200.0,
                "max_spread": 0.00025,
                "pass": "YES",
            }
        ],
        "run_summary": {
            "version": "v1",
            "run_type": "generate_liq_reversal_candidates",
            "inputs": {"rule": "high_liq_reversal_regime", "regime": "liq_reversal_research", "symbols": ["ETHUSDT"]},
            "metrics": {"count": 8},
            "artifacts": {"json": "reports/LIQ_REVERSAL_CANDIDATES.json", "md": "reports/LIQ_REVERSAL_CANDIDATES.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "generate_liq_reversal_candidates"
    assert rsv.validate_payload(payload, "generate_liq_reversal_candidates") == []


def test_validate_run_liq_reversal_e2e_payload() -> None:
    payload = {
        "symbol": "ETHUSDT",
        "rule": "high_liq_reversal_regime",
        "lookback_min": 1440,
        "bucket_sec": 5,
        "coverage_json": "reports/LIQ_REVERSAL_E2E_COVERAGE.json",
        "candidates_json": "reports/LIQ_REVERSAL_E2E_CANDIDATES.json",
        "rank_baseline_json": "reports/LIQ_REVERSAL_E2E_RANK_BASELINE.json",
        "rank_v5_json": "reports/LIQ_REVERSAL_E2E_RANK_V5.json",
        "summary": {
            "coverage": {"windows": 3, "max_rule_fire_count": 12, "max_rule_given_liq_rate": 0.5},
            "candidate_surface": {"count": 8},
            "rank_baseline": {"count": 0, "top": None},
            "rank_v5": {"count": 1, "top": {"symbol": "ETHUSDT"}},
            "decision": {"baseline_tradeable": False, "v5_tradeable": True, "next_step": "inspect_ranked_pockets"},
        },
        "run_summary": {
            "version": "v1",
            "run_type": "run_liq_reversal_e2e",
            "inputs": {"db": "data/microstructure.db", "symbol": "ETHUSDT"},
            "metrics": {"coverage_windows": 3, "candidate_count": 8, "baseline_rank_count": 0, "v5_rank_count": 1},
            "artifacts": {"json": "reports/LIQ_REVERSAL_E2E.json", "md": "reports/LIQ_REVERSAL_E2E.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "run_liq_reversal_e2e"
    assert rsv.validate_payload(payload, "run_liq_reversal_e2e") == []


def test_validate_liquidation_regime_tagger_payload() -> None:
    payload = {
        "symbol": "ETHUSDT",
        "rule": "high_liq_reversal_regime",
        "lookback_min": 1440,
        "bucket_sec": 5,
        "summary": {"rows_total": 10, "tagged_count": 2, "tagged_rate": 0.2},
        "tags": [{"ts_ms": 1, "tag": "high_liq_reversal", "rule_fired": True}],
        "run_summary": {
            "version": "v1",
            "run_type": "liquidation_regime_tagger",
            "inputs": {"db": "data/microstructure.db", "symbol": "ETHUSDT"},
            "metrics": {"rows_total": 10, "tagged_count": 2, "tagged_rate": 0.2},
            "artifacts": {"json": "reports/LIQUIDATION_REGIME_TAGGER.json", "md": "reports/LIQUIDATION_REGIME_TAGGER.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "liquidation_regime_tagger"
    assert rsv.validate_payload(payload, "liquidation_regime_tagger") == []


def test_validate_run_research_event_watchboard_cycle_payload() -> None:
    payload = {
        "watchboard_json": "reports/RESEARCH_EVENT_WATCHBOARD.json",
        "append_json": "reports/RESEARCH_EVENT_WATCHBOARD_SNAPSHOT_APPEND.json",
        "overlap_json": "reports/EVENT_LANE_OVERLAP.json",
        "consolidation_json": "reports/EVENT_LANE_CONSOLIDATION.json",
        "trend_json": "reports/RESEARCH_EVENT_WATCHBOARD_TREND_FROM_HISTORY.json",
        "brief_json": "reports/RESEARCH_EVENT_OPERATOR_BRIEF.json",
        "history_jsonl": "reports/RESEARCH_EVENT_WATCHBOARD_HISTORY.jsonl",
        "summary": {
            "top_lane": "liquidation",
            "top_action": "monitor_only",
            "history_rows": 3,
            "trend": "rising",
            "trimmed_rows": 1,
            "top_overlap_pair": "liquidation::spread_stress",
            "suppression_candidate_count": 1,
        },
        "run_summary": {
            "version": "v1",
            "run_type": "run_research_event_watchboard_cycle",
            "inputs": {"symbols": ["ETHUSDT", "BTCUSDT"], "lookback_min": 240},
            "metrics": {"top_lane": "liquidation", "history_rows": 3, "trend": "rising", "trimmed_rows": 1, "top_overlap_pair": "liquidation::spread_stress"},
            "artifacts": {
                "json": "reports/RESEARCH_EVENT_WATCHBOARD_CYCLE.json",
                "md": "reports/RESEARCH_EVENT_WATCHBOARD_CYCLE.md",
                "watchboard_json": "reports/RESEARCH_EVENT_WATCHBOARD.json",
                "append_json": "reports/RESEARCH_EVENT_WATCHBOARD_SNAPSHOT_APPEND.json",
                "overlap_json": "reports/EVENT_LANE_OVERLAP.json",
                "consolidation_json": "reports/EVENT_LANE_CONSOLIDATION.json",
                "trend_json": "reports/RESEARCH_EVENT_WATCHBOARD_TREND_FROM_HISTORY.json",
                "history_jsonl": "reports/RESEARCH_EVENT_WATCHBOARD_HISTORY.jsonl",
            },
        },
    }
    assert rsv.infer_schema_name(payload) == "run_research_event_watchboard_cycle"
    assert rsv.validate_payload(payload, "run_research_event_watchboard_cycle") == []


def test_validate_event_lane_overlap_payload() -> None:
    payload = {
        "history_jsonl": "reports/RESEARCH_EVENT_WATCHBOARD_HISTORY.jsonl",
        "summary": {
            "available_rows": 3,
            "used_rows": 3,
            "lane_count": 4,
            "active_lane_count": 3,
            "active_snapshot_count": 3,
            "min_level": "elevated",
            "top_overlap_pair": "spread_stress::liquidation",
        },
        "lane_stats": [
            {
                "lane": "liquidation",
                "active_count": 2,
                "active_rate": 0.6667,
                "fresh_active_count": 1,
                "top_count": 1,
            }
        ],
        "pairwise": [
            {
                "lane_a": "spread_stress",
                "lane_b": "liquidation",
                "coactive_count": 2,
                "coactive_rate": 0.6667,
                "jaccard": 1.0,
            }
        ],
        "strongest_overlaps": [
            {
                "lane_a": "spread_stress",
                "lane_b": "liquidation",
                "coactive_count": 2,
                "coactive_rate": 0.6667,
                "jaccard": 1.0,
            }
        ],
        "redundancy_notes": [
            "spread_stress and liquidation co-activate frequently (jaccard=1.00, coactive_count=2)."
        ],
        "run_summary": {
            "version": "v1",
            "run_type": "event_lane_overlap",
            "inputs": {"history_jsonl": "reports/RESEARCH_EVENT_WATCHBOARD_HISTORY.jsonl", "min_level": "elevated", "top_n": 5},
            "metrics": {"available_rows": 3, "lane_count": 4, "active_snapshot_count": 3, "top_overlap_pair": "spread_stress::liquidation"},
            "artifacts": {"json": "reports/EVENT_LANE_OVERLAP.json", "md": "reports/EVENT_LANE_OVERLAP.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "event_lane_overlap"
    assert rsv.validate_payload(payload, "event_lane_overlap") == []


def test_validate_event_lane_consolidation_payload() -> None:
    payload = {
        "watchboard_json": "reports/RESEARCH_EVENT_WATCHBOARD.json",
        "overlap_json": "reports/EVENT_LANE_OVERLAP.json",
        "summary": {
            "top_lane": "spread_stress",
            "top_overlap_pair": "spread_stress::volume_vacuum",
            "decision_count": 2,
            "recommendation_counts": {"candidate_suppress_secondary": 1, "keep_separate": 1},
        },
        "decisions": [
            {
                "lane_a": "spread_stress",
                "lane_b": "volume_vacuum",
                "jaccard": 0.9,
                "coactive_count": 4,
                "secondary_lane": "volume_vacuum",
                "recommendation": "candidate_suppress_secondary",
                "reason": "spread_stress and volume_vacuum move almost together; consider suppressing volume_vacuum when the stronger companion lane is already active.",
            }
        ],
        "run_summary": {
            "version": "v1",
            "run_type": "event_lane_consolidation",
            "inputs": {"watchboard_json": "reports/RESEARCH_EVENT_WATCHBOARD.json", "overlap_json": "reports/EVENT_LANE_OVERLAP.json", "top_n": 5},
            "metrics": {"decision_count": 2, "candidate_suppress_secondary_count": 1, "review_for_merge_count": 0},
            "artifacts": {"json": "reports/EVENT_LANE_CONSOLIDATION.json", "md": "reports/EVENT_LANE_CONSOLIDATION.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "event_lane_consolidation"
    assert rsv.validate_payload(payload, "event_lane_consolidation") == []


def test_validate_event_lane_suppression_policy_payload() -> None:
    payload = {
        "watchboard_json": "reports/RESEARCH_EVENT_WATCHBOARD.json",
        "consolidation_json": "reports/EVENT_LANE_CONSOLIDATION.json",
        "summary": {"top_lane": "spread_stress", "rule_count": 2, "suppressed_lanes": ["volume_vacuum", "volatility_burst"]},
        "rules": [
            {
                "secondary_lane": "volume_vacuum",
                "when_lane_a": "spread_stress",
                "when_lane_b": "volume_vacuum",
                "display_mode": "degrade",
                "reason": "overlap",
                "secondary_level": "severe",
                "secondary_action": "show_caution",
            }
        ],
        "run_summary": {
            "version": "v1",
            "run_type": "event_lane_suppression_policy",
            "inputs": {"watchboard_json": "reports/RESEARCH_EVENT_WATCHBOARD.json", "consolidation_json": "reports/EVENT_LANE_CONSOLIDATION.json"},
            "metrics": {"rule_count": 2, "top_lane": "spread_stress"},
            "artifacts": {"json": "reports/EVENT_LANE_SUPPRESSION_POLICY.json", "md": "reports/EVENT_LANE_SUPPRESSION_POLICY.md"},
        },
    }
    assert rsv.infer_schema_name(payload) == "event_lane_suppression_policy"
    assert rsv.validate_payload(payload, "event_lane_suppression_policy") == []


def test_validate_research_event_operator_brief_payload() -> None:
    payload = {
        "watchboard_json": "reports/RESEARCH_EVENT_WATCHBOARD.json",
        "trend_json": "reports/RESEARCH_EVENT_WATCHBOARD_TREND_FROM_HISTORY.json",
        "overlap_json": "reports/EVENT_LANE_OVERLAP.json",
        "consolidation_json": "reports/EVENT_LANE_CONSOLIDATION.json",
        "summary": {
            "top_lane": "liquidation",
            "top_action": "monitor_only",
            "trend": "flat",
            "severe_lane_count": 1,
            "stale_lane_count": 2,
            "strongest_delta_lane": "return_shock",
            "strongest_delta_trend": "rising_fast",
            "strongest_overlap_pair": "liquidation::spread_stress",
            "suppression_candidate_count": 1,
            "primary_suppression_lane": "spread_stress",
        },
        "brief": {
            "headline": "Research events top=liquidation action=monitor_only trend=flat",
            "operator_note": "top lane liquidation, action monitor_only, trend flat. strongest lane delta: return_shock rising_fast (+125.00). strongest overlap: liquidation + spread_stress (jaccard=0.75, coactive_count=3). suppression candidate: spread_stress behind liquidation / spread_stress. severe lanes: liquidation. stale lanes: liquidation, spread_stress.",
            "top_event": {"lane": "liquidation", "action": "monitor_only", "headline": "top headline"},
            "strongest_delta": {"lane": "return_shock", "trend": "rising_fast", "delta_priority_score": 125.0},
            "strongest_overlap": {"lane_a": "liquidation", "lane_b": "spread_stress", "jaccard": 0.75, "coactive_count": 3},
            "primary_suppression": {"secondary_lane": "spread_stress", "lane_a": "liquidation", "lane_b": "spread_stress", "recommendation": "candidate_suppress_secondary"},
            "severe_lanes": ["liquidation"],
            "stale_lanes": ["liquidation", "spread_stress"],
        },
        "run_summary": {
            "version": "v1",
            "run_type": "research_event_operator_brief",
            "inputs": {
                "watchboard_json": "reports/RESEARCH_EVENT_WATCHBOARD.json",
                "trend_json": "reports/RESEARCH_EVENT_WATCHBOARD_TREND_FROM_HISTORY.json",
                "overlap_json": "reports/EVENT_LANE_OVERLAP.json",
                "consolidation_json": "reports/EVENT_LANE_CONSOLIDATION.json",
            },
            "metrics": {
                "top_lane": "liquidation",
                "top_action": "monitor_only",
                "trend": "flat",
                "severe_lane_count": 1,
                "stale_lane_count": 2,
                "strongest_delta_lane": "return_shock",
                "strongest_delta_trend": "rising_fast",
                "strongest_overlap_pair": "liquidation::spread_stress",
                "suppression_candidate_count": 1,
                "primary_suppression_lane": "spread_stress",
            },
            "artifacts": {
                "json": "reports/RESEARCH_EVENT_OPERATOR_BRIEF.json",
                "md": "reports/RESEARCH_EVENT_OPERATOR_BRIEF.md",
                "watchboard_json": "reports/RESEARCH_EVENT_WATCHBOARD.json",
                "trend_json": "reports/RESEARCH_EVENT_WATCHBOARD_TREND_FROM_HISTORY.json",
                "overlap_json": "reports/EVENT_LANE_OVERLAP.json",
                "consolidation_json": "reports/EVENT_LANE_CONSOLIDATION.json",
            },
        },
    }
    assert rsv.infer_schema_name(payload) == "research_event_operator_brief"
    assert rsv.validate_payload(payload, "research_event_operator_brief") == []


def test_rejects_bad_micro_edge_rule_shape() -> None:
    payload = {
        "ts_utc": "2026-03-05T00:00:00Z",
        "symbol": "ETHUSDT",
        "lookback_min": 120,
        "bucket_sec": 5,
        "horizon_sec": 30,
        "raw_rows": 10,
        "bucket_rows": 5,
        "label_counts": {"up": 1, "down": 1, "flat": 0},
        "baseline_hit_rate": 0.5,
        "min_rule_n": 10,
        "correlations": {},
        "naive_rules": {"bad_rule": {"hit_rate": "0.7", "delta_vs_baseline": 0.1}},
        "label_definition": {
            "timing": "signal at t, entry at t+1 mark, exit at t+1+h mark",
            "label": "sign((mark[t+1+h]/mark[t+1])-1) with threshold",
            "horizon_steps": 6,
            "threshold": 0.0002,
            "label_values": {"up": 1, "flat": 0, "down": -1},
            "hit_definition": "hit",
            "baseline_definition": "baseline",
        },
    }
    errors = rsv.validate_payload(payload, "micro_edge_smoke")
    assert "missing:naive_rules.bad_rule.n" in errors
    assert "bad_type:naive_rules.bad_rule.hit_rate" in errors
    assert "missing:run_summary" in errors or "bad_type:run_summary" in errors


def test_main_validates_jsonl_with_auto_schema(monkeypatch) -> None:
    record = build_json_record(
        rep={
            "symbol": "BTCUSDT",
            "raw_count": 20,
            "bucket_count": 6,
            "up": 1,
            "down": 2,
            "flat": 0,
            "baseline_hit_rate": 0.66,
            "feature_corr": {},
            "rules": {},
            "label_definition": {},
        },
        lookback_min=60,
        bucket_sec=3,
        horizon_sec=30,
        min_rule_n=5,
    )
    path = Path("localtests/test_report_schema_validator/micro_edge.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["x", "--in", str(path), "--schema", "auto"])
    assert rsv.main() == 0


def test_validate_canonical_requires_run_summary() -> None:
    payload = {
        "status": "pass",
        "run_id": "abc123",
        "source": "data/canonical/canonical_merged.parquet",
        "violations": [],
        "column_stats": {},
        "invariant_summary": {},
        "notes": [],
    }
    errors = rsv.validate_payload(payload, "validate_canonical")
    assert "missing:run_summary" in errors or "bad_type:run_summary" in errors
