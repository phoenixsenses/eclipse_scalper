from pathlib import Path

import pytest

from tools.s34_feature_availability import (
    FeatureClass,
    FeatureValue,
    LookaheadViolation,
    assert_feature_available,
    assert_feature_set_available,
    build_registry_payload,
    signal_entry_features,
    write_registry,
)


def test_terminal_cluster_feature_is_illegal_at_entry():
    feature = FeatureValue(
        name="cluster_notional",
        value=500_000.0,
        knowable_at_ts=1_780_000_060_000,
        feature_class=FeatureClass.TERMINAL_CLUSTER,
    )

    with pytest.raises(LookaheadViolation):
        assert_feature_available(feature, 1_780_000_000_000, context="synthetic")


def test_forward_outcome_feature_is_illegal_at_entry_even_if_timestamp_matches():
    feature = FeatureValue(
        name="net_bps",
        value=52.0,
        knowable_at_ts=1_780_000_000_000,
        feature_class=FeatureClass.FORWARD_OUTCOME,
    )

    with pytest.raises(LookaheadViolation):
        assert_feature_available(feature, 1_780_000_000_000, context="synthetic")


def test_running_cluster_feature_is_legal_when_knowable_by_entry():
    feature = FeatureValue(
        name="running_notional",
        value=100_000.0,
        knowable_at_ts=1_780_000_000_000,
        feature_class=FeatureClass.RUNNING_CLUSTER,
    )

    assert_feature_available(feature, 1_780_000_000_000, context="synthetic")


def test_future_point_in_time_feature_is_illegal():
    feature = FeatureValue(
        name="btc_mark_return_bps",
        value=10.0,
        knowable_at_ts=1_780_000_001_000,
        feature_class=FeatureClass.POINT_IN_TIME,
    )

    with pytest.raises(LookaheadViolation):
        assert_feature_available(feature, 1_780_000_000_000, context="synthetic")


def test_runner_signal_features_are_available_at_entry():
    ts = 1_780_000_000_000
    signal = {
        "ts_ms": ts,
        "entry_ts_ms": ts,
        "liq_total_notional": 250_000.0,
        "liq_count": 2,
        "liq_max_notional": 150_000.0,
        "cluster_max_single_liq_share": 60.0,
        "day_trend_bps": 12.5,
    }

    assert_feature_set_available(signal_entry_features(signal), ts, context="runner")


def test_registry_payload_marks_terminal_and_forward_entry_features_as_violations():
    payload = build_registry_payload()
    rows = {row["feature_name"]: row for row in payload["rows"]}

    assert payload["violation_count"] > 0
    assert rows["cluster_notional"]["violation"] is True
    assert rows["net_bps"]["violation"] is True
    assert rows["liq_total_notional"]["violation"] is False


def test_write_registry_outputs_markdown_and_json(tmp_path: Path):
    out_md = tmp_path / "FEATURE_AVAILABILITY_REGISTRY.md"
    out_json = tmp_path / "FEATURE_AVAILABILITY_REGISTRY.json"

    payload = write_registry(out_md, out_json)

    assert payload["violation_count"] > 0
    assert "violation_count" in out_md.read_text(encoding="utf-8")
    assert '"violation_count"' in out_json.read_text(encoding="utf-8")
