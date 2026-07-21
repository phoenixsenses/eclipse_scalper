from tools.s34_v_engine_failure_anatomy import candle_features, fill_delay_bucket, trap_tags


def test_candle_features_classifies_hammer_reversal():
    candle = {"open": 100.0, "high": 101.5, "low": 98.0, "close": 101.0}

    out = candle_features(candle, ref_price=100.0)

    assert out["pattern"] == "hammer_reversal"
    assert out["lower_wick_bps"] == 200.0
    assert out["close_ret_bps"] == 100.0


def test_fill_delay_bucket_boundaries():
    assert fill_delay_bucket(None) == "no_fill_delay"
    assert fill_delay_bucket(30.0) == "fill_0_30s"
    assert fill_delay_bucket(120.0) == "fill_30_120s"
    assert fill_delay_bucket(600.0) == "fill_2_10m"
    assert fill_delay_bucket(601.0) == "fill_10_30m"


def test_trap_tags_prioritizes_rebreak_late_fill_and_btc_context():
    row = {
        "low_rebreak_15m": True,
        "low_rebreak_30m": True,
        "fill_delay_sec": 700.0,
        "first_15m_bucket": "ret15_dump",
        "candle5_pattern": "bear_followthrough",
        "btc_context_bucket": "btc_down_continues",
    }

    tags = trap_tags(row)

    assert "low_rebreak_15m" in tags
    assert "late_fill_gt10m" in tags
    assert "weak_first_15m" in tags
    assert "btc_down_continues" in tags
