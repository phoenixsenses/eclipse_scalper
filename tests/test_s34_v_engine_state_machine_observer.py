from tools.s34_v_engine_state_machine_observer import bps_between, time_bucket


def test_bps_between_measures_long_wait_deterioration():
    assert bps_between(100.0, 101.0) == 100.0
    assert bps_between(100.0, 99.0) == -100.0
    assert bps_between(None, 101.0) is None
    assert bps_between(0.0, 101.0) is None


def test_time_bucket_boundaries():
    assert time_bucket(None) == "no_confirm"
    assert time_bucket(5 * 60) == "confirm_0_5m"
    assert time_bucket(15 * 60) == "confirm_5_15m"
    assert time_bucket(30 * 60) == "confirm_15_30m"
    assert time_bucket(60 * 60) == "confirm_30_60m"
    assert time_bucket(60 * 60 + 1) == "confirm_60m_plus"
