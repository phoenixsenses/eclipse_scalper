from tools.s34_v_engine_adaptive_offset import parse_float_tuple, policies


def test_parse_float_tuple():
    assert parse_float_tuple("1,2, 5") == (1.0, 2.0, 5.0)


def test_vdepth_step_policy():
    p = policies()["vdepth_step_15_20_25"]

    assert p({"vdepth_bps": 31.9}) == 15.0
    assert p({"vdepth_bps": 34.0}) == 20.0
    assert p({"vdepth_bps": 36.0}) == 25.0


def test_missed_winner_rescue_policy_uses_dominance_or_accel():
    p = policies()["missed_winner_rescue"]

    assert p({"single_liq_dominance_pct": 60.0, "running_accel_usd_per_sec": 0.0}) == 10.0
    assert p({"single_liq_dominance_pct": 20.0, "running_accel_usd_per_sec": 6000.0}) == 10.0
    assert p({"single_liq_dominance_pct": 20.0, "running_accel_usd_per_sec": 0.0}) == 20.0
