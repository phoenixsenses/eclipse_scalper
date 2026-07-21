from types import SimpleNamespace

from tools.s34_v_engine_cancel_replace import (
    config_id,
    find_fill_between,
    parse_int_tuple,
    parse_replace_offsets,
    replace_label,
)


def test_parse_helpers():
    assert parse_int_tuple("30, 60,120") == (30, 60, 120)
    assert parse_replace_offsets("cancel,15, none, 5") == (None, 15.0, None, 5.0)


def test_config_id_and_replace_label():
    assert replace_label(None) == "CANCEL"
    assert replace_label(15.0) == "O15"
    assert config_id(initial_offset_bps=20.0, replace_offset_bps=15.0, wait_sec=60, cross_margin_bps=2.0) == "O20_W60_O15_C2"


def test_find_fill_between_respects_activation_and_cancel_window_for_long():
    event = SimpleNamespace(
        fade_direction="LONG",
        path=((1000, 100.0), (2000, 99.0), (3000, 98.0), (4000, 97.0)),
    )

    assert find_fill_between(event, limit_px=99.5, cross_margin_bps=0.0, start_ts_ms=1000, end_ts_ms=2500) == (2000, 99.5)
    assert find_fill_between(event, limit_px=98.5, cross_margin_bps=0.0, start_ts_ms=1000, end_ts_ms=2500) is None
    assert find_fill_between(event, limit_px=98.5, cross_margin_bps=0.0, start_ts_ms=2500, end_ts_ms=None) == (3000, 98.5)


def test_find_fill_between_applies_cross_margin_for_short():
    event = SimpleNamespace(
        fade_direction="SHORT",
        path=((1000, 100.0), (2000, 101.0), (3000, 102.0)),
    )

    assert find_fill_between(event, limit_px=101.0, cross_margin_bps=50.0, start_ts_ms=1000, end_ts_ms=None) == (3000, 101.0)
