from types import SimpleNamespace

from tools.s34_v_engine_protective_stop import anchor_reclaimed, first_stop_ts


def test_first_stop_ts_detects_long_adverse_move_before_deadline():
    event = SimpleNamespace(
        fade_direction="LONG",
        path=((1000, 100.0), (2000, 99.5), (3000, 99.0), (4000, 98.0)),
    )

    assert first_stop_ts(event, fill_ts_ms=1000, entry_px=100.0, sl_bps=75.0, deadline_ms=3500) == 3000
    assert first_stop_ts(event, fill_ts_ms=1000, entry_px=100.0, sl_bps=250.0, deadline_ms=3500) is None


def test_first_stop_ts_detects_short_adverse_move():
    event = SimpleNamespace(
        fade_direction="SHORT",
        path=((1000, 100.0), (2000, 100.5), (3000, 101.0)),
    )

    assert first_stop_ts(event, fill_ts_ms=1000, entry_px=100.0, sl_bps=75.0, deadline_ms=4000) == 3000


def test_anchor_reclaimed_uses_post_fill_path_only():
    event = SimpleNamespace(anchor_mark_price=101.0, path=((1000, 102.0), (2000, 100.5), (3000, 101.1)))

    assert anchor_reclaimed(event, fill_ts_ms=1500, horizon_min=1)
    assert not anchor_reclaimed(event, fill_ts_ms=3000, horizon_min=1)
