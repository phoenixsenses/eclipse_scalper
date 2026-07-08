"""BATCH-P2-002: known-at/available-at timestamp contract tests (Observatory §6).

Run: pytest tests/test_ami_timing_contract.py --basetemp <scratchpad> -p no:cacheprovider
"""
import pytest

from ami.timing.contract import (
    CandleState,
    DataQualityState,
    LookaheadViolation,
    TimestampedValue,
    enforce_known_at,
    known_at_ok,
    reject_if_partial_candle,
)


def test_known_at_ok_true_when_known_before_trigger():
    assert known_at_ok(known_at_ts=100, observer_trigger_ts=200) is True
    assert known_at_ok(known_at_ts=200, observer_trigger_ts=200) is True  # boundary: equal is OK


def test_known_at_ok_false_when_known_after_trigger():
    assert known_at_ok(known_at_ts=201, observer_trigger_ts=200) is False


def test_enforce_known_at_raises_future_information():
    with pytest.raises(LookaheadViolation, match="FUTURE_INFORMATION"):
        enforce_known_at(known_at_ts=500, observer_trigger_ts=100)


def test_enforce_known_at_does_not_raise_when_valid():
    enforce_known_at(known_at_ts=100, observer_trigger_ts=100)  # must not raise


def test_reject_partial_candle_raises():
    with pytest.raises(LookaheadViolation, match="PARTIAL_CANDLE"):
        reject_if_partial_candle(CandleState.PARTIAL_CANDLE)


def test_reject_partial_candle_allows_closed():
    reject_if_partial_candle(CandleState.CLOSED)  # must not raise


def test_timestamped_value_rejects_bad_ordering():
    with pytest.raises(ValueError, match="event_ts"):
        TimestampedValue(event_ts=200, available_at_ts=100, known_at_ts=300)
    with pytest.raises(ValueError, match="available_at_ts"):
        TimestampedValue(event_ts=100, available_at_ts=300, known_at_ts=200)


def test_timestamped_value_validate_for_use_lookahead():
    v = TimestampedValue(event_ts=100, available_at_ts=150, known_at_ts=200)
    with pytest.raises(LookaheadViolation, match="FUTURE_INFORMATION"):
        v.validate_for_use(observer_trigger_ts=199)
    v.validate_for_use(observer_trigger_ts=200)  # must not raise


def test_timestamped_value_validate_for_use_partial_candle():
    v = TimestampedValue(
        event_ts=100, available_at_ts=150, known_at_ts=150,
        candle_state=CandleState.PARTIAL_CANDLE,
    )
    with pytest.raises(LookaheadViolation, match="PARTIAL_CANDLE"):
        v.validate_for_use(observer_trigger_ts=1_000_000)  # far future trigger still rejects


def test_missing_quality_does_not_raise_by_itself():
    # §6.4: MISSING is an allowed, explicit state -- the contract module does
    # not force rejection; it only forbids silent zero-substitution upstream.
    v = TimestampedValue(
        event_ts=100, available_at_ts=150, known_at_ts=150,
        quality=DataQualityState.MISSING,
    )
    v.validate_for_use(observer_trigger_ts=200)  # must not raise
    assert v.quality == DataQualityState.MISSING
