from __future__ import annotations

import pytest

from src.microphys.runtime.order_fsm import OrderFSM, OrderStateError


def test_order_fsm_happy_path_partial_to_filled() -> None:
    f = OrderFSM("ord-1")
    f.on_intent(qty=1.0, ts_ms=1000)
    f.on_ack(ts_ms=1010)
    f.on_fill(fill_qty=0.4, cumulative_qty=0.4, remaining_qty=0.6, ts_ms=1020)
    assert f.state == "PARTIAL"
    f.on_fill(fill_qty=0.6, cumulative_qty=1.0, remaining_qty=0.0, ts_ms=1030)
    assert f.state == "FILLED"
    assert f.is_terminal is True


def test_order_fsm_rejects_invalid_transition() -> None:
    f = OrderFSM("ord-2")
    with pytest.raises(OrderStateError):
        f.on_fill(fill_qty=1.0, cumulative_qty=1.0, remaining_qty=0.0, ts_ms=1000)


def test_order_fsm_ts_must_be_monotonic() -> None:
    f = OrderFSM("ord-3")
    f.on_intent(qty=1.0, ts_ms=1000)
    with pytest.raises(OrderStateError):
        f.on_ack(ts_ms=999)


def test_order_fsm_cancel_path() -> None:
    f = OrderFSM("ord-4")
    f.on_intent(qty=1.0, ts_ms=1000)
    f.on_ack(ts_ms=1005)
    f.on_cancel(ts_ms=1010)
    assert f.state == "CANCELED"
    with pytest.raises(OrderStateError):
        f.on_ack(ts_ms=1020)

