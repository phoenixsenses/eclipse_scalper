from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

OrderState = Literal["NONE", "NEW", "ACKED", "PARTIAL", "FILLED", "CANCELED", "REJECTED"]


@dataclass(frozen=True)
class OrderSnapshot:
    order_id: str
    state: OrderState
    cumulative_qty: float
    remaining_qty: float
    last_ts_ms: int


class OrderStateError(ValueError):
    pass


class OrderFSM:
    def __init__(self, order_id: str) -> None:
        self.order_id = str(order_id)
        self.state: OrderState = "NONE"
        self.cumulative_qty: float = 0.0
        self.remaining_qty: float = 0.0
        self.last_ts_ms: int = 0

    def _touch_ts(self, ts_ms: int) -> None:
        t = int(ts_ms or 0)
        if t < self.last_ts_ms:
            raise OrderStateError(f"ts_not_monotonic:{t}<{self.last_ts_ms}")
        self.last_ts_ms = t

    def on_intent(self, *, qty: float, ts_ms: int) -> None:
        self._touch_ts(ts_ms)
        q = float(qty)
        if q <= 0:
            raise OrderStateError("intent_qty_non_positive")
        if self.state != "NONE":
            raise OrderStateError(f"intent_invalid_from:{self.state}")
        self.state = "NEW"
        self.cumulative_qty = 0.0
        self.remaining_qty = q

    def on_ack(self, *, ts_ms: int) -> None:
        self._touch_ts(ts_ms)
        if self.state not in {"NEW", "ACKED", "PARTIAL"}:
            raise OrderStateError(f"ack_invalid_from:{self.state}")
        self.state = "ACKED" if self.remaining_qty > 0 else "FILLED"

    def on_fill(self, *, fill_qty: float, cumulative_qty: float, remaining_qty: float, ts_ms: int) -> None:
        self._touch_ts(ts_ms)
        if self.state not in {"NEW", "ACKED", "PARTIAL"}:
            raise OrderStateError(f"fill_invalid_from:{self.state}")
        fq = float(fill_qty)
        cq = float(cumulative_qty)
        rq = float(remaining_qty)
        if fq <= 0:
            raise OrderStateError("fill_qty_non_positive")
        if cq < 0 or rq < 0:
            raise OrderStateError("fill_negative_state")
        if cq + 1e-12 < fq:
            raise OrderStateError("fill_cumulative_lt_fill")
        if cq + 1e-12 < self.cumulative_qty:
            raise OrderStateError("fill_cumulative_backwards")
        self.cumulative_qty = cq
        self.remaining_qty = rq
        self.state = "FILLED" if rq <= 0.0 else "PARTIAL"

    def on_cancel(self, *, ts_ms: int) -> None:
        self._touch_ts(ts_ms)
        if self.state in {"FILLED", "REJECTED", "CANCELED", "NONE"}:
            raise OrderStateError(f"cancel_invalid_from:{self.state}")
        self.state = "CANCELED"

    def on_reject(self, *, ts_ms: int) -> None:
        self._touch_ts(ts_ms)
        if self.state in {"FILLED", "REJECTED", "CANCELED"}:
            raise OrderStateError(f"reject_invalid_from:{self.state}")
        self.state = "REJECTED"

    def snapshot(self) -> OrderSnapshot:
        return OrderSnapshot(
            order_id=self.order_id,
            state=self.state,
            cumulative_qty=float(self.cumulative_qty),
            remaining_qty=float(self.remaining_qty),
            last_ts_ms=int(self.last_ts_ms),
        )

    @property
    def is_terminal(self) -> bool:
        return self.state in {"FILLED", "REJECTED", "CANCELED"}

