from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

RuntimeHealth = Literal["ok", "degraded", "failed"]


@dataclass(frozen=True)
class RuntimeHealthSnapshot:
    status: RuntimeHealth
    feed_age_sec: float
    order_age_sec: float
    loop_error_count: int
    reasons: tuple[str, ...]

    @property
    def should_halt(self) -> bool:
        return self.status == "failed"


class RuntimeSupervisor:
    """Runtime guardrail evaluator for feed/order-loop liveness."""

    def __init__(
        self,
        *,
        max_feed_age_sec: float = 5.0,
        max_order_age_sec: float = 30.0,
        max_loop_errors: int = 3,
    ) -> None:
        self.max_feed_age_sec = max(0.1, float(max_feed_age_sec))
        self.max_order_age_sec = max(0.1, float(max_order_age_sec))
        self.max_loop_errors = max(0, int(max_loop_errors))

    def evaluate(
        self,
        *,
        now_ts: float,
        last_feed_ts: float,
        last_order_update_ts: float,
        loop_error_count: int = 0,
    ) -> RuntimeHealthSnapshot:
        now = float(now_ts)
        feed_age = max(0.0, now - float(last_feed_ts))
        order_age = max(0.0, now - float(last_order_update_ts))
        errs = max(0, int(loop_error_count))
        reasons: list[str] = []

        if feed_age > self.max_feed_age_sec:
            reasons.append("feed_stale")
        if order_age > self.max_order_age_sec:
            reasons.append("order_updates_stale")
        if errs > 0:
            reasons.append("loop_errors_present")

        status: RuntimeHealth = "ok"
        if errs >= self.max_loop_errors:
            status = "failed"
            if "loop_errors_present" not in reasons:
                reasons.append("loop_errors_present")
        elif feed_age > (self.max_feed_age_sec * 3.0):
            status = "failed"
        elif reasons:
            status = "degraded"

        return RuntimeHealthSnapshot(
            status=status,
            feed_age_sec=float(feed_age),
            order_age_sec=float(order_age),
            loop_error_count=int(errs),
            reasons=tuple(reasons),
        )

    def thresholds(self) -> Dict[str, float]:
        return {
            "max_feed_age_sec": float(self.max_feed_age_sec),
            "max_order_age_sec": float(self.max_order_age_sec),
            "max_loop_errors": float(self.max_loop_errors),
        }

