from __future__ import annotations

from src.microphys.runtime.event_bus import EventBus
from src.microphys.runtime.supervisor import RuntimeSupervisor


def test_event_bus_idempotent_subscriber_skips_duplicate_event_ids() -> None:
    bus = EventBus(max_seen_per_handler=16)
    seen: list[str] = []

    def _h(topic: str, payload: dict) -> None:
        seen.append(str(payload.get("event_id")))

    bus.subscribe("fills", _h, handler_id="h1", idempotent=True)
    bus.publish("fills", {"event_id": "evt-1", "x": 1})
    bus.publish("fills", {"event_id": "evt-1", "x": 2})
    bus.publish("fills", {"event_id": "evt-2", "x": 3})
    assert seen == ["evt-1", "evt-2"]


def test_event_bus_non_idempotent_subscriber_receives_all() -> None:
    bus = EventBus()
    n = {"count": 0}

    def _h(topic: str, payload: dict) -> None:
        n["count"] += 1

    bus.subscribe("orders", _h, handler_id="h2", idempotent=False)
    bus.publish("orders", {"event_id": "evt-1"})
    bus.publish("orders", {"event_id": "evt-1"})
    assert n["count"] == 2


def test_runtime_supervisor_state_progression() -> None:
    sup = RuntimeSupervisor(max_feed_age_sec=5.0, max_order_age_sec=20.0, max_loop_errors=3)
    ok = sup.evaluate(now_ts=100.0, last_feed_ts=99.0, last_order_update_ts=95.0, loop_error_count=0)
    assert ok.status == "ok"
    degraded = sup.evaluate(now_ts=100.0, last_feed_ts=90.0, last_order_update_ts=99.0, loop_error_count=1)
    assert degraded.status == "degraded"
    failed = sup.evaluate(now_ts=100.0, last_feed_ts=80.0, last_order_update_ts=99.0, loop_error_count=3)
    assert failed.status == "failed"
    assert failed.should_halt is True

