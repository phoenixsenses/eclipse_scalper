from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional

EventHandler = Callable[[str, Any], None]


def _payload_event_key(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        for k in ("event_id", "id", "uid"):
            v = payload.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
    return None


@dataclass
class _Subscriber:
    handler: EventHandler
    handler_id: str
    idempotent: bool
    seen: Deque[str]
    seen_index: set[str]

    def mark_seen(self, key: str, max_seen: int) -> None:
        if key in self.seen_index:
            return
        self.seen.append(key)
        self.seen_index.add(key)
        while len(self.seen) > max_seen:
            old = self.seen.popleft()
            if old in self.seen_index:
                self.seen_index.remove(old)


class EventBus:
    """Simple typed pub/sub bus with optional idempotent delivery."""

    def __init__(self, *, max_seen_per_handler: int = 4096) -> None:
        self._subs: Dict[str, List[_Subscriber]] = defaultdict(list)
        self._max_seen = max(1, int(max_seen_per_handler))

    def subscribe(
        self,
        topic: str,
        handler: EventHandler,
        *,
        handler_id: Optional[str] = None,
        idempotent: bool = False,
    ) -> str:
        hid = str(handler_id or f"{topic}:{id(handler)}")
        for s in self._subs.get(topic, []):
            if s.handler_id == hid:
                return hid
        self._subs[topic].append(
            _Subscriber(
                handler=handler,
                handler_id=hid,
                idempotent=bool(idempotent),
                seen=deque(),
                seen_index=set(),
            )
        )
        return hid

    def unsubscribe(self, topic: str, handler_id: str) -> bool:
        rows = self._subs.get(topic, [])
        before = len(rows)
        self._subs[topic] = [s for s in rows if s.handler_id != str(handler_id)]
        return len(self._subs[topic]) < before

    def publish(self, topic: str, payload: Any) -> int:
        delivered = 0
        key = _payload_event_key(payload)
        for s in list(self._subs.get(topic, [])):
            if s.idempotent and key:
                if key in s.seen_index:
                    continue
                s.mark_seen(key, self._max_seen)
            s.handler(topic, payload)
            delivered += 1
        return delivered

    def subscriber_count(self, topic: str) -> int:
        return len(self._subs.get(topic, []))

