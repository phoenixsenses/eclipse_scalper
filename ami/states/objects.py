"""StateObject / StateBundle — Part IV §14-17."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any

from ami.enums import DataQuality, StateFamily


@dataclass
class StateObject:
    state_id: str
    entity: str                     # ETHUSDT / BTCUSDT / GLOBAL
    timeframe: str                  # 1m..1W or "event"
    family: StateFamily
    label: str
    start_ms: int
    confidence: float = 0.5
    probability: dict[str, float] = field(default_factory=dict)   # label -> p
    supporting: list[str] = field(default_factory=list)
    contradicting: list[str] = field(default_factory=list)
    expected_transitions: dict[str, float] = field(default_factory=dict)
    data_quality: DataQuality = DataQuality.HEALTHY
    knowledge_used: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def age_ms(self) -> int:
        return int(time.time() * 1000) - self.start_ms

    def to_dict(self) -> dict:
        return {"state_id": self.state_id, "entity": self.entity, "timeframe": self.timeframe,
                "family": self.family.value, "label": self.label, "start_ms": self.start_ms,
                "age_min": round(self.age_ms / 60_000, 1), "confidence": self.confidence,
                "probability": self.probability, "supporting": self.supporting,
                "contradicting": self.contradicting, "expected_transitions": self.expected_transitions,
                "data_quality": self.data_quality.value, "knowledge_used": self.knowledge_used,
                "meta": self.meta}


@dataclass
class StateBundle:
    """All states at one moment + timeframe conflict resolution (§17)."""
    ts_ms: int
    states: list[StateObject] = field(default_factory=list)

    def by_family(self, family: StateFamily) -> list[StateObject]:
        return [s for s in self.states if s.family == family]

    def by_timeframe(self, tf: str) -> list[StateObject]:
        return [s for s in self.states if s.timeframe == tf]

    def direction_labels(self) -> dict[str, str]:
        """timeframe -> UP/DOWN/FLAT from structure states."""
        out = {}
        for s in self.states:
            if s.family == StateFamily.STRUCTURE_STATE:
                out[s.timeframe] = s.meta.get("direction", "FLAT")
        return out

    def conflict_report(self) -> dict:
        dirs = self.direction_labels()
        vals = [v for v in dirs.values() if v != "FLAT"]
        if not vals:
            return {"alignment_score": 0.0, "conflict_score": 0.0, "dominant": "FLAT", "by_tf": dirs}
        up = vals.count("UP"); dn = vals.count("DOWN")
        align = max(up, dn) / len(vals)
        return {"alignment_score": round(align, 2), "conflict_score": round(1 - align, 2),
                "dominant": "UP" if up >= dn else "DOWN", "by_tf": dirs}

    def to_dict(self) -> dict:
        return {"ts_ms": self.ts_ms, "states": [s.to_dict() for s in self.states],
                "conflict": self.conflict_report()}
