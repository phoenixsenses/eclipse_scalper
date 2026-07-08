"""DecisionTrace — Part XII §22 + Part XIV §81. Every recommendation is a packet.

decide() composes: StateBundle -> candidate action -> governor authorization ->
immutable trace (appended to data/ami/decisions.jsonl). It NEVER executes anything.
"""
from __future__ import annotations
import json, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ami.enums import Action, StateFamily
from ami.governance.governor import EpistemicGovernor, PermissionDecision
from ami.states.objects import StateBundle

TRACE_PATH = Path(__file__).resolve().parents[2] / "data" / "ami" / "decisions.jsonl"


@dataclass
class DecisionTrace:
    decision_id: str
    action: str
    result: str
    direction_candidates: dict[str, float]
    active_states: dict[str, str]
    support: list[str]
    counterevidence: list[str]
    knowledge_used: list[str]
    permission: dict
    uncertainty: str
    alternatives: list[str]
    context: dict[str, Any]
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def persist(self) -> None:
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.__dict__, default=str) + "\n")

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _direction_probs(bundle: StateBundle) -> dict[str, float]:
    rep = bundle.conflict_report()
    align = rep["alignment_score"]; dom = rep["dominant"]
    p_dom = 0.34 + 0.4 * align
    p_off = max(0.05, (1 - p_dom) * 0.4)
    p_no = max(0.0, 1 - p_dom - p_off)
    if dom == "UP":
        return {"LONG": round(p_dom, 2), "SHORT": round(p_off, 2), "NO_TRADE": round(p_no, 2)}
    if dom == "DOWN":
        return {"LONG": round(p_off, 2), "SHORT": round(p_dom, 2), "NO_TRADE": round(p_no, 2)}
    return {"LONG": 0.25, "SHORT": 0.25, "NO_TRADE": 0.5}


def decide(bundle: StateBundle, governor: EpistemicGovernor,
           knowledge_ids: list[str], context: dict[str, Any],
           proposed: Action | None = None) -> DecisionTrace:
    probs = _direction_probs(bundle)
    best_dir = max(probs, key=probs.get)
    if proposed is None:
        if probs[best_dir] < 0.55 or best_dir == "NO_TRADE":
            proposed = Action.WAIT
        else:
            proposed = Action.OPEN_LONG if best_dir == "LONG" else Action.OPEN_SHORT
    perm: PermissionDecision = governor.authorize(proposed, knowledge_ids, context)
    active = {}
    for s in bundle.states:
        if s.family in (StateFamily.STRUCTURE_STATE, StateFamily.CASCADE_STATE,
                        StateFamily.LEVERAGE_STATE, StateFamily.BOOK_STATE):
            active[f"{s.timeframe}:{s.family.value}"] = s.label
    support = [f"{s.timeframe} {s.label} ({s.meta.get('direction','')})"
               for s in bundle.by_family(StateFamily.STRUCTURE_STATE)
               if s.meta.get("direction") == ("UP" if best_dir == "LONG" else "DOWN")]
    counter = [f"{s.timeframe} {s.label} ({s.meta.get('direction','')})"
               for s in bundle.by_family(StateFamily.STRUCTURE_STATE)
               if s.meta.get("direction") not in ("FLAT", "UP" if best_dir == "LONG" else "DOWN")]
    conf = bundle.conflict_report()
    unc = "LOW" if conf["alignment_score"] > 0.75 else ("MODERATE" if conf["alignment_score"] > 0.5 else "HIGH")
    trace = DecisionTrace(
        decision_id=f"D:{bundle.ts_ms}",
        action=proposed.value, result=perm.result,
        direction_candidates=probs, active_states=active,
        support=support[:6], counterevidence=counter[:6],
        knowledge_used=knowledge_ids, permission=perm.to_dict(),
        uncertainty=unc,
        alternatives=[a for a in ("OPEN_LONG", "OPEN_SHORT", "WAIT", "NO_TRADE") if a != proposed.value],
        context=context)
    trace.persist()
    return trace
