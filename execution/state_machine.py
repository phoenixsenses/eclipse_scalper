from __future__ import annotations

from enum import Enum


class MachineKind(str, Enum):
    ORDER_INTENT = "order_intent"
    POSITION_BELIEF = "position_belief"


_ORDER_INTENT_TRANSITIONS: dict[str, set[str]] = {
    "INTENT_CREATED": {"SUBMITTED", "CANCEL_SENT", "DONE"},
    "SUBMITTED": {"PARTIALLY_FILLED", "FILLED", "CANCEL_SENT", "DONE", "SUBMITTED_UNKNOWN", "CANCEL_SENT_UNKNOWN", "REPLACE_RACE", "FILLED_AFTER_CANCEL"},
    "PARTIALLY_FILLED": {"FILLED", "CANCEL_SENT", "DONE", "SUBMITTED_UNKNOWN", "CANCEL_SENT_UNKNOWN", "REPLACE_RACE"},
    "FILLED": {"DONE"},
    "CANCEL_SENT": {"DONE", "SUBMITTED_UNKNOWN", "CANCEL_SENT_UNKNOWN", "FILLED_AFTER_CANCEL"},
    "CANCEL_SENT_UNKNOWN": {"SUBMITTED", "DONE", "REPLACE_RACE", "FILLED_AFTER_CANCEL"},
    "SUBMITTED_UNKNOWN": {"DONE", "SUBMITTED", "PARTIALLY_FILLED", "FILLED", "CANCEL_SENT", "CANCEL_SENT_UNKNOWN", "REPLACE_RACE"},
    "REPLACE_RACE": {"DONE"},
    "FILLED_AFTER_CANCEL": {"DONE"},
    "DONE": set(),
}

_POSITION_BELIEF_TRANSITIONS: dict[str, set[str]] = {
    "UNKNOWN": {"FLAT", "LONG", "SHORT"},
    "FLAT": {"LONG", "SHORT", "UNKNOWN"},
    "LONG": {"FLAT", "UNKNOWN"},
    "SHORT": {"FLAT", "UNKNOWN"},
}


def is_valid_transition(machine: MachineKind, state_from: str, state_to: str) -> bool:
    frm = str(state_from or "").strip().upper()
    to = str(state_to or "").strip().upper()
    if not frm or not to:
        return False
    if machine == MachineKind.ORDER_INTENT:
        return to in _ORDER_INTENT_TRANSITIONS.get(frm, set())
    if machine == MachineKind.POSITION_BELIEF:
        return to in _POSITION_BELIEF_TRANSITIONS.get(frm, set())
    return False


def transition(machine: MachineKind, state_from: str, state_to: str, reason: str = "") -> dict[str, str]:
    if not is_valid_transition(machine, state_from, state_to):
        raise ValueError(
            f"invalid_transition machine={machine} state_from={state_from} state_to={state_to} reason={reason}"
        )
    return {
        "machine": str(machine),
        "state_from": str(state_from),
        "state_to": str(state_to),
        "reason": str(reason or ""),
    }
