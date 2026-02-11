from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set


class MachineKind(str, Enum):
    ORDER_INTENT = "order_intent"
    POSITION_BELIEF = "position_belief"


class OrderIntentState(str, Enum):
    INTENT_CREATED = "INTENT_CREATED"
    SUBMITTED = "SUBMITTED"
    ACKED = "ACKED"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    DONE = "DONE"
    SUBMITTED_UNKNOWN = "SUBMITTED_UNKNOWN"
    CANCEL_SENT_UNKNOWN = "CANCEL_SENT_UNKNOWN"
    REPLACE_RACE = "REPLACE_RACE"
    FILLED_AFTER_CANCEL = "FILLED_AFTER_CANCEL"


class PositionBeliefState(str, Enum):
    FLAT = "FLAT"
    OPEN_UNKNOWN = "OPEN_UNKNOWN"
    OPEN_CONFIRMED = "OPEN_CONFIRMED"
    CLOSING = "CLOSING"
    CLOSED_UNKNOWN = "CLOSED_UNKNOWN"


@dataclass(frozen=True)
class Transition:
    machine: MachineKind
    state_from: str
    state_to: str
    reason: str = ""


class TransitionError(ValueError):
    pass


_ORDER_TRANSITIONS: Dict[str, Set[str]] = {
    OrderIntentState.INTENT_CREATED.value: {
        OrderIntentState.SUBMITTED.value,
    },
    OrderIntentState.SUBMITTED.value: {
        OrderIntentState.ACKED.value,
        OrderIntentState.OPEN.value,
        OrderIntentState.PARTIAL.value,
        OrderIntentState.FILLED.value,
        OrderIntentState.SUBMITTED_UNKNOWN.value,
        OrderIntentState.CANCEL_SENT_UNKNOWN.value,
        OrderIntentState.REPLACE_RACE.value,
        OrderIntentState.DONE.value,
    },
    OrderIntentState.ACKED.value: {
        OrderIntentState.OPEN.value,
        OrderIntentState.PARTIAL.value,
        OrderIntentState.FILLED.value,
        OrderIntentState.SUBMITTED_UNKNOWN.value,
        OrderIntentState.DONE.value,
    },
    OrderIntentState.OPEN.value: {
        OrderIntentState.PARTIAL.value,
        OrderIntentState.FILLED.value,
        OrderIntentState.CANCEL_SENT_UNKNOWN.value,
        OrderIntentState.REPLACE_RACE.value,
        OrderIntentState.DONE.value,
    },
    OrderIntentState.PARTIAL.value: {
        OrderIntentState.FILLED.value,
        OrderIntentState.CANCEL_SENT_UNKNOWN.value,
        OrderIntentState.REPLACE_RACE.value,
        OrderIntentState.DONE.value,
    },
    OrderIntentState.FILLED.value: {
        OrderIntentState.DONE.value,
        OrderIntentState.FILLED_AFTER_CANCEL.value,
    },
    OrderIntentState.SUBMITTED_UNKNOWN.value: {
        OrderIntentState.ACKED.value,
        OrderIntentState.OPEN.value,
        OrderIntentState.PARTIAL.value,
        OrderIntentState.FILLED.value,
        OrderIntentState.CANCEL_SENT_UNKNOWN.value,
        OrderIntentState.REPLACE_RACE.value,
        OrderIntentState.DONE.value,
    },
    OrderIntentState.CANCEL_SENT_UNKNOWN.value: {
        OrderIntentState.SUBMITTED.value,
        OrderIntentState.OPEN.value,
        OrderIntentState.PARTIAL.value,
        OrderIntentState.FILLED.value,
        OrderIntentState.FILLED_AFTER_CANCEL.value,
        OrderIntentState.REPLACE_RACE.value,
        OrderIntentState.DONE.value,
    },
    OrderIntentState.REPLACE_RACE.value: {
        OrderIntentState.OPEN.value,
        OrderIntentState.PARTIAL.value,
        OrderIntentState.FILLED.value,
        OrderIntentState.CANCEL_SENT_UNKNOWN.value,
        OrderIntentState.DONE.value,
    },
    OrderIntentState.FILLED_AFTER_CANCEL.value: {
        OrderIntentState.DONE.value,
    },
    OrderIntentState.DONE.value: set(),
}


_POSITION_TRANSITIONS: Dict[str, Set[str]] = {
    PositionBeliefState.FLAT.value: {
        PositionBeliefState.OPEN_UNKNOWN.value,
        PositionBeliefState.OPEN_CONFIRMED.value,
        PositionBeliefState.CLOSED_UNKNOWN.value,
    },
    PositionBeliefState.OPEN_UNKNOWN.value: {
        PositionBeliefState.OPEN_CONFIRMED.value,
        PositionBeliefState.CLOSING.value,
        PositionBeliefState.CLOSED_UNKNOWN.value,
    },
    PositionBeliefState.OPEN_CONFIRMED.value: {
        PositionBeliefState.CLOSING.value,
        PositionBeliefState.CLOSED_UNKNOWN.value,
        PositionBeliefState.FLAT.value,
    },
    PositionBeliefState.CLOSING.value: {
        PositionBeliefState.FLAT.value,
        PositionBeliefState.CLOSED_UNKNOWN.value,
        PositionBeliefState.OPEN_CONFIRMED.value,
    },
    PositionBeliefState.CLOSED_UNKNOWN.value: {
        PositionBeliefState.FLAT.value,
        PositionBeliefState.OPEN_UNKNOWN.value,
        PositionBeliefState.OPEN_CONFIRMED.value,
    },
}


def _table(machine: MachineKind) -> Dict[str, Set[str]]:
    if machine == MachineKind.ORDER_INTENT:
        return _ORDER_TRANSITIONS
    if machine == MachineKind.POSITION_BELIEF:
        return _POSITION_TRANSITIONS
    return {}


def is_valid_transition(machine: MachineKind, state_from: str, state_to: str) -> bool:
    frm = str(state_from or "").strip().upper()
    to = str(state_to or "").strip().upper()
    if not frm or not to:
        return False
    return to in _table(machine).get(frm, set())


def transition(machine: MachineKind, state_from: str, state_to: str, reason: str = "") -> Transition:
    if not is_valid_transition(machine, state_from, state_to):
        raise TransitionError(
            f"invalid {machine.value} transition {state_from!s}->{state_to!s}"
            + (f" ({reason})" if reason else "")
        )
    return Transition(machine=machine, state_from=str(state_from).upper(), state_to=str(state_to).upper(), reason=str(reason or ""))


def map_unknown_order_state(current_state: str) -> str:
    cur = str(current_state or "").strip().upper()
    if cur in (
        OrderIntentState.SUBMITTED.value,
        OrderIntentState.ACKED.value,
        OrderIntentState.OPEN.value,
        OrderIntentState.PARTIAL.value,
        OrderIntentState.SUBMITTED_UNKNOWN.value,
    ):
        return OrderIntentState.SUBMITTED_UNKNOWN.value
    if cur in (OrderIntentState.CANCEL_SENT_UNKNOWN.value, OrderIntentState.REPLACE_RACE.value):
        return OrderIntentState.CANCEL_SENT_UNKNOWN.value
    return OrderIntentState.SUBMITTED_UNKNOWN.value
