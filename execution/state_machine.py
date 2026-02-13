# execution/state_machine.py — SCALPER ETERNAL — POSITION STATE MACHINE — 2026 v1.0
# Manages position lifecycle state transitions.
#
# States:
#   FLAT -> ENTRY_PENDING -> OPEN -> EXIT_PENDING -> FLAT
#                              |
#                              +-> ORPHAN_ADOPTED -> OPEN
#                              +-> PHANTOM_DETECTED -> FLAT
#
# Design principles:
# - Explicit state transitions with validation
# - Audit trail for all transitions
# - Fail-safe: unknown states treated as FLAT for entries, OPEN for exits

from __future__ import annotations

import time
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum

from utils.logging import log_core


# Configuration
_STATE_MACHINE_ENABLED = os.getenv("STATE_MACHINE_ENABLED", "1").lower() in ("1", "true", "yes", "on")
_TRANSITION_HISTORY_MAX = int(os.getenv("STATE_MACHINE_HISTORY_MAX", "100"))


class PositionState(Enum):
    """Position lifecycle states."""
    FLAT = "FLAT"                       # No position
    ENTRY_PENDING = "ENTRY_PENDING"     # Entry order submitted
    ENTRY_PARTIAL = "ENTRY_PARTIAL"     # Entry partially filled
    OPEN = "OPEN"                       # Position open, fully filled
    EXIT_PENDING = "EXIT_PENDING"       # Exit order submitted
    EXIT_PARTIAL = "EXIT_PARTIAL"       # Exit partially filled
    ORPHAN_ADOPTED = "ORPHAN_ADOPTED"   # Adopted from exchange (not in brain)
    PHANTOM_DETECTED = "PHANTOM_DETECTED"  # In brain but not on exchange
    UNKNOWN = "UNKNOWN"                 # State unknown


class TransitionResult(Enum):
    """Result of state transition attempt."""
    SUCCESS = "SUCCESS"
    INVALID = "INVALID"           # Invalid transition
    BLOCKED = "BLOCKED"           # Transition blocked by condition
    ERROR = "ERROR"               # Error during transition


# Valid state transitions
_VALID_TRANSITIONS: Dict[PositionState, List[PositionState]] = {
    PositionState.FLAT: [
        PositionState.ENTRY_PENDING,
        PositionState.ORPHAN_ADOPTED,
    ],
    PositionState.ENTRY_PENDING: [
        PositionState.ENTRY_PARTIAL,
        PositionState.OPEN,
        PositionState.FLAT,  # Entry canceled/failed
    ],
    PositionState.ENTRY_PARTIAL: [
        PositionState.OPEN,
        PositionState.FLAT,  # Entry canceled with partial fill closed
    ],
    PositionState.OPEN: [
        PositionState.EXIT_PENDING,
        PositionState.EXIT_PARTIAL,
        PositionState.FLAT,  # Market exit / emergency flat
        PositionState.PHANTOM_DETECTED,
    ],
    PositionState.EXIT_PENDING: [
        PositionState.EXIT_PARTIAL,
        PositionState.FLAT,
        PositionState.OPEN,  # Exit canceled
    ],
    PositionState.EXIT_PARTIAL: [
        PositionState.FLAT,
        PositionState.OPEN,  # Exit canceled with partial still open
    ],
    PositionState.ORPHAN_ADOPTED: [
        PositionState.OPEN,
        PositionState.EXIT_PENDING,
        PositionState.FLAT,
    ],
    PositionState.PHANTOM_DETECTED: [
        PositionState.FLAT,
        PositionState.OPEN,  # Re-detected on exchange
    ],
    PositionState.UNKNOWN: [
        # Can transition to any state from unknown
        PositionState.FLAT,
        PositionState.OPEN,
        PositionState.ENTRY_PENDING,
        PositionState.EXIT_PENDING,
    ],
}


@dataclass
class StateTransition:
    """Record of a state transition."""
    timestamp: float
    symbol: str
    from_state: PositionState
    to_state: PositionState
    result: TransitionResult
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SymbolState:
    """State tracking for a symbol."""
    symbol: str
    current_state: PositionState = PositionState.FLAT
    last_transition_ts: float = 0.0
    entry_pending_since: float = 0.0
    exit_pending_since: float = 0.0
    transition_count: int = 0
    history: List[StateTransition] = field(default_factory=list)


class PositionStateMachine:
    """
    Manages position state transitions with validation.

    Usage:
        sm = PositionStateMachine.get(bot)

        # Check current state
        state = sm.get_state(symbol)

        # Attempt transition
        result = sm.transition(symbol, PositionState.ENTRY_PENDING, reason="Entry signal")

        # Check if transition is valid
        can_enter = sm.can_transition(symbol, PositionState.ENTRY_PENDING)
    """

    _instance: Optional['PositionStateMachine'] = None

    def __init__(self, bot):
        self.bot = bot
        self._states: Dict[str, SymbolState] = {}
        self._global_history: List[StateTransition] = []
        self._transition_hooks: List[Callable[[StateTransition], None]] = []

    @classmethod
    def get(cls, bot) -> 'PositionStateMachine':
        """Get or create state machine (singleton)."""
        if cls._instance is None:
            cls._instance = cls(bot)
        return cls._instance

    @classmethod
    def is_enabled(cls, bot) -> bool:
        """Check if state machine is enabled."""
        if not _STATE_MACHINE_ENABLED:
            return False
        try:
            cfg = getattr(bot, "cfg", None)
            if cfg is not None:
                v = getattr(cfg, "STATE_MACHINE_ENABLED", None)
                if v is not None:
                    if isinstance(v, bool):
                        return v
                    return str(v).lower() in ("1", "true", "yes", "on")
        except Exception:
            pass
        return _STATE_MACHINE_ENABLED

    def _get_symbol_state(self, symbol: str) -> SymbolState:
        """Get or create symbol state."""
        if symbol not in self._states:
            self._states[symbol] = SymbolState(symbol=symbol)
        return self._states[symbol]

    def get_state(self, symbol: str) -> PositionState:
        """Get current state for symbol."""
        return self._get_symbol_state(symbol).current_state

    def get_symbol_state(self, symbol: str) -> SymbolState:
        """Get full symbol state object."""
        return self._get_symbol_state(symbol)

    def can_transition(self, symbol: str, to_state: PositionState) -> bool:
        """Check if transition to target state is valid."""
        current = self.get_state(symbol)
        valid_targets = _VALID_TRANSITIONS.get(current, [])
        return to_state in valid_targets

    def transition(
        self,
        symbol: str,
        to_state: PositionState,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Tuple[TransitionResult, str]:
        """
        Attempt state transition.

        Args:
            symbol: Trading symbol
            to_state: Target state
            reason: Reason for transition
            metadata: Additional metadata to record
            force: Force transition even if invalid (use with caution)

        Returns:
            (result, message) tuple
        """
        now = time.time()
        sym_state = self._get_symbol_state(symbol)
        from_state = sym_state.current_state

        # Same state = no-op success
        if from_state == to_state:
            return (TransitionResult.SUCCESS, "Already in target state")

        # Validate transition
        if not force and not self.can_transition(symbol, to_state):
            transition = StateTransition(
                timestamp=now,
                symbol=symbol,
                from_state=from_state,
                to_state=to_state,
                result=TransitionResult.INVALID,
                reason=f"Invalid: {from_state.value} -> {to_state.value}",
                metadata=metadata or {},
            )
            self._record_transition(sym_state, transition)

            log_core.warning(
                f"[state_machine] INVALID transition {symbol}: "
                f"{from_state.value} -> {to_state.value} | reason={reason}"
            )
            return (TransitionResult.INVALID, f"Cannot transition from {from_state.value} to {to_state.value}")

        # Execute transition
        try:
            sym_state.current_state = to_state
            sym_state.last_transition_ts = now
            sym_state.transition_count += 1

            # Track pending timestamps
            if to_state == PositionState.ENTRY_PENDING:
                sym_state.entry_pending_since = now
            elif to_state == PositionState.EXIT_PENDING:
                sym_state.exit_pending_since = now
            elif to_state in (PositionState.FLAT, PositionState.OPEN):
                sym_state.entry_pending_since = 0.0
                sym_state.exit_pending_since = 0.0

            transition = StateTransition(
                timestamp=now,
                symbol=symbol,
                from_state=from_state,
                to_state=to_state,
                result=TransitionResult.SUCCESS,
                reason=reason,
                metadata=metadata or {},
            )
            self._record_transition(sym_state, transition)

            log_core.info(
                f"[state_machine] {symbol}: {from_state.value} -> {to_state.value} | {reason}"
            )

            # Call hooks
            for hook in self._transition_hooks:
                try:
                    hook(transition)
                except Exception as e:
                    log_core.error(f"[state_machine] hook error: {e}")

            return (TransitionResult.SUCCESS, f"Transitioned to {to_state.value}")

        except Exception as e:
            log_core.error(f"[state_machine] transition error {symbol}: {e}")
            return (TransitionResult.ERROR, str(e))

    def _record_transition(self, sym_state: SymbolState, transition: StateTransition) -> None:
        """Record transition in history."""
        sym_state.history.append(transition)
        self._global_history.append(transition)

        # Trim history
        max_history = _TRANSITION_HISTORY_MAX
        if len(sym_state.history) > max_history:
            sym_state.history = sym_state.history[-max_history:]
        if len(self._global_history) > max_history * 10:
            self._global_history = self._global_history[-(max_history * 10):]

    def set_state(self, symbol: str, state: PositionState, reason: str = "direct_set") -> None:
        """
        Directly set state without validation (for sync with exchange).

        Use with caution - prefer transition() for normal operations.
        """
        sym_state = self._get_symbol_state(symbol)
        from_state = sym_state.current_state

        if from_state != state:
            sym_state.current_state = state
            sym_state.last_transition_ts = time.time()

            transition = StateTransition(
                timestamp=time.time(),
                symbol=symbol,
                from_state=from_state,
                to_state=state,
                result=TransitionResult.SUCCESS,
                reason=f"DIRECT_SET: {reason}",
            )
            self._record_transition(sym_state, transition)

            log_core.debug(f"[state_machine] DIRECT_SET {symbol}: {from_state.value} -> {state.value}")

    def reset(self, symbol: str) -> None:
        """Reset symbol to FLAT state."""
        self.set_state(symbol, PositionState.FLAT, "reset")

    def register_hook(self, hook: Callable[[StateTransition], None]) -> None:
        """Register a transition hook (called on every successful transition)."""
        self._transition_hooks.append(hook)

    def get_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[StateTransition]:
        """Get transition history."""
        if symbol:
            sym_state = self._get_symbol_state(symbol)
            return sym_state.history[-limit:]
        return self._global_history[-limit:]

    def get_pending_entries(self) -> List[str]:
        """Get symbols with pending entries."""
        return [
            sym for sym, state in self._states.items()
            if state.current_state in (PositionState.ENTRY_PENDING, PositionState.ENTRY_PARTIAL)
        ]

    def get_pending_exits(self) -> List[str]:
        """Get symbols with pending exits."""
        return [
            sym for sym, state in self._states.items()
            if state.current_state in (PositionState.EXIT_PENDING, PositionState.EXIT_PARTIAL)
        ]

    def get_open_positions(self) -> List[str]:
        """Get symbols with open positions."""
        return [
            sym for sym, state in self._states.items()
            if state.current_state in (PositionState.OPEN, PositionState.ORPHAN_ADOPTED)
        ]

    def is_flat(self, symbol: str) -> bool:
        """Check if symbol has no position."""
        return self.get_state(symbol) == PositionState.FLAT

    def is_open(self, symbol: str) -> bool:
        """Check if symbol has an open position."""
        return self.get_state(symbol) in (
            PositionState.OPEN,
            PositionState.ORPHAN_ADOPTED,
            PositionState.EXIT_PENDING,
            PositionState.EXIT_PARTIAL,
        )

    def allows_entry(self, symbol: str) -> bool:
        """Check if entry is allowed for symbol."""
        state = self.get_state(symbol)
        # Only allow entry from FLAT state
        return state == PositionState.FLAT

    def allows_exit(self, symbol: str) -> bool:
        """Check if exit is allowed for symbol (always true if position exists)."""
        state = self.get_state(symbol)
        return state in (
            PositionState.OPEN,
            PositionState.ORPHAN_ADOPTED,
            PositionState.ENTRY_PARTIAL,  # Can exit partial fills
        )


# Module-level convenience functions
def get_position_state(bot, symbol: str) -> PositionState:
    """Get current position state for symbol."""
    if not PositionStateMachine.is_enabled(bot):
        return PositionState.UNKNOWN
    return PositionStateMachine.get(bot).get_state(symbol)


def transition_state(bot, symbol: str, to_state: PositionState, reason: str = "", **kwargs) -> Tuple[TransitionResult, str]:
    """Attempt state transition."""
    if not PositionStateMachine.is_enabled(bot):
        return (TransitionResult.SUCCESS, "State machine disabled")
    return PositionStateMachine.get(bot).transition(symbol, to_state, reason, **kwargs)


def allows_entry(bot, symbol: str) -> bool:
    """Check if entry is allowed for symbol."""
    if not PositionStateMachine.is_enabled(bot):
        return True  # Fail-open when disabled
    return PositionStateMachine.get(bot).allows_entry(symbol)


def allows_exit(bot, symbol: str) -> bool:
    """Check if exit is allowed for symbol."""
    # Always allow exits (fail-open)
    return True


def journal_transition(bot, symbol: str, from_state: str, to_state: str, reason: str = "") -> None:
    """
    Journal a state transition (compatibility function).

    This is called from other modules to record transitions.
    """
    if not PositionStateMachine.is_enabled(bot):
        return

    try:
        # Map string states to enum
        state_map = {s.value: s for s in PositionState}
        to_state_enum = state_map.get(to_state.upper(), PositionState.UNKNOWN)

        sm = PositionStateMachine.get(bot)
        sm.transition(symbol, to_state_enum, reason=reason, force=True)

    except Exception as e:
        log_core.error(f"[state_machine] journal_transition error: {e}")
