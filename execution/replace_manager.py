from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

try:
    from execution import state_machine as _state_machine  # type: ignore
except Exception:
    _state_machine = None

STATE_INTENT_CREATED = "INTENT_CREATED"
STATE_CANCEL_SENT_UNKNOWN = "CANCEL_SENT_UNKNOWN"
STATE_REPLACE_RACE = "REPLACE_RACE"
STATE_FILLED_AFTER_CANCEL = "FILLED_AFTER_CANCEL"
STATE_DONE = "DONE"


@dataclass
class ReplaceOutcome:
    success: bool
    state: str
    attempts: int
    reason: str = ""
    cancel_ok: bool = False
    create_ok: bool = False
    last_status: str = ""
    order: Optional[dict[str, Any]] = None
    ambiguity_count: int = 0
    cancel_attempts: int = 0
    create_attempts: int = 0
    status_checks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": bool(self.success),
            "state": str(self.state),
            "attempts": int(self.attempts),
            "reason": str(self.reason or ""),
            "cancel_ok": bool(self.cancel_ok),
            "create_ok": bool(self.create_ok),
            "last_status": str(self.last_status or ""),
            "order_id": (str((self.order or {}).get("id")) if isinstance(self.order, dict) and (self.order or {}).get("id") else ""),
            "ambiguity_count": int(self.ambiguity_count),
            "cancel_attempts": int(self.cancel_attempts),
            "create_attempts": int(self.create_attempts),
            "status_checks": int(self.status_checks),
        }


async def run_cancel_replace(
    *,
    cancel_order_id: str,
    symbol: str,
    max_attempts: int,
    cancel_fn: Callable[[str, str], Awaitable[bool]],
    create_fn: Callable[[], Awaitable[Optional[dict[str, Any]]]],
    status_fn: Optional[Callable[[str, str], Awaitable[Optional[str]]]] = None,
    strict_transitions: bool = False,
    current_exposure_notional: float = 0.0,
    new_order_notional: float = 0.0,
    max_worst_case_notional: float = 0.0,
    max_ambiguity_attempts: int = 0,
) -> ReplaceOutcome:
    attempts = max(1, int(max_attempts or 1))
    state = STATE_INTENT_CREATED
    last_status = ""
    ambiguity_count = 0
    cancel_attempts = 0
    create_attempts = 0
    status_checks = 0

    if float(max_worst_case_notional or 0.0) > 0:
        worst_case = max(0.0, float(current_exposure_notional or 0.0)) + max(0.0, float(new_order_notional or 0.0))
        if worst_case > float(max_worst_case_notional):
            return ReplaceOutcome(
                success=False,
                state=STATE_INTENT_CREATED,
                attempts=0,
                reason="replace_envelope_block",
                cancel_ok=False,
                create_ok=False,
                last_status="",
                order=None,
                ambiguity_count=0,
                cancel_attempts=0,
                create_attempts=0,
                status_checks=0,
            )

    def _advance(next_state: str, reason: str) -> None:
        nonlocal state
        nxt = str(next_state or "").strip().upper()
        if not nxt or nxt == state:
            return
        if _state_machine is not None:
            try:
                _state_machine.transition(_state_machine.MachineKind.ORDER_INTENT, state, nxt, reason)
            except Exception:
                if strict_transitions:
                    raise
        state = nxt

    for idx in range(attempts):
        if idx == 0:
            _advance("SUBMITTED", "replace_begin")
        _advance("CANCEL_SENT_UNKNOWN", "cancel_attempt")
        cancel_ok = False
        try:
            cancel_attempts += 1
            cancel_ok = bool(await cancel_fn(cancel_order_id, symbol))
        except Exception:
            cancel_ok = False

        if not cancel_ok:
            status = ""
            if callable(status_fn):
                try:
                    status_checks += 1
                    status = str((await status_fn(cancel_order_id, symbol)) or "").strip().lower()
                except Exception:
                    status = ""
            last_status = status
            if status in ("closed", "filled", "canceled", "cancelled"):
                _advance(STATE_FILLED_AFTER_CANCEL if status in ("closed", "filled") else STATE_CANCEL_SENT_UNKNOWN, f"status:{status}")
                cancel_ok = True
                if status in ("closed", "filled"):
                    return ReplaceOutcome(
                        success=False,
                        state=STATE_FILLED_AFTER_CANCEL,
                        attempts=idx + 1,
                        reason="filled_after_cancel",
                        cancel_ok=True,
                        create_ok=False,
                        last_status=last_status,
                        order=None,
                        ambiguity_count=int(ambiguity_count),
                        cancel_attempts=int(cancel_attempts),
                        create_attempts=int(create_attempts),
                        status_checks=int(status_checks),
                    )
            else:
                _advance(STATE_CANCEL_SENT_UNKNOWN, "cancel_unknown")
                ambiguity_count += 1

        if not cancel_ok:
            if int(max_ambiguity_attempts or 0) > 0 and ambiguity_count >= int(max_ambiguity_attempts):
                _advance(STATE_REPLACE_RACE, "replace_ambiguity_cap")
                return ReplaceOutcome(
                    success=False,
                    state=STATE_REPLACE_RACE,
                    attempts=idx + 1,
                    reason="replace_ambiguity_cap",
                    cancel_ok=False,
                    create_ok=False,
                    last_status=last_status,
                    order=None,
                    ambiguity_count=int(ambiguity_count),
                    cancel_attempts=int(cancel_attempts),
                    create_attempts=int(create_attempts),
                    status_checks=int(status_checks),
                )
            continue

        try:
            _advance("SUBMITTED", "replace_submit")
            create_attempts += 1
            order = await create_fn()
        except Exception:
            order = None

        if isinstance(order, dict) and order:
            _advance("DONE", "replace_success")
            return ReplaceOutcome(
                success=True,
                state=STATE_DONE,
                attempts=idx + 1,
                reason="replace_success",
                cancel_ok=True,
                create_ok=True,
                last_status=last_status,
                order=order,
                ambiguity_count=int(ambiguity_count),
                cancel_attempts=int(cancel_attempts),
                create_attempts=int(create_attempts),
                status_checks=int(status_checks),
            )

        _advance(STATE_REPLACE_RACE, "replace_create_failed")

    reason = "replace_reconcile_required" if (ambiguity_count > 0 and create_attempts == 0) else "replace_giveup"
    if reason == "replace_reconcile_required":
        _advance(STATE_REPLACE_RACE, "replace_reconcile_required")
    return ReplaceOutcome(
        success=False,
        state=state,
        attempts=attempts,
        reason=reason,
        cancel_ok=False,
        create_ok=False,
        last_status=last_status,
        order=None,
        ambiguity_count=int(ambiguity_count),
        cancel_attempts=int(cancel_attempts),
        create_attempts=int(create_attempts),
        status_checks=int(status_checks),
    )
