"""Archive job state machine (Phase 14). Frozen state names and legal
transitions -- partial outputs never appear verified, restart safely
discards an abandoned partial and rebuilds from the frozen source
identity, and a verified archive can never be silently overwritten.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PLANNED = "PLANNED"
EXPORTING_PARTIAL = "EXPORTING_PARTIAL"
EXPORTED_UNVERIFIED = "EXPORTED_UNVERIFIED"
VERIFYING = "VERIFYING"
VERIFIED_DISPOSABLE = "VERIFIED_DISPOSABLE"
FAILED = "FAILED"
ABANDONED_PARTIAL = "ABANDONED_PARTIAL"

JOB_STATES = frozenset({
    PLANNED, EXPORTING_PARTIAL, EXPORTED_UNVERIFIED, VERIFYING,
    VERIFIED_DISPOSABLE, FAILED, ABANDONED_PARTIAL,
})

# Legal forward transitions. VERIFIED_DISPOSABLE and FAILED are terminal
# except FAILED -> PLANNED (a fresh retry, new job identity in practice).
_LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    PLANNED: frozenset({EXPORTING_PARTIAL, FAILED}),
    EXPORTING_PARTIAL: frozenset({EXPORTED_UNVERIFIED, FAILED, ABANDONED_PARTIAL}),
    EXPORTED_UNVERIFIED: frozenset({VERIFYING, FAILED, ABANDONED_PARTIAL}),
    VERIFYING: frozenset({VERIFIED_DISPOSABLE, FAILED}),
    VERIFIED_DISPOSABLE: frozenset(),  # terminal, immutable
    FAILED: frozenset({PLANNED}),  # a fresh retry starts a new job identity
    ABANDONED_PARTIAL: frozenset({PLANNED}),
}


class IllegalJobTransitionError(Exception):
    """Raised for any transition not in `_LEGAL_TRANSITIONS` -- most
    importantly, nothing may transition INTO or OUT OF
    VERIFIED_DISPOSABLE except arriving from VERIFYING, and nothing may
    leave VERIFIED_DISPOSABLE at all (silent overwrite is structurally
    impossible)."""


@dataclass
class ArchiveJob:
    job_identity: str
    source_watermark_value: int
    state: str = PLANNED
    history: list[str] = field(default_factory=lambda: [PLANNED])

    def transition(self, new_state: str) -> None:
        if new_state not in JOB_STATES:
            raise IllegalJobTransitionError(f"unknown state {new_state!r}")
        if new_state not in _LEGAL_TRANSITIONS[self.state]:
            raise IllegalJobTransitionError(
                f"illegal transition {self.state!r} -> {new_state!r} "
                f"(legal: {sorted(_LEGAL_TRANSITIONS[self.state])})")
        self.state = new_state
        self.history.append(new_state)

    @property
    def is_terminal_verified(self) -> bool:
        return self.state == VERIFIED_DISPOSABLE

    @property
    def is_partial_and_unpublished(self) -> bool:
        return self.state in (EXPORTING_PARTIAL, EXPORTED_UNVERIFIED, VERIFYING, ABANDONED_PARTIAL)


def detect_abandoned_partial(job: ArchiveJob, *, process_still_running: bool) -> bool:
    """A job stuck in EXPORTING_PARTIAL/EXPORTED_UNVERIFIED/VERIFYING
    whose owning process is no longer running is abandoned and safe to
    discard-and-rebuild."""
    return job.is_partial_and_unpublished and not process_still_running


def restart_from_abandoned(job: ArchiveJob) -> ArchiveJob:
    """Discards an abandoned partial job and returns a fresh job at
    PLANNED for the same source identity -- resume is never attempted
    without a fully validated checkpoint (none is implemented in this
    batch, so restart always means a clean rebuild)."""
    if job.state not in (EXPORTING_PARTIAL, EXPORTED_UNVERIFIED, VERIFYING):
        raise IllegalJobTransitionError(f"cannot restart from state {job.state!r} directly")
    job.transition(ABANDONED_PARTIAL)
    job.transition(PLANNED)
    return job


def same_watermark_is_idempotent(prior_job: ArchiveJob, new_watermark: int) -> bool:
    """A repeated execution against the SAME frozen source watermark is
    idempotent (the same content would be produced); a DIFFERENT
    watermark means new data arrived and must be planned as a distinct
    job identity, never silently merged into the old one."""
    return prior_job.source_watermark_value == new_watermark
