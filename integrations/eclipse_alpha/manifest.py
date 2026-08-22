"""Immutable integration manifest for the Eclipse Alpha Agent (E-DER V1).

Producer-owned metadata that lives **outside** the frozen research definition.
Nothing here is derived from a Git SHA, a branch, a filename, or Master Center
state — the identity of an arm must not change because someone committed.

`ARM_VERSION` is not invented: it is the runner's own frozen protocol constant,
re-exported so there is exactly one authoritative source.
"""

from __future__ import annotations

from types import MappingProxyType

# --------------------------------------------------------------------------
# Arm identity
# --------------------------------------------------------------------------
ARM = "E-DER-V1"

ARM_VERSION = "E_DER_V1_PROSPECTIVE_FORWARD_2026_08_21"
"""The runner's own authoritative constant, `PROTOCOL` in
`tools/e_der_v1_forward_shadow.py`.

It is restated here rather than imported. Importing the runner would drag its
entire runtime dependency tree — and its import-time side effects — into every
consumer of this manifest, including tests. The two are held equal by
`test_arm_version_is_the_runners_own_frozen_protocol_constant`, which reads the
runner's source with `ast` instead of importing it. If they ever diverge, that
test fails; the constant cannot drift silently."""

DIRECTION = "long"
"""Declared, not inferred.

The DETECTED event carries no direction field. E-DER V1 is a rebound after
downward forced-selling pressure, so the direction is a property of the frozen
arm rather than of any individual event. Asserting it here keeps the adapter
from guessing per event.
"""

# --------------------------------------------------------------------------
# P2 — no backfill through the live path
#
# Review B1: this used to be a static constant, which resolved to an instant
# before Phase 03B existed and could not survive a restart. The boundary now
# lives in `publication_epoch`, captured per process at publisher startup. The
# constant is deliberately gone rather than deprecated — a dead constant that
# once looked like the gate is an invitation to re-use it.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Producer identity (review B2)
#
# Shape is not provenance. A dict with the right `event` and `status` was being
# labelled E-DER V1, including one whose classification was RETROSPECTIVE and
# whose paper_only was False. These are the immutable traits of the frozen V1
# paper-shadow producer, and all of them must match.
# --------------------------------------------------------------------------
PRODUCER_IDENTITY = MappingProxyType({
    "protocol": ARM_VERSION,
    "classification": "PROSPECTIVE_FORWARD",
    "paper_only": True,
    "real_order_sent": False,
})
"""Who produced it. Checked with `is`-like equality, never coerced."""

T0_DATA_QUALITY = "PENDING_EXACT_OPENS"
"""The only `data_quality_status` a T0 event carries.

A lifecycle marker rather than a producer trait: `mature()` moves it to
ENTRY_EXACT_OPEN / COMPLETE_EXACT_OPENS / ..._UNAVAILABLE. Anything else means
the event has moved on, so it is an eligibility failure, not a provenance one.
"""

MUTATION_MARKERS = frozenset({"updated_at_utc"})
"""Written only by `mature()`. Presence is evidence of mutation, so it is a hard
failure rather than something to filter out (review B4)."""

REQUIRED_T0_MARKERS = frozenset({
    "entry_open", "boundary_open", "gross_return_bps", "net_return_bps",
})
"""`make_event` emits all four, present and None. Absence means the object is
not the real T0 shape — refused rather than tolerated (review B4)."""

# --------------------------------------------------------------------------
# Adapter contract version
# --------------------------------------------------------------------------
INTEGRATION_CONTRACT = "ECLIPSE_ALPHA_V1_ADAPTER_1"
"""Versions the *mapping*, separately from the arm. Bump when the mapping
changes; the arm version must not move because the adapter did."""

# --------------------------------------------------------------------------
# The closed context whitelist
# --------------------------------------------------------------------------
CONTEXT_WHITELIST = frozenset({
    "q_parent", "q_echo", "prior_stress_count", "multiscale_vote_sum",
    "parent_id", "parent_ts_ms", "echo_id", "cascade_id", "product_cohort",
    "session_state", "universe_version", "protocol", "classification",
    "paper_only", "base_ms", "entry_ms", "boundary_ms", "integration_contract",
})
"""Every key here is known at T0. Additions require review."""

OUTCOME_FIELDS = frozenset({
    "gross_return_bps", "net_return_bps", "entry_open", "boundary_open",
})
"""Learned after T0. If any is populated, the event has been matured and is
sealed — the adapter refuses it rather than filtering it."""

ELIGIBLE_EVENT = "DETECTED"
ELIGIBLE_STATUS = "AWAITING_ENTRY"

SUBJECT = "eclipse.alpha.trade_candidate"

_FROZEN = MappingProxyType({
    "arm": ARM, "arm_version": ARM_VERSION, "direction": DIRECTION,
    "integration_contract": INTEGRATION_CONTRACT,
    "subject": SUBJECT,
})


def as_mapping() -> MappingProxyType:
    """Read-only view, for logging or a registration descriptor."""
    return _FROZEN
