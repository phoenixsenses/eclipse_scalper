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
# --------------------------------------------------------------------------
INTEGRATION_BOUNDARY_MS = 1_787_000_000_000
"""Only anchors at or after this instant are eligible for the bus.

A restart replays persisted state; without this, replayed history would be
published as if it were live. The local ledger is unaffected either way.
"""

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
    "integration_boundary_ms": INTEGRATION_BOUNDARY_MS, "subject": SUBJECT,
})


def as_mapping() -> MappingProxyType:
    """Read-only view, for logging or a registration descriptor."""
    return _FROZEN
