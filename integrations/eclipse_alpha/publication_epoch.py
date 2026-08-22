"""The per-process publication epoch — the no-backfill boundary (review B1).

Phase 03B originally gated backfill on a static constant. Two things were wrong
with that. The constant resolved to an instant *before* Phase 03B existed, so it
gated nothing real; and a fixed constant cannot survive a restart — a process
that comes back up would happily republish anchors it had already seen, because
the boundary never moved.

The boundary is therefore captured **at publisher startup, per process**. On
restart the module is fresh, `start()` runs again, and everything older than the
new epoch is history and unpublishable through the live path.

**Nothing here persists.** That is deliberate and is what makes the guarantee
hold: there is no stored epoch to reload, reset, or disagree with. The cost is
that an epoch cannot outlive its process, which is precisely the intent.
"""

from __future__ import annotations

import time

_epoch_ms: int | None = None


class EpochAlreadyStarted(RuntimeError):
    """The epoch is established once, at startup, and never moved.

    Re-establishing it would let a caller slide the no-backfill boundary, which
    is the whole thing this module exists to prevent. A second call is a
    programming error, not a recoverable condition.
    """


def start(epoch_ms: int | None = None) -> int:
    """Establish this process's publication epoch and return it.

    Called once, when the Alpha publisher starts. With no argument the epoch is
    read from the wall clock — never from a constant, a file, or anything that
    could outlive the process.
    """
    global _epoch_ms
    value = int(epoch_ms) if epoch_ms is not None else int(time.time() * 1000)
    if _epoch_ms is not None:
        raise EpochAlreadyStarted(
            f"publication epoch is already {_epoch_ms}; refusing to move it to {value}"
        )
    _epoch_ms = value
    return _epoch_ms


def current() -> int | None:
    """The epoch, or None if the publisher has not started."""
    return _epoch_ms


def reset_for_tests() -> None:
    """Simulate a process boundary. Named so it cannot be mistaken for API.

    Production code must never call this: it is the one operation that could
    move the boundary, and it exists only so tests can model a restart.
    """
    global _epoch_ms
    _epoch_ms = None
