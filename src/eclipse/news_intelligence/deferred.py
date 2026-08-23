"""Everything that must not start while the machine is busy.

The register is the point. A capability that is "not built yet" and one that is
"built and deliberately not running" look identical from the outside, and the
difference decides whether someone spends a week rebuilding it. Each entry says
what it would cost and what has to be true before it runs.

Calling any of these raises. That is deliberate: a deferred collector that
returns quietly is indistinguishable from a working collector on a quiet day,
and this repository has already paid for confusing an outage with silence.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import DeferredUntilPhase1Complete

MARKER = "DEFERRED_UNTIL_PHASE1_COMPLETE"


@dataclass(frozen=True, slots=True)
class DeferredCapability:
    key: str
    what: str
    why_deferred: str
    resource: str
    unblocks_when: str = "the current research phase releases the machine"

    def start(self, *args, **kwargs):
        raise DeferredUntilPhase1Complete(
            f"{MARKER}: {self.key} — {self.what}. Deferred because {self.why_deferred} "
            f"({self.resource}). Unblocks when {self.unblocks_when}."
        )


REGISTER: dict[str, DeferredCapability] = {
    c.key: c
    for c in (
        DeferredCapability(
            "live_collectors",
            "poll official, news and social sources continuously",
            "a live collector is a permanent network process with a queue and retries",
            "network, RAM, a writer process per source",
        ),
        DeferredCapability(
            "historical_backfill",
            "pull the historical archive of every registered source",
            "bulk download is bounded only by the archive size",
            "network, tens of GB of disk",
        ),
        DeferredCapability(
            "embedding_index",
            "vector index for novelty and clustering",
            "building and holding an index competes for exactly the RAM in use",
            "RAM, CPU, disk",
        ),
        DeferredCapability(
            "llm_batch_classification",
            "classify the backlog with a model",
            "batch inference is sustained CPU or a sustained external spend",
            "CPU or API budget",
        ),
        DeferredCapability(
            "market_reaction_measurement",
            "fill reaction requests from the market store",
            "measuring horizons for many events means scanning the large store",
            "disk I/O on the shared database",
        ),
        DeferredCapability(
            "cross_market_backtest",
            "test lead-lag between equities, rates and crypto",
            "a backtest over the full cross-section is the heaviest job in the system",
            "CPU, disk, hours",
        ),
        DeferredCapability(
            "continuous_research_jobs",
            "scheduled re-runs of the research families",
            "a scheduler that wakes up and works is precisely what must not exist yet",
            "CPU, disk",
        ),
    )
}


def start(key: str, *args, **kwargs):
    """Attempt to start a deferred capability. Always raises; that is the contract."""
    try:
        capability = REGISTER[key]
    except KeyError:
        raise DeferredUntilPhase1Complete(
            f"{key!r} is not a registered capability. If it is heavy, register it here first."
        ) from None
    return capability.start(*args, **kwargs)


def register_report() -> list[dict[str, str]]:
    return [
        {
            "key": c.key,
            "what": c.what,
            "resource": c.resource,
            "status": MARKER,
            "unblocks_when": c.unblocks_when,
        }
        for c in sorted(REGISTER.values(), key=lambda c: c.key)
    ]


__all__ = ["DeferredCapability", "REGISTER", "start", "register_report", "MARKER"]
