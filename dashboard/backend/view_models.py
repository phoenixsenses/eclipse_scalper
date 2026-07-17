"""Typed view-model contract for the canonical S34 State Machine operator dashboard.

Every panel rendered by the canonical read-only operator UI is expressed as a
``PanelViewModel``. The UI never consumes raw adapter dicts directly; adapters
produce ``PanelViewModel`` instances (via :mod:`dashboard.backend.aggregator`)
and the server serializes them with :func:`PanelViewModel.to_dict`.

This module is intentionally dependency-free (stdlib only). It is part of the
canonical read-only dashboard and MUST NOT import the legacy, mutation-capable
``dashboard.backend.app`` / ``dashboard.backend.control_actions`` subsystem.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrustState(str, Enum):
    """Trust state of an artifact-backed panel value.

    Ordering (worst → best for surfacing): MISSING/MALFORMED are hard failures,
    STALE means data exists but is older than its freshness threshold, INFERRED
    means the value was derived/defaulted rather than directly observed, CURRENT
    means fresh and directly observed.
    """

    CURRENT = "current"
    STALE = "stale"
    MISSING = "missing"
    MALFORMED = "malformed"
    INFERRED = "inferred"


class ParseStatus(str, Enum):
    OK = "ok"
    MISSING = "missing"
    MALFORMED = "malformed"
    EMPTY = "empty"


class ProvenanceStatus(str, Enum):
    DIRECT = "direct"            # read straight from a runtime/source artifact
    DERIVED = "derived"          # computed from other observed values
    HISTORICAL = "historical"    # historical/research reference evidence
    FALLBACK = "fallback"        # a default used because the source was absent
    UNKNOWN = "unknown"


class Severity(str, Enum):
    OK = "ok"
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_SEVERITY_RANK = {
    Severity.OK: 0,
    Severity.INFO: 1,
    Severity.LOW: 2,
    Severity.MEDIUM: 3,
    Severity.HIGH: 4,
    Severity.CRITICAL: 5,
}

# Trust states that must NEVER be rendered as a healthy/GREEN panel without an
# explicit stale/degraded overlay. Enforced in tests and consulted by the UI.
NON_CURRENT_TRUST = frozenset(
    {TrustState.STALE, TrustState.MISSING, TrustState.MALFORMED, TrustState.INFERRED}
)


def severity_rank(sev: Severity) -> int:
    return _SEVERITY_RANK.get(sev, 0)


def max_severity(*sevs: Severity) -> Severity:
    best = Severity.OK
    for s in sevs:
        if severity_rank(s) > severity_rank(best):
            best = s
    return best


@dataclass
class PanelViewModel:
    """The single typed contract every panel adapter returns.

    ``trust_state != CURRENT`` obliges the UI to visually mute the panel and show
    an explicit stale/missing/malformed/inferred badge (never a bare GREEN).
    """

    key: str
    title: str
    status: str = "ACTIVE"
    severity: Severity = Severity.OK
    summary: str = ""
    fields: dict[str, Any] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)

    # Artifact trust metadata
    source_path: str | None = None
    source_timestamp: float | None = None      # epoch seconds of the source data
    read_timestamp: float | None = None        # epoch seconds when we read it
    age_seconds: float | None = None
    freshness_threshold_seconds: float | None = None
    parse_status: ParseStatus = ParseStatus.OK
    provenance_status: ProvenanceStatus = ProvenanceStatus.DIRECT
    trust_state: TrustState = TrustState.CURRENT

    reason_codes: list[str] = field(default_factory=list)
    raw_evidence: Any = None

    def is_current(self) -> bool:
        return self.trust_state == TrustState.CURRENT

    def add_reason(self, code: str) -> None:
        if code and code not in self.reason_codes:
            self.reason_codes.append(code)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "status": self.status,
            "severity": self.severity.value,
            "summary": self.summary,
            "fields": self.fields,
            "rows": self.rows,
            "source_path": self.source_path,
            "source_timestamp": self.source_timestamp,
            "read_timestamp": self.read_timestamp,
            "age_seconds": (round(self.age_seconds, 3) if isinstance(self.age_seconds, (int, float)) else None),
            "freshness_threshold_seconds": self.freshness_threshold_seconds,
            "parse_status": self.parse_status.value,
            "provenance_status": self.provenance_status.value,
            "trust_state": self.trust_state.value,
            "reason_codes": list(self.reason_codes),
            "raw_evidence": self.raw_evidence,
        }


def failed_panel(
    key: str,
    title: str,
    *,
    error_class: str,
    reason: str,
    severity: Severity = Severity.HIGH,
    source_path: str | None = None,
) -> PanelViewModel:
    """Typed fail-closed panel used by the aggregator when an adapter raises.

    The exception CLASS and a short reason are surfaced; the full exception /
    environment is never dumped (no secret leakage).
    """
    now = time.time()
    vm = PanelViewModel(
        key=key,
        title=title,
        status="ADAPTER_FAILED",
        severity=severity,
        summary=f"panel unavailable: {error_class}",
        source_path=source_path,
        read_timestamp=now,
        parse_status=ParseStatus.MALFORMED,
        provenance_status=ProvenanceStatus.UNKNOWN,
        trust_state=TrustState.MALFORMED,
    )
    vm.add_reason("ADAPTER_EXCEPTION")
    vm.add_reason(f"error_class:{error_class}")
    if reason:
        vm.add_reason(reason[:200])
    return vm
