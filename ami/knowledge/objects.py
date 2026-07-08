"""KnowledgeObject — Part XII §3 + §71 executable contract.

A claim is not AMI knowledge unless it carries provenance, scope, assumptions,
confidence decomposition, falsification conditions and permissions.
"""
from __future__ import annotations
import dataclasses, hashlib, json, time
from dataclasses import dataclass, field
from typing import Any

from ami.constitution import ConstitutionViolation
from ami.enums import (ClaimType, DataQuality, EvidenceLevel, KnowledgeStatus,
                       Permission, PERMISSION_MIN_STATUS, PROMOTION_LADDER)

DAY_MS = 86_400_000


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class Provenance:
    """Part XII §7 — every claim traceable to raw data + code."""
    source_tables: list[str]
    data_time_range: str            # e.g. "2026-02-15..2026-07-02"
    dataset_hash: str = ""          # hash of dataset descriptor
    code_ref: str = ""              # script path or commit
    experiment_id: str = ""
    feature_version: str = "v1"
    execution_model: str = "mark_fill_fee5bps"
    random_seed: int | None = 42

    def is_complete(self) -> bool:
        return bool(self.source_tables and self.data_time_range and self.code_ref)


@dataclass
class KnowledgeObject:
    knowledge_id: str
    claim: str
    claim_type: ClaimType
    status: KnowledgeStatus
    provenance: Provenance
    mechanism: str = ""
    direction: str = ""                       # LONG / SHORT / BOTH / NONE
    effect_size: dict[str, float] = field(default_factory=dict)
    evidence_level: EvidenceLevel = EvidenceLevel.ANECDOTAL
    required_evidence_level: EvidenceLevel = EvidenceLevel.FORWARD_SHADOW
    evidence_families: list[str] = field(default_factory=list)   # §6 dependency model
    replications: int = 0
    holdouts: int = 0
    forward_events: int = 0
    contradictions: list[str] = field(default_factory=list)
    scope: dict[str, Any] = field(default_factory=dict)          # symbols/regimes/sessions/timeframes
    assumptions: list[str] = field(default_factory=list)         # §8 registry
    confidence: dict[str, str] = field(default_factory=dict)     # §9 decomposed
    falsification: list[str] = field(default_factory=list)
    permitted: list[Permission] = field(default_factory=lambda: [Permission.RESEARCH_ONLY])
    forbidden: list[Permission] = field(default_factory=lambda: [Permission.LIVE_ALLOWED, Permission.SIZING_ALLOWED])
    parents: list[str] = field(default_factory=list)             # genealogy §19
    children: list[str] = field(default_factory=list)
    decay_half_life_days: float = 90.0                           # §12
    created_ms: int = field(default_factory=now_ms)
    last_verified_ms: int = field(default_factory=now_ms)
    version: int = 1
    owner: str = "ami"
    frozen: bool = False                                         # §74 freeze
    freeze_hash: str = ""
    history: list[dict] = field(default_factory=list)            # §33 never silently rewrite

    # ---- validation -------------------------------------------------------
    def validate(self) -> None:
        """Constitution: no claim without provenance / no theory without falsification."""
        if not self.claim.strip():
            raise ConstitutionViolation("No claim without provenance. (empty claim)")
        if not self.provenance.is_complete():
            raise ConstitutionViolation("No claim without provenance.")
        if self.claim_type in (ClaimType.MECHANISTIC, ClaimType.CAUSAL) and not self.falsification:
            raise ConstitutionViolation("No theory without falsifiable predictions.")
        if not self.falsification and self.status.value not in ("OBSERVATION", "OPEN_QUESTION"):
            raise ConstitutionViolation("No theory without falsifiable predictions.")
        overlap = set(self.permitted) & set(self.forbidden)
        if overlap:
            raise ConstitutionViolation(f"Permission conflict: {sorted(p.value for p in overlap)}")

    # ---- executable contract (§71) ---------------------------------------
    def is_fresh(self, at_ms: int | None = None) -> bool:
        at = at_ms or now_ms()
        age_days = (at - self.last_verified_ms) / DAY_MS
        return age_days <= 2.0 * self.decay_half_life_days

    def is_applicable(self, context: dict[str, Any]) -> tuple[bool, str]:
        """§72 contextual applicability. Context keys: symbol, session, regime,
        timeframe, data_health. Missing scope key = unrestricted."""
        dq = context.get("data_health")
        if dq is not None and DataQuality(dq) not in (DataQuality.HEALTHY, DataQuality.RECONSTRUCTED):
            return False, f"data_health={dq}"
        if not self.is_fresh(context.get("at_ms")):
            return False, "stale_knowledge"
        for key in ("symbol", "session", "regime", "timeframe", "venue"):
            allowed = self.scope.get(key + "s") or self.scope.get(key)
            if allowed and context.get(key) is not None:
                allowed_list = allowed if isinstance(allowed, list) else [allowed]
                if context[key] not in allowed_list:
                    return False, f"out_of_scope:{key}={context[key]}"
        return True, "ok"

    def is_permitted(self, permission: Permission) -> tuple[bool, str]:
        if permission in self.forbidden:
            return False, "explicitly_forbidden"
        min_status = PERMISSION_MIN_STATUS[permission]
        if self.status_rank() < PROMOTION_LADDER.index(min_status):
            return False, f"status_too_low:{self.status.value}<{min_status.value}"
        if permission not in self.permitted:
            return False, "not_granted"
        if self.contradictions and permission in (Permission.LIVE_ALLOWED, Permission.SIZING_ALLOWED):
            return False, "unresolved_contradiction"
        return True, "ok"

    def status_rank(self) -> int:
        try:
            return PROMOTION_LADDER.index(self.status)
        except ValueError:
            return -1  # demoted/terminal statuses rank below the ladder

    def required_evidence_gap(self) -> int:
        return max(0, int(self.required_evidence_level) - int(self.evidence_level))

    # ---- freeze & versioning (§74) ----------------------------------------
    def compute_freeze_hash(self) -> str:
        payload = json.dumps({"claim": self.claim, "effect": self.effect_size,
                              "scope": self.scope, "feature_version": self.provenance.feature_version,
                              "execution_model": self.provenance.execution_model},
                             sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def freeze(self) -> None:
        self.frozen = True
        self.freeze_hash = self.compute_freeze_hash()

    def touch_version(self, reason: str) -> None:
        """Any material change after freeze -> new version, forward evidence resets (§74)."""
        self.history.append({"version": self.version, "ts_ms": now_ms(), "reason": reason,
                             "status": self.status.value, "forward_events": self.forward_events})
        self.version += 1
        self.forward_events = 0
        self.frozen = False
        self.freeze_hash = ""

    # ---- serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["claim_type"] = self.claim_type.value
        d["status"] = self.status.value
        d["evidence_level"] = int(self.evidence_level)
        d["required_evidence_level"] = int(self.required_evidence_level)
        d["permitted"] = [p.value for p in self.permitted]
        d["forbidden"] = [p.value for p in self.forbidden]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeObject":
        d = dict(d)
        d["claim_type"] = ClaimType(d["claim_type"])
        d["status"] = KnowledgeStatus(d["status"])
        d["evidence_level"] = EvidenceLevel(d["evidence_level"])
        d["required_evidence_level"] = EvidenceLevel(d["required_evidence_level"])
        d["permitted"] = [Permission(p) for p in d["permitted"]]
        d["forbidden"] = [Permission(p) for p in d["forbidden"]]
        d["provenance"] = Provenance(**d["provenance"])
        return cls(**d)
