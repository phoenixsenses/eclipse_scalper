"""EpistemicGovernor — Part XII. The control plane above all other intelligence.

Powers (not suggestions):
- authorize(action, knowledge_ids, context) -> PermissionDecision (can DENY)
- promote(ko) / demote(ko) via coded gates only (§23-24)
- circuit breakers (§75): stale data, contradiction, calibration failure
- belief revision (§13) with history preservation
- assumption invalidation cascade (§8)
"""
from __future__ import annotations
import json, time
from dataclasses import dataclass, field
from typing import Any

from ami.constitution import ConstitutionViolation
from ami.enums import (Action, ACTION_REQUIRED_PERMISSION, BeliefRevision, DataQuality,
                       EvidenceLevel, KnowledgeStatus, Permission, PROMOTION_LADDER)
from ami.knowledge.objects import KnowledgeObject, now_ms
from ami.knowledge.store import KnowledgeStore

# §23 promotion gate requirements: target status -> coded criteria
PROMOTION_GATES: dict[KnowledgeStatus, dict[str, Any]] = {
    KnowledgeStatus.PRELIMINARY: {"min_evidence": EvidenceLevel.IN_SAMPLE},
    KnowledgeStatus.REPLICATED: {"min_evidence": EvidenceLevel.CHRONOLOGICAL, "min_replications": 1},
    KnowledgeStatus.HOLDOUT_VALIDATED: {"min_evidence": EvidenceLevel.UNTOUCHED_HOLDOUT, "min_holdouts": 1},
    KnowledgeStatus.FORWARD_VALIDATING: {"min_evidence": EvidenceLevel.UNTOUCHED_HOLDOUT, "frozen": True},
    KnowledgeStatus.OPERATIONAL_CANDIDATE: {"min_evidence": EvidenceLevel.FORWARD_SHADOW,
                                            "min_forward_events": 20, "no_contradictions": True},
    KnowledgeStatus.PROVISIONALLY_ACCEPTED: {"min_evidence": EvidenceLevel.CONTROLLED_PAPER,
                                             "min_forward_events": 40, "no_contradictions": True},
}


@dataclass
class PermissionDecision:
    action: str
    result: str                      # GRANTED / DENIED / SHADOW_ONLY / OBSERVE_ONLY
    reasons: list[str] = field(default_factory=list)
    knowledge_ids: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    ts_ms: int = field(default_factory=now_ms)

    def to_dict(self) -> dict:
        return {"action": self.action, "result": self.result, "reasons": self.reasons,
                "knowledge_ids": self.knowledge_ids, "context": self.context, "ts_ms": self.ts_ms}


class EpistemicGovernor:
    def __init__(self, store: KnowledgeStore):
        self.store = store
        self.circuit_open: dict[str, str] = {}    # breaker_name -> reason

    # ---- circuit breakers (§75) -------------------------------------------
    def trip_breaker(self, name: str, reason: str) -> None:
        self.circuit_open[name] = reason
        self.store._audit("governor", "BREAKER_TRIP", None, f"{name}: {reason}")
        self.store.conn.commit()

    def reset_breaker(self, name: str) -> None:
        self.circuit_open.pop(name, None)
        self.store._audit("governor", "BREAKER_RESET", None, name)
        self.store.conn.commit()

    def check_data_health(self, feed_ages_min: dict[str, float],
                          max_age_min: dict[str, float]) -> dict[str, DataQuality]:
        """Feed freshness -> DataQuality map; trips breaker on stale critical feeds."""
        out: dict[str, DataQuality] = {}
        for feed, age in feed_ages_min.items():
            limit = max_age_min.get(feed, 10.0)
            if age is None:
                out[feed] = DataQuality.UNAVAILABLE
            elif age <= limit:
                out[feed] = DataQuality.HEALTHY
            elif age <= 10 * limit:
                out[feed] = DataQuality.DEGRADED
            else:
                out[feed] = DataQuality.STALE
        stale = [f for f, q in out.items() if q in (DataQuality.STALE, DataQuality.UNAVAILABLE)]
        if stale:
            self.trip_breaker("data_health", f"stale feeds: {stale}")
        return out

    # ---- authorization (§2, §22) -------------------------------------------
    def authorize(self, action: Action, knowledge_ids: list[str],
                  context: dict[str, Any]) -> PermissionDecision:
        reasons: list[str] = []
        required = ACTION_REQUIRED_PERMISSION.get(action, Permission.RESEARCH_ONLY)
        # observation-class actions are always permitted
        if required == Permission.RESEARCH_ONLY:
            return PermissionDecision(action.value, "GRANTED", ["observation-class action"],
                                      knowledge_ids, context)
        if self.circuit_open:
            return PermissionDecision(action.value, "DENIED",
                                      [f"circuit_breaker:{k}={v}" for k, v in self.circuit_open.items()],
                                      knowledge_ids, context)
        if not knowledge_ids:
            return PermissionDecision(action.value, "DENIED",
                                      ["Every decision must be traceable to active evidence. (no knowledge cited)"],
                                      knowledge_ids, context)
        ok_all = True
        for kid in knowledge_ids:
            ko = self.store.get(kid)
            if ko is None:
                ok_all = False; reasons.append(f"{kid}: unknown knowledge"); continue
            app, why = ko.is_applicable(context)
            if not app:
                ok_all = False; reasons.append(f"{kid}: not_applicable ({why})"); continue
            perm, why = ko.is_permitted(required)
            if not perm:
                ok_all = False; reasons.append(f"{kid}: {required.value} denied ({why})")
        # mutually exclusive knowledge (§21)
        cited = set(knowledge_ids)
        for s, d in self.store.conflicting_pairs():
            if s in cited and d in cited:
                ok_all = False; reasons.append(f"CANNOT_COEXIST: {s} vs {d}")
        if ok_all:
            dec = PermissionDecision(action.value, "GRANTED", ["all gates passed"], knowledge_ids, context)
        else:
            # downgrade path: allow shadow observation if only status blocks it
            shadowable = all("not_applicable" not in r and "unknown" not in r for r in reasons)
            dec = PermissionDecision(action.value, "SHADOW_ONLY" if shadowable else "DENIED",
                                     reasons, knowledge_ids, context)
        self.store._audit("governor", "AUTHORIZE", None,
                          json.dumps({"action": action.value, "result": dec.result, "reasons": reasons[:6]}))
        self.store.conn.commit()
        return dec

    # ---- promotion / demotion (§23-24) --------------------------------------
    def promote(self, ko: KnowledgeObject, target: KnowledgeStatus, actor: str = "governor") -> KnowledgeObject:
        cur = ko.status_rank(); tgt = PROMOTION_LADDER.index(target)
        if tgt != cur + 1:
            raise ConstitutionViolation(f"No agent may override the evidence hierarchy. "
                                        f"(skip {ko.status.value} -> {target.value})")
        gate = PROMOTION_GATES.get(target, {})
        if int(ko.evidence_level) < int(gate.get("min_evidence", EvidenceLevel.ANECDOTAL)):
            raise ConstitutionViolation(f"No operational promotion without untouched validation. "
                                        f"(evidence {int(ko.evidence_level)} < {int(gate['min_evidence'])})")
        if ko.replications < gate.get("min_replications", 0):
            raise ConstitutionViolation("Promotion denied: replications insufficient.")
        if ko.holdouts < gate.get("min_holdouts", 0):
            raise ConstitutionViolation("No operational promotion without untouched validation. (no holdout)")
        if ko.forward_events < gate.get("min_forward_events", 0):
            raise ConstitutionViolation("Promotion denied: forward evidence insufficient.")
        if gate.get("no_contradictions") and ko.contradictions:
            raise ConstitutionViolation("No contradiction is ignored. (unresolved contradictions block promotion)")
        if gate.get("frozen") and not ko.frozen:
            raise ConstitutionViolation("Forward validation requires frozen candidate (§74).")
        ko.history.append({"version": ko.version, "ts_ms": now_ms(),
                           "reason": f"PROMOTE {ko.status.value}->{target.value}", "status": ko.status.value})
        ko.status = target
        ko.last_verified_ms = now_ms()
        self.store.put(ko, actor=actor)
        return ko

    def demote(self, ko: KnowledgeObject, target: KnowledgeStatus, trigger: str,
               actor: str = "governor") -> KnowledgeObject:
        ko.history.append({"version": ko.version, "ts_ms": now_ms(),
                           "reason": f"DEMOTE {ko.status.value}->{target.value}: {trigger}",
                           "status": ko.status.value})
        ko.status = target
        # demotion strips risky permissions automatically
        for p in (Permission.LIVE_ALLOWED, Permission.SIZING_ALLOWED, Permission.OPERATIONAL_CANDIDATE):
            if p in ko.permitted:
                ko.permitted.remove(p)
            if p not in ko.forbidden:
                ko.forbidden.append(p)
        self.store.put(ko, actor=actor)
        return ko

    # ---- belief revision (§13) ----------------------------------------------
    def revise(self, ko: KnowledgeObject, revision: BeliefRevision, detail: str,
               actor: str = "governor") -> KnowledgeObject:
        ko.history.append({"version": ko.version, "ts_ms": now_ms(),
                           "reason": f"{revision.value}: {detail}", "status": ko.status.value})
        if revision == BeliefRevision.REINFORCE:
            ko.last_verified_ms = now_ms()
        elif revision == BeliefRevision.RESTRICT:
            ko.status = KnowledgeStatus.REGIME_LIMITED if ko.status_rank() < 0 else ko.status
        elif revision == BeliefRevision.REVISE:
            ko.touch_version(detail)
        elif revision == BeliefRevision.REJECT:
            ko.status = KnowledgeStatus.REJECTED
            ko.permitted = [Permission.RESEARCH_ONLY]
            ko.forbidden = [p for p in Permission if p != Permission.RESEARCH_ONLY]
        self.store.put(ko, actor=actor)
        return ko

    # ---- assumption invalidation (§8) ---------------------------------------
    def invalidate_assumption(self, assumption_substr: str, actor: str = "governor") -> list[str]:
        """Suspend risky permissions on all knowledge depending on the assumption."""
        affected = []
        for ko in self.store.all():
            if any(assumption_substr.lower() in a.lower() for a in ko.assumptions):
                self.demote(ko, KnowledgeStatus.WEAKENED,
                            f"assumption invalidated: {assumption_substr}", actor=actor)
                affected.append(ko.knowledge_id)
        return affected
