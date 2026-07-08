"""AMI tests — Appendix F minimum initial tests (knowledge + governance).

1. Knowledge provenance required
2. Invalid promotion rejected
3. Stale data blocks applicability
4. Contradiction lowers permission
5. Candidate version change resets forward evidence
6. Research-only knowledge cannot authorize live use
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.constitution import ConstitutionViolation
from ami.enums import (Action, ClaimType, EvidenceLevel, KnowledgeStatus, Permission)
from ami.governance.governor import EpistemicGovernor
from ami.knowledge.objects import KnowledgeObject, Provenance, now_ms
from ami.knowledge.store import KnowledgeStore


def make_ko(kid="K-T-1", status=KnowledgeStatus.PRELIMINARY, **kw) -> KnowledgeObject:
    return KnowledgeObject(
        knowledge_id=kid, claim="test claim", claim_type=ClaimType.PREDICTIVE,
        status=status,
        provenance=Provenance(source_tables=["liquidations"], data_time_range="2026-01..2026-06",
                              code_ref="tests"),
        falsification=["forward avg <= 0"], **kw)


@pytest.fixture()
def store(tmp_path):
    s = KnowledgeStore(tmp_path / "k.sqlite")
    yield s
    s.close()


def test_provenance_required(store):
    ko = make_ko()
    ko.provenance.code_ref = ""
    with pytest.raises(ConstitutionViolation):
        store.put(ko)


def test_mechanistic_claim_needs_falsification(store):
    ko = make_ko()
    ko.claim_type = ClaimType.MECHANISTIC
    ko.falsification = []
    with pytest.raises(ConstitutionViolation):
        store.put(ko)


def test_invalid_promotion_rejected(store):
    gov = EpistemicGovernor(store)
    ko = make_ko(status=KnowledgeStatus.PRELIMINARY)
    store.put(ko)
    # skip a rung
    with pytest.raises(ConstitutionViolation):
        gov.promote(ko, KnowledgeStatus.HOLDOUT_VALIDATED)
    # next rung but no replication evidence
    with pytest.raises(ConstitutionViolation):
        gov.promote(ko, KnowledgeStatus.REPLICATED)
    # with evidence it passes
    ko.evidence_level = EvidenceLevel.CHRONOLOGICAL
    ko.replications = 1
    gov.promote(ko, KnowledgeStatus.REPLICATED)
    assert ko.status == KnowledgeStatus.REPLICATED
    # holdout gate requires holdouts
    ko.evidence_level = EvidenceLevel.UNTOUCHED_HOLDOUT
    with pytest.raises(ConstitutionViolation):
        gov.promote(ko, KnowledgeStatus.HOLDOUT_VALIDATED)
    ko.holdouts = 1
    gov.promote(ko, KnowledgeStatus.HOLDOUT_VALIDATED)
    assert ko.status == KnowledgeStatus.HOLDOUT_VALIDATED


def test_stale_data_blocks_applicability(store):
    ko = make_ko()
    app, why = ko.is_applicable({"data_health": "STALE"})
    assert not app and "data_health" in why
    # stale knowledge itself also blocks
    ko.last_verified_ms = now_ms() - int(1000 * 86400 * 400)
    ko.decay_half_life_days = 30
    app, why = ko.is_applicable({"data_health": "HEALTHY"})
    assert not app and why == "stale_knowledge"


def test_contradiction_lowers_permission(store):
    a = make_ko("K-A", status=KnowledgeStatus.OPERATIONAL_CANDIDATE,
                permitted=[Permission.RESEARCH_ONLY, Permission.SHADOW_ALLOWED, Permission.LIVE_ALLOWED],
                forbidden=[])
    b = make_ko("K-B")
    store.put(a); store.put(b)
    ok, _ = a.is_permitted(Permission.LIVE_ALLOWED)
    assert ok
    store.link("K-A", "CONTRADICTS", "K-B")
    a2 = store.get("K-A")
    ok, why = a2.is_permitted(Permission.LIVE_ALLOWED)
    assert not ok and why == "unresolved_contradiction"


def test_version_change_resets_forward_evidence(store):
    ko = make_ko(status=KnowledgeStatus.FORWARD_VALIDATING)
    ko.forward_events = 37
    ko.freeze()
    assert ko.frozen and ko.freeze_hash
    ko.touch_version("threshold changed 4->3")
    assert ko.forward_events == 0
    assert not ko.frozen
    assert ko.version == 2
    assert ko.history and "threshold" in ko.history[-1]["reason"]


def test_research_only_cannot_authorize_live(store):
    gov = EpistemicGovernor(store)
    ko = make_ko(status=KnowledgeStatus.HOLDOUT_VALIDATED)
    store.put(ko)
    dec = gov.authorize(Action.OPEN_LONG, [ko.knowledge_id], {"data_health": "HEALTHY"})
    assert dec.result != "GRANTED"
    # even a fully permitted KO with forbidden LIVE stays blocked
    ko2 = make_ko("K-T-2", status=KnowledgeStatus.PROVISIONALLY_ACCEPTED,
                  permitted=[Permission.RESEARCH_ONLY, Permission.SHADOW_ALLOWED],
                  forbidden=[Permission.LIVE_ALLOWED])
    store.put(ko2)
    dec2 = gov.authorize(Action.OPEN_LONG, ["K-T-2"], {"data_health": "HEALTHY"})
    assert dec2.result != "GRANTED"


def test_no_silent_delete_and_audit(store):
    ko = make_ko()
    store.put(ko)
    with pytest.raises(ConstitutionViolation):
        store.delete(ko.knowledge_id)
    tail = store.audit_tail(5)
    assert tail and any(t[2] == "PUT" for t in tail)


def test_circuit_breaker_blocks_all(store):
    gov = EpistemicGovernor(store)
    ko = make_ko("K-L", status=KnowledgeStatus.OPERATIONAL_CANDIDATE,
                 permitted=[Permission.RESEARCH_ONLY, Permission.LIVE_ALLOWED], forbidden=[])
    store.put(ko)
    gov.check_data_health({"mark_prices": 999.0}, {"mark_prices": 5.0})
    dec = gov.authorize(Action.OPEN_LONG, ["K-L"], {"data_health": "HEALTHY"})
    assert dec.result == "DENIED"
    assert any("circuit_breaker" in r for r in dec.reasons)


def test_assumption_invalidation_cascade(store):
    gov = EpistemicGovernor(store)
    ko = make_ko("K-AS", status=KnowledgeStatus.OPERATIONAL_CANDIDATE,
                 permitted=[Permission.RESEARCH_ONLY, Permission.LIVE_ALLOWED], forbidden=[])
    ko.assumptions = ["BookTicker fill is representative of executable fill."]
    store.put(ko)
    affected = gov.invalidate_assumption("bookticker fill")
    assert "K-AS" in affected
    k2 = store.get("K-AS")
    assert k2.status == KnowledgeStatus.WEAKENED
    assert Permission.LIVE_ALLOWED in k2.forbidden
