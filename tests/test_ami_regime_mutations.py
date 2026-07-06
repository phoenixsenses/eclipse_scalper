"""Faz 6A-R mutation/adversarial testleri (14 senaryo) — sentetik, ana DB'siz."""
from __future__ import annotations
import sqlite3, sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.constitution import ConstitutionViolation
from ami.enums import Action, ClaimType, EvidenceLevel, KnowledgeStatus, Permission
from ami.governance import epistemic_gates as _gates
from ami.governance.governor import EpistemicGovernor
from ami.knowledge.objects import KnowledgeObject, Provenance
from ami.knowledge.store import KnowledgeStore
from ami.latent.dataset import assert_no_outcome, era_missing_drop
from ami.latent.discovery import align_labels, verify_artifact, assert_versions
from ami.latent.drift_monitor import DriftMonitor
from ami.latent.models import ari, seeded_kmeans
from ami.latent.regime import (RegimeDefiner, check_min_regime_sample,
                               check_regime_fit_boundary, classify_drift,
                               transition_matrix_within)
from ami.research.registry import (EvidenceBundle, ExperimentSpec, ResearchRegistry,
                                   assert_no_overlap)

FEATS = ["ret5m", "rv30m", "ofi10m", "stress10m", "buyliq10m",
         "fund_vel_1h", "spread5m", "trades10m", "ret1h"]


def synth_X(n=2000, seed=0):
    return np.random.RandomState(seed).normal(size=(n, len(FEATS)))


# 1 — outcome ile rejim tanimlama
def test_regime_from_outcome_blocked():
    with pytest.raises(ValueError, match="OUTCOME LEAKAGE"):
        assert_no_outcome(FEATS + ["fwd_ret_6h_regime"])
    rd = RegimeDefiner()
    with pytest.raises(ValueError, match="OUTCOME LEAKAGE"):
        rd.fit(synth_X(), FEATS + ["y_label"] , (0, 100))  # feat listesi kirli


# 2 — validation donemine gore rejim threshold secme
def test_regime_threshold_from_validation_blocked():
    rd = RegimeDefiner().fit(synth_X(), FEATS, fit_range=(0, 999))
    check_regime_fit_boundary(rd, val_start_ts=1000)          # ok
    bad = RegimeDefiner().fit(synth_X(), FEATS, fit_range=(0, 1500))
    with pytest.raises(ConstitutionViolation, match="validation era"):
        check_regime_fit_boundary(bad, val_start_ts=1000)


# 3 — fold overlap
def test_fold_overlap_blocked():
    with pytest.raises(ConstitutionViolation):
        assert_no_overlap({10, 11, 12}, {12, 13})


# 4 — event leakage (ayni event iki fold'da)
def test_event_leakage_blocked():
    tr = {("EV7", "w1")}; va = {("EV7", "w2")}
    with pytest.raises(ConstitutionViolation):
        assert_no_overlap({e for e, _ in tr}, {e for e, _ in va})


# 5 — data-quality drift'ini market rejimi sanma
def test_dq_drift_not_market():
    src, conf = classify_drift(psi_v=0.8, miss_a=0.02, miss_b=0.45)
    assert src.startswith("DATA_ISSUE") and conf == "HIGH"
    src2, _ = classify_drift(psi_v=0.8, miss_a=0.02, miss_b=0.03)
    assert src2 == "MARKET_SHIFT"


# 6 — missingness'i latent state sanma
def test_missingness_not_latent_state():
    miss = np.zeros((1000, 5), dtype=np.int8)
    miss[600:, 4] = 1     # validation erasinda tamamen eksik
    keep = era_missing_drop(miss, cut=600)
    assert 4 not in keep


# 7 — cok kucuk rejimde sahte stabilite
def test_tiny_regime_unknown():
    assert check_min_regime_sample(80, "trend=UP") == "UNKNOWN"
    assert check_min_regime_sample(5000, "trend=UP") == "trend=UP"


# 8 — label alignment hatasi (permutation deterministik hizalanir)
def test_label_alignment():
    ref = np.array([0, 0, 1, 1, 2, 2] * 50)
    perm = np.array([1, 2, 0])[ref]        # adlar permute
    aligned = align_labels(ref, perm, 3)
    assert (aligned == ref).mean() == 1.0
    assert ari(ref, perm) == pytest.approx(1.0)


# 9 — occupancy kriterini sonuctan sonra gevsetme
def test_occupancy_criteria_loosening_blocked(tmp_path):
    reg = ResearchRegistry(tmp_path / "r.sqlite", knowledge_path=tmp_path / "k.sqlite")
    spec = ExperimentSpec("E-R-1", "Q-R", population="grid", target="latent",
                          features=FEATS, threshold_method="occ band [0.3,3.0]",
                          chronological_split="wf", untouched_data="son fold",
                          negative_control="dq", decision_criteria="band [0.3,3.0]",
                          falsification_rule="REJECT",
                          execution_model="research_only_no_execution")
    spec.freeze()
    # BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1: direct gate receipt,
    # same pattern as ami/mutation_suite.py -- this test exercises the
    # freeze/attach_evidence immutability contract, not the gate itself.
    kconn = sqlite3.connect(str(reg.knowledge_path))
    _gates.init_gates_schema(kconn)
    _gates.issue_gate_receipt(kconn, experiment_id=spec.experiment_id,
                              canonical_family_id="REGIME_MUTATION_TEST_HARNESS",
                              split_version=None, nullifier=None, registry_result="TEST_HARNESS_DIRECT_RECEIPT")
    kconn.commit(); kconn.close()
    reg.register_experiment(spec)
    spec.decision_criteria = "band [0.05,20] (sonuca gore gevsetildi)"
    with pytest.raises(ConstitutionViolation):
        reg.attach_evidence(EvidenceBundle("EV-R1", "E-R-1", {}, "SUPPORTS",
                                           dataset_hash="d", code_ref="c"), spec)
    reg.close()


# 10 — transition matrix leakage (fold siniri asan gecis sayilmaz)
def test_transition_boundary_excluded():
    lab = np.array([0, 0, 0, 1, 1, 1])     # sinir index 3'te
    A_leak = transition_matrix_within(lab, 2, boundaries=[])
    A_ok = transition_matrix_within(lab, 2, boundaries=[3])
    assert A_leak[0, 1] > 0                 # sinirsiz: 0->1 gecisi var
    assert A_ok[0, 1] == 0                  # sinir dislandi: leakage yok


# 11 — drift alarmini kapatarak applicability koruma
def test_drift_alarm_cannot_be_silenced():
    mon = DriftMonitor()
    assert mon.recommendations("STABLE") == []
    for st in ("SHIFTED", "UNUSABLE"):
        recs = mon.recommendations(st)
        assert recs, f"{st} durumunda oneri listesi bos olamaz"
        assert any("applicability" in r for r in recs)


# 12 — rejim degismesine ragmen eski artifact kullanimi
def test_stale_artifact_under_shift_blocked():
    mon = DriftMonitor()
    X_ref = synth_X(1500, 1)
    X_cur = synth_X(1500, 2) + 5.0          # buyuk dagilim kaymasi
    rep = mon.assess(X_ref, X_cur, FEATS)
    assert rep["status"] in ("SHIFTED", "UNUSABLE")
    def use_artifact_guard(status):
        if status in ("SHIFTED", "UNUSABLE"):
            raise ConstitutionViolation(f"artifact not applicable under drift status {status}")
    with pytest.raises(ConstitutionViolation):
        use_artifact_guard(rep["status"])


# 13 — state artifact/version mismatch
def test_artifact_version_mismatch():
    art = {"schema_version": "latent_ds_v1", "feature_version": "2026-07-02", "k": 4,
           "centers": [[0.1]], "seed": 11}
    import hashlib, json as _json
    art["artifact_hash"] = hashlib.sha256(_json.dumps(art, sort_keys=True).encode()).hexdigest()[:16]
    assert verify_artifact(art)
    with pytest.raises(ValueError, match="VERSION MISMATCH"):
        assert_versions(art, {"schema_version": "latent_ds_v2", "feature_version": "2026-07-02"})


# 14 — research-only drift sonucuyla LIVE permission isteme
def test_drift_result_cannot_authorize_live(tmp_path):
    store = KnowledgeStore(tmp_path / "k.sqlite")
    gov = EpistemicGovernor(store)
    ko = KnowledgeObject(
        knowledge_id="K-DRIFT-X", claim="drift monitor findings",
        claim_type=ClaimType.META_RESEARCH, status=KnowledgeStatus.HOLDOUT_VALIDATED,
        provenance=Provenance(source_tables=["latent_dataset"], data_time_range="x..y",
                              code_ref="ami/latent/drift_monitor.py"),
        evidence_level=EvidenceLevel.UNTOUCHED_HOLDOUT, replications=1, holdouts=1,
        falsification=["n/a"],
        permitted=[Permission.RESEARCH_ONLY],
        forbidden=[Permission.LIVE_ALLOWED, Permission.SIZING_ALLOWED, Permission.PORTFOLIO_ALLOWED])
    store.put(ko)
    dec = gov.authorize(Action.OPEN_LONG, ["K-DRIFT-X"], {"data_health": "HEALTHY"})
    assert dec.result != "GRANTED"
    store.close()
