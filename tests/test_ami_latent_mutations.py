"""Faz 6A mutation/adversarial testleri (15 senaryo) — sentetik veri, ana DB'siz."""
from __future__ import annotations
import hashlib, json, sqlite3, sys
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
from ami.latent.dataset import (assert_backward_looking, assert_no_outcome,
                                era_missing_drop)
from ami.latent.discovery import assert_versions, verify_artifact
from ami.latent.models import Standardizer, ari, seeded_kmeans
from ami.research.registry import (EvidenceBundle, ExperimentSpec, ResearchRegistry,
                                   assert_no_overlap)


# 1 — outcome leakage
def test_outcome_leakage_blocked():
    with pytest.raises(ValueError, match="OUTCOME LEAKAGE"):
        assert_no_outcome(["ret5m", "fwd_ret_6h"])
    with pytest.raises(ValueError, match="OUTCOME LEAKAGE"):
        assert_no_outcome(["rv30m", "mfe_50_label"])


# 2 — future feature timestamp
def test_future_feature_timestamp_blocked():
    assert_backward_looking(1_000_000, 999_999)      # gecmis veri: ok
    with pytest.raises(ValueError, match="FUTURE FEATURE"):
        assert_backward_looking(1_000_000, 1_000_001)


# 3 — normalization leakage (fit araligi validation'a tasarsa yakalanir)
def test_normalization_leakage_detectable():
    X = np.random.RandomState(0).normal(size=(100, 3))
    cut_ts = 500
    std = Standardizer().fit(X[:70], (0, 499))
    assert std.fit_range[1] < cut_ts                  # dogru kullanim
    bad = Standardizer().fit(X, (0, 999))             # validation'i da gormus
    assert bad.fit_range[1] >= cut_ts                 # denetim bunu yakalar
    # discovery sozlesmesi: fit_range[1] < validation baslangici olmali
    def guard(s, val_start):
        if s.fit_range[1] >= val_start:
            raise ConstitutionViolation("normalization fitted on validation era")
    guard(std, cut_ts)
    with pytest.raises(ConstitutionViolation):
        guard(bad, cut_ts)


# 4 — train/validation sequence overlap
def test_train_val_overlap_blocked():
    with pytest.raises(ConstitutionViolation):
        assert_no_overlap({1, 2, 3}, {3, 9})


# 5 — ayni event farkli pencerelerle iki split'te
def test_same_event_two_windows_blocked():
    train_windows = {("EV1", "pre"), ("EV2", "pre")}
    val_windows = {("EV1", "post")}     # ayni EV1, farkli pencere
    train_events = {e for e, _ in train_windows}
    val_events = {e for e, _ in val_windows}
    with pytest.raises(ConstitutionViolation):
        assert_no_overlap(train_events, val_events)


# 6 — missingness cluster'inin sahte state olusturmasi
def test_missingness_fake_state_blocked():
    miss = np.zeros((100, 4), dtype=np.int8)
    miss[:, 2] = 1                       # feature 2: her yerde eksik
    miss[80:, 3] = 1                     # feature 3: validation erasinda %100 eksik
    keep = era_missing_drop(miss, cut=80)
    assert 2 not in keep and 3 not in keep and 0 in keep and 1 in keep


# 7 — data-source identity leakage
def test_identity_leakage_blocked():
    with pytest.raises(ValueError, match="IDENTITY LEAKAGE"):
        assert_no_outcome(["ret5m", "venue_id"])
    with pytest.raises(ValueError, match="IDENTITY LEAKAGE"):
        assert_no_outcome(["symbol_code", "rv30m"])


# 8 — random-seed instability (yapisiz veri stabil sayilmaz)
def test_seed_instability_rejected():
    rng = np.random.RandomState(1)
    Z = rng.normal(size=(600, 4))        # saf gurultu
    labs = [seeded_kmeans(Z, 4, s, smooth=1)[0] for s in (11, 22, 33)]
    mean_ari = np.mean([ari(labs[0], labs[i]) for i in (1, 2)])
    assert mean_ari < 0.60               # kabul esiginin ALTINDA kalmali -> NO_STABLE_STATE yolu


# 9 — state-label permutation handling
def test_label_permutation_invariance():
    lab = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2] * 10)
    perm = np.array([2, 0, 1])[lab]      # etiket adlari degisti, bolumleme ayni
    assert ari(lab, perm) == pytest.approx(1.0)


# 10 — model artifact/version mismatch
def test_artifact_hash_mismatch_detected():
    art = {"schema_version": "s1", "feature_version": "f1", "k": 3,
           "centers": [[0.0, 1.0]], "seed": 11}
    art["artifact_hash"] = hashlib.sha256(json.dumps(art, sort_keys=True).encode()).hexdigest()[:16]
    assert verify_artifact(art)
    art["centers"] = [[9.9, 9.9]]        # kurcalandi
    assert not verify_artifact(art)


# 11 — feature-version mismatch
def test_feature_version_mismatch_blocked():
    art = {"schema_version": "latent_ds_v1", "feature_version": "2026-07-02"}
    meta_ok = {"schema_version": "latent_ds_v1", "feature_version": "2026-07-02"}
    meta_bad = {"schema_version": "latent_ds_v1", "feature_version": "2026-08-01"}
    assert_versions(art, meta_ok)
    with pytest.raises(ValueError, match="VERSION MISMATCH"):
        assert_versions(art, meta_bad)


# 12 — stale-data propagation (NaN state uretmez; imputasyon kontrollu)
def test_stale_data_no_nan_states():
    X = np.random.RandomState(2).normal(size=(50, 3))
    X[10:20, 1] = np.nan                 # bayat sensor bolgesi
    std = Standardizer().fit(X[:40], (0, 39))
    Z = std.transform(X)
    assert np.isfinite(Z).all()          # NaN sizintisi yok (medyan-imputasyon 0)


# 13 — latent state dogrudan live permission isteyemez
def test_latent_live_permission_forbidden(tmp_path):
    store = KnowledgeStore(tmp_path / "k.sqlite")
    gov = EpistemicGovernor(store)
    ko = KnowledgeObject(
        knowledge_id="K-LATENT-X", claim="latent states", claim_type=ClaimType.DESCRIPTIVE,
        status=KnowledgeStatus.HOLDOUT_VALIDATED,
        provenance=Provenance(source_tables=["mark_prices"], data_time_range="x..y",
                              code_ref="ami/latent/discovery.py"),
        evidence_level=EvidenceLevel.UNTOUCHED_HOLDOUT, holdouts=1, replications=1,
        falsification=["occupancy kaybolursa"],
        permitted=[Permission.RESEARCH_ONLY, Permission.BACKTEST_ALLOWED, Permission.SHADOW_ALLOWED],
        forbidden=[Permission.LIVE_ALLOWED, Permission.SIZING_ALLOWED, Permission.PORTFOLIO_ALLOWED])
    store.put(ko)
    dec = gov.authorize(Action.OPEN_LONG, ["K-LATENT-X"], {"data_health": "HEALTHY"})
    assert dec.result != "GRANTED"
    okp, why = ko.is_permitted(Permission.LIVE_ALLOWED)
    assert not okp and why == "explicitly_forbidden"
    store.close()


def _spec6a_like(eid="E-L-1"):
    s = ExperimentSpec(eid, "Q-L", population="5m grid", target="latent states",
                       features=["ret5m"], threshold_method="k in [2..6], seed-ARI kurali",
                       chronological_split="80/20", untouched_data="son %20",
                       negative_control="missingness maski", decision_criteria="ARI>=0.6",
                       falsification_rule="NO_STABLE_STATE",
                       execution_model="research_only_no_execution")
    s.freeze()
    return s


def _reg_with_receipt(tmp_path, spec):
    """BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1: ResearchRegistry now
    requires a matching M-0034 gate receipt for any new experiment_id. These
    tests are exercising the freeze/attach_evidence immutability contract, not
    the gate itself -- issue a direct receipt in the same disposable
    knowledge.sqlite the registry checks, same pattern as ami/mutation_suite.py."""
    reg = ResearchRegistry(tmp_path / "r.sqlite", knowledge_path=tmp_path / "k.sqlite")
    kconn = sqlite3.connect(str(reg.knowledge_path))
    _gates.init_gates_schema(kconn)
    _gates.issue_gate_receipt(kconn, experiment_id=spec.experiment_id,
                              canonical_family_id="LATENT_MUTATION_TEST_HARNESS",
                              split_version=None, nullifier=None, registry_result="TEST_HARNESS_DIRECT_RECEIPT")
    kconn.commit(); kconn.close()
    reg.register_experiment(spec)
    return reg


# 14 — freeze sonrasi state-count kurali degistirme
def test_state_count_change_after_freeze_blocked(tmp_path):
    spec = _spec6a_like(); reg = _reg_with_receipt(tmp_path, spec)
    spec.threshold_method = "k in [2..12] (exploration sonrasi genisletildi)"
    with pytest.raises(ConstitutionViolation):
        reg.attach_evidence(EvidenceBundle("EV-L1", spec.experiment_id, {"k": 9}, "SUPPORTS",
                                           dataset_hash="d", code_ref="c"), spec)
    reg.close()


# 15 — exploration sonucuna gore holdout secme
def test_holdout_change_after_freeze_blocked(tmp_path):
    spec = _spec6a_like("E-L-2"); reg = _reg_with_receipt(tmp_path, spec)
    spec.untouched_data = "son %5 (exploration iyi gorunen bolge secildi)"
    spec.chronological_split = "95/5"
    with pytest.raises(ConstitutionViolation):
        reg.attach_evidence(EvidenceBundle("EV-L2", spec.experiment_id, {}, "SUPPORTS",
                                           dataset_hash="d", code_ref="c"), spec)
    reg.close()
