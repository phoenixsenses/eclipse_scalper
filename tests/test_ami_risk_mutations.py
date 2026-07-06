"""Faz 6A-R2 mutation suite — risk/applicability ihlalleri YAKALANMALI.

Her test bir metodoloji ihlalini simule eder; guard'in ihlali reddetmesi beklenir.
"""
from __future__ import annotations

import sqlite3, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ami.constitution import ConstitutionViolation
from ami.enums import Permission
from ami.governance import epistemic_gates as _gates
from ami.latent.risk_applicability import (
    MIN_FOLD_CAND,
    fold_verdict,
    guard_artifact_usable,
    guard_artifact_version,
    guard_bootstrap_confidence,
    guard_fold_aggregation,
    guard_frequency_normalized,
    guard_no_retroactive_alarm,
    guard_permissions,
    loss_concentration,
    require_controls,
    require_exposure_normalization,
    require_loss_concentration,
    require_topwinner_disclosure,
    risk_metrics,
)
from ami.research.registry import EvidenceBundle, ExperimentSpec, ResearchRegistry


def _metrics(n=30, seed=1):
    rng = np.random.RandomState(seed)
    nets = rng.normal(0, 50, n)
    maes = nets - np.abs(rng.normal(30, 20, n))
    mfes = nets + np.abs(rng.normal(30, 20, n))
    return risk_metrics(nets, maes, mfes, ["US"] * n, ["UP"] * n, span_days=10.0)


# m01 — N=50 mutlak MDD'yi N=14 ile dogrudan basari sayma
def test_m01_raw_mdd_cross_n_comparison_blocked():
    with pytest.raises(ConstitutionViolation):
        guard_frequency_normalized(50, 14, ["mdd"])
    with pytest.raises(ConstitutionViolation):
        guard_frequency_normalized(50, 14, ["cum"])
    guard_frequency_normalized(14, 14, ["mdd"])           # esit-N mesru
    guard_frequency_normalized(50, 14, ["cvar5", "mean"])  # normalize metrik mesru


# m02 — random-veto kontrolunu atlama
def test_m02_missing_random_veto_control_blocked():
    res = {"matched_count": {}, "regime_only": {}, "latent_only": {}}
    with pytest.raises(ConstitutionViolation, match="random_veto"):
        require_controls(res)


# m03 — exposure normalization eksikligi
def test_m03_missing_exposure_normalization_blocked():
    m = _metrics()
    bad = {k: v for k, v in m.items() if k != "per_active_hour"}
    with pytest.raises(ConstitutionViolation, match="per_active_hour"):
        require_exposure_normalization(bad)
    require_exposure_normalization(m)  # tam set gecer


# m04 — top-winner dependence gizleme
def test_m04_hidden_topwinner_dependence_blocked():
    m = _metrics()
    bad = {k: v for k, v in m.items() if k != "top3_removed"}
    with pytest.raises(ConstitutionViolation, match="top3_removed"):
        require_topwinner_disclosure(bad)


# m05 — post-hoc risk metrigi / kriter degisimi (freeze sonrasi spec mutasyonu)
def test_m05_posthoc_criteria_change_blocked(tmp_path):
    reg = ResearchRegistry(tmp_path / "r.sqlite", knowledge_path=tmp_path / "k.sqlite")
    spec = ExperimentSpec(
        experiment_id="E-MUT-6AR2", question_id="Q-X", population="p", target="t",
        features=["f"], threshold_method="m", untouched_data="beyan",
        decision_criteria="frozen kriterler", falsification_rule="frozen kural")
    spec.freeze()
    # BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1: direct gate receipt
    # (this test exercises freeze/attach_evidence immutability, not the gate).
    kconn = sqlite3.connect(str(reg.knowledge_path))
    _gates.init_gates_schema(kconn)
    _gates.issue_gate_receipt(kconn, experiment_id=spec.experiment_id,
                              canonical_family_id="RISK_MUTATION_TEST_HARNESS",
                              split_version=None, nullifier=None, registry_result="TEST_HARNESS_DIRECT_RECEIPT")
    kconn.commit(); kconn.close()
    reg.register_experiment(spec)
    spec.decision_criteria = "sonuca gore gevsetilmis kriter"   # ihlal
    with pytest.raises(ConstitutionViolation, match="freeze"):
        reg.attach_evidence(EvidenceBundle("EV-MUT", spec.experiment_id, {}, "SUPPORTS"), spec)
    reg.close()


# m06 — alarmi performans dususunden sonra geriye donuk uretme
def test_m06_retroactive_alarm_blocked():
    with pytest.raises(ConstitutionViolation, match="Retroaktif"):
        guard_no_retroactive_alarm(window_end_idx=100, data_used_max_idx=150)
    guard_no_retroactive_alarm(window_end_idx=100, data_used_max_idx=100)


# m07 — good-trade sacrifice raporlamama
def test_m07_unreported_winner_sacrifice_blocked():
    nets = np.array([50.0, -60.0, 20.0, -200.0, 10.0, -30.0, 80.0, -10.0])
    maes = nets - 20.0; mfes = nets + 120.0
    sel = np.array([True, False, True, False, True, True, False, True])
    rep = loss_concentration(nets, maes, mfes, sel)
    assert rep["winner_sacrifice_bps"] is not None
    bad = {k: v for k, v in rep.items() if k != "winner_sacrifice_bps"}
    with pytest.raises(ConstitutionViolation, match="sacrifice"):
        require_loss_concentration(bad)


# m08 — fold cherry-picking (yalniz iyi fold'u rapor etme)
def test_m08_fold_cherrypicking_blocked():
    folds = [{"evaluable": True, "fold_pass": True},
             {"evaluable": True, "fold_pass": False},
             {"evaluable": True, "fold_pass": False}]
    claimed = {"evaluable_folds": 1, "passed_folds": 1,
               "majority_pass": True, "all_folds_reported": 1}
    with pytest.raises(ConstitutionViolation, match="cherry"):
        guard_fold_aggregation(folds, claimed)
    guard_fold_aggregation(folds, fold_verdict(folds))


# m09 — regime-only kontrolunu atlama
def test_m09_missing_regime_only_control_blocked():
    res = {"matched_count": {}, "random_veto": {}, "latent_only": {}}
    with pytest.raises(ConstitutionViolation, match="regime_only"):
        require_controls(res)


# m10 — drift UNUSABLE iken artifact'i trade seciminde kullanma + stale artifact
def test_m10_unusable_drift_artifact_use_blocked():
    with pytest.raises(ConstitutionViolation):
        guard_artifact_usable("UNUSABLE", "trade_selection")
    with pytest.raises(ConstitutionViolation):
        guard_artifact_usable("SHIFTED", "live")
    guard_artifact_usable("UNUSABLE", "research")   # research kullanimi serbest
    with pytest.raises(ConstitutionViolation, match="Stale artifact"):
        guard_artifact_version({"feature_version": "2026-01-01"},
                               {"feature_version": "2026-07-02"})


# m11 — latent risk sonucu ile LIVE/SIZING izni isteme
def test_m11_live_sizing_permission_request_blocked():
    with pytest.raises(ConstitutionViolation, match="LIVE"):
        guard_permissions({Permission.LIVE_ALLOWED})
    with pytest.raises(ConstitutionViolation):
        guard_permissions({Permission.SHADOW_ALLOWED, Permission.SIZING_ALLOWED})
    with pytest.raises(ConstitutionViolation):
        guard_permissions({Permission.PORTFOLIO_ALLOWED})
    guard_permissions({Permission.RESEARCH_ONLY, Permission.SHADOW_ALLOWED})


# m12 — small-N bootstrap guvenini abartma
def test_m12_small_n_bootstrap_overconfidence_flagged():
    assert guard_bootstrap_confidence(MIN_FOLD_CAND - 1) == "INSUFFICIENT_SAMPLE"
    assert guard_bootstrap_confidence(MIN_FOLD_CAND) == "OK"
    assert guard_bootstrap_confidence(1) == "INSUFFICIENT_SAMPLE"


# m13 — pozitif kontrol: tam metrik seti ve fold agregasyonu dogru calisiyor
def test_m13_positive_controls():
    m = _metrics()
    for key in ("cvar5", "downside_dev", "mdd", "avg_dd", "worst", "bottom3_cum",
                "loss_rate", "max_consec_loss", "ret_vol", "downside_dev",
                "per_active_hour", "exposure_hours", "session_conc", "regime_conc",
                "top1_removed", "top3_removed", "top5_removed", "mae_mean", "mae_p10"):
        assert key in m, key
    v = fold_verdict([{"evaluable": True, "fold_pass": True},
                      {"evaluable": False},
                      {"evaluable": True, "fold_pass": True}])
    assert v == {"evaluable_folds": 2, "passed_folds": 2,
                 "majority_pass": True, "all_folds_reported": 3}
