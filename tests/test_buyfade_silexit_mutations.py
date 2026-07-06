"""E-BUYFADE-SILEXIT-001 mutation suite — survivor-bias/lookahead ihlalleri YAKALANMALI."""
from __future__ import annotations

import sqlite3, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ami.constitution import ConstitutionViolation
from ami.governance import epistemic_gates as _gates
from ami.research.registry import EvidenceBundle, ExperimentSpec, ResearchRegistry
from tools.research_s34_buyfade_structural import (
    MIN_CELL, g_split_purge, g_tiny_cell, g_train_only_selection, no_overlap, stat_block)
from tools.research_s34_buyfade_silence_exit import (
    FEE, T30, g_breakdown_causal, g_fee_on_extension, g_no_manage_closed,
    g_no_pre_t30_silence_use, g_no_route_mutation, g_realized_only,
    g_survivor_universe, require_noisy_control)


# 1 — T+30'a ulasamayan trade'leri dislama
def test_x01_survivor_exclusion_blocked():
    with pytest.raises(ConstitutionViolation, match="Survivor"):
        g_survivor_universe(n_silence_all=106, n_reported=99)   # 7 erken-SL atilmis
    g_survivor_universe(106, 106)


# 2 — T+30 oncesi SL'leri silence evreninden silme (ayni guard, farkli senaryo)
def test_x02_pre_t30_sl_removal_blocked():
    with pytest.raises(ConstitutionViolation):
        g_survivor_universe(39, 38)


# 3 — silence bilgisini T0 (veya <T30) kararinda kullanma
def test_x03_silence_before_t30_blocked():
    with pytest.raises(ConstitutionViolation, match="bilinirlik"):
        g_no_pre_t30_silence_use(decision_min=0.0)
    with pytest.raises(ConstitutionViolation):
        g_no_pre_t30_silence_use(decision_min=29.9)
    g_no_pre_t30_silence_use(30.0)


# 4 — gelecekteki breakdown zamanini onceden bilme
def test_x04_future_breakdown_blocked():
    with pytest.raises(ConstitutionViolation, match="breakdown"):
        g_breakdown_causal(exit_min=40.0, breakdown_min=50.0, grace=0.0)
    with pytest.raises(ConstitutionViolation):
        g_breakdown_causal(exit_min=50.5, breakdown_min=50.0, grace=1.0)
    g_breakdown_causal(51.0, 50.0, 1.0)


# 5 — best exit horizon'i full dataset'ten secme
def test_x05_full_dataset_selection_blocked():
    with pytest.raises(ConstitutionViolation):
        g_train_only_selection("val")
    with pytest.raises(ConstitutionViolation):
        g_train_only_selection("untouched")


# 6 — horizon overlap leakage (no_overlap pozitif kontrol)
def test_x06_horizon_overlap_guard():
    evs = [{"ts": i * 3_600_000} for i in range(30)]     # saatlik eventler
    kept = no_overlap(evs, horizon_min=240)              # 4h exit
    for a, b in zip(kept, kept[1:]):
        assert b["ts"] - a["ts"] >= 240 * 60_000


# 7 — ayni event/cycle split leakage (24h purge)
def test_x07_cycle_split_purge():
    with pytest.raises(ConstitutionViolation, match="purge"):
        g_split_purge(t_train_max=0, t_val_min=3_600_000)
    g_split_purge(0, 25 * 3_600_000)


# 8 — T+30 unrealized PnL'yi realized gibi yazma
def test_x08_unrealized_as_realized_blocked():
    with pytest.raises(ConstitutionViolation, match="unrealized"):
        g_realized_only({"t30_unrealized_bps": 22.9, "_unrealized_in_cum": True})
    g_realized_only({"t30_unrealized_bps": 22.9})        # ayri raporlanmasi serbest


# 9 — fee'yi extended hold'da uygulamama
def test_x09_missing_fee_on_extension_blocked():
    with pytest.raises(ConstitutionViolation, match="Fee"):
        g_fee_on_extension(net_gross_diff=0.0)
    g_fee_on_extension(FEE)


# 10 — structural exit feature timestamp ihlali (breakdown-causal ile ayni sinif)
def test_x10_structural_feature_timestamp():
    with pytest.raises(ConstitutionViolation):
        g_breakdown_causal(exit_min=T30, breakdown_min=T30 + 10, grace=0.0)


# 11 — tiny 4h-DOWN hucresini PASS ilan etme
def test_x11_tiny_regime_cell_blocked():
    assert g_tiny_cell(MIN_CELL - 4, "4hDOWN_sil") == "INSUFFICIENT_SAMPLE"
    assert g_tiny_cell(MIN_CELL, "4hDOWN_sil") == "4hDOWN_sil"


# 12 — top-winner dependence gizleme
def test_x12_topwinner_disclosure():
    rows = [{"net": float(x), "mfe": 10.0, "mae": -5.0, "t_mfe": 5.0, "stop_hit_min": None}
            for x in np.random.RandomState(1).normal(5, 20, 25)]
    st = stat_block(rows, span_days=10)
    for k in ("top1_removed", "top3_removed", "top5_removed"):
        assert k in st


# 13 — noisy control'u atlama
def test_x13_noisy_control_required():
    with pytest.raises(ConstitutionViolation, match="Noisy"):
        require_noisy_control({"H_random_exit_timing_train": {}})
    require_noisy_control({"G_noisy_same_exits": {}, "H_random_exit_timing_train": {}})


# 14 — T+30'da kapanmis trade'e sonradan yonetim uygulama
def test_x14_manage_closed_trade_blocked():
    with pytest.raises(ConstitutionViolation, match="kapanmis"):
        g_no_manage_closed(closed_pre_t30=True, action="EXTEND_TO_4H")
    g_no_manage_closed(True, "KEEP_RESULT")
    g_no_manage_closed(False, "EXTEND_TO_4H")


# 15 — mevcut shadow/live route'u otomatik degistirme
def test_x15_route_mutation_blocked():
    for p in ("tools/s34_realtime_shadow_runner.py", ".env",
              "tools/s34_state_machine_live_executor.py", "execution/order.py",
              "risk/limits.py", "brain/policy.py"):
        with pytest.raises(ConstitutionViolation):
            g_no_route_mutation(p)
    g_no_route_mutation("tools/research_s34_buyfade_silence_exit.py")


# bonus — post-hoc kriter degisimi freeze ile bloklu
def test_x16_posthoc_spec_change_blocked(tmp_path):
    reg = ResearchRegistry(tmp_path / "r.sqlite", knowledge_path=tmp_path / "k.sqlite")
    spec = ExperimentSpec(experiment_id="E-MUT-SILEXIT", question_id="Q", population="p",
                          target="t", features=["f"], threshold_method="frozen exits",
                          untouched_data="beyan", decision_criteria="econ>=3",
                          falsification_rule="kriter gevsetme yasak")
    spec.freeze()
    # BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1: direct gate receipt
    # (this test exercises freeze/attach_evidence immutability, not the gate).
    kconn = sqlite3.connect(str(reg.knowledge_path))
    _gates.init_gates_schema(kconn)
    _gates.issue_gate_receipt(kconn, experiment_id=spec.experiment_id,
                              canonical_family_id="SILEXIT_MUTATION_TEST_HARNESS",
                              split_version=None, nullifier=None, registry_result="TEST_HARNESS_DIRECT_RECEIPT")
    kconn.commit(); kconn.close()
    reg.register_experiment(spec)
    spec.decision_criteria = "econ>=1 (sonuca gore)"
    with pytest.raises(ConstitutionViolation):
        reg.attach_evidence(EvidenceBundle("EV", spec.experiment_id, {}, "SUPPORTS"), spec)
    reg.close()
