"""BUY-FADE yapisal + 8A re-entry mutation suite — metodoloji ihlalleri YAKALANMALI."""
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
    MIN_CELL, candidate_pass, g_backward_only, g_completed_bar, g_event_high_causal,
    g_executable, g_no_t0_silence, g_split_purge, g_tiny_cell, g_train_only_selection,
    g_unknown_not_neutral, no_overlap, session_name, stat_block)
from tools.research_s34_buyfade_reentry import (
    g_all_attempts_reported, g_causal_prefix, g_entries_separate, g_fee_per_entry,
    g_flip_separate_claim)


def _rows(n=20, seed=3):
    rng = np.random.RandomState(seed)
    return [{"net": float(x), "mfe": abs(float(x)) + 5, "mae": -abs(float(x)) / 2,
             "t_mfe": 10.0, "stop_hit_min": None} for x in rng.normal(2, 30, n)]


# ── yapisal (§11) ─────────────────────────────────────────────────────────────
def test_m01_future_swing_for_genesis_blocked():
    with pytest.raises(ConstitutionViolation, match="FUTURE"):
        g_backward_only(t0=1000, data_end=2000)
    g_backward_only(t0=1000, data_end=1000)


def test_m02_unfinished_candle_blocked():
    with pytest.raises(ConstitutionViolation, match="UNFINISHED"):
        g_completed_bar(t0=1000, bar_close_ts=1500)
    g_completed_bar(t0=1000, bar_close_ts=1000)


def test_m03_postevent_silence_as_t0_feature_blocked():
    with pytest.raises(ConstitutionViolation, match="yasak"):
        g_no_t0_silence(["ret_60m", "silence_30m"])
    with pytest.raises(ConstitutionViolation):
        g_no_t0_silence(["buy_state"])
    g_no_t0_silence(["ret_60m", "pre_silence_10m"])   # pre-event silence T0'da mesru


def test_m04_delayed_entry_future_event_high_blocked():
    with pytest.raises(ConstitutionViolation, match="event high"):
        g_event_high_causal(trigger_ts=1000, high_window_end=2000)
    g_event_high_causal(trigger_ts=1000, high_window_end=1000)


def test_m05_best_horizon_from_full_dataset_blocked():
    with pytest.raises(ConstitutionViolation, match="TRAIN"):
        g_train_only_selection("val")
    with pytest.raises(ConstitutionViolation):
        g_train_only_selection("untouched")
    g_train_only_selection("train")


def test_m06_same_cascade_split_purge():
    with pytest.raises(ConstitutionViolation, match="purge"):
        g_split_purge(t_train_max=1_000_000, t_val_min=1_000_000 + 3_600_000)  # 1h < 24h
    g_split_purge(t_train_max=0, t_val_min=25 * 3_600_000)


def test_m07_no_overlap_positive_control():
    evs = [{"ts": i * 10 * 60_000} for i in range(10)]           # 10dk arayla
    kept = no_overlap(evs, horizon_min=45)
    for a, b in zip(kept, kept[1:]):
        assert b["ts"] - a["ts"] >= 45 * 60_000


def test_m08_outcome_label_into_features_blocked():
    for bad in ("path_class", "outcome_45m", "mfe_label", "reclaim_flag"):
        with pytest.raises(ConstitutionViolation):
            g_no_t0_silence([bad])


def test_m09_session_timezone_boundaries():
    import datetime
    def ts_at(h):  # UTC saat h
        return int(datetime.datetime(2026, 6, 3, h, 0, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    assert session_name(ts_at(6)) == "OFF"
    assert session_name(ts_at(7)) == "EUROPE"
    assert session_name(ts_at(12)) == "EUROPE"
    assert session_name(ts_at(13)) == "US"
    assert session_name(ts_at(20)) == "US"
    assert session_name(ts_at(21)) == "OFF"


def test_m10_missing_timeframe_not_neutral():
    assert g_unknown_not_neutral("UNKNOWN") == "UNKNOWN"          # UP/RANGE'e cevrilemez
    with pytest.raises(ConstitutionViolation):
        g_unknown_not_neutral("NEUTRAL")


def test_m11_nonexecutable_delayed_fill():
    assert g_executable(None, trigger_ts=1000) is False
    assert g_executable(1000 + 300_000, trigger_ts=1000) is False  # 5dk stale
    assert g_executable(1000 + 60_000, trigger_ts=1000) is True


def test_m12_topwinner_disclosure_in_statblock():
    st = stat_block(_rows(), span_days=10)
    for k in ("top1_removed", "top3_removed", "top5_removed", "mae_p10", "stop_rate"):
        assert k in st


def test_m13_tiny_cell_not_alpha():
    assert g_tiny_cell(MIN_CELL - 1, "cell") == "INSUFFICIENT_SAMPLE"
    assert g_tiny_cell(MIN_CELL, "cell") == "cell"


def test_m14_silence_threshold_posthoc_change_blocked(tmp_path):
    reg = ResearchRegistry(tmp_path / "r.sqlite", knowledge_path=tmp_path / "k.sqlite")
    spec = ExperimentSpec(experiment_id="E-MUT-BF", question_id="Q", population="p",
                          target="t", features=["f"], threshold_method="silence v1 frozen",
                          untouched_data="beyan", decision_criteria="c", falsification_rule="f")
    spec.freeze()
    # BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1: direct gate receipt
    # (this test exercises freeze/attach_evidence immutability, not the gate).
    kconn = sqlite3.connect(str(reg.knowledge_path))
    _gates.init_gates_schema(kconn)
    _gates.issue_gate_receipt(kconn, experiment_id=spec.experiment_id,
                              canonical_family_id="BUYFADE_MUTATION_TEST_HARNESS",
                              split_version=None, nullifier=None, registry_result="TEST_HARNESS_DIRECT_RECEIPT")
    kconn.commit(); kconn.close()
    reg.register_experiment(spec)
    spec.threshold_method = "silence v2 (sonuca gore)"
    with pytest.raises(ConstitutionViolation):
        reg.attach_evidence(EvidenceBundle("EV", spec.experiment_id, {}, "SUPPORTS"), spec)
    reg.close()


def test_m15_multiple_testing_required_for_pass():
    ok, fails = candidate_pass(holdout_net=10, holdout_n=30, top3_removed_net=5,
                               incremental_vs_baseline=True, econ_net=5,
                               folds_same_dir=2, family_p=0.20, tiny=False)
    assert not ok and "family_p" in fails


def test_m16_candidate_pass_positive_and_negatives():
    ok, fails = candidate_pass(holdout_net=10, holdout_n=30, top3_removed_net=5,
                               incremental_vs_baseline=True, econ_net=5,
                               folds_same_dir=2, family_p=0.01, tiny=False)
    assert ok and not fails
    ok, fails = candidate_pass(holdout_net=10, holdout_n=30, top3_removed_net=-5,
                               incremental_vs_baseline=True, econ_net=5,
                               folds_same_dir=2, family_p=0.01, tiny=False)
    assert not ok and "topwinner_dependent" in fails


# ── 8A re-entry (§J) ─────────────────────────────────────────────────────────
def test_r01_future_dip_trigger_blocked():
    with pytest.raises(ConstitutionViolation, match="gelecek"):
        g_causal_prefix(scan_end_idx=30, used_idx=60)
    g_causal_prefix(30, 30)


def test_r02_fee_only_first_entry_blocked():
    with pytest.raises(ConstitutionViolation, match="Fee"):
        g_fee_per_entry(n_entries=3, fee_applied=5.0)
    g_fee_per_entry(3, 15.0)


def test_r03_entries_merged_as_one_trade_blocked():
    with pytest.raises(ConstitutionViolation, match="ayri"):
        g_entries_separate({"cycle_total": {}})
    g_entries_separate({"entry1": {}, "entry2": {}, "cycle_total": {}})


def test_r04_unsuccessful_reentries_dropped_blocked():
    with pytest.raises(ConstitutionViolation, match="cikarilamaz"):
        g_all_attempts_reported(n_attempt=10, n_reported=7)
    g_all_attempts_reported(10, 10)


def test_r05_flip_and_reentry_same_claim_blocked():
    with pytest.raises(ConstitutionViolation):
        g_flip_separate_claim({"S_TO_S": "FLIP_INCREMENTAL", "S_TO_L": "x"})
    g_flip_separate_claim({"S_TO_S": "SHORT_REENTRY_NON_INCREMENTAL",
                           "S_TO_L": "SHORT_TO_LONG_INCREMENTAL"})


def test_r06_cooldown_selection_train_only():
    with pytest.raises(ConstitutionViolation):
        g_train_only_selection("val")


def test_r07_entry3_small_n_flagged():
    assert g_tiny_cell(3, "entry3") == "INSUFFICIENT_SAMPLE"


def test_r08_cycle_split_purge_shared_guard():
    with pytest.raises(ConstitutionViolation):
        g_split_purge(100, 100 + 60_000)
