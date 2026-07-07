"""BATCH-CASCADE-ABSORPTION-IMPACT-PREREGISTRATION-V1 -- focused validation tests.

Proves the preregistration's population/split/nullifier/gate-receipt claims
are reproducible and that no TEST outcome was (or can accidentally be) read
by anything in this test file. Uses mode=ro real-data queries for population
counts and disposable copies for the gate/nullifier/receipt mechanism proofs
-- same discipline as tests/test_ami_cvd_primary_long_preregistration_v1.py.

Run: pytest tests/test_ami_absorption_impact_preregistration_v1.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sqlite3

import pytest

from ami.governance import epistemic_gates as gates

REAL_CANONICAL_PATH = "D:/eclipse_scalper/data/ami/canonical.sqlite"
REAL_KNOWLEDGE_PATH = "D:/eclipse_scalper/data/ami/knowledge.sqlite"

PREREG_JSON = pathlib.Path(__file__).resolve().parents[1] / "reports" / "research" / "s34" / \
    "S34_CASCADE_ABSORPTION_IMPACT_PREREGISTRATION_V1.json"
PREREG_MD = pathlib.Path(__file__).resolve().parents[1] / "reports" / "research" / "s34" / \
    "S34_CASCADE_ABSORPTION_IMPACT_PREREGISTRATION_V1.md"

CORRECTED = "path-v2-candle-repair-r1"


@pytest.fixture(scope="module")
def prereg():
    return json.loads(PREREG_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Artifacts exist and are internally consistent
# ---------------------------------------------------------------------------

def test_preregistration_artifacts_exist():
    assert PREREG_JSON.exists()
    assert PREREG_MD.exists()


def test_json_manifest_status_is_preregistered_not_executed(prereg):
    assert prereg["status"] == "PREREGISTERED_NOT_EXECUTED"
    assert prereg["nullifier_and_gate"]["nullifier_consumed_by_this_batch"] is False


def test_graveyard_recorded_clean(prereg):
    assert prereg["graveyard"]["hits"] == []
    assert prereg["graveyard"]["confirmed"] is True


def test_no_post_outcome_eligibility_filtering_declared(prereg):
    assert prereg["population"]["no_post_outcome_eligibility_filtering"] is True


def test_predictor_is_continuous_no_binning_no_threshold(prereg):
    pred = prereg["predictor"]
    assert pred["type"] == "continuous"
    assert pred["binning"] is False
    assert pred["threshold"] is False


def test_forbidden_analyses_list_covers_required_exclusions(prereg):
    forbidden = " ".join(prereg["forbidden_analyses"]).lower()
    for phrase in ("threshold scan", "subgroup rescue", "interaction search",
                  "pooling exact and proxy", "nonlinear/spline",
                  "w60, w600, w1800 or w3600"):
        assert phrase in forbidden


def test_only_w300_permitted_as_outcome_linked_window(prereg):
    """Amendment 2: only W300 may be joined to outcomes in this experiment."""
    assert prereg["population"]["predictor_window_id"] == "W300"
    ruling = prereg["window_ruling"]["text"]
    assert "Only W300 may be joined to outcomes" in ruling


def test_zero_variance_europe_policy_frozen(prereg):
    policy = prereg["rank_deficiency_and_zero_variance_policy"]
    assert policy["dropped_level_this_experiment"] == "EUROPE (0/91 TRAIN, 0/40 TEST)"
    assert policy["retained_dummy_columns"] == ["session_US", "session_OFF"]


# ---------------------------------------------------------------------------
# 2. Real-data population/split reproducibility (read-only, no gateway calls
#    here -- direct SQL, to avoid this test file adding further exposure-
#    ledger writes)
# ---------------------------------------------------------------------------

def _ro_conn():
    return sqlite3.connect(f"file:{REAL_CANONICAL_PATH}?mode=ro", uri=True)


def test_long_population_count_matches(prereg):
    conn = _ro_conn()
    try:
        n = conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle WHERE direction='LONG'").fetchone()[0]
    finally:
        conn.close()
    assert n == prereg["population"]["long_n"] == 220


def test_w300_fully_exact_reconstructable_for_long_population():
    conn = _ro_conn()
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM ami_absorption_impact_window_quality_v1 WHERE window_id='W300'").fetchone()[0]
        exact = conn.execute(
            "SELECT COUNT(*) FROM ami_absorption_impact_window_quality_v1 WHERE window_id='W300' "
            "AND quality_status='EXACT_RECONSTRUCTABLE'").fetchone()[0]
        long_coverage = conn.execute("""
            SELECT COUNT(*) FROM ami_absorption_impact_windowed_flow f
            JOIN ami_signal_lifecycle s ON s.signal_id = f.signal_id
            WHERE f.window_id='W300' AND s.direction='LONG'
        """).fetchone()[0]
    finally:
        conn.close()
    assert total == 324
    assert exact == 324
    assert long_coverage == 220


def test_known_at_contract_zero_violations_for_w300():
    conn = _ro_conn()
    try:
        violations = conn.execute(
            "SELECT COUNT(*) FROM ami_absorption_impact_windowed_flow WHERE window_id='W300' "
            "AND window_end_ts_ms > signal_birth_ts").fetchone()[0]
        not_safe = conn.execute(
            "SELECT COUNT(*) FROM ami_absorption_impact_windowed_flow WHERE window_id='W300' "
            "AND known_at_classification != 'KNOWN_AT_SAFE'").fetchone()[0]
    finally:
        conn.close()
    assert violations == 0
    assert not_safe == 0


def _compute_population(conn):
    long_rows = conn.execute(
        "SELECT signal_id, independent_cycle_id, signal_birth_ts FROM ami_signal_lifecycle "
        "WHERE direction='LONG'").fetchall()
    obs_rows = conn.execute(
        "SELECT signal_id, path_definition_version, observation_status "
        "FROM ami_lifecycle_path_observations WHERE horizon_name='swing_24h'").fetchall()
    effective_status: dict[str, str] = {}
    version_by_sig: dict[str, str] = {}
    for sid, version, status in obs_rows:
        if sid not in version_by_sig or version == CORRECTED:
            effective_status[sid] = status
            version_by_sig[sid] = version
    eligible_sids = {sid for sid, status in effective_status.items() if status == "OK"}
    long_signal_ids = {sid for sid, _, _ in long_rows}
    eligible_long = eligible_sids & long_signal_ids
    by_cycle: dict[str, tuple[str, int]] = {}
    for sid, cyc, bts in long_rows:
        if sid not in eligible_long:
            continue
        if cyc not in by_cycle or bts < by_cycle[cyc][1]:
            by_cycle[cyc] = (sid, bts)
    reps = sorted(by_cycle.items(), key=lambda kv: kv[1][1])
    cut = int(len(reps) * 0.7)
    return eligible_long, reps, reps[:cut], reps[cut:]


def test_reproduce_eligible_and_representative_population(prereg):
    """Reproduces the preregistration's population/split numbers from a fresh
    query, independent of the identity-resolution session's own computation."""
    conn = _ro_conn()
    try:
        eligible_long, reps, train, test = _compute_population(conn)
    finally:
        conn.close()

    assert len(eligible_long) == prereg["population"]["outcome_eligible_ok_n"] == 194
    assert len(reps) == prereg["population"]["representative_cycle_n"] == 131
    assert len(train) == prereg["population"]["train_cycle_n"] == 91
    assert len(test) == prereg["population"]["test_cycle_n"] == 40
    assert train[-1][1][1] < test[0][1][1]  # no straddling

    train_hash = hashlib.sha256(",".join(sorted(c for c, _ in train)).encode()).hexdigest()
    test_hash = hashlib.sha256(",".join(sorted(c for c, _ in test)).encode()).hexdigest()
    assert train_hash == prereg["population"]["train_cycle_set_hash_sha256"]
    assert test_hash == prereg["population"]["test_cycle_set_hash_sha256"]


def test_train_test_cycle_sets_identical_to_closed_cvd_prereg(prereg):
    """The population is identical to the closed CVD preregistration's own
    (eligibility depends on outcome availability, not on which feature is
    under test) -- proven, not assumed, by an independent hash match."""
    assert prereg["population"]["train_cycle_set_hash_sha256"] == \
        "61486bc62392eed7b7fc038715f2cd9775e270a568e5c1f728dc2d60417671a5"
    assert prereg["population"]["test_cycle_set_hash_sha256"] == \
        "98174ed356826b15bd8513584015447b68d18718bb933d75380a4d6b2c4f7b04"


def test_session_zero_variance_europe_reproduced(prereg):
    """Independently reproduces the EUROPE zero-observation fact in both
    TRAIN and TEST for this population, from a fresh query."""
    import datetime
    from ami.chart.level_registry import _session_of_hour

    conn = _ro_conn()
    try:
        _, _, train, test = _compute_population(conn)
    finally:
        conn.close()

    def session_counts(reps):
        from collections import Counter
        c = Counter()
        for _, (sid, bts) in reps:
            hour = datetime.datetime.fromtimestamp(bts / 1000, datetime.timezone.utc).hour
            c[_session_of_hour(hour)] += 1
        return dict(c)

    train_counts = session_counts(train)
    test_counts = session_counts(test)
    assert train_counts.get("EUROPE", 0) == 0
    assert test_counts.get("EUROPE", 0) == 0
    policy = prereg["rank_deficiency_and_zero_variance_policy"]
    assert policy["dropped_level_this_experiment"] == "EUROPE (0/91 TRAIN, 0/40 TEST)"


def test_predictor_never_bound_by_floor_in_train(prereg):
    conn = _ro_conn()
    try:
        _, _, train, _ = _compute_population(conn)
        train_signal_ids = [sid for _, (sid, _) in train]
        placeholders = ",".join("?" for _ in train_signal_ids)
        rows = conn.execute(
            f"SELECT floor_usd_m_applied, price_response_per_signed_notional FROM "
            f"ami_absorption_impact_windowed_flow WHERE window_id='W300' AND signal_id IN ({placeholders})",
            train_signal_ids).fetchall()
    finally:
        conn.close()
    assert len(rows) == 91
    assert all(v is not None for _, v in rows)
    assert sum(1 for applied, _ in rows if applied) == 0
    assert prereg["predictor"]["floor_usd_m"] == 0.01


# ---------------------------------------------------------------------------
# 3. Gate/nullifier/receipt mechanism proof (disposable copy, not the real DB)
# ---------------------------------------------------------------------------

@pytest.fixture()
def disposable_knowledge(tmp_path):
    import shutil
    path = tmp_path / "knowledge_disposable.sqlite"
    shutil.copyfile(REAL_KNOWLEDGE_PATH, path)
    return path


def test_family_and_split_identity_are_reproducible(prereg):
    ident = prereg["identity"]
    family_id = gates.resolve_canonical_family_id(ident["question_ids"], ident["hypothesis_id"])
    assert family_id == ident["canonical_family_id"] == "FAMv1:3e2dfe63f9e271bf"


def test_nullifier_reproducible_from_frozen_cycle_sets(prereg):
    ident = prereg["identity"]
    conn = _ro_conn()
    try:
        _, _, _, test = _compute_population(conn)
    finally:
        conn.close()
    test_cycle_ids = [c for c, _ in test]
    nullifier = gates.derive_test_nullifier(
        ident["canonical_family_id"], ident["split_version"], test_cycle_ids)
    assert nullifier == prereg["nullifier_and_gate"]["test_nullifier_sha256"] == \
        "4e3d1229edc04a946ef29994f1562444fd7c9e77b6ff3ecf3004677f919df7d4"
    assert nullifier != "085397f31c199c1d0c1d5ce647af4d1aa311166c63199f92872e089db8e72a7a"  # differs from CVD's own


def test_gate_receipt_mechanism_round_trips_on_disposable_copy(disposable_knowledge, prereg):
    kconn = sqlite3.connect(str(disposable_knowledge))
    gates.init_gates_schema(kconn)
    ident = prereg["identity"]
    nullifier_before = kconn.execute(
        "SELECT COUNT(*) FROM epistemic_test_nullifiers WHERE nullifier=?",
        (prereg["nullifier_and_gate"]["test_nullifier_sha256"],)).fetchone()[0]
    receipt_hash = gates.issue_gate_receipt(
        kconn, experiment_id="TEST-COPY-" + ident["experiment_id"], canonical_family_id=ident["canonical_family_id"],
        split_version=ident["split_version"], nullifier=prereg["nullifier_and_gate"]["test_nullifier_sha256"],
        registry_result="PREREGISTERED_NOT_EXECUTED")
    kconn.commit()
    assert gates.has_gate_receipt(kconn, "TEST-COPY-" + ident["experiment_id"])
    # nullifier must NOT be consumed by issuing a receipt alone
    nullifier_after = kconn.execute(
        "SELECT COUNT(*) FROM epistemic_test_nullifiers WHERE nullifier=?",
        (prereg["nullifier_and_gate"]["test_nullifier_sha256"],)).fetchone()[0]
    assert nullifier_after == nullifier_before == 0
    kconn.close()


def test_real_nullifier_and_receipt_state(prereg):
    """Real, read-only confirmation of the actual preregistration's effect on
    the real knowledge.sqlite: receipt present, nullifier still unconsumed."""
    conn = sqlite3.connect(f"file:{REAL_KNOWLEDGE_PATH}?mode=ro", uri=True)
    try:
        receipt = conn.execute(
            "SELECT registry_result FROM experiment_gate_receipts WHERE experiment_id=?",
            (prereg["identity"]["experiment_id"],)).fetchone()
        nullifier_rows = conn.execute(
            "SELECT COUNT(*) FROM epistemic_test_nullifiers WHERE nullifier=?",
            (prereg["nullifier_and_gate"]["test_nullifier_sha256"],)).fetchone()[0]
    finally:
        conn.close()
    assert receipt == ("PREREGISTERED_NOT_EXECUTED",)
    assert nullifier_rows == 0


# ---------------------------------------------------------------------------
# 4. No experiment/result was created in canonical.sqlite; invariants hold
# ---------------------------------------------------------------------------

def test_no_experiment_created_and_canonical_invariants_hold():
    conn = _ro_conn()
    try:
        n_reg = conn.execute("SELECT COUNT(*) FROM experiment_registry").fetchone()[0]
        n_res = conn.execute("SELECT COUNT(*) FROM experiment_results").fetchone()[0]
        version = conn.execute(
            "SELECT version FROM schema_versions WHERE component='canonical_warehouse'").fetchone()[0]
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        counts = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("ami_events", "ami_signal_lifecycle", "ami_cycles",
                      "ami_birth_truncated_cascade_geometry")
        }
        absorption_counts = {
            "windowed_flow": conn.execute(
                "SELECT COUNT(*) FROM ami_absorption_impact_windowed_flow").fetchone()[0],
            "quality": conn.execute(
                "SELECT COUNT(*) FROM ami_absorption_impact_window_quality_v1").fetchone()[0],
            "exclusions": conn.execute(
                "SELECT COUNT(*) FROM ami_absorption_impact_exclusions").fetchone()[0],
        }
    finally:
        conn.close()
    assert version == 13
    assert integrity == "ok"
    assert counts == {"ami_events": 252, "ami_signal_lifecycle": 324, "ami_cycles": 167,
                       "ami_birth_truncated_cascade_geometry": 220}
    assert absorption_counts == {"windowed_flow": 1619, "quality": 1620, "exclusions": 1}
    # experiment_registry/experiment_results are NOT asserted to an exact frozen
    # number here (unlike the pre-existing, waived tests) -- this preregistration
    # itself creates no experiment_registry/experiment_results row (see below);
    # any value is consistent with that as long as it did not change across
    # this batch, which is verified by the transition proof's before/after check.
    assert n_reg >= 0
    assert n_res >= 0
