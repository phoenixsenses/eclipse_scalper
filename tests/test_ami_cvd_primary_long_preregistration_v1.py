"""G2-CVD-PRIMARY-LONG-PREREGISTRATION-V1 -- focused validation tests.

Proves the preregistration's population/split/nullifier/gate-receipt claims
are reproducible and that no TEST outcome was (or can accidentally be) read
by anything in this test file. Uses mode=ro real-data queries for population
counts (no gateway calls here, so no further researcher_exposure_ledger
writes from this test file itself -- the identity-resolution session's own
exposure appends are already accounted for in the transition proof) and
disposable copies for the gate/nullifier/receipt mechanism proofs.

Run: pytest tests/test_ami_cvd_primary_long_preregistration_v1.py --basetemp <scratchpad> -p no:cacheprovider
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
    "S34_CVD_PRIMARY_LONG_PREREGISTRATION_V1.json"
PREREG_MD = pathlib.Path(__file__).resolve().parents[1] / "reports" / "research" / "s34" / \
    "S34_CVD_PRIMARY_LONG_PREREGISTRATION_V1.md"


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
                  "pooling exact and proxy", "nonlinear/spline"):
        assert phrase in forbidden


# ---------------------------------------------------------------------------
# 2. Real-data population/split reproducibility (read-only, no gateway calls
#    here -- direct SQL matching the gateway's own already-audited queries,
#    to avoid this test file adding further exposure-ledger writes)
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


def test_w300_fully_exact_reconstructable_no_source_gapped():
    conn = _ro_conn()
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM ami_cvd_window_quality_v1 WHERE window_id='W300'").fetchone()[0]
        exact = conn.execute(
            "SELECT COUNT(*) FROM ami_cvd_window_quality_v1 WHERE window_id='W300' "
            "AND quality_status='EXACT_RECONSTRUCTABLE'").fetchone()[0]
    finally:
        conn.close()
    assert total == 324
    assert exact == 324


def test_bucket_exclusions_are_100_percent_short():
    conn = _ro_conn()
    try:
        rows = conn.execute(
            "SELECT direction, COUNT(*) FROM ami_cvd_bucket_exclusions GROUP BY direction").fetchall()
    finally:
        conn.close()
    assert dict(rows) == {"SHORT": 104}


def test_known_at_contract_zero_violations_for_w300():
    conn = _ro_conn()
    try:
        violations = conn.execute(
            "SELECT COUNT(*) FROM ami_cvd_windowed_flow WHERE window_id='W300' "
            "AND window_end_ts_ms > signal_birth_ts").fetchone()[0]
        not_safe = conn.execute(
            "SELECT COUNT(*) FROM ami_cvd_windowed_flow WHERE window_id='W300' "
            "AND known_at_classification != 'KNOWN_AT_SAFE'").fetchone()[0]
    finally:
        conn.close()
    assert violations == 0
    assert not_safe == 0


def test_reproduce_eligible_and_representative_population(prereg):
    """Reproduces the preregistration's population/split numbers from a
    fresh query, independent of the identity-resolution session's own
    computation, to prove the numbers are not a one-off/hand-picked result."""
    conn = _ro_conn()
    try:
        # LONG signals + cycle id + birth ts
        long_rows = conn.execute(
            "SELECT signal_id, independent_cycle_id, signal_birth_ts FROM ami_signal_lifecycle "
            "WHERE direction='LONG'").fetchall()
        # effective (corrected-preferred) swing_24h observation_status per signal
        obs_rows = conn.execute(
            "SELECT signal_id, path_definition_version, observation_status "
            "FROM ami_lifecycle_path_observations WHERE horizon_name='swing_24h'").fetchall()
    finally:
        conn.close()

    CORRECTED = "path-v2-candle-repair-r1"
    effective_status: dict[str, str] = {}
    version_by_sig: dict[str, str] = {}
    for sid, version, status in obs_rows:
        if sid not in version_by_sig or version == CORRECTED:
            effective_status[sid] = status
            version_by_sig[sid] = version

    eligible_sids = {sid for sid, status in effective_status.items() if status == "OK"}
    long_signal_ids = {sid for sid, _, _ in long_rows}
    eligible_long = eligible_sids & long_signal_ids
    assert len(eligible_long) == prereg["population"]["outcome_eligible_ok_n"] == 194

    by_cycle: dict[str, tuple[str, int]] = {}
    for sid, cyc, bts in long_rows:
        if sid not in eligible_long:
            continue
        if cyc not in by_cycle or bts < by_cycle[cyc][1]:
            by_cycle[cyc] = (sid, bts)
    assert len(by_cycle) == prereg["population"]["representative_cycle_n"] == 131

    reps = sorted(by_cycle.items(), key=lambda kv: kv[1][1])
    cut = int(len(reps) * 0.7)
    train, test = reps[:cut], reps[cut:]
    assert len(train) == prereg["population"]["train_cycle_n"] == 91
    assert len(test) == prereg["population"]["test_cycle_n"] == 40
    assert train[-1][1][1] < test[0][1][1]  # no straddling

    train_hash = hashlib.sha256(",".join(sorted(c for c, _ in train)).encode()).hexdigest()
    test_hash = hashlib.sha256(",".join(sorted(c for c, _ in test)).encode()).hexdigest()
    assert train_hash == prereg["population"]["train_cycle_set_hash_sha256"]
    assert test_hash == prereg["population"]["test_cycle_set_hash_sha256"]


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
    assert family_id == ident["canonical_family_id"] == "FAMv1:bec99d8d36f7d6a1"


def test_nullifier_reproducible_from_frozen_cycle_sets(prereg):
    # order-invariance is already covered by the M-0033 test suite; here we
    # confirm THIS preregistration's specific nullifier is reproducible from
    # its own recorded family_id/split_version and the frozen TEST cycle
    # hash's underlying set is unavailable here (only the hash is stored),
    # so this test instead confirms determinism given the SAME inputs used
    # at preregistration time round-trips to the recorded nullifier.
    ident = prereg["identity"]
    pop = prereg["population"]
    # reconstruct via the real DB (read-only) rather than storing 40 raw ids in JSON
    conn = _ro_conn()
    try:
        long_rows = conn.execute(
            "SELECT signal_id, independent_cycle_id, signal_birth_ts FROM ami_signal_lifecycle "
            "WHERE direction='LONG'").fetchall()
        obs_rows = conn.execute(
            "SELECT signal_id, path_definition_version, observation_status "
            "FROM ami_lifecycle_path_observations WHERE horizon_name='swing_24h'").fetchall()
    finally:
        conn.close()
    CORRECTED = "path-v2-candle-repair-r1"
    effective_status, version_by_sig = {}, {}
    for sid, version, status in obs_rows:
        if sid not in version_by_sig or version == CORRECTED:
            effective_status[sid] = status
            version_by_sig[sid] = version
    long_signal_ids = {sid for sid, _, _ in long_rows}
    eligible_long = {sid for sid, status in effective_status.items() if status == "OK"} & long_signal_ids
    by_cycle = {}
    for sid, cyc, bts in long_rows:
        if sid not in eligible_long:
            continue
        if cyc not in by_cycle or bts < by_cycle[cyc][1]:
            by_cycle[cyc] = (sid, bts)
    reps = sorted(by_cycle.items(), key=lambda kv: kv[1][1])
    cut = int(len(reps) * 0.7)
    test_cycle_ids = [c for c, _ in reps[cut:]]

    nullifier = gates.derive_test_nullifier(
        ident["canonical_family_id"], ident["split_version"], test_cycle_ids)
    assert nullifier == prereg["nullifier_and_gate"]["test_nullifier_sha256"] == \
        "085397f31c199c1d0c1d5ce647af4d1aa311166c63199f92872e089db8e72a7a"


def test_gate_receipt_mechanism_round_trips_on_disposable_copy(disposable_knowledge, prereg):
    kconn = sqlite3.connect(str(disposable_knowledge))
    gates.init_gates_schema(kconn)
    ident = prereg["identity"]
    receipt_hash = gates.issue_gate_receipt(
        kconn, experiment_id="TEST-COPY-" + ident["experiment_id"], canonical_family_id=ident["canonical_family_id"],
        split_version=ident["split_version"], nullifier=prereg["nullifier_and_gate"]["test_nullifier_sha256"],
        registry_result="PREREGISTERED_NOT_EXECUTED")
    kconn.commit()
    assert gates.has_gate_receipt(kconn, "TEST-COPY-" + ident["experiment_id"])
    # nullifier must NOT be consumed by issuing a receipt alone
    n = kconn.execute("SELECT COUNT(*) FROM epistemic_test_nullifiers").fetchone()[0]
    assert n == 0
    kconn.close()


def test_real_nullifier_and_receipt_state():
    """Real, read-only confirmation of the actual preregistration's effect on
    the real knowledge.sqlite: receipt present, nullifier still unconsumed."""
    conn = sqlite3.connect(f"file:{REAL_KNOWLEDGE_PATH}?mode=ro", uri=True)
    try:
        receipt = conn.execute(
            "SELECT registry_result FROM experiment_gate_receipts WHERE experiment_id=?",
            ("E-CVD-PRIMARY-LONG-W300-PREREG-001",)).fetchone()
        nullifier_rows = conn.execute(
            "SELECT COUNT(*) FROM epistemic_test_nullifiers WHERE nullifier=?",
            ("085397f31c199c1d0c1d5ce647af4d1aa311166c63199f92872e089db8e72a7a",)).fetchone()[0]
    finally:
        conn.close()
    assert receipt == ("PREREGISTERED_NOT_EXECUTED",)
    assert nullifier_rows == 0


# ---------------------------------------------------------------------------
# 4. No experiment/result was created; canonical invariants hold
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
        cvd_counts = {
            "repaired_trades": conn.execute("SELECT COUNT(*) FROM ami_agg_trades_repaired").fetchone()[0],
            "exact": conn.execute("SELECT COUNT(*) FROM ami_cvd_windowed_flow").fetchone()[0],
            "proxy": conn.execute("SELECT COUNT(*) FROM ami_cvd_windowed_flow_proxy").fetchone()[0],
        }
    finally:
        conn.close()
    assert n_reg == 22
    assert n_res == 323
    assert version == 12
    assert integrity == "ok"
    assert counts == {"ami_events": 252, "ami_signal_lifecycle": 324, "ami_cycles": 167,
                       "ami_birth_truncated_cascade_geometry": 220}
    assert cvd_counts == {"repaired_trades": 40934, "exact": 1840, "proxy": 1840}
