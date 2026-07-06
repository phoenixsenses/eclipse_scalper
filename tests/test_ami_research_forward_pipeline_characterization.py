"""PHASE 7A-0: characterization + safety-invariant closure tests for
ami/research/forward_pipeline.py (ForwardEvidencePipeline -- already LIVE
with 2 real frozen bindings, previously without dedicated test coverage).

NO_PRODUCTION_CODE_CHANGE: every test below exercises EXISTING, UNMODIFIED
code. Findings are separated into CURRENT_BEHAVIOR_CHARACTERIZED vs
REQUIRED_SAFETY_INVARIANT_MET/NOT_MET -- see SYSTEM_STATE.md Phase 7A-0
report for the consolidated verdict. NO_NEW_BINDING against the real system:
every ForwardEvidencePipeline/ResearchRegistry/KnowledgeStore instance here
is backed by a tmp_path fixture file, never the real data/ami/*.sqlite.

The two real, live bindings (E-HOUR17-FWD-001 / E-CONVCOMP-FWD-001) are
checked with a SEPARATE, read-only (mode=ro) test at the bottom that only
locks their immutable identity fields (spec_hash/frozen_ms/dataset_hash/
candidate_version/signal/knowledge_id) -- NOT forward_n, which is expected
to grow over time via legitimate cron operation and would make a hardcoded
assertion misleadingly fail on healthy operation.

Run: pytest tests/test_ami_research_forward_pipeline_characterization.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import ast
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from ami.enums import ClaimType, KnowledgeStatus
from ami.governance.governor import EpistemicGovernor
from ami.knowledge.objects import KnowledgeObject, Provenance
from ami.knowledge.store import KnowledgeStore
from ami.research.forward_pipeline import ForwardEvidencePipeline
from ami.research.registry import ExperimentSpec, ResearchRegistry

PIPELINE_SRC_PATH = Path(__file__).resolve().parents[1] / "ami" / "research" / "forward_pipeline.py"
_FORBIDDEN_IMPORT_PREFIXES = ("execution", "risk", "brain")

EXP_ID = "E-TEST"
KO_ID = "K-TEST"
SIGNAL = "TEST_SIGNAL"
EXEC_MODEL = "mark_fill_fee5bps"
DATASET_HASH = "test-dataset-hash"


def _write_ledger(path, trades: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in trades:
            row = {"event": "CLOSE", "signal": SIGNAL, **t}
            f.write(json.dumps(row) + "\n")


def _mk_ko(knowledge_id: str = KO_ID, version: int = 1) -> KnowledgeObject:
    prov = Provenance(source_tables=["test_table"], data_time_range="2026-01-01..2026-01-02",
                       dataset_hash=DATASET_HASH, code_ref="tests/fixture", execution_model=EXEC_MODEL)
    ko = KnowledgeObject(knowledge_id=knowledge_id, claim="test claim", claim_type=ClaimType.PREDICTIVE,
                         status=KnowledgeStatus.FORWARD_VALIDATING, provenance=prov,
                         falsification=["avg_bps<=0 at n>=20"])
    ko.version = version
    return ko


@dataclass
class _Env:
    store: KnowledgeStore
    reg: ResearchRegistry
    gov: EpistemicGovernor
    pipe: ForwardEvidencePipeline
    store_path: Path
    reg_path: Path
    ledger_path: Path


def _build_env(tmp_path: Path, frozen_ms: int, trades: list[dict],
                experiment_id: str = EXP_ID, knowledge_id: str = KO_ID, min_conviction=None) -> _Env:
    tmp_path.mkdir(parents=True, exist_ok=True)
    store_path = tmp_path / "knowledge.sqlite"
    reg_path = tmp_path / "research.sqlite"
    ledger_path = tmp_path / "shadow.jsonl"
    _write_ledger(ledger_path, trades)

    store = KnowledgeStore(path=store_path)
    ko = _mk_ko(knowledge_id)
    store.put(ko, actor="test")

    reg = ResearchRegistry(path=reg_path, knowledge_path=store_path)
    spec = ExperimentSpec(
        experiment_id=experiment_id, question_id="Q-TEST", population="test population",
        target="net_bps", features=["route_gate"], threshold_method="frozen(route gate)",
        untouched_data="all shadow CLOSEs after freeze", negative_control="pre-freeze rejected",
        min_sample=1, decision_criteria="avg net > 0", falsification_rule="avg net <= 0",
        execution_model=EXEC_MODEL)
    spec.freeze()
    # BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1: direct gate receipt
    # in the same disposable knowledge.sqlite -- this fixture characterizes
    # ForwardEvidencePipeline behavior, not the gate itself.
    from ami.governance import epistemic_gates as _gates
    _gates.init_gates_schema(store.conn)
    _gates.issue_gate_receipt(store.conn, experiment_id=experiment_id,
                              canonical_family_id="FORWARD_PIPELINE_TEST_HARNESS",
                              split_version=None, nullifier=None, registry_result="TEST_HARNESS_DIRECT_RECEIPT")
    store.conn.commit()
    reg.register_experiment(spec)
    # Direct forward_bindings insert (NOT pipe.bind(), which stamps frozen_ms=now())
    # so the boundary tests below can control frozen_ms exactly -- this is a
    # tmp_path fixture DB, never the real system (NO_NEW_BINDING is about the
    # real store, not about constructing disposable test fixtures).
    reg.conn.execute(
        "INSERT INTO forward_bindings (experiment_id, knowledge_id, signal, min_conviction, spec_hash, "
        "frozen_ms, dataset_hash, code_ref, execution_model, candidate_version) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (experiment_id, knowledge_id, SIGNAL, min_conviction, spec.frozen_hash, frozen_ms,
         DATASET_HASH, "tests/fixture", EXEC_MODEL, ko.version),
    )
    reg.conn.commit()

    gov = EpistemicGovernor(store)
    pipe = ForwardEvidencePipeline(store, reg, gov, ledger_path=ledger_path)
    return _Env(store, reg, gov, pipe, store_path, reg_path, ledger_path)


def _reopen(env: _Env) -> _Env:
    """Simulate crash/restart: fresh handles against the SAME on-disk files."""
    store2 = KnowledgeStore(path=env.store_path)
    reg2 = ResearchRegistry(path=env.reg_path)
    gov2 = EpistemicGovernor(store2)
    pipe2 = ForwardEvidencePipeline(store2, reg2, gov2, ledger_path=env.ledger_path)
    return _Env(store2, reg2, gov2, pipe2, env.store_path, env.reg_path, env.ledger_path)


# ---- deterministic signal identity ----

def test_trade_dedup_key_is_deterministic(tmp_path):
    env = _build_env(tmp_path, frozen_ms=1000, trades=[
        {"id": "t1", "entry_ts_ms": 1100, "net_bps": 10.0},
    ])
    r1 = env.pipe.run_once()
    env2 = _reopen(env)
    r2 = env2.pipe.run_once()
    assert r1["bindings"][0]["accepted"] == 1
    assert r2["bindings"][0]["accepted"] == 0
    assert r2["bindings"][0]["duplicates"] == 1


# ---- restart idempotency ----

def test_restart_idempotency_same_pipe_instance(tmp_path):
    env = _build_env(tmp_path, frozen_ms=1000, trades=[
        {"id": "t1", "entry_ts_ms": 1100, "net_bps": 10.0},
        {"id": "t2", "entry_ts_ms": 900, "net_bps": 5.0},  # PRE_FREEZE
    ])
    r1 = env.pipe.run_once()
    r2 = env.pipe.run_once()
    assert r1["bindings"][0]["accepted"] == 1
    assert r1["bindings"][0]["rejected"] == 1
    assert r2["bindings"][0]["accepted"] == 0
    assert r2["bindings"][0]["rejected"] == 0
    assert r2["bindings"][0]["duplicates"] == 2


# ---- duplicate transition suppression ----

def test_duplicate_trade_processed_only_once(tmp_path):
    env = _build_env(tmp_path, frozen_ms=1000, trades=[
        {"id": "t1", "entry_ts_ms": 1100, "net_bps": 10.0},
    ])
    env.pipe.run_once()
    env.pipe.run_once()
    n_rows = env.reg.conn.execute(
        "SELECT COUNT(*) FROM processed_trades WHERE experiment_id=? AND trade_id='t1'", (EXP_ID,)
    ).fetchone()[0]
    assert n_rows == 1


# ---- append-only enforcement ----

def test_no_delete_or_update_statements_in_source():
    src = PIPELINE_SRC_PATH.read_text(encoding="utf-8")
    for forbidden in ("DELETE FROM processed_trades", "DELETE FROM forward_bindings",
                      "UPDATE processed_trades", "UPDATE forward_bindings"):
        assert forbidden not in src, f"found forbidden statement: {forbidden}"


def test_processed_trades_row_count_never_shrinks_across_runs(tmp_path):
    env = _build_env(tmp_path, frozen_ms=1000, trades=[
        {"id": "t1", "entry_ts_ms": 1100, "net_bps": 10.0},
        {"id": "t2", "entry_ts_ms": 900, "net_bps": 5.0},
    ])
    env.pipe.run_once()
    n1 = env.reg.conn.execute("SELECT COUNT(*) FROM processed_trades").fetchone()[0]
    env.pipe.run_once()
    n2 = env.reg.conn.execute("SELECT COUNT(*) FROM processed_trades").fetchone()[0]
    assert n2 == n1 == 2


# ---- historical replay does not increase FORWARD_N ----

def test_historical_replay_does_not_increase_forward_n(tmp_path):
    env = _build_env(tmp_path, frozen_ms=1_000_000, trades=[
        {"id": "t-old-1", "entry_ts_ms": 100, "net_bps": 50.0},
        {"id": "t-old-2", "entry_ts_ms": 500, "net_bps": 50.0},
    ])
    env.pipe.run_once()
    stats = env.pipe._forward_stats(EXP_ID)
    assert stats.get("n", 0) == 0


def test_pre_freeze_trades_never_get_evidence_attached(tmp_path):
    env = _build_env(tmp_path, frozen_ms=1000, trades=[
        {"id": "t-hist", "entry_ts_ms": 500, "net_bps": 999.0},
    ])
    env.pipe.run_once()
    n_evidence = env.reg.conn.execute(
        "SELECT COUNT(*) FROM evidence WHERE experiment_id=?", (EXP_ID,)
    ).fetchone()[0]
    assert n_evidence == 0


# ---- activation boundary (characterization vs operator's preferred contract) ----

def test_activation_boundary_characterization(tmp_path):
    frozen_ms = 1000
    env = _build_env(tmp_path, frozen_ms=frozen_ms, trades=[
        {"id": "t-before", "entry_ts_ms": frozen_ms - 1, "net_bps": 5.0},
        {"id": "t-at", "entry_ts_ms": frozen_ms, "net_bps": 5.0},
        {"id": "t-after", "entry_ts_ms": frozen_ms + 1, "net_bps": 5.0},
    ])
    env.pipe.run_once()
    accepted = dict(env.reg.conn.execute(
        "SELECT trade_id, accepted FROM processed_trades WHERE experiment_id=?", (EXP_ID,)
    ).fetchall())
    reasons = dict(env.reg.conn.execute(
        "SELECT trade_id, reject_reason FROM processed_trades WHERE experiment_id=?", (EXP_ID,)
    ).fetchall())

    # CURRENT_BEHAVIOR_CHARACTERIZED: code uses `entry <= frozen_ms -> PRE_FREEZE`.
    assert accepted["t-before"] == 0 and reasons["t-before"] == "PRE_FREEZE"
    assert accepted["t-at"] == 0 and reasons["t-at"] == "PRE_FREEZE"   # <-- the tie
    assert accepted["t-after"] == 1

    # DOCUMENTED DISCREPANCY vs operator's preferred contract:
    #   event_ts <  activation_ts -> HISTORICAL_REPLAY    (code AGREES)
    #   event_ts >= activation_ts -> FORWARD_OBSERVATION  (code DISAGREES at the
    #       exact tie: entry_ts_ms == frozen_ms is rejected as PRE_FREEZE here,
    #       instead of being accepted as forward per the preferred contract).
    # This is the STRICTER direction (a tie is rejected, never wrongly counted
    # as forward) -- the core no-lookahead invariant ("nothing with
    # event_ts < activation_ts is ever accepted as forward") still holds.
    # See SYSTEM_STATE.md Phase 7A-0 report: REQUIRED_SAFETY_INVARIANT_MET for
    # the no-lookahead guarantee; boundary-tie CONVENTION mismatch reported
    # separately, not silently changed (no production code touched).


# ---- crash before commit ----

def test_crash_before_commit_uncommitted_row_is_rolled_back(tmp_path):
    db_path = tmp_path / "research.sqlite"
    reg = ResearchRegistry(path=db_path)
    reg.conn.execute(
        "INSERT INTO processed_trades (experiment_id, trade_id, entry_ts_ms, net_bps, accepted, "
        "reject_reason, processed_ms) VALUES (?,?,?,?,?,?,?)",
        (EXP_ID, "crash-trade", 500, None, 0, "PRE_FREEZE", 123),
    )
    reg.conn.close()  # simulated crash: connection dies WITHOUT commit

    reg2 = ResearchRegistry(path=db_path)
    row = reg2.conn.execute(
        "SELECT 1 FROM processed_trades WHERE experiment_id=? AND trade_id=?", (EXP_ID, "crash-trade")
    ).fetchone()
    reg2.close()
    assert row is None  # uncommitted insert did not survive -> safely reprocessable


# ---- crash after commit: commit -> crash -> restart -> replay -> 0 dup rows, FORWARD_N unchanged ----

def test_crash_after_commit_restart_replay_is_idempotent(tmp_path):
    env = _build_env(tmp_path, frozen_ms=1000, trades=[
        {"id": "t1", "entry_ts_ms": 1100, "net_bps": 30.0},
        {"id": "t2", "entry_ts_ms": 900, "net_bps": 5.0},
    ])
    env.pipe.run_once()  # commits (accepted trade's attach_evidence commits; end-of-binding commit too)
    n_before = env.reg.conn.execute("SELECT COUNT(*) FROM processed_trades").fetchone()[0]
    n_forward_before = env.pipe._forward_stats(EXP_ID).get("n", 0)

    env.store.close()
    env.reg.close()  # simulated crash + process exit

    env2 = _reopen(env)  # restart: fresh instances, same on-disk files
    report2 = env2.pipe.run_once()  # same ledger replayed
    n_after = env2.reg.conn.execute("SELECT COUNT(*) FROM processed_trades").fetchone()[0]
    n_forward_after = env2.pipe._forward_stats(EXP_ID).get("n", 0)

    assert n_after == n_before  # zero duplicate rows
    assert n_forward_after == n_forward_before  # FORWARD_N unchanged
    assert report2["bindings"][0]["accepted"] == 0
    assert report2["bindings"][0]["rejected"] == 0
    assert report2["bindings"][0]["duplicates"] == 2


# ---- partial-batch recovery (characterization: commit granularity) ----

def test_partial_batch_commit_granularity_characterization(tmp_path):
    # CURRENT_BEHAVIOR_CHARACTERIZED: an ACCEPTED trade's processed_trades row
    # and its evidence row are committed TOGETHER, immediately, because
    # ResearchRegistry.attach_evidence() calls self.conn.commit() internally
    # and shares the same connection as the processed_trades INSERT that
    # precedes it in run_once()'s loop. A REJECTED/duplicate trade's
    # processed_trades row is NOT committed until either a later accepted
    # trade in the same binding's loop (via attach_evidence) or the final
    # per-binding self.reg.conn.commit() fires. This test proves the
    # ACCEPTED-trade atomicity half of that claim directly.
    env = _build_env(tmp_path, frozen_ms=1000, trades=[
        {"id": "t1", "entry_ts_ms": 1100, "net_bps": 10.0},
    ])
    env.pipe.run_once()
    # both rows must exist together (atomicity), on a fresh connection (so we
    # are reading committed state, not the in-memory connection's own view)
    reg2 = ResearchRegistry(path=env.reg_path)
    n_processed = reg2.conn.execute(
        "SELECT COUNT(*) FROM processed_trades WHERE experiment_id=? AND trade_id='t1'", (EXP_ID,)
    ).fetchone()[0]
    n_evidence = reg2.conn.execute(
        "SELECT COUNT(*) FROM evidence WHERE experiment_id=?", (EXP_ID,)
    ).fetchone()[0]
    reg2.close()
    assert n_processed == 1
    assert n_evidence == 1


# ---- late and out-of-order observations ----

def test_out_of_order_ledger_lines_do_not_change_outcome(tmp_path):
    trades = [
        {"id": "t1", "entry_ts_ms": 1100, "net_bps": 10.0},
        {"id": "t2", "entry_ts_ms": 1200, "net_bps": -5.0},
    ]
    env_in_order = _build_env(tmp_path / "a", frozen_ms=1000, trades=trades)
    env_shuffled = _build_env(tmp_path / "b", frozen_ms=1000, trades=list(reversed(trades)))
    r1 = env_in_order.pipe.run_once()
    r2 = env_shuffled.pipe.run_once()
    assert r1["bindings"][0]["accepted"] == r2["bindings"][0]["accepted"] == 2


# ---- no import of order router/executor/position manager ----

def test_no_execution_risk_brain_import():
    tree = ast.parse(PIPELINE_SRC_PATH.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[0] in _FORBIDDEN_IMPORT_PREFIXES]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in _FORBIDDEN_IMPORT_PREFIXES:
                found.append(node.module)
    assert found == []


# ---- no trading credential requirement ----

def test_no_credential_or_env_access():
    src = PIPELINE_SRC_PATH.read_text(encoding="utf-8")
    assert "os.environ" not in src
    assert "getenv" not in src
    assert "API_KEY" not in src.upper()


# ---- mutation test proving no order creation ----

def test_authorize_is_never_called_by_forward_pipeline(monkeypatch, tmp_path):
    env = _build_env(tmp_path, frozen_ms=1000, trades=[
        {"id": "t1", "entry_ts_ms": 1100, "net_bps": 500.0},  # large win, could tempt a promotion path
    ])
    calls = []
    original = EpistemicGovernor.authorize

    def spy(self, *a, **kw):
        calls.append((a, kw))
        return original(self, *a, **kw)

    monkeypatch.setattr(EpistemicGovernor, "authorize", spy)
    env.pipe.run_once()
    assert calls == []


def test_promote_never_grants_live_or_sizing_permission(tmp_path):
    # characterization + mutation guard: governor.promote() (the only
    # permission-adjacent call forward_pipeline ever makes, via
    # _governor_review) must never add LIVE_ALLOWED/SIZING_ALLOWED to a KO's
    # permitted list -- only authorize() grants permissions, and it is never
    # called by this module (see test above).
    from ami.enums import Permission

    env = _build_env(tmp_path, frozen_ms=1000, trades=[
        {"id": f"t{i}", "entry_ts_ms": 1100 + i, "net_bps": 50.0} for i in range(25)
    ])
    env.pipe.run_once()
    ko = env.store.get(KO_ID)
    assert Permission.LIVE_ALLOWED not in ko.permitted
    assert Permission.SIZING_ALLOWED not in ko.permitted


# ---- existing two real bindings remain behaviorally unchanged (READ-ONLY) ----

_REAL_BINDINGS_EXPECTED = {
    "E-HOUR17-FWD-001": {
        "knowledge_id": "K-S34-HOUR17-001", "signal": "LONG_HOUR17_HOLD6H",
        "spec_hash": "62861d5c6bf98581", "frozen_ms": 1783023164493,
        "dataset_hash": "s34-2026H1", "candidate_version": 1,
    },
    "E-CONVCOMP-FWD-001": {
        "knowledge_id": "K-S34-MECH-COMPOSITE-001", "signal": "LONG_HOUR17_COMPOSITE",
        "spec_hash": "15d4fc5c2c1a2038", "frozen_ms": 1783023164510,
        "dataset_hash": "s34-2026H1", "candidate_version": 1,
    },
}


def test_real_bindings_identity_fields_unchanged_read_only():
    from ami.research.registry import DEFAULT_PATH

    conn = sqlite3.connect(f"file:{DEFAULT_PATH}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT experiment_id, knowledge_id, signal, spec_hash, frozen_ms, dataset_hash, "
            "candidate_version FROM forward_bindings"
        ).fetchall()
    finally:
        conn.close()
    by_id = {r[0]: r for r in rows}
    assert set(by_id) == set(_REAL_BINDINGS_EXPECTED)
    for exp_id, expected in _REAL_BINDINGS_EXPECTED.items():
        _, knowledge_id, signal, spec_hash, frozen_ms, dataset_hash, candidate_version = by_id[exp_id]
        assert knowledge_id == expected["knowledge_id"]
        assert signal == expected["signal"]
        assert spec_hash == expected["spec_hash"]
        assert frozen_ms == expected["frozen_ms"]
        assert dataset_hash == expected["dataset_hash"]
        assert candidate_version == expected["candidate_version"]
