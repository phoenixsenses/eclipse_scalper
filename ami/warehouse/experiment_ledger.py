"""BATCH: AMI EFFECTIVE-PATH AND EXPERIMENT-IMMUTABILITY SAFETY HARDENING (GOAL B).

Single, mandatory write path for experiment_registry/experiment_results
(ami/warehouse/schema.py's Phase 6 "scientific governance" tables). Before
this module, every research wave's freeze_and_record() built its own
"INSERT ... ON CONFLICT(experiment_id) DO UPDATE SET dataset_hash=...,
completed_at=..., updated_ms=..." statement, plus a blind
"DELETE FROM experiment_results WHERE experiment_id=?" before reinserting.
That meant re-executing ANY research module's main()/freeze_and_record() a
second time -- even with a genuinely DIFFERENT population (e.g. a stale
script rerun after a canonical backfill changed what a query returns) --
would silently REPLACE an already-completed, immutable experiment's stored
content under the SAME experiment_id, with no content check at all.

New contract (operator-mandated):
  - new experiment_id                          -> INSERT
  - existing experiment_id, IDENTICAL content   -> deterministic no-op (NOOP_IDENTICAL, no write)
  - existing experiment_id, ANY content diff    -> ImmutableExperimentConflict
    ("IMMUTABLE_EXPERIMENT_CONFLICT"), refuses to write anything at all

"Content" excludes the five inherently-volatile bookkeeping timestamp columns
(preregistered_at/started_at/completed_at/created_ms/updated_ms) on
experiment_registry -- these differ between the original run and a later
byte-identical rerun purely because each call captures time.time()
independently; they are bookkeeping, not scientific content. Every other
experiment_registry column (frozen_population/frozen_features/frozen_target/
frozen_thresholds/frozen_splits/frozen_economic_gate/frozen_statistical_gate/
code_commit/dataset_hash/software_verdict/scientific_verdict/
mutation_test_count/mutation_test_passed/supersedes_experiment_id/
report_artifact_id/schema_version/provenance) IS content and must match
exactly for a no-op. For experiment_results, the full (metric_name,
metric_value) set is content (schema_version/provenance/created_ms on that
table are bookkeeping only, never compared).

No draft/non-completed experiment lifecycle exists anywhere in this codebase
today -- every current freeze_and_record() writes registry+results together
in one commit, with software_verdict already finalized at write time. This
module therefore treats every experiment_registry row as completed/immutable
from the moment it is first inserted; there is no separate mutable-draft
state to scope this guard around (verified by inspecting every
ami/research/w8_*.py freeze_and_record(), which are the only current writers
of these two tables -- none of them stage a row before finalizing it).

---

BATCH-EPISTEMIC-NULLIFIER-ENFORCEMENT-WIRING-V1 (M-0033) addition:
`register_experiment_with_gates()` below is now the single MANDATORY entry
point for any NEW research wave's freeze_and_record() to call -- it wraps
`record_experiment_registry`/`record_experiment_results` with the
BATCH-EPISTEMIC-NULLIFIER-GATES-V1 mechanism (graveyard slash-set +
TEST-evidence nullifier, `ami/governance/epistemic_gates.py`), making both
disciplines mandatory rather than advisory, in one atomic cross-database
transaction (canonical.sqlite + knowledge.sqlite via `ATTACH DATABASE`).

The two raw functions (`record_experiment_registry`/`record_experiment_results`)
are left unchanged and still directly callable -- this preserves byte-for-byte
backward compatibility for the 10 already-completed `ami/research/w8_*.py`
modules that call them directly (their experiments are immutable history;
touching them would be repository-wide cleanup, explicitly out of scope for
this batch). See the enforcement-wiring transition-proof
(`reports/governance/EPISTEMIC_NULLIFIER_ENFORCEMENT_WIRING_V1_STATE_TRANSITION_PROOF.md`)
for the full audit: 10 OLDER research modules
(candidate_universe/w1/w3/w4/w5a/w6/w6rs_confirmation/w6rs_confound_resolution/
w7a/w10a) each still hold their own inline
`INSERT INTO experiment_registry ... ON CONFLICT(experiment_id) DO UPDATE`
statement and a hardcoded EXPERIMENT_ID, bypassing BOTH this module's
immutability guard AND the new gates structurally -- this is a KNOWN,
UNCLOSED bypass surface, not touched by this batch (closing it means editing
10 historical/frozen research files, which is repository-wide cleanup).
"""
from __future__ import annotations

import sqlite3
import time

from ami.governance import epistemic_gates as _gates
from ami.research.registry import assert_no_overlap

REGISTRY_COLUMNS = (
    "experiment_id", "question_ids", "hypothesis_id", "preregistered_at",
    "frozen_population", "frozen_features", "frozen_target", "frozen_thresholds", "frozen_splits",
    "frozen_economic_gate", "frozen_statistical_gate", "code_commit", "dataset_hash", "started_at",
    "completed_at", "software_verdict", "scientific_verdict", "mutation_test_count", "mutation_test_passed",
    "supersedes_experiment_id", "report_artifact_id", "schema_version", "provenance", "created_ms", "updated_ms",
)

# bookkeeping timestamps: allowed to differ across runs, never part of the content-equality check,
# never rewritten on a no-op (the stored row's own original timestamps are left untouched)
_VOLATILE_BOOKKEEPING_COLUMNS = frozenset(
    {"preregistered_at", "started_at", "completed_at", "created_ms", "updated_ms"}
)
_REGISTRY_CONTENT_COLUMNS = tuple(
    c for c in REGISTRY_COLUMNS if c not in _VOLATILE_BOOKKEEPING_COLUMNS and c != "experiment_id"
)


class ImmutableExperimentConflict(Exception):
    """Raised when a write would change an already-completed experiment's
    stored registry/result content. See module docstring for the exact
    fail-closed contract -- never overwrites in place; a revised or
    corrected-data computation must use a new experiment_id instead."""


def record_experiment_registry(conn, values: dict) -> str:
    """`values` must contain every key in REGISTRY_COLUMNS. Returns
    'INSERTED' or 'NOOP_IDENTICAL'. Raises ImmutableExperimentConflict on any
    content difference -- never overwrites an existing row in place (no
    ON CONFLICT DO UPDATE anywhere in this function)."""
    missing = set(REGISTRY_COLUMNS) - set(values)
    if missing:
        raise ValueError(f"record_experiment_registry: missing required column(s): {sorted(missing)}")
    experiment_id = values["experiment_id"]
    existing = conn.execute(
        f"SELECT {', '.join(REGISTRY_COLUMNS)} FROM experiment_registry WHERE experiment_id=?",
        (experiment_id,),
    ).fetchone()
    if existing is None:
        conn.execute(
            f"INSERT INTO experiment_registry ({', '.join(REGISTRY_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in REGISTRY_COLUMNS)})",
            [values[c] for c in REGISTRY_COLUMNS],
        )
        return "INSERTED"

    existing_map = dict(zip(REGISTRY_COLUMNS, existing))
    diffs = {
        c: (existing_map[c], values[c])
        for c in _REGISTRY_CONTENT_COLUMNS
        if existing_map[c] != values[c]
    }
    if diffs:
        raise ImmutableExperimentConflict(
            f"IMMUTABLE_EXPERIMENT_CONFLICT: experiment_registry row for experiment_id={experiment_id!r} "
            f"already exists with different content in column(s) {sorted(diffs)}. A completed "
            "experiment's registry row is immutable -- a corrected-data or revised rerun must use a "
            f"NEW experiment_id (e.g. set supersedes_experiment_id={experiment_id!r} on the new one), "
            f"never overwrite this one in place. Diffs: {diffs}"
        )
    return "NOOP_IDENTICAL"


def record_experiment_results(conn, experiment_id: str, results: list[tuple[str, str]],
                               schema_version: int, provenance: str, created_ms: int) -> str:
    """`results`: list of (metric_name, metric_value) pairs (metric_value
    already str()-ified by the caller, matching the existing convention).
    Returns 'INSERTED' or 'NOOP_IDENTICAL'. Raises ImmutableExperimentConflict
    if the already-stored (metric_name, metric_value) set differs from the
    proposed one (compared as a set -- row order/result_id/schema_version/
    provenance/created_ms are bookkeeping, never part of the content check).
    Never deletes existing rows (no DELETE-then-reinsert anywhere here)."""
    existing_rows = conn.execute(
        "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=?",
        (experiment_id,),
    ).fetchall()
    existing_set = {(r[0], r[1]) for r in existing_rows}
    proposed_set = {(name, value) for name, value in results}

    if not existing_rows:
        for name, value in results:
            conn.execute(
                "INSERT INTO experiment_results (experiment_id, metric_name, metric_value, schema_version, "
                "provenance, created_ms) VALUES (?,?,?,?,?,?)",
                (experiment_id, name, value, schema_version, provenance, created_ms),
            )
        return "INSERTED"

    if existing_set != proposed_set:
        missing_in_stored = sorted(proposed_set - existing_set)[:5]
        extra_in_stored = sorted(existing_set - proposed_set)[:5]
        raise ImmutableExperimentConflict(
            f"IMMUTABLE_EXPERIMENT_CONFLICT: experiment_results for experiment_id={experiment_id!r} "
            f"already exist and differ from the proposed write. missing_in_stored (up to 5 shown)="
            f"{missing_in_stored} extra_in_stored (up to 5 shown)={extra_in_stored}. A completed "
            "experiment's results are immutable -- use a NEW experiment_id for any revised computation."
        )
    return "NOOP_IDENTICAL"


# ---------------------------------------------------------------------------
# BATCH-EPISTEMIC-NULLIFIER-ENFORCEMENT-WIRING-V1 (M-0033)
# ---------------------------------------------------------------------------

def _audit_kb(conn, action: str, detail: str) -> None:
    """Same shape as epistemic_gates._audit -- duplicated locally rather than
    importing that leading-underscore helper across module boundaries."""
    try:
        conn.execute(
            "INSERT INTO audit_log (ts_ms, actor, action, knowledge_id, detail) VALUES (?,?,?,?,?)",
            (int(time.time() * 1000), "experiment_ledger", action, None, detail[:400]))
    except Exception:
        pass


def register_experiment_with_gates(
    canonical_conn,
    *,
    registry_values: dict,
    results: list[tuple[str, str]],
    results_schema_version: int,
    results_provenance: str,
    results_created_ms: int,
    test_cycle_ids,
    train_cycle_ids=None,
    knowledge_db_path=None,
    retry_token: str | None = None,
    supersession_token: str | None = None,
) -> dict:
    """Mandatory gated entry point (REQUIRED ENFORCEMENT ORDER, steps 1-11):

    1. resolve canonical family identity
    2. evaluate graveyard slash-set gate
    3. validate retry authorization if blocked
    4. resolve frozen split version
    5. resolve the ordered TEST independent-cycle set (+ TRAIN/TEST leakage
       check, reusing `ami.research.registry.assert_no_overlap`, when the
       caller supplies `train_cycle_ids`)
    6. compute the TEST-evidence nullifier
    7. check prior nullifier consumption
    8. validate supersession authorization if required
    9. persist gate decision + audit record
    10. only then permit experiment freeze/registration
    11. only after successful experiment registration, mark the nullifier
        (and any authorization token used) consumed

    Everything from the ATTACH onward is ONE transaction on `canonical_conn`
    (knowledge.sqlite is ATTACHed onto it as schema `kb`) -- a single final
    `commit()` covers canonical.sqlite's `experiment_registry`/
    `experiment_results` AND knowledge.sqlite's audit/nullifier/authorization
    tables together. Any exception before that commit triggers `rollback()`,
    so a blocked/failed attempt writes nothing to either database and
    consumes no nullifier and no authorization token (invariants required by
    the enforcement-wiring batch). `canonical_conn` must already be open
    (e.g. via `ami.warehouse.schema.connect()`); this function does not open
    or close it, only attaches/detaches `kb` on it.
    """
    if knowledge_db_path is None:
        from ami.knowledge.store import DEFAULT_PATH as knowledge_db_path  # noqa: PLC0415

    # Schema setup is idempotent infra, not part of the atomic transaction,
    # and MUST run on a direct connection: unqualified CREATE TABLE always
    # targets the connection's `main` schema, so running it through the
    # attached connection below would (incorrectly) try to create these
    # tables inside canonical.sqlite.
    _direct_kb = sqlite3.connect(str(knowledge_db_path))
    try:
        _gates.init_gates_schema(_direct_kb)
    finally:
        _direct_kb.close()

    experiment_id = registry_values["experiment_id"]
    question_ids = registry_values.get("question_ids", "")
    hypothesis_id = registry_values.get("hypothesis_id", "")
    frozen_splits = registry_values.get("frozen_splits", "")

    # step 1
    family_id = _gates.resolve_canonical_family_id(question_ids, hypothesis_id)
    # step 4
    split_version = _gates.resolve_split_version(frozen_splits)
    # step 5 (leakage check before anything else touches the DB)
    if train_cycle_ids is not None:
        assert_no_overlap(set(str(c) for c in train_cycle_ids), set(str(c) for c in test_cycle_ids))
    _, test_set_hash, test_set_n = _gates._normalized_test_set_hash(test_cycle_ids)  # noqa: SLF001
    # step 6
    nullifier = _gates.derive_test_nullifier(family_id, split_version, test_cycle_ids)

    already_attached = any(
        r[1] == "kb" for r in canonical_conn.execute("PRAGMA database_list").fetchall())
    if not already_attached:
        canonical_conn.execute("ATTACH DATABASE ? AS kb", (str(knowledge_db_path),))
    attached_by_us = not already_attached

    try:
        spec_text = " | ".join(str(x) for x in (
            hypothesis_id, question_ids,
            registry_values.get("frozen_population", ""),
            registry_values.get("frozen_features", ""),
            registry_values.get("frozen_target", ""),
            registry_values.get("frozen_thresholds", ""),
        ))

        # steps 2-3: graveyard gate, validating any retry token BEFORE it is
        # ever handed to assert_not_graveyard (which -- for backward
        # compatibility with the V1 seed batch -- accepts any non-empty
        # string; the real validation happens here, first).
        hits = _gates.match_graveyard(canonical_conn, spec_text)
        retry_auth_id = None
        if hits:
            if retry_token is None:
                raise _gates.GraveyardRetestBlocked(
                    f"GRAVEYARD_RETEST_BLOCKED: family {family_id!r} matches "
                    f"{len(hits)} slashed famil{'y' if len(hits) == 1 else 'ies'}; no retry_token supplied.")
            retry_auth_id = _gates._reserve_authorization(  # noqa: SLF001
                canonical_conn, raw_token=retry_token, authorization_type="RETRY",
                canonical_family_id=family_id)
        _gates.assert_not_graveyard(canonical_conn, spec_text, retry_token=retry_token, _autocommit=False)

        # steps 7-8: prior nullifier consumption + supersession validation
        prior = canonical_conn.execute(
            "SELECT consumed_by_experiment_id FROM epistemic_test_nullifiers"
            " WHERE nullifier=? ORDER BY consumed_ms", (nullifier,)).fetchall()
        is_rerun_of_self = any(p[0] == experiment_id for p in prior)
        supersession_auth_id = None
        if prior and not is_rerun_of_self:
            if supersession_token is None:
                raise _gates.TestEvidenceReuseBlocked(
                    f"TEST_EVIDENCE_REUSE_BLOCKED: family {family_id!r} already consumed this TEST"
                    f" set (split={split_version}); no supersession_token supplied.")
            supersession_auth_id = _gates._reserve_authorization(  # noqa: SLF001
                canonical_conn, raw_token=supersession_token, authorization_type="SUPERSESSION",
                canonical_family_id=family_id, split_version=split_version, test_set_hash=test_set_hash)

        # step 9: persist the gate decision itself, before permitting registration
        _audit_kb(canonical_conn, "EXPERIMENT_GATE_DECISION", str({
            "experiment_id": experiment_id, "family_id": family_id, "split_version": split_version,
            "nullifier": nullifier, "test_set_n": test_set_n,
            "graveyard_hits": [h["family_label"] for h in hits],
            "retry_auth_used": bool(retry_auth_id), "supersession_auth_used": bool(supersession_auth_id),
        }))

        # step 10: permit registration
        registry_result = record_experiment_registry(canonical_conn, registry_values)
        results_result = record_experiment_results(
            canonical_conn, experiment_id, results, schema_version=results_schema_version,
            provenance=results_provenance, created_ms=results_created_ms)

        # step 11: only now mark the nullifier (and any tokens) consumed
        nullifier_result = _gates.consume_test_evidence(
            canonical_conn, family_id=family_id, split_version=split_version,
            test_cycle_ids=test_cycle_ids, experiment_id=experiment_id,
            supersession_token=supersession_token, _autocommit=False)
        if retry_auth_id:
            _gates._finalize_authorization_consumption(  # noqa: SLF001
                canonical_conn, authorization_id=retry_auth_id, resulting_experiment_id=experiment_id)
        if supersession_auth_id:
            _gates._finalize_authorization_consumption(  # noqa: SLF001
                canonical_conn, authorization_id=supersession_auth_id, resulting_experiment_id=experiment_id)

        canonical_conn.commit()
        return {
            "experiment_id": experiment_id, "family_id": family_id, "split_version": split_version,
            "nullifier": nullifier, "registry_result": registry_result, "results_result": results_result,
            "nullifier_result": nullifier_result,
        }
    except BaseException:
        canonical_conn.rollback()
        raise
    finally:
        if attached_by_us:
            canonical_conn.execute("DETACH DATABASE kb")
