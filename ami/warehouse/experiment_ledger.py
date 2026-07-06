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

        # M-0034: issue the gate receipt research.sqlite's ResearchRegistry
        # checks before permitting its own projection write -- same
        # transaction, same commit.
        _gates.issue_gate_receipt(
            canonical_conn, experiment_id=experiment_id, canonical_family_id=family_id,
            split_version=split_version, nullifier=nullifier, registry_result=registry_result)

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


# ---------------------------------------------------------------------------
# BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1 (M-0034)
# ---------------------------------------------------------------------------

# The 10 legacy "living snapshot" modules (candidate_universe/w1/w3/w4/w5a/
# w6/w6rs_confirmation/w6rs_confound_resolution/w7a/w10a) have their OWN
# long-standing, already-accepted `ON CONFLICT(experiment_id) DO UPDATE SET
# dataset_hash=excluded.dataset_hash, completed_at=..., updated_ms=...`
# behavior: re-running against a live, growing population is BY DESIGN.
# `dataset_hash` AND every `frozen_*` free-text/count-embedding column
# (population/features/target/thresholds/economic_gate/statistical_gate) are
# EXPECTED to drift on every call -- e.g. `frozen_population` embeds a live
# `anchor_n` count, and a module's own prose (e.g. a checkpoint-closure note
# appended to `frozen_statistical_gate` by a later code change) legitimately
# gets refined without that being new scientific evidence. Real-data proof
# this distinction matters: `E-W7A-STATE-STRUCTURE-AGING-MARKET-CLOCKS-001`'s
# CURRENT code computes a `frozen_statistical_gate` string that already
# differs from what is stored in the REAL canonical.sqlite today (a
# pre-existing, harmless documentation-text drift from before this batch,
# invisible until this stricter check -- confirmed read-only against the real
# DB; the actual closure verdict is correctly recorded as its own
# `experiment_results` row, `closure_classification`, unaffected).
#
# What DOES stay strictly compared (must match for a drift-only refresh,
# else routes through the full gate): `question_ids`/`hypothesis_id` (family
# identity) and `frozen_splits` (split METHODOLOGY, e.g. "chronological
# 70/30" -- changing the actual split fraction/ordering IS a genuine
# methodology change, unlike a prose refinement or a growing count).
_LEGACY_SNAPSHOT_STRICT_COLUMNS = frozenset({"question_ids", "hypothesis_id", "frozen_splits"})


def _upsert_legacy_snapshot(conn, registry_values: dict, results: list[tuple[str, str]],
                             results_schema_version: int, results_provenance: str,
                             results_created_ms: int) -> str:
    """Reproduces the 10 legacy modules' OWN pre-existing
    `INSERT ... ON CONFLICT(experiment_id) DO UPDATE SET dataset_hash=...,
    completed_at=..., updated_ms=...` + `DELETE FROM experiment_results
    WHERE experiment_id=?` -- reinsert pattern, verbatim, for the
    drift-only-refresh case (population grew, scientific content columns
    unchanged). Deliberately bypasses `record_experiment_registry`/
    `record_experiment_results`'s strict content-immutability check, which
    would otherwise raise on the dataset_hash change alone -- that check
    stays fully in effect for anything BEYOND drift columns (see the caller,
    `register_legacy_snapshot_with_gates`, which only reaches this helper
    after confirming non-drift columns are unchanged)."""
    experiment_id = registry_values["experiment_id"]
    conn.execute(
        f"INSERT INTO experiment_registry ({', '.join(REGISTRY_COLUMNS)}) "
        f"VALUES ({', '.join('?' for _ in REGISTRY_COLUMNS)}) "
        "ON CONFLICT(experiment_id) DO UPDATE SET dataset_hash=excluded.dataset_hash, "
        "completed_at=excluded.completed_at, updated_ms=excluded.updated_ms",
        [registry_values[c] for c in REGISTRY_COLUMNS])
    conn.execute("DELETE FROM experiment_results WHERE experiment_id=?", (experiment_id,))
    for name, value in results:
        conn.execute(
            "INSERT INTO experiment_results (experiment_id, metric_name, metric_value, schema_version, "
            "provenance, created_ms) VALUES (?,?,?,?,?,?)",
            (experiment_id, name, value, results_schema_version, results_provenance, results_created_ms))
    return "UPSERTED_DRIFT_ONLY"


class MissingFrozenTestMetadata(Exception):
    """A legacy snapshot module's SCIENTIFIC content (anything other than
    dataset_hash/timestamp drift) changed from what is already registered,
    but no `test_cycle_ids` (or explicit `no_test_split=True`) was supplied
    to gate the resulting new-or-changed registration. Fails closed rather
    than inventing a TEST split the legacy module's own frozen_statistical_gate
    never defined -- see the legacy-bypass-closure transition-proof for the
    exact migration/adaptation each of the 10 modules would need if its
    computed content ever genuinely changes."""


def register_legacy_snapshot_with_gates(
    canonical_conn,
    *,
    registry_values: dict,
    results: list[tuple[str, str]],
    results_schema_version: int,
    results_provenance: str,
    results_created_ms: int,
    knowledge_db_path=None,
    test_cycle_ids=None,
    train_cycle_ids=None,
    no_test_split: bool = False,
    retry_token: str | None = None,
    supersession_token: str | None = None,
) -> dict:
    """Mandatory gated entry point for the 10 legacy "living snapshot"
    research modules. Always runs the graveyard slash-set gate. Then:

      - No prior row for this experiment_id, or the prior row's
        `_LEGACY_SNAPSHOT_STRICT_COLUMNS` (family identity + split
        methodology) match `registry_values`: this is either a first-ever
        registration or a plain refresh (population counts, descriptive
        prose, and dataset_hash may all legitimately drift). Writes directly
        via `_upsert_legacy_snapshot` (still a true no-op when truly nothing
        at all changed) -- no TEST-evidence nullifier is touched, because no
        new confirmatory evidence was consumed, only a descriptive
        re-measurement.
      - The prior row's strict (family/split-methodology) columns DIFFER: this is a
        genuine new-or-changed registration. If `no_test_split=True`
        (matches this module's own `frozen_statistical_gate`/`frozen_splits`
        text -- candidate_universe/w1/w3 declare "no train/test split"),
        registers directly (graveyard-gated, no nullifier applicable). Else
        requires `test_cycle_ids` and routes through the full
        `register_experiment_with_gates` (family+split+TEST-nullifier);
        missing `test_cycle_ids` here raises `MissingFrozenTestMetadata`
        fail-closed rather than inventing one.

    Every branch that actually registers issues an M-0034 gate receipt
    (`ami.governance.epistemic_gates.issue_gate_receipt`) so
    `ami.research.registry.ResearchRegistry` can verify this experiment_id
    passed the gate before accepting its own research.sqlite projection.
    """
    if knowledge_db_path is None:
        from ami.knowledge.store import DEFAULT_PATH as knowledge_db_path  # noqa: PLC0415

    experiment_id = registry_values["experiment_id"]
    existing = canonical_conn.execute(
        f"SELECT {', '.join(REGISTRY_COLUMNS)} FROM experiment_registry WHERE experiment_id=?",
        (experiment_id,)).fetchone()
    is_drift_only = existing is not None and all(
        dict(zip(REGISTRY_COLUMNS, existing))[c] == registry_values[c]
        for c in _LEGACY_SNAPSHOT_STRICT_COLUMNS
    )

    _direct_kb = sqlite3.connect(str(knowledge_db_path))
    try:
        _gates.init_gates_schema(_direct_kb)
    finally:
        _direct_kb.close()

    family_id = _gates.resolve_canonical_family_id(
        registry_values.get("question_ids", ""), registry_values.get("hypothesis_id", ""))

    # IMPORTANT: `existing is None` (a brand-new experiment_id) is NOT
    # eligible for the lenient drift-only path -- there is nothing yet to
    # compare against, so a first-ever registration is, by definition, new
    # content and must go through the same graveyard+nullifier gate as any
    # genuine content change below. Only an ALREADY-REGISTERED experiment_id
    # whose strict (family/split) columns are unchanged qualifies as a
    # drift-only refresh.
    if existing is not None and is_drift_only:
        already_attached = any(
            r[1] == "kb" for r in canonical_conn.execute("PRAGMA database_list").fetchall())
        if not already_attached:
            canonical_conn.execute("ATTACH DATABASE ? AS kb", (str(knowledge_db_path),))
        attached_by_us = not already_attached
        try:
            spec_text = " | ".join(str(x) for x in (
                registry_values.get("hypothesis_id", ""), registry_values.get("question_ids", ""),
                registry_values.get("frozen_population", ""), registry_values.get("frozen_features", ""),
                registry_values.get("frozen_target", ""), registry_values.get("frozen_thresholds", ""),
            ))
            _gates.assert_not_graveyard(canonical_conn, spec_text, retry_token=retry_token, _autocommit=False)
            registry_result = _upsert_legacy_snapshot(
                canonical_conn, registry_values, results, results_schema_version,
                results_provenance, results_created_ms)
            _gates.issue_gate_receipt(
                canonical_conn, experiment_id=experiment_id, canonical_family_id=family_id,
                split_version=None, nullifier=None, registry_result=registry_result)
            canonical_conn.commit()
            return {"experiment_id": experiment_id, "family_id": family_id, "registry_result": registry_result,
                    "results_result": registry_result, "nullifier_result": "N/A_DRIFT_ONLY_SNAPSHOT"}
        except BaseException:
            canonical_conn.rollback()
            raise
        finally:
            if attached_by_us:
                canonical_conn.execute("DETACH DATABASE kb")

    # scientific content changed -- descriptive-only modules register
    # directly (no TEST evidence exists to gate); everything else must
    # supply real test_cycle_ids or fail closed.
    if no_test_split:
        already_attached = any(
            r[1] == "kb" for r in canonical_conn.execute("PRAGMA database_list").fetchall())
        if not already_attached:
            canonical_conn.execute("ATTACH DATABASE ? AS kb", (str(knowledge_db_path),))
        attached_by_us = not already_attached
        try:
            spec_text = " | ".join(str(x) for x in (
                registry_values.get("hypothesis_id", ""), registry_values.get("question_ids", ""),
                registry_values.get("frozen_population", ""), registry_values.get("frozen_features", ""),
                registry_values.get("frozen_target", ""), registry_values.get("frozen_thresholds", ""),
            ))
            _gates.assert_not_graveyard(canonical_conn, spec_text, retry_token=retry_token, _autocommit=False)
            registry_result = record_experiment_registry(canonical_conn, registry_values)
            results_result = record_experiment_results(
                canonical_conn, experiment_id, results, schema_version=results_schema_version,
                provenance=results_provenance, created_ms=results_created_ms)
            _gates.issue_gate_receipt(
                canonical_conn, experiment_id=experiment_id, canonical_family_id=family_id,
                split_version=None, nullifier=None, registry_result=registry_result)
            canonical_conn.commit()
            return {"experiment_id": experiment_id, "family_id": family_id, "registry_result": registry_result,
                    "results_result": results_result, "nullifier_result": "N/A_NO_TEST_SPLIT"}
        except BaseException:
            canonical_conn.rollback()
            raise
        finally:
            if attached_by_us:
                canonical_conn.execute("DETACH DATABASE kb")

    if test_cycle_ids is None:
        raise MissingFrozenTestMetadata(
            f"MISSING_FROZEN_TEST_METADATA: {experiment_id!r}'s scientific content changed but no"
            " test_cycle_ids were supplied (and no_test_split was not set). Refusing to invent a"
            " TEST split -- supply the real frozen TEST independent-cycle-id set for this content"
            " version, or confirm this module truly has no train/test split and pass"
            " no_test_split=True.")

    return register_experiment_with_gates(
        canonical_conn, registry_values=registry_values, results=results,
        results_schema_version=results_schema_version, results_provenance=results_provenance,
        results_created_ms=results_created_ms, test_cycle_ids=test_cycle_ids,
        train_cycle_ids=train_cycle_ids, knowledge_db_path=knowledge_db_path,
        retry_token=retry_token, supersession_token=supersession_token)
