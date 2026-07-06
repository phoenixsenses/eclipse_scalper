"""EPISTEMIC GATES — fail-closed enforcement layer for two disciplines that
previously existed only as documentation habits (BATCH-EPISTEMIC-NULLIFIER-
GATES-V1, design transfer from the validator-unlinkability path, P2
"slash-by-nullifier" + "no-double-vote"):

1. GRAVEYARD SLASH-SET GATE — the failure archive (knowledge.sqlite
   `failure_archive`, master protocol §12 / AMI_RESEARCH_PROTOCOLS §5) is the
   canonical list of falsified/rejected hypothesis families. Until now the
   only mechanism was the advisory `KnowledgeStore.is_known_failure()`
   substring check, which nothing forces a researcher to call. This module
   turns the archive into an enforced slash-set: `assert_not_graveyard()`
   matches a proposed experiment's frozen-spec text against curated kinship
   fingerprints and raises `GraveyardRetestBlocked` unless the operator's
   retry-condition token is explicitly supplied (and then the pass itself is
   audit-logged). Fail-closed: a keyword hit blocks; breadth lives in the
   curated fingerprint list, never in fuzzy scoring.

2. TEST-EVIDENCE NULLIFIER — confirmatory TEST evidence is single-use per
   hypothesis family (evidence-contamination law, whitepaper §70). Until now
   `researcher_exposure_ledger` RECORDED exposure but nothing BLOCKED a
   family from re-reading the same TEST cycles under a new experiment id.
   `consume_test_evidence()` derives a deterministic nullifier from
   (family_id, split_version, sorted TEST cycle ids) and refuses a second
   consumption by a different experiment in the same family — unless an
   operator supersession token accompanies it (the corrected-data-rerun
   precedent: -002/-003/-004 style reruns were individually operator-
   approved; the token records that approval in data). Idempotent reruns of
   the SAME frozen experiment stay legal (NOOP_IDENTICAL), matching the
   experiment_ledger's immutability contract.

Storage: two additive tables in knowledge.sqlite (CREATE TABLE IF NOT
EXISTS), the same store that already holds failure_archive and audit_log.
Canonical.sqlite is never opened by this module.

NOT_CALLED_AUTOMATICALLY / NOT_YET_WIRED (as of V1 seed batch): every function
took an explicit connection; importing this module opened nothing. Wiring
these gates into `ami/warehouse/experiment_ledger.py` was flagged as a
SEPARATE, operator-approved step -- that step is `M-0033` /
`BATCH-EPISTEMIC-NULLIFIER-ENFORCEMENT-WIRING-V1`, see
`ami/warehouse/experiment_ledger.py:register_experiment_with_gates()`, the
new single mandatory entry point that calls this module's gates before
`record_experiment_registry`/`record_experiment_results`.

WIRING-V1 additions to this module (schema v2, additive-only, `M-0033`):
  - `epistemic_authorization_tokens` -- rich retry/supersession authorization
    records (§ AUTHORIZATION REQUIREMENTS of the enforcement-wiring batch),
    superseding the free-form `supersession_token` string that
    `epistemic_test_nullifiers` already carried (that column is kept for
    backward compatibility with the V1 seed batch's own tests; new callers
    go through `issue_retry_authorization`/`issue_supersession_authorization`
    + `_reserve_authorization` instead of a bare string).
  - a partial UNIQUE index on `epistemic_test_nullifiers(nullifier)` (only
    where `supersession_token IS NULL`) -- a database-level backstop so two
    concurrent *first* consumptions of the same nullifier cannot both commit,
    even if two processes both pass the application-level "no prior row"
    check before either commits.
  - `resolve_canonical_family_id`/`resolve_split_version` -- minimal,
    versioned identity adapters (no `hypothesis_family_id` column exists on
    `experiment_registry`; see the enforcement-wiring transition-proof for
    the residual bypass this leaves: differently-worded but semantically
    identical hypothesis_id/frozen_splits text resolve to different ids).
  - `_autocommit` kwarg added to `assert_not_graveyard`/`consume_test_evidence`
    (default `True`, preserving the V1 seed batch's existing standalone
    behavior/tests byte-for-byte) so the new orchestrator can defer the
    commit and fold graveyard-audit + nullifier-consumption + registry/result
    writes + authorization-consumption into ONE cross-database transaction.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import time

GATES_SCHEMA_VERSION = 2

_GATES_SCHEMA = """
CREATE TABLE IF NOT EXISTS graveyard_slash_fingerprints (
    fingerprint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT NOT NULL UNIQUE,
    failure_archive_id INTEGER,
    family_label TEXT NOT NULL,
    source TEXT NOT NULL,
    added_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS epistemic_test_nullifiers (
    nullifier TEXT NOT NULL,
    family_id TEXT NOT NULL,
    split_version TEXT NOT NULL,
    test_set_hash TEXT NOT NULL,
    consumed_by_experiment_id TEXT NOT NULL,
    supersession_token TEXT,
    consumed_ms INTEGER NOT NULL,
    PRIMARY KEY (nullifier, consumed_by_experiment_id)
);
CREATE INDEX IF NOT EXISTS idx_test_nullifiers_nullifier
    ON epistemic_test_nullifiers(nullifier);
"""

# WIRING-V1 (M-0033) additive schema -- rich authorization ledger + the
# concurrency backstop index. Rollback: DROP the table + DROP the index; no
# existing row in any table is touched (pure addition, same discipline as
# M-0032). Migration is idempotent (CREATE ... IF NOT EXISTS throughout).
_AUTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS epistemic_authorization_tokens (
    authorization_id TEXT PRIMARY KEY,
    authorization_type TEXT NOT NULL CHECK (authorization_type IN ('RETRY', 'SUPERSESSION')),
    canonical_family_id TEXT NOT NULL,
    related_experiment_id TEXT,
    related_nullifier TEXT,
    split_version TEXT,
    test_set_hash TEXT,
    approver TEXT NOT NULL,
    justification TEXT NOT NULL,
    retry_condition_satisfied TEXT,
    input_manifest_root TEXT,
    issued_ms INTEGER NOT NULL,
    expiry_ms INTEGER,
    single_use INTEGER NOT NULL DEFAULT 1,
    token_commitment TEXT NOT NULL UNIQUE,
    consumed INTEGER NOT NULL DEFAULT 0,
    resulting_experiment_id TEXT,
    consumed_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_auth_tokens_family
    ON epistemic_authorization_tokens(canonical_family_id, authorization_type);

CREATE UNIQUE INDEX IF NOT EXISTS uq_test_nullifiers_first_consumption
    ON epistemic_test_nullifiers(nullifier) WHERE supersession_token IS NULL;
"""


class GraveyardRetestBlocked(Exception):
    """Proposed spec matches a slashed (graveyarded) hypothesis family and no
    operator retry-token was supplied."""


class TestEvidenceReuseBlocked(Exception):
    """A hypothesis family attempted to consume the same TEST evidence set a
    second time under a new experiment id without an operator supersession
    token -- OR two concurrent first-consumption attempts raced and this one
    lost (caught IntegrityError on the partial unique index)."""


class AuthorizationInvalid(Exception):
    """A supplied retry/supersession token does not resolve to a valid,
    unconsumed, unexpired, correctly-scoped `epistemic_authorization_tokens`
    row. Fail-closed: missing, wrong family, wrong split, wrong test-set,
    already-consumed, and expired tokens all raise this."""


def init_gates_schema(conn) -> None:
    """Idempotent, additive-only. Caller supplies the knowledge.sqlite
    connection (real or disposable copy). Runs both the V1 seed-batch schema
    and the WIRING-V1 (M-0033) additions -- safe to call repeatedly and safe
    to call on a DB that already has either or both."""
    conn.executescript(_GATES_SCHEMA)
    conn.executescript(_AUTH_SCHEMA)
    conn.commit()


def _audit(conn, action: str, detail: str) -> None:
    """Reuses knowledge.sqlite's existing audit_log (same shape KnowledgeStore
    writes); if the caller's DB has no audit_log (bare disposable), skip
    silently — auditing never blocks the gate itself."""
    try:
        conn.execute(
            "INSERT INTO audit_log (ts_ms, actor, action, knowledge_id, detail) VALUES (?,?,?,?,?)",
            (int(time.time() * 1000), "epistemic_gates", action, None, detail[:400]))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 1. Graveyard slash-set gate
# ---------------------------------------------------------------------------

# Curated kinship fingerprints. keyword -> (failure_archive_id or None, family_label).
# Sources: the 21 failure_archive rows (ids below verified against the real DB
# 2026-07-06) + the FAILURE_ARCHIVE.md index families that post-date the last
# DB seed (id=None, source records the MD index). Keywords are lowercase
# substrings; curation errs broad (fail-closed) — a false-positive costs one
# operator token, a false-negative costs a graveyard re-test.
_CURATED_FINGERPRINTS: tuple[tuple[str, int | None, str], ...] = (
    ("pre-cascade entry", 1, "naive early pre-cascade entry"),
    ("micro-timing", 2, "broad micro-timing optimization"),
    ("tight stop", 3, "tight volatility-scaled stops"),
    ("volatility-scaled stop", 3, "tight volatility-scaled stops"),
    ("partial exit", 4, "partial exits"),
    ("limit-entry", 5, "limit-entry improvement"),
    ("limit entry", 5, "limit-entry improvement"),
    ("bk_refill", 6, "bk_refill universal separator"),
    ("gentleness", 7, "cascade gentleness/grind signal"),
    ("grind signal", 7, "cascade gentleness/grind signal"),
    ("buy-side fade", 8, "BUY-side short-squeeze fade"),
    ("short-squeeze fade", 8, "BUY-side short-squeeze fade"),
    ("buyfade", 8, "BUY-side fade family (all overlays 16-21)"),
    ("cross-asset transfer", 9, "cross-asset transfer"),
    ("time-stop", 10, "loser time-stops"),
    ("clock-targeted exit", 11, "clock-targeted exits"),
    ("silence as t0", 12, "silence as T0-tradeable main alpha (lookahead)"),
    ("silence t0", 12, "silence as T0-tradeable main alpha (lookahead)"),
    ("mfe50", 13, "MFE50 giveback separation"),
    ("latent state", 14, "Faz6A latent states"),
    ("kmeans", 14, "Faz6A latent states"),
    ("latent risk overlay", 15, "Faz6A-R2 regime+latent overlay"),
    ("re-entry", 19, "S->S re-entry churn"),
    ("reversal harvest", None, "reversal harvest (FAILURE_ARCHIVE.md index)"),
    ("failed_cascade_short", None, "failed_cascade_SHORT (FAILURE_ARCHIVE.md index)"),
    ("delayed entry", None, "delayed-entry silence (FAILURE_ARCHIVE.md index)"),
    ("delayed-entry", None, "delayed-entry silence (FAILURE_ARCHIVE.md index)"),
    ("early-exit", None, "early-exit kills edge (FAILURE_ARCHIVE.md index)"),
    ("early exit rule", None, "early-exit kills edge (FAILURE_ARCHIVE.md index)"),
    ("ofi momentum", None, "agg-trade OFI momentum (S34_ORDERFLOW_LEAD graveyard)"),
    ("ofi-momentum", None, "agg-trade OFI momentum (S34_ORDERFLOW_LEAD graveyard)"),
)


def seed_slash_fingerprints(conn) -> dict:
    """Idempotent seed (INSERT OR IGNORE on UNIQUE keyword)."""
    now = int(time.time() * 1000)
    inserted = 0
    for keyword, fa_id, label in _CURATED_FINGERPRINTS:
        cur = conn.execute(
            "INSERT OR IGNORE INTO graveyard_slash_fingerprints"
            " (keyword, failure_archive_id, family_label, source, added_ms) VALUES (?,?,?,?,?)",
            (keyword, fa_id,
             label, "failure_archive" if fa_id is not None else "FAILURE_ARCHIVE.md index",
             now))
        inserted += cur.rowcount
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM graveyard_slash_fingerprints").fetchone()[0]
    return {"inserted": inserted, "total": total}


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def match_graveyard(conn, spec_text: str) -> list[dict]:
    """Pure matcher (no raise): every fingerprint whose keyword occurs as a
    substring of the normalized spec text, with the linked archive row's
    retry_condition when available."""
    norm = _normalize(spec_text)
    hits = []
    for fp_id, keyword, fa_id, label in conn.execute(
            "SELECT fingerprint_id, keyword, failure_archive_id, family_label"
            " FROM graveyard_slash_fingerprints ORDER BY fingerprint_id"):
        if keyword in norm:
            retry = None
            if fa_id is not None:
                row = conn.execute(
                    "SELECT retry_condition FROM failure_archive WHERE id=?", (fa_id,)).fetchone()
                retry = row[0] if row else None
            hits.append({"fingerprint_id": fp_id, "keyword": keyword,
                         "failure_archive_id": fa_id, "family_label": label,
                         "retry_condition": retry})
    return hits


def assert_not_graveyard(conn, spec_text: str, *, retry_token: str | None = None,
                          _autocommit: bool = True) -> dict:
    """Fail-closed gate. A hit without a token raises; a hit WITH a token
    passes but the invocation is audit-logged (the token is the operator's
    recorded acknowledgment that the registered retry-condition is met).

    `_autocommit=False` (WIRING-V1 addition): skip the internal commit so a
    caller can fold this write into a larger atomic transaction (see
    `ami/warehouse/experiment_ledger.py:register_experiment_with_gates`).
    Default stays `True` so every V1-seed-batch caller/test is unaffected."""
    hits = match_graveyard(conn, spec_text)
    if not hits:
        return {"result": "CLEAN", "hits": []}
    if retry_token:
        _audit(conn, "GRAVEYARD_RETRY_TOKEN_USED",
               json.dumps({"token": retry_token, "hits": [h["family_label"] for h in hits]}))
        if _autocommit:
            conn.commit()
        return {"result": "PASSED_WITH_RETRY_TOKEN", "hits": hits, "retry_token": retry_token}
    families = "; ".join(f"{h['family_label']} (keyword={h['keyword']!r},"
                         f" retry_condition={h['retry_condition']!r})" for h in hits)
    raise GraveyardRetestBlocked(
        f"GRAVEYARD_RETEST_BLOCKED: spec matches {len(hits)} slashed famil"
        f"{'y' if len(hits) == 1 else 'ies'}: {families}. Re-testing requires the"
        " registered retry-condition to be met AND an operator retry_token.")


# ---------------------------------------------------------------------------
# 2. TEST-evidence nullifier
# ---------------------------------------------------------------------------

def _normalized_test_set_hash(test_cycle_ids) -> tuple[str, str, int]:
    """Deterministic, order-invariant. Fails closed on duplicate cycle ids
    (WIRING-V1: a duplicate silently halving the effective independent-N
    would understate the true single-use exposure -- reject instead)."""
    raw = [str(c) for c in test_cycle_ids]
    if len(raw) != len(set(raw)):
        dupes = sorted({c for c in raw if raw.count(c) > 1})
        raise ValueError(f"duplicate TEST cycle id(s) not allowed: {dupes[:5]}")
    ids = ",".join(sorted(raw))
    return ids, hashlib.sha256(ids.encode()).hexdigest(), len(raw)


def derive_test_nullifier(family_id: str, split_version: str, test_cycle_ids) -> str:
    """Deterministic, order-invariant over the TEST cycle-id set."""
    _, test_set_hash, _ = _normalized_test_set_hash(test_cycle_ids)
    return hashlib.sha256(f"{family_id}|{split_version}|{test_set_hash}".encode()).hexdigest()


def consume_test_evidence(conn, *, family_id: str, split_version: str, test_cycle_ids,
                          experiment_id: str, supersession_token: str | None = None,
                          _autocommit: bool = True) -> str:
    """Single-use law for confirmatory TEST evidence.

    Returns 'CONSUMED' (first use), 'NOOP_IDENTICAL' (idempotent rerun of the
    same experiment), or 'CONSUMED_WITH_SUPERSESSION' (new experiment id on an
    already-consumed set, operator token recorded). Raises
    TestEvidenceReuseBlocked otherwise -- including when a concurrent first
    consumption of the same nullifier won the race (caught as an
    IntegrityError against the WIRING-V1 partial unique index, translated
    here so callers see one exception type regardless of whether the
    app-level pre-check or the database-level backstop caught the race).

    `_autocommit=False` (WIRING-V1): defer the commit for the orchestrator's
    single cross-database transaction; default `True` preserves the V1 seed
    batch's standalone behavior exactly.
    """
    _, test_set_hash, _ = _normalized_test_set_hash(test_cycle_ids)
    nullifier = hashlib.sha256(f"{family_id}|{split_version}|{test_set_hash}".encode()).hexdigest()

    prior = conn.execute(
        "SELECT consumed_by_experiment_id, supersession_token FROM epistemic_test_nullifiers"
        " WHERE nullifier=? ORDER BY consumed_ms", (nullifier,)).fetchall()
    if any(p[0] == experiment_id for p in prior):
        return "NOOP_IDENTICAL"
    if prior and supersession_token is None:
        consumers = ", ".join(p[0] for p in prior)
        raise TestEvidenceReuseBlocked(
            f"TEST_EVIDENCE_REUSE_BLOCKED: family {family_id!r} already consumed this TEST set"
            f" (split={split_version}, consumers: {consumers}). A revised computation on the"
            " SAME test evidence requires an operator supersession_token (corrected-rerun"
            " precedent); genuinely new evidence means a different test set and therefore a"
            " different nullifier.")
    try:
        conn.execute(
            "INSERT INTO epistemic_test_nullifiers (nullifier, family_id, split_version,"
            " test_set_hash, consumed_by_experiment_id, supersession_token, consumed_ms)"
            " VALUES (?,?,?,?,?,?,?)",
            (nullifier, family_id, split_version, test_set_hash, experiment_id,
             supersession_token, int(time.time() * 1000)))
    except Exception as exc:  # sqlite3.IntegrityError on the partial unique index
        if "UNIQUE" in str(exc).upper() or "unique" in type(exc).__name__.lower():
            raise TestEvidenceReuseBlocked(
                f"TEST_EVIDENCE_REUSE_BLOCKED: concurrent first-consumption race on family"
                f" {family_id!r} (split={split_version}) -- another process committed the same"
                " nullifier first (database-level backstop, not just the app-level check).") from exc
        raise
    if prior:
        _audit(conn, "TEST_NULLIFIER_SUPERSESSION",
               json.dumps({"family": family_id, "experiment": experiment_id,
                           "token": supersession_token}))
    if _autocommit:
        conn.commit()
    return "CONSUMED_WITH_SUPERSESSION" if prior else "CONSUMED"


# ---------------------------------------------------------------------------
# 3. Retro-audit (measurement only — P0 discipline: measure before enforcing)
# ---------------------------------------------------------------------------

def retro_audit_experiment_registry(knowledge_conn, canonical_ro) -> list[dict]:
    """Runs the graveyard matcher over every EXISTING experiment_registry row
    (canonical opened read-only by the caller). Pure measurement: no gate is
    raised, nothing is written to canonical — the output says which past
    registrations WOULD have required a retry token had the gate existed."""
    out = []
    rows = canonical_ro.execute(
        "SELECT experiment_id, COALESCE(hypothesis_id,''), COALESCE(question_ids,''),"
        " COALESCE(frozen_population,''), COALESCE(frozen_features,''),"
        " COALESCE(frozen_target,''), COALESCE(frozen_thresholds,'')"
        " FROM experiment_registry ORDER BY experiment_id").fetchall()
    for r in rows:
        spec_text = " | ".join(str(x) for x in r[1:])
        hits = match_graveyard(knowledge_conn, spec_text)
        out.append({"experiment_id": r[0],
                    "would_block": bool(hits),
                    "hits": [h["family_label"] for h in hits]})
    return out


# ---------------------------------------------------------------------------
# 4. Family / split identity adapter (WIRING-V1, M-0033)
#
# `experiment_registry` has no `hypothesis_family_id` column and none is added
# here (schema_version 12 stays untouched -- out of scope for a
# "wire enforcement only" batch, and a full ontology redesign is explicitly
# out of scope too). This is the minimum-compatible adapter: deterministic,
# versioned (`FAMv1`/`SPLITv1` prefixes so a future, richer identity scheme
# can coexist without colliding), immutable once frozen (inputs are the
# experiment_registry content columns, which are themselves immutable per
# `experiment_ledger.py`'s NOOP_IDENTICAL/ImmutableExperimentConflict
# contract), auditable (the resolved ids are what gets written to the audit
# record), and backward-compatible with the curated fingerprint mechanism
# (graveyard matching still runs on raw spec_text, untouched by this).
#
# KNOWN RESIDUAL BYPASS (report verbatim in the transition proof): both
# adapters hash free-form researcher-authored text. A semantically identical
# hypothesis re-worded under a different `hypothesis_id` string, or a
# `frozen_splits` description reworded without changing its actual meaning,
# resolves to a DIFFERENT family_id/split_version and therefore a different
# nullifier -- the single-use law does not see through paraphrase. Closing
# this fully requires a real canonical hypothesis-family ontology, which is
# explicitly out of scope for this batch.
# ---------------------------------------------------------------------------

FAMILY_ID_ADAPTER_VERSION = "FAMv1"
SPLIT_VERSION_ADAPTER_VERSION = "SPLITv1"


def resolve_canonical_family_id(question_ids: str, hypothesis_id: str) -> str:
    """Deterministic family identity from the two free-text identity columns
    `experiment_registry` already has. Each field is normalized (whitespace
    collapsed, lowercased) INDEPENDENTLY before joining, so leading/trailing
    whitespace on either field can never change the result -- normalizing the
    already-concatenated string would leave whitespace touching the join
    delimiter uncollapsed relative to the no-whitespace case."""
    norm = f"{_normalize(question_ids or '')}|{_normalize(hypothesis_id or '')}"
    return f"{FAMILY_ID_ADAPTER_VERSION}:{hashlib.sha256(norm.encode()).hexdigest()[:16]}"


def resolve_split_version(frozen_splits: str) -> str:
    """Deterministic split-version identity from the already-frozen
    `frozen_splits` description column (e.g. 'chronological 70/30 by
    signal_birth_ts, never randomized'). Two experiments describing the same
    split methodology in the same words share a split_version; a genuinely
    different split methodology (different fraction, different ordering
    field) produces a different one."""
    norm = _normalize(frozen_splits or "")
    return f"{SPLIT_VERSION_ADAPTER_VERSION}:{hashlib.sha256(norm.encode()).hexdigest()[:16]}"


# ---------------------------------------------------------------------------
# 5. Authorization tokens (WIRING-V1, M-0033) -- retry + supersession
#
# Replaces the V1 seed batch's bare `retry_token`/`supersession_token`
# strings (still accepted by `assert_not_graveyard`/`consume_test_evidence`
# for backward compatibility) with a real, auditable, scoped, single-use
# record. `register_experiment_with_gates` (experiment_ledger.py) is the only
# caller that should ever produce the validated token string to hand down to
# those two functions -- it does so by calling `_reserve_authorization` here
# FIRST, which raises AuthorizationInvalid fail-closed on any mismatch.
#
# The raw token is a `secrets.token_urlsafe` string, returned ONCE to the
# operator/approver at issuance time and never stored -- only its sha256
# "commitment" is persisted (`token_commitment`), so a knowledge.sqlite leak
# or audit_log read cannot recover a usable token.
# ---------------------------------------------------------------------------

def _commitment(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode()).hexdigest()


def _issue_authorization(conn, *, authorization_type: str, canonical_family_id: str,
                          approver: str, justification: str,
                          related_experiment_id: str | None = None,
                          related_nullifier: str | None = None,
                          split_version: str | None = None,
                          test_set_hash: str | None = None,
                          retry_condition_satisfied: str | None = None,
                          input_manifest_root: str | None = None,
                          expiry_ms: int | None = None) -> str:
    if not approver or not justification:
        raise ValueError("authorization requires both approver and justification (fail-closed)")
    raw_token = secrets.token_urlsafe(32)
    authorization_id = f"AUTH-{authorization_type}-{secrets.token_hex(8)}"
    now = int(time.time() * 1000)
    conn.execute(
        "INSERT INTO epistemic_authorization_tokens (authorization_id, authorization_type,"
        " canonical_family_id, related_experiment_id, related_nullifier, split_version,"
        " test_set_hash, approver, justification, retry_condition_satisfied,"
        " input_manifest_root, issued_ms, expiry_ms, single_use, token_commitment, consumed)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,0)",
        (authorization_id, authorization_type, canonical_family_id, related_experiment_id,
         related_nullifier, split_version, test_set_hash, approver, justification,
         retry_condition_satisfied, input_manifest_root, now, expiry_ms, _commitment(raw_token)))
    _audit(conn, f"{authorization_type}_AUTHORIZATION_ISSUED",
           json.dumps({"authorization_id": authorization_id, "family": canonical_family_id,
                       "approver": approver}))
    conn.commit()
    return raw_token


def issue_retry_authorization(conn, *, canonical_family_id: str, approver: str, justification: str,
                               related_experiment_id: str, retry_condition_satisfied: str,
                               input_manifest_root: str | None = None,
                               expiry_ms: int | None = None) -> str:
    """Operator/out-of-band action, done BEFORE a graveyard-blocked
    registration is retried. Returns the raw token (shown once)."""
    return _issue_authorization(
        conn, authorization_type="RETRY", canonical_family_id=canonical_family_id,
        approver=approver, justification=justification,
        related_experiment_id=related_experiment_id,
        retry_condition_satisfied=retry_condition_satisfied,
        input_manifest_root=input_manifest_root, expiry_ms=expiry_ms)


def issue_supersession_authorization(conn, *, canonical_family_id: str, approver: str,
                                      justification: str, related_nullifier: str,
                                      split_version: str, test_set_hash: str,
                                      input_manifest_root: str | None = None,
                                      expiry_ms: int | None = None) -> str:
    """Operator/out-of-band action, done BEFORE a re-consumption of an
    already-consumed TEST-evidence nullifier. Scoped to one family + one
    split_version + one test_set_hash -- non-transferable across any of the
    three. Returns the raw token (shown once)."""
    return _issue_authorization(
        conn, authorization_type="SUPERSESSION", canonical_family_id=canonical_family_id,
        approver=approver, justification=justification,
        related_nullifier=related_nullifier, split_version=split_version,
        test_set_hash=test_set_hash, input_manifest_root=input_manifest_root,
        expiry_ms=expiry_ms)


def _reserve_authorization(conn, *, raw_token: str, authorization_type: str,
                            canonical_family_id: str, split_version: str | None = None,
                            test_set_hash: str | None = None) -> str:
    """Fail-closed validation: resolves `raw_token` to a row, checks type,
    family scope, split/test-set scope (when the row records one), not
    already consumed, not expired. Does NOT commit and does NOT mark the row
    consumed yet -- that happens in `_finalize_authorization_consumption`
    only after the caller's overall transaction is about to succeed, so a
    downstream failure leaves the token unconsumed and reusable.
    Returns the `authorization_id` on success; raises AuthorizationInvalid
    otherwise (missing / wrong type / wrong family / wrong split / wrong
    test-set / already consumed / expired -- every branch fails closed)."""
    if not raw_token:
        raise AuthorizationInvalid(f"AUTHORIZATION_INVALID: no {authorization_type} token supplied.")
    row = conn.execute(
        "SELECT authorization_id, authorization_type, canonical_family_id, split_version,"
        " test_set_hash, expiry_ms, consumed FROM epistemic_authorization_tokens"
        " WHERE token_commitment=?", (_commitment(raw_token),)).fetchone()
    if row is None:
        raise AuthorizationInvalid("AUTHORIZATION_INVALID: token does not match any issued authorization.")
    auth_id, auth_type, family_id, auth_split, auth_test_set, expiry_ms, consumed = row
    if auth_type != authorization_type:
        raise AuthorizationInvalid(
            f"AUTHORIZATION_INVALID: token is a {auth_type} authorization, not {authorization_type}.")
    if consumed:
        raise AuthorizationInvalid(f"AUTHORIZATION_INVALID: token {auth_id!r} already consumed (single-use).")
    if expiry_ms is not None and int(time.time() * 1000) > expiry_ms:
        raise AuthorizationInvalid(f"AUTHORIZATION_INVALID: token {auth_id!r} expired at {expiry_ms}.")
    if family_id != canonical_family_id:
        raise AuthorizationInvalid(
            f"AUTHORIZATION_INVALID: token {auth_id!r} is scoped to family {family_id!r},"
            f" not {canonical_family_id!r} (non-transferable between families).")
    if authorization_type == "SUPERSESSION":
        if auth_split != split_version:
            raise AuthorizationInvalid(
                f"AUTHORIZATION_INVALID: token {auth_id!r} is scoped to split_version {auth_split!r},"
                f" not {split_version!r} (non-transferable between split versions).")
        if auth_test_set != test_set_hash:
            raise AuthorizationInvalid(
                f"AUTHORIZATION_INVALID: token {auth_id!r} is scoped to a different TEST cycle-set"
                " (non-transferable between TEST cycle sets).")
    return auth_id


def _finalize_authorization_consumption(conn, *, authorization_id: str, resulting_experiment_id: str) -> None:
    """Marks the row consumed. Caller commits (part of the orchestrator's
    single atomic transaction) -- this function itself never commits."""
    now = int(time.time() * 1000)
    conn.execute(
        "UPDATE epistemic_authorization_tokens SET consumed=1, resulting_experiment_id=?,"
        " consumed_ms=? WHERE authorization_id=? AND consumed=0",
        (resulting_experiment_id, now, authorization_id))
    if conn.execute("SELECT changes()").fetchone()[0] != 1:
        # Someone else consumed it between _reserve_authorization and here
        # (concurrent race) -- fail closed rather than silently proceed.
        raise AuthorizationInvalid(
            f"AUTHORIZATION_INVALID: token {authorization_id!r} was consumed by a concurrent"
            " request between validation and finalization.")
    _audit(conn, f"AUTHORIZATION_CONSUMED",
           json.dumps({"authorization_id": authorization_id, "experiment_id": resulting_experiment_id}))
