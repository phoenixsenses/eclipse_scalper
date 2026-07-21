"""Phase A evidence-safety ledger WRITERS (whitepaper §70 / §77 Phase A).

Closes gap-analysis items A1 and A3 (`docs/ami/AMI_GAP_ANALYSIS.md` §3/§6):
`evidence_contamination` and `mt_family_registry` were created as empty schema
in `BATCH-P2-001` and explicitly deferred (`SYSTEM_STATE.md:2388` -- "Hepsi boş
... bu batch yalnız iskelet"). Since then the ONLY Phase A table with a writer
was `researcher_exposure_ledger`, and that one only by accident
(`ami/research/feature_gateway.py:90` logs a row per fetch).

This module gives the other two a writer, hooked at the one place where the
required facts are already resolved and already inside a transaction:
`ami.warehouse.experiment_ledger.register_experiment_with_gates` (M-0033). That
hook is what makes these ledgers records of ENFORCEMENT rather than of
paperwork -- the gate already knows the canonical family, the frozen split, the
TEST nullifier, and whether that TEST evidence was consumed before.

SCOPE LIMIT -- READ THIS BEFORE TRUSTING A ROW
----------------------------------------------
These ledgers see exactly what the gated entry point sees. The 10 legacy
`w*` research modules bypass `register_experiment_with_gates` entirely and call
`record_experiment_registry`/`record_experiment_results` directly -- a
"KNOWN, UNCLOSED bypass surface" per `experiment_ledger.py`'s own docstring
(lines 46-69). So an ABSENT row here is NOT evidence that a family was never
tested; it may mean the caller went around the gate. Closing that surface is a
separate item (gap analysis §6.3).

WHAT IS DERIVED VS WHAT IS REFUSED
----------------------------------
Only fields the gate can actually establish are written. The rest stay NULL
with an explicit reason rather than being filled with a plausible number:

  mt_family_registry
    variants_tested             DERIVED  (distinct experiments per family)
    effective_trials            DERIVED  (a rerun of the SAME frozen experiment
                                          is not a new trial -- it is the
                                          immutability contract's NOOP)
    family_adjusted_significance DERIVED (Bonferroni: alpha / effective_trials)
    threshold_stability         REFUSED  (needs the per-variant threshold sweep,
                                          which the gate never sees)
    researcher_freedom_score    REFUSED  (needs the analyst's full decision
                                          space, not observable here)
    minimum_economic_effect     REFUSED  (an economic policy input, not a
                                          measurement)

  evidence_contamination
    evidence_status             DERIVED  (from prior nullifier consumption --
                                          see `classify_evidence_status`)
    hypothesis_origin_split     CALLER   (the gate cannot know where a
                                          hypothesis was born; UNDECLARED is
                                          recorded honestly, never guessed)
    contaminated_splits         DERIVED  (only when origin split is declared)
    fresh_forward_required      DERIVED
    maximum_evidence_ceiling    DERIVED

Bonferroni, not Holm: Holm needs the whole family's p-values at once, and this
hook sees one experiment at a time as it arrives. Bonferroni is the honest
online bound. `family_adjusted_significance` is the ALPHA a variant must beat,
not a p-value.

TWO IDENTITIES, ON PURPOSE (do not "simplify" this)
---------------------------------------------------
`epistemic_gates.resolve_canonical_family_id(question_ids, hypothesis_id)` hashes
BOTH columns, so every distinct hypothesis is its own gate-family. That is right
for the NULLIFIER (it must not let hypothesis A's TEST consumption silently
license hypothesis B) but it is useless for multiple-testing: if every hypothesis
is its own family, `variants_tested` is permanently 1 and the Bonferroni alpha
never tightens off 0.05 -- a ledger that always answers "one trial" is a decoration.

So the MT family key here is derived from `question_ids` ALONE
(`resolve_mt_family_id`), which is where the real `FAM_*` family strings live
(e.g. `FAM_SIGNAL_AGING_CLOCK_ROUTE_HOLD_EXECUTION`). Many hypotheses in one
`FAM_*` = many trials in one multiple-testing family, which is the whole point.
The gate's family_id is deliberately NOT reused here, and this module never
mutates it -- changing `resolve_canonical_family_id` would invalidate every
nullifier already consumed.
"""
from __future__ import annotations

import time
import uuid

from ami.governance.epistemic_gates import (
    _normalize,
    resolve_split_version,
)

SCHEMA_VERSION = 5
MT_FAMILY_ADAPTER_VERSION = "MTFAMv1"
DEFAULT_ALPHA = 0.05
PROVENANCE = "batch-phase-a-evidence-ledgers-v1"

# evidence_contamination.evidence_status (whitepaper §70.1)
INDEPENDENT_EVIDENCE = "INDEPENDENT_EVIDENCE"
REUSED_EVIDENCE = "REUSED_EVIDENCE"
CONTAMINATED_FOR_CONFIRMATION = "CONTAMINATED_FOR_CONFIRMATION"
FORWARD_ONLY_CONFIRMATION_REQUIRED = "FORWARD_ONLY_CONFIRMATION_REQUIRED"

ALL_EVIDENCE_STATUSES = (
    INDEPENDENT_EVIDENCE, REUSED_EVIDENCE,
    CONTAMINATED_FOR_CONFIRMATION, FORWARD_ONLY_CONFIRMATION_REQUIRED,
)

UNDECLARED_ORIGIN_SPLIT = "UNDECLARED"

# maximum_evidence_ceiling
CEILING_CONFIRMATORY = "CONFIRMATORY_ALLOWED"
CEILING_NO_UPGRADE = "NO_CONFIRMATORY_UPGRADE"
CEILING_FORWARD_ONLY = "FORWARD_ONLY"

_REFUSED = None  # explicit: a field the gate cannot establish


def resolve_mt_family_id(question_ids: str) -> str:
    """Multiple-testing family identity from `question_ids` ALONE.

    Deliberately NOT `epistemic_gates.resolve_canonical_family_id`, which also
    hashes `hypothesis_id` and therefore puts every hypothesis in its own
    family -- correct for the nullifier, fatal for multiple-testing (see the
    module docstring). Normalized the same way so whitespace/case cannot fork
    a family.
    """
    return f"{MT_FAMILY_ADAPTER_VERSION}:{_normalize(question_ids or '')}"


def classify_evidence_status(
    *,
    prior_consumption_count: int,
    is_rerun_of_self: bool,
    supersession_used: bool,
    hypothesis_origin_split: str | None,
    split_version: str,
) -> tuple[str, bool, str]:
    """Derive (evidence_status, fresh_forward_required, ceiling).

    Both split arguments MUST already be in the same vocabulary. The gate works
    in RESOLVED split identity (`resolve_split_version` -> "SPLITv1:<hash>"),
    never the raw `frozen_splits` prose, so a caller-supplied raw origin string
    must be resolved before it reaches here -- otherwise the comparison below
    can never match and the contamination branch is silently dead code.
    `record_evidence_contamination` does that resolution; this function assumes
    it was done.

    Precedence, most-specific first:

    1. The hypothesis was BORN on the very split it is now being confirmed on.
       That is contamination regardless of nullifier bookkeeping: the split
       cannot confirm a hypothesis it generated. Beats everything else because
       no amount of first-time-ness repairs it.
    2. TEST evidence for this family+split was already consumed by a DIFFERENT
       experiment -> reuse. Only reachable at all if an operator supersession
       token was accepted upstream.
    3. Otherwise -> independent. A rerun of the SAME frozen experiment stays
       independent: it is the immutability contract's idempotent NOOP, not a
       second look at the data.
    """
    origin_declared = (
        hypothesis_origin_split is not None
        and hypothesis_origin_split != UNDECLARED_ORIGIN_SPLIT
    )
    if origin_declared and hypothesis_origin_split == split_version:
        # Born on this split AND the split is already spent by someone else:
        # nothing untouched remains, so only fresh forward data can confirm.
        if prior_consumption_count > 0 and not is_rerun_of_self:
            return FORWARD_ONLY_CONFIRMATION_REQUIRED, True, CEILING_FORWARD_ONLY
        return CONTAMINATED_FOR_CONFIRMATION, True, CEILING_FORWARD_ONLY
    if prior_consumption_count > 0 and not is_rerun_of_self:
        return REUSED_EVIDENCE, True, CEILING_NO_UPGRADE
    return INDEPENDENT_EVIDENCE, False, CEILING_CONFIRMATORY


def record_evidence_contamination(
    conn,
    *,
    hypothesis_id: str,
    family_id: str,
    split_version: str,
    prior_consumption_count: int,
    is_rerun_of_self: bool,
    supersession_used: bool,
    hypothesis_origin_split: str | None = None,
    hypothesis_birth_ts: int | None = None,
    splits_seen_before_freeze: str | None = None,
    eligible_validation_splits: str | None = None,
    origin_split_is_resolved: bool = False,
    provenance: str = PROVENANCE,
    now_ms: int | None = None,
    _autocommit: bool = True,
) -> dict:
    """Upsert one contamination row, keyed by (hypothesis_id, split_version).

    Idempotent by construction: the same gated registration re-run rewrites the
    same row rather than appending a second verdict for the same hypothesis on
    the same split. That matches the experiment ledger's immutability contract
    -- unlike `researcher_exposure_ledger`, which is genuinely append-per-fetch.
    """
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    # Callers declare the origin split in the SAME prose vocabulary as
    # `frozen_splits` ("chronological 70/30 by signal_birth_ts"). The gate
    # compares RESOLVED identities, so resolve it here -- skipping this makes
    # the whole contamination branch unreachable (caught end-to-end: a
    # hypothesis declaring its own split as origin was still classified
    # INDEPENDENT_EVIDENCE because "split-v1" != "SPLITv1:26f773a2...").
    if hypothesis_origin_split is None or hypothesis_origin_split == UNDECLARED_ORIGIN_SPLIT:
        origin = UNDECLARED_ORIGIN_SPLIT
    elif origin_split_is_resolved:
        origin = hypothesis_origin_split
    else:
        origin = resolve_split_version(hypothesis_origin_split)
    status, fresh_forward, ceiling = classify_evidence_status(
        prior_consumption_count=prior_consumption_count,
        is_rerun_of_self=is_rerun_of_self,
        supersession_used=supersession_used,
        hypothesis_origin_split=origin,
        split_version=split_version,
    )
    contamination_id = f"CONT-{family_id}-{split_version}-{hypothesis_id}"[:180]
    contaminated_splits = split_version if status in (
        CONTAMINATED_FOR_CONFIRMATION, FORWARD_ONLY_CONFIRMATION_REQUIRED, REUSED_EVIDENCE) else None
    existing = conn.execute(
        "SELECT created_ms FROM evidence_contamination WHERE contamination_id=?",
        (contamination_id,)).fetchone()
    created = int(existing[0]) if existing else now
    conn.execute(
        "INSERT INTO evidence_contamination (contamination_id, hypothesis_id, hypothesis_birth_ts,"
        " hypothesis_origin_split, splits_seen_before_freeze, contaminated_splits,"
        " eligible_validation_splits, fresh_forward_required, evidence_status,"
        " maximum_evidence_ceiling, schema_version, provenance, created_ms, updated_ms)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        " ON CONFLICT(contamination_id) DO UPDATE SET"
        "  hypothesis_birth_ts=excluded.hypothesis_birth_ts,"
        "  hypothesis_origin_split=excluded.hypothesis_origin_split,"
        "  splits_seen_before_freeze=excluded.splits_seen_before_freeze,"
        "  contaminated_splits=excluded.contaminated_splits,"
        "  eligible_validation_splits=excluded.eligible_validation_splits,"
        "  fresh_forward_required=excluded.fresh_forward_required,"
        "  evidence_status=excluded.evidence_status,"
        "  maximum_evidence_ceiling=excluded.maximum_evidence_ceiling,"
        "  updated_ms=excluded.updated_ms",
        (contamination_id, hypothesis_id, hypothesis_birth_ts, origin,
         splits_seen_before_freeze, contaminated_splits, eligible_validation_splits,
         1 if fresh_forward else 0, status, ceiling, SCHEMA_VERSION, provenance, created, now),
    )
    if _autocommit:
        conn.commit()
    return {
        "contamination_id": contamination_id,
        "evidence_status": status,
        "fresh_forward_required": fresh_forward,
        "maximum_evidence_ceiling": ceiling,
        "hypothesis_origin_split": origin,
        "origin_declared": origin != UNDECLARED_ORIGIN_SPLIT,
    }


def bonferroni_alpha(effective_trials: int, alpha: float = DEFAULT_ALPHA) -> float:
    """The alpha a single variant must beat given the family's trial count.

    Online-safe: this hook sees one experiment at a time, so Holm (which needs
    every p-value at once) is not available. Bonferroni is the honest bound.
    """
    return alpha / max(1, int(effective_trials))


def record_family_variant(
    conn,
    *,
    question_ids: str,
    experiment_id: str,
    is_rerun_of_self: bool,
    alpha: float = DEFAULT_ALPHA,
    provenance: str = PROVENANCE,
    now_ms: int | None = None,
    _autocommit: bool = True,
) -> dict:
    """Upsert the family's multiple-testing row for one registered variant.

    `variants_tested` / `effective_trials` are recomputed from
    `experiment_registry` rather than incremented, so the row is idempotent and
    self-healing: replaying the same registration cannot inflate the count, and
    a row rebuilt after a crash converges to the true distinct-experiment count.

    Requires the caller to have ALREADY inserted the experiment row (this is
    called after step 10 in the gated entry point), so the current experiment is
    included in the recount.
    """
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    family_id = resolve_mt_family_id(question_ids)
    # Exact normalized match on question_ids, NOT a LIKE substring scan: a
    # substring match would fold FAM_LONG into FAM_LONG_SHORT_TRANSITIONS and
    # silently merge two unrelated families' trial counts.
    n = conn.execute(
        "SELECT COUNT(DISTINCT experiment_id) FROM experiment_registry"
        " WHERE TRIM(LOWER(COALESCE(question_ids,''))) = ?",
        (_normalize(question_ids or ""),)).fetchone()[0]
    variants = max(1, int(n))
    # A rerun of the same frozen experiment is not an extra look at the data.
    effective = variants
    adjusted = bonferroni_alpha(effective, alpha)
    existing = conn.execute(
        "SELECT created_ms FROM mt_family_registry WHERE family_id=?", (family_id,)).fetchone()
    created = int(existing[0]) if existing else now
    conn.execute(
        "INSERT INTO mt_family_registry (family_id, variants_tested, effective_trials,"
        " family_adjusted_significance, threshold_stability, researcher_freedom_score,"
        " minimum_economic_effect, schema_version, provenance, created_ms, updated_ms)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)"
        " ON CONFLICT(family_id) DO UPDATE SET"
        "  variants_tested=excluded.variants_tested,"
        "  effective_trials=excluded.effective_trials,"
        "  family_adjusted_significance=excluded.family_adjusted_significance,"
        "  updated_ms=excluded.updated_ms",
        (family_id, variants, effective, adjusted,
         _REFUSED, _REFUSED, _REFUSED, SCHEMA_VERSION, provenance, created, now),
    )
    if _autocommit:
        conn.commit()
    return {
        "family_id": family_id,
        "variants_tested": variants,
        "effective_trials": effective,
        "family_adjusted_significance": adjusted,
        "alpha_input": alpha,
        "is_rerun_of_self": is_rerun_of_self,
        "refused_fields": ["threshold_stability", "researcher_freedom_score", "minimum_economic_effect"],
        "refused_reason": "not establishable at the gate; left NULL rather than fabricated",
    }
