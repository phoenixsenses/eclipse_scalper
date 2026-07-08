"""PHASE 7A.1: Disposable, deterministic historical-safe backfill for the
canonical lifecycle tables (ami/lifecycle/canonical_schema.py).

DISPOSABLE_DB_ONLY: `conn_source` here is ALWAYS a caller-supplied
connection -- this module never opens ami.warehouse.schema.DEFAULT_PATH
itself. Callers in this batch only ever pass a disposable-copy connection
(see ami/lifecycle/migration_rehearsal.py). NO_REAL_CANONICAL_DB_BACKFILL.

NOT the same population as ami/lifecycle/engine.py's LONG_HOUR17_*/
LONG_SILENCE shadow-ledger replay -- this backfill reconstructs canonical
lifecycle identity for the Phase 1-6 ami_events/ami_cycles (SELL-cascade)
population directly from the warehouse tables (no shadow-ledger JSONL
parsing here). The existing LONG_HOUR17_*/LONG_SILENCE bindings are not
touched, extended, or reconciled by this module (deferred, per Phase 7-8
audit's own scoping note).

NO_TIMING_PATH_LABEL_ENGINE: only two transitions are ever backfilled per
signal -- SIGNAL_BIRTH (None->OPEN) and, when a terminal_ts is known,
TERMINAL_CLOSE (OPEN->CLOSED). No HEALTHY/ACCELERATING/STALLING/etc path
labels are computed here (that is Phase 7B's job, not approved yet).

FIELD_CLASSIFICATION documents, per field, exactly how its backfilled value
was derived (or why it is left NULL) -- see module docstring for the full
rationale of each entry.
"""
from __future__ import annotations
import hashlib
import time

from ami.enums import TradeLifecycleState
from ami.lifecycle.canonical_schema import (
    IDENTITY_VERSION,
    LIFECYCLE_SCHEMA_VERSION,
    SETUP_VERSION_DEFAULT,
    EvidenceLayer,
    ExecutabilityStatus,
    LifecycleReasonCode,
    ObservationMode,
    classify_direction_from_setup_id,
    generate_signal_id,
    insert_transition,
)

# [PHASE 7A-P1 semantic closure, additional identity contract] see
# canonical_schema.UNKNOWN_SETUP_VERSION_TOKEN docstring: SETUP_VERSION_DEFAULT
# is a placeholder identity-hash input, never a real observed setup version.
from ami.lifecycle.canonical_schema import UNKNOWN_SETUP_VERSION_TOKEN  # noqa: F401 (re-exported for callers)

FIELD_CLASSIFICATION = {
    "source_event_id": "DETERMINISTIC_HISTORICAL_SAFE",       # ami_events.event_id, verbatim
    "independent_cycle_id": "DETERMINISTIC_HISTORICAL_SAFE",   # event_cycle_membership canonical-v1, verbatim
    "signal_birth_ts": "DETERMINISTIC_HISTORICAL_SAFE",        # ami_events.anchor_ts_ms, verbatim
    "terminal_ts": "NOT_IMPLEMENTED",       # [PHASE 7A-P1 semantic closure] ami_events.event_end_ts_ms
                                             # is a source-event-level concept (last associated trade
                                             # close), never validated as the true lifecycle TERMINAL
                                             # transition time -- no longer copied into this column
    "symbol": "DETERMINISTIC_HISTORICAL_SAFE",
    "setup_id": "DETERMINISTIC_HISTORICAL_SAFE",               # ami_events.route_version token, verbatim
    "route_version": "DETERMINISTIC_HISTORICAL_SAFE",
    "evidence_layer": "DETERMINISTIC_HISTORICAL_SAFE",         # ami_events.source_quality, verbatim
    "is_proxy": "DETERMINISTIC_HISTORICAL_SAFE",
    "setup_version": "NOT_IMPLEMENTED",     # no per-route rule/parameter version exists upstream;
                                             # frozen placeholder constant (SETUP_VERSION_DEFAULT) used
    "direction": "HISTORICAL_PROXY",        # heuristic prefix-parse of setup_id (LONG_/SHORT_/BUY_FADE);
                                             # no canonical route-direction registry exists -- 'UNKNOWN'
                                             # returned rather than guessed for unrecognized prefixes
    "timeframe": "NOT_IMPLEMENTED",         # no canonical per-signal timeframe assignment exists yet
    "first_known_ts": "NOT_IMPLEMENTED",    # ami_events.feature_available_ts_ms reserved but never
                                             # computed anywhere upstream (always NULL)
    "first_executable_ts": "FORWARD_ONLY",  # true live-executability timestamp cannot be reconstructed
                                             # from candle/mark data after the fact -- left NULL, per instruction
    "last_confirmation_ts": "NOT_IMPLEMENTED",  # "confirmation" is undefined without a path/progress
                                                 # engine (explicitly out of scope this batch)
    "invalidation_ts": "NOT_IMPLEMENTED",       # same reason
}


def fetch_source_events(conn_source) -> list[dict]:
    """Direct warehouse-layer read (this is migration/backfill infrastructure,
    not a Phase 6 research engine -- feature_gateway's mandate covers
    research population assembly, not schema backfill code; existing seed
    scripts like cycle_resolver.py/candidate_universe.py follow the same
    direct-SQL convention)."""
    cols = ["event_id", "symbol", "anchor_ts_ms", "event_end_ts_ms", "source_quality", "route_version"]
    rows = conn_source.execute(f"SELECT {', '.join(cols)} FROM ami_events").fetchall()
    events = [dict(zip(cols, r)) for r in rows]

    membership = conn_source.execute(
        "SELECT event_id, candidate_cycle_key FROM event_cycle_membership "
        "WHERE cycle_definition_version='canonical-v1' AND is_canonical=1"
    ).fetchall()
    event_to_cycle = {r[0]: r[1] for r in membership}
    for e in events:
        e["independent_cycle_id"] = event_to_cycle.get(e["event_id"])
    return events


def _setup_tokens(route_version: str | None) -> list[str]:
    if not route_version:
        return ["UNKNOWN_SETUP"]
    return [t.strip() for t in route_version.split(",") if t.strip()] or ["UNKNOWN_SETUP"]


def derive_signals(events: list[dict]) -> list[dict]:
    """One signal per (source_event_id, setup token) pair -- an anchor event
    with N attached routes (real data: some events carry BOTH LONG_* and
    SHORT_* route tokens simultaneously, since independent strategies watch
    the same cascade and decide independently) yields N signal rows, not 1.
    This supersedes the earlier "signal_id=event_id 1:1" simplification
    floated in the Phase 7-8 audit -- that simplification could not carry a
    single well-defined `direction` per the operator's identity contract,
    so it is not used here (documented, not silently dropped)."""
    out = []
    for e in events:
        is_real = e["source_quality"] == "REAL_LIQUIDATION"
        evidence_layer = EvidenceLayer.REAL.value if is_real else EvidenceLayer.PROXY.value
        is_proxy = 0 if is_real else 1
        for setup_id in _setup_tokens(e["route_version"]):
            direction = classify_direction_from_setup_id(setup_id)
            signal_id = generate_signal_id(
                setup_id=setup_id, setup_version=SETUP_VERSION_DEFAULT, symbol=e["symbol"],
                direction=direction, source_event_id=e["event_id"],
            )
            source_hash = hashlib.sha256(
                f"{e['event_id']}|{setup_id}|{e['anchor_ts_ms']}|{e['event_end_ts_ms']}".encode("utf-8")
            ).hexdigest()[:16]
            out.append({
                "signal_id": signal_id,
                "setup_id": setup_id,
                # [PHASE 7A-P1 semantic closure] setup_version is intentionally
                # None here (canonical column) -- no real applied per-route
                # versioning source exists. SETUP_VERSION_DEFAULT is still
                # used ABOVE as an identity-hash input (signal_id stability),
                # but identity_version and setup_version are distinct
                # concepts and are never conflated in the stored data.
                "setup_version": None,
                "source_event_id": e["event_id"],
                "independent_cycle_id": e["independent_cycle_id"],
                "symbol": e["symbol"],
                "direction": direction,
                "timeframe": None,
                "route_version": e["route_version"],
                "signal_birth_ts": e["anchor_ts_ms"],
                "first_known_ts": None,
                "first_executable_ts": None,
                "last_confirmation_ts": None,
                "invalidation_ts": None,
                # [PHASE 7A-P1 semantic closure, round 2] terminal_ts is None
                # (canonical column) -- ami_events.event_end_ts_ms is a
                # distinct source-event-level concept (last associated trade
                # close), not a validated lifecycle TERMINAL transition time.
                # Round 1 of this closure had already nulled this column but
                # left a contradiction in place: the ledger below still wrote
                # a TERMINAL_CLOSE transition (OPEN->CLOSED) using
                # event_end_ts_ms as its evidentiary basis, so
                # rebuild_current_state() still asserted CLOSED off the very
                # same unvalidated timestamp terminal_ts refused to echo.
                # Corrected here: this backfill no longer derives ANY
                # lifecycle_status/transition claim from event_end_ts_ms --
                # every freshly-backfilled signal starts and stays OPEN
                # (SIGNAL_BIRTH only) until a genuinely validated terminal
                # detection mechanism exists (future phase, NOT_IMPLEMENTED
                # here). event_end_ts_ms remains available, unmodified, via
                # a direct ami_events join (source-event semantics) -- it is
                # never lost, only no longer conflated with signal lifecycle
                # termination. See correct_unvalidated_terminal_close() for
                # the companion retroactive fix on ledgers that already
                # contain the old (pre-correction) TERMINAL_CLOSE rows.
                "terminal_ts": None,
                "evidence_layer": evidence_layer,
                "is_proxy": is_proxy,
                "executability_status": ExecutabilityStatus.FORWARD_ONLY.value,
                "source_hash": source_hash,
            })
    return out


def backfill_lifecycle(conn_target, conn_source,
                        provenance: str = "phase-7a1-disposable-backfill") -> dict:
    """Idempotent: rerunning against unchanged source data upserts identical
    values (row count and content hash unchanged) and inserts zero NEW
    transitions (insert_transition's own dedup check makes re-inserting the
    same genesis/terminal transition a no-op). Never touches
    data/ami/research.sqlite or forward_bindings/processed_trades -- this
    module has no import of ami.research.forward_pipeline at all."""
    events = fetch_source_events(conn_source)
    signals = derive_signals(events)
    now = int(time.time() * 1000)

    n_signals = 0
    n_transitions = 0
    for s in signals:
        # [PHASE 7A-P1 semantic closure, round 2] No lifecycle_status/reason_code/
        # transition claim is ever derived from ami_events.event_end_ts_ms --
        # every signal is backfilled as OPEN/SIGNAL_BIRTH only. A genuinely
        # validated terminal-detection mechanism is a future, separately
        # approved phase (NOT_IMPLEMENTED here); until it exists, asserting
        # CLOSED would overclaim knowledge this backfill does not have.
        current_status = TradeLifecycleState.OPEN.value
        reason_code = LifecycleReasonCode.SIGNAL_BIRTH.value
        # [PHASE 7A-P canonical migration -- idempotency fix] lifecycle_status/
        # lifecycle_reason_code are DELIBERATELY EXCLUDED from the ON CONFLICT
        # UPDATE SET below (a dry-run against a disposable copy of the real DB
        # caught this): once correct_unvalidated_terminal_close() has upgraded
        # a signal's denormalized status/reason to (OPEN, CORRECTION), a LATER
        # rerun of this function must not clobber it back to (OPEN,
        # SIGNAL_BIRTH) -- these two columns are the ledger-derived cache
        # that only the transition ledger (genesis on first INSERT, or an
        # explicit correction) may set; a plain backfill rerun leaves them
        # untouched on conflict.
        conn_target.execute(
            "INSERT INTO ami_signal_lifecycle (signal_id, setup_id, setup_version, source_event_id, "
            "independent_cycle_id, symbol, direction, timeframe, route_version, signal_birth_ts, "
            "first_known_ts, first_executable_ts, last_confirmation_ts, invalidation_ts, terminal_ts, "
            "lifecycle_status, lifecycle_reason_code, observation_mode, evidence_layer, is_proxy, "
            "executability_status, identity_version, schema_version, source_hash, code_commit, "
            "provenance, created_at, updated_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(signal_id) DO UPDATE SET "
            "setup_version=excluded.setup_version, "
            "terminal_ts=excluded.terminal_ts, independent_cycle_id=excluded.independent_cycle_id, "
            "source_hash=excluded.source_hash, updated_ms=excluded.updated_ms",
            (s["signal_id"], s["setup_id"], s["setup_version"], s["source_event_id"],
             s["independent_cycle_id"], s["symbol"], s["direction"], s["timeframe"], s["route_version"],
             s["signal_birth_ts"], s["first_known_ts"], s["first_executable_ts"], s["last_confirmation_ts"],
             s["invalidation_ts"], s["terminal_ts"], current_status, reason_code,
             ObservationMode.HISTORICAL_REPLAY.value, s["evidence_layer"], s["is_proxy"],
             s["executability_status"], IDENTITY_VERSION, LIFECYCLE_SCHEMA_VERSION, s["source_hash"],
             "ami/lifecycle/canonical_backfill.py", provenance, now, now),
        )
        n_signals += 1

        insert_transition(
            conn_target, signal_id=s["signal_id"], previous_status=None,
            new_status=TradeLifecycleState.OPEN.value, transition_ts=s["signal_birth_ts"],
            known_at_ts=s["signal_birth_ts"], reason_code=LifecycleReasonCode.SIGNAL_BIRTH.value,
            observation_mode=ObservationMode.HISTORICAL_REPLAY.value, provenance=provenance,
        )
        n_transitions += 1
        # [PHASE 7A-P1 semantic closure, round 2] no second (TERMINAL_CLOSE)
        # transition is ever written here -- see the comment above and
        # correct_unvalidated_terminal_close() below for retroactively fixing
        # ledgers that already contain the old, pre-correction TERMINAL_CLOSE rows.

    conn_target.commit()
    return {"signals_upserted": n_signals, "transitions_attempted": n_transitions,
            "source_event_n": len(events)}


def correct_unvalidated_terminal_close(
    conn, provenance: str = "phase-7a-p1-terminal-semantic-correction",
) -> dict:
    """[PHASE 7A-P1 semantic closure, round 2] Retroactive companion fix for
    `backfill_lifecycle()`'s prior (corrected above) behavior: finds every
    ami_lifecycle_transitions row that ever asserted a terminal CLOSED status
    via reason_code=TERMINAL_CLOSE (the ONLY code path that historically ever
    wrote that reason_code -- grep-verified, no other writer exists) and, for
    every signal whose CURRENT (ledger-derived) state is still CLOSED because
    of it, appends an explicit CORRECTION transition (CLOSED->OPEN,
    transition_version=2, correction_of=<original transition_id>). The
    original TERMINAL_CLOSE row is NEVER updated or deleted (append-only
    ledger discipline preserved) -- this only adds a new row on top, exactly
    like the correction pattern already established in
    canonical_schema.py's docstring/tests (`validate=False` is the documented,
    sanctioned bypass for reversing a terminal status precisely because a
    correction revises a past claim rather than extending the live chain).
    Also updates the denormalized ami_signal_lifecycle.lifecycle_status/
    lifecycle_reason_code columns so they never diverge from what
    rebuild_current_state() would now compute from the ledger.
    Idempotent: a signal whose latest recorded status is already OPEN
    (either never had a TERMINAL_CLOSE, or was already corrected by a prior
    run of this function) is skipped -- reruns insert zero new rows."""
    rows = conn.execute(
        "SELECT transition_id, signal_id, transition_ts FROM ami_lifecycle_transitions "
        "WHERE reason_code=? AND new_status=?",
        (LifecycleReasonCode.TERMINAL_CLOSE.value, TradeLifecycleState.CLOSED.value),
    ).fetchall()
    now = int(time.time() * 1000)
    n_corrected = 0
    n_already_open = 0
    for transition_id, signal_id, transition_ts in rows:
        latest = conn.execute(
            "SELECT new_status FROM ami_lifecycle_transitions WHERE signal_id=? "
            "ORDER BY transition_ts DESC, transition_version DESC LIMIT 1",
            (signal_id,),
        ).fetchone()
        if latest is None or latest[0] != TradeLifecycleState.CLOSED.value:
            n_already_open += 1
            continue  # already corrected by a prior run, or superseded by a later transition
        known_at_ts = max(transition_ts, now)
        insert_transition(
            conn, signal_id=signal_id, previous_status=TradeLifecycleState.CLOSED.value,
            new_status=TradeLifecycleState.OPEN.value, transition_ts=transition_ts,
            known_at_ts=known_at_ts, reason_code=LifecycleReasonCode.CORRECTION.value,
            observation_mode=ObservationMode.HISTORICAL_REPLAY.value, transition_version=2,
            correction_of=transition_id, provenance=provenance, validate=False,
        )
        conn.execute(
            "UPDATE ami_signal_lifecycle SET lifecycle_status=?, lifecycle_reason_code=?, updated_ms=? "
            "WHERE signal_id=?",
            (TradeLifecycleState.OPEN.value, LifecycleReasonCode.CORRECTION.value, now, signal_id),
        )
        n_corrected += 1
    conn.commit()
    return {"terminal_close_transitions_found": len(rows), "signals_corrected": n_corrected,
            "already_open_or_corrected": n_already_open}
