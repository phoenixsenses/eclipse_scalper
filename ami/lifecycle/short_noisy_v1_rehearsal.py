"""BATCH: SHORT_NOISY_BTC200K_CONFIRMED_V1 -- disposable canonicalization
rehearsal for the SHORT_NOISY known-at-safe candidate identified by the
"SHORT_NOISY KNOWN-AT OVERLAP RECONCILIATION" read-only batch.

DISPOSABLE_DB_ONLY / NO_REAL_CANONICAL_WRITE: identical safety posture to
ami/lifecycle/canonical_backfill.py and ami/lifecycle/path_migration_rehearsal.py
-- every function here takes caller-supplied connections; this module never
opens ami.warehouse.schema.DEFAULT_PATH itself, and never writes to
data/microstructure.db (liquidations are read-only source data).

FROZEN DEFINITION (operator-specified, unchanged from the read-only
reconciliation that preceded this approval):
  - source event: existing canonical ami_events ONLY, anchor = the existing
    ETH SELL>=200K canonical event population (all 252 rows used directly;
    no new event/event_family is ever created here)
  - noisy follow-on: first ETH SELL liquidation >= 50,000 notional with
    ts_ms in (anchor_ts_ms + 1m, anchor_ts_ms + 30m]
  - BTC confirmation: first BTC SELL liquidation >= 200,000 notional with
    ts_ms in (noisy_ts + 5m, noisy_ts + 30m]
  - signal_birth_ts = BTC confirmation timestamp (conf_ts) -- NEVER noisy_ts
  - direction: SHORT (fixed; not swept, not conviction-scored)
  - no hour filter / funding filter / threshold sweep / outcome-based
    selection anywhere in this identification

NOT SHORT_NOISY_BTC1M_D5_H180: that pre-existing route bakes in a fixed
D5/H180 management horizon and (per tools/research_s34_short_conviction.py's
own S5 comparison, graveyarded) used noisy_ts -- not a BTC-confirmation
timestamp -- as its entry point, a lookahead violation the operator's
forensic reconciliation identified. SETUP_ID below is a deliberately
DIFFERENT, new identity; the old route is never reused, renamed, or merged.

WHY A SEPARATE MODULE (not a `derive_signals()` route_version token): the
existing Phase 7A backfill derives one signal per (source_event_id,
route_version comma-token) pair, using ami_events.anchor_ts_ms verbatim as
signal_birth_ts for every token. This setup's signal_birth_ts is NOT
anchor_ts_ms -- it is a materially later, independently-scanned BTC
liquidation timestamp. Reusing derive_signals()/backfill_lifecycle()
unmodified would silently backdate this signal's birth to the anchor event,
re-introducing the exact lookahead-adjacent conflation this batch exists to
avoid. A parallel, narrow builder is used instead; ami_signal_lifecycle's
identity contract (generate_signal_id) and transition ledger
(insert_transition) are reused verbatim -- only the row-assembly step is new.

FIELD PROVENANCE HONESTY: ami.lifecycle.canonical_field_provenance.FIELD_PROVENANCE_SPECS
is reused for 13 of its 16 fields verbatim (their actual derivation for THIS
setup is identical to the Phase 7A backfill: source_event_id/independent_cycle_id/
symbol/evidence_layer/is_proxy/direction/terminal_ts/setup_version/timeframe/
first_known_ts/first_executable_ts/last_confirmation_ts/invalidation_ts are all
derived the same way regardless of which setup_id a signal carries). 3 fields
are NOT reused verbatim because their FACTUAL derivation differs for this
setup: signal_birth_ts (BTC-confirmation liquidation scan, not
ami_events.anchor_ts_ms), setup_id (a frozen literal identity, not a
route_version comma-split token), route_version (set equal to setup_id, not
copied from ami_events.route_version). All three overrides keep the SAME
field_classification (DETERMINISTIC_HISTORICAL_SAFE) as the base spec --
only the derivation_method/source_reference/limitations text is corrected to
match what was actually done. This is not a proxy->safe upgrade (classification
unchanged) and not a downgrade -- it is a factual correction of free-text
description for a cohort whose real derivation differs from the base spec's
assumption. Path-observation field provenance (23 fields, path_field_provenance.py)
needs NO override: those fields are computed generically from signal_birth_ts +
direction + candle data, independent of how signal_birth_ts was itself derived.
"""
from __future__ import annotations
import bisect
import hashlib
import time

from ami.enums import TradeLifecycleState
from ami.lifecycle.canonical_field_provenance import (
    FIELD_PROVENANCE_SCHEMA_VERSION,
    FIELD_PROVENANCE_SPECS,
    PROVENANCE_VERSION,
    _provenance_id,
)
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

SETUP_ID = "SHORT_NOISY_BTC200K_CONFIRMED_V1"
ROUTE_VERSION = SETUP_ID  # single-token identity; not derived from ami_events.route_version
IDENTITY_DEFINITION_VERSION = "short-noisy-btc200k-confirmed-v1"

ANCHOR_SYMBOL = "ETHUSDT"
NOISY_SYMBOL = "ETHUSDT"
CONFIRM_SYMBOL = "BTCUSDT"
NOISY_MIN_NOTIONAL = 50_000.0
BTC_CONFIRM_MIN_NOTIONAL = 200_000.0
MIN1_MS = 60_000
MIN5_MS = 5 * 60_000
MIN30_MS = 30 * 60_000

PROVENANCE_TAG = "short-noisy-v1-disposable-rehearsal"


def _sell_liquidation_ts(conn_liq, symbol: str, min_notional: float) -> list[int]:
    rows = conn_liq.execute(
        "SELECT ts_ms FROM liquidations WHERE symbol=? AND side='SELL' AND notional>=? ORDER BY ts_ms",
        (symbol, min_notional),
    ).fetchall()
    return [r[0] for r in rows]


def _first_in_window(sorted_ts: list[int], start_exclusive: int, end_inclusive: int) -> int | None:
    lo = bisect.bisect_right(sorted_ts, start_exclusive)
    hi = bisect.bisect_right(sorted_ts, end_inclusive)
    return sorted_ts[lo] if lo < hi else None


def identify_candidates(conn_events, conn_liq) -> list[dict]:
    """Deterministic, no-lookahead identification over the FULL existing
    canonical ami_events population (all 252 ETH SELL>=200K anchors, used
    directly -- no re-filtering, no new event created). Every timestamp used
    to construct a candidate's identity (anchor_ts_ms, noisy_ts, conf_ts) is
    <= conf_ts (the eventual signal_birth_ts): anchor_ts_ms is the event's own
    frozen anchor; noisy_ts is found by a forward-only scan strictly after
    anchor_ts_ms; conf_ts is found by a forward-only scan strictly after
    noisy_ts. No candidate ever uses information dated after its own conf_ts."""
    events = conn_events.execute(
        "SELECT event_id, symbol, anchor_ts_ms, source_quality FROM ami_events"
    ).fetchall()
    noisy_ts_list = _sell_liquidation_ts(conn_liq, NOISY_SYMBOL, NOISY_MIN_NOTIONAL)
    btc_ts_list = _sell_liquidation_ts(conn_liq, CONFIRM_SYMBOL, BTC_CONFIRM_MIN_NOTIONAL)

    out: list[dict] = []
    for event_id, symbol, anchor_ts_ms, source_quality in events:
        if symbol != ANCHOR_SYMBOL:
            continue
        noisy_ts = _first_in_window(noisy_ts_list, anchor_ts_ms + MIN1_MS, anchor_ts_ms + MIN30_MS)
        if noisy_ts is None:
            continue
        conf_ts = _first_in_window(btc_ts_list, noisy_ts + MIN5_MS, noisy_ts + MIN30_MS)
        if conf_ts is None:
            continue
        assert anchor_ts_ms < noisy_ts < conf_ts, "identity must be strictly forward-only"
        out.append({
            "event_id": event_id, "symbol": symbol, "anchor_ts_ms": anchor_ts_ms,
            "source_quality": source_quality, "noisy_ts": noisy_ts, "conf_ts": conf_ts,
        })
    return out


def _event_to_cycle_map(conn_events) -> dict[str, str]:
    rows = conn_events.execute(
        "SELECT event_id, candidate_cycle_key FROM event_cycle_membership "
        "WHERE cycle_definition_version='canonical-v1' AND is_canonical=1"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def build_signal_rows(candidates: list[dict], event_to_cycle: dict[str, str]) -> list[dict]:
    direction = classify_direction_from_setup_id(SETUP_ID)
    assert direction == "SHORT", f"SETUP_ID {SETUP_ID!r} must classify as SHORT, got {direction!r}"
    out = []
    for c in candidates:
        is_real = c["source_quality"] == "REAL_LIQUIDATION"
        evidence_layer = EvidenceLayer.REAL.value if is_real else EvidenceLayer.PROXY.value
        is_proxy = 0 if is_real else 1
        signal_id = generate_signal_id(
            setup_id=SETUP_ID, setup_version=SETUP_VERSION_DEFAULT, symbol=c["symbol"],
            direction=direction, source_event_id=c["event_id"],
        )
        source_hash = hashlib.sha256(
            f"{c['event_id']}|{SETUP_ID}|{c['anchor_ts_ms']}|{c['noisy_ts']}|{c['conf_ts']}".encode("utf-8")
        ).hexdigest()[:16]
        out.append({
            "signal_id": signal_id, "setup_id": SETUP_ID, "setup_version": None,
            "source_event_id": c["event_id"], "independent_cycle_id": event_to_cycle.get(c["event_id"]),
            "symbol": c["symbol"], "direction": direction, "timeframe": None,
            "route_version": ROUTE_VERSION, "signal_birth_ts": c["conf_ts"],
            "first_known_ts": None, "first_executable_ts": None, "last_confirmation_ts": None,
            "invalidation_ts": None, "terminal_ts": None,
            "evidence_layer": evidence_layer, "is_proxy": is_proxy,
            "executability_status": ExecutabilityStatus.FORWARD_ONLY.value,
            "source_hash": source_hash,
            "_anchor_ts_ms": c["anchor_ts_ms"], "_noisy_ts": c["noisy_ts"], "_conf_ts": c["conf_ts"],
        })
    return out


def backfill_short_noisy_v1(conn_target, conn_events, conn_liq,
                             provenance: str = PROVENANCE_TAG) -> dict:
    """Idempotent: rerunning against unchanged source data upserts identical
    ami_signal_lifecycle rows and inserts zero new transitions (insert_transition's
    own dedup check). Only ever writes rows for SETUP_ID -- never touches any
    other setup_id's signals/transitions."""
    candidates = identify_candidates(conn_events, conn_liq)
    event_to_cycle = _event_to_cycle_map(conn_events)
    signals = build_signal_rows(candidates, event_to_cycle)
    now = int(time.time() * 1000)

    n_signals = 0
    n_transitions = 0
    for s in signals:
        current_status = TradeLifecycleState.OPEN.value
        reason_code = LifecycleReasonCode.SIGNAL_BIRTH.value
        conn_target.execute(
            "INSERT INTO ami_signal_lifecycle (signal_id, setup_id, setup_version, source_event_id, "
            "independent_cycle_id, symbol, direction, timeframe, route_version, signal_birth_ts, "
            "first_known_ts, first_executable_ts, last_confirmation_ts, invalidation_ts, terminal_ts, "
            "lifecycle_status, lifecycle_reason_code, observation_mode, evidence_layer, is_proxy, "
            "executability_status, identity_version, schema_version, source_hash, code_commit, "
            "provenance, created_at, updated_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(signal_id) DO UPDATE SET "
            "independent_cycle_id=excluded.independent_cycle_id, "
            "source_hash=excluded.source_hash, updated_ms=excluded.updated_ms",
            (s["signal_id"], s["setup_id"], s["setup_version"], s["source_event_id"],
             s["independent_cycle_id"], s["symbol"], s["direction"], s["timeframe"], s["route_version"],
             s["signal_birth_ts"], s["first_known_ts"], s["first_executable_ts"], s["last_confirmation_ts"],
             s["invalidation_ts"], s["terminal_ts"], current_status, reason_code,
             ObservationMode.HISTORICAL_REPLAY.value, s["evidence_layer"], s["is_proxy"],
             s["executability_status"], IDENTITY_VERSION, LIFECYCLE_SCHEMA_VERSION, s["source_hash"],
             "ami/lifecycle/short_noisy_v1_rehearsal.py", provenance, now, now),
        )
        n_signals += 1
        insert_transition(
            conn_target, signal_id=s["signal_id"], previous_status=None,
            new_status=TradeLifecycleState.OPEN.value, transition_ts=s["signal_birth_ts"],
            known_at_ts=s["signal_birth_ts"], reason_code=LifecycleReasonCode.SIGNAL_BIRTH.value,
            observation_mode=ObservationMode.HISTORICAL_REPLAY.value, provenance=provenance,
        )
        n_transitions += 1

    conn_target.commit()
    return {
        "signal_ids": [s["signal_id"] for s in signals],
        "signals": signals,
        "signals_upserted": n_signals,
        "transitions_attempted": n_transitions,
        "candidate_n": len(candidates),
    }


# ---- field-level provenance: 13 fields reused verbatim, 3 corrected (see module docstring) ----

_SIGNAL_BIRTH_TS_OVERRIDE = {
    "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
    "derivation_method": "first_in_window_btc_sell_liquidation_scan",
    "source_reference": (
        "microstructure.db:liquidations (BTCUSDT, side=SELL, notional>=200000), first row with "
        "ts_ms in (noisy_ts+5m, noisy_ts+30m]; noisy_ts is itself the first ETHUSDT SELL "
        "liquidation>=50000 in (anchor_ts_ms+1m, anchor_ts_ms+30m]. NOT ami_events.anchor_ts_ms "
        "(that source_reference applies only to the Phase 7A route_version-derived backfill, "
        "not this setup)."
    ),
    "limitations": (
        "Deterministic given the frozen liquidation history and window bounds. This IS the "
        "eligible-entry timestamp for this setup (never noisy_ts, never anchor_ts_ms)."
    ),
}
_SETUP_ID_OVERRIDE = {
    "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
    "derivation_method": "frozen_literal_setup_identity",
    "source_reference": (
        "ami.lifecycle.short_noisy_v1_rehearsal.SETUP_ID -- a frozen literal, not derived by "
        "route_version comma-split; this setup_id is not necessarily a token present in the "
        "source event's own ami_events.route_version value."
    ),
    "limitations": None,
}
_ROUTE_VERSION_OVERRIDE = {
    "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
    "derivation_method": "set_equal_to_setup_id",
    "source_reference": (
        "ami.lifecycle.short_noisy_v1_rehearsal.ROUTE_VERSION (== SETUP_ID) -- NOT copied from "
        "ami_events.route_version."
    ),
    "limitations": (
        "This signal's setup is independent of whatever token list is present in the source "
        "event's own route_version column; the two are not expected to match."
    ),
}
FIELD_SPEC_OVERRIDES = {
    "signal_birth_ts": _SIGNAL_BIRTH_TS_OVERRIDE,
    "setup_id": _SETUP_ID_OVERRIDE,
    "route_version": _ROUTE_VERSION_OVERRIDE,
}


def backfill_short_noisy_v1_field_provenance(conn_target, signal_ids: list[str],
                                              provenance: str = PROVENANCE_TAG) -> dict:
    """Same table/provenance_version/idempotency contract as
    canonical_field_provenance.backfill_field_provenance() -- only the
    per-field spec text differs for the 3 overridden fields (see
    FIELD_SPEC_OVERRIDES); all field_classification values are IDENTICAL to
    the base spec (no proxy->safe upgrade, no downgrade)."""
    now = int(time.time() * 1000)
    n_written = 0
    for signal_id in signal_ids:
        for field_name, base_spec in FIELD_PROVENANCE_SPECS.items():
            spec = FIELD_SPEC_OVERRIDES.get(field_name, base_spec)
            assert spec["field_classification"] == base_spec["field_classification"], (
                f"override for {field_name!r} must not change field_classification"
            )
            pid = _provenance_id(signal_id, field_name, PROVENANCE_VERSION)
            is_proxy = 1 if spec["field_classification"] == "HISTORICAL_PROXY" else 0
            conn_target.execute(
                "INSERT INTO ami_lifecycle_field_provenance (provenance_id, signal_id, field_name, "
                "field_classification, is_proxy, derivation_method, source_reference, limitations, "
                "provenance_version, schema_version, code_commit, source_hash, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(signal_id, field_name, provenance_version) DO UPDATE SET "
                "field_classification=excluded.field_classification, is_proxy=excluded.is_proxy, "
                "derivation_method=excluded.derivation_method, source_reference=excluded.source_reference, "
                "limitations=excluded.limitations",
                (pid, signal_id, field_name, spec["field_classification"], is_proxy,
                 spec["derivation_method"], spec["source_reference"], spec["limitations"],
                 PROVENANCE_VERSION, FIELD_PROVENANCE_SCHEMA_VERSION,
                 "ami/lifecycle/short_noisy_v1_rehearsal.py", None, now),
            )
            n_written += 1
    conn_target.commit()
    return {"signals_covered": len(signal_ids), "fields_per_signal": len(FIELD_PROVENANCE_SPECS),
            "provenance_rows_written": n_written}


def rollback_short_noisy_v1(conn_target) -> dict:
    """Deletes ONLY rows tied to SETUP_ID -- every other setup's signals,
    transitions, path observations, and field-provenance rows (including the
    270 pre-existing signals and their 1080/6210/4320 rows) are untouched."""
    signal_ids = [
        r[0] for r in conn_target.execute(
            "SELECT signal_id FROM ami_signal_lifecycle WHERE setup_id=?", (SETUP_ID,)
        ).fetchall()
    ]
    n_path_obs = 0
    n_path_prov = 0
    n_lifecycle_prov = 0
    n_transitions = 0
    if signal_ids:
        placeholders = ",".join("?" for _ in signal_ids)
        n_path_obs = conn_target.execute(
            f"DELETE FROM ami_lifecycle_path_observations WHERE signal_id IN ({placeholders})",
            signal_ids,
        ).rowcount
        n_path_prov = conn_target.execute(
            f"DELETE FROM ami_lifecycle_field_provenance WHERE signal_id IN ({placeholders}) "
            f"AND field_name LIKE 'path_observations.%'",
            signal_ids,
        ).rowcount
        n_lifecycle_prov = conn_target.execute(
            f"DELETE FROM ami_lifecycle_field_provenance WHERE signal_id IN ({placeholders}) "
            f"AND field_name NOT LIKE 'path_observations.%'",
            signal_ids,
        ).rowcount
        n_transitions = conn_target.execute(
            f"DELETE FROM ami_lifecycle_transitions WHERE signal_id IN ({placeholders})",
            signal_ids,
        ).rowcount
    n_signals = conn_target.execute(
        "DELETE FROM ami_signal_lifecycle WHERE setup_id=?", (SETUP_ID,)
    ).rowcount
    conn_target.commit()
    return {
        "signals_deleted": n_signals, "transitions_deleted": n_transitions,
        "path_observations_deleted": n_path_obs, "path_provenance_deleted": n_path_prov,
        "lifecycle_provenance_deleted": n_lifecycle_prov,
    }
