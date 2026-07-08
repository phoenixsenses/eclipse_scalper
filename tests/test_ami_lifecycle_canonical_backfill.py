"""PHASE 7A.1: tests for ami/lifecycle/canonical_backfill.py.
DISPOSABLE_DB_ONLY: source events are always synthetic sqlite3 fixtures
(:memory: or tmp_path); the real data/ami/canonical.sqlite is never opened
for writing here.

Run: pytest tests/test_ami_lifecycle_canonical_backfill.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import ast
import sqlite3
from pathlib import Path

from ami.lifecycle.canonical_backfill import (
    FIELD_CLASSIFICATION,
    backfill_lifecycle,
    correct_unvalidated_terminal_close,
    derive_signals,
    fetch_source_events,
)
from ami.lifecycle.canonical_schema import (
    UNKNOWN_SETUP_VERSION_TOKEN,
    SETUP_VERSION_DEFAULT,
    generate_signal_id,
    init_lifecycle_schema,
    insert_transition,
    rebuild_current_state,
)

BACKFILL_SRC_PATH = Path(__file__).resolve().parents[1] / "ami" / "lifecycle" / "canonical_backfill.py"


def _mk_source_db(path, events, memberships=()):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE ami_events (event_id TEXT PRIMARY KEY, symbol TEXT, anchor_ts_ms INTEGER, "
        "event_end_ts_ms INTEGER, source_quality TEXT, route_version TEXT)"
    )
    conn.execute(
        "CREATE TABLE event_cycle_membership (event_id TEXT, candidate_cycle_key TEXT, "
        "cycle_definition_version TEXT, is_canonical INTEGER)"
    )
    for e in events:
        conn.execute(
            "INSERT INTO ami_events VALUES (?,?,?,?,?,?)",
            (e["event_id"], e["symbol"], e["anchor_ts_ms"], e.get("event_end_ts_ms"),
             e.get("source_quality", "REAL_LIQUIDATION"), e.get("route_version")),
        )
    for m in memberships:
        conn.execute(
            "INSERT INTO event_cycle_membership VALUES (?,?,?,?)",
            (m["event_id"], m["candidate_cycle_key"], "canonical-v1", 1),
        )
    conn.commit()
    return conn


# ---- LONG/SHORT identity symmetry (event with both attached routes) ----

def test_event_with_long_and_short_routes_yields_two_distinct_signals():
    events = [{"event_id": "EVT-1", "symbol": "ETHUSDT", "anchor_ts_ms": 1000,
               "event_end_ts_ms": 2000, "route_version": "LONG_SILENCE,SHORT_NEITHER",
               "source_quality": "REAL_LIQUIDATION", "independent_cycle_id": None}]
    signals = derive_signals(events)
    assert len(signals) == 2
    directions = {s["direction"] for s in signals}
    assert directions == {"LONG", "SHORT"}
    assert len({s["signal_id"] for s in signals}) == 2  # distinct identities


# ---- event-less / route-less signal handling ----

def test_event_with_no_route_version_gets_unknown_setup():
    events = [{"event_id": "EVT-2", "symbol": "ETHUSDT", "anchor_ts_ms": 1000,
               "event_end_ts_ms": None, "route_version": None,
               "source_quality": "REAL_LIQUIDATION", "independent_cycle_id": None}]
    signals = derive_signals(events)
    assert len(signals) == 1
    assert signals[0]["setup_id"] == "UNKNOWN_SETUP"
    assert signals[0]["direction"] == "UNKNOWN"


# ---- historical/proxy/forward separation ----

def test_evidence_layer_reflects_source_quality():
    events = [
        {"event_id": "EVT-REAL", "symbol": "ETHUSDT", "anchor_ts_ms": 1000, "event_end_ts_ms": 2000,
         "route_version": "LONG_SILENCE", "source_quality": "REAL_LIQUIDATION", "independent_cycle_id": None},
        {"event_id": "EVT-PROXY", "symbol": "ETHUSDT", "anchor_ts_ms": 1000, "event_end_ts_ms": 2000,
         "route_version": "LONG_SILENCE", "source_quality": "PROXY_OTHER", "independent_cycle_id": None},
    ]
    signals = derive_signals(events)
    by_event = {s["source_event_id"]: s for s in signals}
    assert by_event["EVT-REAL"]["evidence_layer"] == "REAL" and by_event["EVT-REAL"]["is_proxy"] == 0
    assert by_event["EVT-PROXY"]["evidence_layer"] == "PROXY" and by_event["EVT-PROXY"]["is_proxy"] == 1


# ---- FORWARD_ONLY fields remain NULL; missing data stays explicit (not zero) ----

def test_forward_only_and_not_implemented_fields_stay_null():
    events = [{"event_id": "EVT-1", "symbol": "ETHUSDT", "anchor_ts_ms": 1000,
               "event_end_ts_ms": 2000, "route_version": "LONG_SILENCE",
               "source_quality": "REAL_LIQUIDATION", "independent_cycle_id": None}]
    signals = derive_signals(events)
    s = signals[0]
    assert s["first_known_ts"] is None
    assert s["first_executable_ts"] is None
    assert s["last_confirmation_ts"] is None
    assert s["invalidation_ts"] is None
    assert s["executability_status"] == "FORWARD_ONLY"
    # explicitly NOT zero -- missing must never silently become a valid-looking 0
    assert s["first_known_ts"] != 0
    assert s["first_executable_ts"] != 0


def test_field_classification_matrix_covers_all_forward_only_and_not_implemented_fields():
    for field in ("first_executable_ts",):
        assert FIELD_CLASSIFICATION[field] == "FORWARD_ONLY"
    for field in ("setup_version", "timeframe", "first_known_ts", "last_confirmation_ts",
                  "invalidation_ts", "terminal_ts"):
        assert FIELD_CLASSIFICATION[field] == "NOT_IMPLEMENTED"
    assert FIELD_CLASSIFICATION["direction"] == "HISTORICAL_PROXY"
    for field in ("source_event_id", "independent_cycle_id", "signal_birth_ts", "symbol"):
        assert FIELD_CLASSIFICATION[field] == "DETERMINISTIC_HISTORICAL_SAFE"


# ---- book-gap executability remains blocked ----

def test_executability_status_always_forward_only_this_batch():
    # NO_TIMING_PATH_LABEL_ENGINE / book_ticker coverage unresolved (Phase
    # 7-8 audit finding) -- every backfilled signal must carry
    # executability_status=FORWARD_ONLY, never a fabricated EXECUTABLE claim.
    events = [{"event_id": f"EVT-{i}", "symbol": "ETHUSDT", "anchor_ts_ms": 1000 + i,
               "event_end_ts_ms": 2000 + i, "route_version": "LONG_SILENCE",
               "source_quality": "REAL_LIQUIDATION", "independent_cycle_id": None} for i in range(5)]
    signals = derive_signals(events)
    assert all(s["executability_status"] == "FORWARD_ONLY" for s in signals)


# ---- backfill rerun idempotency (disposable fixture) ----

def test_backfill_rerun_idempotent_row_counts(tmp_path):
    source_path = tmp_path / "source.sqlite"
    conn_source = _mk_source_db(
        source_path,
        events=[
            {"event_id": "EVT-1", "symbol": "ETHUSDT", "anchor_ts_ms": 1000,
             "event_end_ts_ms": 2000, "route_version": "LONG_SILENCE,SHORT_NEITHER"},
            {"event_id": "EVT-2", "symbol": "ETHUSDT", "anchor_ts_ms": 3000,
             "event_end_ts_ms": None, "route_version": "LONG_SILENCE"},
        ],
        memberships=[{"event_id": "EVT-1", "candidate_cycle_key": "CYC-1"}],
    )
    conn_target = sqlite3.connect(tmp_path / "target.sqlite")
    init_lifecycle_schema(conn_target)

    r1 = backfill_lifecycle(conn_target, conn_source)
    n_sig_1 = conn_target.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0]
    n_trn_1 = conn_target.execute("SELECT COUNT(*) FROM ami_lifecycle_transitions").fetchone()[0]

    r2 = backfill_lifecycle(conn_target, conn_source)
    n_sig_2 = conn_target.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0]
    n_trn_2 = conn_target.execute("SELECT COUNT(*) FROM ami_lifecycle_transitions").fetchone()[0]

    assert r1["signals_upserted"] == 3  # EVT-1 has 2 routes, EVT-2 has 1
    assert n_sig_1 == n_sig_2 == 3
    # [PHASE 7A-P1 semantic closure, round 2] every signal gets genesis (SIGNAL_BIRTH)
    # only -- no TERMINAL_CLOSE is ever derived from event_end_ts_ms (EVT-1 having
    # one set makes no difference to transition count)
    assert n_trn_1 == n_trn_2
    assert n_trn_1 == 3

    # independent_cycle_id correctly attached only to EVT-1's signals
    cyc = {r[0] for r in conn_target.execute(
        "SELECT independent_cycle_id FROM ami_signal_lifecycle WHERE source_event_id='EVT-1'"
    ).fetchall()}
    assert cyc == {"CYC-1"}
    cyc_none = {r[0] for r in conn_target.execute(
        "SELECT independent_cycle_id FROM ami_signal_lifecycle WHERE source_event_id='EVT-2'"
    ).fetchall()}
    assert cyc_none == {None}


# ---- historical backfill does not increase FORWARD_N ----

def test_backfill_module_has_no_forward_pipeline_dependency():
    tree = ast.parse(BACKFILL_SRC_PATH.read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules += [a.name for a in node.names]
    assert not any("forward_pipeline" in m for m in modules)
    assert not any(m.startswith(("execution", "risk", "brain")) for m in modules)


def test_real_forward_bindings_processed_trades_unaffected_by_backfill(tmp_path):
    # Runs the REAL backfill code against a disposable fixture, then verifies
    # (read-only) that the real research.sqlite:processed_trades count is
    # untouched -- backfill_lifecycle has no path to that store at all.
    from ami.research.registry import DEFAULT_PATH as REAL_RESEARCH_PATH

    before = sqlite3.connect(f"file:{REAL_RESEARCH_PATH}?mode=ro", uri=True).execute(
        "SELECT COUNT(*) FROM processed_trades"
    ).fetchone()[0]

    source_path = tmp_path / "source.sqlite"
    conn_source = _mk_source_db(source_path, events=[
        {"event_id": "EVT-1", "symbol": "ETHUSDT", "anchor_ts_ms": 1000,
         "event_end_ts_ms": 2000, "route_version": "LONG_SILENCE"},
    ])
    conn_target = sqlite3.connect(tmp_path / "target.sqlite")
    init_lifecycle_schema(conn_target)
    backfill_lifecycle(conn_target, conn_source)

    after = sqlite3.connect(f"file:{REAL_RESEARCH_PATH}?mode=ro", uri=True).execute(
        "SELECT COUNT(*) FROM processed_trades"
    ).fetchone()[0]
    assert after == before


# ---- disposable DB only ----

def test_backfill_never_imports_default_canonical_path():
    tree = ast.parse(BACKFILL_SRC_PATH.read_text(encoding="utf-8"))
    imported_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_names += [a.name for a in node.names]
    assert "DEFAULT_PATH" not in imported_names


# ---- real-data smoke (read-only source, disposable target) ----

def test_fetch_source_events_real_data_smoke():
    conn = sqlite3.connect("file:data/ami/canonical.sqlite?mode=ro", uri=True)
    try:
        events = fetch_source_events(conn)
    finally:
        conn.close()
    assert len(events) > 0
    assert all(e["source_quality"] == "REAL_LIQUIDATION" for e in events)
    signals = derive_signals(events)
    assert len(signals) >= len(events)  # some events have >1 attached route


# ---- PHASE 7A-P canonical migration idempotency fix: backfill_lifecycle
# rerun must never clobber a correction's (OPEN, CORRECTION) denormalized
# status back to (OPEN, SIGNAL_BIRTH) -- caught by a pre-migration dry run
# against a disposable copy of the real canonical.sqlite ----

def test_backfill_rerun_after_correction_does_not_clobber_reason_code(tmp_path):
    source_path = tmp_path / "source.sqlite"
    conn_source = _mk_source_db(source_path, events=[
        {"event_id": "EVT-1", "symbol": "ETHUSDT", "anchor_ts_ms": 1000,
         "event_end_ts_ms": 2000, "route_version": "LONG_SILENCE"},
    ])
    conn_target = sqlite3.connect(tmp_path / "target.sqlite")
    init_lifecycle_schema(conn_target)
    backfill_lifecycle(conn_target, conn_source)

    # simulate the legacy (pre-fix) ledger state: a TERMINAL_CLOSE row
    # inherited from before this correction existed
    signal_id = conn_target.execute("SELECT signal_id FROM ami_signal_lifecycle").fetchone()[0]
    insert_transition(conn_target, signal_id, "OPEN", "CLOSED", 2000, 2000, "TERMINAL_CLOSE",
                      "HISTORICAL_REPLAY", provenance="legacy-fixture")
    conn_target.execute(
        "UPDATE ami_signal_lifecycle SET lifecycle_status='CLOSED', lifecycle_reason_code='TERMINAL_CLOSE' "
        "WHERE signal_id=?", (signal_id,))
    conn_target.commit()

    correct_unvalidated_terminal_close(conn_target)
    status_after_correction = conn_target.execute(
        "SELECT lifecycle_status, lifecycle_reason_code FROM ami_signal_lifecycle WHERE signal_id=?", (signal_id,)
    ).fetchone()
    assert status_after_correction == ("OPEN", "CORRECTION")

    # rerun backfill_lifecycle (e.g. a future incremental backfill run) --
    # must NOT reset lifecycle_reason_code back to SIGNAL_BIRTH
    backfill_lifecycle(conn_target, conn_source)
    status_after_rerun = conn_target.execute(
        "SELECT lifecycle_status, lifecycle_reason_code FROM ami_signal_lifecycle WHERE signal_id=?", (signal_id,)
    ).fetchone()
    assert status_after_rerun == ("OPEN", "CORRECTION")  # unchanged, not clobbered


# ---- PHASE 7A-P1 semantic closure, round 2: terminal_ts/lifecycle_status
# consistency (event_end_ts_ms must never drive a CLOSED/TERMINAL_CLOSE claim) ----

def test_fresh_backfill_never_writes_terminal_close_even_with_event_end_ts(tmp_path):
    source_path = tmp_path / "source.sqlite"
    conn_source = _mk_source_db(source_path, events=[
        {"event_id": "EVT-1", "symbol": "ETHUSDT", "anchor_ts_ms": 1000,
         "event_end_ts_ms": 2000, "route_version": "LONG_SILENCE"},
    ])
    conn_target = sqlite3.connect(tmp_path / "target.sqlite")
    init_lifecycle_schema(conn_target)
    backfill_lifecycle(conn_target, conn_source)

    row = conn_target.execute(
        "SELECT lifecycle_status, lifecycle_reason_code, terminal_ts FROM ami_signal_lifecycle"
    ).fetchone()
    assert row == ("OPEN", "SIGNAL_BIRTH", None)

    reason_codes = {r[0] for r in conn_target.execute(
        "SELECT DISTINCT reason_code FROM ami_lifecycle_transitions"
    ).fetchall()}
    assert reason_codes == {"SIGNAL_BIRTH"}  # TERMINAL_CLOSE never written

    signal_id = conn_target.execute("SELECT signal_id FROM ami_signal_lifecycle").fetchone()[0]
    rebuilt = rebuild_current_state(conn_target, signal_id)
    assert rebuilt["current_status"] == "OPEN"  # ledger and denormalized column agree


# ---- correct_unvalidated_terminal_close: retroactive fix for ledgers that
# already contain the old (pre-correction) TERMINAL_CLOSE rows ----

def _seed_pre_correction_closed_signal(conn, signal_id="SIG-OLD"):
    """Simulates the state a ledger was left in by the OLD (pre-fix)
    backfill_lifecycle() -- a signal with genesis + an unvalidated
    TERMINAL_CLOSE transition, and lifecycle_status/terminal_ts already
    matching that (now-corrected-elsewhere) claim."""
    now = 0
    conn.execute(
        "INSERT INTO ami_signal_lifecycle (signal_id, setup_id, setup_version, source_event_id, "
        "independent_cycle_id, symbol, direction, timeframe, route_version, signal_birth_ts, "
        "first_known_ts, first_executable_ts, last_confirmation_ts, invalidation_ts, terminal_ts, "
        "lifecycle_status, lifecycle_reason_code, observation_mode, evidence_layer, is_proxy, "
        "executability_status, identity_version, schema_version, source_hash, code_commit, "
        "provenance, created_at, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (signal_id, "LONG_SILENCE", None, "EVT-OLD", None, "ETHUSDT", "LONG", None,
         "LONG_SILENCE", 1000, None, None, None, None, None, "CLOSED", "TERMINAL_CLOSE",
         "HISTORICAL_REPLAY", "REAL", 0, "FORWARD_ONLY", "signal-identity-v1", 1, "h", "test",
         "pre-correction-fixture", now, now),
    )
    insert_transition(conn, signal_id, None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY",
                      provenance="pre-correction-fixture")
    tid = insert_transition(conn, signal_id, "OPEN", "CLOSED", 2000, 2000, "TERMINAL_CLOSE",
                            "HISTORICAL_REPLAY", provenance="pre-correction-fixture")
    conn.commit()
    return tid


def test_correct_unvalidated_terminal_close_reverses_pre_existing_closed_status():
    conn = sqlite3.connect(":memory:")
    init_lifecycle_schema(conn)
    original_tid = _seed_pre_correction_closed_signal(conn, "SIG-OLD")

    report = correct_unvalidated_terminal_close(conn)
    assert report == {"terminal_close_transitions_found": 1, "signals_corrected": 1,
                       "already_open_or_corrected": 0}

    rebuilt = rebuild_current_state(conn, "SIG-OLD")
    assert rebuilt["current_status"] == "OPEN"

    row = conn.execute(
        "SELECT lifecycle_status, lifecycle_reason_code FROM ami_signal_lifecycle WHERE signal_id='SIG-OLD'"
    ).fetchone()
    assert row == ("OPEN", "CORRECTION")

    # original TERMINAL_CLOSE row is untouched (append-only) -- proof this is
    # a correction, not an UPDATE/DELETE
    original_row = conn.execute(
        "SELECT previous_status, new_status, reason_code, transition_ts FROM ami_lifecycle_transitions "
        "WHERE transition_id=?", (original_tid,),
    ).fetchone()
    assert original_row == ("OPEN", "CLOSED", "TERMINAL_CLOSE", 2000)

    # the new correction row references the original via correction_of
    correction_row = conn.execute(
        "SELECT previous_status, new_status, reason_code, transition_version, correction_of "
        "FROM ami_lifecycle_transitions WHERE signal_id='SIG-OLD' AND reason_code='CORRECTION'"
    ).fetchone()
    assert correction_row == ("CLOSED", "OPEN", "CORRECTION", 2, original_tid)

    n_transitions = conn.execute(
        "SELECT COUNT(*) FROM ami_lifecycle_transitions WHERE signal_id='SIG-OLD'"
    ).fetchone()[0]
    assert n_transitions == 3  # genesis + original TERMINAL_CLOSE + correction, nothing removed


def test_correct_unvalidated_terminal_close_is_idempotent():
    conn = sqlite3.connect(":memory:")
    init_lifecycle_schema(conn)
    _seed_pre_correction_closed_signal(conn, "SIG-OLD")

    r1 = correct_unvalidated_terminal_close(conn)
    n1 = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_transitions").fetchone()[0]
    r2 = correct_unvalidated_terminal_close(conn)
    n2 = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_transitions").fetchone()[0]

    assert r1["signals_corrected"] == 1
    assert r2["signals_corrected"] == 0  # already OPEN, nothing new to correct
    assert n1 == n2  # rerun inserts zero new rows


def test_correct_unvalidated_terminal_close_skips_signals_already_open():
    conn = sqlite3.connect(":memory:")
    init_lifecycle_schema(conn)
    conn.execute(
        "INSERT INTO ami_signal_lifecycle (signal_id, setup_id, setup_version, source_event_id, "
        "independent_cycle_id, symbol, direction, timeframe, route_version, signal_birth_ts, "
        "first_known_ts, first_executable_ts, last_confirmation_ts, invalidation_ts, terminal_ts, "
        "lifecycle_status, lifecycle_reason_code, observation_mode, evidence_layer, is_proxy, "
        "executability_status, identity_version, schema_version, source_hash, code_commit, "
        "provenance, created_at, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("SIG-OPEN", "LONG_SILENCE", None, "EVT-OPEN", None, "ETHUSDT", "LONG", None,
         "LONG_SILENCE", 1000, None, None, None, None, None, "OPEN", "SIGNAL_BIRTH",
         "HISTORICAL_REPLAY", "REAL", 0, "FORWARD_ONLY", "signal-identity-v1", 1, "h", "test", "x", 0, 0),
    )
    insert_transition(conn, "SIG-OPEN", None, "OPEN", 1000, 1000, "SIGNAL_BIRTH", "HISTORICAL_REPLAY",
                      provenance="x")
    conn.commit()

    report = correct_unvalidated_terminal_close(conn)
    assert report == {"terminal_close_transitions_found": 0, "signals_corrected": 0,
                       "already_open_or_corrected": 0}


# ---- UNKNOWN_SETUP_VERSION_TOKEN identity contract (setup_version=NULL
# canonically, but the identity-hash input must remain stable/unmutated) ----

def test_unknown_setup_version_token_equals_frozen_default_and_is_hash_stable():
    assert UNKNOWN_SETUP_VERSION_TOKEN == SETUP_VERSION_DEFAULT == "setup-v1"
    # signal_id computed with the token must match one computed with the
    # literal frozen constant -- proves this is a naming/contract clarification,
    # never a value change that would silently mutate existing signal_ids
    a = generate_signal_id(setup_id="LONG_SILENCE", setup_version=UNKNOWN_SETUP_VERSION_TOKEN,
                           symbol="ETHUSDT", direction="LONG", source_event_id="EVT-1")
    b = generate_signal_id(setup_id="LONG_SILENCE", setup_version=SETUP_VERSION_DEFAULT,
                           symbol="ETHUSDT", direction="LONG", source_event_id="EVT-1")
    assert a == b


def test_derive_signals_setup_version_column_none_but_identity_hash_uses_token():
    events = [{"event_id": "EVT-1", "symbol": "ETHUSDT", "anchor_ts_ms": 1000,
               "event_end_ts_ms": None, "route_version": "LONG_SILENCE",
               "source_quality": "REAL_LIQUIDATION", "independent_cycle_id": None}]
    signals = derive_signals(events)
    assert signals[0]["setup_version"] is None  # canonical column: never a fake real value
    expected_id = generate_signal_id(setup_id="LONG_SILENCE", setup_version=UNKNOWN_SETUP_VERSION_TOKEN,
                                     symbol="ETHUSDT", direction="LONG", source_event_id="EVT-1")
    assert signals[0]["signal_id"] == expected_id  # identity hash input unchanged
