"""AMI BIRTH-TRUNCATED CASCADE GEOMETRY -- tests for
ami/geometry/birth_truncated_cascade_geometry.py (DISPOSABLE ONLY: no
function under test ever opens the real canonical.sqlite; every test here
uses an in-memory or tmp_path sqlite connection).

Quality state is NOT part of this module (post source-quality-contract-v2
migration review: the old row-level data_quality_status column +
ami_birth_truncated_geometry_quality_assessment table/effective-view were
removed here to avoid a duplicate quality-state mechanism against the
authoritative field-level ami_birth_truncated_geometry_field_quality_v2
ledger in ami/geometry/liquidation_source_quality_contract_v2.py, which has
its own dedicated test file).

Run: pytest tests/test_ami_geometry_birth_truncated_cascade_geometry.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations

import sqlite3

import pytest

from ami.geometry import birth_truncated_cascade_geometry as geo

_FORBIDDEN_TERMS = ("forward_fill", "interpolate", "synthesize", "synthetic_liquidation")


def test_no_interpolation_or_fabrication_terms_in_module_source():
    import inspect
    src = inspect.getsource(geo).lower()
    hits = [t for t in _FORBIDDEN_TERMS if t in src]
    assert hits == [], f"forbidden fabrication terms found: {hits}"


def test_no_quality_status_column_or_table_in_this_module():
    """Guard against reintroducing the removed duplicate quality-state
    mechanism: this module must never define a data_quality_status column
    or a quality_assessment table of its own -- ami_birth_truncated_geometry_
    field_quality_v2 (liquidation_source_quality_contract_v2.py) is the SOLE
    quality mechanism."""
    assert "data_quality_status" not in geo._SCHEMA
    assert "coverage_assessment_version" not in geo._SCHEMA
    assert "quality_assessment" not in geo._SCHEMA.lower()


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys=ON")
    # minimal stand-ins for the canonical tables this module's FKs reference
    c.executescript(
        """
        CREATE TABLE ami_signal_lifecycle (signal_id TEXT PRIMARY KEY);
        CREATE TABLE ami_events (event_id TEXT PRIMARY KEY, anchor_ts_ms INTEGER);
        CREATE TABLE ami_cycles (cycle_id TEXT PRIMARY KEY);
        """
    )
    geo.init_schema(c)
    return c


_EARLY_TS = 1_000_000 - 60_000  # 60s before the default anchor_ts_ms (1_000_000) -- same 300s bucket


def _liq(ts_ms, notional):
    return {"ts_ms": ts_ms, "price": 1000.0, "quantity": notional / 1000.0,
            "notional": notional, "trade_time_ms": ts_ms}


def _fake_reconstruct_anchors(rows, *, bucket_sec, min_gap_sec, thresholds, accel_window_sec):
    """Deterministic stand-in mirroring reconstruct_anchors()'s public
    contract for one bucket/one threshold, so these tests do not depend on
    tools.research_s34_knowable_anchor_continuation's implementation."""
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: r["ts_ms"])
    first_ts = rows[0]["ts_ms"]
    total = 0.0
    max_single = 0.0
    for idx, r in enumerate(rows):
        total += r["notional"]
        max_single = max(max_single, r["notional"])
        if total >= thresholds[0]:
            t = r["ts_ms"]
            elapsed = max(0.0, (t - first_ts) / 1000.0)

            class _Anc:
                pass

            a = _Anc()
            a.anchor_ts_ms = t
            a.first_ts_ms = first_ts
            a.running_notional = total
            a.running_liq_count = idx + 1
            a.running_rate = total / max(elapsed, 1.0)
            a.running_accel = 0.0
            a.running_single_liq_dominance = max_single / total * 100.0
            a.max_single_notional = max_single
            a.elapsed_since_first_sec = elapsed
            return [a]
    return []


def _setup_one_signal(conn, sid="SIG-1", eid="EVT-1", cyc="CYC-1", anchor_ts=1_000_000, birth=None):
    birth = anchor_ts if birth is None else birth
    conn.execute("INSERT INTO ami_signal_lifecycle VALUES (?)", (sid,))
    conn.execute("INSERT INTO ami_events VALUES (?,?)", (eid, anchor_ts))
    conn.execute("INSERT INTO ami_cycles VALUES (?)", (cyc,))
    conn.commit()
    signals = [{"signal_id": sid, "source_event_id": eid, "independent_cycle_id": cyc, "signal_birth_ts": birth}]
    events_by_id = {eid: {"anchor_ts_ms": anchor_ts}}
    return signals, events_by_id


def test_init_schema_is_idempotent(conn):
    geo.init_schema(conn)
    geo.init_schema(conn)  # must not raise


def test_reconstruct_signal_geometry_uses_only_rows_leq_birth():
    liqs = [_liq(1_000_000 - 60_000, 150_000.0), _liq(1_000_000, 60_000.0), _liq(1_000_001, 999_999_999.0)]
    geo_row = geo.reconstruct_signal_geometry(
        liqs, anchor_ts_ms=1_000_000, signal_birth_ts=1_000_000,
        reconstruct_anchors_fn=_fake_reconstruct_anchors,
    )
    assert geo_row is not None
    assert geo_row["running_notional"] == pytest.approx(210_000.0)
    assert geo_row["source_window_end_ts_ms"] <= 1_000_000


def test_reconstruct_signal_geometry_returns_none_if_anchor_not_reproducible():
    liqs = [_liq(1_000_000, 60_000.0)]  # never crosses any threshold
    result = geo.reconstruct_signal_geometry(
        liqs, anchor_ts_ms=1_000_000, signal_birth_ts=1_000_000,
        reconstruct_anchors_fn=_fake_reconstruct_anchors,
    )
    assert result is None


def test_inter_cluster_gap_sec_none_for_first_anchor():
    assert geo.inter_cluster_gap_sec([500, 1000, 2000], 500) is None


def test_inter_cluster_gap_sec_computed_for_later_anchor():
    assert geo.inter_cluster_gap_sec([500, 1000, 2000], 2000) == pytest.approx(1.0)


def test_backfill_accepts_reconstructable_signal_and_writes_both_tables(conn):
    liqs = [_liq(_EARLY_TS, 150_000.0), _liq(1_000_000, 60_000.0)]
    signals, events_by_id = _setup_one_signal(conn, anchor_ts=1_000_000)
    result = geo.backfill(
        conn, liqs, signals, events_by_id,
        reconstruct_anchors_fn=_fake_reconstruct_anchors, provenance="test",
    )
    assert result["accepted_n"] == 1
    assert result["rejected_n"] == 0
    counts = geo.row_counts(conn)
    assert counts["geometry"] == 1
    assert counts["field_provenance"] == len(geo._FEATURE_FIELDS)


def test_backfill_rejects_anchor_after_birth(conn):
    signals, events_by_id = _setup_one_signal(conn, anchor_ts=2_000_000, birth=1_000_000)
    result = geo.backfill(
        conn, [], signals, events_by_id,
        reconstruct_anchors_fn=_fake_reconstruct_anchors, provenance="test",
    )
    assert result["accepted_n"] == 0
    assert result["rejected"]["ANCHOR_AFTER_BIRTH"] == ["SIG-1"]


def test_backfill_rejects_no_source_event(conn):
    conn.execute("INSERT INTO ami_signal_lifecycle VALUES (?)", ("SIG-1",))
    conn.commit()
    signals = [{"signal_id": "SIG-1", "source_event_id": "MISSING", "independent_cycle_id": None,
                "signal_birth_ts": 1_000_000}]
    result = geo.backfill(
        conn, [], signals, {},
        reconstruct_anchors_fn=_fake_reconstruct_anchors, provenance="test",
    )
    assert result["rejected"]["NO_SOURCE_EVENT"] == ["SIG-1"]


def test_backfill_is_idempotent_on_identical_rerun(conn):
    liqs = [_liq(_EARLY_TS, 150_000.0), _liq(1_000_000, 60_000.0)]
    signals, events_by_id = _setup_one_signal(conn, anchor_ts=1_000_000)
    geo.backfill(conn, liqs, signals, events_by_id,
                 reconstruct_anchors_fn=_fake_reconstruct_anchors, provenance="test")
    hash1 = geo.content_hash(conn)
    result2 = geo.backfill(conn, liqs, signals, events_by_id,
                            reconstruct_anchors_fn=_fake_reconstruct_anchors, provenance="test-rerun")
    hash2 = geo.content_hash(conn)
    assert result2["accepted_n"] == 1
    assert hash1 == hash2
    assert geo.row_counts(conn)["geometry"] == 1


def test_backfill_fails_closed_on_content_conflict(conn):
    liqs = [_liq(_EARLY_TS, 150_000.0), _liq(1_000_000, 60_000.0)]
    signals, events_by_id = _setup_one_signal(conn, anchor_ts=1_000_000)
    geo.backfill(conn, liqs, signals, events_by_id,
                 reconstruct_anchors_fn=_fake_reconstruct_anchors, provenance="test")
    conn.execute(
        "UPDATE ami_birth_truncated_cascade_geometry SET running_notional = running_notional + 1 "
        "WHERE signal_id='SIG-1'"
    )
    conn.commit()
    with pytest.raises(geo.ImmutableGeometryConflict):
        geo.backfill(conn, liqs, signals, events_by_id,
                     reconstruct_anchors_fn=_fake_reconstruct_anchors, provenance="test-conflict")


def test_schema_check_constraint_rejects_window_end_after_feature_available(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ami_birth_truncated_cascade_geometry (feature_id, feature_definition_version, "
            "signal_id, source_event_id, independent_cycle_id, feature_available_ts_ms, "
            "source_window_start_ts_ms, source_window_end_ts_ms, source_row_count, "
            "source_row_manifest_sha256, running_notional, running_liq_count, max_single_notional, "
            "running_single_liq_dominance, running_rate, running_accel, elapsed_since_first_sec, "
            "inter_cluster_gap_sec, known_at_classification, "
            "derivation_reference, schema_version, provenance, created_ms) "
            "VALUES ('F1','v1','SIG-X','EVT-X',NULL,100,0,200,1,'h',1.0,1,1.0,1.0,1.0,0.0,1.0,NULL,"
            "'KNOWN_AT_SAFE','ref',1,'p',0)"
        )


def test_schema_foreign_key_rejects_unknown_cycle_id(conn):
    conn.execute("INSERT INTO ami_signal_lifecycle VALUES ('SIG-X')")
    conn.execute("INSERT INTO ami_events VALUES ('EVT-X', 100)")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ami_birth_truncated_cascade_geometry (feature_id, feature_definition_version, "
            "signal_id, source_event_id, independent_cycle_id, feature_available_ts_ms, "
            "source_window_start_ts_ms, source_window_end_ts_ms, source_row_count, "
            "source_row_manifest_sha256, running_notional, running_liq_count, max_single_notional, "
            "running_single_liq_dominance, running_rate, running_accel, elapsed_since_first_sec, "
            "inter_cluster_gap_sec, known_at_classification, "
            "derivation_reference, schema_version, provenance, created_ms) "
            "VALUES ('F1','v1','SIG-X','EVT-X','NONEXISTENT-CYCLE',100,0,100,1,'h',1.0,1,1.0,1.0,1.0,0.0,1.0,"
            "NULL,'KNOWN_AT_SAFE','ref',1,'p',0)"
        )
        conn.commit()


def test_field_provenance_uses_existing_repo_classification_vocabulary(conn):
    liqs = [_liq(_EARLY_TS, 150_000.0), _liq(1_000_000, 60_000.0)]
    signals, events_by_id = _setup_one_signal(conn, anchor_ts=1_000_000)
    geo.backfill(conn, liqs, signals, events_by_id,
                 reconstruct_anchors_fn=_fake_reconstruct_anchors, provenance="test")
    rows = conn.execute(
        "SELECT field_name, field_classification FROM ami_birth_truncated_geometry_field_provenance"
    ).fetchall()
    assert len(rows) == len(geo._FEATURE_FIELDS)
    assert {name for name, _ in rows} == set(geo._FEATURE_FIELDS)
    assert all(cls == "DETERMINISTIC_HISTORICAL_SAFE" for _, cls in rows)


def test_rollback_drops_all_new_objects_and_reapply_reproduces_content(conn):
    liqs = [_liq(_EARLY_TS, 150_000.0), _liq(1_000_000, 60_000.0)]
    signals, events_by_id = _setup_one_signal(conn, anchor_ts=1_000_000)
    geo.backfill(conn, liqs, signals, events_by_id,
                 reconstruct_anchors_fn=_fake_reconstruct_anchors, provenance="test")
    hash_before = geo.content_hash(conn)

    geo.rollback(conn)
    remaining = conn.execute(
        "SELECT name FROM sqlite_master WHERE name LIKE 'ami_birth_truncated%'"
    ).fetchall()
    assert remaining == []

    geo.init_schema(conn)
    geo.backfill(conn, liqs, signals, events_by_id,
                 reconstruct_anchors_fn=_fake_reconstruct_anchors, provenance="test-reapply")
    hash_after = geo.content_hash(conn)
    assert hash_after == hash_before


def test_row_content_tuple_stable_field_order():
    row = {f: 1.0 for f in geo._FEATURE_FIELDS}
    row.update({"source_window_start_ts_ms": 1, "source_window_end_ts_ms": 2,
                "source_row_count": 3, "source_row_manifest_sha256": "h"})
    t1 = geo._row_content_tuple(row)
    t2 = geo._row_content_tuple(dict(row))
    assert t1 == t2


def test_ami_events_semantic_collision_guard_is_documented():
    """Goal F: the module docstring must explicitly warn that ami_events'
    final-geometry fields are not equivalent to this module's fields."""
    import inspect
    src = inspect.getsource(geo)
    assert "event_count" in src and "route" in src.lower()
    assert "event_end_ts_ms" in src and "signal_birth_ts" in src


def test_module_never_writes_to_ami_events():
    """Goal F: this module must never mutate ami_events -- no INSERT/UPDATE/
    DELETE statement anywhere in its source targets that table."""
    import inspect
    import re
    src = inspect.getsource(geo)
    mutating = re.findall(r"(?i)\b(INSERT INTO|UPDATE|DELETE FROM)\s+ami_events\b", src)
    assert mutating == []
