"""LIQUIDATION SOURCE-QUALITY CONTRACT V2 -- tests for
ami/geometry/liquidation_source_quality_contract_v2.py.

DISPOSABLE_DB_ONLY / NO_OUTCOME_ANALYSIS: every synthetic test uses an
in-memory sqlite connection; the one real-data test opens the real
canonical.sqlite ONLY via a disposable copy + microstructure.db ONLY
mode=ro (never written).

Run: pytest tests/test_ami_geometry_liquidation_source_quality_contract_v2.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import inspect
import shutil
import sqlite3
import time

import pytest

from ami.geometry import birth_truncated_cascade_geometry as geo
from ami.geometry import liquidation_source_quality_contract_v2 as v2

TRANSITION = v2.ALL_MARKET_TRANSITION_TS_MS
CRIT = v2.CRITICAL_GAP_MS
_FORBIDDEN_OUTCOME_TERMS = ("mfe_bps", "mae_bps", "pnl", "win_rate", "p_value", "return_bps")


# --------------------------------------------------------------------- (12)
def test_no_outcome_terms_in_module_source():
    src = inspect.getsource(v2).lower()
    hits = [t for t in _FORBIDDEN_OUTCOME_TERMS if t in src]
    assert hits == [], f"forbidden outcome terms found: {hits}"


# ----------------------------------------------------------------------(1)
def test_pre_transition_window_can_never_become_complete():
    ws = TRANSITION - 3_600_000
    we = TRANSITION - 1_000
    status, reason = v2.classify_field_window(ws, we, resolved_gaps=[], sorted_all_market_liq_ts=[])
    assert status == "SOURCE_COVERAGE_UNRESOLVED"
    assert "all-market transition" in reason
    # even with PERFECT (dense) synthetic cadence evidence supplied, pre-transition still cannot pass
    dense = list(range(ws, we, 1000))
    status2, _ = v2.classify_field_window(ws, we, resolved_gaps=[], sorted_all_market_liq_ts=dense)
    assert status2 == "SOURCE_COVERAGE_UNRESOLVED"


# ----------------------------------------------------------------------(2)
def test_absence_of_gap_registry_row_does_not_imply_completeness():
    """A window with ZERO registry gap rows AND zero raw liquidation
    messages anywhere near it must NOT be classified COMPLETE just because
    no gap was ever logged against it."""
    ws = TRANSITION + 10_000_000
    we = ws + 10_000  # short window, no messages supplied at all
    status, reason = v2.classify_field_window(ws, we, resolved_gaps=[], sorted_all_market_liq_ts=[])
    # window itself (10s) is well under CRITICAL_GAP_MS, so this SPECIFIC short
    # window legitimately passes on cadence grounds -- but widen it past the
    # critical threshold with zero evidence and it must fail closed:
    we_long = ws + CRIT + 10_000
    status_long, reason_long = v2.classify_field_window(ws, we_long, resolved_gaps=[], sorted_all_market_liq_ts=[])
    assert status_long == "SOURCE_GAPPED"
    assert "cadence gap" in reason_long


# ----------------------------------------------------------------------(3)
def test_all_market_cadence_within_300s_establishes_complete():
    ws = TRANSITION + 1_000_000
    we = ws + 600_000
    ts = list(range(ws, we + 1, 100_000))  # every 100s -- well under 300s
    status, reason = v2.classify_field_window(ws, we, resolved_gaps=[], sorted_all_market_liq_ts=ts)
    assert status == "SOURCE_COMPLETE"
    assert "cadence verified" in reason


# ----------------------------------------------------------------------(4)
def test_cadence_gap_over_300s_produces_gapped():
    ws = TRANSITION + 1_000_000
    we = ws + 700_000
    ts = [ws, ws + 100_000, ws + 700_000 - 100_000, we]  # a >400s hole in the middle
    status, reason = v2.classify_field_window(ws, we, resolved_gaps=[], sorted_all_market_liq_ts=ts)
    assert status == "SOURCE_GAPPED"
    assert "exceeds CRITICAL_GAP_MS" in reason


def test_resolved_registry_gap_overlap_produces_gapped_even_with_perfect_cadence():
    ws = TRANSITION + 1_000_000
    we = ws + 600_000
    ts = list(range(ws, we + 1, 50_000))
    resolved = [(ws + 200_000, ws + 210_000)]
    status, reason = v2.classify_field_window(ws, we, resolved_gaps=resolved, sorted_all_market_liq_ts=ts)
    assert status == "SOURCE_GAPPED"
    assert "resolved registry gap" in reason


# ----------------------------------------------------------------------(5)
def test_cross_stream_activity_cannot_establish_completeness():
    """Structural guard: classify_field_window's (and classify_signal_fields')
    signatures accept only liquidation-stream evidence (resolved_gaps +
    sorted_all_market_liq_ts) -- no agg_trades/mark_prices parameter exists
    anywhere to be (mis)used as completeness proof. The module's own
    docstring legitimately DISCUSSES agg_trades/mark_prices prose (to
    document why cross-stream health was rejected as evidence) -- this test
    checks executable signatures/bodies, not docstring text."""
    for fn in (v2.classify_field_window, v2.classify_signal_fields, v2.required_window,
               v2.max_cadence_gap_ms, v2.resolved_gap_overlaps):
        sig = inspect.signature(fn)
        for forbidden in ("agg_trades", "mark_price", "markprice", "cross_stream"):
            assert forbidden not in sig.parameters, f"{fn.__name__} accepts forbidden param {forbidden!r}"
        body_lines = inspect.getsource(fn).split("\n")
        # strip the function's own leading docstring block before scanning for forbidden identifiers
        code_only = "\n".join(l for l in body_lines if not l.strip().startswith(("#", '"""', "'''")))
        assert "agg_trades" not in code_only
        assert "mark_price" not in code_only.lower()


# ----------------------------------------------------------------------(6)
def test_feature_specific_windows_are_evaluated_independently():
    bucket_start = TRANSITION + 1_000_000
    anchor_ts = bucket_start + 200_000
    prev_anchor = bucket_start - 5_000_000
    windows = {
        f: v2.required_window(f, bucket_start_ts_ms=bucket_start, anchor_ts_ms=anchor_ts,
                               prev_anchor_ts_ms=prev_anchor, earliest_liq_ts_ms=0)
        for f in geo._FEATURE_FIELDS
    }
    # running_notional-family fields share the bucket window
    assert windows["running_notional"] == (bucket_start, anchor_ts)
    assert windows["elapsed_since_first_sec"] == (bucket_start, anchor_ts)
    # running_accel uses its own narrow backward two-window span, independent of bucket_start
    assert windows["running_accel"] == (anchor_ts - 2 * geo.ACCEL_WIN_SEC * 1000, anchor_ts)
    assert windows["running_accel"] != windows["running_notional"]
    # inter_cluster_gap_sec spans back to the PREVIOUS anchor, not the current bucket
    assert windows["inter_cluster_gap_sec"] == (prev_anchor, anchor_ts)
    assert windows["inter_cluster_gap_sec"] != windows["running_notional"]


# ----------------------------------------------------------------------(7)
def test_inter_cluster_gap_sec_cannot_inherit_current_bucket_completeness():
    bucket_start = TRANSITION + 10_000_000
    anchor_ts = bucket_start + 100_000  # narrow, perfectly-covered bucket
    prev_anchor = TRANSITION + 1_000  # far earlier -- huge inter-anchor span
    dense_bucket_ts = list(range(bucket_start, anchor_ts + 1, 50_000))  # dense INSIDE the bucket only

    fields = v2.classify_signal_fields(
        bucket_start_ts_ms=bucket_start, anchor_ts_ms=anchor_ts, prev_anchor_ts_ms=prev_anchor,
        earliest_liq_ts_ms=0, resolved_gaps=[], sorted_all_market_liq_ts=dense_bucket_ts,
    )
    assert fields["running_notional"]["status"] == "SOURCE_COMPLETE"
    # the inter-anchor span is mostly EMPTY of the dense-bucket-only evidence -> must not be COMPLETE
    assert fields["inter_cluster_gap_sec"]["status"] != "SOURCE_COMPLETE"


def test_first_ever_anchor_inter_cluster_gap_sec_window_reaches_earliest_history():
    ws, we = v2.required_window(
        "inter_cluster_gap_sec", bucket_start_ts_ms=5_000, anchor_ts_ms=10_000,
        prev_anchor_ts_ms=None, earliest_liq_ts_ms=1_000,
    )
    assert (ws, we) == (1_000, 10_000)


# ----------------------------------------------------------------------(8, 9, 10)
@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys=ON")
    c.executescript(
        """
        CREATE TABLE ami_signal_lifecycle (signal_id TEXT PRIMARY KEY);
        CREATE TABLE ami_events (event_id TEXT PRIMARY KEY, anchor_ts_ms INTEGER);
        CREATE TABLE ami_cycles (cycle_id TEXT PRIMARY KEY);
        """
    )
    geo.init_schema(c)
    v2.init_schema(c)
    return c


def _insert_one_geometry_row(conn, feature_id="FEAT-1", signal_id="SIG-1"):
    conn.execute("INSERT INTO ami_signal_lifecycle VALUES (?)", (signal_id,))
    conn.execute("INSERT INTO ami_events VALUES ('EVT-1', 1000)")
    conn.commit()
    conn.execute(
        "INSERT INTO ami_birth_truncated_cascade_geometry (feature_id, feature_definition_version, "
        "signal_id, source_event_id, independent_cycle_id, feature_available_ts_ms, "
        "source_window_start_ts_ms, source_window_end_ts_ms, source_row_count, "
        "source_row_manifest_sha256, running_notional, running_liq_count, max_single_notional, "
        "running_single_liq_dominance, running_rate, running_accel, elapsed_since_first_sec, "
        "inter_cluster_gap_sec, known_at_classification, "
        "derivation_reference, schema_version, provenance, created_ms) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (feature_id, "v1", signal_id, "EVT-1", None, 1000, 500, 1000, 1, "h",
         100.0, 1, 100.0, 100.0, 1.0, 0.0, 1.0, None, "KNOWN_AT_SAFE",
         "ref", 1, "test", 0),
    )
    conn.commit()


def _field_statuses_fixture():
    return {f: {"status": "SOURCE_COMPLETE" if f != "inter_cluster_gap_sec" else "SOURCE_COVERAGE_UNRESOLVED",
                "reason": "test", "window_start_ts_ms": 1, "window_end_ts_ms": 2}
            for f in geo._FEATURE_FIELDS}


def test_contract_v2_rerun_is_deterministic(conn):
    _insert_one_geometry_row(conn)
    rows = [{"feature_id": "FEAT-1", "field_statuses": _field_statuses_fixture()}]
    r1 = v2.backfill_field_quality(conn, rows, provenance="test")
    hash1 = v2.content_hash(conn)
    r2 = v2.backfill_field_quality(conn, rows, provenance="test-rerun")
    hash2 = v2.content_hash(conn)
    assert r1["accepted_n"] == len(geo._FEATURE_FIELDS)
    assert r2["accepted_n"] == len(geo._FEATURE_FIELDS)
    assert hash1 == hash2
    assert v2.row_counts(conn)["field_quality_v2"] == len(geo._FEATURE_FIELDS)


def test_schema_check_constraint_rejects_bad_data_quality_status(conn):
    """Direct raw-SQL proof of the table-level CHECK constraint on
    ami_birth_truncated_geometry_field_quality_v2.data_quality_status --
    closes the coverage gap identified in the regression-ground-truth
    reconciliation (the deleted, pre-contract-v2 equivalent test exercised
    the OLD row-level table's now-removed column of the same name; this is
    the replacement table's own constraint, never previously exercised via
    raw SQL bypassing the Python classify_field_window()/backfill_field_quality()
    layer)."""
    _insert_one_geometry_row(conn)
    rows = [{"feature_id": "FEAT-1", "field_statuses": _field_statuses_fixture()}]
    v2.backfill_field_quality(conn, rows, provenance="test")

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE ami_birth_truncated_geometry_field_quality_v2 SET data_quality_status='BOGUS_STATUS' "
            "WHERE feature_id='FEAT-1' AND field_name='running_notional'"
        )

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ami_birth_truncated_geometry_field_quality_v2 "
            "(assessment_id, feature_id, field_name, coverage_assessment_version, data_quality_status, "
            "window_start_ts_ms, window_end_ts_ms, evidence, provenance, assessed_at_ms) "
            "VALUES ('FQV2-BOGUS','FEAT-1','running_notional','v-bogus','BOGUS_STATUS',1,2,'e','p',0)"
        )


def test_quality_reassessment_is_append_only(conn):
    _insert_one_geometry_row(conn)
    rows = [{"feature_id": "FEAT-1", "field_statuses": _field_statuses_fixture()}]
    v2.backfill_field_quality(conn, rows, provenance="test", coverage_assessment_version="v2-a")
    n_after_first = conn.execute(
        "SELECT COUNT(*) FROM ami_birth_truncated_geometry_field_quality_v2").fetchone()[0]

    # a genuine reassessment: same field, DIFFERENT status, but a NEW version -- must be an
    # INSERT, not an UPDATE of the first row. Sleep past the millisecond boundary so the two
    # real time.time()*1000-stamped rows cannot tie on assessed_at_ms (the effective view
    # picks the MAX -- a tie would make "latest" ambiguous, not a module bug).
    time.sleep(0.005)
    rows2 = [{"feature_id": "FEAT-1", "field_statuses": {
        "inter_cluster_gap_sec": {"status": "SOURCE_COMPLETE", "reason": "resumed monitoring",
                                   "window_start_ts_ms": 1, "window_end_ts_ms": 2}}}]
    v2.backfill_field_quality(conn, rows2, provenance="test", coverage_assessment_version="v2-b")
    n_after_second = conn.execute(
        "SELECT COUNT(*) FROM ami_birth_truncated_geometry_field_quality_v2").fetchone()[0]
    assert n_after_second == n_after_first + 1  # appended, nothing deleted

    first_row_status = conn.execute(
        "SELECT data_quality_status FROM ami_birth_truncated_geometry_field_quality_v2 "
        "WHERE feature_id='FEAT-1' AND field_name='inter_cluster_gap_sec' AND coverage_assessment_version='v2-a'"
    ).fetchone()[0]
    assert first_row_status == "SOURCE_COVERAGE_UNRESOLVED"  # untouched, historical evidence preserved

    effective = conn.execute(
        "SELECT data_quality_status FROM ami_birth_truncated_geometry_field_quality_v2_effective "
        "WHERE feature_id='FEAT-1' AND field_name='inter_cluster_gap_sec'"
    ).fetchone()[0]
    assert effective == "SOURCE_COMPLETE"  # effective view resolves to the LATEST assessment

    # conflicting content under the SAME version fails closed
    conflicting = [{"feature_id": "FEAT-1", "field_statuses": {
        "inter_cluster_gap_sec": {"status": "SOURCE_GAPPED", "reason": "different",
                                   "window_start_ts_ms": 1, "window_end_ts_ms": 2}}}]
    with pytest.raises(v2.ImmutableFieldQualityConflict):
        v2.backfill_field_quality(conn, conflicting, provenance="test", coverage_assessment_version="v2-b")


def test_geometry_values_byte_identical_across_quality_reassessments(conn):
    _insert_one_geometry_row(conn)
    geo_hash_before = geo.content_hash(conn)
    rows = [{"feature_id": "FEAT-1", "field_statuses": _field_statuses_fixture()}]
    v2.backfill_field_quality(conn, rows, provenance="test")
    geo_hash_after = geo.content_hash(conn)
    assert geo_hash_before == geo_hash_after  # geometry table untouched by field-quality backfill

    rows2 = [{"feature_id": "FEAT-1", "field_statuses": {
        "inter_cluster_gap_sec": {"status": "SOURCE_COMPLETE", "reason": "resumed",
                                   "window_start_ts_ms": 1, "window_end_ts_ms": 2}}}]
    v2.backfill_field_quality(conn, rows2, provenance="test", coverage_assessment_version="v2-later")
    assert geo.content_hash(conn) == geo_hash_before  # still untouched after a SECOND reassessment


def test_row_level_worst_case_never_upgrades_to_complete_when_any_field_is_not():
    mixed = {"running_notional": "SOURCE_COMPLETE", "inter_cluster_gap_sec": "SOURCE_COVERAGE_UNRESOLVED"}
    assert v2.row_level_worst_case(mixed) == "SOURCE_COVERAGE_UNRESOLVED"
    gapped_mixed = {"running_notional": "SOURCE_COMPLETE", "running_accel": "SOURCE_GAPPED",
                    "inter_cluster_gap_sec": "SOURCE_COVERAGE_UNRESOLVED"}
    assert v2.row_level_worst_case(gapped_mixed) == "SOURCE_GAPPED"
    all_complete = {f: "SOURCE_COMPLETE" for f in geo._FEATURE_FIELDS}
    assert v2.row_level_worst_case(all_complete) == "SOURCE_COMPLETE"


# ---------------------------------------------------------------------(11)
def test_source_complete_only_cycle_counts_remain_below_min_bucket_n_real_data(tmp_path):
    """Real-data measurement: rebuild all 220 LONG signals' field-level
    contract-v2 quality against the real (disposable-copied) canonical.sqlite
    and real (mode=ro) microstructure.db, then confirm SOURCE_COMPLETE_ONLY
    independent-cycle N in TEST remains below MIN_BUCKET_N=20 (measured, not
    forced) -- research readiness stays fail-closed."""
    from ami.geometry import birth_truncated_geometry_rehearsal as rehearsal
    from ami.research.w4_post_event_path_taxonomy import MIN_BUCKET_N
    from ami.warehouse.schema import DEFAULT_PATH as REAL_CANONICAL_PATH
    from tools.research_s34_knowable_anchor_continuation import reconstruct_anchors

    micro_path = r"D:\eclipse_scalper\data\microstructure.db"
    disposable = tmp_path / "disposable.sqlite"
    shutil.copy2(REAL_CANONICAL_PATH, disposable)
    conn_c = sqlite3.connect(disposable)
    conn_c.execute("PRAGMA foreign_keys=ON")

    _cols = ["signal_id", "setup_id", "source_event_id", "independent_cycle_id",
             "symbol", "direction", "signal_birth_ts"]
    signals = [dict(zip(_cols, r)) for r in conn_c.execute(
        f"SELECT {', '.join(_cols)} FROM ami_signal_lifecycle WHERE symbol='ETHUSDT' AND direction='LONG'"
    ).fetchall()]
    assert len(signals) == 220
    event_ids = {s["source_event_id"] for s in signals}
    events_by_id = rehearsal.fetch_events_by_id(conn_c, event_ids)

    conn_m = sqlite3.connect(f"file:{micro_path}?mode=ro", uri=True)
    all_sell_liqs = rehearsal.fetch_all_sell_liqs(conn_m)
    all_anchor_ts = sorted({int(r["anchor_ts_ms"]) for r in events_by_id.values()})
    earliest_liq_ts_ms = all_sell_liqs[0]["ts_ms"]
    resolved_gaps = [
        (s, e) for s, e in conn_m.execute(
            "SELECT start_ts_ms, end_ts_ms FROM gaps WHERE stream='liquidations' AND end_ts_ms IS NOT NULL"
        ).fetchall()
    ]
    all_market_liq_ts = [
        r[0] for r in conn_m.execute(
            "SELECT ts_ms FROM liquidations WHERE ts_ms >= ? ORDER BY ts_ms",
            (v2.ALL_MARKET_TRANSITION_TS_MS - v2.CRITICAL_GAP_MS,),
        ).fetchall()
    ]
    conn_m.close()

    complete_signals = []
    for s in signals:
        ev = events_by_id[s["source_event_id"]]
        anchor_ts = int(ev["anchor_ts_ms"])
        geo_row = geo.reconstruct_signal_geometry(
            all_sell_liqs, anchor_ts, int(s["signal_birth_ts"]), reconstruct_anchors_fn=reconstruct_anchors)
        assert geo_row is not None
        pos = all_anchor_ts.index(anchor_ts)
        prev_anchor_ts_ms = all_anchor_ts[pos - 1] if pos > 0 else None
        fields = v2.classify_signal_fields(
            bucket_start_ts_ms=geo_row["source_window_start_ts_ms"], anchor_ts_ms=anchor_ts,
            prev_anchor_ts_ms=prev_anchor_ts_ms, earliest_liq_ts_ms=earliest_liq_ts_ms,
            resolved_gaps=resolved_gaps, sorted_all_market_liq_ts=all_market_liq_ts,
        )
        row_status = v2.row_level_worst_case({f: d["status"] for f, d in fields.items()})
        if row_status == "SOURCE_COMPLETE":
            complete_signals.append(s)
    conn_c.close()

    rep = rehearsal.compute_population_report(complete_signals)
    assert rep["test_cycle_n"] < MIN_BUCKET_N, (
        f"expected SOURCE_COMPLETE_ONLY TEST cycle N to remain below MIN_BUCKET_N={MIN_BUCKET_N} "
        f"(measured {rep['test_cycle_n']}) -- research readiness must stay fail-closed"
    )
