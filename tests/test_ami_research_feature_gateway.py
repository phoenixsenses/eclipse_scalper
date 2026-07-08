"""BATCH-P6-001: mandatory feature gateway tests (FABLE REVIEW A F3 / REVIEW B F-B5).

Run: pytest tests/test_ami_research_feature_gateway.py --basetemp <scratchpad> -p no:cacheprovider
"""
import pytest

from ami.research.feature_gateway import (
    AmbiguousPathVersionError,
    FeatureGatewayViolation,
    fetch_chart_feature,
    fetch_cycles,
    fetch_events,
    fetch_level_features,
    fetch_lifecycle_effective_transitions,
    fetch_lifecycle_signals,
    fetch_path_observations,
)
from ami.warehouse.schema import connect, init_schema

NOW = 0


def _insert_event(conn, eid, symbol, source_quality):
    conn.execute(
        "INSERT INTO ami_events (event_id, event_family, symbol, anchor_ts_ms, source_quality, "
        "event_definition_version, censor_status, schema_version, provenance, created_ms, updated_ms) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (eid, "FAM_A", symbol, 0, source_quality, "test-v1", "COMPLETED", 5, "test", NOW, NOW),
    )


def _insert_level(conn, lid, symbol="ETHUSDT"):
    conn.execute(
        "INSERT INTO ami_levels (level_id, symbol, level_type, price, origin_ts, known_at_ts, timeframe, "
        "touch_count, source_type, level_definition_version, schema_version, provenance, created_ms, "
        "updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (lid, symbol, "SESSION_HIGH", 100.0, 0, 1000, "1m", 3, "SESSION", "level-v2-boundary-safe",
         5, "test", NOW, NOW),
    )


def _insert_candle(conn, cid, symbol="ETHUSDT"):
    conn.execute(
        "INSERT INTO ami_candles (candle_id, symbol, timeframe, open_ts_ms, close_ts_ms, open, high, low, "
        "close, is_closed, partial_status, known_at_ts, data_quality, candle_definition_version, "
        "schema_version, provenance, created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (cid, symbol, "1m", 0, 60_000, 100.0, 101.0, 99.0, 100.5, 1, "CLOSED", 60_000, "AVAILABLE",
         "test-v1", 5, "test", NOW, NOW),
    )


def _db(tmp_path):
    conn = connect(tmp_path / "canonical.sqlite")
    init_schema(conn)
    return conn


def test_fetch_events_pure_real_population_succeeds(tmp_path):
    conn = _db(tmp_path)
    _insert_event(conn, "E1", "ETHUSDT", "REAL_LIQUIDATION")
    _insert_event(conn, "E2", "ETHUSDT", "REAL_LIQUIDATION")
    conn.commit()
    rows = fetch_events(conn, "ctx-1")
    conn.close()
    assert len(rows) == 2


def test_fetch_events_includes_notional_for_confounder_checks(tmp_path):
    # BATCH-P6-008: notional (event size) added so research waves can run
    # confounder checks without a second, unsanctioned ami_events access path.
    conn = _db(tmp_path)
    conn.execute(
        "INSERT INTO ami_events (event_id, event_family, symbol, anchor_ts_ms, notional, source_quality, "
        "event_definition_version, censor_status, schema_version, provenance, created_ms, updated_ms) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("E1", "FAM_A", "ETHUSDT", 0, 500_000.0, "REAL_LIQUIDATION", "test-v1", "COMPLETED", 5, "test", NOW, NOW),
    )
    conn.commit()
    rows = fetch_events(conn, "ctx-notional")
    conn.close()
    assert rows[0]["notional"] == 500_000.0


def test_fetch_events_includes_event_end_ts_ms_for_liquidation_age(tmp_path):
    # BATCH-P6-010 (W7A): event_end_ts_ms exposed so liquidation-age features
    # can exclude the anchor's own cascade window using its true canonical
    # boundary instead of a generic wall-clock heuristic.
    conn = _db(tmp_path)
    conn.execute(
        "INSERT INTO ami_events (event_id, event_family, symbol, anchor_ts_ms, event_end_ts_ms, "
        "source_quality, event_definition_version, censor_status, schema_version, provenance, "
        "created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("E1", "FAM_A", "ETHUSDT", 1000, 5000, "REAL_LIQUIDATION", "test-v1", "COMPLETED", 5, "test", NOW, NOW),
    )
    conn.commit()
    rows = fetch_events(conn, "ctx-event-end")
    conn.close()
    assert rows[0]["event_end_ts_ms"] == 5000


def test_fetch_events_event_end_ts_ms_none_when_not_set(tmp_path):
    conn = _db(tmp_path)
    _insert_event(conn, "E1", "ETHUSDT", "REAL_LIQUIDATION")
    conn.commit()
    rows = fetch_events(conn, "ctx-event-end-none")
    conn.close()
    assert rows[0]["event_end_ts_ms"] is None


def test_fetch_cycles_returns_rows_and_records_exposure(tmp_path):
    conn = _db(tmp_path)
    conn.execute(
        "INSERT INTO ami_cycles (cycle_id, symbol, cycle_definition_version, event_count, "
        "direction_conflict, censored, confidence, schema_version, provenance, created_ms, updated_ms) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("CYC-1", "ETHUSDT", "canonical-v1", 2, 0, "COMPLETED", 1.0, 6, "test", NOW, NOW),
    )
    conn.commit()
    rows = fetch_cycles(conn, "ctx-cycles-1")
    n_exposure = conn.execute(
        "SELECT COUNT(*) FROM researcher_exposure_ledger WHERE claim_or_hypothesis_id='ctx-cycles-1'"
    ).fetchone()[0]
    conn.close()
    assert len(rows) == 1
    assert rows[0]["cycle_id"] == "CYC-1"
    assert n_exposure == 1


def test_fetch_events_pooled_population_raises(tmp_path):
    conn = _db(tmp_path)
    _insert_event(conn, "E1", "ETHUSDT", "REAL_LIQUIDATION")
    _insert_event(conn, "E2", "ETHUSDT", "PROXY_CASCADE_6H_GAP")
    conn.commit()
    with pytest.raises(FeatureGatewayViolation, match="R-09"):
        fetch_events(conn, "ctx-1")
    conn.close()


def test_fetch_events_records_exposure_ledger(tmp_path):
    conn = _db(tmp_path)
    _insert_event(conn, "E1", "ETHUSDT", "REAL_LIQUIDATION")
    conn.commit()
    fetch_events(conn, "ctx-exposure-1")
    n = conn.execute(
        "SELECT COUNT(*) FROM researcher_exposure_ledger WHERE claim_or_hypothesis_id='ctx-exposure-1'"
    ).fetchone()[0]
    conn.close()
    assert n == 1


def test_fetch_level_features_safe_column_succeeds(tmp_path):
    conn = _db(tmp_path)
    _insert_level(conn, "L1")
    conn.commit()
    rows = fetch_level_features(conn, "ctx-2", ["level_id", "price"])
    conn.close()
    assert rows == [{"level_id": "L1", "price": 100.0}]


def test_fetch_level_features_blocked_column_raises(tmp_path):
    conn = _db(tmp_path)
    _insert_level(conn, "L1")
    conn.commit()
    with pytest.raises(FeatureGatewayViolation, match="not point-in-time-safe"):
        fetch_level_features(conn, "ctx-2", ["price", "touch_count"])
    conn.close()


def test_fetch_level_features_unknown_column_raises(tmp_path):
    conn = _db(tmp_path)
    _insert_level(conn, "L1")
    conn.commit()
    with pytest.raises(FeatureGatewayViolation, match="unknown"):
        fetch_level_features(conn, "ctx-2", ["not_a_real_column"])
    conn.close()


def test_fetch_chart_feature_known_table_succeeds(tmp_path):
    conn = _db(tmp_path)
    _insert_candle(conn, "C1")
    conn.commit()
    rows = fetch_chart_feature(conn, "ctx-3", "ami_candles", ["candle_id", "close"])
    conn.close()
    assert rows == [{"candle_id": "C1", "close": 100.5}]


def test_fetch_chart_feature_rejects_levels_and_events_tables(tmp_path):
    conn = _db(tmp_path)
    with pytest.raises(FeatureGatewayViolation, match="dedicated"):
        fetch_chart_feature(conn, "ctx-3", "ami_levels", ["price"])
    with pytest.raises(FeatureGatewayViolation, match="dedicated"):
        fetch_chart_feature(conn, "ctx-3", "ami_events", ["event_id"])
    conn.close()


def test_fetch_chart_feature_unknown_table_raises(tmp_path):
    conn = _db(tmp_path)
    with pytest.raises(FeatureGatewayViolation, match="unknown feature table"):
        fetch_chart_feature(conn, "ctx-3", "some_random_table", ["x"])
    conn.close()


def test_fetch_events_real_data_smoke(tmp_path):
    # Integration smoke test against the real, already-ingested canonical.sqlite.
    from ami.warehouse.schema import DEFAULT_PATH, connect as real_connect

    conn = real_connect(DEFAULT_PATH, read_only=True)
    try:
        rows = fetch_events_readonly_smoke(conn)
    finally:
        conn.close()
    assert rows >= 0  # no exception raised == pooling guard passed on real data


def test_fetch_chart_feature_ami_candidate_universe_is_known_and_filterable(tmp_path):
    # BATCH-P6-005: ami_candidate_universe added to the gateway allowlist so
    # W4 can use it as a negative-control (random-time) population.
    conn = _db(tmp_path)
    now = 0
    conn.execute(
        "INSERT INTO ami_candidate_universe (candidate_id, symbol, timeframe, slot_ts_ms, known_at_ts, "
        "data_quality, is_event_aligned, aligned_event_id, universe_definition_version, schema_version, "
        "provenance, created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("CND-1", "ETHUSDT", "1m", 0, 60_000, "AVAILABLE", 0, None, "candidate-universe-v1", 7, "test", now, now),
    )
    conn.commit()
    rows = fetch_chart_feature(
        conn, "ctx-cu", "ami_candidate_universe", ["candidate_id", "slot_ts_ms"],
        symbol="ETHUSDT", equals={"timeframe": "1m", "is_event_aligned": 0},
    )
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "CND-1"


def _insert_signal(conn, sid, symbol="ETHUSDT", direction="LONG", evidence_layer="REAL", is_proxy=0):
    conn.execute(
        "INSERT INTO ami_signal_lifecycle (signal_id, setup_id, setup_version, source_event_id, "
        "independent_cycle_id, symbol, direction, timeframe, route_version, signal_birth_ts, "
        "first_known_ts, first_executable_ts, last_confirmation_ts, invalidation_ts, terminal_ts, "
        "lifecycle_status, lifecycle_reason_code, observation_mode, evidence_layer, is_proxy, "
        "executability_status, identity_version, schema_version, source_hash, code_commit, "
        "provenance, created_at, updated_ms) VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (sid, "LONG_SILENCE", None, "EVT-1", "CYC-1", symbol, direction, None, "LONG_SILENCE",
         1_000_000, None, None, None, None, None, "OPEN", "SIGNAL_BIRTH", "HISTORICAL_REPLAY",
         evidence_layer, is_proxy, "FORWARD_ONLY", "signal-identity-v1", 10, None, None, "test", 0, 0),
    )


def _insert_transition(conn, tid, sid, prev, new, ts, reason="SIGNAL_BIRTH", version=1, correction_of=None):
    conn.execute(
        "INSERT INTO ami_lifecycle_transitions (transition_id, signal_id, previous_status, new_status, "
        "transition_ts, known_at_ts, reason_code, transition_version, observation_mode, correction_of, "
        "evidence_ref, schema_version, provenance, created_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (tid, sid, prev, new, ts, ts, reason, version, "HISTORICAL_REPLAY", correction_of, None, 10, "test", 0),
    )


def _insert_path_observation(conn, oid, sid, horizon="scalp_30m", status="OK", vol_status="OK"):
    conn.execute(
        "INSERT INTO ami_lifecycle_path_observations (observation_id, signal_id, horizon_name, "
        "horizon_end_ts, known_at_ts, as_of_ts, observation_status, volatility_status, reference_price, "
        "reference_price_ts, effective_path_start_ts, endpoint_return_bps, mfe_bps, mae_bps, "
        "time_to_mfe_ms, time_to_mae_ms, intrabar_order_status, realized_vol_at_anchor, "
        "endpoint_return_anchor_vol_units, mfe_anchor_vol_units, mae_anchor_vol_units, "
        "horizon_outcome_class, expected_candle_count, observed_candle_count, gap_count, "
        "candle_definition_version, path_definition_version, observation_mode, provenance, "
        "schema_version, created_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (oid, sid, horizon, 2_000_000, 2_000_000, 2_000_000, status, vol_status, 100.0, 999_000,
         1_000_000, 5.0, 10.0, -3.0, 60_000, 120_000, "MFE_FIRST", 0.001, 0.5, 1.0, -0.3, "CHOP",
         30, 30, 0, "candle-agg_trades-v1", "path-v2", "HISTORICAL_REPLAY", "test", 2, 0),
    )


def test_fetch_lifecycle_signals_returns_curated_columns(tmp_path):
    conn = _db(tmp_path)
    _insert_signal(conn, "SIG-1")
    conn.commit()
    rows = fetch_lifecycle_signals(conn, "ctx-lifecycle-1")
    conn.close()
    assert len(rows) == 1
    assert rows[0]["signal_id"] == "SIG-1"
    assert rows[0]["direction"] == "LONG"
    assert "schema_version" not in rows[0]  # bookkeeping columns excluded


def test_fetch_lifecycle_signals_records_exposure(tmp_path):
    conn = _db(tmp_path)
    _insert_signal(conn, "SIG-1")
    conn.commit()
    fetch_lifecycle_signals(conn, "ctx-lifecycle-exposure")
    n = conn.execute(
        "SELECT COUNT(*) FROM researcher_exposure_ledger WHERE claim_or_hypothesis_id='ctx-lifecycle-exposure'"
    ).fetchone()[0]
    conn.close()
    assert n == 1


def test_fetch_lifecycle_signals_pooled_population_raises(tmp_path):
    conn = _db(tmp_path)
    _insert_signal(conn, "SIG-1", evidence_layer="REAL", is_proxy=0)
    _insert_signal(conn, "SIG-2", evidence_layer="PROXY", is_proxy=1)
    conn.commit()
    with pytest.raises(FeatureGatewayViolation, match="REAL.*PROXY|mixes"):
        fetch_lifecycle_signals(conn, "ctx-lifecycle-pooled")
    conn.close()


def test_fetch_lifecycle_signals_symbol_and_evidence_layer_filter(tmp_path):
    conn = _db(tmp_path)
    _insert_signal(conn, "SIG-1", symbol="ETHUSDT")
    _insert_signal(conn, "SIG-2", symbol="BTCUSDT")
    conn.commit()
    rows = fetch_lifecycle_signals(conn, "ctx-lifecycle-filter", symbol="ETHUSDT")
    conn.close()
    assert [r["signal_id"] for r in rows] == ["SIG-1"]


def test_fetch_lifecycle_effective_transitions_excludes_superseded(tmp_path):
    conn = _db(tmp_path)
    _insert_signal(conn, "SIG-1")
    _insert_transition(conn, "TRN-1", "SIG-1", None, "OPEN", 1000, reason="SIGNAL_BIRTH")
    _insert_transition(conn, "TRN-2", "SIG-1", "OPEN", "CLOSED", 2000, reason="TERMINAL_CLOSE", version=1)
    # append-only CORRECTION that reverses TRN-2 (pure reversal -> excluded from the effective view)
    _insert_transition(conn, "TRN-3", "SIG-1", "CLOSED", "OPEN", 2000, reason="CORRECTION", version=2,
                       correction_of="TRN-2")
    conn.commit()
    rows = fetch_lifecycle_effective_transitions(conn, "ctx-effective-1", signal_id="SIG-1")
    ids = {r["transition_id"] for r in rows}
    conn.close()
    # TRN-2 (the raw TERMINAL_CLOSE) and TRN-3 (its pure-reversal correction) are BOTH excluded
    # from the effective view -- only the genuine genesis transition remains, proving raw
    # TERMINAL_CLOSE rows are never surfaced as valid terminal evidence through this gateway.
    assert ids == {"TRN-1"}


def test_fetch_lifecycle_effective_transitions_records_exposure(tmp_path):
    conn = _db(tmp_path)
    _insert_signal(conn, "SIG-1")
    _insert_transition(conn, "TRN-1", "SIG-1", None, "OPEN", 1000)
    conn.commit()
    fetch_lifecycle_effective_transitions(conn, "ctx-effective-exposure")
    n = conn.execute(
        "SELECT COUNT(*) FROM researcher_exposure_ledger WHERE claim_or_hypothesis_id='ctx-effective-exposure'"
    ).fetchone()[0]
    conn.close()
    assert n == 1


def test_fetch_path_observations_returns_curated_columns(tmp_path):
    conn = _db(tmp_path)
    _insert_signal(conn, "SIG-1")
    _insert_path_observation(conn, "OBS-1", "SIG-1")
    conn.commit()
    rows = fetch_path_observations(conn, "ctx-path-1")
    conn.close()
    assert len(rows) == 1
    assert rows[0]["mfe_bps"] == 10.0
    assert "provenance" not in rows[0]  # bookkeeping columns excluded


def test_fetch_path_observations_equals_filter(tmp_path):
    conn = _db(tmp_path)
    _insert_signal(conn, "SIG-1")
    _insert_path_observation(conn, "OBS-1", "SIG-1", horizon="scalp_30m")
    _insert_path_observation(conn, "OBS-2", "SIG-1", horizon="scalp_1h")
    conn.commit()
    rows = fetch_path_observations(conn, "ctx-path-2", equals={"horizon_name": "scalp_1h"})
    conn.close()
    assert [r["observation_id"] for r in rows] == ["OBS-2"]


def test_fetch_path_observations_unknown_filter_raises(tmp_path):
    conn = _db(tmp_path)
    with pytest.raises(FeatureGatewayViolation, match="non-filterable"):
        fetch_path_observations(conn, "ctx-path-3", equals={"reference_price": 100.0})
    conn.close()


def test_fetch_path_observations_records_exposure(tmp_path):
    conn = _db(tmp_path)
    _insert_signal(conn, "SIG-1")
    _insert_path_observation(conn, "OBS-1", "SIG-1")
    conn.commit()
    fetch_path_observations(conn, "ctx-path-exposure")
    n = conn.execute(
        "SELECT COUNT(*) FROM researcher_exposure_ledger WHERE claim_or_hypothesis_id='ctx-path-exposure'"
    ).fetchone()[0]
    conn.close()
    assert n == 1


def test_fetch_chart_feature_rejects_lifecycle_and_path_tables(tmp_path):
    conn = _db(tmp_path)
    with pytest.raises(FeatureGatewayViolation, match="dedicated"):
        fetch_chart_feature(conn, "ctx-4", "ami_signal_lifecycle", ["signal_id"])
    with pytest.raises(FeatureGatewayViolation, match="dedicated"):
        fetch_chart_feature(conn, "ctx-4", "ami_lifecycle_effective_transitions", ["transition_id"])
    with pytest.raises(FeatureGatewayViolation, match="dedicated"):
        fetch_chart_feature(conn, "ctx-4", "ami_lifecycle_path_observations", ["observation_id"])
    conn.close()


def test_fetch_lifecycle_signals_real_data_smoke():
    from ami.warehouse.schema import DEFAULT_PATH, connect as real_connect

    conn = real_connect(DEFAULT_PATH)
    try:
        rows = fetch_lifecycle_signals(conn, "ctx-lifecycle-real-smoke", symbol="ETHUSDT")
    finally:
        conn.close()
    # 324 = 270 original + 54 SHORT_NOISY_BTC200K_CONFIRMED_V1 (BATCH-SHORT-NOISY-V1-CANON-BACKFILL)
    assert len(rows) == 324


def test_fetch_path_observations_real_data_smoke():
    """[BATCH: AMI EFFECTIVE-PATH AND EXPERIMENT-IMMUTABILITY SAFETY HARDENING,
    GOAL A/D] The real (isolated-copy) canonical.sqlite now legitimately holds
    two path_definition_version values (candle-repair correction), so an
    unpinned observation_status='OK' query is genuinely ambiguous -- it must
    fail closed rather than silently mixing physical versions. Pinning the
    exact historical version reproduces the pre-repair population only."""
    from ami.warehouse.schema import DEFAULT_PATH, connect as real_connect, init_schema

    conn = real_connect(DEFAULT_PATH)
    try:
        init_schema(conn)  # ensures ami_lifecycle_path_observations exists even if not yet backfilled
        with pytest.raises(AmbiguousPathVersionError):
            fetch_path_observations(conn, "ctx-path-real-smoke-unscoped", equals={"observation_status": "OK"})

        rows = fetch_path_observations(
            conn, "ctx-path-real-smoke-pinned",
            equals={"observation_status": "OK", "path_definition_version": "path-v2"},
        )
        assert all(r["observation_status"] == "OK" for r in rows)
        assert all(r["path_definition_version"] == "path-v2" for r in rows)
        assert rows  # non-empty: proves the pin didn't silently degrade to "nothing matched"
    finally:
        conn.close()


def _insert_path_observation_versioned(conn, oid, sid, path_definition_version, horizon="scalp_30m", status="OK"):
    """Same column set as _insert_path_observation(), parameterized on
    path_definition_version so a test can create two physical versions of
    the same (signal_id, horizon_name) pair without a UNIQUE-constraint clash."""
    conn.execute(
        "INSERT INTO ami_lifecycle_path_observations (observation_id, signal_id, horizon_name, "
        "horizon_end_ts, known_at_ts, as_of_ts, observation_status, volatility_status, reference_price, "
        "reference_price_ts, effective_path_start_ts, endpoint_return_bps, mfe_bps, mae_bps, "
        "time_to_mfe_ms, time_to_mae_ms, intrabar_order_status, realized_vol_at_anchor, "
        "endpoint_return_anchor_vol_units, mfe_anchor_vol_units, mae_anchor_vol_units, "
        "horizon_outcome_class, expected_candle_count, observed_candle_count, gap_count, "
        "candle_definition_version, path_definition_version, observation_mode, provenance, "
        "schema_version, created_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (oid, sid, horizon, 2_000_000, 2_000_000, 2_000_000, status, "OK", 100.0, 999_000,
         1_000_000, 5.0, 10.0, -3.0, 60_000, 120_000, "MFE_FIRST", 0.001, 0.5, 1.0, -0.3, "CHOP",
         30, 30, 0, "candle-agg_trades-v1", path_definition_version, "HISTORICAL_REPLAY", "test", 2, 0),
    )


def test_fetch_path_observations_ambiguous_version_fails_closed(tmp_path):
    """Synthetic reproduction of GOAL A's contract: two path_definition_version
    rows for the SAME (signal_id, horizon_name), unscoped fetch must raise."""
    conn = _db(tmp_path)
    _insert_signal(conn, "SIG-1")
    _insert_path_observation_versioned(conn, "OBS-old", "SIG-1", "path-v2", horizon="scalp_30m")
    _insert_path_observation_versioned(conn, "OBS-new", "SIG-1", "path-v2-candle-repair-r1", horizon="scalp_30m")
    conn.commit()
    with pytest.raises(AmbiguousPathVersionError, match="path_definition_version"):
        fetch_path_observations(conn, "ctx-ambiguous", equals={"observation_status": "OK"})
    conn.close()


def test_fetch_path_observations_explicit_version_pin_bypasses_ambiguity_guard(tmp_path):
    """An exact path_definition_version pin never raises, even when multiple
    versions exist for the same (signal_id, horizon_name) -- the guard only
    fires on the UNSCOPED case, and reproduces the historical version only."""
    conn = _db(tmp_path)
    _insert_signal(conn, "SIG-1")
    _insert_path_observation_versioned(conn, "OBS-old", "SIG-1", "path-v2", horizon="scalp_30m")
    _insert_path_observation_versioned(conn, "OBS-new", "SIG-1", "path-v2-candle-repair-r1", horizon="scalp_30m")
    conn.commit()
    rows = fetch_path_observations(
        conn, "ctx-pinned", equals={"observation_status": "OK", "path_definition_version": "path-v2"},
    )
    assert len(rows) == 1
    assert rows[0]["observation_id"] == "OBS-old"
    conn.close()


def test_fetch_path_observations_effective_true_bypasses_ambiguity_guard(tmp_path):
    """effective=True is the other sanctioned bypass, reserved for the
    effective-path selector's own internal use -- proven here against a
    genuinely ambiguous (two-version) population, not a trivially single-
    version one."""
    conn = _db(tmp_path)
    _insert_signal(conn, "SIG-1")
    _insert_path_observation_versioned(conn, "OBS-old", "SIG-1", "path-v2", horizon="scalp_30m")
    _insert_path_observation_versioned(conn, "OBS-new", "SIG-1", "path-v2-candle-repair-r1", horizon="scalp_30m")
    conn.commit()
    rows = fetch_path_observations(conn, "ctx-effective-bypass", equals={"observation_status": "OK"}, effective=True)
    assert len(rows) == 2  # both physical versions returned -- caller (the effective selector) does its own reduction
    conn.close()


def test_effective_fetch_real_data_returns_1296_unique_pairs_zero_duplicates():
    """GOAL D items 3+4: the effective selector, run against the real
    (isolated-copy) canonical.sqlite, must return exactly 1296 rows (one per
    (signal_id, horizon_name)) with zero duplicate pairs."""
    import ami.warehouse.schema as schema_mod
    from ami.lifecycle.path_candle_repair_correction import fetch_effective_path_observations

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        rows = fetch_effective_path_observations(conn, "ctx-effective-1296")
        pairs = [(r["signal_id"], r["horizon_name"]) for r in rows]
        assert len(rows) == 1296
        assert len(pairs) == len(set(pairs))
    finally:
        conn.close()


def test_effective_fetch_corrected_rows_supersede_old_only_in_effective_queries():
    """GOAL D item 5: a corrected (signal, horizon) pair must appear as the
    corrected row in the effective fetch, while the raw, unpinned physical
    fetch remains ambiguous (proving the correction never silently mutated
    or removed the original 'path-v2' row)."""
    import ami.warehouse.schema as schema_mod
    from ami.lifecycle.path_candle_repair_correction import (
        OLD_PATH_DEFINITION_VERSION,
        PATH_DATA_VERSION_CANDLE_REPAIR_R1,
        identify_affected_pairs,
        fetch_effective_path_observations,
    )

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        affected = identify_affected_pairs(conn)
        sid, horizon = affected["class_a"][0]

        effective_rows = fetch_effective_path_observations(conn, "ctx-supersede-check")
        eff_row = next(r for r in effective_rows if r["signal_id"] == sid and r["horizon_name"] == horizon)
        assert eff_row["path_definition_version"] == PATH_DATA_VERSION_CANDLE_REPAIR_R1

        old_row = conn.execute(
            "SELECT observation_status FROM ami_lifecycle_path_observations "
            "WHERE signal_id=? AND horizon_name=? AND path_definition_version=?",
            (sid, horizon, OLD_PATH_DEFINITION_VERSION),
        ).fetchone()
        assert old_row is not None  # original row still exists, untouched

        with pytest.raises(AmbiguousPathVersionError):
            fetch_path_observations(conn, "ctx-supersede-ambiguous", equals={"signal_id": sid, "horizon_name": horizon})
    finally:
        conn.close()


def fetch_events_readonly_smoke(conn) -> int:
    # can't write exposure ledger on a read-only connection -- just prove the
    # pooling check itself passes on real data by calling assert_not_pooled directly.
    from ami.identity.event_identity import assert_not_pooled

    rows = conn.execute("SELECT source_quality FROM ami_events").fetchall()
    assert_not_pooled(r[0] for r in rows)
    return len(rows)
