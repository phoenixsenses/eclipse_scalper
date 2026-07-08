"""PHASE 7B-0 / 7B-0.1: tests for ami/lifecycle/path_metrics.py (candle
boundary semantics, MFE/MAE zero-clamp + zero-extremum timing, LONG/SHORT
sign symmetry, known-at safety, gap detection, intrabar ambiguity,
independent observation_status/volatility_status axes, anchor-vol
normalization). DISPOSABLE_DB_ONLY: synthetic fixtures are plain dicts (no DB
needed for compute_observation); the real-data smoke tests at the bottom go
through tests/conftest.py's session-scoped isolation (DEFAULT_PATH already
redirected to a disposable copy for the whole test session -- see
tests/conftest.py -- so this never touches the real canonical.sqlite).

Run: pytest tests/test_ami_lifecycle_path_metrics.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations

import ami.research.w4_post_event_path_taxonomy as w4
from ami.lifecycle.path_metrics import (
    CANDLE_MS,
    VOL_WINDOW_CANDLES,
    _CandleOHLCIndex,
    _realized_vol_at,
    compute_all,
    compute_observation,
    fetch_signals,
    fetch_symbol_candles,
    freeze_and_record,
)
from ami.lifecycle.path_schema import PATH_DEFINITION_VERSION, init_path_schema
from ami.research.w4_post_event_path_taxonomy import PATH_HORIZONS_MS, classify_path

PROV = "test"


def _mk(open_ts, high, low, close, quality="AVAILABLE", cdv="cdv-test"):
    return {"open_ts_ms": open_ts, "close_ts_ms": open_ts + CANDLE_MS, "high": high, "low": low,
            "close": close, "data_quality": quality, "candle_definition_version": cdv}


def _flat_vol_window(end_close_ts_ms: int, price: float) -> list[dict]:
    """60 identical-close candles ending at end_close_ts_ms -- realized_vol == 0."""
    rows = []
    ts = end_close_ts_ms - VOL_WINDOW_CANDLES * CANDLE_MS
    for _ in range(VOL_WINDOW_CANDLES):
        rows.append(_mk(ts, price, price, price))
        ts += CANDLE_MS
    return rows


def _varying_vol_window(end_close_ts_ms: int, base_price: float) -> list[dict]:
    """60 candles alternating +/-0.1% closes ending at end_close_ts_ms -- non-zero realized_vol."""
    rows = []
    ts = end_close_ts_ms - VOL_WINDOW_CANDLES * CANDLE_MS
    price = base_price
    for i in range(VOL_WINDOW_CANDLES):
        price = base_price * (1.001 if i % 2 == 0 else 0.999)
        rows.append(_mk(ts, price * 1.0001, price * 0.9999, price))
        ts += CANDLE_MS
    return rows


BIRTH = 100_000_000_000  # aligned to a 60_000ms boundary for most tests
AS_OF_FAR_FUTURE = BIRTH + 100 * 24 * 3600_000


def _flat_path(start_ts, n, price):
    rows = []
    ts = start_ts
    for _ in range(n):
        rows.append(_mk(ts, price, price, price))
        ts += CANDLE_MS
    return rows


# ---- reference price / candle boundary semantics ----

def test_reference_price_uses_last_fully_closed_candle_before_birth():
    pre = _flat_vol_window(BIRTH, 100.0)  # ends exactly at BIRTH (aligned)
    path = _flat_path(BIRTH, 30, 100.0)
    idx = _CandleOHLCIndex(pre + path)
    obs = compute_observation(idx, "SIG-1", "LONG", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                               AS_OF_FAR_FUTURE)
    assert obs["reference_price"] == 100.0
    assert obs["reference_price_ts"] == BIRTH


def test_partial_first_candle_excluded_from_path_high_low():
    # birth falls MID-candle: a candle opens 30s before birth and closes 30s after --
    # its high/low are extreme and must NEVER enter the MFE/MAE computation.
    straddle_open = BIRTH - 30_000
    pre = _flat_vol_window(straddle_open, 100.0)  # vol window ends just before the straddling candle
    straddling = _mk(straddle_open, high=999.0, low=1.0, close=100.0)  # extreme, must be excluded
    path = _flat_path(straddle_open + CANDLE_MS, 30, 100.0)  # clean path candles start after straddle closes
    idx = _CandleOHLCIndex(pre + [straddling] + path)
    obs = compute_observation(idx, "SIG-1", "LONG", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                               AS_OF_FAR_FUTURE)
    assert obs["observation_status"] == "OK"
    assert obs["volatility_status"] in ("OK", "INVALID_VOLATILITY_BASELINE")
    assert obs["mfe_bps"] == 0.0  # flat path after straddle -- the 999.0 high must not leak in
    assert obs["mae_bps"] == 0.0
    assert obs["time_to_mfe_ms"] == 0  # zero-extremum timing: reference point IS the achieving point
    assert obs["time_to_mae_ms"] == 0
    assert obs["effective_path_start_ts"] == straddle_open + CANDLE_MS  # first candle with open_ts>=BIRTH
    # the excluded straddling candle structurally shortens the observable window by <1 candle --
    # this is a documented boundary effect, not a data gap, so gap_count must be 0 here
    assert obs["gap_count"] == 0


# ---- MFE/MAE zero-clamp + zero-extremum timing (PHASE 7B-0.1 lock #2) ----

def test_long_only_falls_mfe_zero_time_to_mfe_zero():
    pre = _flat_vol_window(BIRTH, 100.0)
    path = [_mk(BIRTH, 100.0, 95.0, 95.0), _mk(BIRTH + CANDLE_MS, 95.0, 90.0, 90.0)]
    path += _flat_path(BIRTH + 2 * CANDLE_MS, 28, 90.0)
    idx = _CandleOHLCIndex(pre + path)
    obs = compute_observation(idx, "SIG-1", "LONG", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                               AS_OF_FAR_FUTURE)
    assert obs["mfe_bps"] == 0.0  # never negative, even though price only dropped
    assert obs["time_to_mfe_ms"] == 0  # the reference point (t=0) IS the achieving point -- never a
                                       # silently-picked real candle's timestamp
    assert obs["mae_bps"] < 0.0


def test_long_only_rises_mae_zero_time_to_mae_zero():
    pre = _flat_vol_window(BIRTH, 100.0)
    path = [_mk(BIRTH, 105.0, 100.0, 105.0), _mk(BIRTH + CANDLE_MS, 110.0, 105.0, 110.0)]
    path += _flat_path(BIRTH + 2 * CANDLE_MS, 28, 110.0)
    idx = _CandleOHLCIndex(pre + path)
    obs = compute_observation(idx, "SIG-1", "LONG", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                               AS_OF_FAR_FUTURE)
    assert obs["mae_bps"] == 0.0  # never positive
    assert obs["time_to_mae_ms"] == 0
    assert obs["mfe_bps"] > 0.0


def test_short_only_rises_mfe_zero_time_to_mfe_zero_mirrored():
    # SHORT mirror of test_long_only_falls: price RISING is adverse for SHORT, so this is the
    # SHORT-direction case where favorable excursion never happens.
    pre = _flat_vol_window(BIRTH, 100.0)
    path = [_mk(BIRTH, 105.0, 100.0, 105.0), _mk(BIRTH + CANDLE_MS, 110.0, 105.0, 110.0)]
    path += _flat_path(BIRTH + 2 * CANDLE_MS, 28, 110.0)
    idx = _CandleOHLCIndex(pre + path)
    obs = compute_observation(idx, "SIG-1", "SHORT", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                               AS_OF_FAR_FUTURE)
    assert obs["mfe_bps"] == 0.0  # price only rose -- never favorable for SHORT
    assert obs["time_to_mfe_ms"] == 0
    assert obs["mae_bps"] < 0.0  # rising price is adverse for SHORT


def test_short_only_falls_mae_zero_time_to_mae_zero_mirrored():
    # SHORT mirror of test_long_only_rises: price FALLING is favorable-only for SHORT (never adverse).
    pre = _flat_vol_window(BIRTH, 100.0)
    path = [_mk(BIRTH, 100.0, 95.0, 95.0), _mk(BIRTH + CANDLE_MS, 95.0, 90.0, 90.0)]
    path += _flat_path(BIRTH + 2 * CANDLE_MS, 28, 90.0)
    idx = _CandleOHLCIndex(pre + path)
    obs = compute_observation(idx, "SIG-1", "SHORT", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                               AS_OF_FAR_FUTURE)
    assert obs["mae_bps"] == 0.0  # never adverse for SHORT when price only falls
    assert obs["time_to_mae_ms"] == 0
    assert obs["mfe_bps"] > 0.0


def test_perfectly_flat_path_both_extrema_at_reference_point():
    pre = _flat_vol_window(BIRTH, 100.0)
    path = _flat_path(BIRTH, 30, 100.0)  # never moves at all
    idx = _CandleOHLCIndex(pre + path)
    obs = compute_observation(idx, "SIG-1", "LONG", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                               AS_OF_FAR_FUTURE)
    assert obs["mfe_bps"] == 0.0
    assert obs["mae_bps"] == 0.0
    assert obs["time_to_mfe_ms"] == 0
    assert obs["time_to_mae_ms"] == 0


# ---- LONG/SHORT sign symmetry (mirrored path) ----

def test_long_short_mirror_symmetry():
    ref = 100.0
    pre = _flat_vol_window(BIRTH, ref)
    path_a = [
        _mk(BIRTH, high=103.0, low=99.0, close=101.0),
        _mk(BIRTH + CANDLE_MS, high=104.0, low=98.0, close=99.0),
    ] + _flat_path(BIRTH + 2 * CANDLE_MS, 28, ref)
    idx_a = _CandleOHLCIndex(pre + path_a)
    obs_long = compute_observation(idx_a, "SIG-1", "LONG", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                                    AS_OF_FAR_FUTURE)

    # mirror every path candle through ref: new_high = 2ref-low, new_low = 2ref-high, new_close = 2ref-close
    def _mirror(c):
        return _mk(c["open_ts_ms"], high=2 * ref - c["low"], low=2 * ref - c["high"],
                    close=2 * ref - c["close"])

    path_b = [_mirror(c) for c in path_a]
    idx_b = _CandleOHLCIndex(pre + path_b)
    obs_short = compute_observation(idx_b, "SIG-1", "SHORT", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                                     AS_OF_FAR_FUTURE)

    assert obs_long["mfe_bps"] == obs_short["mfe_bps"]
    assert obs_long["mae_bps"] == obs_short["mae_bps"]
    assert obs_long["time_to_mfe_ms"] == obs_short["time_to_mfe_ms"]
    assert obs_long["time_to_mae_ms"] == obs_short["time_to_mae_ms"]


# ---- intrabar ambiguity ----

def test_intrabar_same_candle_unknown():
    pre = _flat_vol_window(BIRTH, 100.0)
    # a single candle dominates BOTH the max favorable and max adverse excursion
    path = [_mk(BIRTH, high=120.0, low=80.0, close=100.0)] + _flat_path(BIRTH + CANDLE_MS, 29, 100.0)
    idx = _CandleOHLCIndex(pre + path)
    obs = compute_observation(idx, "SIG-1", "LONG", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                               AS_OF_FAR_FUTURE)
    assert obs["intrabar_order_status"] == "SAME_CANDLE_UNKNOWN"


def test_intrabar_mfe_first():
    pre = _flat_vol_window(BIRTH, 100.0)
    path = [_mk(BIRTH, high=110.0, low=100.0, close=105.0),          # MFE candle (t=0)
            _mk(BIRTH + CANDLE_MS, high=105.0, low=90.0, close=95.0)]  # MAE candle (t=1)
    path += _flat_path(BIRTH + 2 * CANDLE_MS, 28, 100.0)
    idx = _CandleOHLCIndex(pre + path)
    obs = compute_observation(idx, "SIG-1", "LONG", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                               AS_OF_FAR_FUTURE)
    assert obs["intrabar_order_status"] == "MFE_FIRST"
    assert obs["time_to_mfe_ms"] < obs["time_to_mae_ms"]


def test_intrabar_mae_first():
    pre = _flat_vol_window(BIRTH, 100.0)
    path = [_mk(BIRTH, high=105.0, low=90.0, close=95.0),             # MAE candle (t=0)
            _mk(BIRTH + CANDLE_MS, high=110.0, low=100.0, close=105.0)]  # MFE candle (t=1)
    path += _flat_path(BIRTH + 2 * CANDLE_MS, 28, 100.0)
    idx = _CandleOHLCIndex(pre + path)
    obs = compute_observation(idx, "SIG-1", "LONG", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                               AS_OF_FAR_FUTURE)
    assert obs["intrabar_order_status"] == "MAE_FIRST"
    assert obs["time_to_mae_ms"] < obs["time_to_mfe_ms"]


# ---- known-at / no-lookahead ----

def test_future_candle_beyond_horizon_does_not_change_result():
    pre = _flat_vol_window(BIRTH, 100.0)
    path = _flat_path(BIRTH, 30, 100.0)
    idx_before = _CandleOHLCIndex(pre + path)
    obs_before = compute_observation(idx_before, "SIG-1", "LONG", BIRTH, "scalp_30m",
                                      PATH_HORIZONS_MS["scalp_30m"], AS_OF_FAR_FUTURE)

    horizon_end = BIRTH + PATH_HORIZONS_MS["scalp_30m"]
    future_candle = _mk(horizon_end + CANDLE_MS, high=999.0, low=1.0, close=500.0)  # extreme, AFTER horizon
    idx_after = _CandleOHLCIndex(pre + path + [future_candle])
    obs_after = compute_observation(idx_after, "SIG-1", "LONG", BIRTH, "scalp_30m",
                                     PATH_HORIZONS_MS["scalp_30m"], AS_OF_FAR_FUTURE)

    assert obs_before["mfe_bps"] == obs_after["mfe_bps"]
    assert obs_before["mae_bps"] == obs_after["mae_bps"]
    assert obs_before["endpoint_return_bps"] == obs_after["endpoint_return_bps"]


# ---- gap detection ----

def test_missing_row_in_path_window_triggers_missing_internal_gap():
    pre = _flat_vol_window(BIRTH, 100.0)
    path = _flat_path(BIRTH, 30, 100.0)
    del path[15]  # remove one candle -- a literal missing minute slot
    idx = _CandleOHLCIndex(pre + path)
    obs = compute_observation(idx, "SIG-1", "LONG", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                               AS_OF_FAR_FUTURE)
    assert obs["observation_status"] == "MISSING_INTERNAL_GAP"
    assert obs["volatility_status"] == "NOT_APPLICABLE"
    assert obs["gap_count"] > 0
    assert obs["mfe_bps"] is None
    assert obs["mae_bps"] is None


def test_gapped_quality_candle_in_path_window_triggers_missing_internal_gap():
    pre = _flat_vol_window(BIRTH, 100.0)
    path = _flat_path(BIRTH, 30, 100.0)
    path[10]["data_quality"] = "GAPPED"
    idx = _CandleOHLCIndex(pre + path)
    obs = compute_observation(idx, "SIG-1", "LONG", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                               AS_OF_FAR_FUTURE)
    assert obs["observation_status"] == "MISSING_INTERNAL_GAP"
    assert obs["volatility_status"] == "NOT_APPLICABLE"
    assert obs["gap_count"] == 1


def test_clean_window_reports_expected_observed_gap_counts():
    pre = _flat_vol_window(BIRTH, 100.0)
    path = _flat_path(BIRTH, 30, 100.0)
    idx = _CandleOHLCIndex(pre + path)
    obs = compute_observation(idx, "SIG-1", "LONG", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                               AS_OF_FAR_FUTURE)
    assert obs["expected_candle_count"] == 30
    assert obs["observed_candle_count"] == 30
    assert obs["gap_count"] == 0


# ---- maturity / excluded ----

def test_excluded_no_horizon_data_when_not_yet_matured():
    pre = _flat_vol_window(BIRTH, 100.0)
    path = _flat_path(BIRTH, 10, 100.0)  # only 10 candles -- far short of a 30m horizon
    idx = _CandleOHLCIndex(pre + path)
    as_of_ts = BIRTH + 10 * CANDLE_MS  # matches available data -- horizon has not matured
    obs = compute_observation(idx, "SIG-1", "LONG", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"], as_of_ts)
    assert obs["observation_status"] == "EXCLUDED_NO_HORIZON_DATA"
    assert obs["volatility_status"] == "NOT_APPLICABLE"
    assert obs["reference_price"] == 100.0  # reference IS still recorded
    assert obs["mfe_bps"] is None


# ---- missing reference price ----

def test_missing_reference_price_when_no_candle_before_birth():
    path = _flat_path(BIRTH, 30, 100.0)  # nothing before BIRTH at all
    idx = _CandleOHLCIndex(path)
    obs = compute_observation(idx, "SIG-1", "LONG", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                               AS_OF_FAR_FUTURE)
    assert obs["observation_status"] == "MISSING_REFERENCE_PRICE"
    assert obs["volatility_status"] == "NOT_APPLICABLE"
    assert obs["reference_price"] is None
    assert obs["mfe_bps"] is None


# ---- direction ----

def test_not_computable_direction_for_unknown():
    pre = _flat_vol_window(BIRTH, 100.0)
    path = _flat_path(BIRTH, 30, 100.0)
    idx = _CandleOHLCIndex(pre + path)
    obs = compute_observation(idx, "SIG-1", "UNKNOWN", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                               AS_OF_FAR_FUTURE)
    assert obs["observation_status"] == "NOT_COMPUTABLE_DIRECTION"
    assert obs["volatility_status"] == "NOT_APPLICABLE"
    assert obs["reference_price"] is None
    assert obs["mfe_bps"] is None


# ---- observation_status / volatility_status independence (PHASE 7B-0.1 lock #1) ----

def test_invalid_volatility_baseline_is_observation_ok_not_a_separate_observation_status():
    pre = _flat_vol_window(BIRTH, 100.0)[:VOL_WINDOW_CANDLES - 1]  # 59, one short of the 60 needed
    path = _flat_path(BIRTH, 30, 100.0)
    idx = _CandleOHLCIndex(pre + path)
    obs = compute_observation(idx, "SIG-1", "LONG", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                               AS_OF_FAR_FUTURE)
    assert obs["observation_status"] == "OK"  # path itself IS computable
    assert obs["volatility_status"] == "INVALID_VOLATILITY_BASELINE"  # only the vol baseline fails
    assert obs["endpoint_return_bps"] is not None
    assert obs["mfe_bps"] is not None
    assert obs["mae_bps"] is not None
    assert obs["time_to_mfe_ms"] is not None
    assert obs["horizon_outcome_class"] is not None
    assert obs["realized_vol_at_anchor"] is None
    assert obs["endpoint_return_anchor_vol_units"] is None
    assert obs["mfe_anchor_vol_units"] is None
    assert obs["mae_anchor_vol_units"] is None


def test_invalid_volatility_baseline_zero_vol():
    pre = _flat_vol_window(BIRTH, 100.0)  # all identical closes -> realized_vol == 0
    path = _flat_path(BIRTH, 30, 100.0)
    idx = _CandleOHLCIndex(pre + path)
    obs = compute_observation(idx, "SIG-1", "LONG", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                               AS_OF_FAR_FUTURE)
    assert obs["observation_status"] == "OK"
    assert obs["volatility_status"] == "INVALID_VOLATILITY_BASELINE"
    assert obs["mfe_bps"] is not None


def test_ok_status_with_varying_volatility():
    pre = _varying_vol_window(BIRTH, 100.0)
    path = [_mk(BIRTH, high=105.0, low=100.0, close=103.0)] + _flat_path(BIRTH + CANDLE_MS, 29, 103.0)
    idx = _CandleOHLCIndex(pre + path)
    obs = compute_observation(idx, "SIG-1", "LONG", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                               AS_OF_FAR_FUTURE)
    assert obs["observation_status"] == "OK"
    assert obs["volatility_status"] == "OK"
    assert obs["realized_vol_at_anchor"] > 0
    assert obs["mfe_anchor_vol_units"] is not None


def test_non_ok_observation_status_always_pairs_with_not_applicable_volatility():
    # NOT_COMPUTABLE_DIRECTION, MISSING_REFERENCE_PRICE, EXCLUDED_NO_HORIZON_DATA,
    # MISSING_INTERNAL_GAP -- every non-OK observation_status must carry
    # volatility_status=NOT_APPLICABLE (matches the DB-level CHECK coupling).
    pre = _flat_vol_window(BIRTH, 100.0)
    path = _flat_path(BIRTH, 30, 100.0)
    idx = _CandleOHLCIndex(pre + path)

    obs_dir = compute_observation(idx, "SIG-1", "UNKNOWN", BIRTH, "scalp_30m",
                                   PATH_HORIZONS_MS["scalp_30m"], AS_OF_FAR_FUTURE)
    assert obs_dir["observation_status"] != "OK" and obs_dir["volatility_status"] == "NOT_APPLICABLE"

    obs_ref = compute_observation(_CandleOHLCIndex(path), "SIG-1", "LONG", BIRTH, "scalp_30m",
                                   PATH_HORIZONS_MS["scalp_30m"], AS_OF_FAR_FUTURE)
    assert obs_ref["observation_status"] != "OK" and obs_ref["volatility_status"] == "NOT_APPLICABLE"


def test_vol_units_formula_lock():
    pre = _varying_vol_window(BIRTH, 100.0)
    path = [_mk(BIRTH, high=105.0, low=100.0, close=103.0)] + _flat_path(BIRTH + CANDLE_MS, 29, 103.0)
    idx = _CandleOHLCIndex(pre + path)
    realized_vol = _realized_vol_at(idx, BIRTH)
    obs = compute_observation(idx, "SIG-1", "LONG", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                               AS_OF_FAR_FUTURE)
    vol_bps = realized_vol * 1e4
    assert obs["mfe_anchor_vol_units"] == round(obs["mfe_bps"] / vol_bps, 4)
    assert obs["mae_anchor_vol_units"] == round(obs["mae_bps"] / vol_bps, 4)
    assert obs["endpoint_return_anchor_vol_units"] == round(obs["endpoint_return_bps"] / vol_bps, 4)


def test_realized_vol_at_anchor_identical_across_horizons_same_signal():
    # Locks the naming rationale (PHASE 7B-0.1 lock #5): realized_vol_at_anchor does not
    # depend on horizon_ms at all -- it must be IDENTICAL for the same signal across all
    # 4 horizon rows (never rescaled/recomputed per horizon).
    pre = _varying_vol_window(BIRTH, 100.0)
    path = _flat_path(BIRTH, 1440, 100.0)  # enough candles to cover even swing_24h
    idx = _CandleOHLCIndex(pre + path)
    results = {
        name: compute_observation(idx, "SIG-1", "LONG", BIRTH, name, ms, AS_OF_FAR_FUTURE)
        for name, ms in PATH_HORIZONS_MS.items()
    }
    vols = {r["realized_vol_at_anchor"] for r in results.values()}
    assert len(vols) == 1  # one single value across all 4 horizons


# ---- horizon_outcome_class reuse (no parallel taxonomy) ----

def test_horizon_outcome_class_reuses_w4_classify_path_verbatim():
    from ami.lifecycle import path_metrics
    assert path_metrics.classify_path is w4.classify_path


def test_horizon_outcome_class_matches_direct_classify_path_call():
    pre = _flat_vol_window(BIRTH, 100.0)
    path = [_mk(BIRTH, high=125.0, low=100.0, close=125.0)] + _flat_path(BIRTH + CANDLE_MS, 29, 125.0)
    idx = _CandleOHLCIndex(pre + path)
    obs = compute_observation(idx, "SIG-1", "LONG", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                               AS_OF_FAR_FUTURE)
    assert obs["horizon_outcome_class"] == classify_path(obs["endpoint_return_bps"])
    assert obs["horizon_outcome_class"] == "REVERSAL"  # +2000bps, well past the +-20bps band


def test_horizon_outcome_class_direction_independent():
    # endpoint_return_bps/horizon_outcome_class describe the raw PRICE PATH, not the
    # signal's own directional bet -- identical for LONG and SHORT rows on the same path.
    pre = _flat_vol_window(BIRTH, 100.0)
    path = [_mk(BIRTH, high=125.0, low=100.0, close=125.0)] + _flat_path(BIRTH + CANDLE_MS, 29, 125.0)
    idx = _CandleOHLCIndex(pre + path)
    obs_long = compute_observation(idx, "SIG-1", "LONG", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                                    AS_OF_FAR_FUTURE)
    obs_short = compute_observation(idx, "SIG-1", "SHORT", BIRTH, "scalp_30m", PATH_HORIZONS_MS["scalp_30m"],
                                     AS_OF_FAR_FUTURE)
    assert obs_long["endpoint_return_bps"] == obs_short["endpoint_return_bps"]
    assert obs_long["horizon_outcome_class"] == obs_short["horizon_outcome_class"]


# ---- identity ----

def test_observation_id_deterministic():
    from ami.lifecycle.path_metrics import _observation_id
    a = _observation_id("SIG-1", "scalp_30m", PATH_DEFINITION_VERSION)
    b = _observation_id("SIG-1", "scalp_30m", PATH_DEFINITION_VERSION)
    assert a == b


def test_observation_id_differs_by_horizon():
    from ami.lifecycle.path_metrics import _observation_id
    a = _observation_id("SIG-1", "scalp_30m", PATH_DEFINITION_VERSION)
    b = _observation_id("SIG-1", "scalp_1h", PATH_DEFINITION_VERSION)
    assert a != b


# ---- horizons frozen from W4, no new horizon invented ----

def test_horizons_reused_verbatim_from_w4():
    from ami.lifecycle import path_metrics
    assert path_metrics.PATH_HORIZONS_MS is w4.PATH_HORIZONS_MS
    assert set(path_metrics.PATH_HORIZONS_MS.keys()) == {"scalp_30m", "scalp_1h", "swing_4h", "swing_24h"}


# ---- real-data smoke test (disposable copy only, via conftest isolation) ----

def test_real_data_smoke_compute_all():
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH, read_only=True)
    try:
        result = compute_all(conn)
    finally:
        conn.close()

    # 324 = 270 original + 54 SHORT_NOISY_BTC200K_CONFIRMED_V1 (BATCH-SHORT-NOISY-V1-CANON-BACKFILL)
    assert result["raw_signal_n"] == 324
    assert result["raw_observation_row_n"] == result["raw_observation_row_n_expected_max"]
    assert result["raw_observation_row_n_expected_max"] == 324 * 4
    assert result["distinct_independent_cycle_n"] <= result["raw_signal_n"]
    assert result["distinct_source_event_n"] <= result["raw_signal_n"]
    total = sum(result["status_counts"].values())
    assert total == result["raw_observation_row_n"]
    total_vol = sum(result["volatility_status_counts"].values())
    assert total_vol == result["raw_observation_row_n"]
    assert result["path_valid_n"] == result["status_counts"].get("OK", 0)
    assert result["volatility_valid_n"] == result["volatility_status_counts"].get("OK", 0)
    assert result["volatility_valid_n"] <= result["path_valid_n"]  # vol-valid is a SUBSET of path-valid
    # NOT_COMPUTABLE_DIRECTION must be 0 today (direction distribution verified LONG=220/SHORT=50/UNKNOWN=0)
    assert result["status_counts"].get("NOT_COMPUTABLE_DIRECTION", 0) == 0
    # every non-OK observation_status row must carry volatility_status=NOT_APPLICABLE
    non_ok_rows = [r for r in result["rows"] if r["observation_status"] != "OK"]
    assert all(r["volatility_status"] == "NOT_APPLICABLE" for r in non_ok_rows)
    # at least one signal's horizon has not matured yet (most recent signal_birth_ts is
    # only ~2.8h before the candle table's own maturity cutoff)
    assert result["status_counts"].get("EXCLUDED_NO_HORIZON_DATA", 0) > 0


def test_real_data_smoke_realized_vol_identical_across_horizons_per_signal():
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH, read_only=True)
    try:
        result = compute_all(conn)
    finally:
        conn.close()

    by_signal: dict[str, set] = {}
    for r in result["rows"]:
        if r["observation_status"] == "OK":
            by_signal.setdefault(r["signal_id"], set()).add(r["realized_vol_at_anchor"])
    assert by_signal  # at least some OK rows exist
    assert all(len(vols) == 1 for vols in by_signal.values())


def test_real_data_smoke_freeze_and_record_idempotent():
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        init_path_schema(conn)
        r1 = freeze_and_record(conn)
        count_1 = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0]
        r2 = freeze_and_record(conn)
        count_2 = conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0]
        assert count_1 == count_2
        assert r1["raw_observation_row_n"] == r2["raw_observation_row_n"]
    finally:
        conn.close()
