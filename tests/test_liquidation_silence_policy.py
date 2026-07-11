"""Pure decision-policy tests for tools/liquidation_silence_policy.py (v2).

No I/O, no database, no filesystem -- every test constructs its inputs
directly and checks the returned classification/severity/reasons. See
tests/test_liquidation_silence_detector.py for I/O, DB-replay, evaluation-
mode, and writer-ownership coverage.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.liquidation_silence_policy import (
    ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC,
    ALL_SYMBOL_SILENCE_WARNING_AGE_SEC,
    CONTROL_STREAM_FRESH_AGE_SEC,
    FUTURE_TIMESTAMP_TOLERANCE_SEC,
    POLICY_FINGERPRINT,
    POLICY_SPEC,
    POLICY_VERSION,
    SEVERITY_GREEN,
    SEVERITY_RED,
    SEVERITY_UNKNOWN,
    SEVERITY_YELLOW,
    STATUS_ALL_SYMBOL_SILENCE_WARNING,
    STATUS_CONTROL_STREAMS_STALE,
    STATUS_HEALTHY,
    STATUS_LIQUIDATION_TRANSPORT_OUTAGE,
    STATUS_NATIVE_WS_UNHEALTHY,
    STATUS_PARTIAL_SYMBOL_EVIDENCE,
    STATUS_SYMBOL_SILENCE_WARNING,
    STATUS_UNKNOWN,
    SUPERSEDED_V1_POLICY_FINGERPRINT,
    SYMBOL_SILENCE_WARNING_AGE_SEC,
    _compute_policy_fingerprint,
    _fingerprint_of,
    _policy_spec,
    compose_with_overall_severity,
    evaluate_liquidation_silence,
    normalize_tracked_symbols,
)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
NOW = 1_800_000_000.0


def _ts_ms(age_sec: float) -> int:
    return int((NOW - age_sec) * 1000)


def _evaluate(**overrides):
    kwargs = dict(
        now_ts=NOW,
        tracked_symbols=list(SYMBOLS),
        last_liquidation_ts_ms={s: _ts_ms(1.0) for s in SYMBOLS},
        mark_prices_age_sec=1.0,
        agg_trades_age_sec=1.0,
        native_ws_status="GREEN",
        collector_process_alive=True,
    )
    kwargs.update(overrides)
    return evaluate_liquidation_silence(**kwargs)


# 1. recent liquidation + active controls -> GREEN
def test_healthy_all_fresh_is_green():
    r = _evaluate()
    assert r["status"] == STATUS_HEALTHY
    assert r["severity"] == SEVERITY_GREEN
    assert r["reasons"] == []
    assert r["complete_symbol_evidence"] is True
    assert r["tracked_symbol_count"] == 3
    assert r["known_symbol_count"] == 3
    assert r["missing_symbols"] == []


# 2. one symbol silent, others active -> warning classification, not all-symbol outage
def test_single_symbol_silence_is_symbol_warning_not_all_symbol():
    ts = {s: _ts_ms(1.0) for s in SYMBOLS}
    ts["SOLUSDT"] = _ts_ms(SYMBOL_SILENCE_WARNING_AGE_SEC + 10.0)
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["status"] == STATUS_SYMBOL_SILENCE_WARNING
    assert r["severity"] == SEVERITY_YELLOW
    assert any("SOLUSDT" in reason for reason in r["reasons"])
    assert r["status"] != STATUS_ALL_SYMBOL_SILENCE_WARNING


# 3. all symbols silent beyond warning while controls advance -> YELLOW
def test_all_symbol_silence_beyond_warning_is_yellow():
    ts = {s: _ts_ms(ALL_SYMBOL_SILENCE_WARNING_AGE_SEC + 10.0) for s in SYMBOLS}
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["status"] == STATUS_ALL_SYMBOL_SILENCE_WARNING
    assert r["severity"] == SEVERITY_YELLOW
    assert r["complete_symbol_evidence"] is True


# 4. all symbols silent beyond critical while controls advance -> RED (complete evidence)
def test_all_symbol_silence_beyond_critical_is_red():
    ts = {s: _ts_ms(ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC + 10.0) for s in SYMBOLS}
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["status"] == STATUS_LIQUIDATION_TRANSPORT_OUTAGE
    assert r["severity"] == SEVERITY_RED
    assert r["complete_symbol_evidence"] is True


# --- CORRECTION 1: complete-evidence gating ---------------------------------

# Review reproduction: BTC/ETH critically silent + SOL evidence missing must
# NOT declare LIQUIDATION_TRANSPORT_OUTAGE / RED.
def test_partial_evidence_critical_silence_is_not_transport_outage():
    ts = {
        "BTCUSDT": _ts_ms(ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC + 100.0),
        "ETHUSDT": _ts_ms(ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC + 100.0),
        "SOLUSDT": None,
    }
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["status"] == STATUS_PARTIAL_SYMBOL_EVIDENCE
    assert r["severity"] == SEVERITY_UNKNOWN
    assert r["status"] != STATUS_LIQUIDATION_TRANSPORT_OUTAGE
    assert r["complete_symbol_evidence"] is False
    assert r["missing_symbols"] == ["SOLUSDT"]
    # must not use the all-symbol reason codes on incomplete evidence
    assert "ALL_SYMBOL_SILENCE_BEYOND_CRITICAL" not in r["reasons"]
    assert "ALL_SYMBOL_SILENCE_BEYOND_WARNING" not in r["reasons"]
    assert any("PARTIAL_SYMBOL_EVIDENCE_MISSING" in x for x in r["reasons"])
    assert "KNOWN_SYMBOL_SILENCE_BEYOND_CRITICAL" in r["reasons"]


def test_partial_evidence_warning_silence_is_not_all_symbol_warning():
    ts = {
        "BTCUSDT": _ts_ms(ALL_SYMBOL_SILENCE_WARNING_AGE_SEC + 100.0),
        "ETHUSDT": _ts_ms(ALL_SYMBOL_SILENCE_WARNING_AGE_SEC + 100.0),
        "SOLUSDT": None,
    }
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["status"] == STATUS_PARTIAL_SYMBOL_EVIDENCE
    assert r["severity"] == SEVERITY_UNKNOWN
    assert r["status"] != STATUS_ALL_SYMBOL_SILENCE_WARNING
    assert "ALL_SYMBOL_SILENCE_BEYOND_WARNING" not in r["reasons"]
    assert "KNOWN_SYMBOL_SILENCE_BEYOND_WARNING" in r["reasons"]


# item D: partial evidence while native WS RED -> upstream RED preserved
def test_partial_evidence_with_native_ws_red_preserves_red():
    ts = {
        "BTCUSDT": _ts_ms(ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC + 100.0),
        "ETHUSDT": _ts_ms(ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC + 100.0),
        "SOLUSDT": None,
    }
    r = _evaluate(last_liquidation_ts_ms=ts, native_ws_status="RED")
    assert r["status"] == STATUS_NATIVE_WS_UNHEALTHY
    assert r["severity"] == SEVERITY_RED
    assert r["status"] != STATUS_LIQUIDATION_TRANSPORT_OUTAGE


# item E: isolated known-symbol warning still works under complete evidence
def test_isolated_symbol_warning_under_complete_evidence():
    ts = {s: _ts_ms(1.0) for s in SYMBOLS}
    ts["BTCUSDT"] = _ts_ms(SYMBOL_SILENCE_WARNING_AGE_SEC + 10.0)
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["status"] == STATUS_SYMBOL_SILENCE_WARNING
    assert r["complete_symbol_evidence"] is True


# 5. liquidation silent but mark/agg controls stale -> UNKNOWN/CONTROL_STALE, not liquidation-outage RED
def test_controls_stale_prevents_transport_outage_claim():
    ts = {s: _ts_ms(ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC + 10.0) for s in SYMBOLS}
    r = _evaluate(
        last_liquidation_ts_ms=ts,
        mark_prices_age_sec=CONTROL_STREAM_FRESH_AGE_SEC + 500.0,
        agg_trades_age_sec=CONTROL_STREAM_FRESH_AGE_SEC + 500.0,
    )
    assert r["status"] == STATUS_CONTROL_STREAMS_STALE
    assert r["severity"] == SEVERITY_UNKNOWN
    assert r["status"] != STATUS_LIQUIDATION_TRANSPORT_OUTAGE


def test_controls_stale_also_suppresses_warning_tier_transport_claim():
    ts = {s: _ts_ms(ALL_SYMBOL_SILENCE_WARNING_AGE_SEC + 10.0) for s in SYMBOLS}
    r = _evaluate(
        last_liquidation_ts_ms=ts,
        mark_prices_age_sec=CONTROL_STREAM_FRESH_AGE_SEC + 5.0,
        agg_trades_age_sec=1.0,
    )
    assert r["status"] == STATUS_CONTROL_STREAMS_STALE
    assert r["severity"] == SEVERITY_UNKNOWN


# 6. native WS RED -> severe state preserved
def test_native_ws_red_is_preserved_even_when_liquidation_healthy():
    r = _evaluate(native_ws_status="RED")
    assert r["status"] == STATUS_NATIVE_WS_UNHEALTHY
    assert r["severity"] == SEVERITY_RED


def test_native_ws_degraded_preserved_when_no_liquidation_specific_evidence():
    r = _evaluate(native_ws_status="DEGRADED")
    assert r["status"] == STATUS_NATIVE_WS_UNHEALTHY
    assert r["severity"] == SEVERITY_YELLOW


# 7. exact warning boundary
def test_all_symbol_warning_exact_boundary_triggers():
    ts = {s: _ts_ms(ALL_SYMBOL_SILENCE_WARNING_AGE_SEC) for s in SYMBOLS}
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["status"] == STATUS_ALL_SYMBOL_SILENCE_WARNING


def test_all_symbol_warning_just_below_boundary_is_healthy():
    ts = {s: _ts_ms(ALL_SYMBOL_SILENCE_WARNING_AGE_SEC - 0.5) for s in SYMBOLS}
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["status"] == STATUS_HEALTHY


# 8. exact critical boundary
def test_all_symbol_critical_exact_boundary_triggers():
    ts = {s: _ts_ms(ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC) for s in SYMBOLS}
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["status"] == STATUS_LIQUIDATION_TRANSPORT_OUTAGE


def test_all_symbol_critical_just_below_boundary_is_warning_not_outage():
    ts = {s: _ts_ms(ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC - 0.5) for s in SYMBOLS}
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["status"] == STATUS_ALL_SYMBOL_SILENCE_WARNING


# 9. threshold inclusivity/exclusivity (per-symbol tier)
def test_symbol_silence_exact_boundary_triggers_inclusive():
    ts = {s: _ts_ms(1.0) for s in SYMBOLS}
    ts["ETHUSDT"] = _ts_ms(SYMBOL_SILENCE_WARNING_AGE_SEC)
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["status"] == STATUS_SYMBOL_SILENCE_WARNING


def test_symbol_silence_just_below_boundary_is_healthy():
    ts = {s: _ts_ms(1.0) for s in SYMBOLS}
    ts["ETHUSDT"] = _ts_ms(SYMBOL_SILENCE_WARNING_AGE_SEC - 0.5)
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["status"] == STATUS_HEALTHY


# 10. missing symbol evidence (known symbols healthy -> stays healthy, partial reason)
def test_missing_symbol_evidence_partial_still_evaluates_remaining():
    ts = {s: _ts_ms(1.0) for s in SYMBOLS}
    ts["SOLUSDT"] = None
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["status"] == STATUS_HEALTHY
    assert any("PARTIAL_SYMBOL_EVIDENCE_MISSING" in reason for reason in r["reasons"])
    assert r["per_symbol_silence_age_sec"]["SOLUSDT"] is None
    assert r["complete_symbol_evidence"] is False


def test_all_symbol_evidence_missing_is_unknown():
    ts = {s: None for s in SYMBOLS}
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["status"] == STATUS_UNKNOWN
    assert r["severity"] == SEVERITY_UNKNOWN


# missing symbol + native WS RED -> RED preserved (required test 4)
def test_missing_symbol_with_native_ws_red_is_red():
    ts = {s: _ts_ms(1.0) for s in SYMBOLS}
    ts["SOLUSDT"] = None
    r = _evaluate(last_liquidation_ts_ms=ts, native_ws_status="RED")
    assert r["status"] == STATUS_NATIVE_WS_UNHEALTHY
    assert r["severity"] == SEVERITY_RED


# 11. missing control evidence
def test_missing_control_evidence_when_silence_is_material_is_unknown():
    ts = {s: _ts_ms(ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC + 10.0) for s in SYMBOLS}
    r = _evaluate(last_liquidation_ts_ms=ts, mark_prices_age_sec=None, agg_trades_age_sec=1.0)
    assert r["status"] == STATUS_UNKNOWN
    assert r["status"] != STATUS_LIQUIDATION_TRANSPORT_OUTAGE


def test_missing_control_evidence_irrelevant_when_liquidation_healthy():
    r = _evaluate(mark_prices_age_sec=None, agg_trades_age_sec=None)
    assert r["status"] == STATUS_HEALTHY


# 12. malformed component input
def test_malformed_last_liquidation_input_is_unknown():
    r = _evaluate(last_liquidation_ts_ms="not-a-dict")
    assert r["status"] == STATUS_UNKNOWN


def test_empty_tracked_symbols_is_unknown():
    r = _evaluate(tracked_symbols=[])
    assert r["status"] == STATUS_UNKNOWN


def test_non_list_tracked_symbols_is_unknown():
    r = _evaluate(tracked_symbols="BTCUSDT")
    assert r["status"] == STATUS_UNKNOWN


def test_non_numeric_symbol_timestamp_treated_as_missing():
    ts = {s: _ts_ms(1.0) for s in SYMBOLS}
    ts["BTCUSDT"] = "garbage"
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["per_symbol_silence_age_sec"]["BTCUSDT"] is None
    assert any("BTCUSDT" in reason for reason in r["reasons"])


def test_collector_process_missing_is_unknown_not_red():
    r = _evaluate(collector_process_alive=False)
    assert r["status"] == STATUS_UNKNOWN


# --- CORRECTION 8: symbol normalization -------------------------------------
def test_normalize_dedup_and_case_and_whitespace():
    norm = normalize_tracked_symbols([" btcusdt ", "BTCUSDT", "ethusdt", "", "  ", "SOLUSDT"])
    assert norm["symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_duplicate_symbols_are_deduplicated_before_evaluation():
    # A duplicate must not inflate the tracked count nor break the all-symbol
    # aggregation. BTCUSDT duplicated + all three silent beyond critical.
    dup = ["BTCUSDT", "btcusdt", "ETHUSDT", "SOLUSDT"]
    ts = {s: _ts_ms(ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC + 10.0) for s in SYMBOLS}
    r = _evaluate(tracked_symbols=dup, last_liquidation_ts_ms=ts)
    assert r["tracked_symbol_count"] == 3
    assert r["status"] == STATUS_LIQUIDATION_TRANSPORT_OUTAGE


def test_empty_normalized_universe_fails_visibly():
    r = _evaluate(tracked_symbols=["", "   ", None])
    assert r["status"] == STATUS_UNKNOWN
    assert r["severity"] == SEVERITY_UNKNOWN


# --- CORRECTION 7: future-timestamp handling --------------------------------
def test_future_liquidation_within_tolerance_is_healthy():
    ts = {s: _ts_ms(-(FUTURE_TIMESTAMP_TOLERANCE_SEC - 10.0)) for s in SYMBOLS}  # ~50s in the future
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["status"] == STATUS_HEALTHY
    assert all(v == 0.0 for v in r["per_symbol_silence_age_sec"].values())


def test_future_liquidation_beyond_tolerance_is_anomaly_unknown():
    ts = {s: _ts_ms(-5000.0) for s in SYMBOLS}  # far in the future
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["status"] == STATUS_UNKNOWN
    assert any("FUTURE_LIQUIDATION_TS_ANOMALY" in x for x in r["reasons"])
    assert all(v is None for v in r["per_symbol_silence_age_sec"].values())


def test_future_control_timestamp_beyond_tolerance_is_anomaly_not_advancing():
    ts = {s: _ts_ms(ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC + 10.0) for s in SYMBOLS}
    # mark_prices reported as far in the future (negative age beyond tolerance)
    r = _evaluate(last_liquidation_ts_ms=ts, mark_prices_age_sec=-5000.0, agg_trades_age_sec=1.0)
    assert r["status"] != STATUS_LIQUIDATION_TRANSPORT_OUTAGE
    assert r["status"] == STATUS_CONTROL_STREAMS_STALE
    assert any("FUTURE_CONTROL_TS_ANOMALY" in x for x in r["reasons"])


def test_future_control_timestamp_within_tolerance_is_fresh():
    ts = {s: _ts_ms(ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC + 10.0) for s in SYMBOLS}
    r = _evaluate(last_liquidation_ts_ms=ts, mark_prices_age_sec=-30.0, agg_trades_age_sec=-30.0)
    assert r["status"] == STATUS_LIQUIDATION_TRANSPORT_OUTAGE


# --- deterministic serialization / fingerprint ------------------------------
def test_policy_fingerprint_is_deterministic():
    assert POLICY_FINGERPRINT == _compute_policy_fingerprint()
    assert POLICY_FINGERPRINT == _compute_policy_fingerprint()


def test_new_fingerprint_differs_from_superseded_v1():
    assert POLICY_FINGERPRINT != SUPERSEDED_V1_POLICY_FINGERPRINT
    assert POLICY_VERSION.startswith("liquidation_silence_policy_v2")


def test_fingerprint_insensitive_to_key_ordering():
    spec = _policy_spec()
    reordered = json.loads(json.dumps(spec))  # round-trip
    # rebuild with reversed top-level key order
    reordered = {k: reordered[k] for k in reversed(list(reordered.keys()))}
    assert _fingerprint_of(reordered) == POLICY_FINGERPRINT


def test_fingerprint_changes_on_threshold_change():
    spec = copy.deepcopy(POLICY_SPEC)
    spec["thresholds"]["all_symbol_silence_critical_age_sec"] = 9999.0
    assert _fingerprint_of(spec) != POLICY_FINGERPRINT


def test_fingerprint_changes_on_boundary_operator_change():
    spec = copy.deepcopy(POLICY_SPEC)
    spec["boundary_operators"]["silence_tier"] = ">"
    assert _fingerprint_of(spec) != POLICY_FINGERPRINT


def test_fingerprint_changes_on_precedence_change():
    spec = copy.deepcopy(POLICY_SPEC)
    spec["decision_precedence"] = list(reversed(spec["decision_precedence"]))
    assert _fingerprint_of(spec) != POLICY_FINGERPRINT


def test_fingerprint_changes_on_complete_evidence_change():
    spec = copy.deepcopy(POLICY_SPEC)
    spec["complete_evidence_required_for_all_symbol_claim"] = False
    assert _fingerprint_of(spec) != POLICY_FINGERPRINT


def test_fingerprint_changes_on_native_ws_precedence_change():
    spec = copy.deepcopy(POLICY_SPEC)
    spec["native_ws_precedence"]["RED"] = "ignored"
    assert _fingerprint_of(spec) != POLICY_FINGERPRINT


def test_result_payload_is_json_round_trippable_and_stable():
    r1 = _evaluate()
    r2 = _evaluate()
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


# --- future-composition mapping (isolated) ----------------------------------
def test_compose_with_overall_severity_yellow_raises_green():
    assert compose_with_overall_severity(SEVERITY_GREEN, SEVERITY_YELLOW) == SEVERITY_YELLOW


def test_compose_with_overall_severity_red_forces_red():
    assert compose_with_overall_severity(SEVERITY_GREEN, SEVERITY_RED) == SEVERITY_RED
    assert compose_with_overall_severity(SEVERITY_YELLOW, SEVERITY_RED) == SEVERITY_RED


def test_compose_with_overall_severity_green_never_downgrades():
    assert compose_with_overall_severity(SEVERITY_RED, SEVERITY_GREEN) == SEVERITY_RED
    assert compose_with_overall_severity(SEVERITY_YELLOW, SEVERITY_GREEN) == SEVERITY_YELLOW


def test_compose_with_overall_severity_unknown_raises_at_least_yellow():
    assert compose_with_overall_severity(SEVERITY_GREEN, SEVERITY_UNKNOWN) == SEVERITY_YELLOW
