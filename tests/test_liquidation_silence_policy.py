"""Pure decision-policy tests for tools/liquidation_silence_policy.py.

No I/O, no database, no filesystem -- every test constructs its inputs
directly and checks the returned classification/severity/reasons. See
tests/test_liquidation_silence_detector.py for I/O, DB-replay, and
writer-ownership coverage.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.liquidation_silence_policy import (
    ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC,
    ALL_SYMBOL_SILENCE_WARNING_AGE_SEC,
    CONTROL_STREAM_FRESH_AGE_SEC,
    POLICY_FINGERPRINT,
    SEVERITY_GREEN,
    SEVERITY_RED,
    SEVERITY_UNKNOWN,
    SEVERITY_YELLOW,
    STATUS_ALL_SYMBOL_SILENCE_WARNING,
    STATUS_CONTROL_STREAMS_STALE,
    STATUS_HEALTHY,
    STATUS_LIQUIDATION_TRANSPORT_OUTAGE,
    STATUS_NATIVE_WS_UNHEALTHY,
    STATUS_SYMBOL_SILENCE_WARNING,
    STATUS_UNKNOWN,
    SYMBOL_SILENCE_WARNING_AGE_SEC,
    _compute_policy_fingerprint,
    compose_with_overall_severity,
    evaluate_liquidation_silence,
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


# 4. all symbols silent beyond critical while controls advance -> RED
def test_all_symbol_silence_beyond_critical_is_red():
    ts = {s: _ts_ms(ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC + 10.0) for s in SYMBOLS}
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["status"] == STATUS_LIQUIDATION_TRANSPORT_OUTAGE
    assert r["severity"] == SEVERITY_RED


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


# 10. missing symbol evidence
def test_missing_symbol_evidence_partial_still_evaluates_remaining():
    ts = {s: _ts_ms(1.0) for s in SYMBOLS}
    ts["SOLUSDT"] = None
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["status"] == STATUS_HEALTHY
    assert any("PARTIAL_SYMBOL_EVIDENCE_MISSING" in reason for reason in r["reasons"])
    assert r["per_symbol_silence_age_sec"]["SOLUSDT"] is None


def test_all_symbol_evidence_missing_is_unknown():
    ts = {s: None for s in SYMBOLS}
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["status"] == STATUS_UNKNOWN
    assert r["severity"] == SEVERITY_UNKNOWN


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


def test_non_numeric_symbol_timestamp_treated_as_missing():
    ts = {s: _ts_ms(1.0) for s in SYMBOLS}
    ts["BTCUSDT"] = "garbage"
    r = _evaluate(last_liquidation_ts_ms=ts)
    assert r["per_symbol_silence_age_sec"]["BTCUSDT"] is None
    assert any("BTCUSDT" in reason for reason in r["reasons"])


def test_collector_process_missing_is_unknown_not_red():
    r = _evaluate(collector_process_alive=False)
    assert r["status"] == STATUS_UNKNOWN


# 15 (policy half). deterministic serialization / fingerprint
def test_policy_fingerprint_is_deterministic():
    assert POLICY_FINGERPRINT == _compute_policy_fingerprint()
    assert POLICY_FINGERPRINT == _compute_policy_fingerprint()


def test_result_payload_is_json_round_trippable_and_stable():
    import json

    r1 = _evaluate()
    r2 = _evaluate()
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


# 20. enabled synthetic component maps YELLOW/RED correctly in isolated tests
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
