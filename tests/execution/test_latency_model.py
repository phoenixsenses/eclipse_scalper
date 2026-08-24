from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.microphys.execution.latency import (
    build_latency_timeline,
    latency_bars,
    parse_latency_profile,
    sample_stage_latency,
    stage_to_legacy_components,
)


def test_latency_fixed_profile_deterministic() -> None:
    params = {
        "latency_enabled": True,
        "latency_profile": "fixed",
        "latency_send_ms": 10.0,
        "latency_exchange_recv_ms": 20.0,
        "latency_book_effective_ms": 30.0,
        "latency_ack_ms": 40.0,
        "latency_fill_ms": 50.0,
    }
    prof = parse_latency_profile(params)
    s1 = sample_stage_latency(prof, seed=42, event_id="evt_a")
    s2 = sample_stage_latency(prof, seed=999, event_id="evt_b")
    assert s1 == s2
    leg = stage_to_legacy_components(s1)
    assert abs(float(leg["decision_to_ack_ms"]) - 70.0) < 1e-9
    assert abs(float(leg["queue_entry_ms"]) - 30.0) < 1e-9
    assert abs(float(leg["feed_lag_ms"]) - 50.0) < 1e-9
    assert int(latency_bars(float(leg["total_ms"]), 1.0)) == 0


def test_latency_normal_profile_bounded_jitter() -> None:
    params = {
        "latency_enabled": True,
        "latency_profile": "normal",
        "latency_send_ms": 20.0,
        "latency_exchange_recv_ms": 30.0,
        "latency_book_effective_ms": 10.0,
        "latency_ack_ms": 10.0,
        "latency_fill_ms": 5.0,
        "latency_jitter_ms": 3.0,
    }
    prof = parse_latency_profile(params)
    s = sample_stage_latency(prof, seed=123, event_id="evt_x")
    # bounded +/- jitter around each stage mean
    assert 17.0 <= float(s.send_ms) <= 23.0
    assert 27.0 <= float(s.exchange_recv_ms) <= 33.0
    assert 7.0 <= float(s.book_effective_ms) <= 13.0
    assert 7.0 <= float(s.ack_ms) <= 13.0
    assert 2.0 <= float(s.fill_ms) <= 8.0


def test_latency_empirical_profile_deterministic_bucket_pick() -> None:
    params = {
        "latency_enabled": True,
        "latency_profile": "empirical",
        "latency_send_ms": 2.0,
        "latency_exchange_recv_ms": 2.0,
        "latency_book_effective_ms": 2.0,
        "latency_ack_ms": 2.0,
        "latency_fill_ms": 2.0,
        "latency_empirical_ms": [20.0, 80.0, 200.0],
        "latency_empirical_weights": [0.7, 0.2, 0.1],
    }
    prof = parse_latency_profile(params)
    s1 = sample_stage_latency(prof, seed=42, event_id="evt_emp")
    s2 = sample_stage_latency(prof, seed=42, event_id="evt_emp")
    assert s1 == s2
    assert float(s1.total_ms) in {20.0, 80.0, 200.0}


def test_latency_timeline_monotonic() -> None:
    params = {
        "latency_enabled": True,
        "latency_profile": "fixed",
        "latency_send_ms": 10.0,
        "latency_exchange_recv_ms": 20.0,
        "latency_book_effective_ms": 30.0,
        "latency_ack_ms": 40.0,
        "latency_fill_ms": 50.0,
    }
    prof = parse_latency_profile(params)
    s = sample_stage_latency(prof, seed=1, event_id="evt_tl")
    tl = build_latency_timeline(decision_ts_ms=1_772_000_000_000, stage=s)
    assert tl["decision_ts"] <= tl["send_ts"] <= tl["exchange_recv_ts"] <= tl["book_effective_ts"] <= tl["ack_ts"] <= tl["fill_ts"]
