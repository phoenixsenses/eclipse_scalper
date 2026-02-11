#!/usr/bin/env python3
from __future__ import annotations

import types
import unittest
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution.belief_controller import BeliefController  # noqa: E402


class _Clock:
    def __init__(self, start: float = 0.0):
        self.t = float(start)

    def now(self) -> float:
        return float(self.t)

    def tick(self, seconds: float) -> None:
        self.t += float(seconds)


class BeliefControllerTests(unittest.TestCase):
    def test_monotone_risk_mapping(self):
        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=100.0,
            BELIEF_GROWTH_REF_PER_MIN=120.0,
            BELIEF_SYMBOL_WEIGHT=0.0,
            BELIEF_STREAK_WEIGHT=0.0,
            BELIEF_YELLOW_SCORE=99.0,
            BELIEF_ORANGE_SCORE=199.0,
            BELIEF_RED_SCORE=299.0,
            BELIEF_YELLOW_GROWTH=99.0,
            BELIEF_ORANGE_GROWTH=199.0,
            BELIEF_RED_GROWTH=299.0,
            BELIEF_RISK_SLOPE=0.45,
            FIXED_NOTIONAL_USDT=100.0,
            LEVERAGE=20,
            ENTRY_MIN_CONFIDENCE=0.25,
            BELIEF_MIN_CONF_SPREAD=0.5,
        )
        clock = _Clock(10.0)
        ctl = BeliefController(clock=clock.now)

        k1 = ctl.update({"belief_debt_sec": 0.0, "belief_debt_symbols": 0, "mismatch_streak": 0}, cfg)
        clock.tick(5.0)
        k2 = ctl.update({"belief_debt_sec": 80.0, "belief_debt_symbols": 0, "mismatch_streak": 0}, cfg)
        clock.tick(5.0)
        k3 = ctl.update({"belief_debt_sec": 140.0, "belief_debt_symbols": 0, "mismatch_streak": 0}, cfg)

        self.assertGreaterEqual(k1.max_notional_usdt, k2.max_notional_usdt)
        self.assertGreaterEqual(k2.max_notional_usdt, k3.max_notional_usdt)
        self.assertGreaterEqual(k1.max_leverage, k2.max_leverage)
        self.assertGreaterEqual(k2.max_leverage, k3.max_leverage)
        self.assertLessEqual(k1.min_entry_conf, k2.min_entry_conf)
        self.assertLessEqual(k2.min_entry_conf, k3.min_entry_conf)

    def test_hysteresis_prevents_flapping(self):
        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=100.0,
            BELIEF_SYMBOL_WEIGHT=0.0,
            BELIEF_STREAK_WEIGHT=0.0,
            BELIEF_YELLOW_SCORE=0.8,
            BELIEF_ORANGE_SCORE=1.2,
            BELIEF_RED_SCORE=9.0,
            BELIEF_YELLOW_GROWTH=9.0,
            BELIEF_ORANGE_GROWTH=9.0,
            BELIEF_RED_GROWTH=99.0,
            BELIEF_MODE_PERSIST_SEC=4.0,
            BELIEF_MODE_RECOVER_SEC=6.0,
            BELIEF_DOWN_HYST=0.8,
            FIXED_NOTIONAL_USDT=50.0,
            LEVERAGE=10,
            ENTRY_MIN_CONFIDENCE=0.2,
        )
        clock = _Clock(10.0)
        ctl = BeliefController(clock=clock.now)

        ctl.update({"belief_debt_sec": 130.0, "belief_debt_symbols": 0, "mismatch_streak": 0}, cfg)
        self.assertEqual(ctl.mode, "GREEN")
        clock.tick(5.0)
        ctl.update({"belief_debt_sec": 130.0, "belief_debt_symbols": 0, "mismatch_streak": 0}, cfg)
        self.assertEqual(ctl.mode, "ORANGE")

        clock.tick(1.0)
        ctl.update({"belief_debt_sec": 95.0, "belief_debt_symbols": 0, "mismatch_streak": 0}, cfg)
        self.assertEqual(ctl.mode, "ORANGE")
        clock.tick(7.0)
        ctl.update({"belief_debt_sec": 95.0, "belief_debt_symbols": 0, "mismatch_streak": 0}, cfg)
        self.assertEqual(ctl.mode, "YELLOW")

    def test_growth_burst_trips_red(self):
        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=100.0,
            BELIEF_SYMBOL_WEIGHT=0.0,
            BELIEF_STREAK_WEIGHT=0.0,
            BELIEF_YELLOW_SCORE=9.0,
            BELIEF_ORANGE_SCORE=10.0,
            BELIEF_RED_SCORE=11.0,
            BELIEF_YELLOW_GROWTH=0.1,
            BELIEF_ORANGE_GROWTH=0.2,
            BELIEF_RED_GROWTH=0.3,
            FIXED_NOTIONAL_USDT=100.0,
            LEVERAGE=20,
        )
        clock = _Clock(10.0)
        ctl = BeliefController(clock=clock.now)

        k1 = ctl.update({"belief_debt_sec": 0.0, "belief_debt_symbols": 0, "mismatch_streak": 0}, cfg)
        self.assertFalse(k1.kill_switch_trip)
        clock.tick(1.0)
        k2 = ctl.update({"belief_debt_sec": 75.0, "belief_debt_symbols": 0, "mismatch_streak": 0}, cfg)
        self.assertTrue(k2.kill_switch_trip)
        self.assertEqual(k2.mode, "RED")

    def test_exit_intent_always_allowed(self):
        ctl = BeliefController(clock=lambda: 0.0)
        self.assertTrue(ctl.allows_intent("exit"))
        self.assertTrue(ctl.allows_intent("stop"))
        self.assertTrue(ctl.allows_intent("tp"))

    def test_evidence_only_ws_degrade_does_not_force_red(self):
        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=300.0,
            BELIEF_SYMBOL_WEIGHT=0.0,
            BELIEF_STREAK_WEIGHT=0.0,
            BELIEF_EVIDENCE_WEIGHT=0.4,
            BELIEF_EVIDENCE_SOURCE_WEIGHT=0.05,
            BELIEF_YELLOW_SCORE=0.6,
            BELIEF_ORANGE_SCORE=1.4,
            BELIEF_RED_SCORE=2.0,
            BELIEF_YELLOW_GROWTH=99.0,
            BELIEF_ORANGE_GROWTH=99.0,
            BELIEF_RED_GROWTH=99.0,
            BELIEF_MODE_PERSIST_SEC=1.0,
        )
        clock = _Clock(10.0)
        ctl = BeliefController(clock=clock.now)
        clock.tick(2.0)
        k = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "evidence_confidence": 0.65,
                "evidence_degraded_sources": 1,
            },
            cfg,
        )
        self.assertNotEqual(k.mode, "RED")
        self.assertFalse(k.kill_switch_trip)

    def test_evidence_dual_degrade_can_trip_red(self):
        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=300.0,
            BELIEF_SYMBOL_WEIGHT=0.0,
            BELIEF_STREAK_WEIGHT=0.0,
            BELIEF_EVIDENCE_WEIGHT=1.4,
            BELIEF_EVIDENCE_SOURCE_WEIGHT=0.4,
            BELIEF_YELLOW_SCORE=0.7,
            BELIEF_ORANGE_SCORE=1.2,
            BELIEF_RED_SCORE=1.6,
            BELIEF_YELLOW_GROWTH=99.0,
            BELIEF_ORANGE_GROWTH=99.0,
            BELIEF_RED_GROWTH=99.0,
        )
        clock = _Clock(10.0)
        ctl = BeliefController(clock=clock.now)
        k = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "evidence_confidence": 0.0,
                "evidence_degraded_sources": 3,
            },
            cfg,
        )
        self.assertEqual(k.mode, "RED")
        self.assertTrue(k.kill_switch_trip)

    def test_contradiction_weight_escalates_faster_than_plain_evidence_drop(self):
        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=300.0,
            BELIEF_SYMBOL_WEIGHT=0.0,
            BELIEF_STREAK_WEIGHT=0.0,
            BELIEF_EVIDENCE_WEIGHT=0.3,
            BELIEF_EVIDENCE_SOURCE_WEIGHT=0.05,
            BELIEF_EVIDENCE_CONTRADICTION_WEIGHT=1.0,
            BELIEF_EVIDENCE_CONTRADICTION_STREAK_WEIGHT=0.2,
            BELIEF_YELLOW_SCORE=0.7,
            BELIEF_ORANGE_SCORE=1.2,
            BELIEF_RED_SCORE=3.0,
            BELIEF_YELLOW_GROWTH=99.0,
            BELIEF_ORANGE_GROWTH=99.0,
            BELIEF_RED_GROWTH=99.0,
            BELIEF_MODE_PERSIST_SEC=0.0,
        )
        clock = _Clock(10.0)
        ctl = BeliefController(clock=clock.now)

        # Plain evidence degradation without contradiction pressure.
        k_plain = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "evidence_confidence": 0.55,
                "evidence_degraded_sources": 1,
                "evidence_contradiction_score": 0.0,
                "evidence_contradiction_streak": 0,
            },
            cfg,
        )

        clock.tick(1.0)
        # Same base confidence but with contradiction pressure should tighten posture more.
        k_contrad = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "evidence_confidence": 0.55,
                "evidence_degraded_sources": 1,
                "evidence_contradiction_score": 0.9,
                "evidence_contradiction_streak": 2,
            },
            cfg,
        )
        self.assertLessEqual(float(k_contrad.max_notional_usdt), float(k_plain.max_notional_usdt))
        self.assertGreaterEqual(float(k_contrad.min_entry_conf), float(k_plain.min_entry_conf))
        self.assertIn(str(k_contrad.mode), ("ORANGE", "RED"))

    def test_runtime_gate_degraded_freezes_entries_but_keeps_exit_invariant(self):
        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=300.0,
            BELIEF_YELLOW_SCORE=9.0,
            BELIEF_ORANGE_SCORE=10.0,
            BELIEF_RED_SCORE=11.0,
            BELIEF_YELLOW_GROWTH=9.0,
            BELIEF_ORANGE_GROWTH=9.0,
            BELIEF_RED_GROWTH=9.0,
            FIXED_NOTIONAL_USDT=100.0,
            LEVERAGE=20,
            ENTRY_MIN_CONFIDENCE=0.2,
            BELIEF_RUNTIME_GATE_RECOVER_SEC=30.0,
        )
        clock = _Clock(10.0)
        ctl = BeliefController(clock=clock.now)
        k = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "runtime_gate_degraded": True,
                "runtime_gate_reason": "mismatch>0",
            },
            cfg,
        )
        self.assertFalse(k.allow_entries)
        self.assertEqual(k.max_notional_usdt, 0.0)
        self.assertIn("runtime_gate_degraded", k.reason)
        self.assertTrue(bool(getattr(k, "runtime_gate_degraded", False)))
        self.assertEqual(str(getattr(k, "runtime_gate_reason", "")), "mismatch>0")
        self.assertTrue(ctl.allows_intent("exit"))

    def test_protection_gap_ttl_breach_freezes_entries(self):
        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=300.0,
            BELIEF_YELLOW_SCORE=9.0,
            BELIEF_ORANGE_SCORE=10.0,
            BELIEF_RED_SCORE=11.0,
            BELIEF_YELLOW_GROWTH=9.0,
            BELIEF_ORANGE_GROWTH=9.0,
            BELIEF_RED_GROWTH=9.0,
            FIXED_NOTIONAL_USDT=100.0,
            LEVERAGE=20,
            ENTRY_MIN_CONFIDENCE=0.2,
            BELIEF_PROTECTION_GAP_TRIP_SEC=60.0,
        )
        clock = _Clock(10.0)
        ctl = BeliefController(clock=clock.now)
        k = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "protection_coverage_gap_seconds": 75.0,
                "protection_coverage_gap_symbols": 1,
                "protection_coverage_ttl_breaches": 1,
            },
            cfg,
        )
        self.assertFalse(k.allow_entries)
        self.assertEqual(float(k.max_notional_usdt), 0.0)
        self.assertIn("protection_gap_degraded", str(k.reason))
        self.assertTrue(ctl.allows_intent("exit"))

    def test_runtime_gate_recovery_warmup_applies_tighter_posture(self):
        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=300.0,
            BELIEF_YELLOW_SCORE=9.0,
            BELIEF_ORANGE_SCORE=10.0,
            BELIEF_RED_SCORE=11.0,
            BELIEF_YELLOW_GROWTH=9.0,
            BELIEF_ORANGE_GROWTH=9.0,
            BELIEF_RED_GROWTH=9.0,
            FIXED_NOTIONAL_USDT=100.0,
            LEVERAGE=20,
            ENTRY_MIN_CONFIDENCE=0.2,
            BELIEF_RUNTIME_GATE_RECOVER_SEC=60.0,
            BELIEF_RUNTIME_GATE_WARMUP_NOTIONAL_SCALE=0.5,
            BELIEF_RUNTIME_GATE_WARMUP_LEVERAGE_SCALE=0.5,
        )
        clock = _Clock(10.0)
        ctl = BeliefController(clock=clock.now)
        ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "runtime_gate_degraded": True,
            },
            cfg,
        )
        clock.tick(1.0)
        k = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "runtime_gate_degraded": False,
            },
            cfg,
        )
        self.assertTrue(k.allow_entries)
        self.assertIn("runtime_gate_warmup", k.reason)
        self.assertLessEqual(k.max_notional_usdt, 50.0)
        self.assertLessEqual(k.max_leverage, 10)

    def test_reconcile_first_spike_freezes_entries_without_runtime_gate(self):
        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=300.0,
            BELIEF_YELLOW_SCORE=9.0,
            BELIEF_ORANGE_SCORE=10.0,
            BELIEF_RED_SCORE=11.0,
            BELIEF_YELLOW_GROWTH=9.0,
            BELIEF_ORANGE_GROWTH=9.0,
            BELIEF_RED_GROWTH=9.0,
            FIXED_NOTIONAL_USDT=100.0,
            LEVERAGE=20,
            ENTRY_MIN_CONFIDENCE=0.2,
            BELIEF_RECONCILE_FIRST_GATE_COUNT_THRESHOLD=2,
            BELIEF_RECONCILE_FIRST_GATE_SEVERITY_THRESHOLD=0.85,
            BELIEF_RECONCILE_FIRST_GATE_STREAK_THRESHOLD=2,
        )
        clock = _Clock(10.0)
        ctl = BeliefController(clock=clock.now)
        k = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "runtime_gate_degraded": False,
                "reconcile_first_gate_count": 2,
                "reconcile_first_gate_max_severity": 0.9,
                "reconcile_first_gate_max_streak": 2,
                "reconcile_first_gate_last_reason": "runtime_gate_reconcile_first",
            },
            cfg,
        )
        self.assertFalse(k.allow_entries)
        self.assertEqual(k.max_notional_usdt, 0.0)
        self.assertTrue(bool(getattr(k, "reconcile_first_gate_degraded", False)))
        self.assertEqual(str(getattr(k, "recovery_stage", "")), "RECONCILE_FIRST_GATE_DEGRADED")

    def test_reconcile_first_spike_recovery_uses_runtime_warmup(self):
        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=300.0,
            BELIEF_YELLOW_SCORE=9.0,
            BELIEF_ORANGE_SCORE=10.0,
            BELIEF_RED_SCORE=11.0,
            BELIEF_YELLOW_GROWTH=9.0,
            BELIEF_ORANGE_GROWTH=9.0,
            BELIEF_RED_GROWTH=9.0,
            FIXED_NOTIONAL_USDT=100.0,
            LEVERAGE=20,
            ENTRY_MIN_CONFIDENCE=0.2,
            BELIEF_RUNTIME_GATE_RECOVER_SEC=60.0,
            BELIEF_RUNTIME_GATE_WARMUP_NOTIONAL_SCALE=0.5,
            BELIEF_RECONCILE_FIRST_GATE_COUNT_THRESHOLD=2,
            BELIEF_RECONCILE_FIRST_GATE_SEVERITY_THRESHOLD=0.85,
            BELIEF_RECONCILE_FIRST_GATE_STREAK_THRESHOLD=2,
        )
        clock = _Clock(10.0)
        ctl = BeliefController(clock=clock.now)
        ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "reconcile_first_gate_count": 2,
                "reconcile_first_gate_max_severity": 0.9,
                "reconcile_first_gate_max_streak": 2,
            },
            cfg,
        )
        clock.tick(1.0)
        k = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "reconcile_first_gate_count": 0,
                "reconcile_first_gate_max_severity": 0.0,
                "reconcile_first_gate_max_streak": 0,
            },
            cfg,
        )
        self.assertTrue(k.allow_entries)
        self.assertIn("runtime_gate_warmup", str(k.reason))
        self.assertEqual(str(getattr(k, "recovery_stage", "")), "RUNTIME_GATE_WARMUP")

    def test_per_symbol_budget_output(self):
        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=100.0,
            BELIEF_SYMBOL_WEIGHT=0.0,
            BELIEF_STREAK_WEIGHT=0.0,
            BELIEF_YELLOW_SCORE=99.0,
            BELIEF_ORANGE_SCORE=199.0,
            BELIEF_RED_SCORE=299.0,
            BELIEF_YELLOW_GROWTH=99.0,
            BELIEF_ORANGE_GROWTH=199.0,
            BELIEF_RED_GROWTH=299.0,
            BELIEF_RISK_SLOPE=0.5,
            BELIEF_PER_SYMBOL_WEIGHT=1.0,
            FIXED_NOTIONAL_USDT=100.0,
            LEVERAGE=20,
            ENTRY_MIN_CONFIDENCE=0.2,
            BELIEF_MIN_CONF_SPREAD=0.5,
        )
        clock = _Clock(10.0)
        ctl = BeliefController(clock=clock.now)
        k = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "symbol_belief_debt_sec": {"BTCUSDT": 140.0, "ETHUSDT": 10.0},
            },
            cfg,
        )
        per = dict(getattr(k, "per_symbol", {}) or {})
        self.assertIn("BTCUSDT", per)
        self.assertIn("ETHUSDT", per)
        btc = per["BTCUSDT"]
        eth = per["ETHUSDT"]
        self.assertLessEqual(float(btc.get("max_notional_usdt") or 0.0), float(eth.get("max_notional_usdt") or 0.0))
        self.assertGreaterEqual(float(btc.get("min_entry_conf") or 0.0), float(eth.get("min_entry_conf") or 0.0))

    def test_runtime_gate_soft_scaling_tightens_even_without_hard_degraded(self):
        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=300.0,
            BELIEF_SYMBOL_WEIGHT=0.0,
            BELIEF_STREAK_WEIGHT=0.0,
            BELIEF_YELLOW_SCORE=99.0,
            BELIEF_ORANGE_SCORE=199.0,
            BELIEF_RED_SCORE=299.0,
            BELIEF_YELLOW_GROWTH=99.0,
            BELIEF_ORANGE_GROWTH=199.0,
            BELIEF_RED_GROWTH=299.0,
            FIXED_NOTIONAL_USDT=100.0,
            LEVERAGE=20,
            ENTRY_MIN_CONFIDENCE=0.2,
            BELIEF_RUNTIME_GATE_WEIGHT=0.5,
            BELIEF_RUNTIME_GATE_SCORE_WEIGHT=1.0,
        )
        clock = _Clock(10.0)
        ctl = BeliefController(clock=clock.now)
        base = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "runtime_gate_degraded": False,
            },
            cfg,
        )
        clock.tick(1.0)
        pressured = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "runtime_gate_degraded": False,
                "runtime_gate_replay_mismatch_count": 1,
                "runtime_gate_invalid_transition_count": 0,
                "runtime_gate_journal_coverage_ratio": 0.95,
                "runtime_gate_degrade_score": 0.5,
            },
            cfg,
        )
        self.assertTrue(pressured.allow_entries)
        self.assertLessEqual(float(pressured.max_notional_usdt), float(base.max_notional_usdt))
        self.assertLessEqual(int(pressured.max_leverage), int(base.max_leverage))
        self.assertGreaterEqual(float(pressured.min_entry_conf), float(base.min_entry_conf))
        self.assertIn("runtime_gate_soft_scale", str(pressured.reason))

    def test_runtime_gate_category_pressure_tightens_posture(self):
        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=300.0,
            BELIEF_SYMBOL_WEIGHT=0.0,
            BELIEF_STREAK_WEIGHT=0.0,
            BELIEF_YELLOW_SCORE=99.0,
            BELIEF_ORANGE_SCORE=199.0,
            BELIEF_RED_SCORE=299.0,
            BELIEF_YELLOW_GROWTH=99.0,
            BELIEF_ORANGE_GROWTH=199.0,
            BELIEF_RED_GROWTH=299.0,
            FIXED_NOTIONAL_USDT=100.0,
            LEVERAGE=20,
            ENTRY_MIN_CONFIDENCE=0.2,
            BELIEF_RUNTIME_GATE_WEIGHT=0.2,
            BELIEF_RUNTIME_GATE_CAT_LEDGER_WEIGHT=1.0,
            BELIEF_RUNTIME_GATE_CAT_TRANSITION_WEIGHT=1.0,
        )
        clock = _Clock(10.0)
        ctl = BeliefController(clock=clock.now)
        base = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
            },
            cfg,
        )
        clock.tick(1.0)
        pressured = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "runtime_gate_degraded": False,
                "runtime_gate_cat_ledger": 1,
                "runtime_gate_cat_transition": 1,
            },
            cfg,
        )
        self.assertLessEqual(float(pressured.max_notional_usdt), float(base.max_notional_usdt))
        self.assertLessEqual(int(pressured.max_leverage), int(base.max_leverage))

    def test_runtime_gate_critical_categories_tighten_more_than_unknown(self):
        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=300.0,
            BELIEF_SYMBOL_WEIGHT=0.0,
            BELIEF_STREAK_WEIGHT=0.0,
            BELIEF_YELLOW_SCORE=99.0,
            BELIEF_ORANGE_SCORE=199.0,
            BELIEF_RED_SCORE=299.0,
            BELIEF_YELLOW_GROWTH=99.0,
            BELIEF_ORANGE_GROWTH=199.0,
            BELIEF_RED_GROWTH=299.0,
            FIXED_NOTIONAL_USDT=100.0,
            LEVERAGE=20,
            ENTRY_MIN_CONFIDENCE=0.2,
            BELIEF_RUNTIME_GATE_WEIGHT=0.2,
            BELIEF_RUNTIME_GATE_CRITICAL_WEIGHT=0.5,
            BELIEF_RUNTIME_GATE_CAT_POSITION_WEIGHT=1.0,
            BELIEF_RUNTIME_GATE_CAT_ORPHAN_WEIGHT=1.0,
            BELIEF_RUNTIME_GATE_CAT_REPLACE_RACE_WEIGHT=1.0,
            BELIEF_RUNTIME_GATE_CAT_CONTRADICTION_WEIGHT=1.0,
            BELIEF_RUNTIME_GATE_CAT_UNKNOWN_WEIGHT=1.0,
        )
        clock = _Clock(10.0)
        ctl = BeliefController(clock=clock.now)

        unknown_only = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "runtime_gate_degraded": False,
                "runtime_gate_cat_unknown": 2,
            },
            cfg,
        )
        clock.tick(1.0)
        critical_mix = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "runtime_gate_degraded": False,
                "runtime_gate_cat_position": 1,
                "runtime_gate_cat_orphan": 1,
            },
            cfg,
        )
        self.assertLessEqual(float(critical_mix.max_notional_usdt), float(unknown_only.max_notional_usdt))
        self.assertGreaterEqual(float(critical_mix.min_entry_conf), float(unknown_only.min_entry_conf))

    def test_runtime_gate_critical_trip_freezes_then_warms_up(self):
        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=300.0,
            BELIEF_SYMBOL_WEIGHT=0.0,
            BELIEF_STREAK_WEIGHT=0.0,
            BELIEF_YELLOW_SCORE=99.0,
            BELIEF_ORANGE_SCORE=199.0,
            BELIEF_RED_SCORE=299.0,
            BELIEF_YELLOW_GROWTH=99.0,
            BELIEF_ORANGE_GROWTH=199.0,
            BELIEF_RED_GROWTH=299.0,
            FIXED_NOTIONAL_USDT=100.0,
            LEVERAGE=20,
            ENTRY_MIN_CONFIDENCE=0.2,
            BELIEF_RUNTIME_GATE_RECOVER_SEC=60.0,
            BELIEF_RUNTIME_GATE_CRITICAL_TRIP_THRESHOLD=2.0,
            BELIEF_RUNTIME_GATE_CRITICAL_CLEAR_THRESHOLD=1.0,
            BELIEF_RUNTIME_GATE_WARMUP_NOTIONAL_SCALE=0.5,
            BELIEF_RUNTIME_GATE_WARMUP_LEVERAGE_SCALE=0.5,
        )
        clock = _Clock(10.0)
        ctl = BeliefController(clock=clock.now)
        tripped = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "runtime_gate_degraded": False,
                "runtime_gate_cat_position": 1,
                "runtime_gate_cat_replace_race": 1,
            },
            cfg,
        )
        self.assertFalse(tripped.allow_entries)
        self.assertEqual(float(tripped.max_notional_usdt), 0.0)
        self.assertTrue(bool(getattr(tripped, "runtime_gate_degraded", False)))
        self.assertIn("runtime_gate_degraded", str(tripped.reason))

        clock.tick(1.0)
        warm = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "runtime_gate_degraded": False,
                "runtime_gate_cat_position": 0,
                "runtime_gate_cat_replace_race": 0,
            },
            cfg,
        )
        self.assertTrue(warm.allow_entries)
        self.assertIn("runtime_gate_warmup", str(warm.reason))
        self.assertLessEqual(float(warm.max_notional_usdt), 50.0)

    def test_post_red_recovery_ladder_applies_warmup_caps(self):
        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=100.0,
            BELIEF_SYMBOL_WEIGHT=0.0,
            BELIEF_STREAK_WEIGHT=0.0,
            BELIEF_YELLOW_SCORE=0.8,
            BELIEF_ORANGE_SCORE=1.2,
            BELIEF_RED_SCORE=2.0,
            BELIEF_YELLOW_GROWTH=9.0,
            BELIEF_ORANGE_GROWTH=9.0,
            BELIEF_RED_GROWTH=9.0,
            BELIEF_MODE_PERSIST_SEC=0.0,
            BELIEF_MODE_RECOVER_SEC=2.0,
            BELIEF_DOWN_HYST=0.8,
            BELIEF_POST_RED_WARMUP_SEC=10.0,
            BELIEF_POST_RED_WARMUP_NOTIONAL_SCALE=0.5,
            BELIEF_POST_RED_WARMUP_LEVERAGE_SCALE=0.5,
            FIXED_NOTIONAL_USDT=100.0,
            LEVERAGE=20,
            ENTRY_MIN_CONFIDENCE=0.2,
        )
        clock = _Clock(10.0)
        ctl = BeliefController(clock=clock.now)
        k_red = ctl.update(
            {"belief_debt_sec": 300.0, "belief_debt_symbols": 0, "mismatch_streak": 0},
            cfg,
        )
        self.assertEqual(k_red.mode, "RED")
        self.assertFalse(k_red.allow_entries)
        self.assertIn("RED_LOCK", str(getattr(k_red, "recovery_stage", "")))

        clock.tick(3.0)
        ctl.update(
            {"belief_debt_sec": 0.0, "belief_debt_symbols": 0, "mismatch_streak": 0},
            cfg,
        )
        clock.tick(3.0)
        k_orange = ctl.update(
            {"belief_debt_sec": 0.0, "belief_debt_symbols": 0, "mismatch_streak": 0},
            cfg,
        )
        self.assertIn(k_orange.mode, ("ORANGE", "YELLOW"))
        self.assertIn("post_red_warmup", str(k_orange.reason))
        self.assertEqual(str(getattr(k_orange, "recovery_stage", "")), "POST_RED_WARMUP")
        self.assertGreater(float(getattr(k_orange, "next_unlock_sec", 0.0) or 0.0), 0.0)

    def test_fixture_runtime_gate_snapshot_freeze_then_warmup(self):
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "belief_runtime_gate_snapshot.json"
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        degraded_state = dict(payload.get("degraded_state") or {})
        recovered_state = dict(payload.get("recovered_state") or {})

        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=300.0,
            BELIEF_SYMBOL_WEIGHT=0.0,
            BELIEF_STREAK_WEIGHT=0.0,
            BELIEF_YELLOW_SCORE=99.0,
            BELIEF_ORANGE_SCORE=199.0,
            BELIEF_RED_SCORE=299.0,
            BELIEF_YELLOW_GROWTH=99.0,
            BELIEF_ORANGE_GROWTH=199.0,
            BELIEF_RED_GROWTH=299.0,
            FIXED_NOTIONAL_USDT=100.0,
            LEVERAGE=20,
            ENTRY_MIN_CONFIDENCE=0.2,
            BELIEF_RUNTIME_GATE_RECOVER_SEC=120.0,
            BELIEF_RUNTIME_GATE_WARMUP_NOTIONAL_SCALE=0.5,
            BELIEF_RUNTIME_GATE_WARMUP_LEVERAGE_SCALE=0.6,
        )
        clock = _Clock(100.0)
        ctl = BeliefController(clock=clock.now)

        k1 = ctl.update(degraded_state, cfg)
        self.assertFalse(k1.allow_entries)
        self.assertEqual(float(k1.max_notional_usdt), 0.0)
        self.assertTrue(bool(getattr(k1, "runtime_gate_degraded", False)))
        self.assertIn("runtime_gate_degraded", str(k1.reason))

        clock.tick(1.0)
        k2 = ctl.update(recovered_state, cfg)
        self.assertTrue(k2.allow_entries)
        self.assertIn("runtime_gate_warmup", str(k2.reason))
        self.assertEqual(str(getattr(k2, "recovery_stage", "")), "RUNTIME_GATE_WARMUP")
        self.assertLessEqual(float(k2.max_notional_usdt), 50.0)

    def test_fixture_runtime_gate_critical_snapshot_freeze_then_warmup(self):
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "belief_runtime_gate_critical_snapshot.json"
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        critical_trip_state = dict(payload.get("critical_trip_state") or {})
        critical_clear_state = dict(payload.get("critical_clear_state") or {})

        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=300.0,
            BELIEF_SYMBOL_WEIGHT=0.0,
            BELIEF_STREAK_WEIGHT=0.0,
            BELIEF_YELLOW_SCORE=99.0,
            BELIEF_ORANGE_SCORE=199.0,
            BELIEF_RED_SCORE=299.0,
            BELIEF_YELLOW_GROWTH=99.0,
            BELIEF_ORANGE_GROWTH=199.0,
            BELIEF_RED_GROWTH=299.0,
            FIXED_NOTIONAL_USDT=100.0,
            LEVERAGE=20,
            ENTRY_MIN_CONFIDENCE=0.2,
            BELIEF_RUNTIME_GATE_RECOVER_SEC=120.0,
            BELIEF_RUNTIME_GATE_CRITICAL_TRIP_THRESHOLD=2.0,
            BELIEF_RUNTIME_GATE_CRITICAL_CLEAR_THRESHOLD=1.0,
            BELIEF_RUNTIME_GATE_WARMUP_NOTIONAL_SCALE=0.5,
            BELIEF_RUNTIME_GATE_WARMUP_LEVERAGE_SCALE=0.6,
        )
        clock = _Clock(200.0)
        ctl = BeliefController(clock=clock.now)

        k1 = ctl.update(critical_trip_state, cfg)
        self.assertFalse(k1.allow_entries)
        self.assertEqual(float(k1.max_notional_usdt), 0.0)
        self.assertTrue(bool(getattr(k1, "runtime_gate_degraded", False)))
        self.assertIn("runtime_gate_critical", str(getattr(k1, "runtime_gate_reason", "")))

        clock.tick(1.0)
        k2 = ctl.update(critical_clear_state, cfg)
        self.assertTrue(k2.allow_entries)
        self.assertIn("runtime_gate_warmup", str(k2.reason))
        self.assertEqual(str(getattr(k2, "recovery_stage", "")), "RUNTIME_GATE_WARMUP")
        self.assertLessEqual(float(k2.max_notional_usdt), 50.0)

    def test_fixture_observed_runtime_gate_spike_freeze_then_warmup(self):
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "belief_runtime_gate_observed_spike.json"
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        spike_state = dict(payload.get("observed_spike_state") or {})
        clear_state = dict(payload.get("observed_clear_state") or {})

        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=300.0,
            BELIEF_SYMBOL_WEIGHT=0.0,
            BELIEF_STREAK_WEIGHT=0.0,
            BELIEF_YELLOW_SCORE=99.0,
            BELIEF_ORANGE_SCORE=199.0,
            BELIEF_RED_SCORE=299.0,
            BELIEF_YELLOW_GROWTH=99.0,
            BELIEF_ORANGE_GROWTH=199.0,
            BELIEF_RED_GROWTH=299.0,
            FIXED_NOTIONAL_USDT=100.0,
            LEVERAGE=20,
            ENTRY_MIN_CONFIDENCE=0.2,
            BELIEF_RUNTIME_GATE_RECOVER_SEC=120.0,
            BELIEF_RUNTIME_GATE_CRITICAL_TRIP_THRESHOLD=4.0,
            BELIEF_RUNTIME_GATE_CRITICAL_CLEAR_THRESHOLD=2.0,
            BELIEF_RUNTIME_GATE_WARMUP_NOTIONAL_SCALE=0.5,
            BELIEF_RUNTIME_GATE_WARMUP_LEVERAGE_SCALE=0.6,
        )
        clock = _Clock(300.0)
        ctl = BeliefController(clock=clock.now)

        k1 = ctl.update(spike_state, cfg)
        self.assertFalse(k1.allow_entries)
        self.assertEqual(float(k1.max_notional_usdt), 0.0)
        self.assertTrue(bool(getattr(k1, "runtime_gate_degraded", False)))
        self.assertIn("runtime_gate_degraded", str(k1.reason))

        clock.tick(1.0)
        k2 = ctl.update(clear_state, cfg)
        self.assertTrue(k2.allow_entries)
        self.assertIn("runtime_gate_warmup", str(k2.reason))
        self.assertEqual(str(getattr(k2, "recovery_stage", "")), "RUNTIME_GATE_WARMUP")
        self.assertLessEqual(float(k2.max_notional_usdt), 50.0)

    def test_transition_trace_includes_dominant_cause_tags(self):
        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=100.0,
            BELIEF_SYMBOL_WEIGHT=0.0,
            BELIEF_STREAK_WEIGHT=0.0,
            BELIEF_YELLOW_SCORE=0.2,
            BELIEF_ORANGE_SCORE=0.4,
            BELIEF_RED_SCORE=0.6,
            BELIEF_YELLOW_GROWTH=99.0,
            BELIEF_ORANGE_GROWTH=99.0,
            BELIEF_RED_GROWTH=99.0,
            BELIEF_MODE_PERSIST_SEC=0.0,
            FIXED_NOTIONAL_USDT=100.0,
            LEVERAGE=20,
            ENTRY_MIN_CONFIDENCE=0.2,
            BELIEF_RUNTIME_GATE_WEIGHT=0.2,
            BELIEF_RUNTIME_GATE_CAT_POSITION_WEIGHT=1.0,
            BELIEF_RUNTIME_GATE_CAT_REPLACE_RACE_WEIGHT=1.0,
            BELIEF_RUNTIME_GATE_CRITICAL_WEIGHT=0.5,
        )
        clock = _Clock(1.0)
        ctl = BeliefController(clock=clock.now)
        ctl.update(
            {
                "belief_debt_sec": 100.0,
                "runtime_gate_cat_position": 2,
                "runtime_gate_cat_replace_race": 1,
            },
            cfg,
        )
        trace = ctl.explain().to_dict()
        self.assertIn("mode_transition", str(trace.get("cause_tags", "")))
        self.assertIn("position", str(trace.get("dominant_contributors", "")))

    def test_unlock_conditions_snapshot_contains_tick_coverage_and_contradiction(self):
        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=300.0,
            BELIEF_SYMBOL_WEIGHT=0.0,
            BELIEF_STREAK_WEIGHT=0.0,
            BELIEF_YELLOW_SCORE=99.0,
            BELIEF_ORANGE_SCORE=199.0,
            BELIEF_RED_SCORE=299.0,
            BELIEF_YELLOW_GROWTH=99.0,
            BELIEF_ORANGE_GROWTH=99.0,
            BELIEF_RED_GROWTH=99.0,
            BELIEF_UNLOCK_HEALTHY_TICKS_REQUIRED=3,
            BELIEF_UNLOCK_MIN_JOURNAL_COVERAGE=0.95,
            BELIEF_UNLOCK_CONTRADICTION_CLEAR_SEC=60,
            FIXED_NOTIONAL_USDT=100.0,
            LEVERAGE=20,
            ENTRY_MIN_CONFIDENCE=0.2,
        )
        clock = _Clock(10.0)
        ctl = BeliefController(clock=clock.now)
        k = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "runtime_gate_journal_coverage_ratio": 0.80,
                "runtime_gate_cat_contradiction": 1,
                "evidence_contradiction_score": 1.0,
            },
            cfg,
        )
        msg = str(getattr(k, "unlock_conditions", ""))
        self.assertIn("healthy_ticks", msg)
        self.assertIn("journal_coverage", msg)
        self.assertIn("contradiction_clear", msg)

    def test_contradiction_burn_rate_escalates_faster_than_staleness(self):
        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=300.0,
            BELIEF_SYMBOL_WEIGHT=0.0,
            BELIEF_STREAK_WEIGHT=0.0,
            BELIEF_YELLOW_SCORE=0.6,
            BELIEF_ORANGE_SCORE=1.0,
            BELIEF_RED_SCORE=2.5,
            BELIEF_YELLOW_GROWTH=99.0,
            BELIEF_ORANGE_GROWTH=99.0,
            BELIEF_RED_GROWTH=99.0,
            BELIEF_MODE_PERSIST_SEC=0.0,
            FIXED_NOTIONAL_USDT=100.0,
            LEVERAGE=20,
            ENTRY_MIN_CONFIDENCE=0.2,
            BELIEF_EVIDENCE_WEIGHT=0.2,
            BELIEF_EVIDENCE_SOURCE_WEIGHT=0.05,
            BELIEF_EVIDENCE_CONTRADICTION_WEIGHT=0.4,
            BELIEF_EVIDENCE_CONTRADICTION_STREAK_WEIGHT=0.05,
            BELIEF_EVIDENCE_CONTRADICTION_BURN_WEIGHT=0.4,
            BELIEF_EVIDENCE_CONTRADICTION_BURN_REF=3.0,
        )
        clock = _Clock(1.0)
        ctl = BeliefController(clock=clock.now)

        stale_only = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "evidence_confidence": 0.75,
                "evidence_degraded_sources": 1,
                "evidence_contradiction_score": 0.0,
                "evidence_contradiction_streak": 0,
                "evidence_contradiction_burn_rate": 0.0,
            },
            cfg,
        )

        clock.tick(1.0)
        contrad_burn = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "evidence_confidence": 0.75,
                "evidence_degraded_sources": 1,
                "evidence_contradiction_score": 0.7,
                "evidence_contradiction_streak": 2,
                "evidence_contradiction_burn_rate": 4.0,
            },
            cfg,
        )
        self.assertLessEqual(float(contrad_burn.max_notional_usdt), float(stale_only.max_notional_usdt))
        self.assertGreaterEqual(float(contrad_burn.min_entry_conf), float(stale_only.min_entry_conf))
        self.assertIn(str(contrad_burn.mode), ("YELLOW", "ORANGE", "RED"))


if __name__ == "__main__":
    unittest.main()
