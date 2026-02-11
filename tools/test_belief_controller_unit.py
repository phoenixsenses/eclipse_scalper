#!/usr/bin/env python3
from __future__ import annotations

import types
import unittest
import sys
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


if __name__ == "__main__":
    unittest.main()
