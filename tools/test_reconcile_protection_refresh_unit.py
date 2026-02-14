#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution import reconcile  # noqa: E402
from eclipse_scalper.execution.belief_controller import BeliefController  # noqa: E402


class ReconcileProtectionRefreshTests(unittest.IsolatedAsyncioTestCase):
    def _mk_bot(self, *, ttl_breached: bool) -> SimpleNamespace:
        cfg = SimpleNamespace(
            GUARDIAN_ENSURE_STOP=True,
            GUARDIAN_RESPECT_KILL_SWITCH=False,
            RECONCILE_STOP_THROTTLE_SEC=0.0,
            RECONCILE_REPAIR_COOLDOWN_SEC=0.0,
            RECONCILE_STOP_MIN_COVERAGE_RATIO=0.98,
            STOP_ATR_MULT=2.0,
            GUARDIAN_STOP_BUFFER_ATR_MULT=0.0,
            GUARDIAN_STOP_FALLBACK_PCT=0.0035,
            RECONCILE_STOP_REFRESH_FORCE_COVERAGE_RATIO=0.80,
            RECONCILE_STOP_REFRESH_MIN_DELTA_RATIO=0.0,
            RECONCILE_STOP_REFRESH_MIN_DELTA_ABS=0.0,
            RECONCILE_STOP_REFRESH_MAX_INTERVAL_SEC=3600.0,
            RECONCILE_STOP_REFRESH_BUDGET_WINDOW_SEC=60.0,
            RECONCILE_STOP_REFRESH_MAX_PER_WINDOW=1,
            RECONCILE_STOP_REPLACE_RETRIES=1,
            RECONCILE_ALERT_COOLDOWN_SEC=1.0,
        )
        run_context = {
            "protection_refresh": {
                "BTCUSDT": {"qty": 1.0, "ts": 100.0, "refresh_events": [130.0]},
            },
            "protection_gap_state": {
                "BTCUSDT": {"ttl_breached": bool(ttl_breached)},
            },
        }
        state = SimpleNamespace(run_context=run_context)
        bot = SimpleNamespace(cfg=cfg, state=state, ex=None)
        return bot

    async def test_stop_refresh_budget_ttl_breach_forces_refresh(self):
        bot = self._mk_bot(ttl_breached=True)
        pos = SimpleNamespace(side="long", entry_price=100.0, atr=1.0)
        forced_args: list[bool] = []
        patched = {}

        async def _fake_fetch_open_orders_best_effort(_bot, _sym_raw):
            return []

        def _fake_assess_stop_coverage(_orders, **_kwargs):
            return {
                "covered": False,
                "needs_refresh": True,
                "order_id": "old-stop",
                "existing_qty": 1.0,
                "coverage_ratio": 1.0,
                "reason": "qty_under_covered",
            }

        async def _fake_cancel_replace_order(*_args, **_kwargs):
            return {"id": "new-stop"}

        def _fake_should_allow_refresh_budget(_state, **kwargs):
            forced_args.append(bool(kwargs.get("force", False)))
            # Simulate exhausted budget that should only pass with force=True.
            return {"allowed": bool(kwargs.get("force", False))}

        for name, value in {
            "_fetch_open_orders_best_effort": _fake_fetch_open_orders_best_effort,
            "_assess_stop_coverage": _fake_assess_stop_coverage,
            "cancel_replace_order": _fake_cancel_replace_order,
            "_should_refresh_protection": lambda **_kwargs: True,
            "_should_allow_refresh_budget": _fake_should_allow_refresh_budget,
            "_alert_ok": lambda *_a, **_k: False,
        }.items():
            patched[name] = getattr(reconcile, name)
            setattr(reconcile, name, value)

        try:
            out = await reconcile._ensure_protective_stop(bot, "BTCUSDT", pos, "long", 1.0)
        finally:
            for name, prev in patched.items():
                setattr(reconcile, name, prev)

        self.assertEqual(out, "restored")
        self.assertTrue(forced_args and forced_args[-1])
        rm = reconcile._ensure_reconcile_metrics(bot)
        self.assertEqual(int(rm.get("protection_refresh_stop_force_override_count", 0)), 1)
        self.assertEqual(int(rm.get("protection_refresh_stop_budget_blocked_count", 0)), 0)

    async def test_stop_refresh_budget_blocks_without_ttl_breach(self):
        bot = self._mk_bot(ttl_breached=False)
        pos = SimpleNamespace(side="long", entry_price=100.0, atr=1.0)
        patched = {}

        async def _fake_fetch_open_orders_best_effort(_bot, _sym_raw):
            return []

        def _fake_assess_stop_coverage(_orders, **_kwargs):
            return {
                "covered": False,
                "needs_refresh": True,
                "order_id": "old-stop",
                "existing_qty": 1.0,
                "coverage_ratio": 1.0,
                "reason": "qty_under_covered",
            }

        for name, value in {
            "_fetch_open_orders_best_effort": _fake_fetch_open_orders_best_effort,
            "_assess_stop_coverage": _fake_assess_stop_coverage,
            "_should_refresh_protection": lambda **_kwargs: True,
            "_should_allow_refresh_budget": lambda _s, **_kwargs: {"allowed": False},
            "_alert_ok": lambda *_a, **_k: False,
        }.items():
            patched[name] = getattr(reconcile, name)
            setattr(reconcile, name, value)

        try:
            out = await reconcile._ensure_protective_stop(bot, "BTCUSDT", pos, "long", 1.0)
        finally:
            for name, prev in patched.items():
                setattr(reconcile, name, prev)

        self.assertEqual(out, "refresh_deferred")
        rm = reconcile._ensure_reconcile_metrics(bot)
        self.assertEqual(int(rm.get("protection_refresh_stop_budget_blocked_count", 0)), 1)
        self.assertEqual(int(rm.get("protection_refresh_stop_force_override_count", 0)), 0)


class ReconcileRuntimeGateSummaryTests(unittest.TestCase):
    def test_position_envelope_widens_on_ambiguity_then_narrows_on_consistent_evidence(self):
        bot = SimpleNamespace(state=SimpleNamespace(run_context={"position_envelopes": {}}))
        state_positions = {"BTCUSDT": SimpleNamespace(size=1.0)}
        ex_map = {"BTCUSDT": [{"symbol": "BTC/USDT:USDT", "contracts": 2.0, "side": "long"}]}

        widened = reconcile._update_position_envelopes(  # type: ignore[attr-defined]
            bot,
            state_positions=state_positions,
            ex_map=ex_map,
            tracked_syms={"BTCUSDT"},
            mismatch_symbols={"BTCUSDT"},
        )
        store = bot.state.run_context.get("position_envelopes", {})
        env = dict(store.get("BTCUSDT") or {})
        self.assertGreaterEqual(float(env.get("position_interval_max", 0.0)), 2.0)
        self.assertAlmostEqual(float(env.get("position_interval_min", 1.0)), 0.0, places=6)
        self.assertIn("UNKNOWN", str(env.get("live_state_set", "")))
        self.assertEqual(int(widened.get("belief_envelope_ambiguous_symbols", 0)), 1)

        # Consistent evidence narrows the envelope.
        state_positions["BTCUSDT"] = SimpleNamespace(size=1.5)
        ex_map["BTCUSDT"] = [{"symbol": "BTC/USDT:USDT", "contracts": 1.5, "side": "long"}]
        narrowed = reconcile._update_position_envelopes(  # type: ignore[attr-defined]
            bot,
            state_positions=state_positions,
            ex_map=ex_map,
            tracked_syms={"BTCUSDT"},
            mismatch_symbols=set(),
        )
        env2 = dict(store.get("BTCUSDT") or {})
        self.assertAlmostEqual(float(env2.get("position_interval_min", 0.0)), 1.5, places=6)
        self.assertAlmostEqual(float(env2.get("position_interval_max", 0.0)), 1.5, places=6)
        self.assertEqual(str(env2.get("live_state_set", "")), "LIVE")
        self.assertEqual(int(narrowed.get("belief_envelope_ambiguous_symbols", 1)), 0)

    def test_reconcile_metrics_include_evidence_coverage_defaults(self):
        bot = SimpleNamespace(state=SimpleNamespace(reconcile_metrics={}))
        rm = reconcile._ensure_reconcile_metrics(bot)
        self.assertIn("evidence_ws_coverage_ratio", rm)
        self.assertIn("evidence_rest_coverage_ratio", rm)
        self.assertIn("evidence_fill_coverage_ratio", rm)
        self.assertIn("evidence_coverage_ratio", rm)

    def test_runtime_gate_cause_summary_includes_peak_hints(self):
        summary = reconcile._runtime_gate_cause_summary(  # type: ignore[attr-defined]
            {
                "runtime_gate_degraded": True,
                "runtime_gate_reason": "position_peak>1,coverage_gap_sec_peak>5.0",
                "runtime_gate_position_mismatch_count": 0,
                "runtime_gate_position_mismatch_count_peak": 2,
                "runtime_gate_protection_coverage_gap_seconds": 0.0,
                "runtime_gate_protection_coverage_gap_seconds_peak": 12.0,
            }
        )
        self.assertIn("position_peak=2 current=0", summary)
        self.assertIn("coverage_gap_peak=12.0s current=0.0s", summary)

    def test_runtime_gate_summary_propagates_into_controller_clamps(self):
        class _Clock:
            def __init__(self, start: float = 0.0):
                self.t = float(start)

            def now(self) -> float:
                return float(self.t)

            def tick(self, sec: float) -> None:
                self.t += float(sec)

        metrics = {
            "runtime_gate_degraded": True,
            "runtime_gate_reason": "position_peak>1,coverage_gap_sec_peak>5.0",
            "runtime_gate_position_mismatch_count": 0,
            "runtime_gate_position_mismatch_count_peak": 2,
            "runtime_gate_protection_coverage_gap_seconds": 0.0,
            "runtime_gate_protection_coverage_gap_seconds_peak": 12.0,
        }
        cause_summary = reconcile._runtime_gate_cause_summary(metrics)  # type: ignore[attr-defined]
        self.assertTrue(cause_summary)

        cfg = SimpleNamespace(
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
            ENTRY_COOLDOWN_SEC=5.0,
            BELIEF_RUNTIME_GATE_POSITION_PEAK_NOTIONAL_SCALE=0.75,
            BELIEF_RUNTIME_GATE_POSITION_PEAK_LEVERAGE_SCALE=0.85,
            BELIEF_RUNTIME_GATE_COVERAGE_GAP_PEAK_MIN_CONF_EXTRA=0.05,
            BELIEF_RUNTIME_GATE_COVERAGE_GAP_PEAK_COOLDOWN_EXTRA_SEC=12.0,
        )
        clock = _Clock(10.0)
        ctl = BeliefController(clock=clock.now)
        base = ctl.update(
            {"belief_debt_sec": 0.0, "belief_debt_symbols": 0, "mismatch_streak": 0},
            cfg,
        )
        clock.tick(1.0)
        clamped = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "runtime_gate_cause_summary": cause_summary,
            },
            cfg,
        )
        self.assertLess(float(clamped.max_notional_usdt), float(base.max_notional_usdt))
        self.assertLessEqual(int(clamped.max_leverage), int(base.max_leverage))
        self.assertGreater(float(clamped.entry_cooldown_seconds), float(base.entry_cooldown_seconds))


if __name__ == "__main__":
    unittest.main()
