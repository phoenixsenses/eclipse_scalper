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


class ReconcileCorrelationStateTests(unittest.IsolatedAsyncioTestCase):
    def _bot(self) -> SimpleNamespace:
        cfg = SimpleNamespace(
            RECONCILE_FULL_SCAN_ORPHANS=True,
            RUNTIME_RELIABILITY_COUPLING=True,
        )
        state = SimpleNamespace(
            positions={},
            run_context={},
            kill_metrics={},
            reconcile_metrics={},
        )
        return SimpleNamespace(cfg=cfg, state=state, active_symbols=set(), ex=None)

    async def test_reconcile_emits_correlation_state_and_updates_metrics(self):
        bot = self._bot()
        events: list[tuple[str, dict]] = []
        patched = {}

        async def _fake_fetch_positions_best_effort(_bot, _symbols):
            return [], True

        async def _fake_tel_emit(_bot, event, data=None, level="info"):
            events.append((str(event), dict(data or {})))

        def _fake_corr(_bot, _metrics, _cfg):
            return {
                "corr_pressure": 0.83,
                "corr_regime": "STRESS",
                "corr_confidence": 0.62,
                "corr_roll": 0.8,
                "corr_downside": 0.84,
                "corr_tail_coupling": 0.88,
                "corr_uncertainty_uplift": 0.12,
                "corr_group_drift_debt": 0.2,
                "corr_hidden_exposure_risk": 0.1,
                "corr_worst_group": "MAJOR",
                "corr_reason_tags": "tail_coupling,belief_uplift",
            }

        for name, value in {
            "_fetch_positions_best_effort": _fake_fetch_positions_best_effort,
            "_tel_emit": _fake_tel_emit,
            "_compute_correlation_risk": _fake_corr,
            "_ensure_belief_controller": lambda _bot: None,
        }.items():
            patched[name] = getattr(reconcile, name)
            setattr(reconcile, name, value)

        try:
            await reconcile.reconcile_tick(bot)
        finally:
            for name, prev in patched.items():
                setattr(reconcile, name, prev)

        rm = reconcile._ensure_reconcile_metrics(bot)
        self.assertAlmostEqual(float(rm.get("corr_pressure", 0.0)), 0.83, places=6)
        self.assertEqual(str(rm.get("corr_regime", "")), "STRESS")
        self.assertEqual(str(rm.get("corr_worst_group", "")), "MAJOR")
        evt = [e for e in events if e[0] == "execution.correlation_state"]
        self.assertTrue(evt, "execution.correlation_state event was not emitted")
        payload = evt[-1][1]
        self.assertAlmostEqual(float(payload.get("corr_pressure", 0.0)), 0.83, places=6)
        self.assertEqual(str(payload.get("corr_regime", "")), "STRESS")
        self.assertIn("corr_reason_tags", payload)


if __name__ == "__main__":
    unittest.main()
