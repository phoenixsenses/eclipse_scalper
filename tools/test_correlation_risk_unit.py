#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution import correlation_risk  # noqa: E402


class CorrelationRiskTests(unittest.TestCase):
    def _bot(self):
        returns = {
            "BTCUSDT": [-0.03, -0.02, -0.01, -0.02, -0.03, -0.01],
            "ETHUSDT": [-0.028, -0.018, -0.012, -0.021, -0.029, -0.011],
        }
        state = SimpleNamespace(
            run_context={"corr_returns": returns},
            positions={
                "BTCUSDT": SimpleNamespace(size=1.0),
                "ETHUSDT": SimpleNamespace(size=0.5),
            },
        )
        return SimpleNamespace(
            state=state,
            CORRELATION_GROUPS={"MAJOR": ["BTCUSDT", "ETHUSDT"]},
        )

    def _cfg(self, **kwargs):
        base = dict(
            CORR_STRESS_ENTER=0.75,
            CORR_STRESS_EXIT=0.55,
            CORR_TIGHTEN_ENTER=0.45,
            CORR_TIGHTEN_EXIT=0.30,
            CORR_HYST_UP_SEC=5.0,
            CORR_HYST_DOWN_SEC=8.0,
            CORR_RETURNS_KEY="corr_returns",
        )
        base.update(kwargs)
        return SimpleNamespace(**base)

    def test_uncertainty_uplift_increases_pressure(self):
        bot = self._bot()
        cfg = self._cfg()
        low = correlation_risk.compute_correlation_risk(
            bot,
            {
                "belief_debt_sec": 0.0,
                "runtime_gate_cat_position": 0,
                "runtime_gate_cat_orphan": 0,
                "intent_unknown_count": 0,
                "evidence_contradiction_score": 0.0,
            },
            cfg,
        )
        high = correlation_risk.compute_correlation_risk(
            bot,
            {
                "belief_debt_sec": 600.0,
                "runtime_gate_cat_position": 2,
                "runtime_gate_cat_orphan": 2,
                "intent_unknown_count": 1,
                "evidence_contradiction_score": 0.8,
            },
            cfg,
        )
        self.assertGreater(float(high["corr_uncertainty_uplift"]), float(low["corr_uncertainty_uplift"]))
        self.assertGreater(float(high["corr_pressure"]), float(low["corr_pressure"]))

    def test_hysteresis_prevents_mode_flapping(self):
        bot = self._bot()
        cfg = self._cfg()
        metrics = {"belief_debt_sec": 0.0, "evidence_contradiction_score": 0.0}
        with mock.patch("eclipse_scalper.execution.correlation_risk._now", side_effect=[1000.0, 1002.0, 1006.0, 1007.0, 1016.0]):
            r1 = correlation_risk.compute_correlation_risk(bot, metrics, cfg)
            self.assertEqual(str(r1["corr_regime"]), "NORMAL")
            r2 = correlation_risk.compute_correlation_risk(bot, metrics, cfg)
            self.assertEqual(str(r2["corr_regime"]), "NORMAL")
            r3 = correlation_risk.compute_correlation_risk(bot, metrics, cfg)
            self.assertIn(str(r3["corr_regime"]), ("TIGHTENING", "STRESS"))
            bot.state.run_context["corr_returns"] = {
                "BTCUSDT": [0.02, -0.02, 0.02, -0.02, 0.02, -0.02],
                "ETHUSDT": [-0.02, 0.02, -0.02, 0.02, -0.02, 0.02],
            }
            r4 = correlation_risk.compute_correlation_risk(bot, metrics, cfg)
            self.assertEqual(str(r4["corr_regime"]), str(r3["corr_regime"]))
            r5 = correlation_risk.compute_correlation_risk(bot, metrics, cfg)
            self.assertEqual(str(r5["corr_regime"]), "NORMAL")

    def test_output_shape_is_stable(self):
        bot = self._bot()
        out = correlation_risk.compute_correlation_risk(bot, {}, self._cfg())
        for key in (
            "corr_pressure",
            "corr_regime",
            "corr_confidence",
            "corr_reason_tags",
            "corr_group_scores",
        ):
            self.assertIn(key, out)
        self.assertIsInstance(out["corr_group_scores"], dict)


if __name__ == "__main__":
    unittest.main()
