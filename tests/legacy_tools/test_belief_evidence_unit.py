#!/usr/bin/env python3
from __future__ import annotations

import time
import types
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution import belief_evidence  # noqa: E402


class BeliefEvidenceTests(unittest.TestCase):
    def _bot(self):
        return types.SimpleNamespace(
            state=types.SimpleNamespace(run_context={}, kill_metrics={}),
            ex=types.SimpleNamespace(last_health_check=0.0),
            data=types.SimpleNamespace(last_poll_ts={}),
        )

    def test_missing_signals_are_neutral(self):
        bot = self._bot()
        out = belief_evidence.compute_belief_evidence(bot, types.SimpleNamespace(), now=1000.0)
        self.assertAlmostEqual(float(out["evidence_confidence"]), 1.0, places=6)
        self.assertEqual(int(out["evidence_degraded_sources"]), 0)

    def test_ws_degraded_rest_healthy_is_not_catastrophic(self):
        bot = self._bot()
        now = time.time()
        bot.state.run_context["ws_last_event_ts"] = now - 120.0
        bot.state.run_context["rest_last_ok_ts"] = now - 5.0
        out = belief_evidence.compute_belief_evidence(bot, types.SimpleNamespace(), now=now)
        self.assertLess(float(out["evidence_ws_score"]), 0.7)
        self.assertGreater(float(out["evidence_rest_score"]), 0.9)
        self.assertGreater(float(out["evidence_confidence"]), 0.45)

    def test_ws_and_rest_degraded_confidence_drops(self):
        bot = self._bot()
        now = time.time()
        bot.state.run_context["ws_last_event_ts"] = now - 300.0
        bot.state.run_context["rest_last_ok_ts"] = now - 300.0
        out = belief_evidence.compute_belief_evidence(bot, types.SimpleNamespace(), now=now)
        self.assertLess(float(out["evidence_confidence"]), 0.5)
        self.assertGreaterEqual(int(out["evidence_degraded_sources"]), 2)

    def test_contradiction_penalizes_confidence_more_than_single_source_stale(self):
        bot = self._bot()
        now = time.time()
        cfg = types.SimpleNamespace(BELIEF_CONTRADICTION_SEVERE_DELTA=0.3)
        # Baseline: all sources similarly stale -> lower confidence but low contradiction.
        bot.state.run_context["ws_last_event_ts"] = now - 70.0
        bot.state.run_context["rest_last_ok_ts"] = now - 70.0
        bot.state.run_context["fills_last_ts"] = now - 70.0
        baseline = belief_evidence.compute_belief_evidence(bot, cfg, now=now)

        # Contradiction: one source fresh and one critical stale creates high disagreement.
        bot2 = self._bot()
        bot2.state.run_context["ws_last_event_ts"] = now - 2.0
        bot2.state.run_context["rest_last_ok_ts"] = now - 130.0
        bot2.state.run_context["fills_last_ts"] = now - 2.0
        contrad = belief_evidence.compute_belief_evidence(bot2, cfg, now=now)

        self.assertGreater(float(contrad.get("evidence_contradiction_score", 0.0)), float(baseline.get("evidence_contradiction_score", 0.0)))
        self.assertLess(float(contrad["evidence_confidence"]), float(baseline["evidence_confidence"]))
        self.assertGreaterEqual(int(contrad.get("evidence_contradiction_streak", 0)), 1)


if __name__ == "__main__":
    unittest.main()
