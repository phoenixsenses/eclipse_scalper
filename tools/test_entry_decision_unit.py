from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

logging.disable(logging.CRITICAL)


class EntryDecisionUnitTests(unittest.TestCase):
    def test_compute_entry_decision_is_deterministic_for_same_inputs(self):
        from eclipse_scalper.execution.entry_decision import compute_entry_decision

        kwargs = dict(
            symbol="DOGEUSDT",
            signal={"action": "buy", "confidence": 0.62, "type": "market"},
            guard_knobs={"mode": "GREEN", "allow_entries": True, "min_entry_conf": 0.3},
            min_confidence=0.30,
            amount=10.0,
            order_type="market",
            price=None,
            planned_notional=6.0,
            stage="propose",
        )
        rec1 = compute_entry_decision(**kwargs).to_dict()
        rec2 = compute_entry_decision(**kwargs).to_dict()

        rec1.pop("ts", None)
        rec2.pop("ts", None)
        self.assertEqual(rec1, rec2)

    def test_reason_ranking_prefers_guard_over_confidence(self):
        from eclipse_scalper.execution.entry_decision import compute_entry_decision

        rec = compute_entry_decision(
            symbol="DOGEUSDT",
            signal={"action": "buy", "confidence": 0.10, "type": "market"},
            guard_knobs={"mode": "RED", "allow_entries": False, "min_entry_conf": 0.6},
            min_confidence=0.60,
            amount=5.0,
            order_type="market",
            price=None,
            planned_notional=3.0,
            stage="propose",
        )
        self.assertEqual(rec.action, "DENY")
        self.assertTrue(rec.reasons)
        self.assertEqual(rec.reason_primary, "guard_blocked")

    def test_monotone_under_tighter_guard(self):
        from eclipse_scalper.execution.entry_decision import compute_entry_decision

        loose = compute_entry_decision(
            symbol="DOGEUSDT",
            signal={"action": "buy", "confidence": 0.45, "type": "market"},
            guard_knobs={"mode": "GREEN", "allow_entries": True, "min_entry_conf": 0.3},
            min_confidence=0.30,
            amount=5.0,
            order_type="market",
            price=None,
            planned_notional=3.0,
            stage="propose",
        )
        tight = compute_entry_decision(
            symbol="DOGEUSDT",
            signal={"action": "buy", "confidence": 0.45, "type": "market"},
            guard_knobs={"mode": "YELLOW", "allow_entries": True, "min_entry_conf": 0.5},
            min_confidence=0.50,
            amount=5.0,
            order_type="market",
            price=None,
            planned_notional=3.0,
            stage="propose",
        )
        self.assertEqual(loose.action, "ALLOW")
        self.assertIn(tight.action, ("SCALE", "DENY"))


if __name__ == "__main__":
    unittest.main()
