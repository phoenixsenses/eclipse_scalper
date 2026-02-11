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


class EntryCommitRaceUnitTests(unittest.TestCase):
    def _base_record(self):
        from eclipse_scalper.execution.entry_decision import compute_entry_decision

        return compute_entry_decision(
            symbol="DOGEUSDT",
            signal={"action": "buy", "confidence": 0.7, "type": "market"},
            guard_knobs={"mode": "GREEN", "allow_entries": True, "min_entry_conf": 0.3},
            min_confidence=0.30,
            amount=10.0,
            order_type="market",
            price=None,
            planned_notional=6.0,
            stage="propose",
        )

    def test_commit_denied_when_posture_changes(self):
        from eclipse_scalper.execution.entry_decision import commit_entry_intent

        rec = self._base_record()
        ok, out = commit_entry_intent(
            rec,
            current_guard_knobs={"mode": "RED", "allow_entries": False, "min_entry_conf": 0.6},
            in_position_fn=lambda: False,
            pending_fn=lambda: False,
        )
        self.assertFalse(ok)
        self.assertEqual(out.action, "DEFER")
        self.assertIn(out.reason_primary, ("posture_changed", "guard_blocked"))

    def test_commit_denied_when_position_appears(self):
        from eclipse_scalper.execution.entry_decision import commit_entry_intent

        rec = self._base_record()
        ok, out = commit_entry_intent(
            rec,
            current_guard_knobs={"mode": "GREEN", "allow_entries": True, "min_entry_conf": 0.3},
            in_position_fn=lambda: True,
            pending_fn=lambda: False,
        )
        self.assertFalse(ok)
        self.assertEqual(out.action, "DEFER")
        self.assertEqual(out.reason_primary, "risk_in_position")

    def test_commit_allowed_when_state_stable(self):
        from eclipse_scalper.execution.entry_decision import commit_entry_intent

        rec = self._base_record()
        ok, out = commit_entry_intent(
            rec,
            current_guard_knobs={"mode": "GREEN", "allow_entries": True, "min_entry_conf": 0.3},
            in_position_fn=lambda: False,
            pending_fn=lambda: False,
        )
        self.assertTrue(ok)
        self.assertEqual(out.action, "ALLOW")
        self.assertEqual(out.reason_primary, "")


if __name__ == "__main__":
    unittest.main()
