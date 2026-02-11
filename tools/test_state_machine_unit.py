#!/usr/bin/env python3
from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution import state_machine as sm  # noqa: E402


class StateMachineTests(unittest.TestCase):
    def test_valid_order_transition(self):
        tr = sm.transition(
            sm.MachineKind.ORDER_INTENT,
            sm.OrderIntentState.INTENT_CREATED.value,
            sm.OrderIntentState.SUBMITTED.value,
            "send",
        )
        self.assertEqual(tr.state_to, sm.OrderIntentState.SUBMITTED.value)

    def test_invalid_order_transition_raises(self):
        with self.assertRaises(sm.TransitionError):
            sm.transition(
                sm.MachineKind.ORDER_INTENT,
                sm.OrderIntentState.INTENT_CREATED.value,
                sm.OrderIntentState.FILLED.value,
                "invalid_skip",
            )

    def test_unknown_order_maps_to_uncertainty_state(self):
        out = sm.map_unknown_order_state(sm.OrderIntentState.SUBMITTED.value)
        self.assertEqual(out, sm.OrderIntentState.SUBMITTED_UNKNOWN.value)
        out2 = sm.map_unknown_order_state(sm.OrderIntentState.REPLACE_RACE.value)
        self.assertEqual(out2, sm.OrderIntentState.CANCEL_SENT_UNKNOWN.value)

    def test_position_transition(self):
        self.assertTrue(
            sm.is_valid_transition(
                sm.MachineKind.POSITION_BELIEF,
                sm.PositionBeliefState.FLAT.value,
                sm.PositionBeliefState.OPEN_CONFIRMED.value,
            )
        )


if __name__ == "__main__":
    unittest.main()
