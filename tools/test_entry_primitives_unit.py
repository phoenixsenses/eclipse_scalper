import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution import entry_primitives


class EntryPrimitivesPlanTests(unittest.TestCase):
    def test_plan_none_when_unfilled(self):
        plan = entry_primitives.build_staged_protection_plan(
            requested_qty=10.0,
            filled_qty=0.0,
            min_fill_ratio=0.5,
            trailing_enabled=True,
        )
        self.assertEqual(plan["stage"], "none")
        self.assertFalse(plan["stage1_required"])
        self.assertFalse(plan["stage2_required"])
        self.assertFalse(plan["stage3_required"])

    def test_plan_partial_underfill_requires_flatten(self):
        plan = entry_primitives.build_staged_protection_plan(
            requested_qty=10.0,
            filled_qty=2.0,
            min_fill_ratio=0.5,
            trailing_enabled=True,
        )
        self.assertEqual(plan["stage"], "stage1_emergency_partial_underfill")
        self.assertTrue(plan["stage1_required"])
        self.assertFalse(plan["stage2_required"])
        self.assertTrue(plan["flatten_required"])

    def test_plan_full_fill_advances_to_trailing_stage(self):
        plan = entry_primitives.build_staged_protection_plan(
            requested_qty=10.0,
            filled_qty=10.0,
            min_fill_ratio=0.5,
            trailing_enabled=True,
        )
        self.assertEqual(plan["stage"], "stage3_trailing")
        self.assertTrue(plan["stage1_required"])
        self.assertTrue(plan["stage2_required"])
        self.assertTrue(plan["stage3_required"])
        self.assertFalse(plan["flatten_required"])


if __name__ == "__main__":
    unittest.main()
