import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution import protection_manager


class EntryStageHintTests(unittest.TestCase):
    def test_record_and_get_hint(self):
        rc = {}
        plan = {"stage": "stage1_emergency", "stage1_required": True}
        out = protection_manager.record_entry_stage_hint(
            rc,
            symbol="DOGEUSDT",
            plan=plan,
            requested_qty=100.0,
            filled_qty=80.0,
            ttl_sec=60.0,
            now_ts=1000.0,
        )
        self.assertEqual(out["stage"], "stage1_emergency")
        hint = protection_manager.get_entry_stage_hint(rc, symbol="DOGEUSDT", now_ts=1001.0)
        self.assertIsNotNone(hint)
        self.assertTrue(hint["stage1_required"])
        self.assertAlmostEqual(hint["filled_qty"], 80.0)

    def test_hint_expires(self):
        rc = {}
        protection_manager.record_entry_stage_hint(
            rc,
            symbol="DOGEUSDT",
            plan={"stage1_required": True},
            requested_qty=10.0,
            filled_qty=10.0,
            ttl_sec=5.0,
            now_ts=100.0,
        )
        hint = protection_manager.get_entry_stage_hint(rc, symbol="DOGEUSDT", now_ts=106.0)
        self.assertIsNone(hint)

    def test_consume_hint(self):
        rc = {}
        protection_manager.record_entry_stage_hint(
            rc,
            symbol="DOGEUSDT",
            plan={"stage1_required": True},
            requested_qty=10.0,
            filled_qty=10.0,
            ttl_sec=60.0,
            now_ts=100.0,
        )
        hint = protection_manager.get_entry_stage_hint(rc, symbol="DOGEUSDT", now_ts=101.0, consume=True)
        self.assertIsNotNone(hint)
        hint2 = protection_manager.get_entry_stage_hint(rc, symbol="DOGEUSDT", now_ts=101.0)
        self.assertIsNone(hint2)


if __name__ == "__main__":
    unittest.main()
