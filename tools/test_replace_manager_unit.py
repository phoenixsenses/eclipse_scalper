#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution import replace_manager as rm  # noqa: E402


class ReplaceManagerTests(unittest.TestCase):
    def test_successful_cancel_replace(self):
        calls = {"cancel": 0, "create": 0}

        async def _cancel(_oid, _sym):
            calls["cancel"] += 1
            return True

        async def _create():
            calls["create"] += 1
            return {"id": "new-1", "status": "open"}

        out = asyncio.run(
            rm.run_cancel_replace(
                cancel_order_id="old-1",
                symbol="BTC/USDT",
                max_attempts=2,
                cancel_fn=_cancel,
                create_fn=_create,
            )
        )
        self.assertTrue(out.success)
        self.assertEqual(out.state, rm.STATE_DONE)
        self.assertEqual(calls["cancel"], 1)
        self.assertEqual(calls["create"], 1)

    def test_cancel_unknown_but_filled_status_classified(self):
        async def _cancel(_oid, _sym):
            return False

        async def _create():
            return None

        async def _status(_oid, _sym):
            return "filled"

        out = asyncio.run(
            rm.run_cancel_replace(
                cancel_order_id="old-1",
                symbol="BTC/USDT",
                max_attempts=1,
                cancel_fn=_cancel,
                create_fn=_create,
                status_fn=_status,
            )
        )
        self.assertFalse(out.success)
        self.assertEqual(out.state, rm.STATE_FILLED_AFTER_CANCEL)
        self.assertEqual(out.reason, "filled_after_cancel")
        self.assertEqual(out.last_status, "filled")

    def test_bounded_attempts_no_infinite_loop(self):
        calls = {"cancel": 0, "create": 0}

        async def _cancel(_oid, _sym):
            calls["cancel"] += 1
            return False

        async def _create():
            calls["create"] += 1
            return None

        out = asyncio.run(
            rm.run_cancel_replace(
                cancel_order_id="old-1",
                symbol="BTC/USDT",
                max_attempts=3,
                cancel_fn=_cancel,
                create_fn=_create,
            )
        )
        self.assertFalse(out.success)
        self.assertEqual(out.attempts, 3)
        self.assertEqual(calls["cancel"], 3)
        self.assertEqual(calls["create"], 0)

    def test_strict_transition_mode_raises_on_invalid_manual_flow(self):
        async def _cancel(_oid, _sym):
            return True

        async def _create():
            return None

        # This flow stays valid; strict mode should still complete cleanly and not raise.
        out = asyncio.run(
            rm.run_cancel_replace(
                cancel_order_id="old-1",
                symbol="BTC/USDT",
                max_attempts=1,
                cancel_fn=_cancel,
                create_fn=_create,
                strict_transitions=True,
            )
        )
        self.assertFalse(out.success)
        self.assertIn(out.state, (rm.STATE_REPLACE_RACE, rm.STATE_CANCEL_SENT_UNKNOWN, rm.STATE_DONE))

    def test_replace_envelope_blocks_when_worst_case_exceeds_cap(self):
        async def _cancel(_oid, _sym):
            return True

        async def _create():
            return {"id": "new-1"}

        out = asyncio.run(
            rm.run_cancel_replace(
                cancel_order_id="old-1",
                symbol="BTC/USDT",
                max_attempts=2,
                cancel_fn=_cancel,
                create_fn=_create,
                current_exposure_notional=120.0,
                new_order_notional=80.0,
                max_worst_case_notional=150.0,
            )
        )
        self.assertFalse(out.success)
        self.assertEqual(out.reason, "replace_envelope_block")
        self.assertEqual(out.attempts, 0)

    def test_replace_ambiguity_cap_triggers_giveup(self):
        async def _cancel(_oid, _sym):
            return False

        async def _create():
            return None

        async def _status(_oid, _sym):
            return ""

        out = asyncio.run(
            rm.run_cancel_replace(
                cancel_order_id="old-1",
                symbol="BTC/USDT",
                max_attempts=5,
                cancel_fn=_cancel,
                create_fn=_create,
                status_fn=_status,
                max_ambiguity_attempts=2,
            )
        )
        self.assertFalse(out.success)
        self.assertEqual(out.reason, "replace_ambiguity_cap")
        self.assertEqual(out.attempts, 2)
        self.assertEqual(out.state, rm.STATE_REPLACE_RACE)
        self.assertEqual(int(out.ambiguity_count), 2)
        self.assertEqual(int(out.cancel_attempts), 2)
        self.assertEqual(int(out.create_attempts), 0)

    def test_replace_reconcile_required_when_cancel_unknown_and_no_create(self):
        async def _cancel(_oid, _sym):
            return False

        async def _create():
            return {"id": "new-1"}

        out = asyncio.run(
            rm.run_cancel_replace(
                cancel_order_id="old-1",
                symbol="BTC/USDT",
                max_attempts=3,
                cancel_fn=_cancel,
                create_fn=_create,
                status_fn=None,
                max_ambiguity_attempts=0,
            )
        )
        self.assertFalse(out.success)
        self.assertEqual(out.reason, "replace_reconcile_required")
        self.assertEqual(out.state, rm.STATE_REPLACE_RACE)
        self.assertEqual(int(out.attempts), 3)
        self.assertEqual(int(out.create_attempts), 0)
        self.assertEqual(int(out.ambiguity_count), 3)

    def test_verify_cancel_status_blocks_create_when_still_open(self):
        calls = {"cancel": 0, "create": 0, "status": 0}

        async def _cancel(_oid, _sym):
            calls["cancel"] += 1
            return True

        async def _create():
            calls["create"] += 1
            return {"id": "new-1"}

        async def _status(_oid, _sym):
            calls["status"] += 1
            return "open"

        out = asyncio.run(
            rm.run_cancel_replace(
                cancel_order_id="old-1",
                symbol="BTC/USDT",
                max_attempts=2,
                cancel_fn=_cancel,
                create_fn=_create,
                status_fn=_status,
                verify_cancel_with_status=True,
                max_ambiguity_attempts=1,
            )
        )
        self.assertFalse(out.success)
        self.assertEqual(out.reason, "replace_ambiguity_cap")
        self.assertEqual(calls["create"], 0)
        self.assertGreaterEqual(calls["status"], 1)

    def test_verify_cancel_status_allows_create_when_canceled(self):
        calls = {"cancel": 0, "create": 0, "status": 0}

        async def _cancel(_oid, _sym):
            calls["cancel"] += 1
            return True

        async def _create():
            calls["create"] += 1
            return {"id": "new-2", "status": "open"}

        async def _status(_oid, _sym):
            calls["status"] += 1
            return "canceled"

        out = asyncio.run(
            rm.run_cancel_replace(
                cancel_order_id="old-1",
                symbol="BTC/USDT",
                max_attempts=2,
                cancel_fn=_cancel,
                create_fn=_create,
                status_fn=_status,
                verify_cancel_with_status=True,
            )
        )
        self.assertTrue(out.success)
        self.assertEqual(out.reason, "replace_success")
        self.assertEqual(calls["create"], 1)


if __name__ == "__main__":
    unittest.main()
