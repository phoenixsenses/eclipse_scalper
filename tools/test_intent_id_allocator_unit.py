#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import threading
import types
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution import intent_allocator  # noqa: E402


class IntentAllocatorTests(unittest.TestCase):
    def _bot(self, state_path: Path, *, run_context=None):
        if run_context is None:
            run_context = {}
        return types.SimpleNamespace(
            cfg=types.SimpleNamespace(
                INTENT_ALLOCATOR_ENABLED=True,
                INTENT_ALLOCATOR_STATE_PATH=str(state_path),
                BOT_INSTANCE_ID="PHX01",
            ),
            state=types.SimpleNamespace(run_context=run_context),
        )

    def test_allocate_is_unique_under_threads(self):
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "intent_allocator_state.json"
            bot = self._bot(state_path)
            out: list[str] = []
            out_lock = threading.Lock()

            def worker():
                local: list[str] = []
                for _ in range(300):
                    local.append(
                        intent_allocator.allocate_intent_id(
                            bot,
                            component="entry_loop",
                            intent_kind="ENTRY",
                            symbol="BTCUSDT",
                            is_exit=False,
                        )
                    )
                with out_lock:
                    out.extend(local)

            threads = [threading.Thread(target=worker) for _ in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(len(out), 1800)
            self.assertEqual(len(set(out)), 1800)
            self.assertTrue(all(s.startswith("I2.") for s in out))

    def test_restart_continues_sequence(self):
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "intent_allocator_state.json"
            bot1 = self._bot(state_path)
            first = intent_allocator.allocate_intent_id(
                bot1, component="entry_loop", intent_kind="ENTRY", symbol="ETHUSDT"
            )
            second = intent_allocator.allocate_intent_id(
                bot1, component="entry_loop", intent_kind="ENTRY", symbol="ETHUSDT"
            )
            self.assertNotEqual(first, second)
            seq2 = int(second.rsplit(".", 1)[-1])

            bot2 = self._bot(state_path, run_context={})
            third = intent_allocator.allocate_intent_id(
                bot2, component="entry_loop", intent_kind="ENTRY", symbol="ETHUSDT"
            )
            seq3 = int(third.rsplit(".", 1)[-1])
            self.assertGreater(seq3, seq2)

    def test_derive_client_order_id_deterministic_and_length_safe(self):
        iid = "I2.PHX01.1739318400.entry_loop.ENTRY.BTCUSDT.00000042"
        c1 = intent_allocator.derive_client_order_id(intent_id=iid, prefix="ENTRY", symbol="BTCUSDT")
        c2 = intent_allocator.derive_client_order_id(intent_id=iid, prefix="ENTRY", symbol="BTCUSDT")
        c3 = intent_allocator.derive_client_order_id(intent_id=iid + "_x", prefix="ENTRY", symbol="BTCUSDT")
        self.assertEqual(c1, c2)
        self.assertNotEqual(c1, c3)
        self.assertLessEqual(len(c1), 35)
        self.assertTrue(all(ch.isalnum() or ch in ("_", "-") for ch in c1))

    def test_derive_client_order_id_prefix_sanitized(self):
        iid = "I2.PHX01.1739318400.reconcile.stop.REPAIR.BTCUSDT.00000101"
        coid = intent_allocator.derive_client_order_id(
            intent_id=iid,
            prefix="ENTRY BAD PREFIX %%%%%%% LONGVALUE",
            symbol="BTC/USDT:USDT",
        )
        self.assertLessEqual(len(coid), 35)
        self.assertTrue(coid.startswith("ENTRY"))


if __name__ == "__main__":
    unittest.main()
