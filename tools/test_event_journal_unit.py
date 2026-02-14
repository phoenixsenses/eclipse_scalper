#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import types
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution import event_journal  # noqa: E402


class EventJournalTests(unittest.TestCase):
    def test_append_event_and_transition(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "journal.jsonl"
            bot = types.SimpleNamespace(cfg=types.SimpleNamespace(EVENT_JOURNAL_PATH=str(path)))

            ok1 = event_journal.append_event(bot, "order.retry", {"k": "BTCUSDT"})
            ok2 = event_journal.journal_transition(
                bot,
                machine="order_intent",
                entity="CID-1",
                state_from="INTENT_CREATED",
                state_to="SUBMITTED",
                reason="send",
                correlation_id="CID-1",
                meta={"attempt": 1},
            )
            self.assertTrue(ok1)
            self.assertTrue(ok2)
            text = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(text), 2)
            ev1 = json.loads(text[0])
            ev2 = json.loads(text[1])
            self.assertEqual(ev1.get("event"), "order.retry")
            self.assertEqual(ev2.get("event"), "state.transition")
            self.assertEqual((ev2.get("data") or {}).get("state_to"), "SUBMITTED")


if __name__ == "__main__":
    unittest.main()
