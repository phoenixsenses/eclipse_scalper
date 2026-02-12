#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tools import replay_trade  # noqa: E402


class ReplayTradeTests(unittest.TestCase):
    def test_replay_filters_correlation_and_reports_last_state(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "journal.jsonl"
            rows = [
                {
                    "ts": 1.0,
                    "event": "state.transition",
                    "data": {
                        "machine": "order_intent",
                        "entity": "CID-1",
                        "state_from": "INTENT_CREATED",
                        "state_to": "SUBMITTED",
                        "reason": "send",
                        "correlation_id": "CID-1",
                    },
                },
                {
                    "ts": 2.0,
                    "event": "state.transition",
                    "data": {
                        "machine": "order_intent",
                        "entity": "CID-1",
                        "state_from": "SUBMITTED",
                        "state_to": "DONE",
                        "reason": "terminal",
                        "correlation_id": "CID-1",
                    },
                },
                {
                    "ts": 3.0,
                    "event": "state.transition",
                    "data": {
                        "machine": "order_intent",
                        "entity": "CID-OTHER",
                        "state_from": "INTENT_CREATED",
                        "state_to": "SUBMITTED",
                        "reason": "send",
                        "correlation_id": "CID-OTHER",
                    },
                },
            ]
            with path.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            out = replay_trade.replay(path, correlation_id="CID-1")
            self.assertEqual(int(out.get("count", 0)), 2)
            self.assertEqual(str(out.get("last_state") or ""), "DONE")
            self.assertEqual(len(out.get("transitions", [])), 2)


if __name__ == "__main__":
    unittest.main()
