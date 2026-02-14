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

from tools import replay_latest_summary as rls  # noqa: E402


class ReplayLatestSummaryTests(unittest.TestCase):
    def test_build_summary_uses_latest_corr(self):
        with tempfile.TemporaryDirectory() as td:
            telemetry = Path(td) / "telemetry.jsonl"
            journal = Path(td) / "execution_journal.jsonl"
            telemetry_rows = [
                {"event": "order.retry", "data": {"correlation_id": "CID-X", "k": "BTCUSDT"}},
            ]
            journal_rows = [
                {
                    "ts": 1.0,
                    "event": "state.transition",
                    "data": {
                        "machine": "order_intent",
                        "entity": "CID-X",
                        "state_from": "INTENT_CREATED",
                        "state_to": "SUBMITTED",
                        "reason": "send",
                        "correlation_id": "CID-X",
                    },
                },
            ]
            telemetry.write_text("\n".join(json.dumps(r) for r in telemetry_rows) + "\n", encoding="utf-8")
            journal.write_text("\n".join(json.dumps(r) for r in journal_rows) + "\n", encoding="utf-8")

            out = rls.build_summary(telemetry, journal, limit=5)
            self.assertIn("correlation_id=CID-X", out)
            self.assertIn("INTENT_CREATED->SUBMITTED", out)


if __name__ == "__main__":
    unittest.main()
