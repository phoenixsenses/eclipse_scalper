#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tools import reliability_gate_refresh as rgr  # noqa: E402


class ReliabilityGateRefreshTests(unittest.TestCase):
    def test_refresh_main_uses_env_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            journ = Path(td) / "journal.jsonl"
            out = Path(td) / "reliability_gate.txt"
            tele.write_text(
                json.dumps({"event": "order.retry", "data": {"correlation_id": "CID-A"}}) + "\n",
                encoding="utf-8",
            )
            journ.write_text(
                json.dumps(
                    {
                        "event": "state.transition",
                        "data": {
                            "machine": "order_intent",
                            "state_from": "INTENT_CREATED",
                            "state_to": "SUBMITTED",
                            "correlation_id": "CID-A",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            os.environ["TELEMETRY_PATH"] = str(tele)
            os.environ["EVENT_JOURNAL_PATH"] = str(journ)
            os.environ["RELIABILITY_GATE_PATH"] = str(out)
            os.environ["RELIABILITY_GATE_WINDOW_SECONDS"] = "3600"
            rc = rgr.main([])
            self.assertEqual(rc, 0)
            txt = out.read_text(encoding="utf-8")
            self.assertIn("window_seconds=3600.0", txt)
            self.assertIn("replay_mismatch_count=0", txt)


if __name__ == "__main__":
    unittest.main()
