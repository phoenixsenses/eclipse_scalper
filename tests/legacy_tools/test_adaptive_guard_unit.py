#!/usr/bin/env python3
"""
Unit test for adaptive guard guard-history events.
"""

from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import sys

# Ensure repo root and eclipse_scalper package dir are on sys.path
ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


class AdaptiveGuardTests(unittest.TestCase):
    def test_guard_history_event_applies_global_overrides(self):
        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)
            events_path = temp_dir / "guard_history_events.jsonl"
            state_path = temp_dir / "adaptive_guard_state.json"
            telemetry_path = temp_dir / "telemetry.jsonl"
            drift_path = temp_dir / "telemetry_drift.jsonl"

            payload = {
                "ts": 1234567890.0,
                "event": "telemetry.guard_history_spike",
                "data": {
                    "hit_count": 2,
                    "hit_rate": 0.4,
                    "confidence_delta": 0.1,
                    "confidence_duration": 900.0,
                    "leverage_scale": 0.85,
                    "leverage_duration": 900.0,
                    "notional_scale": 0.8,
                    "notional_duration": 900.0,
                },
            }
            events_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

            old_env = {
                "TELEMETRY_GUARD_HISTORY_EVENTS": os.getenv("TELEMETRY_GUARD_HISTORY_EVENTS", ""),
                "ADAPTIVE_GUARD_STATE": os.getenv("ADAPTIVE_GUARD_STATE", ""),
                "TELEMETRY_PATH": os.getenv("TELEMETRY_PATH", ""),
                "TELEMETRY_DRIFT_PATH": os.getenv("TELEMETRY_DRIFT_PATH", ""),
            }
            try:
                os.environ["TELEMETRY_GUARD_HISTORY_EVENTS"] = str(events_path)
                os.environ["ADAPTIVE_GUARD_STATE"] = str(state_path)
                os.environ["TELEMETRY_PATH"] = str(telemetry_path)
                os.environ["TELEMETRY_DRIFT_PATH"] = str(drift_path)

                from eclipse_scalper.execution import adaptive_guard

                importlib.reload(adaptive_guard)
                adaptive_guard.refresh_state()

                min_conf, reason = adaptive_guard.get_override("BTCUSDT", 0.5)
                scale, scale_reason = adaptive_guard.get_leverage_scale("BTCUSDT")
                notional_scale, notional_reason = adaptive_guard.get_notional_scale("BTCUSDT")

                self.assertAlmostEqual(min_conf, 0.6, places=3)
                self.assertTrue("guard_history" in reason)
                self.assertAlmostEqual(scale, 0.85, places=3)
                self.assertTrue("guard_history" in scale_reason)
                self.assertAlmostEqual(notional_scale, 0.8, places=3)
                self.assertTrue("guard_history" in notional_reason)
            finally:
                for key, value in old_env.items():
                    if value:
                        os.environ[key] = value
                    elif key in os.environ:
                        os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
