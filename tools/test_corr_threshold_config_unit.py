#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path


class CorrThresholdConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.scalper_root = self.repo_root / "eclipse_scalper"

    def _read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def test_env_example_has_canonical_corr_thresholds(self):
        env_example = self.repo_root / ".env.example"
        text = self._read(env_example)
        self.assertIn("CORR_STRESS_THRESHOLD=", text)
        self.assertIn("CORR_PRESSURE_THRESHOLD=", text)
        self.assertIn("CORR_EXIT_PNL_DROP_THRESHOLD=", text)

    def test_run_profile_has_canonical_corr_thresholds(self):
        profile = self.scalper_root / "run-bot-ps2.ps1"
        text = self._read(profile)
        self.assertIn("$env:CORR_STRESS_THRESHOLD", text)
        self.assertIn("$env:CORR_PRESSURE_THRESHOLD", text)
        self.assertIn("$env:CORR_EXIT_PNL_DROP_THRESHOLD", text)

    def test_workflows_use_canonical_cli_flags(self):
        files = [
            self.scalper_root / ".github" / "workflows" / "telemetry-dashboard.yml",
            self.scalper_root / ".github" / "workflows" / "telemetry-smoke.yml",
        ]
        for path in files:
            text = self._read(path)
            self.assertIn("--corr-stress-threshold", text, str(path))
            self.assertIn("--corr-pressure-threshold", text, str(path))
            self.assertIn("--corr-exit-pnl-drop-threshold", text, str(path))
            self.assertNotIn("--corr-stress-critical-threshold", text, str(path))
            self.assertNotIn("--corr-pressure-critical-threshold", text, str(path))
            self.assertNotIn("--corr-exit-pnl-drop-critical-threshold", text, str(path))


if __name__ == "__main__":
    unittest.main()

