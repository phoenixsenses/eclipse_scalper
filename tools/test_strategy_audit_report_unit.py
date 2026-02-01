#!/usr/bin/env python3
"""
Unit-style tests for strategy_audit_report filters and CSV export.
"""

from __future__ import annotations

import csv
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import NamedTemporaryFile

# Ensure repo root is on sys.path for local imports.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import strategy_audit_report


class StrategyAuditReportTests(unittest.TestCase):
    def _run(self, args: list[str]) -> str:
        buf = io.StringIO()
        argv = sys.argv[:]
        try:
            sys.argv = ["strategy_audit_report.py"] + args
            with redirect_stdout(buf):
                strategy_audit_report.main()
        finally:
            sys.argv = argv
        return buf.getvalue()

    def _write_audit_csv(self, rows: list[dict]) -> str:
        with NamedTemporaryFile("w+", delete=False, encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["ts", "symbol", "outcome", "blockers"],
            )
            w.writeheader()
            for r in rows:
                w.writerow(r)
            return f.name

    def test_csv_export(self):
        path = self._write_audit_csv(
            [
                {"ts": "1", "symbol": "DOGEUSDT", "outcome": "signal", "blockers": ""},
                {"ts": "2", "symbol": "DOGEUSDT", "outcome": "blocked", "blockers": "cooldown|session"},
            ]
        )

        with NamedTemporaryFile("w+", delete=False, encoding="utf-8") as out_f:
            out_path = out_f.name

        _ = self._run(["--path", path, "--csv", out_path])

        with open(out_path, "r", encoding="utf-8") as f:
            csv_text = f.read()

        self.assertIn("section,item,count", csv_text)
        self.assertIn("outcome,signal,1", csv_text)
        self.assertIn("blocker,cooldown,1", csv_text)
        self.assertIn("blocker,session,1", csv_text)

    def test_top_sort_min(self):
        path = self._write_audit_csv(
            [
                {"ts": "1", "symbol": "DOGEUSDT", "outcome": "aaa", "blockers": "x"},
                {"ts": "2", "symbol": "DOGEUSDT", "outcome": "bbb", "blockers": "x|y"},
                {"ts": "3", "symbol": "DOGEUSDT", "outcome": "bbb", "blockers": ""},
            ]
        )

        out = self._run(["--path", path, "--top", "1", "--sort", "count"])
        self.assertIn("bbb: 2", out)
        self.assertNotIn("aaa: 1", out)

        out2 = self._run(["--path", path, "--sort", "alpha", "--min", "2"])
        self.assertIn("bbb: 2", out2)
        self.assertNotIn("aaa: 1", out2)


if __name__ == "__main__":
    unittest.main()
