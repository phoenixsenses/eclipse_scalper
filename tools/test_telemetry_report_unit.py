#!/usr/bin/env python3
"""
Unit-style tests for telemetry_report filters.
"""

from __future__ import annotations

import io
import json
import sys
import time
import unittest
from contextlib import redirect_stdout
from tempfile import NamedTemporaryFile
from pathlib import Path

# Ensure repo root is on sys.path for local imports.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import telemetry_report


class TelemetryReportTests(unittest.TestCase):
    def _run(self, args: list[str]) -> str:
        buf = io.StringIO()
        argv = sys.argv[:]
        try:
            sys.argv = ["telemetry_report.py"] + args
            with redirect_stdout(buf):
                telemetry_report.main()
        finally:
            sys.argv = argv
        return buf.getvalue()

    def test_symbol_and_event_filtering(self):
        now = time.time()
        records = [
            {"event": "entry.blocked", "data": {"reason": "cooldown", "k": "DOGEUSDT"}, "ts": now},
            {"event": "entry.blocked", "data": {"reason": "signal not present", "k": "BTCUSDT"}, "ts": now},
            {"event": "order.retry", "data": {"reason": "retry", "k": "DOGEUSDT"}, "ts": now},
        ]

        with NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            path = f.name

        out = self._run(["--path", path, "--symbol", "DOGEUSDT", "--event", "entry.blocked"])
        self.assertIn("entry.blocked total: 1", out)
        self.assertIn("cooldown: 1", out)
        self.assertNotIn("signal not present", out)

        out2 = self._run(["--path", path, "--symbol", "DOGEUSDT", "--event", "order.retry"])
        self.assertIn("order.retry total: 1", out2)

    def test_reason_contains_filter(self):
        now = time.time()
        records = [
            {"event": "entry.blocked", "data": {"reason": "cooldown", "k": "DOGEUSDT"}, "ts": now},
            {"event": "entry.blocked", "data": {"reason": "signal not present", "k": "DOGEUSDT"}, "ts": now},
        ]

        with NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            path = f.name

        out = self._run(["--path", path, "--reason-contains", "cool"])
        self.assertIn("cooldown: 1", out)
        self.assertNotIn("signal not present", out)

    def test_summary_only(self):
        now = time.time()
        records = [
            {"event": "entry.blocked", "data": {"reason": "cooldown", "k": "DOGEUSDT"}, "ts": now},
        ]

        with NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            path = f.name

        out = self._run(["--path", path, "--summary-only"])
        self.assertIn("entry.blocked total: 1", out)
        self.assertIn("top guard buckets:", out)
        self.assertNotIn("cooldown: 1", out)

    def test_since_filter(self):
        now = time.time()
        records = [
            {"event": "entry.blocked", "data": {"reason": "cooldown", "k": "DOGEUSDT"}, "ts": now - 7200},
            {"event": "entry.blocked", "data": {"reason": "signal not present", "k": "DOGEUSDT"}, "ts": now},
        ]

        with NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            path = f.name

        out = self._run(["--path", path, "--since", "60"])
        self.assertIn("entry.blocked total: 1", out)
        self.assertIn("signal not present: 1", out)
        self.assertNotIn("cooldown: 1", out)

    def test_top_sort_min(self):
        now = time.time()
        records = [
            {"event": "entry.blocked", "data": {"reason": "bbb", "k": "DOGEUSDT"}, "ts": now},
            {"event": "entry.blocked", "data": {"reason": "aaa", "k": "DOGEUSDT"}, "ts": now},
            {"event": "entry.blocked", "data": {"reason": "aaa", "k": "DOGEUSDT"}, "ts": now},
            {"event": "entry.blocked", "data": {"reason": "ccc", "k": "DOGEUSDT"}, "ts": now},
        ]

        with NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            path = f.name

        out = self._run(["--path", path, "--top", "1", "--sort", "count"])
        self.assertIn("aaa: 2", out)
        self.assertNotIn("bbb: 1", out)

        out2 = self._run(["--path", path, "--sort", "alpha", "--min", "2"])
        self.assertIn("aaa: 2", out2)
        self.assertNotIn("bbb: 1", out2)

    def test_csv_output(self):
        now = time.time()
        records = [
            {"event": "entry.blocked", "data": {"reason": "cooldown", "k": "DOGEUSDT"}, "ts": now},
            {"event": "entry.blocked", "data": {"reason": "signal not present", "k": "DOGEUSDT"}, "ts": now},
        ]

        with NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            path = f.name

        with NamedTemporaryFile("w+", delete=False, encoding="utf-8") as out_f:
            out_path = out_f.name

        _ = self._run(["--path", path, "--csv", out_path])

        with open(out_path, "r", encoding="utf-8") as f:
            csv_text = f.read()

        self.assertIn("event,reason,count", csv_text)
        self.assertIn("entry.blocked,cooldown,1", csv_text)
        self.assertIn("entry.blocked,signal not present,1", csv_text)

    def test_csv_output_respects_filters(self):
        now = time.time()
        records = [
            {"event": "entry.blocked", "data": {"reason": "cooldown", "k": "DOGEUSDT"}, "ts": now},
            {"event": "entry.blocked", "data": {"reason": "signal not present", "k": "BTCUSDT"}, "ts": now},
            {"event": "order.retry", "data": {"reason": "retry", "k": "DOGEUSDT"}, "ts": now},
        ]

        with NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            path = f.name

        with NamedTemporaryFile("w+", delete=False, encoding="utf-8") as out_f:
            out_path = out_f.name

        _ = self._run(["--path", path, "--csv", out_path, "--symbol", "DOGEUSDT", "--event", "entry.blocked"])

        with open(out_path, "r", encoding="utf-8") as f:
            csv_text = f.read()

        self.assertIn("entry.blocked,cooldown,1", csv_text)
        self.assertNotIn("signal not present", csv_text)
        self.assertNotIn("order.retry", csv_text)

    def test_csv_output_reason_contains(self):
        now = time.time()
        records = [
            {"event": "entry.blocked", "data": {"reason": "cooldown", "k": "DOGEUSDT"}, "ts": now},
            {"event": "entry.blocked", "data": {"reason": "signal not present", "k": "DOGEUSDT"}, "ts": now},
        ]

        with NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            path = f.name

        with NamedTemporaryFile("w+", delete=False, encoding="utf-8") as out_f:
            out_path = out_f.name

        _ = self._run(["--path", path, "--csv", out_path, "--reason-contains", "cool"])

        with open(out_path, "r", encoding="utf-8") as f:
            csv_text = f.read()

        self.assertIn("entry.blocked,cooldown,1", csv_text)
        self.assertNotIn("signal not present", csv_text)

    def test_csv_output_custom_event_header(self):
        now = time.time()
        records = [
            {"event": "order.retry", "data": {"reason": "retry", "k": "DOGEUSDT"}, "ts": now},
        ]

        with NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            path = f.name

        with NamedTemporaryFile("w+", delete=False, encoding="utf-8") as out_f:
            out_path = out_f.name

        _ = self._run(["--path", path, "--csv", out_path, "--event", "order.retry"])

        with open(out_path, "r", encoding="utf-8") as f:
            csv_text = f.read()

        self.assertIn("event,reason,count", csv_text)
        self.assertIn("order.retry,retry,1", csv_text)

    def test_csv_output_with_summary_only(self):
        now = time.time()
        records = [
            {"event": "entry.blocked", "data": {"reason": "cooldown", "k": "DOGEUSDT"}, "ts": now},
        ]

        with NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            path = f.name

        with NamedTemporaryFile("w+", delete=False, encoding="utf-8") as out_f:
            out_path = out_f.name

        out = self._run(["--path", path, "--csv", out_path, "--summary-only"])
        self.assertIn("entry.blocked total: 1", out)

        with open(out_path, "r", encoding="utf-8") as f:
            csv_text = f.read()

        self.assertIn("event,reason,count", csv_text)
        self.assertIn("entry.blocked,cooldown,1", csv_text)

    def test_csv_output_min_filter(self):
        now = time.time()
        records = [
            {"event": "entry.blocked", "data": {"reason": "cooldown", "k": "DOGEUSDT"}, "ts": now},
            {"event": "entry.blocked", "data": {"reason": "cooldown", "k": "DOGEUSDT"}, "ts": now},
            {"event": "entry.blocked", "data": {"reason": "signal not present", "k": "DOGEUSDT"}, "ts": now},
        ]

        with NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            path = f.name

        with NamedTemporaryFile("w+", delete=False, encoding="utf-8") as out_f:
            out_path = out_f.name

        _ = self._run(["--path", path, "--csv", out_path, "--min", "2"])

        with open(out_path, "r", encoding="utf-8") as f:
            csv_text = f.read()

        self.assertIn("entry.blocked,cooldown,2", csv_text)
        self.assertNotIn("signal not present", csv_text)


if __name__ == "__main__":
    unittest.main()
