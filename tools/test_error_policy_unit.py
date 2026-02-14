#!/usr/bin/env python3
from __future__ import annotations

import types
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution import error_policy  # noqa: E402


class ErrorPolicyTests(unittest.TestCase):
    def test_binance_duplicate_client_id_is_retry_modify(self):
        out = error_policy.classify_order_error_policy(Exception("clientOrderId is duplicated -4116"))
        self.assertEqual(out["error_class"], error_policy.ERROR_CLASS_RETRYABLE_MOD)
        self.assertTrue(out["retryable"])

    def test_unknown_order_is_idempotent_safe(self):
        out = error_policy.classify_order_error_policy(Exception("Unknown order -2011"))
        self.assertEqual(out["error_class"], error_policy.ERROR_CLASS_IDEMPOTENT)
        self.assertFalse(out["retryable"])

    def test_coinbase_rate_limit_is_retryable(self):
        ex = types.SimpleNamespace(id="coinbase")
        out = error_policy.classify_order_error_policy(Exception("rate limit exceeded"), ex=ex, sym_raw="BTC/USD")
        self.assertEqual(out["reason"], "exchange_busy")
        self.assertEqual(out["error_class"], error_policy.ERROR_CLASS_RETRYABLE)
        self.assertTrue(out["retryable"])

    def test_unknown_error_defaults_conservative(self):
        out = error_policy.classify_order_error_policy(Exception("weird upstream blew up"))
        self.assertEqual(out["reason"], "unknown")
        self.assertEqual(out["error_class"], error_policy.ERROR_CLASS_FATAL)
        self.assertFalse(out["retryable"])


if __name__ == "__main__":
    unittest.main()
