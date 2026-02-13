#!/usr/bin/env python3
"""
Unit tests for entry_loop._entry_idempotency_key — deterministic
clientOrderId generation for Binance entry orders.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure Unicode log lines don't crash on Windows codepages
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Ensure repo root and eclipse_scalper package dir are on sys.path
ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution.entry_loop import _entry_idempotency_key  # noqa: E402


class TestEntryIdempotencyKey(unittest.TestCase):
    """Validate that the idempotency key is stable across retries."""

    def test_same_candle_produces_same_key(self):
        """Two calls within the same 60-second candle must return identical IDs."""
        fixed_ts = 1700000040.0  # arbitrary point inside a candle bucket
        with patch("eclipse_scalper.execution.entry_loop.time") as mock_time:
            mock_time.time.return_value = fixed_ts
            key1 = _entry_idempotency_key("BTCUSDT", "buy")
            # Simulate a retry 15 seconds later, still in the same candle
            mock_time.time.return_value = fixed_ts + 15.0
            key2 = _entry_idempotency_key("BTCUSDT", "buy")
        self.assertEqual(key1, key2)

    def test_different_candle_produces_different_key(self):
        """Calls in different 60-second candles must return different IDs."""
        with patch("eclipse_scalper.execution.entry_loop.time") as mock_time:
            mock_time.time.return_value = 1700000040.0
            key1 = _entry_idempotency_key("BTCUSDT", "buy")
            # Next candle (61 seconds later)
            mock_time.time.return_value = 1700000040.0 + 61.0
            key2 = _entry_idempotency_key("BTCUSDT", "buy")
        self.assertNotEqual(key1, key2)

    def test_different_symbol_produces_different_key(self):
        """Same candle, different symbol → different key."""
        fixed_ts = 1700000040.0
        with patch("eclipse_scalper.execution.entry_loop.time") as mock_time:
            mock_time.time.return_value = fixed_ts
            key_btc = _entry_idempotency_key("BTCUSDT", "buy")
            key_eth = _entry_idempotency_key("ETHUSDT", "buy")
        self.assertNotEqual(key_btc, key_eth)

    def test_different_side_produces_different_key(self):
        """Same candle + symbol but buy vs sell → different key."""
        fixed_ts = 1700000040.0
        with patch("eclipse_scalper.execution.entry_loop.time") as mock_time:
            mock_time.time.return_value = fixed_ts
            key_buy = _entry_idempotency_key("BTCUSDT", "buy")
            key_sell = _entry_idempotency_key("BTCUSDT", "sell")
        self.assertNotEqual(key_buy, key_sell)

    def test_key_under_36_chars(self):
        """Binance requires clientOrderId < 36 chars."""
        fixed_ts = 1700000040.0
        with patch("eclipse_scalper.execution.entry_loop.time") as mock_time:
            mock_time.time.return_value = fixed_ts
            for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "1000SHIBUSDT", "AVAXUSDT"):
                for action in ("buy", "sell"):
                    key = _entry_idempotency_key(sym, action)
                    self.assertLess(len(key), 36, f"Key too long for {sym}/{action}: {key!r} ({len(key)})")

    def test_prefix_format(self):
        """Key must start with SE_E_ prefix."""
        fixed_ts = 1700000040.0
        with patch("eclipse_scalper.execution.entry_loop.time") as mock_time:
            mock_time.time.return_value = fixed_ts
            key = _entry_idempotency_key("BTCUSDT", "buy")
        self.assertTrue(key.startswith("SE_E_"), f"Bad prefix: {key!r}")

    def test_long_symbol_still_under_limit(self):
        """Even with a very long symbol name, key must stay under 36 chars."""
        fixed_ts = 1700000040.0
        with patch("eclipse_scalper.execution.entry_loop.time") as mock_time:
            mock_time.time.return_value = fixed_ts
            key = _entry_idempotency_key("1000000PEPEUSDT", "buy")
        self.assertLess(len(key), 36, f"Long-symbol key too long: {key!r} ({len(key)})")
        self.assertTrue(key.startswith("SE_E_"))


if __name__ == "__main__":
    unittest.main()
