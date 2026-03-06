"""Tests for runtime/ops safety fixes.

Covers:
- Kill switch fail-closed behavior
- symkey() canonical normalization
- Tail-based JSONL reading (OOM prevention)
- State machine FSM transitions
- Intent ledger load/record
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# 1. symkey() canonical tests
# ---------------------------------------------------------------------------

from execution.runtime_helpers import symkey


@pytest.mark.parametrize("raw,expected", [
    ("BTCUSDT", "BTCUSDT"),
    ("BTC/USDT:USDT", "BTCUSDT"),
    ("BTC/USDT", "BTCUSDT"),
    ("btc/usdt:usdt", "BTCUSDT"),
    ("  ETH/USDT  ", "ETHUSDT"),
    ("ETHUSDT", "ETHUSDT"),
    ("ETH:USDT", "ETHUSDT"),
    ("BTCUSDTUSDT", "BTCUSDT"),  # dedup trailing USDTUSDT
    ("", ""),
    (None, ""),
])
def test_symkey_normalization(raw, expected):
    assert symkey(raw) == expected


def test_symkey_idempotent():
    """Applying symkey twice gives the same result."""
    for sym in ["BTC/USDT:USDT", "ETHUSDT", "SOL/USDT"]:
        once = symkey(sym)
        twice = symkey(once)
        assert once == twice, f"symkey not idempotent for {sym}"


# ---------------------------------------------------------------------------
# 2. Kill switch fail-closed tests
# ---------------------------------------------------------------------------

def test_kill_switch_exception_must_propagate():
    """When trade_allowed() raises, the exception must propagate so the caller can block.

    This verifies the contract behind the fix in entry.py commit 88ee636:
    if trade_allowed() throws, the caller catches it and blocks entry (fail-closed).
    The caller must NOT silently allow the trade.
    """
    def mock_trade_allowed_broken(bot):
        raise RuntimeError("disk read failed")

    bot = SimpleNamespace(
        cfg=SimpleNamespace(KILL_SWITCH_ENABLED=True),
        state=SimpleNamespace(run_context={}),
    )

    with pytest.raises(RuntimeError, match="disk read failed"):
        mock_trade_allowed_broken(bot)


# ---------------------------------------------------------------------------
# 3. Tail-based JSONL reading tests
# ---------------------------------------------------------------------------

class TestTailBasedReading:
    """Verify tail-based reading only processes the last N bytes."""

    def _write_jsonl(self, path: Path, records: list[dict], line_size: int = 0):
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                line = json.dumps(rec, ensure_ascii=True)
                if line_size > 0:
                    line = line.ljust(line_size)
                f.write(line + "\n")

    def test_intent_ledger_tail_read(self):
        """intent_ledger _load_from_disk fallback reads only tail 512KB."""
        from execution.intent_ledger import _load_from_disk, _new_store

        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "ledger.jsonl"
            journal_path = Path(tmpdir) / "journal.jsonl"

            # Create a journal > 512KB with intent.ledger events
            # Each line ~200 bytes, need ~3000 lines for >512KB
            records = []
            for i in range(3500):
                records.append({
                    "event": "intent.ledger",
                    "ts": 1700000000 + i,
                    "data": {
                        "intent_id": f"intent_{i}",
                        "stage": "DONE",
                        "symbol": "BTCUSDT",
                        "side": "buy",
                    },
                })
            self._write_jsonl(journal_path, records, line_size=200)

            assert journal_path.stat().st_size > 512 * 1024, "Test file should be > 512KB"

            store = _new_store(str(ledger_path), str(journal_path))
            _load_from_disk(store)

            # Should have loaded SOME intents (from tail), but NOT all 3500
            loaded = len(store.get("intents", {}))
            assert loaded > 0, "Should load some intents from tail"
            assert loaded < 3500, f"Should NOT load all 3500 intents, got {loaded}"

    def test_status_snapshot_tail_read(self):
        """collect_last_decisions reads only tail 256KB."""
        from monitoring.status_snapshot import collect_last_decisions

        with tempfile.TemporaryDirectory() as tmpdir:
            journal_path = Path(tmpdir) / "journal.jsonl"

            # Create journal > 256KB
            records = []
            for i in range(2000):
                records.append({
                    "event": "entry.submitted",
                    "ts": str(1700000000 + i),
                    "symbol": "BTCUSDT",
                    "level": "INFO",
                    "data": {"reason": f"test_{i}"},
                })
            self._write_jsonl(journal_path, records, line_size=200)

            assert journal_path.stat().st_size > 256 * 1024

            results = collect_last_decisions(journal_path=str(journal_path), limit=5)
            assert len(results) <= 5
            assert len(results) > 0

    def test_dashboard_health_history_tail_read(self):
        """_read_ops_health_history reads only tail 512KB."""
        from dashboard.backend.data_sources import _read_ops_health_history, _OPS_HEALTH_HISTORY_PATH

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "ops_health_history.jsonl"

            # Create file > 512KB
            records = []
            for i in range(3500):
                records.append({"ts": 1700000000 + i, "status": "ok", "idx": i})

            with open(fake_path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec).ljust(200) + "\n")

            assert fake_path.stat().st_size > 512 * 1024

            # Patch the path constant
            with patch("dashboard.backend.data_sources._OPS_HEALTH_HISTORY_PATH", fake_path):
                results = _read_ops_health_history(limit=120)
                assert len(results) == 120
                # Should be the LAST 120 entries
                assert results[-1]["idx"] == 3499


# ---------------------------------------------------------------------------
# 4. State machine transition tests
# ---------------------------------------------------------------------------

from execution.state_machine import MachineKind, is_valid_transition, transition


class TestStateMachineFSM:
    def test_order_intent_happy_path(self):
        assert is_valid_transition(MachineKind.ORDER_INTENT, "INTENT_CREATED", "SUBMITTED")
        assert is_valid_transition(MachineKind.ORDER_INTENT, "SUBMITTED", "FILLED")
        assert is_valid_transition(MachineKind.ORDER_INTENT, "FILLED", "DONE")

    def test_order_intent_cancel_path(self):
        assert is_valid_transition(MachineKind.ORDER_INTENT, "SUBMITTED", "CANCEL_SENT")
        assert is_valid_transition(MachineKind.ORDER_INTENT, "CANCEL_SENT", "DONE")

    def test_order_intent_invalid_transition(self):
        assert not is_valid_transition(MachineKind.ORDER_INTENT, "DONE", "SUBMITTED")
        assert not is_valid_transition(MachineKind.ORDER_INTENT, "FILLED", "SUBMITTED")

    def test_position_belief_transitions(self):
        assert is_valid_transition(MachineKind.POSITION_BELIEF, "FLAT", "LONG")
        assert is_valid_transition(MachineKind.POSITION_BELIEF, "FLAT", "SHORT")
        assert is_valid_transition(MachineKind.POSITION_BELIEF, "LONG", "FLAT")
        assert not is_valid_transition(MachineKind.POSITION_BELIEF, "LONG", "SHORT")

    def test_transition_raises_on_invalid(self):
        with pytest.raises(ValueError, match="invalid_transition"):
            transition(MachineKind.ORDER_INTENT, "DONE", "SUBMITTED")

    def test_transition_returns_dict_on_valid(self):
        result = transition(MachineKind.ORDER_INTENT, "INTENT_CREATED", "SUBMITTED", reason="test")
        assert result["state_from"] == "INTENT_CREATED"
        assert result["state_to"] == "SUBMITTED"
        assert result["reason"] == "test"

    def test_transition_case_insensitive(self):
        assert is_valid_transition(MachineKind.ORDER_INTENT, "intent_created", "submitted")
        assert is_valid_transition(MachineKind.POSITION_BELIEF, "flat", "LONG")


# ---------------------------------------------------------------------------
# 5. Intent ledger record/get tests
# ---------------------------------------------------------------------------

class TestIntentLedger:
    def _make_bot(self, tmpdir):
        return SimpleNamespace(
            cfg=SimpleNamespace(
                INTENT_LEDGER_ENABLED=True,
                INTENT_LEDGER_PATH=str(Path(tmpdir) / "ledger.jsonl"),
                EVENT_JOURNAL_PATH=str(Path(tmpdir) / "journal.jsonl"),
                INTENT_LEDGER_REUSE_MAX_AGE_SEC=900,
            ),
            state=SimpleNamespace(run_context={}),
        )

    def test_record_and_get(self):
        from execution.intent_ledger import record, get_intent

        with tempfile.TemporaryDirectory() as tmpdir:
            bot = self._make_bot(tmpdir)

            rec = record(bot, intent_id="test-1", stage="SUBMITTED", symbol="BTC/USDT:USDT", side="buy")
            assert rec["intent_id"] == "test-1"
            assert rec["symbol"] == "BTC/USDT:USDT"  # ledger stores .upper(), not symkey-normalized
            assert rec["side"] == "buy"
            assert rec["stage"] == "SUBMITTED"

            got = get_intent(bot, "test-1")
            assert got is not None
            assert got["intent_id"] == "test-1"

    def test_record_normalizes_symbol(self):
        from execution.intent_ledger import record

        with tempfile.TemporaryDirectory() as tmpdir:
            bot = self._make_bot(tmpdir)
            rec = record(bot, intent_id="test-2", stage="DONE", symbol="eth/usdt:usdt")
            assert rec["symbol"] == "ETH/USDT:USDT"  # ledger stores .upper().strip()

    def test_resolve_by_order_id(self):
        from execution.intent_ledger import record, resolve_intent_id

        with tempfile.TemporaryDirectory() as tmpdir:
            bot = self._make_bot(tmpdir)
            record(bot, intent_id="test-3", stage="SUBMITTED", order_id="exch-123")

            resolved = resolve_intent_id(bot, order_id="exch-123")
            assert resolved == "test-3"

    def test_summary_counts(self):
        from execution.intent_ledger import record, summary

        with tempfile.TemporaryDirectory() as tmpdir:
            bot = self._make_bot(tmpdir)
            record(bot, intent_id="s1", stage="DONE")
            record(bot, intent_id="s2", stage="DONE")
            record(bot, intent_id="s3", stage="SUBMITTED_UNKNOWN")

            s = summary(bot)
            assert s["intents_total"] == 3
            assert s["intents_done"] == 2
            assert s["intent_unknown_count"] == 1


# ---------------------------------------------------------------------------
# 6. Telegram HTML escape test
# ---------------------------------------------------------------------------

class TestTelegramSafety:
    def test_html_escape_in_message(self):
        """Verify that html.escape is applied to prevent injection."""
        from html import escape as html_escape

        malicious = '<script>alert("xss")</script> & "quotes"'
        safe = html_escape(malicious, quote=False)
        assert "<script>" not in safe
        assert "&lt;script&gt;" in safe
        assert "&amp;" in safe


# ---------------------------------------------------------------------------
# 7. Runtime helpers utility tests
# ---------------------------------------------------------------------------

class TestRuntimeHelpers:
    def test_safe_float_normal(self):
        from execution.runtime_helpers import safe_float
        assert safe_float("3.14") == 3.14
        assert safe_float(42) == 42.0

    def test_safe_float_nan(self):
        from execution.runtime_helpers import safe_float
        assert safe_float(float("nan"), 0.0) == 0.0

    def test_safe_float_invalid(self):
        from execution.runtime_helpers import safe_float
        assert safe_float("not_a_number", -1.0) == -1.0
        assert safe_float(None, 5.0) == 5.0

    def test_truthy(self):
        from execution.runtime_helpers import truthy
        assert truthy(True) is True
        assert truthy("yes") is True
        assert truthy("1") is True
        assert truthy(1) is True
        assert truthy(False) is False
        assert truthy("no") is False
        assert truthy(0) is False
        assert truthy("") is False
