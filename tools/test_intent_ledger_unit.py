#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import types
import unittest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution import intent_ledger  # noqa: E402


class IntentLedgerTests(unittest.TestCase):
    def _bot(self, path: Path):
        journal_path = path.parent / "execution_journal.jsonl"
        return types.SimpleNamespace(
            cfg=types.SimpleNamespace(
                INTENT_LEDGER_ENABLED=True,
                INTENT_LEDGER_PATH=str(path),
                INTENT_LEDGER_REUSE_MAX_AGE_SEC=3600.0,
                EVENT_JOURNAL_PATH=str(journal_path),
            ),
            state=types.SimpleNamespace(run_context={}),
        )

    def test_record_and_reload_from_disk(self):
        with tempfile.TemporaryDirectory() as td:
            ledger_path = Path(td) / "intent_ledger.jsonl"
            bot = self._bot(ledger_path)
            out = intent_ledger.record(
                bot,
                intent_id="CID-1",
                stage="OPEN",
                symbol="BTCUSDT",
                side="buy",
                order_type="MARKET",
                client_order_id="ENTRY_BTC_ABC",
                order_id="OID-1",
                status="open",
                reason="exchange_ack",
            )
            self.assertEqual(out.get("intent_id"), "CID-1")
            self.assertTrue(ledger_path.exists())
            raw = ledger_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(raw), 1)
            payload = json.loads(raw[0])
            self.assertEqual(payload.get("stage"), "OPEN")

            bot2 = self._bot(ledger_path)
            got = intent_ledger.get_intent(bot2, "CID-1")
            self.assertIsNotNone(got)
            self.assertEqual(str(got.get("order_id")), "OID-1")
            reuse = intent_ledger.find_reusable_intent(bot2, intent_id="CID-1")
            self.assertIsNotNone(reuse)

    def test_resolve_by_order_or_client_id(self):
        with tempfile.TemporaryDirectory() as td:
            ledger_path = Path(td) / "intent_ledger.jsonl"
            bot = self._bot(ledger_path)
            intent_ledger.record(
                bot,
                intent_id="CID-2",
                stage="ACKED",
                symbol="ETHUSDT",
                side="sell",
                order_type="LIMIT",
                client_order_id="ENTRY_ETH_DEF",
                order_id="OID-2",
                status="open",
            )
            self.assertEqual(intent_ledger.resolve_intent_id(bot, order_id="OID-2"), "CID-2")
            self.assertEqual(intent_ledger.resolve_intent_id(bot, client_order_id="ENTRY_ETH_DEF"), "CID-2")

    def test_summary_tracks_unknown_count_and_resolve_time(self):
        with tempfile.TemporaryDirectory() as td:
            ledger_path = Path(td) / "intent_ledger.jsonl"
            bot = self._bot(ledger_path)

            orig_now = intent_ledger.time.time
            try:
                intent_ledger.time.time = lambda: 100.0
                intent_ledger.record(bot, intent_id="CID-1", stage="SUBMITTED_UNKNOWN", symbol="BTCUSDT")
                intent_ledger.time.time = lambda: 130.0
                intent_ledger.record(bot, intent_id="CID-1", stage="DONE", symbol="BTCUSDT")
                intent_ledger.time.time = lambda: 150.0
                intent_ledger.record(bot, intent_id="CID-2", stage="CANCEL_SENT_UNKNOWN", symbol="ETHUSDT")
                out = intent_ledger.summary(bot, now_ts=200.0)
            finally:
                intent_ledger.time.time = orig_now

            self.assertEqual(int(out.get("intents_total", 0)), 2)
            self.assertEqual(int(out.get("intents_done", 0)), 1)
            self.assertEqual(int(out.get("intent_unknown_count", 0)), 1)
            self.assertGreaterEqual(float(out.get("intent_unknown_oldest_sec", 0.0)), 50.0)
            self.assertGreaterEqual(float(out.get("intent_unknown_mean_resolve_sec", 0.0)), 30.0)

    def test_load_from_journal_when_ledger_file_missing(self):
        with tempfile.TemporaryDirectory() as td:
            ledger_path = Path(td) / "intent_ledger.jsonl"
            journal_path = Path(td) / "execution_journal.jsonl"
            journal_payload = {
                "event": "intent.ledger",
                "data": {
                    "ts": 123.0,
                    "intent_id": "CID-JRN-1",
                    "stage": "OPEN",
                    "symbol": "SOLUSDT",
                    "side": "buy",
                    "type": "LIMIT",
                    "status": "open",
                    "reason": "journal_recovery",
                    "client_order_id": "ENTRY_SOL_A",
                    "order_id": "OID-JRN-1",
                    "is_exit": False,
                    "meta": {},
                },
            }
            journal_path.write_text(json.dumps(journal_payload) + "\n", encoding="utf-8")
            bot = self._bot(ledger_path)
            got = intent_ledger.get_intent(bot, "CID-JRN-1")
            self.assertIsNotNone(got)
            self.assertEqual(str(got.get("order_id") or ""), "OID-JRN-1")
            self.assertEqual(
                str(intent_ledger.resolve_intent_id(bot, client_order_id="ENTRY_SOL_A") or ""),
                "CID-JRN-1",
            )

    def test_find_pending_unknown_intent_respects_stage_and_age(self):
        with tempfile.TemporaryDirectory() as td:
            ledger_path = Path(td) / "intent_ledger.jsonl"
            bot = self._bot(ledger_path)
            bot.cfg.INTENT_LEDGER_UNKNOWN_MAX_AGE_SEC = 300.0

            orig_now = intent_ledger.time.time
            try:
                intent_ledger.time.time = lambda: 100.0
                intent_ledger.record(
                    bot,
                    intent_id="CID-UNK-1",
                    stage="SUBMITTED_UNKNOWN",
                    symbol="BTCUSDT",
                    client_order_id="ENTRY_BTC_UNK",
                )
                intent_ledger.time.time = lambda: 200.0
                pending = intent_ledger.find_pending_unknown_intent(bot, client_order_id="ENTRY_BTC_UNK")
                self.assertIsNotNone(pending)
                intent_ledger.time.time = lambda: 500.0
                expired = intent_ledger.find_pending_unknown_intent(bot, client_order_id="ENTRY_BTC_UNK")
                self.assertIsNone(expired)
            finally:
                intent_ledger.time.time = orig_now


if __name__ == "__main__":
    unittest.main()
