#!/usr/bin/env python3
from __future__ import annotations

import asyncio
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

from eclipse_scalper.execution import rebuild  # noqa: E402


class _DummyEx:
    def __init__(self, positions=None, orders=None, trades=None):
        self._positions = list(positions or [])
        self._orders = list(orders or [])
        self._trades = list(trades or [])
        self.cancel_calls = []

    async def fetch_positions(self, *_args, **_kwargs):
        return list(self._positions)

    async def fetch_open_orders(self, *_args, **_kwargs):
        return list(self._orders)

    async def fetch_my_trades(self, sym=None, _since=None):
        if sym is None:
            return list(self._trades)
        return [t for t in self._trades if str(t.get("symbol") or "") == str(sym)]

    async def cancel_order(self, order_id, symbol=None):
        self.cancel_calls.append((str(order_id or ""), str(symbol or "")))
        return {"id": order_id, "status": "canceled"}


class RebuildUnitTests(unittest.TestCase):
    def _bot(self, ex, td: str):
        st = types.SimpleNamespace(positions={}, run_context={}, halt=False)
        cfg = types.SimpleNamespace(
            INTENT_LEDGER_ENABLED=True,
            INTENT_LEDGER_PATH=str(Path(td) / "intent_ledger.jsonl"),
            EVENT_JOURNAL_PATH=str(Path(td) / "execution_journal.jsonl"),
            INTENT_LEDGER_REUSE_MAX_AGE_SEC=3600.0,
        )
        return types.SimpleNamespace(ex=ex, state=st, cfg=cfg)

    def test_rebuild_positions_and_orphans(self):
        with tempfile.TemporaryDirectory() as td:
            ex = _DummyEx(
                positions=[
                    {"symbol": "BTC/USDT:USDT", "contracts": 2.0, "side": "long", "entryPrice": 101.0},
                ],
                orders=[
                    {
                        "id": "o1",
                        "symbol": "DOGE/USDT:USDT",
                        "status": "open",
                        "type": "LIMIT",
                        "clientOrderId": "ENTRY_DOGE_1",
                        "params": {},
                    },
                    {
                        "id": "o2",
                        "symbol": "BTC/USDT:USDT",
                        "status": "open",
                        "type": "STOP_MARKET",
                        "params": {"reduceOnly": True},
                    },
                ],
                trades=[
                    {"symbol": "BTC/USDT:USDT", "timestamp": 1700000000000},
                ],
            )
            bot = self._bot(ex, td)
            out = asyncio.run(rebuild.rebuild_local_state(bot, symbols=["BTC/USDT:USDT", "DOGE/USDT:USDT"]))
            self.assertTrue(out.get("ok"))
            self.assertIn("BTCUSDT", bot.state.positions)
            self.assertEqual(int(out.get("orphans", 0)), 1)
            self.assertEqual(str(out.get("orphans_list", [{}])[0].get("symbol")), "DOGEUSDT")
            self.assertEqual(str(out.get("orphans_list", [{}])[0].get("action")), "CANCEL")
            self.assertEqual(str(out.get("orphans_list", [{}])[0].get("class")), "orphan_entry_order")
            self.assertGreaterEqual(len(ex.cancel_calls), 1)
            ledger_path = Path(td) / "intent_ledger.jsonl"
            self.assertTrue(ledger_path.exists())
            rows = [json.loads(x) for x in ledger_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            self.assertTrue(any(str(r.get("client_order_id") or "") == "ENTRY_DOGE_1" for r in rows))
            self.assertTrue(any(str(r.get("reason") or "") == "rebuild_orphan_canceled" for r in rows))

    def test_rebuild_can_freeze_on_orphans(self):
        with tempfile.TemporaryDirectory() as td:
            ex = _DummyEx(
                positions=[],
                orders=[
                    {
                        "id": "o1",
                        "symbol": "DOGE/USDT:USDT",
                        "status": "open",
                        "type": "LIMIT",
                        "clientOrderId": "ENTRY_DOGE_2",
                        "params": {},
                    }
                ],
                trades=[],
            )
            bot = self._bot(ex, td)
            out = asyncio.run(rebuild.rebuild_local_state(bot, freeze_on_orphans=True, adopt_orphans=False))
            self.assertTrue(bool(out.get("halted", False)))
            self.assertTrue(bool(bot.state.halt))
            self.assertEqual(str(getattr(bot.state, "shutdown_source", "")), "execution.rebuild")
            self.assertEqual(str(getattr(bot.state, "shutdown_reason", "")), "rebuild_orphan_freeze")
            ledger_path = Path(td) / "intent_ledger.jsonl"
            rows = [json.loads(x) for x in ledger_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            self.assertTrue(any(str(r.get("reason") or "") == "rebuild_orphan_frozen" for r in rows))
            self.assertTrue(any(str(r.get("stage") or "") == "OPEN_UNKNOWN" for r in rows))

    def test_rebuild_keeps_protective_reduce_only_with_position(self):
        with tempfile.TemporaryDirectory() as td:
            ex = _DummyEx(
                positions=[{"symbol": "BTC/USDT:USDT", "contracts": 1.0, "side": "long", "entryPrice": 100.0}],
                orders=[
                    {
                        "id": "s1",
                        "symbol": "BTC/USDT:USDT",
                        "status": "open",
                        "type": "STOP_MARKET",
                        "clientOrderId": "SL_BTC_1",
                        "params": {"reduceOnly": True},
                    }
                ],
                trades=[],
            )
            bot = self._bot(ex, td)
            out = asyncio.run(rebuild.rebuild_local_state(bot, freeze_on_orphans=False, adopt_orphans=True))
            self.assertTrue(out.get("ok"))
            self.assertEqual(int(out.get("orphans", 0)), 0)
            self.assertEqual(len(ex.cancel_calls), 0)
            ledger_path = Path(td) / "intent_ledger.jsonl"
            rows = [json.loads(x) for x in ledger_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            self.assertTrue(any(str(r.get("client_order_id") or "") == "SL_BTC_1" for r in rows))
            self.assertTrue(any(str(r.get("reason") or "") == "rebuild_open_order_seen" for r in rows))


if __name__ == "__main__":
    unittest.main()
