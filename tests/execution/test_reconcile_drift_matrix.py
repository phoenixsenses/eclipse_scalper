# tests/execution/test_reconcile_drift_matrix.py
# Comprehensive unit tests for execution/reconcile.py helper functions
# Focus: drift scenario matrix + pure helper coverage

import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from execution.reconcile import (
    _canonical_to_binance_linear_raw,
    _compute_belief_debt,
    _ensure_reconcile_metrics,
    _evict_stale_phantoms,
    _extract_entry_price,
    _extract_pos_size_side,
    _is_reduce_only,
    _is_stop_like,
    _is_tp_like,
    _make_pos_obj,
    _phantom_store,
    _position_belief_store,
    _record_symbol_mismatch,
    _resolve_raw_symbol,
    _set_position_belief_state,
    _safe_float,
    _truthy,
)


# ---------------------------------------------------------------------------
# Fake bot for tests
# ---------------------------------------------------------------------------

class _FakeBot:
    def __init__(self, positions=None, run_context=None):
        self.state = SimpleNamespace(
            positions=positions or {},
            run_context=run_context if run_context is not None else {},
            current_equity=1000.0,
        )
        self.cfg = SimpleNamespace(
            ACTIVE_SYMBOLS=["BTCUSDT", "ETHUSDT"],
            RECONCILE_STRICT_POSITION_TRANSITIONS=False,
        )


# ===================================================================
# 1. _extract_pos_size_side
# ===================================================================

class TestExtractPosSizeSide(unittest.TestCase):

    def test_contracts_long(self):
        pos = {"contracts": 1.5, "side": "long"}
        size, side = _extract_pos_size_side(pos)
        self.assertAlmostEqual(size, 1.5)
        self.assertEqual(side, "long")

    def test_contracts_short(self):
        pos = {"contracts": 2.0, "side": "short"}
        size, side = _extract_pos_size_side(pos)
        self.assertAlmostEqual(size, 2.0)
        self.assertEqual(side, "short")

    def test_contracts_negative_abs(self):
        pos = {"contracts": -3.0, "side": "short"}
        size, side = _extract_pos_size_side(pos)
        self.assertAlmostEqual(size, 3.0)
        self.assertEqual(side, "short")

    def test_contracts_unknown_side(self):
        pos = {"contracts": 1.0, "side": "hedged"}
        size, side = _extract_pos_size_side(pos)
        self.assertAlmostEqual(size, 1.0)
        self.assertIsNone(side)

    def test_info_position_amt_positive(self):
        pos = {"info": {"positionAmt": "5.0"}}
        size, side = _extract_pos_size_side(pos)
        self.assertAlmostEqual(size, 5.0)
        self.assertEqual(side, "long")

    def test_info_position_amt_negative(self):
        pos = {"info": {"positionAmt": "-2.5"}}
        size, side = _extract_pos_size_side(pos)
        self.assertAlmostEqual(size, 2.5)
        self.assertEqual(side, "short")

    def test_amount_fallback(self):
        pos = {"amount": 7.7}
        size, side = _extract_pos_size_side(pos)
        self.assertAlmostEqual(size, 7.7)
        self.assertIsNone(side)

    def test_empty_position(self):
        size, side = _extract_pos_size_side({})
        self.assertAlmostEqual(size, 0.0)
        self.assertIsNone(side)


# ===================================================================
# 2. _extract_entry_price
# ===================================================================

class TestExtractEntryPrice(unittest.TestCase):

    def test_entry_price_top_level(self):
        self.assertAlmostEqual(_extract_entry_price({"entryPrice": 50000.0}), 50000.0)

    def test_entry_price_snake(self):
        self.assertAlmostEqual(_extract_entry_price({"entry_price": 3000.0}), 3000.0)

    def test_average_field(self):
        self.assertAlmostEqual(_extract_entry_price({"average": 1800.5}), 1800.5)

    def test_info_entry_price(self):
        pos = {"info": {"entryPrice": "42000.1"}}
        self.assertAlmostEqual(_extract_entry_price(pos), 42000.1)

    def test_info_avg_price(self):
        pos = {"info": {"avgPrice": "99.9"}}
        self.assertAlmostEqual(_extract_entry_price(pos), 99.9)

    def test_missing_returns_zero(self):
        self.assertAlmostEqual(_extract_entry_price({}), 0.0)


# ===================================================================
# 3. _is_reduce_only
# ===================================================================

class TestIsReduceOnly(unittest.TestCase):

    def test_top_level_true(self):
        self.assertTrue(_is_reduce_only({"reduceOnly": True}))

    def test_info_reduce_only(self):
        self.assertTrue(_is_reduce_only({"info": {"reduceOnly": "true"}}))

    def test_info_close_position(self):
        self.assertTrue(_is_reduce_only({"info": {"closePosition": "true"}}))

    def test_params_reduce_only(self):
        self.assertTrue(_is_reduce_only({"params": {"reduceOnly": 1}}))

    def test_params_close_position(self):
        self.assertTrue(_is_reduce_only({"params": {"closePosition": "yes"}}))

    def test_not_reduce_only(self):
        self.assertFalse(_is_reduce_only({"reduceOnly": False}))

    def test_empty_order(self):
        self.assertFalse(_is_reduce_only({}))


# ===================================================================
# 4. _is_stop_like
# ===================================================================

class TestIsStopLike(unittest.TestCase):

    def test_stop_market(self):
        self.assertTrue(_is_stop_like({"type": "STOP_MARKET"}))

    def test_stop_limit(self):
        self.assertTrue(_is_stop_like({"type": "stop_limit"}))

    def test_info_type_stop(self):
        self.assertTrue(_is_stop_like({"info": {"type": "STOP_MARKET"}}))

    def test_stop_price_present(self):
        self.assertTrue(_is_stop_like({"stopPrice": 100.0}))

    def test_info_stop_price(self):
        self.assertTrue(_is_stop_like({"info": {"stopPrice": "200"}}))

    def test_limit_order_not_stop(self):
        self.assertFalse(_is_stop_like({"type": "LIMIT"}))

    def test_empty(self):
        self.assertFalse(_is_stop_like({}))


# ===================================================================
# 5. _is_tp_like
# ===================================================================

class TestIsTpLike(unittest.TestCase):

    def test_take_profit_market(self):
        self.assertTrue(_is_tp_like({"type": "TAKE_PROFIT_MARKET"}))

    def test_take_profit_limit(self):
        self.assertTrue(_is_tp_like({"type": "TAKE_PROFIT_LIMIT"}))

    def test_tp_abbreviation(self):
        self.assertTrue(_is_tp_like({"type": "TP"}))

    def test_info_take_profit(self):
        self.assertTrue(_is_tp_like({"info": {"type": "TAKE_PROFIT"}}))

    def test_not_tp(self):
        self.assertFalse(_is_tp_like({"type": "STOP_MARKET"}))

    def test_empty(self):
        self.assertFalse(_is_tp_like({}))


# ===================================================================
# 6. _phantom_store
# ===================================================================

class TestPhantomStore(unittest.TestCase):

    def test_creates_store(self):
        bot = _FakeBot()
        store = _phantom_store(bot)
        self.assertIsInstance(store, dict)
        self.assertIs(store, bot.state.run_context["phantom"])

    def test_returns_existing(self):
        existing = {"BTCUSDT": {"first_ts": 100}}
        bot = _FakeBot(run_context={"phantom": existing})
        store = _phantom_store(bot)
        self.assertIs(store, existing)


# ===================================================================
# 7. _evict_stale_phantoms
# ===================================================================

class TestEvictStalePhantoms(unittest.TestCase):

    def test_empty_dict(self):
        phantom = {}
        _evict_stale_phantoms(phantom)
        self.assertEqual(phantom, {})

    def test_evicts_old_entries(self):
        old_ts = time.time() - 90000  # >24h ago
        phantom = {
            "BTCUSDT": {"first_ts": old_ts},
            "ETHUSDT": {"first_ts": time.time()},
        }
        _evict_stale_phantoms(phantom)
        self.assertNotIn("BTCUSDT", phantom)
        self.assertIn("ETHUSDT", phantom)

    def test_caps_at_500(self):
        now = time.time()
        phantom = {
            f"SYM{i}USDT": {"first_ts": now - i}
            for i in range(520)
        }
        _evict_stale_phantoms(phantom)
        self.assertLessEqual(len(phantom), 500)

    @patch("execution.reconcile._now")
    def test_eviction_with_monkeypatched_time(self, mock_now):
        # Freeze time at T=200000
        mock_now.return_value = 200000.0
        phantom = {
            "OLD": {"first_ts": 100000.0},      # age=100000 > 86400 -> evict
            "RECENT": {"first_ts": 190000.0},    # age=10000 < 86400 -> keep
        }
        _evict_stale_phantoms(phantom)
        self.assertNotIn("OLD", phantom)
        self.assertIn("RECENT", phantom)


# ===================================================================
# 8. _make_pos_obj
# ===================================================================

class TestMakePosObj(unittest.TestCase):

    def test_creates_namespace(self):
        p = _make_pos_obj("BTCUSDT", side="long", size_abs=0.5, entry_price=50000.0)
        self.assertEqual(p.symbol, "BTCUSDT")
        self.assertEqual(p.side, "long")
        self.assertAlmostEqual(p.size, 0.5)
        self.assertAlmostEqual(p.entry_price, 50000.0)

    def test_has_atr_default(self):
        p = _make_pos_obj("ETHUSDT", side="short", size_abs=1.0, entry_price=3000.0)
        self.assertAlmostEqual(p.atr, 0.0)

    def test_none_side(self):
        p = _make_pos_obj("XRPUSDT", side=None, size_abs=100.0, entry_price=0.5)
        self.assertIsNone(p.side)


# ===================================================================
# 9. _position_belief_store
# ===================================================================

class TestPositionBeliefStore(unittest.TestCase):

    def test_creates_store(self):
        bot = _FakeBot()
        store = _position_belief_store(bot)
        self.assertIsInstance(store, dict)
        self.assertIs(store, bot.state.run_context["position_belief_state"])

    def test_returns_existing(self):
        existing = {"BTCUSDT": "LONG"}
        bot = _FakeBot(run_context={"position_belief_state": existing})
        store = _position_belief_store(bot)
        self.assertIs(store, existing)


# ===================================================================
# 10. _set_position_belief_state
# ===================================================================

class TestSetPositionBeliefState(unittest.TestCase):

    def test_sets_state(self):
        bot = _FakeBot()
        _set_position_belief_state(bot, "BTCUSDT", "LONG", "test")
        store = _position_belief_store(bot)
        self.assertEqual(store.get("BTCUSDT"), "LONG")

    def test_noop_same_state(self):
        bot = _FakeBot(run_context={"position_belief_state": {"BTCUSDT": "LONG"}})
        _set_position_belief_state(bot, "BTCUSDT", "LONG", "no-op")
        # Should not raise; state unchanged
        store = _position_belief_store(bot)
        self.assertEqual(store["BTCUSDT"], "LONG")

    def test_noop_empty_symbol(self):
        bot = _FakeBot()
        _set_position_belief_state(bot, "", "LONG", "empty-sym")
        store = _position_belief_store(bot)
        self.assertEqual(len(store), 0)

    def test_transition(self):
        bot = _FakeBot(run_context={"position_belief_state": {"ETHUSDT": "FLAT"}})
        _set_position_belief_state(bot, "ETHUSDT", "SHORT", "signal")
        store = _position_belief_store(bot)
        self.assertEqual(store["ETHUSDT"], "SHORT")


# ===================================================================
# 11. _canonical_to_binance_linear_raw
# ===================================================================

class TestCanonicalToBinanceLinearRaw(unittest.TestCase):

    def test_dogeusdt(self):
        self.assertEqual(_canonical_to_binance_linear_raw("DOGEUSDT"), "DOGE/USDT:USDT")

    def test_btcusdt(self):
        self.assertEqual(_canonical_to_binance_linear_raw("BTCUSDT"), "BTC/USDT:USDT")

    def test_non_usdt_passthrough(self):
        result = _canonical_to_binance_linear_raw("BTCBUSD")
        # Not ending in USDT, so returns the symkey as-is
        self.assertNotIn("/", result)

    def test_bare_usdt(self):
        # Edge case: just "USDT" -> base is empty -> returns as-is
        result = _canonical_to_binance_linear_raw("USDT")
        self.assertEqual(result, "USDT")


# ===================================================================
# 12. _resolve_raw_symbol
# ===================================================================

class TestResolveRawSymbol(unittest.TestCase):

    def test_uses_raw_symbol_map(self):
        bot = _FakeBot()
        bot.data = SimpleNamespace(raw_symbol={"BTCUSDT": "BTC/USDT:USDT"})
        result = _resolve_raw_symbol(bot, "BTCUSDT")
        self.assertEqual(result, "BTC/USDT:USDT")

    def test_fallback_canonical(self):
        bot = _FakeBot()
        # No data, no exchange -> returns canonical
        result = _resolve_raw_symbol(bot, "ETHUSDT")
        self.assertEqual(result, "ETHUSDT")

    def test_futures_heuristic(self):
        bot = _FakeBot()
        inner = SimpleNamespace(options={"defaultType": "future"})
        bot.ex = SimpleNamespace(exchange=inner)
        result = _resolve_raw_symbol(bot, "DOGEUSDT")
        self.assertEqual(result, "DOGE/USDT:USDT")


# ===================================================================
# 13. _ensure_reconcile_metrics
# ===================================================================

class TestEnsureReconcileMetrics(unittest.TestCase):

    def test_creates_metrics(self):
        bot = _FakeBot()
        metrics = _ensure_reconcile_metrics(bot)
        self.assertIsInstance(metrics, dict)
        self.assertEqual(metrics["mismatch_streak"], 0)
        self.assertAlmostEqual(metrics["belief_confidence"], 1.0)

    def test_returns_existing(self):
        bot = _FakeBot()
        bot.state.reconcile_metrics = {"mismatch_streak": 5}
        metrics = _ensure_reconcile_metrics(bot)
        self.assertEqual(metrics["mismatch_streak"], 5)

    def test_no_state_returns_empty(self):
        bot = SimpleNamespace(state=None, cfg=SimpleNamespace())
        metrics = _ensure_reconcile_metrics(bot)
        self.assertEqual(metrics, {})


# ===================================================================
# 14. _record_symbol_mismatch
# ===================================================================

class TestRecordSymbolMismatch(unittest.TestCase):

    def test_creates_entry(self):
        metrics = {"mismatch_by_symbol": {}}
        _record_symbol_mismatch(metrics, "BTCUSDT")
        entry = metrics["mismatch_by_symbol"]["BTCUSDT"]
        self.assertEqual(entry["count"], 1)
        self.assertGreater(entry["first_ts"], 0)

    def test_increments_count(self):
        now = time.time()
        metrics = {"mismatch_by_symbol": {
            "BTCUSDT": {"first_ts": now - 10, "last_ts": now - 5, "count": 3}
        }}
        _record_symbol_mismatch(metrics, "BTCUSDT")
        self.assertEqual(metrics["mismatch_by_symbol"]["BTCUSDT"]["count"], 4)

    def test_empty_symbol_noop(self):
        metrics = {"mismatch_by_symbol": {}}
        _record_symbol_mismatch(metrics, "")
        self.assertEqual(len(metrics["mismatch_by_symbol"]), 0)

    def test_none_metrics_noop(self):
        # Should not raise
        _record_symbol_mismatch(None, "BTCUSDT")


# ===================================================================
# 15. _compute_belief_debt
# ===================================================================

class TestComputeBeliefDebt(unittest.TestCase):

    @patch("execution.reconcile._now")
    def test_active_mismatch(self, mock_now):
        mock_now.return_value = 10000.0
        metrics = {"mismatch_by_symbol": {
            "BTCUSDT": {"first_ts": 9000.0, "last_ts": 9990.0, "count": 2}
        }}
        debt_sec, active = _compute_belief_debt(metrics)
        self.assertAlmostEqual(debt_sec, 1000.0)
        self.assertEqual(active, 1)

    @patch("execution.reconcile._now")
    def test_stale_evicted(self, mock_now):
        mock_now.return_value = 20000.0
        metrics = {"mismatch_by_symbol": {
            "BTCUSDT": {"first_ts": 1000.0, "last_ts": 1000.0, "count": 1}  # >3600s old
        }}
        debt_sec, active = _compute_belief_debt(metrics)
        self.assertAlmostEqual(debt_sec, 0.0)
        self.assertEqual(active, 0)
        self.assertNotIn("BTCUSDT", metrics["mismatch_by_symbol"])

    @patch("execution.reconcile._now")
    def test_clear_set(self, mock_now):
        mock_now.return_value = 10000.0
        metrics = {"mismatch_by_symbol": {
            "BTCUSDT": {"first_ts": 9500.0, "last_ts": 9999.0, "count": 1},
            "ETHUSDT": {"first_ts": 9500.0, "last_ts": 9999.0, "count": 1},
        }}
        debt_sec, active = _compute_belief_debt(metrics, clear={"BTCUSDT"})
        self.assertEqual(active, 1)
        self.assertNotIn("BTCUSDT", metrics["mismatch_by_symbol"])
        self.assertIn("ETHUSDT", metrics["mismatch_by_symbol"])

    def test_none_metrics(self):
        debt_sec, active = _compute_belief_debt(None)
        self.assertAlmostEqual(debt_sec, 0.0)
        self.assertEqual(active, 0)

    def test_empty_mismatch(self):
        metrics = {"mismatch_by_symbol": {}}
        debt_sec, active = _compute_belief_debt(metrics)
        self.assertAlmostEqual(debt_sec, 0.0)
        self.assertEqual(active, 0)


# ===================================================================
# DRIFT SCENARIO MATRIX (parameterized)
# ===================================================================

class TestDriftScenarioMatrix(unittest.TestCase):
    """
    Tests that verify drift detection building blocks by simulating
    bot-state vs exchange-state comparisons using the helper functions.
    """

    def _make_bot_with_position(self, symbol, side, size, entry_price):
        """Helper: create a bot with a single position."""
        bot = _FakeBot()
        pos_obj = _make_pos_obj(symbol, side=side, size_abs=size, entry_price=entry_price)
        bot.state.positions[symbol] = pos_obj
        return bot

    # --- Scenario 1: Both agree (no drift) ---
    def test_no_drift_same_size(self):
        """Bot and exchange have same position -> no drift."""
        bot = self._make_bot_with_position("BTCUSDT", "long", 0.5, 50000.0)
        exchange_pos = {"contracts": 0.5, "side": "long", "entryPrice": 50000.0}
        ex_size, ex_side = _extract_pos_size_side(exchange_pos)
        bot_pos = bot.state.positions["BTCUSDT"]
        self.assertAlmostEqual(ex_size, bot_pos.size)
        self.assertEqual(ex_side, bot_pos.side)

    # --- Scenario 2: Size mismatch ---
    def test_size_mismatch(self):
        """Bot has 0.5, exchange has 0.3 -> drift detected."""
        bot = self._make_bot_with_position("BTCUSDT", "long", 0.5, 50000.0)
        exchange_pos = {"contracts": 0.3, "side": "long"}
        ex_size, _ = _extract_pos_size_side(exchange_pos)
        self.assertNotAlmostEqual(ex_size, bot.state.positions["BTCUSDT"].size)

    # --- Scenario 3: Bot has position, exchange has none (phantom) ---
    def test_phantom_orphan_bot_has_exchange_empty(self):
        """Bot thinks it has a position, exchange says nothing -> phantom."""
        bot = self._make_bot_with_position("ETHUSDT", "short", 1.0, 3000.0)
        exchange_pos = {}
        ex_size, ex_side = _extract_pos_size_side(exchange_pos)
        self.assertAlmostEqual(ex_size, 0.0)
        self.assertIsNone(ex_side)
        # Bot still believes it has a position
        self.assertAlmostEqual(bot.state.positions["ETHUSDT"].size, 1.0)

    # --- Scenario 4: Bot has NO position, exchange has one (untracked) ---
    def test_untracked_position(self):
        """Exchange has position bot doesn't know about."""
        bot = _FakeBot()  # no positions
        exchange_pos = {"contracts": 2.0, "side": "long", "entryPrice": 45000.0}
        ex_size, ex_side = _extract_pos_size_side(exchange_pos)
        self.assertAlmostEqual(ex_size, 2.0)
        self.assertEqual(ex_side, "long")
        self.assertNotIn("BTCUSDT", bot.state.positions)

    # --- Scenario 5: Both have position, different sides (critical) ---
    def test_critical_side_mismatch(self):
        """Bot says long, exchange says short -> critical mismatch."""
        bot = self._make_bot_with_position("BTCUSDT", "long", 0.5, 50000.0)
        exchange_pos = {"info": {"positionAmt": "-0.5"}}
        ex_size, ex_side = _extract_pos_size_side(exchange_pos)
        self.assertAlmostEqual(ex_size, 0.5)
        self.assertEqual(ex_side, "short")
        self.assertEqual(bot.state.positions["BTCUSDT"].side, "long")
        # Side mismatch is critical
        self.assertNotEqual(ex_side, bot.state.positions["BTCUSDT"].side)

    # --- Scenario 6: Multiple positions, mixed states ---
    def test_mixed_positions(self):
        """Multiple symbols: one matches, one drifts, one untracked."""
        bot = _FakeBot()
        bot.state.positions["BTCUSDT"] = _make_pos_obj(
            "BTCUSDT", side="long", size_abs=0.5, entry_price=50000.0
        )
        bot.state.positions["ETHUSDT"] = _make_pos_obj(
            "ETHUSDT", side="short", size_abs=1.0, entry_price=3000.0
        )

        exchange_positions = {
            "BTCUSDT": {"contracts": 0.5, "side": "long"},       # matches
            "ETHUSDT": {"contracts": 2.0, "side": "short"},      # size drift
            "DOGEUSDT": {"contracts": 1000, "side": "long"},     # untracked
        }

        # BTC: no drift
        btc_size, btc_side = _extract_pos_size_side(exchange_positions["BTCUSDT"])
        self.assertAlmostEqual(btc_size, bot.state.positions["BTCUSDT"].size)

        # ETH: size drift
        eth_size, _ = _extract_pos_size_side(exchange_positions["ETHUSDT"])
        self.assertNotAlmostEqual(eth_size, bot.state.positions["ETHUSDT"].size)

        # DOGE: untracked
        doge_size, doge_side = _extract_pos_size_side(exchange_positions["DOGEUSDT"])
        self.assertAlmostEqual(doge_size, 1000.0)
        self.assertNotIn("DOGEUSDT", bot.state.positions)


# ===================================================================
# Bonus: _safe_float and _truthy edge cases
# ===================================================================

class TestSafeFloatAndTruthy(unittest.TestCase):

    def test_safe_float_nan(self):
        self.assertAlmostEqual(_safe_float(float("nan"), 99.0), 99.0)

    def test_safe_float_none(self):
        self.assertAlmostEqual(_safe_float(None, 5.5), 5.5)

    def test_safe_float_string(self):
        self.assertAlmostEqual(_safe_float("3.14", 0.0), 3.14)

    def test_truthy_string_variants(self):
        for val in ("true", "1", "yes", "y", "t", "on", "True", "YES"):
            self.assertTrue(_truthy(val), f"Expected truthy for {val!r}")

    def test_truthy_false_variants(self):
        for val in ("false", "0", "no", "n", "off", "", None, False):
            self.assertFalse(_truthy(val), f"Expected falsy for {val!r}")


if __name__ == "__main__":
    unittest.main()
