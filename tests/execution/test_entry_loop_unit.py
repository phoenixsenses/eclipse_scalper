# tests/execution/test_entry_loop_unit.py
# Comprehensive unit tests for pure/near-pure helpers in execution/entry_loop.py
# Designed to be importable WITHOUT the full bot stack.
"""
Strategy: entry_loop.py wraps all optional imports in try/except, so only the
bare-minimum top-level imports (utils.logging, execution.runtime_helpers, etc.)
need to resolve.  Those modules exist in this repo.  We just import the module
normally and test the pure helper functions directly.

Between tests we reset the mutable module-level dicts (_PENDING_UNTIL, etc.)
so tests are fully isolated.
"""

from __future__ import annotations

import os
import time
import types
import unittest
from typing import Any, Dict, Optional
from unittest import mock

# ---------------------------------------------------------------------------
# Import the module under test (relies on real repo layout on sys.path)
# ---------------------------------------------------------------------------
import execution.entry_loop as _el
import execution.entry_signals as _es

# Convenience aliases for the functions under test
_pending_active = _el._pending_active
_set_pending = _el._set_pending
_evict_stale_pending = _el._evict_stale_pending
_record_partial_fill_hit = _el._record_partial_fill_hit
_reset_partial_fill_hits = _el._reset_partial_fill_hits
_parse_action = _el._parse_action
_parse_order_type = _el._parse_order_type
_parse_amount = _el._parse_amount
_parse_price = _el._parse_price
_tuple_to_sig_dict = _el._tuple_to_sig_dict
_parse_regime_mode = _el._parse_regime_mode
_should_block_regime = _el._should_block_regime
_effective_min_conf = _el._effective_min_conf
_order_filled = _el._order_filled
_guard_block_reason_code = _el._guard_block_reason_code
_parse_groups = _el._parse_groups
_env_float = _el._env_float
_cfg_float = _el._cfg_float
_confidence_notional_scale = _el._confidence_notional_scale
_micro_signal_to_entry_sig = _el._micro_signal_to_entry_sig
_parse_micro_pockets = _el._parse_micro_pockets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot(**overrides):
    """Minimal bot-like object accepted by cfg_env_float / cfg_env_bool."""
    bot = types.SimpleNamespace(
        cfg=types.SimpleNamespace(),
        state=types.SimpleNamespace(
            positions={},
            run_context={},
            current_equity=1000.0,
            telemetry=None,
        ),
        data=None,
        ex=None,
    )
    for k, v in overrides.items():
        setattr(bot, k, v)
    return bot


def _reset_module_state():
    """Clear mutable module-level dicts."""
    _el._PENDING_UNTIL.clear()
    _el._PENDING_ORDER_ID.clear()
    _el._PARTIAL_FILL_STREAK.clear()
    _el._PENDING_LAST_EVICTION_TS = 0.0


# ===========================================================================
# TESTS
# ===========================================================================


class TestEffectiveMinConf(unittest.TestCase):
    """_effective_min_conf(base, guard, adaptive_enabled)"""

    def test_adaptive_disabled_returns_base(self):
        self.assertEqual(_effective_min_conf(0.5, 0.9, adaptive_enabled=False), 0.5)

    def test_adaptive_enabled_returns_max_of_both(self):
        self.assertEqual(_effective_min_conf(0.3, 0.7, adaptive_enabled=True), 0.7)

    def test_adaptive_enabled_base_higher_wins(self):
        self.assertEqual(_effective_min_conf(0.8, 0.5, adaptive_enabled=True), 0.8)

    def test_negative_base_clamped_to_zero(self):
        self.assertEqual(_effective_min_conf(-1.0, 0.0, adaptive_enabled=False), 0.0)

    def test_negative_guard_clamped_to_zero(self):
        # guard negative => max(0.0, -0.5) = 0.0, then max(base, 0.0)
        self.assertEqual(_effective_min_conf(0.3, -0.5, adaptive_enabled=True), 0.3)

    def test_both_zero(self):
        self.assertEqual(_effective_min_conf(0.0, 0.0, adaptive_enabled=True), 0.0)

    def test_adaptive_disabled_ignores_guard(self):
        # Even if guard is very high, disabled means base only
        self.assertEqual(_effective_min_conf(0.1, 0.99, adaptive_enabled=False), 0.1)


class TestParseGroups(unittest.TestCase):
    """_parse_groups(raw) -- parses 'GROUP:SYM1,SYM2;GROUP2:SYM3' format"""

    def test_basic_single_group(self):
        result = _parse_groups("MEME:BTCUSDT,SHIBUSDT")
        self.assertIn("MEME", result)
        self.assertEqual(len(result["MEME"]), 2)

    def test_multiple_groups_semicolon(self):
        result = _parse_groups("MEME:BTCUSDT,SHIBUSDT;MAJOR:ETHUSDT")
        self.assertIn("MEME", result)
        self.assertIn("MAJOR", result)

    def test_empty_string_returns_empty_dict(self):
        self.assertEqual(_parse_groups(""), {})

    def test_none_returns_empty_dict(self):
        self.assertEqual(_parse_groups(None), {})

    def test_no_colon_skips_group(self):
        result = _parse_groups("BTCUSDT,ETHUSDT")
        self.assertEqual(result, {})

    def test_group_name_uppercased(self):
        result = _parse_groups("meme:BTCUSDT")
        self.assertIn("MEME", result)
        self.assertNotIn("meme", result)

    def test_empty_group_name_skipped(self):
        result = _parse_groups(":BTCUSDT")
        self.assertEqual(result, {})

    def test_empty_symbols_skipped(self):
        result = _parse_groups("MEME:")
        # No members -> group not added
        self.assertEqual(result, {})

    def test_whitespace_handling(self):
        result = _parse_groups("  MEME : BTCUSDT , ETHUSDT ; ")
        self.assertIn("MEME", result)


class TestParseAction(unittest.TestCase):
    """_parse_action(sig) -- extracts 'long'/'short' from signal dict"""

    def test_buy(self):
        self.assertEqual(_parse_action({"action": "buy"}), "buy")

    def test_buy_uppercase(self):
        self.assertEqual(_parse_action({"action": "BUY"}), "buy")

    def test_long_maps_to_buy(self):
        self.assertEqual(_parse_action({"action": "long"}), "buy")

    def test_long_uppercase(self):
        self.assertEqual(_parse_action({"action": "LONG"}), "buy")

    def test_sell(self):
        self.assertEqual(_parse_action({"action": "sell"}), "sell")

    def test_sell_uppercase(self):
        self.assertEqual(_parse_action({"action": "SELL"}), "sell")

    def test_short_maps_to_sell(self):
        self.assertEqual(_parse_action({"action": "short"}), "sell")

    def test_short_uppercase(self):
        self.assertEqual(_parse_action({"action": "SHORT"}), "sell")

    def test_side_key_fallback(self):
        self.assertEqual(_parse_action({"side": "long"}), "buy")

    def test_unknown_returns_none(self):
        self.assertIsNone(_parse_action({"action": "hold"}))

    def test_empty_dict(self):
        self.assertIsNone(_parse_action({}))


class TestParseOrderType(unittest.TestCase):
    """_parse_order_type(sig) -- extracts order type, default 'MARKET'"""

    def test_market_default(self):
        self.assertEqual(_parse_order_type({}), "market")

    def test_limit(self):
        self.assertEqual(_parse_order_type({"type": "limit"}), "limit")

    def test_market_explicit(self):
        self.assertEqual(_parse_order_type({"type": "market"}), "market")

    def test_order_type_key(self):
        self.assertEqual(_parse_order_type({"order_type": "limit"}), "limit")

    def test_unknown_type_defaults_to_market(self):
        self.assertEqual(_parse_order_type({"type": "stop_limit"}), "market")


class TestParseAmount(unittest.TestCase):
    """_parse_amount(sig) -- extracts amount float from signal dict"""

    def test_amount_key(self):
        self.assertEqual(_parse_amount({"amount": 1.5}), 1.5)

    def test_qty_key(self):
        self.assertEqual(_parse_amount({"qty": 2.0}), 2.0)

    def test_size_key(self):
        self.assertEqual(_parse_amount({"size": 0.01}), 0.01)

    def test_zero_returns_none(self):
        self.assertIsNone(_parse_amount({"amount": 0}))

    def test_negative_returns_none(self):
        self.assertIsNone(_parse_amount({"amount": -1.0}))

    def test_missing_returns_none(self):
        self.assertIsNone(_parse_amount({}))

    def test_non_numeric_returns_none(self):
        self.assertIsNone(_parse_amount({"amount": "abc"}))


class TestParsePrice(unittest.TestCase):
    """_parse_price(sig) -- extracts price float from signal dict"""

    def test_price_key(self):
        self.assertEqual(_parse_price({"price": 100.5}), 100.5)

    def test_limit_price_key(self):
        self.assertEqual(_parse_price({"limit_price": 200.0}), 200.0)

    def test_zero_returns_none(self):
        self.assertIsNone(_parse_price({"price": 0}))

    def test_negative_returns_none(self):
        self.assertIsNone(_parse_price({"price": -50.0}))

    def test_missing_returns_none(self):
        self.assertIsNone(_parse_price({}))

    def test_non_numeric_returns_none(self):
        self.assertIsNone(_parse_price({"price": "not_a_number"}))


class TestOrderFilled(unittest.TestCase):
    """_order_filled(order) -- extracts filled amount from order dict"""

    def test_filled_key(self):
        self.assertEqual(_order_filled({"filled": 1.5}), 1.5)

    def test_info_executedQty(self):
        self.assertEqual(_order_filled({"info": {"executedQty": "2.3"}}), 2.3)

    def test_none_order(self):
        self.assertEqual(_order_filled(None), 0.0)

    def test_empty_dict(self):
        self.assertEqual(_order_filled({}), 0.0)

    def test_filled_none_value(self):
        self.assertEqual(_order_filled({"filled": None}), 0.0)

    def test_filled_zero(self):
        self.assertEqual(_order_filled({"filled": 0.0}), 0.0)


class TestEnvFloat(unittest.TestCase):
    """_env_float(*names, default) -- reads env vars as float"""

    def test_reads_env_var(self):
        with mock.patch.dict(os.environ, {"TEST_FLOAT_A": "3.14"}):
            self.assertAlmostEqual(_env_float("TEST_FLOAT_A"), 3.14)

    def test_first_match_wins(self):
        with mock.patch.dict(os.environ, {"A1": "1.0", "A2": "2.0"}):
            self.assertAlmostEqual(_env_float("A1", "A2"), 1.0)

    def test_falls_through_to_second(self):
        with mock.patch.dict(os.environ, {"B2": "9.9"}, clear=False):
            # Make sure B1 is not set
            os.environ.pop("B1", None)
            self.assertAlmostEqual(_env_float("B1", "B2"), 9.9)

    def test_default_when_not_set(self):
        os.environ.pop("NONEXISTENT_TEST_VAR", None)
        self.assertEqual(_env_float("NONEXISTENT_TEST_VAR", default=42.0), 42.0)

    def test_default_zero(self):
        os.environ.pop("NONEXISTENT_TEST_VAR2", None)
        self.assertEqual(_env_float("NONEXISTENT_TEST_VAR2"), 0.0)

    def test_invalid_value_skips(self):
        with mock.patch.dict(os.environ, {"BAD_FLOAT": "not_a_float"}):
            self.assertEqual(_env_float("BAD_FLOAT", default=5.0), 5.0)

    def test_empty_string_skips(self):
        with mock.patch.dict(os.environ, {"EMPTY_FLOAT": ""}):
            self.assertEqual(_env_float("EMPTY_FLOAT", default=7.0), 7.0)


class TestCfgFloat(unittest.TestCase):
    """_cfg_float(cfg_obj, *names, default) -- reads cfg attrs as float"""

    def test_reads_attribute(self):
        cfg = types.SimpleNamespace(MY_VALUE=3.5)
        self.assertAlmostEqual(_cfg_float(cfg, "MY_VALUE"), 3.5)

    def test_first_match_wins(self):
        cfg = types.SimpleNamespace(A=1.1, B=2.2)
        self.assertAlmostEqual(_cfg_float(cfg, "A", "B"), 1.1)

    def test_skips_none_values(self):
        cfg = types.SimpleNamespace(A=None, B=5.5)
        self.assertAlmostEqual(_cfg_float(cfg, "A", "B"), 5.5)

    def test_skips_zero_values(self):
        # _cfg_float skips when float(v) == 0.0
        cfg = types.SimpleNamespace(A=0.0, B=8.8)
        self.assertAlmostEqual(_cfg_float(cfg, "A", "B"), 8.8)

    def test_default_when_not_found(self):
        cfg = types.SimpleNamespace()
        self.assertEqual(_cfg_float(cfg, "MISSING", default=99.0), 99.0)

    def test_none_cfg_returns_default(self):
        self.assertEqual(_cfg_float(None, "ANY", default=11.0), 11.0)

    def test_default_zero(self):
        cfg = types.SimpleNamespace()
        self.assertEqual(_cfg_float(cfg, "MISSING"), 0.0)


class TestParseRegimeMode(unittest.TestCase):
    """_parse_regime_mode(raw) -- parses regime mode string"""

    def test_up(self):
        self.assertEqual(_parse_regime_mode("up"), "up")

    def test_up_uppercase(self):
        self.assertEqual(_parse_regime_mode("UP"), "up")

    def test_down(self):
        self.assertEqual(_parse_regime_mode("down"), "down")

    def test_down_mixed_case(self):
        self.assertEqual(_parse_regime_mode("Down"), "down")

    def test_none_string(self):
        self.assertEqual(_parse_regime_mode("none"), "none")

    def test_empty_string(self):
        self.assertEqual(_parse_regime_mode(""), "none")

    def test_none_value(self):
        self.assertEqual(_parse_regime_mode(None), "none")

    def test_unrecognized_string(self):
        self.assertEqual(_parse_regime_mode("sideways"), "none")

    def test_numeric_input(self):
        self.assertEqual(_parse_regime_mode(123), "none")


class TestShouldBlockRegime(unittest.TestCase):
    """_should_block_regime(...) -- checks if regime blocks entry"""

    def test_mode_none_no_block_normal(self):
        blocked, reason = _should_block_regime(
            mode="none", current_regime="UP",
            block_transition=False, block_unknown=False,
        )
        self.assertFalse(blocked)

    def test_block_transition(self):
        blocked, reason = _should_block_regime(
            mode="none", current_regime="TRANSITION",
            block_transition=True, block_unknown=False,
        )
        self.assertTrue(blocked)
        self.assertEqual(reason, "regime_transition")

    def test_block_unknown(self):
        blocked, reason = _should_block_regime(
            mode="none", current_regime="UNKNOWN",
            block_transition=False, block_unknown=True,
        )
        self.assertTrue(blocked)
        self.assertEqual(reason, "regime_unknown")

    def test_unknown_allowed_during_warmup(self):
        blocked, _ = _should_block_regime(
            mode="up", current_regime="UNKNOWN",
            block_transition=False, block_unknown=True,
            warmup_active=True,
        )
        self.assertFalse(blocked)

    def test_unknown_allowed_via_allow_unknown(self):
        blocked, _ = _should_block_regime(
            mode="up", current_regime="UNKNOWN",
            block_transition=False, block_unknown=True,
            allow_unknown=True,
        )
        self.assertFalse(blocked)

    def test_mode_up_current_up_pass(self):
        blocked, _ = _should_block_regime(
            mode="up", current_regime="UP",
            block_transition=False, block_unknown=False,
        )
        self.assertFalse(blocked)

    def test_mode_up_current_down_blocked(self):
        blocked, reason = _should_block_regime(
            mode="up", current_regime="DOWN",
            block_transition=False, block_unknown=False,
        )
        self.assertTrue(blocked)
        self.assertEqual(reason, "regime_mismatch")

    def test_mode_down_current_down_pass(self):
        blocked, _ = _should_block_regime(
            mode="down", current_regime="DOWN",
            block_transition=False, block_unknown=False,
        )
        self.assertFalse(blocked)

    def test_mode_down_current_up_blocked(self):
        blocked, reason = _should_block_regime(
            mode="down", current_regime="UP",
            block_transition=False, block_unknown=False,
        )
        self.assertTrue(blocked)
        self.assertEqual(reason, "regime_mismatch")

    def test_mode_up_transition_blocked(self):
        blocked, reason = _should_block_regime(
            mode="up", current_regime="TRANSITION",
            block_transition=True, block_unknown=False,
        )
        self.assertTrue(blocked)
        self.assertEqual(reason, "regime_transition")


class TestTupleToSigDict(unittest.TestCase):
    """_tuple_to_sig_dict(...) -- converts (long, short, conf) tuple to signal dict"""

    def test_long_signal(self):
        sig = _tuple_to_sig_dict("BTCUSDT", (True, False, 0.85))
        self.assertIsNotNone(sig)
        self.assertEqual(sig["action"], "buy")
        self.assertEqual(sig["confidence"], 0.85)
        self.assertEqual(sig["symbol"], "BTCUSDT")
        self.assertEqual(sig["type"], "market")

    def test_short_signal(self):
        sig = _tuple_to_sig_dict("ETHUSDT", (False, True, 0.70))
        self.assertIsNotNone(sig)
        self.assertEqual(sig["action"], "sell")
        self.assertEqual(sig["confidence"], 0.70)

    def test_both_true_returns_none(self):
        self.assertIsNone(_tuple_to_sig_dict("X", (True, True, 0.9)))

    def test_both_false_returns_none(self):
        self.assertIsNone(_tuple_to_sig_dict("X", (False, False, 0.5)))

    def test_short_tuple_returns_none(self):
        self.assertIsNone(_tuple_to_sig_dict("X", (True, False)))

    def test_not_tuple_returns_none(self):
        self.assertIsNone(_tuple_to_sig_dict("X", "garbage"))

    def test_none_confidence_becomes_zero(self):
        sig = _tuple_to_sig_dict("Z", (True, False, None))
        self.assertIsNotNone(sig)
        self.assertEqual(sig["confidence"], 0.0)

    def test_list_input_works(self):
        sig = _tuple_to_sig_dict("Z", [True, False, 0.5])
        self.assertIsNotNone(sig)
        self.assertEqual(sig["action"], "buy")

    def test_none_input(self):
        self.assertIsNone(_tuple_to_sig_dict("Z", None))


class TestMicroSignalToEntrySig(unittest.TestCase):
    """_micro_signal_to_entry_sig(msig) -- converts MicroSignalResult to entry signal"""

    def test_none_returns_none(self):
        self.assertIsNone(_micro_signal_to_entry_sig(None))

    def test_buy_signal(self):
        features = types.SimpleNamespace(
            mark_price=50000.0,
            imbalance=0.6,
            trade_intensity=4000.0,
            spread=0.0002,
        )
        msig = types.SimpleNamespace(
            side="buy",
            confidence=0.75,
            order_type="limit",
            symbol="BTCUSDT",
            pocket_name="pocket_0",
            features=features,
            regime="UP",
            regime_age_sec=120.0,
            fill_timeout_sec=10.0,
        )
        sig = _micro_signal_to_entry_sig(msig)
        self.assertIsNotNone(sig)
        self.assertEqual(sig["action"], "buy")
        self.assertEqual(sig["confidence"], 0.75)
        self.assertEqual(sig["type"], "limit")
        self.assertEqual(sig["price"], 50000.0)
        self.assertEqual(sig["source"], "micro_signal")
        self.assertEqual(sig["symbol"], "BTCUSDT")

    def test_sell_signal(self):
        features = types.SimpleNamespace(
            mark_price=3000.0,
            imbalance=0.5,
            trade_intensity=2000.0,
            spread=0.0003,
        )
        msig = types.SimpleNamespace(
            side="sell",
            confidence=0.80,
            order_type="market",
            symbol="ETHUSDT",
            pocket_name="pocket_1",
            features=features,
            regime="DOWN",
            regime_age_sec=60.0,
            fill_timeout_sec=5.0,
        )
        sig = _micro_signal_to_entry_sig(msig)
        self.assertIsNotNone(sig)
        self.assertEqual(sig["action"], "sell")
        self.assertEqual(sig["type"], "market")
        self.assertIsNone(sig["price"])  # market order -> no price

    def test_invalid_side_returns_none(self):
        features = types.SimpleNamespace(mark_price=100.0, imbalance=0, trade_intensity=0, spread=0)
        msig = types.SimpleNamespace(
            side="hold",
            confidence=0.5,
            order_type="market",
            symbol="X",
            pocket_name="",
            features=features,
            regime="UNKNOWN",
            regime_age_sec=0,
            fill_timeout_sec=10,
        )
        self.assertIsNone(_micro_signal_to_entry_sig(msig))

    def test_present_false_wrapper_returns_none(self):
        """If msig has present=False (MicroSignalResult wrapper), returns None."""
        msig = types.SimpleNamespace(present=False, reason="no signal", meta={})
        self.assertIsNone(_micro_signal_to_entry_sig(msig))

    def test_present_true_wrapper_unwraps_inner(self):
        """If msig has present=True, it looks inside meta['signal']."""
        features = types.SimpleNamespace(
            mark_price=60000.0, imbalance=0.7, trade_intensity=5000.0, spread=0.0001,
        )
        inner = types.SimpleNamespace(
            side="buy", confidence=0.9, order_type="limit",
            symbol="BTCUSDT", pocket_name="p0", features=features,
            regime="UP", regime_age_sec=30.0, fill_timeout_sec=8.0,
        )
        msig = types.SimpleNamespace(
            present=True, reason="ok",
            meta={"signal": inner},
        )
        sig = _micro_signal_to_entry_sig(msig)
        self.assertIsNotNone(sig)
        self.assertEqual(sig["action"], "buy")
        self.assertEqual(sig["confidence"], 0.9)

    def test_limit_order_with_zero_price_sets_none(self):
        features = types.SimpleNamespace(
            mark_price=0.0, imbalance=0, trade_intensity=0, spread=0,
        )
        msig = types.SimpleNamespace(
            side="buy", confidence=0.5, order_type="limit",
            symbol="X", pocket_name="", features=features,
            regime="UNKNOWN", regime_age_sec=0, fill_timeout_sec=10,
        )
        sig = _micro_signal_to_entry_sig(msig)
        self.assertIsNotNone(sig)
        # price should be None since mark_price is 0
        self.assertIsNone(sig["price"])


class TestConfidenceNotionalScale(unittest.TestCase):
    """_confidence_notional_scale(bot, confidence) -- scales notional by confidence"""

    def test_disabled_returns_1(self):
        bot = _make_bot()
        # Default ENTRY_CONF_SCALE_ENABLED is False
        scale, reason = _confidence_notional_scale(bot, 0.5)
        self.assertEqual(scale, 1.0)
        self.assertEqual(reason, "")

    def test_enabled_mid_confidence(self):
        cfg = types.SimpleNamespace(
            ENTRY_CONF_SCALE_ENABLED=True,
            ENTRY_CONF_SCALE_MIN_CONF=0.0,
            ENTRY_CONF_SCALE_MAX_CONF=1.0,
            ENTRY_CONF_SCALE_MIN=0.5,
            ENTRY_CONF_SCALE_MAX=1.0,
        )
        bot = _make_bot(cfg=cfg)
        scale, reason = _confidence_notional_scale(bot, 0.5)
        # ratio = (0.5 - 0.0) / (1.0 - 0.0) = 0.5
        # scale = 0.5 + 0.5 * (1.0 - 0.5) = 0.75
        self.assertAlmostEqual(scale, 0.75, places=4)

    def test_enabled_below_min_conf(self):
        cfg = types.SimpleNamespace(
            ENTRY_CONF_SCALE_ENABLED=True,
            ENTRY_CONF_SCALE_MIN_CONF=0.5,
            ENTRY_CONF_SCALE_MAX_CONF=1.0,
            ENTRY_CONF_SCALE_MIN=0.3,
            ENTRY_CONF_SCALE_MAX=1.0,
        )
        bot = _make_bot(cfg=cfg)
        scale, reason = _confidence_notional_scale(bot, 0.2)
        self.assertAlmostEqual(scale, 0.3)
        self.assertIn("confidence<", reason)

    def test_enabled_above_max_conf(self):
        cfg = types.SimpleNamespace(
            ENTRY_CONF_SCALE_ENABLED=True,
            ENTRY_CONF_SCALE_MIN_CONF=0.0,
            ENTRY_CONF_SCALE_MAX_CONF=0.8,
            ENTRY_CONF_SCALE_MIN=0.5,
            ENTRY_CONF_SCALE_MAX=1.0,
        )
        bot = _make_bot(cfg=cfg)
        scale, reason = _confidence_notional_scale(bot, 0.95)
        self.assertAlmostEqual(scale, 1.0)


class TestPendingActive(unittest.TestCase):
    """_pending_active / _set_pending"""

    def setUp(self):
        _reset_module_state()

    def tearDown(self):
        _reset_module_state()

    def test_no_entry_returns_false(self):
        self.assertFalse(_pending_active("BTCUSDT"))

    def test_active_after_set(self):
        _set_pending("BTCUSDT", sec=10.0)
        self.assertTrue(_pending_active("BTCUSDT"))

    def test_inactive_after_expiry(self):
        _el._PENDING_UNTIL["ETHUSDT"] = time.time() - 1.0
        self.assertFalse(_pending_active("ETHUSDT"))

    def test_set_pending_stores_order_id(self):
        _set_pending("SOLUSDT", sec=5.0, order_id="ord-123")
        self.assertEqual(_el._PENDING_ORDER_ID["SOLUSDT"], "ord-123")

    def test_set_pending_no_order_id(self):
        _set_pending("SOLUSDT", sec=5.0)
        self.assertNotIn("SOLUSDT", _el._PENDING_ORDER_ID)

    def test_set_pending_negative_sec_clamped(self):
        _set_pending("XRPUSDT", sec=-5.0)
        # max(0.0, -5.0) == 0.0, so _now() + 0 => expires immediately
        self.assertFalse(_pending_active("XRPUSDT"))


class TestEvictStalePending(unittest.TestCase):
    """_evict_stale_pending() -- evicts stale pending entries"""

    def setUp(self):
        _reset_module_state()

    def tearDown(self):
        _reset_module_state()

    def test_removes_stale_entries(self):
        now = time.time()
        _el._PENDING_UNTIL["OLD"] = now - 600  # expired 600s ago (> 300 max age)
        _el._PENDING_ORDER_ID["OLD"] = "ord-old"
        _el._PENDING_UNTIL["FRESH"] = now + 60
        _evict_stale_pending()
        self.assertNotIn("OLD", _el._PENDING_UNTIL)
        self.assertNotIn("OLD", _el._PENDING_ORDER_ID)
        self.assertIn("FRESH", _el._PENDING_UNTIL)

    def test_respects_60s_throttle(self):
        now = time.time()
        _el._PENDING_LAST_EVICTION_TS = now  # just ran
        _el._PENDING_UNTIL["OLD"] = now - 600
        _evict_stale_pending()
        # Should NOT have evicted because we ran < 60s ago
        self.assertIn("OLD", _el._PENDING_UNTIL)

    def test_empty_dict_is_noop(self):
        _evict_stale_pending()  # should not raise


class TestRecordPartialFillHit(unittest.TestCase):
    """_record_partial_fill_hit / _reset_partial_fill_hits"""

    def setUp(self):
        _reset_module_state()

    def tearDown(self):
        _reset_module_state()

    def test_first_hit_no_penalty(self):
        bot = _make_bot()
        penalty = _record_partial_fill_hit(bot, "BTCUSDT")
        self.assertEqual(penalty, 0.0)
        self.assertEqual(_el._PARTIAL_FILL_STREAK["BTCUSDT"]["count"], 1)

    def test_threshold_triggers_penalty(self):
        bot = _make_bot()
        # Default threshold is 3
        _record_partial_fill_hit(bot, "X")
        _record_partial_fill_hit(bot, "X")
        penalty = _record_partial_fill_hit(bot, "X")  # 3rd hit
        self.assertGreater(penalty, 0)  # default penalty is 120.0
        # Count resets after threshold
        self.assertEqual(_el._PARTIAL_FILL_STREAK["X"]["count"], 0)

    def test_reset_clears_streak(self):
        bot = _make_bot()
        _record_partial_fill_hit(bot, "Y")
        _reset_partial_fill_hits("Y")
        self.assertNotIn("Y", _el._PARTIAL_FILL_STREAK)

    def test_window_expiry_resets_count(self):
        bot = _make_bot()
        _record_partial_fill_hit(bot, "Z")
        # Simulate the last_ts being far in the past
        _el._PARTIAL_FILL_STREAK["Z"]["last_ts"] = time.time() - 99999
        _record_partial_fill_hit(bot, "Z")
        # Count should have been reset to 0 then incremented to 1
        self.assertEqual(_el._PARTIAL_FILL_STREAK["Z"]["count"], 1)

    def test_reset_nonexistent_is_noop(self):
        _reset_partial_fill_hits("NEVER_SET")  # should not raise


class TestGuardBlockReasonCode(unittest.TestCase):
    """_guard_block_reason_code(guard_knobs)"""

    def test_runtime_gate_degraded_flag(self):
        reason, code = _guard_block_reason_code({"runtime_gate_degraded": True})
        self.assertEqual(reason, "runtime_gate_reconcile_first")
        self.assertEqual(code, "ERR_RELIABILITY_GATE")

    def test_runtime_gate_in_reason_string(self):
        reason, code = _guard_block_reason_code({
            "reason": "runtime_gate_degraded detected",
            "runtime_gate_degraded": False,
        })
        self.assertEqual(reason, "runtime_gate_reconcile_first")

    def test_reconcile_first_gate_degraded(self):
        reason, code = _guard_block_reason_code({"reconcile_first_gate_degraded": True})
        self.assertEqual(reason, "reconcile_first_pressure")
        self.assertEqual(code, "ERR_RELIABILITY_GATE")

    def test_default_belief_controller(self):
        reason, code = _guard_block_reason_code({})
        self.assertEqual(reason, "belief_controller_block")
        self.assertEqual(code, "ERR_ROUTER_BLOCK")


class TestParseMicroPockets(unittest.TestCase):
    """_parse_micro_pockets(raw) -- parses pocket schedule string"""

    def test_empty_string(self):
        result = _parse_micro_pockets("")
        self.assertEqual(result, [])

    def test_none_string(self):
        result = _parse_micro_pockets(None)
        self.assertEqual(result, [])

    def test_requires_pocket_filter_class(self):
        # If PocketFilter is None (not imported), returns empty
        original = _es.PocketFilter
        try:
            _es.PocketFilter = None
            result = _parse_micro_pockets("0.5,3500,0.0003")
            self.assertEqual(result, [])
        finally:
            _es.PocketFilter = original

    def test_valid_single_pocket(self):
        if _es.PocketFilter is None:
            self.skipTest("PocketFilter not available")
        result = _parse_micro_pockets("0.50,3500,0.000300")
        self.assertEqual(len(result), 1)

    def test_valid_multiple_pockets(self):
        if _es.PocketFilter is None:
            self.skipTest("PocketFilter not available")
        result = _parse_micro_pockets("0.50,3500,0.000300;0.40,2500,0.000500")
        self.assertEqual(len(result), 2)

    def test_insufficient_values_skipped(self):
        if _es.PocketFilter is None:
            self.skipTest("PocketFilter not available")
        result = _parse_micro_pockets("0.50,3500")  # only 2 values, need 3
        self.assertEqual(result, [])

    def test_mixed_valid_and_invalid(self):
        if _es.PocketFilter is None:
            self.skipTest("PocketFilter not available")
        result = _parse_micro_pockets("0.50,3500,0.0003;bad;0.40,2500,0.0005")
        # First and third are valid, second is invalid (no 3 comma-sep values)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
