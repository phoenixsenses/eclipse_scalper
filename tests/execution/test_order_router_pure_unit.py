# tests/execution/test_order_router_pure_unit.py
# Pure-function unit tests for execution/order_router.py helpers.
# No external dependencies beyond stdlib. No network, no asyncio.

from __future__ import annotations

import hashlib
import os
import sys
import unittest
from unittest import mock

# ---------------------------------------------------------------------------
# Bootstrap: ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from execution.order_router import (
    _cfg_env,
    _parse_symbol_overrides,
    _parse_group_kv,
    _parse_retry_policy_entry,
    _error_text,
    _extract_binance_code,
    _binance_filter_reason,
    _classify_order_error,
    _clamp,
    _normalize_callback_rate,
    _normalize_pct,
    _to_float_if_possible,
    _merge_params,
    _normalize_type_for_ccxt,
    _is_number_like,
    _strip_none_params,
    _normalize_bool_params,
    _infer_position_side,
    _make_client_order_id,
    _sanitize_client_order_id,
    _is_futures_symbol,
    _looks_like_binance_reduceonly_not_required,
    _looks_like_binance_client_id_duplicate,
    _looks_like_binance_client_id_too_long,
    _looks_like_unknown_order,
    _BINANCE_CLIENT_ID_MAX,
)


# ---------------------------------------------------------------------------
# Fake bot for functions that need a bot object
# ---------------------------------------------------------------------------
class _FakeBot:
    def __init__(self, **cfg_kwargs):
        self.cfg = type("C", (), cfg_kwargs)()
        self.state = type("S", (), {"positions": {}, "run_context": {}})()


# ===========================================================================
# Tests
# ===========================================================================


class TestCfgEnv(unittest.TestCase):
    """_cfg_env(bot, name, default) — config → env → default."""

    def test_reads_from_bot_cfg(self):
        bot = _FakeBot(MY_SETTING="hello")
        self.assertEqual(_cfg_env(bot, "MY_SETTING", "fallback"), "hello")

    def test_falls_back_to_env_var(self):
        bot = _FakeBot()  # no MY_VAR on cfg
        with mock.patch.dict(os.environ, {"MY_VAR": "from_env"}):
            self.assertEqual(_cfg_env(bot, "MY_VAR", "nope"), "from_env")

    def test_returns_default_when_nothing(self):
        bot = _FakeBot()
        # Make sure env var is absent
        os.environ.pop("NONEXISTENT_12345", None)
        self.assertEqual(_cfg_env(bot, "NONEXISTENT_12345", 42), 42)


class TestParseSymbolOverrides(unittest.TestCase):
    """_parse_symbol_overrides(raw) — 'BTCUSDT=0.5,ETHUSDT=0.3' → dict."""

    def test_basic(self):
        result = _parse_symbol_overrides("BTCUSDT=0.5,ETHUSDT=0.3")
        self.assertAlmostEqual(result["BTCUSDT"], 0.5)
        self.assertAlmostEqual(result["ETHUSDT"], 0.3)

    def test_semicolons(self):
        result = _parse_symbol_overrides("BTCUSDT=1;ETHUSDT=2")
        self.assertIn("BTCUSDT", result)
        self.assertIn("ETHUSDT", result)

    def test_none_returns_empty(self):
        self.assertEqual(_parse_symbol_overrides(None), {})

    def test_empty_string_returns_empty(self):
        self.assertEqual(_parse_symbol_overrides(""), {})

    def test_no_equals_skipped(self):
        result = _parse_symbol_overrides("BTCUSDT,ETHUSDT=0.3")
        self.assertNotIn("BTCUSDT", result)
        self.assertAlmostEqual(result["ETHUSDT"], 0.3)


class TestParseGroupKv(unittest.TestCase):
    """_parse_group_kv(raw) — 'MEME=5,MAJOR=10' → dict."""

    def test_basic(self):
        result = _parse_group_kv("MEME=5,MAJOR=10")
        self.assertAlmostEqual(result["MEME"], 5.0)
        self.assertAlmostEqual(result["MAJOR"], 10.0)

    def test_none(self):
        self.assertEqual(_parse_group_kv(None), {})

    def test_empty(self):
        self.assertEqual(_parse_group_kv(""), {})


class TestParseRetryPolicyEntry(unittest.TestCase):
    """_parse_retry_policy_entry(value) — colon-separated 'max:delay:max_delay:extra'."""

    def test_full_parse(self):
        result = _parse_retry_policy_entry("3:0.5:2.0:1")
        self.assertEqual(result["max_attempts"], 3)
        self.assertAlmostEqual(result["base_delay"], 0.5)
        self.assertAlmostEqual(result["max_delay"], 2.0)
        self.assertEqual(result["extra_attempts"], 1)

    def test_single_value(self):
        result = _parse_retry_policy_entry("5")
        self.assertEqual(result["max_attempts"], 5)
        self.assertNotIn("base_delay", result)

    def test_empty_returns_empty(self):
        self.assertEqual(_parse_retry_policy_entry(""), {})

    def test_min_attempts_is_one(self):
        result = _parse_retry_policy_entry("0")
        self.assertEqual(result["max_attempts"], 1)


class TestErrorText(unittest.TestCase):
    """_error_text(err) — normalizes exception to lowercase text."""

    def test_basic_exception(self):
        err = ValueError("Something Failed")
        text = _error_text(err)
        self.assertIn("something failed", text)
        # repr + str, both lowered
        self.assertIn("valueerror", text)

    def test_returns_lowercase(self):
        err = RuntimeError("UPPER CASE")
        text = _error_text(err)
        self.assertEqual(text, text.lower())


class TestExtractBinanceCode(unittest.TestCase):
    """_extract_binance_code(msg) — finds -NNNN codes."""

    def test_finds_code(self):
        self.assertEqual(_extract_binance_code("binance error -4015 something"), -4015)

    def test_finds_five_digit(self):
        self.assertEqual(_extract_binance_code("error -20019 blah"), -20019)

    def test_no_code_returns_none(self):
        self.assertIsNone(_extract_binance_code("no code here"))

    def test_first_code_wins(self):
        self.assertEqual(_extract_binance_code("-1001 and -2002"), -1001)


class TestBinanceFilterReason(unittest.TestCase):
    """_binance_filter_reason(msg) — extracts filter name."""

    def test_no_filter_failure(self):
        self.assertIsNone(_binance_filter_reason("just a normal error"))

    def test_price_filter(self):
        self.assertEqual(_binance_filter_reason("filter failure price_filter"), "price_filter")

    def test_min_notional(self):
        self.assertEqual(_binance_filter_reason("filter failure min_notional"), "min_notional")

    def test_lot_size(self):
        self.assertEqual(_binance_filter_reason("filter failure lot_size"), "lot_size")

    def test_generic_filter_failure(self):
        self.assertEqual(_binance_filter_reason("filter failure unknown_type"), "filter_failure")

    def test_notional_alias(self):
        self.assertEqual(_binance_filter_reason("filter failure notional too low"), "min_notional")


class TestClassifyOrderError(unittest.TestCase):
    """_classify_order_error(err) — (retryable, reason, class)."""

    def test_margin_insufficient(self):
        err = Exception("binance -2019 margin is insufficient")
        retryable, reason, _ = _classify_order_error(err)
        self.assertFalse(retryable)
        self.assertEqual(reason, "margin_insufficient")

    def test_timestamp_error(self):
        err = Exception("binance -1021 recvwindow timestamp")
        retryable, reason, _ = _classify_order_error(err)
        self.assertTrue(retryable)
        self.assertEqual(reason, "timestamp")

    def test_network_error(self):
        err = Exception("connection timeout timed out")
        retryable, reason, _ = _classify_order_error(err)
        self.assertTrue(retryable)
        self.assertEqual(reason, "network")

    def test_invalid_symbol(self):
        err = Exception("symbol not found for XYZUSDT")
        retryable, reason, _ = _classify_order_error(err)
        self.assertFalse(retryable)
        self.assertEqual(reason, "invalid_symbol")


class TestClamp(unittest.TestCase):
    """_clamp(x, lo, hi)."""

    def test_within_range(self):
        self.assertEqual(_clamp(5.0, 1.0, 10.0), 5.0)

    def test_below_lo(self):
        self.assertEqual(_clamp(-1.0, 0.0, 10.0), 0.0)

    def test_above_hi(self):
        self.assertEqual(_clamp(100.0, 0.0, 10.0), 10.0)

    def test_edge_equals(self):
        self.assertEqual(_clamp(5.0, 5.0, 5.0), 5.0)


class TestNormalizeCallbackRate(unittest.TestCase):
    """_normalize_callback_rate(val) — [0.1, 5.0], divides >5 by 100."""

    def test_normal_value(self):
        self.assertAlmostEqual(_normalize_callback_rate(1.5), 1.5)

    def test_too_large_divided(self):
        # 45 → 45/100 = 0.45
        self.assertAlmostEqual(_normalize_callback_rate(45), 0.45)

    def test_clamped_low(self):
        self.assertAlmostEqual(_normalize_callback_rate(0.01), 0.1)

    def test_clamped_high(self):
        # 600 → 600/100 = 6.0 → clamped to 5.0
        self.assertAlmostEqual(_normalize_callback_rate(600), 5.0)

    def test_zero_becomes_min(self):
        self.assertAlmostEqual(_normalize_callback_rate(0), 0.1)


class TestNormalizePct(unittest.TestCase):
    """_normalize_pct(val, default) — >1 treated as percent."""

    def test_fraction_unchanged(self):
        self.assertAlmostEqual(_normalize_pct(0.5, 0.01), 0.5)

    def test_greater_than_one_divided(self):
        # 2 → 2/100 = 0.02
        self.assertAlmostEqual(_normalize_pct(2, 0.01), 0.02)

    def test_negative_returns_zero(self):
        self.assertAlmostEqual(_normalize_pct(-1, 0.01), 0.0)

    def test_zero_returns_zero(self):
        self.assertAlmostEqual(_normalize_pct(0, 0.01), 0.0)


class TestToFloatIfPossible(unittest.TestCase):
    """_to_float_if_possible(x)."""

    def test_int(self):
        self.assertEqual(_to_float_if_possible(5), 5.0)

    def test_float(self):
        self.assertEqual(_to_float_if_possible(3.14), 3.14)

    def test_string_number(self):
        self.assertEqual(_to_float_if_possible("2.5"), 2.5)

    def test_none_returns_none(self):
        self.assertIsNone(_to_float_if_possible(None))

    def test_non_numeric_string_passthrough(self):
        self.assertEqual(_to_float_if_possible("hello"), "hello")

    def test_empty_string_passthrough(self):
        self.assertEqual(_to_float_if_possible(""), "")


class TestMergeParams(unittest.TestCase):
    """_merge_params(base, extra) — merges two dicts."""

    def test_basic_merge(self):
        result = _merge_params({"a": 1}, {"b": 2})
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_extra_overrides_base(self):
        result = _merge_params({"a": 1}, {"a": 99})
        self.assertEqual(result["a"], 99)

    def test_none_base(self):
        result = _merge_params(None, {"b": 2})
        self.assertEqual(result, {"b": 2})

    def test_none_extra(self):
        result = _merge_params({"a": 1}, None)
        self.assertEqual(result, {"a": 1})

    def test_both_none(self):
        self.assertEqual(_merge_params(None, None), {})


class TestNormalizeTypeForCcxt(unittest.TestCase):
    """_normalize_type_for_ccxt(type_u) — internal → ccxt type name."""

    def test_market(self):
        self.assertEqual(_normalize_type_for_ccxt("MARKET"), "market")

    def test_limit(self):
        self.assertEqual(_normalize_type_for_ccxt("LIMIT"), "limit")

    def test_stop_market_variants(self):
        for v in ("STOP_MARKET", "STOP", "STOPMARKET"):
            self.assertEqual(_normalize_type_for_ccxt(v), "stop_market", f"failed for {v}")

    def test_take_profit_variants(self):
        for v in ("TAKE_PROFIT_MARKET", "TP_MARKET", "TAKEPROFITMARKET"):
            self.assertEqual(_normalize_type_for_ccxt(v), "take_profit_market", f"failed for {v}")

    def test_trailing_variants(self):
        for v in ("TRAILING_STOP_MARKET", "TRAILING", "TRAILINGSTOPMARKET"):
            self.assertEqual(_normalize_type_for_ccxt(v), "trailing_stop_market", f"failed for {v}")

    def test_lowercase_passthrough(self):
        self.assertEqual(_normalize_type_for_ccxt("custom_type"), "custom_type")


class TestIsNumberLike(unittest.TestCase):
    """_is_number_like(x)."""

    def test_int(self):
        self.assertTrue(_is_number_like(5))

    def test_float(self):
        self.assertTrue(_is_number_like(3.14))

    def test_string_number(self):
        self.assertTrue(_is_number_like("2.5"))

    def test_non_numeric(self):
        self.assertFalse(_is_number_like("hello"))

    def test_none(self):
        self.assertFalse(_is_number_like(None))


class TestStripNoneParams(unittest.TestCase):
    """_strip_none_params(p) — removes None values."""

    def test_removes_nones(self):
        result = _strip_none_params({"a": 1, "b": None, "c": "x"})
        self.assertEqual(result, {"a": 1, "c": "x"})

    def test_empty_dict(self):
        self.assertEqual(_strip_none_params({}), {})

    def test_all_none(self):
        self.assertEqual(_strip_none_params({"a": None, "b": None}), {})

    def test_none_input(self):
        self.assertEqual(_strip_none_params(None), {})


class TestNormalizeBoolParams(unittest.TestCase):
    """_normalize_bool_params(p, keys) — string bools → actual bools (in-place)."""

    def test_string_true(self):
        p = {"reduceOnly": "true"}
        _normalize_bool_params(p, ("reduceOnly",))
        self.assertIs(p["reduceOnly"], True)

    def test_string_false(self):
        p = {"reduceOnly": "false"}
        _normalize_bool_params(p, ("reduceOnly",))
        self.assertIs(p["reduceOnly"], False)

    def test_missing_key_no_error(self):
        p = {"other": "value"}
        _normalize_bool_params(p, ("reduceOnly",))
        self.assertNotIn("reduceOnly", p)


class TestInferPositionSide(unittest.TestCase):
    """_infer_position_side(side_hint) — 'buy' → 'LONG', etc."""

    def test_long(self):
        self.assertEqual(_infer_position_side("LONG"), "LONG")

    def test_short(self):
        self.assertEqual(_infer_position_side("SHORT"), "SHORT")

    def test_buy_maps_to_long(self):
        self.assertEqual(_infer_position_side("buy"), "LONG")

    def test_sell_maps_to_short(self):
        self.assertEqual(_infer_position_side("sell"), "SHORT")

    def test_none_returns_none(self):
        self.assertIsNone(_infer_position_side(None))

    def test_empty_returns_none(self):
        self.assertIsNone(_infer_position_side(""))

    def test_case_insensitive(self):
        self.assertEqual(_infer_position_side("long"), "LONG")
        self.assertEqual(_infer_position_side("Short"), "SHORT")


class TestMakeClientOrderId(unittest.TestCase):
    """_make_client_order_id(...) — generates ID < 36 chars."""

    def test_under_36_chars(self):
        coid = _make_client_order_id(
            prefix="SE",
            sym_raw="BTC/USDT:USDT",
            type_norm="market",
            side_l="buy",
            amount=0.001,
            price=None,
            stop_price=None,
        )
        self.assertLessEqual(len(coid), _BINANCE_CLIENT_ID_MAX)

    def test_deterministic(self):
        kwargs = dict(
            prefix="SE", sym_raw="BTC/USDT:USDT", type_norm="limit",
            side_l="sell", amount=1.0, price=50000, stop_price=None,
        )
        a = _make_client_order_id(**kwargs)
        b = _make_client_order_id(**kwargs)
        self.assertEqual(a, b)

    def test_different_inputs_differ(self):
        base = dict(
            prefix="SE", sym_raw="BTC/USDT:USDT", type_norm="market",
            side_l="buy", amount=1.0, price=None, stop_price=None,
        )
        a = _make_client_order_id(**base)
        b = _make_client_order_id(**{**base, "amount": 2.0})
        self.assertNotEqual(a, b)


class TestSanitizeClientOrderId(unittest.TestCase):
    """_sanitize_client_order_id(coid) — keeps [A-Za-z0-9_-], truncates."""

    def test_clean_passthrough(self):
        self.assertEqual(_sanitize_client_order_id("ABC_123"), "ABC_123")

    def test_strips_special_chars(self):
        result = _sanitize_client_order_id("A@B#C$D")
        self.assertEqual(result, "ABCD")

    def test_none_returns_none(self):
        self.assertIsNone(_sanitize_client_order_id(None))

    def test_empty_returns_none(self):
        self.assertIsNone(_sanitize_client_order_id(""))

    def test_long_id_truncated(self):
        long_id = "A" * 100
        result = _sanitize_client_order_id(long_id)
        self.assertLessEqual(len(result), _BINANCE_CLIENT_ID_MAX)

    def test_truncated_is_deterministic(self):
        long_id = "ABCDEFGHIJ" * 10
        a = _sanitize_client_order_id(long_id)
        b = _sanitize_client_order_id(long_id)
        self.assertEqual(a, b)


class TestIsFuturesSymbol(unittest.TestCase):
    """_is_futures_symbol(sym_raw) — string heuristics."""

    def test_colon_usdt(self):
        self.assertTrue(_is_futures_symbol("BTC/USDT:USDT"))

    def test_colon_usd(self):
        self.assertTrue(_is_futures_symbol("BTC/USD:USD"))

    def test_perp(self):
        self.assertTrue(_is_futures_symbol("BTCPERP"))

    def test_spot(self):
        self.assertFalse(_is_futures_symbol("BTCUSDT"))

    def test_empty(self):
        self.assertFalse(_is_futures_symbol(""))


class TestLooksLikeReduceOnlyNotRequired(unittest.TestCase):
    """_looks_like_binance_reduceonly_not_required(err)."""

    def test_match(self):
        err = Exception("ReduceOnly not required for this order")
        self.assertTrue(_looks_like_binance_reduceonly_not_required(err))

    def test_no_match(self):
        err = Exception("some other error")
        self.assertFalse(_looks_like_binance_reduceonly_not_required(err))

    def test_sent_when_not_required(self):
        err = Exception("ReduceOnly sent when not required")
        self.assertTrue(_looks_like_binance_reduceonly_not_required(err))


class TestLooksLikeClientIdDuplicate(unittest.TestCase):
    """_looks_like_binance_client_id_duplicate(err)."""

    def test_code_match(self):
        err = Exception("binance error -4116 clientOrderId is duplicated")
        self.assertTrue(_looks_like_binance_client_id_duplicate(err))

    def test_text_match(self):
        err = Exception("Client order id is duplicated")
        self.assertTrue(_looks_like_binance_client_id_duplicate(err))

    def test_no_match(self):
        err = Exception("something else entirely")
        self.assertFalse(_looks_like_binance_client_id_duplicate(err))


class TestLooksLikeClientIdTooLong(unittest.TestCase):
    """_looks_like_binance_client_id_too_long(err)."""

    def test_code_match(self):
        err = Exception("binance error -4015 too long")
        self.assertTrue(_looks_like_binance_client_id_too_long(err))

    def test_text_match(self):
        err = Exception("client order id length should be less than 36")
        self.assertTrue(_looks_like_binance_client_id_too_long(err))

    def test_no_match(self):
        err = Exception("random error")
        self.assertFalse(_looks_like_binance_client_id_too_long(err))


class TestLooksLikeUnknownOrder(unittest.TestCase):
    """_looks_like_unknown_order(err) — idempotent cancel patterns."""

    def test_code_2011(self):
        err = Exception("binance -2011 Unknown order sent")
        self.assertTrue(_looks_like_unknown_order(err))

    def test_unknown_order_text(self):
        err = Exception("unknown order")
        self.assertTrue(_looks_like_unknown_order(err))

    def test_order_not_found(self):
        err = Exception("order not found")
        self.assertTrue(_looks_like_unknown_order(err))

    def test_order_does_not_exist(self):
        err = Exception("order does not exist")
        self.assertTrue(_looks_like_unknown_order(err))

    def test_no_match(self):
        err = Exception("margin insufficient")
        self.assertFalse(_looks_like_unknown_order(err))


if __name__ == "__main__":
    unittest.main()
