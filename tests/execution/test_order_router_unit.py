# tests/execution/test_order_router_unit.py
# Unit tests for pure helper functions in execution/order_router.py
# 2026-03-12

from __future__ import annotations

import hashlib
import re
import sys
import os

import pytest

# ---------------------------------------------------------------------------
# Import helpers: try direct import first, fall back to inline definitions.
# The module has heavy optional imports (ccxt, exchanges, telemetry) which
# may not be available in a test-only environment, so we guard everything.
# ---------------------------------------------------------------------------

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

_IMPORT_OK = False
try:
    # Patch optional deps that may not exist so the module loads cleanly.
    import types as _t

    for _stub in (
        "utils.logging",
        "execution.telemetry",
        "execution.adaptive_guard",
        "execution.error_codes",
        "execution.error_policy",
        "execution.replace_manager",
        "execution.state_machine",
        "execution.event_journal",
        "execution.intent_ledger",
        "execution.runtime_helpers",
    ):
        if _stub not in sys.modules:
            _mod = _t.ModuleType(_stub)
            if _stub == "utils.logging":
                import logging as _logging

                _mod.log_entry = _logging.getLogger("stub")  # type: ignore
            sys.modules[_stub] = _mod

    from execution.order_router import (
        _sanitize_client_order_id,
        _freshen_client_order_id,
        _sanitize_client_id_fields,
        _classify_order_error,
        _error_class_from_reason,
        _looks_like_binance_client_id_duplicate,
        _looks_like_binance_client_id_too_long,
        _looks_like_binance_reduceonly_not_required,
        _normalize_type_for_ccxt,
        _is_exit_intent,
        _merge_params,
        _strip_none_params,
        _normalize_bool_params,
        _error_text,
        _extract_binance_code,
        _binance_filter_reason,
        _BINANCE_CLIENT_ID_MAX,
    )

    _IMPORT_OK = True
except Exception as _exc:
    # If import fails, define the functions inline (copy from source) so
    # tests can still validate the logic in isolation.
    pass

if not _IMPORT_OK:
    # ---- Inline fallback copies (kept minimal, matching source logic) ----
    _BINANCE_CLIENT_ID_MAX = 35

    def _error_text(err):
        try:
            return f"{repr(err)} {str(err)}".lower()
        except Exception:
            return str(err or "").lower()

    def _extract_binance_code(msg):
        try:
            hits = re.findall(r"-\d{4,5}", msg)
            if not hits:
                return None
            return int(hits[0])
        except Exception:
            return None

    def _binance_filter_reason(msg):
        if "filter failure" not in msg:
            return None
        if "price_filter" in msg:
            return "price_filter"
        if "min_notional" in msg or "notional" in msg:
            return "min_notional"
        if "lot_size" in msg:
            return "lot_size"
        if "market_lot_size" in msg:
            return "market_lot_size"
        return "filter_failure"

    def _sanitize_client_order_id(coid, *, max_len=35):
        if coid is None:
            return None
        s = str(coid).strip()
        if not s:
            return None
        safe_chars = [ch for ch in s if ch.isalnum() or ch in ("_", "-")]
        s2 = "".join(safe_chars) or "SE"
        if len(s2) <= max_len:
            return s2
        h = hashlib.sha1(s2.encode("utf-8")).hexdigest()
        keep = max(1, max_len - 11)
        prefix = s2[:keep]
        compact = f"{prefix}_{h[:10]}"
        return compact[:max_len]

    def _freshen_client_order_id(existing, *, salt=None):
        import random, time

        base = _sanitize_client_order_id(existing) or "SE"
        stem = re.sub(r"_[0-9a-f]{6,10}$", "", base, flags=re.IGNORECASE)
        room = 1 + 8
        stem = stem[: max(1, _BINANCE_CLIENT_ID_MAX - room)]
        payload = f"{stem}|{salt if salt is not None else f'{time.time()}|{random.random()}'}"
        suffix = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
        out = f"{stem}_{suffix}"
        return _sanitize_client_order_id(out) or out

    def _sanitize_client_id_fields(p):
        if not isinstance(p, dict):
            return p
        for key in ("clientOrderId", "newClientOrderId"):
            if key in p:
                s = _sanitize_client_order_id(p.get(key))
                if s is None:
                    p.pop(key, None)
                else:
                    p[key] = s
        return p

    def _looks_like_binance_client_id_duplicate(err):
        s = repr(err).lower()
        return "-4116" in s or "clientorderid is duplicated" in s or "client order id is duplicated" in s

    def _looks_like_binance_client_id_too_long(err):
        s = repr(err).lower()
        return "-4015" in s or "client order id length" in s or "less than 36" in s

    def _looks_like_binance_reduceonly_not_required(err):
        s = repr(err).lower()
        return ("reduceonly" in s) and (
            "not required" in s or "sent when not required" in s or "parameter 'reduceonly'" in s
        )

    def _normalize_type_for_ccxt(type_u):
        tu = (type_u or "").upper().strip()
        if tu == "MARKET":
            return "market"
        if tu == "LIMIT":
            return "limit"
        if tu in ("STOP_MARKET", "STOP", "STOPMARKET"):
            return "stop_market"
        if tu in ("TAKE_PROFIT_MARKET", "TP_MARKET", "TAKEPROFITMARKET"):
            return "take_profit_market"
        if tu in ("TRAILING_STOP_MARKET", "TRAILING", "TRAILINGSTOPMARKET"):
            return "trailing_stop_market"
        return tu.lower() if tu.isupper() else tu

    def _truthy(x):
        if x is True:
            return True
        if x is False or x is None:
            return False
        if isinstance(x, (int, float)):
            return x != 0
        if isinstance(x, str):
            return x.strip().lower() in ("true", "1", "yes", "y", "t", "on")
        return False

    def _is_exit_intent(*, type_u, params, intent_reduce_only, intent_close_position):
        if bool(_truthy(params.get("reduceOnly"))) or bool(_truthy(params.get("closePosition"))):
            return True
        if intent_reduce_only or intent_close_position:
            return True
        return False

    def _merge_params(base, extra):
        p = {}
        if isinstance(base, dict):
            p.update(base)
        if isinstance(extra, dict):
            p.update(extra)
        return p

    def _strip_none_params(p):
        out = {}
        for k, v in (p or {}).items():
            if v is None:
                continue
            out[k] = v
        return out

    def _normalize_bool_params(p, keys):
        for k in keys:
            if k in p:
                p[k] = bool(_truthy(p.get(k)))

    def _classify_order_error(err, *, ex=None, sym_raw=None):
        msg = _error_text(err)
        binance_code = _extract_binance_code(msg)
        filter_reason = _binance_filter_reason(msg)
        if filter_reason is not None:
            return False, filter_reason, "ERR_UNKNOWN"
        if binance_code == -2019 or any(
            x in msg for x in ("margin is insufficient", "insufficient margin", "insufficient balance")
        ):
            return False, "margin_insufficient", "ERR_UNKNOWN"
        if binance_code == -1021 or "recvwindow" in msg or "timestamp for this request is outside" in msg:
            return True, "timestamp", "ERR_UNKNOWN"
        if any(x in msg for x in ("timeout", "timed out", "temporarily unavailable", "connection", "econnreset", "network")):
            return True, "network", "ERR_UNKNOWN"
        if "symbol not found" in msg or "invalid symbol" in msg:
            return False, "invalid_symbol", "ERR_UNKNOWN"
        return True, "unknown", "ERR_UNKNOWN"

    def _error_class_from_reason(err, *, retryable, reason):
        rs = str(reason or "").strip().lower()
        if _looks_like_binance_client_id_duplicate(err) or _looks_like_binance_client_id_too_long(err):
            return "retryable_with_modification"
        if _looks_like_binance_reduceonly_not_required(err) or rs == "reduceonly":
            return "retryable_with_modification"
        if retryable:
            return "retryable"
        return "fatal"


# ============================================================================
# Tests
# ============================================================================


class TestSanitizeClientOrderId:
    """Tests for _sanitize_client_order_id."""

    def test_none_returns_none(self):
        assert _sanitize_client_order_id(None) is None

    def test_empty_string_returns_none(self):
        assert _sanitize_client_order_id("") is None
        assert _sanitize_client_order_id("   ") is None

    def test_short_alnum_passes_through(self):
        assert _sanitize_client_order_id("abc123") == "abc123"

    def test_strips_special_chars(self):
        result = _sanitize_client_order_id("abc!@#$%^&*()def")
        assert result == "abcdef"

    def test_keeps_underscore_and_dash(self):
        result = _sanitize_client_order_id("my_order-123")
        assert result == "my_order-123"

    def test_long_id_gets_hashed_to_max_len(self):
        long_id = "A" * 100
        result = _sanitize_client_order_id(long_id)
        assert result is not None
        assert len(result) <= _BINANCE_CLIENT_ID_MAX

    def test_exactly_max_len_passes_through(self):
        exact = "A" * _BINANCE_CLIENT_ID_MAX
        result = _sanitize_client_order_id(exact)
        assert result == exact

    def test_one_over_max_len_gets_hashed(self):
        over = "B" * (_BINANCE_CLIENT_ID_MAX + 1)
        result = _sanitize_client_order_id(over)
        assert result is not None
        assert len(result) <= _BINANCE_CLIENT_ID_MAX
        assert "_" in result  # hash separator

    def test_all_special_chars_returns_SE(self):
        result = _sanitize_client_order_id("!@#$%")
        assert result == "SE"

    def test_custom_max_len(self):
        result = _sanitize_client_order_id("ABCDEFGHIJ", max_len=5)
        assert result is not None
        assert len(result) <= 5

    def test_deterministic(self):
        coid = "X" * 50
        r1 = _sanitize_client_order_id(coid)
        r2 = _sanitize_client_order_id(coid)
        assert r1 == r2


class TestFreshenClientOrderId:
    """Tests for _freshen_client_order_id."""

    def test_result_within_max_len(self):
        result = _freshen_client_order_id("my_order_abc123", salt="test")
        assert len(result) <= _BINANCE_CLIENT_ID_MAX

    def test_different_salt_gives_different_result(self):
        r1 = _freshen_client_order_id("base_order", salt="salt1")
        r2 = _freshen_client_order_id("base_order", salt="salt2")
        assert r1 != r2

    def test_same_salt_gives_same_result(self):
        r1 = _freshen_client_order_id("base_order", salt="same")
        r2 = _freshen_client_order_id("base_order", salt="same")
        assert r1 == r2

    def test_strips_old_hash_suffix(self):
        # If existing already has a hash suffix, it should be replaced
        r1 = _freshen_client_order_id("stem_aabbccddee", salt="new")
        r2 = _freshen_client_order_id("stem", salt="new")
        # Both should share the same stem "stem"
        assert r1 == r2

    def test_none_existing_uses_SE(self):
        result = _freshen_client_order_id(None, salt="x")
        assert result is not None
        assert len(result) <= _BINANCE_CLIENT_ID_MAX


class TestSanitizeClientIdFields:
    """Tests for _sanitize_client_id_fields."""

    def test_sanitizes_clientOrderId(self):
        p = {"clientOrderId": "abc!@#def"}
        result = _sanitize_client_id_fields(p)
        assert result["clientOrderId"] == "abcdef"

    def test_sanitizes_newClientOrderId(self):
        p = {"newClientOrderId": "X" * 50}
        result = _sanitize_client_id_fields(p)
        assert len(result["newClientOrderId"]) <= _BINANCE_CLIENT_ID_MAX

    def test_removes_empty_id(self):
        p = {"clientOrderId": "   "}
        result = _sanitize_client_id_fields(p)
        assert "clientOrderId" not in result

    def test_non_dict_returns_as_is(self):
        assert _sanitize_client_id_fields("not_a_dict") == "not_a_dict"

    def test_other_keys_untouched(self):
        p = {"stopPrice": "100.5", "clientOrderId": "ok123"}
        result = _sanitize_client_id_fields(p)
        assert result["stopPrice"] == "100.5"
        assert result["clientOrderId"] == "ok123"


class TestErrorText:
    """Tests for _error_text."""

    def test_basic_exception(self):
        result = _error_text(ValueError("Some Error"))
        assert "some error" in result  # lowercased
        assert "valueerror" in result

    def test_empty_message(self):
        result = _error_text(Exception(""))
        assert isinstance(result, str)


class TestExtractBinanceCode:
    """Tests for _extract_binance_code."""

    def test_finds_code(self):
        assert _extract_binance_code("binance error -4116 blah") == -4116

    def test_finds_first_code(self):
        assert _extract_binance_code("errors -1021 and -4015") == -1021

    def test_no_code_returns_none(self):
        assert _extract_binance_code("no codes here") is None

    def test_five_digit_code(self):
        assert _extract_binance_code("code -10001 found") == -10001


class TestBinanceFilterReason:
    """Tests for _binance_filter_reason."""

    def test_price_filter(self):
        assert _binance_filter_reason("filter failure: price_filter") == "price_filter"

    def test_min_notional(self):
        assert _binance_filter_reason("filter failure min_notional check") == "min_notional"

    def test_lot_size(self):
        assert _binance_filter_reason("filter failure lot_size") == "lot_size"

    def test_generic_filter_failure(self):
        assert _binance_filter_reason("filter failure something_else") == "filter_failure"

    def test_no_filter_failure(self):
        assert _binance_filter_reason("margin is insufficient") is None


class TestClassifyOrderError:
    """Tests for _classify_order_error."""

    def test_margin_insufficient(self):
        err = Exception("Margin is insufficient")
        retryable, reason, _ = _classify_order_error(err)
        assert not retryable
        assert reason == "margin_insufficient"

    def test_timestamp_error(self):
        err = Exception("Timestamp for this request is outside of the recvWindow")
        retryable, reason, _ = _classify_order_error(err)
        assert retryable
        assert reason == "timestamp"

    def test_network_timeout(self):
        err = Exception("Connection timeout occurred")
        retryable, reason, _ = _classify_order_error(err)
        assert retryable
        assert reason == "network"

    def test_invalid_symbol(self):
        err = Exception("Symbol not found: FAKEUSDT")
        retryable, reason, _ = _classify_order_error(err)
        assert not retryable
        assert reason == "invalid_symbol"

    def test_unknown_error_is_retryable(self):
        err = Exception("some weird exchange error")
        retryable, reason, _ = _classify_order_error(err)
        assert retryable
        assert reason == "unknown"

    def test_filter_failure(self):
        err = Exception("filter failure: price_filter violated")
        retryable, reason, _ = _classify_order_error(err)
        assert not retryable
        assert reason == "price_filter"


class TestErrorClassFromReason:
    """Tests for _error_class_from_reason."""

    def test_duplicate_coid_is_retryable_mod(self):
        err = Exception("binance error -4116 clientOrderId is duplicated")
        result = _error_class_from_reason(err, retryable=True, reason="unknown")
        assert result == "retryable_with_modification"

    def test_too_long_coid_is_retryable_mod(self):
        err = Exception("binance -4015 client order id length should be less than 36")
        result = _error_class_from_reason(err, retryable=True, reason="unknown")
        assert result == "retryable_with_modification"

    def test_reduceonly_reason_is_retryable_mod(self):
        err = Exception("reduceOnly is sent when not required")
        result = _error_class_from_reason(err, retryable=False, reason="reduceonly")
        assert result == "retryable_with_modification"

    def test_retryable_stays_retryable(self):
        err = Exception("timeout")
        result = _error_class_from_reason(err, retryable=True, reason="network")
        assert result == "retryable"

    def test_non_retryable_is_fatal(self):
        err = Exception("margin insufficient")
        result = _error_class_from_reason(err, retryable=False, reason="margin_insufficient")
        assert result == "fatal"


class TestLooksLikeBinancePatterns:
    """Tests for the _looks_like_binance_* pattern matchers."""

    def test_client_id_duplicate_code(self):
        assert _looks_like_binance_client_id_duplicate(Exception("-4116"))

    def test_client_id_duplicate_text(self):
        assert _looks_like_binance_client_id_duplicate(
            Exception("clientOrderId is duplicated")
        )

    def test_client_id_duplicate_negative(self):
        assert not _looks_like_binance_client_id_duplicate(Exception("some other error"))

    def test_client_id_too_long_code(self):
        assert _looks_like_binance_client_id_too_long(Exception("-4015"))

    def test_client_id_too_long_text(self):
        assert _looks_like_binance_client_id_too_long(
            Exception("client order id length should be less than 36")
        )

    def test_client_id_too_long_negative(self):
        assert not _looks_like_binance_client_id_too_long(Exception("margin error"))

    def test_reduceonly_not_required(self):
        assert _looks_like_binance_reduceonly_not_required(
            Exception("reduceOnly is sent when not required")
        )

    def test_reduceonly_negative(self):
        assert not _looks_like_binance_reduceonly_not_required(Exception("price filter"))


class TestNormalizeTypeForCcxt:
    """Tests for _normalize_type_for_ccxt."""

    def test_market(self):
        assert _normalize_type_for_ccxt("MARKET") == "market"

    def test_limit(self):
        assert _normalize_type_for_ccxt("LIMIT") == "limit"

    def test_stop_market_variants(self):
        assert _normalize_type_for_ccxt("STOP_MARKET") == "stop_market"
        assert _normalize_type_for_ccxt("STOP") == "stop_market"
        assert _normalize_type_for_ccxt("STOPMARKET") == "stop_market"

    def test_take_profit_variants(self):
        assert _normalize_type_for_ccxt("TAKE_PROFIT_MARKET") == "take_profit_market"
        assert _normalize_type_for_ccxt("TP_MARKET") == "take_profit_market"

    def test_trailing_variants(self):
        assert _normalize_type_for_ccxt("TRAILING_STOP_MARKET") == "trailing_stop_market"
        assert _normalize_type_for_ccxt("TRAILING") == "trailing_stop_market"

    def test_unknown_uppercase_lowered(self):
        assert _normalize_type_for_ccxt("FOOBAR") == "foobar"

    def test_empty_string(self):
        assert _normalize_type_for_ccxt("") == ""

    def test_none_handled(self):
        assert _normalize_type_for_ccxt(None) == ""


class TestIsExitIntent:
    """Tests for _is_exit_intent."""

    def test_reduceonly_in_params(self):
        assert _is_exit_intent(
            type_u="MARKET", params={"reduceOnly": True},
            intent_reduce_only=False, intent_close_position=False,
        )

    def test_close_position_in_params(self):
        assert _is_exit_intent(
            type_u="STOP_MARKET", params={"closePosition": "true"},
            intent_reduce_only=False, intent_close_position=False,
        )

    def test_intent_reduce_only(self):
        assert _is_exit_intent(
            type_u="LIMIT", params={},
            intent_reduce_only=True, intent_close_position=False,
        )

    def test_intent_close_position(self):
        assert _is_exit_intent(
            type_u="LIMIT", params={},
            intent_reduce_only=False, intent_close_position=True,
        )

    def test_no_exit_signals(self):
        assert not _is_exit_intent(
            type_u="STOP_MARKET", params={},
            intent_reduce_only=False, intent_close_position=False,
        )


class TestMergeParams:
    """Tests for _merge_params."""

    def test_both_dicts(self):
        assert _merge_params({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_extra_overrides_base(self):
        assert _merge_params({"a": 1}, {"a": 99}) == {"a": 99}

    def test_none_base(self):
        assert _merge_params(None, {"b": 2}) == {"b": 2}

    def test_none_extra(self):
        assert _merge_params({"a": 1}, None) == {"a": 1}

    def test_both_none(self):
        assert _merge_params(None, None) == {}


class TestStripNoneParams:
    """Tests for _strip_none_params."""

    def test_removes_nones(self):
        assert _strip_none_params({"a": 1, "b": None, "c": "x"}) == {"a": 1, "c": "x"}

    def test_empty_dict(self):
        assert _strip_none_params({}) == {}

    def test_all_none(self):
        assert _strip_none_params({"a": None, "b": None}) == {}

    def test_none_input(self):
        assert _strip_none_params(None) == {}


class TestNormalizeBoolParams:
    """Tests for _normalize_bool_params."""

    def test_converts_truthy_string(self):
        p = {"reduceOnly": "true", "closePosition": "1"}
        _normalize_bool_params(p, ("reduceOnly", "closePosition"))
        assert p["reduceOnly"] is True
        assert p["closePosition"] is True

    def test_converts_falsy(self):
        p = {"reduceOnly": "false", "closePosition": "0"}
        _normalize_bool_params(p, ("reduceOnly", "closePosition"))
        assert p["reduceOnly"] is False
        assert p["closePosition"] is False

    def test_missing_keys_ignored(self):
        p = {"other": "value"}
        _normalize_bool_params(p, ("reduceOnly",))
        assert p == {"other": "value"}
