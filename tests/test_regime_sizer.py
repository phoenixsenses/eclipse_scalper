"""Tests for execution.regime_sizer."""
import importlib
import os

import pytest


def _reload():
    import execution.regime_sizer as m
    importlib.reload(m)
    return m


class TestRegimeSizeScale:
    def test_up_buy_default(self):
        m = _reload()
        assert m.regime_size_scale("UP", "buy") == pytest.approx(1.25)

    def test_up_sell_default(self):
        m = _reload()
        assert m.regime_size_scale("UP", "sell") == pytest.approx(1.10)

    def test_down_buy_default(self):
        m = _reload()
        assert m.regime_size_scale("DOWN", "buy") == pytest.approx(0.80)

    def test_down_sell_default(self):
        m = _reload()
        assert m.regime_size_scale("DOWN", "sell") == pytest.approx(1.00)

    def test_no_regime_neutral(self):
        m = _reload()
        assert m.regime_size_scale("", "buy") == pytest.approx(1.00)
        assert m.regime_size_scale("", "sell") == pytest.approx(1.00)

    def test_unknown_regime_neutral(self):
        m = _reload()
        assert m.regime_size_scale("SIDEWAYS", "buy") == pytest.approx(1.00)

    def test_case_insensitive_regime(self):
        m = _reload()
        assert m.regime_size_scale("up", "buy") == pytest.approx(1.25)
        assert m.regime_size_scale("down", "sell") == pytest.approx(1.00)

    def test_case_insensitive_side(self):
        m = _reload()
        assert m.regime_size_scale("UP", "BUY") == pytest.approx(1.25)

    def test_invalid_side_returns_one(self):
        m = _reload()
        assert m.regime_size_scale("UP", "long") == pytest.approx(1.00)

    def test_max_scale_cap(self, monkeypatch):
        monkeypatch.setenv("REGIME_SIZER_UP_BUY", "3.0")
        monkeypatch.setenv("REGIME_SIZER_MAX_SCALE", "1.5")
        m = _reload()
        assert m.regime_size_scale("UP", "buy") == pytest.approx(1.5)

    def test_env_override_up_buy(self, monkeypatch):
        monkeypatch.setenv("REGIME_SIZER_UP_BUY", "1.4")
        m = _reload()
        assert m.regime_size_scale("UP", "buy") == pytest.approx(1.4)

    def test_env_override_down_buy(self, monkeypatch):
        monkeypatch.setenv("REGIME_SIZER_DOWN_BUY", "0.5")
        m = _reload()
        assert m.regime_size_scale("DOWN", "buy") == pytest.approx(0.5)

    def test_disabled_returns_neutral(self, monkeypatch):
        monkeypatch.setenv("REGIME_SIZER_ENABLED", "0")
        monkeypatch.setenv("REGIME_SIZER_UP_BUY", "1.4")
        m = _reload()
        assert m.regime_size_scale("UP", "buy") == pytest.approx(1.0)
