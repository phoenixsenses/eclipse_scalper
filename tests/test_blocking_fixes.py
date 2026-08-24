from __future__ import annotations

import importlib
import os
from pathlib import Path

from core.micro_features import MicroFeatures
from core.micro_signal import MicroSignalConfig, MicroSignalProvider, PocketFilter
from execution.entry_loop import _effective_min_conf


class _FakeFeatureEngine:
    def __init__(self, feat: MicroFeatures | None):
        self.current_features = feat


def _mk_feat(*, signed: float, intensity: float, spread: float, age: float = 1.0) -> MicroFeatures:
    return MicroFeatures(
        timestamp=1.0,
        symbol="ETHUSDT",
        imbalance=abs(float(signed)),
        imbalance_signed=float(signed),
        trade_intensity=float(intensity),
        spread=float(spread),
        mark_price=2500.0,
        age_sec=float(age),
    )


def _mk_provider(feat: MicroFeatures | None) -> MicroSignalProvider:
    cfg = MicroSignalConfig(
        pockets=[PocketFilter(min_imbalance=0.50, min_intensity=3500.0, max_spread=0.0003, priority=10)],
        active_sides=["sell", "buy"],
        required_regime="up",
        max_feature_age_sec=5.0,
        signal_cooldown_sec=0.0,
    )
    return MicroSignalProvider(_FakeFeatureEngine(feat), None, cfg)


def test_settings_min_conf_env_override() -> None:
    old = os.environ.get("ENTRY_MIN_CONFIDENCE")
    try:
        os.environ["ENTRY_MIN_CONFIDENCE"] = "0.00"
        mod = importlib.import_module("config.settings")
        mod = importlib.reload(mod)
        cfg = mod.Config()
        assert float(cfg.MIN_CONFIDENCE) == 0.0
        assert float(cfg.ENTRY_MIN_CONFIDENCE) == 0.0
    finally:
        if old is None:
            os.environ.pop("ENTRY_MIN_CONFIDENCE", None)
        else:
            os.environ["ENTRY_MIN_CONFIDENCE"] = old


def test_adaptive_guard_disable_keeps_base_min_conf() -> None:
    assert _effective_min_conf(0.10, 0.60, adaptive_enabled=True) == 0.60
    assert _effective_min_conf(0.10, 0.60, adaptive_enabled=False) == 0.10


def test_micro_signal_binary_confidence_only() -> None:
    old = os.environ.get("MICRO_SIGNAL_REQUIRE_REGIME")
    try:
        os.environ["MICRO_SIGNAL_REQUIRE_REGIME"] = "0"
        p_ok = _mk_provider(_mk_feat(signed=0.60, intensity=3600.0, spread=0.0002))
        r_ok = p_ok.evaluate(regime_override={"current_regime": "UP", "regime_age_sec": 10.0})
        assert r_ok.present is True
        assert float(r_ok.confidence) == 1.0

        p_no = _mk_provider(_mk_feat(signed=0.60, intensity=1000.0, spread=0.0002))
        r_no = p_no.evaluate(regime_override={"current_regime": "UP", "regime_age_sec": 10.0})
        assert r_no.present is False
        assert float(r_no.confidence) == 0.0
    finally:
        if old is None:
            os.environ.pop("MICRO_SIGNAL_REQUIRE_REGIME", None)
        else:
            os.environ["MICRO_SIGNAL_REQUIRE_REGIME"] = old


def test_bootstrap_dotenv_loader_prefers_env_paper(monkeypatch) -> None:
    from execution import bootstrap as bs

    tmp = Path("localtests") / "dotenv_blocking_fix"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        (tmp / ".env.paper").write_text("ZZ_BLOCKING_FIX_DOTENV=paper\n", encoding="utf-8")
        (tmp / ".env").write_text("ZZ_BLOCKING_FIX_DOTENV=default\n", encoding="utf-8")
        monkeypatch.chdir(tmp)
        monkeypatch.delenv("ZZ_BLOCKING_FIX_DOTENV", raising=False)
        src = bs._load_dotenv_best_effort()
        assert src == ".env.paper"
        assert os.getenv("ZZ_BLOCKING_FIX_DOTENV") == "paper"
    finally:
        for f in (tmp / ".env.paper", tmp / ".env"):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            tmp.rmdir()
        except Exception:
            pass
