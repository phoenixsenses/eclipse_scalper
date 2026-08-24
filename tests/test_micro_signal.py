from __future__ import annotations

import time
from pathlib import Path

try:
    from core.micro_features import MicroFeatures
    from core.micro_signal import MicroSignalConfig, MicroSignalProvider, PocketFilter
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.micro_features import MicroFeatures
    from core.micro_signal import MicroSignalConfig, MicroSignalProvider, PocketFilter


class _Engine:
    def __init__(self, feat: MicroFeatures | None):
        self._feat = feat

    @property
    def current_features(self) -> MicroFeatures | None:
        return self._feat


class _Regime:
    def __init__(self, regime: str = "UP", age: float = 10.0):
        self.current_regime = regime
        self.regime_age_sec = age


def _mk_feat(*, signed: float, intensity: float, spread: float, age: float = 1.0) -> MicroFeatures:
    now = time.time()
    return MicroFeatures(
        timestamp=now - age,
        symbol="ETHUSDT",
        imbalance=abs(signed),
        imbalance_signed=signed,
        trade_intensity=float(intensity),
        spread=float(spread),
        mark_price=100.0,
        age_sec=age,
    )


def test_pocket_match_and_priority_sell() -> None:
    feat = _mk_feat(signed=0.62, intensity=3600, spread=0.0002)
    cfg = MicroSignalConfig(
        pockets=[
            PocketFilter(0.40, 2500, 0.0005, priority=1),
            PocketFilter(0.50, 3500, 0.0003, priority=3),
        ],
        active_sides=["sell"],
        required_regime="up",
    )
    p = MicroSignalProvider(_Engine(feat), _Regime("UP"), cfg)
    res = p.evaluate()
    assert bool(res.present) is True
    assert str(res.reason) == "match"
    assert str(res.side) == "sell"
    assert float(res.confidence) == 1.0


def test_regime_gating_blocks_wrong_regime(monkeypatch) -> None:
    monkeypatch.setenv("MICRO_SIGNAL_REQUIRE_REGIME", "1")
    monkeypatch.setenv("MICRO_SIGNAL_ALLOW_UNKNOWN_REGIME", "0")
    monkeypatch.setenv("MICRO_SIGNAL_REGIME_WARMUP_SEC", "0")
    feat = _mk_feat(signed=0.60, intensity=3600, spread=0.0002)
    cfg = MicroSignalConfig(
        pockets=[PocketFilter(0.50, 3500, 0.0003, priority=1)],
        active_sides=["sell"],
        required_regime="up",
    )
    p = MicroSignalProvider(_Engine(feat), _Regime("DOWN"), cfg)
    res = p.evaluate()
    assert bool(res.present) is False
    assert str(res.reason) == "regime_mismatch"


def test_buy_uses_negative_imbalance_direction() -> None:
    feat = _mk_feat(signed=-0.55, intensity=4000, spread=0.0002)
    cfg = MicroSignalConfig(
        pockets=[PocketFilter(0.50, 3500, 0.0003, priority=1)],
        active_sides=["buy"],
        required_regime="up",
    )
    p = MicroSignalProvider(_Engine(feat), _Regime("UP"), cfg)
    res = p.evaluate()
    assert bool(res.present) is True
    assert str(res.side) == "buy"
    assert float(res.confidence) == 1.0


def test_cooldown_prevents_rapid_resignal() -> None:
    feat = _mk_feat(signed=0.6, intensity=3800, spread=0.0002)
    cfg = MicroSignalConfig(
        pockets=[PocketFilter(0.50, 3500, 0.0003, priority=1)],
        active_sides=["sell"],
        required_regime="up",
        signal_cooldown_sec=60.0,
    )
    p = MicroSignalProvider(_Engine(feat), _Regime("UP"), cfg)
    res1 = p.evaluate()
    res2 = p.evaluate()
    assert bool(res1.present) is True
    assert bool(res2.present) is False
    assert str(res2.reason) == "cooldown"


def test_feature_staleness_blocks_signal() -> None:
    feat = _mk_feat(signed=0.6, intensity=3800, spread=0.0002, age=20.0)
    cfg = MicroSignalConfig(
        pockets=[PocketFilter(0.50, 3500, 0.0003, priority=1)],
        active_sides=["sell"],
        required_regime="up",
        max_feature_age_sec=5.0,
    )
    p = MicroSignalProvider(_Engine(feat), _Regime("UP"), cfg)
    res = p.evaluate()
    assert bool(res.present) is False
    assert str(res.reason) == "stale_features"


def test_binary_confidence_is_zero_when_partial_thresholds_fail() -> None:
    from core import micro_signal as ms

    feat = _mk_feat(signed=0.60, intensity=1000, spread=0.0002)
    pocket = PocketFilter(0.50, 3500, 0.0003, priority=1)
    assert float(ms._confidence(feat, pocket)) == 0.0


def test_unknown_regime_allowed_when_require_regime_disabled(monkeypatch) -> None:
    monkeypatch.setenv("MICRO_SIGNAL_REQUIRE_REGIME", "0")
    monkeypatch.setenv("MICRO_SIGNAL_ALLOW_UNKNOWN_REGIME", "0")
    monkeypatch.setenv("MICRO_SIGNAL_REGIME_WARMUP_SEC", "0")
    feat = _mk_feat(signed=0.60, intensity=3600, spread=0.0002)
    cfg = MicroSignalConfig(
        pockets=[PocketFilter(0.50, 3500, 0.0003, priority=1)],
        active_sides=["sell"],
        required_regime="up",
    )
    p = MicroSignalProvider(_Engine(feat), _Regime("UNKNOWN"), cfg)
    res = p.evaluate()
    assert str(res.reason) == "match"
    assert bool(res.present) is True
    assert float(res.confidence) == 1.0


def test_unknown_regime_allowed_in_warmup(monkeypatch) -> None:
    monkeypatch.setenv("MICRO_SIGNAL_REQUIRE_REGIME", "1")
    monkeypatch.setenv("MICRO_SIGNAL_ALLOW_UNKNOWN_REGIME", "0")
    monkeypatch.setenv("MICRO_SIGNAL_REGIME_WARMUP_SEC", "3600")
    feat = _mk_feat(signed=0.60, intensity=3600, spread=0.0002)
    cfg = MicroSignalConfig(
        pockets=[PocketFilter(0.50, 3500, 0.0003, priority=1)],
        active_sides=["sell"],
        required_regime="up",
    )
    p = MicroSignalProvider(_Engine(feat), _Regime("UNKNOWN"), cfg)
    res = p.evaluate()
    assert str(res.reason) == "match"
    assert bool(res.present) is True


def test_unknown_regime_strict_blocks_with_regime_unknown(monkeypatch) -> None:
    monkeypatch.setenv("MICRO_SIGNAL_REQUIRE_REGIME", "1")
    monkeypatch.setenv("MICRO_SIGNAL_ALLOW_UNKNOWN_REGIME", "0")
    monkeypatch.setenv("MICRO_SIGNAL_REGIME_WARMUP_SEC", "0")
    feat = _mk_feat(signed=0.60, intensity=3600, spread=0.0002)
    cfg = MicroSignalConfig(
        pockets=[PocketFilter(0.50, 3500, 0.0003, priority=1)],
        active_sides=["sell"],
        required_regime="up",
    )
    p = MicroSignalProvider(_Engine(feat), _Regime("UNKNOWN"), cfg)
    res = p.evaluate()
    assert bool(res.present) is False
    assert str(res.reason) == "regime_unknown"
