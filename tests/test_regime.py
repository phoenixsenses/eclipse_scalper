from __future__ import annotations

import math
import threading

try:
    from core.regime import RegimeClassifier
except ModuleNotFoundError:  # pragma: no cover - fallback for isolated pytest envs
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.regime import RegimeClassifier


def _feed_linear(cls: RegimeClassifier, start_ts: int, n: int, p0: float, dp: float) -> None:
    for i in range(n):
        cls.update(float(start_ts + i), p0 + (dp * i))


def test_steady_uptrend() -> None:
    cls = RegimeClassifier(lookback_sec=3600, debounce_sec=0)
    _feed_linear(cls, start_ts=0, n=8000, p0=100.0, dp=0.001)
    assert cls.current_regime == "UP"
    assert cls.rolling_return > 0.0


def test_steady_downtrend() -> None:
    cls = RegimeClassifier(lookback_sec=3600, debounce_sec=0)
    _feed_linear(cls, start_ts=0, n=8000, p0=200.0, dp=-0.001)
    assert cls.current_regime == "DOWN"
    assert cls.rolling_return < 0.0


def test_flat_market_no_crash() -> None:
    cls = RegimeClassifier(lookback_sec=3600, debounce_sec=0)
    for i in range(8000):
        cls.update(float(i), 100.0 + (0.0001 if (i % 2 == 0) else -0.0001))
    assert cls.current_regime in {"UP", "DOWN"}
    assert math.isfinite(cls.rolling_return)


def test_debounce_prevents_whipsaw() -> None:
    cls = RegimeClassifier(lookback_sec=3600, debounce_sec=60)
    _feed_linear(cls, start_ts=0, n=8000, p0=100.0, dp=0.001)
    assert cls.current_regime == "UP"
    # Brief dip (30 seconds) should not confirm a flip.
    for i in range(8000, 8030):
        cls.update(float(i), 90.0)
    assert cls.current_regime == "TRANSITION"
    for i in range(8030, 8100):
        cls.update(float(i), 120.0)
    assert cls.current_regime == "UP"


def test_debounce_confirms_real_flip() -> None:
    cls = RegimeClassifier(lookback_sec=3600, debounce_sec=60)
    _feed_linear(cls, start_ts=0, n=8000, p0=100.0, dp=0.001)
    assert cls.current_regime == "UP"
    for i in range(8000, 8125):
        cls.update(float(i), 90.0)
    assert cls.current_regime == "DOWN"


def test_transition_state() -> None:
    cls = RegimeClassifier(lookback_sec=3600, debounce_sec=60)
    _feed_linear(cls, start_ts=0, n=8000, p0=100.0, dp=0.001)
    for i in range(8000, 8010):
        cls.update(float(i), 90.0)
    assert cls.current_regime == "TRANSITION"


def test_unknown_insufficient_data() -> None:
    cls = RegimeClassifier(lookback_sec=3600, debounce_sec=0)
    _feed_linear(cls, start_ts=0, n=600, p0=100.0, dp=0.001)
    assert cls.current_regime == "UNKNOWN"
    assert math.isnan(cls.rolling_return)


def test_rolling_return_accuracy() -> None:
    cls = RegimeClassifier(lookback_sec=10, debounce_sec=0)
    prices = [100.0 + i for i in range(25)]
    for i, p in enumerate(prices):
        cls.update(float(i), p)
    expected = math.log(prices[-1] / prices[-11])
    assert abs(cls.rolling_return - expected) < 1e-6


def test_thread_safety() -> None:
    cls = RegimeClassifier(lookback_sec=60, debounce_sec=5)
    errors: list[str] = []

    def _writer() -> None:
        try:
            for i in range(2000):
                cls.update(float(i), 100.0 + (i * 0.01))
        except Exception as e:  # pragma: no cover - regression safety
            errors.append(f"writer:{e}")

    def _reader() -> None:
        try:
            for _ in range(2000):
                _ = cls.current_regime
                _ = cls.regime_age_sec
                _ = cls.rolling_return
                _ = cls.state_dict()
        except Exception as e:  # pragma: no cover - regression safety
            errors.append(f"reader:{e}")

    t1 = threading.Thread(target=_writer)
    t2 = threading.Thread(target=_reader)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert errors == []


def test_state_dict() -> None:
    cls = RegimeClassifier(lookback_sec=10, debounce_sec=2)
    for i in range(30):
        cls.update(float(i), 100.0 + i)
    st = cls.state_dict()
    for k in (
        "source",
        "lookback_sec",
        "debounce_sec",
        "current_regime",
        "raw_regime",
        "confirmed_regime",
        "pending_regime",
        "regime_age_sec",
        "rolling_return",
        "buffer_size",
        "current_ts",
        "last_change_ts",
        "data_ready",
    ):
        assert k in st
