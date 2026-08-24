from __future__ import annotations

from pathlib import Path

try:
    import strategies.eclipse_scalper as strat
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import strategies.eclipse_scalper as strat


class _DummyData:
    def __init__(self) -> None:
        self.ohlcv = {"ETHUSDT": [[0, 1, 1, 1, 1, 1]]}


def test_scalper_signal_marks_scorer_exception(monkeypatch) -> None:
    monkeypatch.setenv("SCALPER_DATA_MAX_STALE_SEC", "0")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(strat, "_get_df_flexible", _boom)
    long_sig, short_sig, conf = strat.scalper_signal("ETHUSDT", data=_DummyData(), cfg=None)
    assert long_sig is False
    assert short_sig is False
    assert conf == 0.0
    diag = strat.get_last_signal_diag("ETHUSDT")
    assert str(diag.get("reason")) == "scorer_exception"
