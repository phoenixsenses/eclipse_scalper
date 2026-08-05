"""The two sealed forward ledgers must survive a Phase-4 frozen-segment delete.

`open_union_ro` raises `RotationStateError` when a segment listed in
rotation_state.json is missing. Both ledgers used to call it OUTSIDE their loop's
try block, so that raise would end `main()` for good. Nothing supervises or
restarts these roles, and the forward days they would miss cannot be re-mined --
they are the un-burned sample the echo and hold-horizon arms rest on.

These tests never touch the real ledgers: LEDGER/STATE are redirected to tmp_path
and the connection factory raises before any write path is reached.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from ami.storage.union_reader import RotationStateError


LEDGER_MODULES = [
    "tools.research_s34_echo_forward_ledger",
    "tools.research_s34_hold_horizon_forward_ledger",
]


def _isolate(mod, monkeypatch, tmp_path):
    """Point the module's artifacts at tmp_path so a real ledger can never be written."""
    monkeypatch.setattr(mod, "LEDGER", tmp_path / "ledger.jsonl", raising=False)
    monkeypatch.setattr(mod, "STATE", tmp_path / "state.json", raising=False)
    monkeypatch.setattr(sys, "argv", [mod.__name__, "--once"])


@pytest.mark.parametrize("modname", LEDGER_MODULES)
def test_ledger_survives_a_missing_frozen_segment(modname, monkeypatch, tmp_path, capsys):
    mod = importlib.import_module(modname)
    _isolate(mod, monkeypatch, tmp_path)

    def _segment_gone(*_a, **_k):
        raise RotationStateError("frozen segment file does not exist: 'microstructure.db'")

    monkeypatch.setattr(mod, "open_union_ro", _segment_gone)

    mod.main()  # must return, not propagate

    out = capsys.readouterr().out
    assert "ROTATION_STATE_ERROR" in out
    assert "microstructure.db" in out


@pytest.mark.parametrize("modname", LEDGER_MODULES)
def test_the_raise_still_reaches_the_operator(modname, monkeypatch, tmp_path, capsys):
    """Surviving must not mean going quiet -- a swallowed error is its own failure."""
    mod = importlib.import_module(modname)
    _isolate(mod, monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "open_union_ro",
                        lambda *a, **k: (_ for _ in ()).throw(RotationStateError("boom")))
    mod.main()
    assert "boom" in capsys.readouterr().out


@pytest.mark.parametrize("modname", LEDGER_MODULES)
def test_unexpected_errors_are_not_swallowed(modname, monkeypatch, tmp_path):
    """Only rotation-state failures are survivable; a real bug must still surface."""
    mod = importlib.import_module(modname)
    _isolate(mod, monkeypatch, tmp_path)

    def _real_bug(*_a, **_k):
        raise ValueError("a genuine defect")

    monkeypatch.setattr(mod, "open_union_ro", _real_bug)
    with pytest.raises(ValueError, match="a genuine defect"):
        mod.main()


@pytest.mark.parametrize("modname", LEDGER_MODULES)
def test_connection_is_closed_when_the_cycle_succeeds(modname, monkeypatch, tmp_path):
    """The finally-block must not regress into leaking a connection per cycle."""
    mod = importlib.import_module(modname)
    _isolate(mod, monkeypatch, tmp_path)

    closed = []

    class _Conn:
        def close(self):
            closed.append(True)

    monkeypatch.setattr(mod, "open_union_ro", lambda *a, **k: _Conn())
    monkeypatch.setattr(mod, "run_once", lambda conn, st: (0, 0, 0, 0)[: _ARITY[modname]])
    mod.main()
    assert closed == [True]


_ARITY = {
    "tools.research_s34_echo_forward_ledger": 3,
    "tools.research_s34_hold_horizon_forward_ledger": 4,
}
