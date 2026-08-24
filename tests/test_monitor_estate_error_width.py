"""The two roles that actually reach 14 days back must survive what a delete throws.

`7e3a0d4b` widened the ledgers' catch and pre-added `InsufficientHistoryError` "so
landing the coverage assertions cannot silently start killing this role" -- but it
added it to the two ledgers, which never call the coverage code, and left it off
`liq_anomaly_monitor` and `liq_tip_forward`, which reach 14 days back through
`liq_indicator_library` and are the only two that can raise it. The hardening was
applied to the wrong pair.
"""

from __future__ import annotations

import importlib
import sqlite3
import sys

import pytest

from ami.storage.union_reader import InsufficientHistoryError, RotationStateError


ROLES = ["tools.liq_anomaly_monitor", "tools.liq_tip_forward"]

ESTATE_FAILURES = [
    RotationStateError("frozen segment file does not exist"),
    sqlite3.DatabaseError("file is not a database"),        # truncated / mid-replace
    sqlite3.OperationalError("unable to open database"),    # path is a directory
    InsufficientHistoryError("needs 14 days, estate holds 12"),
]


@pytest.mark.parametrize("modname", ROLES)
@pytest.mark.parametrize("exc", ESTATE_FAILURES)
def test_survives_every_failure_a_delete_can_produce(modname, exc, monkeypatch, tmp_path, capsys):
    mod = importlib.import_module(modname)
    monkeypatch.setattr(sys, "argv", [modname, "--once"])
    monkeypatch.setattr(mod, "open_union_ro", lambda *a, **k: (_ for _ in ()).throw(exc))
    mod.main()  # must return, not propagate
    out = capsys.readouterr().out
    assert "ESTATE_UNAVAILABLE" in out
    assert type(exc).__name__ in out, "the operator must see WHICH failure occurred"


@pytest.mark.parametrize("modname", ROLES)
def test_an_error_inside_the_cycle_still_surfaces(modname, monkeypatch, tmp_path):
    """Only the connection is retried; a real bug in the cycle must stay visible."""
    mod = importlib.import_module(modname)
    monkeypatch.setattr(sys, "argv", [modname, "--once"])

    class _Conn:
        def close(self):
            pass

    monkeypatch.setattr(mod, "open_union_ro", lambda *a, **k: _Conn())
    if hasattr(mod, "open_live_ro"):
        monkeypatch.setattr(mod, "open_live_ro", lambda *a, **k: _Conn())
    monkeypatch.setattr(mod, "run_once", lambda *a, **k: (_ for _ in ()).throw(
        sqlite3.OperationalError("no such table: mark_prices")))
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        mod.main()
