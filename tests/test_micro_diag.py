from __future__ import annotations

import sys
from pathlib import Path

try:
    from tools import micro_diag
except ModuleNotFoundError:  # pragma: no cover
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools import micro_diag

from tests.fixtures.microstructure import build_collector_schema_fixture, cleanup_temp_path, make_temp_micro_db


def test_micro_diag_smoke(monkeypatch, capsys) -> None:
    db = make_temp_micro_db(prefix="test_micro_diag")
    build_collector_schema_fixture(db, symbols=["BTCUSDT"], rows_per_symbol=20)
    try:
        monkeypatch.setattr(sys, "argv", ["micro_diag", "--db", str(db), "--symbol", "BTCUSDT", "--window-sec", "30"])
        code = micro_diag.main()
        out = capsys.readouterr().out
        assert code in (0, 1)
        assert '"symbol": "BTCUSDT"' in out
        assert '"reason":' in out
    finally:
        cleanup_temp_path(db)

