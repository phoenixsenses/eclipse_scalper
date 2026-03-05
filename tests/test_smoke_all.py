from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import smoke_all as sa


def test_smoke_all_skip_db_when_missing() -> None:
    repo_root = Path(".").resolve()
    missing_db = repo_root / "data" / "definitely_missing_for_smoke.db"
    res = sa.run_smoke(repo_root=repo_root, db_path=missing_db)
    assert isinstance(res["checks"], list)
    db_rows = [c for c in res["checks"] if c[0] == "db_check"]
    assert db_rows
    assert db_rows[0][1] is True
    assert "skipped_missing_db" in db_rows[0][2]


def test_smoke_all_exit_code_success(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["x", "--db", "data/definitely_missing_for_smoke.db"])
    rc = sa.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "smoke_all" in out
    assert "PASS" in out

