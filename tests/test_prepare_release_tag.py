from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from tools import prepare_release_tag as prt


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"prepare_tag_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def test_prepare_release_tag_dry_run(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    cwd = Path.cwd()
    try:
        monkeypatch.chdir(tmp)
        (tmp / ".git").mkdir(parents=True, exist_ok=True)

        responses = {
            ("git", "rev-parse", "--abbrev-ref", "HEAD"): (0, "main"),
            ("git", "rev-parse", "--short", "HEAD"): (0, "abc1234"),
            ("git", "status", "--porcelain"): (0, ""),
        }

        def _fake_run(cmd: list[str]) -> tuple[int, str]:
            return responses.get(tuple(cmd), (1, "unexpected"))

        monkeypatch.setattr(prt, "_run", _fake_run)
        monkeypatch.setattr("sys.argv", ["x", "--tag", "v2026.03.04-test"])
        assert prt.main() == 0
    finally:
        monkeypatch.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_prepare_release_tag_fails_on_dirty_tree(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    cwd = Path.cwd()
    try:
        monkeypatch.chdir(tmp)
        (tmp / ".git").mkdir(parents=True, exist_ok=True)

        responses = {
            ("git", "rev-parse", "--abbrev-ref", "HEAD"): (0, "main"),
            ("git", "rev-parse", "--short", "HEAD"): (0, "abc1234"),
            ("git", "status", "--porcelain"): (0, " M x.py"),
        }

        def _fake_run(cmd: list[str]) -> tuple[int, str]:
            return responses.get(tuple(cmd), (1, "unexpected"))

        monkeypatch.setattr(prt, "_run", _fake_run)
        monkeypatch.setattr("sys.argv", ["x", "--tag", "v2026.03.04-test"])
        assert prt.main() == 1
    finally:
        monkeypatch.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)
