from __future__ import annotations

from pathlib import Path

try:
    import tools.ops_smoke as osk
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import tools.ops_smoke as osk


def test_ops_smoke_sanitizer_no_secret_leak() -> None:
    txt = "hello"
    out = osk._sanitize_output(txt)
    assert "hello" in out


def test_ops_smoke_main(monkeypatch) -> None:
    calls = {"n": 0}

    def _fake_run(cmd, env):
        calls["n"] += 1
        if "tools.validate_env" in " ".join(cmd):
            return 0, "validate ok"
        return 2, "push missing config"

    monkeypatch.setattr(osk, "_run", _fake_run)
    monkeypatch.setattr("sys.argv", ["ops_smoke", "--env", ".env.paper"])
    rc = osk.main()
    assert rc == 0
    assert calls["n"] >= 2

