from __future__ import annotations

import json
from pathlib import Path

try:
    import tools.health_check as hc
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import tools.health_check as hc


def test_health_check_missing_file(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["health_check", "--health", "logs/health/missing_test.json"])
    rc = hc.main()
    out = capsys.readouterr().out
    assert rc == 2
    assert "missing_or_invalid" in out


def test_health_check_ok_and_degraded(monkeypatch, capsys) -> None:
    p = Path("logs/health/test_overall_health.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(json.dumps({"state": "ok", "reason": "", "components": {"collector": {"status": "ok"}}}), encoding="utf-8")
        monkeypatch.setattr("sys.argv", ["health_check", "--health", str(p)])
        assert hc.main() == 0
        p.write_text(json.dumps({"state": "degraded", "reason": "x", "components": {}}), encoding="utf-8")
        monkeypatch.setattr("sys.argv", ["health_check", "--health", str(p)])
        assert hc.main() == 1
    finally:
        p.unlink(missing_ok=True)

