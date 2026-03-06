from __future__ import annotations

import json
import sys
import shutil
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import reconnection_audit


def _workdir() -> Path:
    p = Path("eclipse_scalper/localtests/phase2_tests") / uuid.uuid4().hex
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_reconnection_audit_writes_report(monkeypatch) -> None:
    wd = _workdir()
    hb = wd / "collector_heartbeat.json"
    out = wd / "RECONNECTION_AUDIT.md"
    hb.write_text(
        json.dumps(
            {
                "connected": True,
                "current_backoff_seconds": 1.5,
                "backend": "websockets",
                "last_error": "",
                "wal_size_mb": 12.0,
                "wal_alert": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["reconnection_audit", "--heartbeat", str(hb), "--out", str(out)],
    )
    try:
        rc = reconnection_audit.main()
        assert rc == 0
        text = out.read_text(encoding="utf-8")
        assert "Reconnection Audit Report" in text
        assert "Connected" in text
    finally:
        shutil.rmtree(wd, ignore_errors=True)
