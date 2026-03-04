from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from tools import freeze_runtime_profile as frp


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"freeze_profile_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_freeze_runtime_profile_write_and_enforce(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        out_json = tmp / "lock.json"
        out_md = tmp / "lock.md"
        monkeypatch.setenv("SCALPER_DRY_RUN", "1")
        monkeypatch.setenv("ACTIVE_SYMBOLS", "ETHUSDT")
        monkeypatch.setattr(
            "sys.argv",
            ["x", "--out-json", str(out_json), "--out-md", str(out_md), "--write-lock"],
        )
        assert frp.main() == 0
        assert out_json.exists()
        assert out_md.exists()
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload.get("hash")

        monkeypatch.setattr(
            "sys.argv",
            ["x", "--out-json", str(out_json), "--out-md", str(out_md), "--enforce"],
        )
        assert frp.main() == 0

        monkeypatch.setenv("ACTIVE_SYMBOLS", "BTCUSDT")
        monkeypatch.setattr(
            "sys.argv",
            ["x", "--out-json", str(out_json), "--out-md", str(out_md), "--enforce"],
        )
        assert frp.main() == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

