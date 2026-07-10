from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_health_", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, separators=(",", ":"))
        os.replace(tmp_name, str(p))
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass


def write_component_health(component: str, status: Dict[str, Any], root: str | Path = "logs/health") -> None:
    """Writes a single component's own dedicated file (e.g. paper_trader.json,
    replay.json). component="overall" is rejected -- logs/health/overall.json
    is the canonical payload owned solely by tools/heartbeat_watchdog.py; no
    other writer, including this generic helper, may produce it. See
    reports/research/s34/CANONICAL_OPERATIONAL_HEALTH_2026-07-10.md."""
    if str(component).strip().lower() == "overall":
        raise ValueError("write_component_health may not target 'overall' -- overall.json is owned solely by tools/heartbeat_watchdog.py")
    payload = dict(status or {})
    payload.setdefault("ts_utc", utc_now_iso())
    atomic_write_json(Path(root) / f"{component}.json", payload)

