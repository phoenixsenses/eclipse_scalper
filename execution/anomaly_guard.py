"""
Anomaly guard that tracks telemetry anomalies and pauses new entries.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


_STATE_PATH = Path(os.getenv("TELEMETRY_ANOMALY_STATE", "logs/telemetry_anomaly_state.json"))


def _load_state() -> dict[str, float]:
    if not _STATE_PATH.exists():
        return {}
    try:
        with _STATE_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def should_pause() -> tuple[bool, float]:
    state = _load_state()
    now = time.time()
    pause_until = float(state.get("pause_until") or 0.0)
    if pause_until > now:
        return True, pause_until - now
    return False, 0.0
