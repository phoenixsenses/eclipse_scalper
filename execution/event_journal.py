from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return default
        return v
    except Exception:
        return default


def _path_from_bot(bot) -> Path:
    p = ""
    try:
        cfg = getattr(bot, "cfg", None)
        p = str(getattr(cfg, "EVENT_JOURNAL_PATH", "") or "").strip()
    except Exception:
        p = ""
    if not p:
        p = str(os.getenv("EVENT_JOURNAL_PATH", "") or "").strip()
    if not p:
        p = "logs/execution_journal.jsonl"
    return Path(p)


def append_event(bot, event: str, data: Optional[Dict[str, Any]] = None) -> bool:
    try:
        path = _path_from_bot(bot)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": time.time(),
            "event": str(event or "").strip(),
            "data": dict(data or {}),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        return True
    except Exception:
        return False


def journal_transition(
    bot,
    *,
    machine: str,
    entity: str,
    state_from: str,
    state_to: str,
    reason: str = "",
    correlation_id: str = "",
    meta: Optional[Dict[str, Any]] = None,
) -> bool:
    return append_event(
        bot,
        "state.transition",
        {
            "machine": str(machine or ""),
            "entity": str(entity or ""),
            "state_from": str(state_from or ""),
            "state_to": str(state_to or ""),
            "reason": str(reason or ""),
            "correlation_id": str(correlation_id or ""),
            "meta": dict(meta or {}),
            "age_ms": int(_safe_float((meta or {}).get("age_ms"), 0.0)),
        },
    )
