from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = str(raw or "").strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def replay(path: str | Path, *, correlation_id: str = "", symbol: str = "") -> dict[str, Any]:
    journal_path = Path(path)
    corr = str(correlation_id or "").strip()
    sym = str(symbol or "").strip().upper()
    transitions: list[dict[str, Any]] = []
    for row in _load_jsonl(journal_path):
        if str(row.get("event") or "") != "state.transition":
            continue
        data = row.get("data") if isinstance(row.get("data"), dict) else {}
        row_corr = str(data.get("correlation_id") or data.get("entity") or row.get("correlation_id") or "").strip()
        row_symbol = str(data.get("symbol") or data.get("k") or row.get("symbol") or "").strip().upper()
        if corr and row_corr != corr:
            continue
        if (not corr) and sym and row_symbol and row_symbol != sym:
            continue
        transitions.append(
            {
                "ts": row.get("ts"),
                "machine": str(data.get("machine") or ""),
                "entity": str(data.get("entity") or row_corr or ""),
                "from": str(data.get("state_from") or ""),
                "to": str(data.get("state_to") or ""),
                "reason": str(data.get("reason") or ""),
            }
        )
    last_state = transitions[-1]["to"] if transitions else ""
    return {
        "count": len(transitions),
        "last_state": last_state,
        "transitions": transitions,
    }
