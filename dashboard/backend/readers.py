"""Shared fail-closed, bounded read helpers for adapters.

Every helper returns typed status alongside the data so adapters never raise on
missing/malformed input and can populate the trust model uniformly.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from dashboard.backend.view_models import ParseStatus

MAX_JSON_BYTES = 8 * 1024 * 1024      # never load an absurdly large json wholesale
MAX_LEDGER_BYTES = 16 * 1024 * 1024   # bounded ledger scan


def stat_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def load_json(path: Path) -> tuple[Any, ParseStatus, float | None]:
    """Return (data, parse_status, source_mtime)."""
    if not path.exists():
        return None, ParseStatus.MISSING, None
    mtime = stat_mtime(path)
    try:
        size = path.stat().st_size
    except OSError:
        return None, ParseStatus.MISSING, None
    if size == 0:
        return None, ParseStatus.EMPTY, mtime
    if size > MAX_JSON_BYTES:
        # Refuse to load an unexpectedly huge json wholesale.
        return None, ParseStatus.MALFORMED, mtime
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data, ParseStatus.OK, mtime
    except (json.JSONDecodeError, OSError, ValueError):
        return None, ParseStatus.MALFORMED, mtime


def iter_jsonl_tail(path: Path, *, max_bytes: int = MAX_LEDGER_BYTES) -> tuple[list[dict], ParseStatus, float | None]:
    """Bounded read of a JSONL file (tail up to ``max_bytes``). Malformed lines
    are skipped, not fatal. Returns (rows, parse_status, mtime)."""
    if not path.exists():
        return [], ParseStatus.MISSING, None
    mtime = stat_mtime(path)
    try:
        size = path.stat().st_size
    except OSError:
        return [], ParseStatus.MISSING, None
    if size == 0:
        return [], ParseStatus.EMPTY, mtime
    rows: list[dict] = []
    try:
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()  # discard partial first line
            raw = f.read()
        for line in raw.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except (json.JSONDecodeError, ValueError):
                continue
        return rows, ParseStatus.OK, mtime
    except OSError:
        return [], ParseStatus.MALFORMED, mtime


def open_ro(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite DB strictly read-only (mode=ro). Raises if the file is
    missing (URI ro mode does not create)."""
    uri = f"file:{db_path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=2.0)


def parse_iso(ts: Any) -> float | None:
    """Parse an ISO-8601 string to epoch seconds; None on failure."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        import datetime as _dt
        s = str(ts).strip().replace("Z", "+00:00")
        return _dt.datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return None


def load_env_numeric(path: Path, keys: tuple[str, ...]) -> dict[str, float]:
    """Read ONLY the requested numeric keys from a .env file. Never returns
    secret/credential values — only the whitelisted numeric keys asked for."""
    out: dict[str, float] = {}
    if not path.exists():
        return out
    try:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k not in keys:
                continue
            v = v.strip().strip('"').strip("'")
            try:
                out[k] = float(v)
            except ValueError:
                continue
    except OSError:
        return out
    return out
