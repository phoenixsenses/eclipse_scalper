"""SSE log tailer for Eclipse Scalper Dashboard.

Streams new lines from a log file as Server-Sent Events.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import AsyncIterator

_POLL_INTERVAL = 0.5  # seconds between file checks


async def tail_sse(path: Path, last_n: int = 50) -> AsyncIterator[str]:
    """Async generator that yields SSE-formatted strings.

    Emits the last `last_n` lines on connect, then streams new lines.
    Yields ``data: <line>\\n\\n`` per SSE spec.
    """
    if not path.exists():
        yield f"data: [file not found: {path.name}]\n\n"
        return

    # Seed: tail last_n lines (read only tail to avoid OOM on large files)
    _TAIL_BYTES = 256 * 1024  # 256 KB is enough for last_n lines
    lines: list[str] = []
    try:
        file_size = path.stat().st_size
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            if file_size > _TAIL_BYTES:
                fh.seek(file_size - _TAIL_BYTES)
                fh.readline()  # skip partial first line
            lines = fh.readlines()
    except Exception as exc:
        yield f"data: [read error: {exc}]\n\n"
        return

    for line in lines[-last_n:]:
        yield f"data: {line.rstrip()}\n\n"

    # Stream new lines
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(0, os.SEEK_END)
            while True:
                line = fh.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                else:
                    await asyncio.sleep(_POLL_INTERVAL)
                    # keepalive comment every poll cycle
                    yield ": keepalive\n\n"
    except asyncio.CancelledError:
        return
    except Exception as exc:
        yield f"data: [stream error: {exc}]\n\n"
