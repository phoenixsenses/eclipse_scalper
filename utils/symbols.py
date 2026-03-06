from __future__ import annotations

import re


_SEP_RE = re.compile(r"[\s/\-:_]+")


def canonical_symbol(sym: str) -> str:
    if sym is None:
        raise ValueError("symbol is None")
    raw = str(sym).strip().upper()
    if not raw:
        raise ValueError("symbol is empty")
    out = _SEP_RE.sub("", raw)
    if not out:
        raise ValueError("symbol is empty after normalization")
    return out

