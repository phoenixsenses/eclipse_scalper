#!/usr/bin/env python3
"""
Summarize top error/exit codes per symbol from telemetry JSONL.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Dict, List, Tuple


def _load_jsonl(path: Path, limit: int = 50000):
    if not path.exists():
        return []
    out = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
                if len(out) >= limit:
                    break
    except Exception:
        return out
    return out


def summarize_codes(events: Iterable[dict], *, top: int = 5) -> Dict[str, List[Tuple[str, int]]]:
    per_sym: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ev in events:
        data = ev.get("data") or {}
        code = data.get("code")
        if not code:
            continue
        symbol = ev.get("symbol") or data.get("symbol") or data.get("k")
        if not symbol:
            continue
        per_sym[str(symbol)][str(code)] += 1

    top = max(1, int(top)) if top else 1
    result: Dict[str, List[Tuple[str, int]]] = {}
    for sym, codes in per_sym.items():
        sorted_codes = sorted(codes.items(), key=lambda kv: kv[1], reverse=True)[:top]
        result[sym] = sorted_codes
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Telemetry codes per symbol")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Path to telemetry.jsonl")
    parser.add_argument("--limit", type=int, default=5000, help="Max events to load")
    parser.add_argument("--top", type=int, default=5, help="Top codes per symbol")
    args = parser.parse_args()

    path = Path(args.path)
    events = _load_jsonl(path, limit=max(1, int(args.limit)))
    if not events:
        print(f"No events found at {path}")
        return 0

    top_codes = summarize_codes(events, top=args.top)
    if not top_codes:
        print("No codes found.")
        return 0

    for sym, codes in sorted(top_codes.items()):
        print(f"{sym}: " + ", ".join([f"{k}={v}" for k, v in codes]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
