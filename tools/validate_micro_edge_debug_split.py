from __future__ import annotations

"""
Validate that debug trade events are split cleanly by side.

Example:
  python -m tools.validate_micro_edge_debug_split --long logs/dbg_long.jsonl --short logs/dbg_short.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _event_key(row: Dict) -> str:
    return f"{row.get('symbol')}|{row.get('ts_bucket')}|{row.get('signal_idx')}"


def validate_split(long_rows: Iterable[Dict], short_rows: Iterable[Dict]) -> Dict[str, int]:
    long_keys: Set[str] = {_event_key(r) for r in long_rows}
    short_keys: Set[str] = {_event_key(r) for r in short_rows}
    inter = long_keys & short_keys
    return {
        "long_count": len(long_keys),
        "short_count": len(short_keys),
        "intersection_count": len(inter),
        "union_count": len(long_keys | short_keys),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate long/short debug stream split.")
    p.add_argument("--long", default="logs/dbg_long.jsonl")
    p.add_argument("--short", default="logs/dbg_short.jsonl")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    long_rows = _load_jsonl(Path(args.long))
    short_rows = _load_jsonl(Path(args.short))
    stats = validate_split(long_rows, short_rows)
    print(
        f"debug_split long={stats['long_count']} short={stats['short_count']} "
        f"intersection={stats['intersection_count']} union={stats['union_count']}"
    )
    if stats["intersection_count"] > 0:
        print("FAIL debug_split_overlap_detected")
        return 2
    print("OK debug_split_clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

