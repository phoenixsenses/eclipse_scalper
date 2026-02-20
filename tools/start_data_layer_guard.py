from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Set


def extract_matching_pids(process_rows: Sequence[Dict[str, Any]], module_substring: str) -> List[int]:
    out: List[int] = []
    needle = str(module_substring or "")
    for row in process_rows:
        cl = str((row or {}).get("CommandLine") or "")
        pid = (row or {}).get("ProcessId")
        if not needle or needle not in cl:
            continue
        try:
            out.append(int(pid))
        except Exception:
            continue
    # stable unique
    seen: Set[int] = set()
    uniq: List[int] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def should_start_instance(force_restart: bool, existing_pids: Iterable[int]) -> bool:
    if force_restart:
        return True
    for _ in existing_pids:
        return False
    return True
