#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load(path: Path) -> List[dict]:
    out: List[dict] = []
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


def _matches(ev: dict, correlation_id: str, symbol: str) -> bool:
    data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
    corr = str(data.get("correlation_id") or ev.get("correlation_id") or "")
    entity = str(data.get("entity") or "")
    sym = str(data.get("k") or data.get("symbol") or "")
    if correlation_id:
        if correlation_id == corr or correlation_id == entity:
            return True
    if symbol:
        s = symbol.upper()
        if s in (str(sym).upper(), str(entity).upper()):
            return True
    if not correlation_id and not symbol:
        return True
    return False


def replay(path: Path, *, correlation_id: str = "", symbol: str = "") -> Dict[str, Any]:
    events = _load(path)
    flt = [e for e in events if _matches(e, correlation_id, symbol)]
    transitions = []
    for e in flt:
        if str(e.get("event") or "") != "state.transition":
            continue
        data = e.get("data") if isinstance(e.get("data"), dict) else {}
        transitions.append(
            {
                "ts": e.get("ts"),
                "machine": data.get("machine"),
                "entity": data.get("entity"),
                "from": data.get("state_from"),
                "to": data.get("state_to"),
                "reason": data.get("reason"),
            }
        )
    last_state = transitions[-1]["to"] if transitions else ""
    return {
        "count": len(flt),
        "transitions": transitions,
        "last_state": last_state,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Replay execution journal lifecycle for one trade/symbol")
    ap.add_argument("--path", default="logs/execution_journal.jsonl")
    ap.add_argument("--correlation-id", default="")
    ap.add_argument("--symbol", default="")
    args = ap.parse_args(argv)

    out = replay(Path(args.path), correlation_id=str(args.correlation_id or ""), symbol=str(args.symbol or ""))
    print(f"events: {int(out.get('count', 0))}")
    print(f"last_state: {str(out.get('last_state') or '')}")
    for tr in out.get("transitions", []):
        print(
            f"{tr.get('ts')} | {tr.get('machine')} | {tr.get('entity')} | "
            f"{tr.get('from')} -> {tr.get('to')} | {tr.get('reason')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
