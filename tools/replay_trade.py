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


def _symkey(sym: str) -> str:
    s = (sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _matches(ev: dict, correlation_id: str, symbol: str) -> bool:
    data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
    corr = str(data.get("correlation_id") or ev.get("correlation_id") or "")
    entity = str(data.get("entity") or "")
    ev_sym = str(
        data.get("k")
        or data.get("symbol")
        or data.get("symbol_key")
        or data.get("sym")
        or ev.get("symbol")
        or ev.get("symbol_key")
        or ""
    )
    if correlation_id:
        if correlation_id == corr or correlation_id == entity:
            return True
    if symbol:
        want = _symkey(symbol)
        got = _symkey(ev_sym)
        ent = _symkey(entity)
        if want and want in (got, ent):
            return True
        # Fallback for entity strings like "POSITION:BTCUSDT"
        if want and want in str(entity).upper():
            return True
    if not correlation_id and not symbol:
        return True
    return False


def replay(path: Path, *, correlation_id: str = "", symbol: str = "") -> Dict[str, Any]:
    events = _load(path)
    flt = [e for e in events if _matches(e, correlation_id, symbol)]
    transitions = []
    position_transitions = []
    for e in flt:
        if str(e.get("event") or "") != "state.transition":
            continue
        data = e.get("data") if isinstance(e.get("data"), dict) else {}
        row = {
            "ts": e.get("ts"),
            "machine": data.get("machine"),
            "entity": data.get("entity"),
            "from": data.get("state_from"),
            "to": data.get("state_to"),
            "reason": data.get("reason"),
        }
        transitions.append(row)
        if str(data.get("machine") or "").strip().lower() == "position_belief":
            position_transitions.append(row)
    # For symbol-scoped replay, prefer the authoritative position-belief lifecycle
    # when present to avoid unrelated order-intent noise dominating last_state.
    if symbol and (not correlation_id) and position_transitions:
        last_state = position_transitions[-1]["to"] if position_transitions else ""
    else:
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
