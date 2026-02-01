#!/usr/bin/env python3
"""
Lightweight telemetry dashboard (JSONL).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from eclipse_scalper.tools.telemetry_codes_by_symbol import summarize_codes
except Exception:
    summarize_codes = None  # type: ignore[assignment]

def _load_jsonl(path: Path, limit: int = 5000):
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


def _count(events, key: str):
    out = {}
    for ev in events:
        k = ev.get(key)
        if k is None:
            continue
        out[k] = out.get(k, 0) + 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Telemetry dashboard")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Path to telemetry.jsonl")
    parser.add_argument("--limit", type=int, default=100, help="Max events to load from JSONL")
    parser.add_argument("--codes-per-symbol", action="store_true", help="Display top codes per symbol")
    parser.add_argument("--codes-top", type=int, default=3, help="Top codes per symbol when --codes-per-symbol")
    args = parser.parse_args()

    path = Path(args.path)
    events = _load_jsonl(path, limit=max(1, int(args.limit)))
    print("Telemetry Dashboard (Snapshot)")
    print(f"- File events loaded: {len(events)} from {path}")

    if not events:
        return 0

    by_event = _count(events, "event")
    by_code = _count([ev.get("data", {}) for ev in events], "code")

    print("- Top events:")
    for k, v in sorted(by_event.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"  {k}: {v}")

    if by_code:
        print("- Top codes:")
        for k, v in sorted(by_code.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            print(f"  {k}: {v}")

    exit_events = [ev for ev in events if str(ev.get("event", "")).startswith("exit.")]
    if exit_events:
        print("- Exit events summary:")
        exit_by_event = _count(exit_events, "event")
        exit_by_code = _count([ev.get("data", {}) for ev in exit_events], "code")
        if exit_by_event:
            for k, v in sorted(exit_by_event.items(), key=lambda kv: kv[1], reverse=True):
                print(f"  {k}: {v}")
        if exit_by_code:
            print("  exit codes:")
            for k, v in sorted(exit_by_code.items(), key=lambda kv: kv[1], reverse=True):
                print(f"    {k}: {v}")

    recent = events[-20:]
    print("- Recent (last 20):")
    for ev in recent:
        code = (ev.get("data", {}) or {}).get("code")
        print(f"  {ev.get('ts')} {ev.get('event')} {ev.get('symbol')} {code or ''}")

    if args.codes_per_symbol:
        if callable(summarize_codes):
            per_symbol = summarize_codes(events, top=args.codes_top)
            if per_symbol:
                print("- Top codes per symbol:")
                for sym, codes in sorted(per_symbol.items()):
                    print(f"  {sym}: " + ", ".join([f"{k}={v}" for k, v in codes]))
            else:
                print("- Top codes per symbol: none")
        else:
            print("- Top codes per symbol: helper unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
