from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_gate_events(path: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    file_path = Path(path)
    if not file_path.exists():
        return events
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            payload = json.loads(raw_line)
        except Exception:
            continue
        if str(payload.get("event") or "") != "entry.event_lane_gate":
            continue
        events.append(payload)
    return events


def build_summary(path: str, symbol: str = "") -> Dict[str, Any]:
    rows = _load_gate_events(path)
    if symbol:
        rows = [
            row
            for row in rows
            if str(row.get("symbol") or row.get("data", {}).get("symbol") or "").upper() == symbol.upper()
        ]

    decisions = {"allowed": 0, "would_block": 0, "blocked": 0}
    lane_counts: Dict[str, int] = {}
    latest_event: Dict[str, Any] | None = None
    for row in rows:
        data = row.get("data") or {}
        decision = str(data.get("decision") or "allowed")
        if decision in decisions:
            decisions[decision] += 1
        for lane in list(data.get("blocking_lanes") or []):
            lane_key = str(lane)
            lane_counts[lane_key] = lane_counts.get(lane_key, 0) + 1
        latest_event = row

    total = len(rows)
    allowed = decisions["allowed"]
    would_block = decisions["would_block"] + decisions["blocked"]
    return {
        "telemetry_path": str(path),
        "symbol": symbol.upper() if symbol else "",
        "rows_total": total,
        "allowed_count": allowed,
        "would_block_count": would_block,
        "allowed_rate": (allowed / total) if total else 0.0,
        "would_block_rate": (would_block / total) if total else 0.0,
        "blocking_lane_counts": lane_counts,
        "latest": latest_event.get("data") if latest_event else None,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize shadow-mode event lane gate telemetry.")
    parser.add_argument("--telemetry-path", default="logs/telemetry.jsonl")
    parser.add_argument("--symbol", default="")
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = build_summary(path=str(args.telemetry_path), symbol=str(args.symbol))
    if args.out_json:
        out_path = Path(str(args.out_json))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
