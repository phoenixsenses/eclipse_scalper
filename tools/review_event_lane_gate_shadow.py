from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_gate_events(path: str, last_min: Optional[float] = None) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    file_path = Path(path)
    if not file_path.exists():
        return events
    cutoff_ts: Optional[float] = (time.time() - last_min * 60.0) if last_min else None
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
        if cutoff_ts is not None:
            try:
                ts = float(payload.get("ts") or payload.get("timestamp") or 0.0)
                if ts < cutoff_ts:
                    continue
            except Exception:
                pass
        events.append(payload)
    return events


def build_summary(
    path: str,
    symbol: str = "",
    last_min: Optional[float] = None,
) -> Dict[str, Any]:
    rows = _load_gate_events(path, last_min=last_min)
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
        "last_min": last_min,
        "rows_total": total,
        "allowed_count": allowed,
        "would_block_count": would_block,
        "allowed_rate": round((allowed / total), 4) if total else 0.0,
        "would_block_rate": round((would_block / total), 4) if total else 0.0,
        "blocking_lane_counts": lane_counts,
        "latest": latest_event.get("data") if latest_event else None,
    }


def _print_human(summary: Dict[str, Any]) -> None:
    sym = summary.get("symbol") or "ALL"
    total = summary["rows_total"]
    allowed = summary["allowed_count"]
    would_block = summary["would_block_count"]
    a_rate = summary["allowed_rate"] * 100
    b_rate = summary["would_block_rate"] * 100
    window = f" (last {summary['last_min']:.0f} min)" if summary.get("last_min") else ""
    print(f"Event Lane Gate Shadow Review  symbol={sym}{window}")
    print(f"  total events : {total}")
    print(f"  allowed      : {allowed}  ({a_rate:.1f}%)")
    print(f"  would_block  : {would_block}  ({b_rate:.1f}%)")
    lanes = summary.get("blocking_lane_counts") or {}
    if lanes:
        print(f"  blocking lanes:")
        for lane, cnt in sorted(lanes.items(), key=lambda x: -x[1]):
            print(f"    {lane}: {cnt}")
    latest = summary.get("latest")
    if latest:
        print(f"  latest ts_ms : {latest.get('latest_ts_ms')}  imb={latest.get('latest_abs_imbalance')}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize shadow-mode event lane gate telemetry.")
    parser.add_argument("--telemetry-path", default="logs/telemetry.jsonl")
    parser.add_argument("--symbol", default="")
    parser.add_argument("--last-min", type=float, default=None, help="Only consider events in the last N minutes")
    parser.add_argument("--json", action="store_true", dest="json_out", help="Output raw JSON (default: human-readable)")
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = build_summary(
        path=str(args.telemetry_path),
        symbol=str(args.symbol),
        last_min=args.last_min,
    )
    if args.json_out:
        print(json.dumps(summary, indent=2))
    else:
        _print_human(summary)
    if args.out_json:
        out_path = Path(str(args.out_json))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
