from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.run_summary import build_run_summary


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_history_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_history_record(*, history_path: Path, record: Dict[str, Any], max_history: int = 0) -> Dict[str, int]:
    rows = _load_history_rows(history_path)
    rows.append(record)
    trimmed_rows = 0
    if max_history > 0 and len(rows) > max_history:
        trimmed_rows = len(rows) - max_history
        rows = rows[-max_history:]
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    return {"history_rows": len(rows), "trimmed_rows": trimmed_rows}


def _snapshot_record(payload: Dict[str, Any], source: str) -> Dict[str, Any]:
    summary = dict(payload.get("summary") or {})
    top_event = dict(payload.get("top_event") or {})
    banner = dict(payload.get("banner") or {})
    lanes = list(payload.get("lanes") or [])
    run_summary = dict(payload.get("run_summary") or {})
    return {
        "source": str(source),
        "top_lane": str(summary.get("top_lane") or top_event.get("lane") or ""),
        "state_counts": dict(summary.get("state_counts") or {}),
        "lanes": [
            {
                "lane": str(lane.get("lane") or ""),
                "priority_score": float(lane.get("priority_score") or 0.0),
                "level": str(lane.get("level") or "quiet"),
                "freshness_status": str(lane.get("freshness_status") or "stale"),
                "recommended_action": str(lane.get("recommended_action") or "monitor_only"),
            }
            for lane in lanes
            if str(lane.get("lane") or "")
        ],
        "top_event": {
            "lane": str(top_event.get("lane") or ""),
            "level": str(top_event.get("level") or "quiet"),
            "recommended_action": str(top_event.get("recommended_action") or "monitor_only"),
            "headline": str(top_event.get("headline") or ""),
        },
        "banner": {
            "headline": str(banner.get("headline") or ""),
            "recommended_action": str(banner.get("recommended_action") or "monitor_only"),
            "top_lane": str(banner.get("top_lane") or ""),
            "top_level": str(banner.get("top_level") or "quiet"),
        },
        "upstream_run_type": str(run_summary.get("run_type") or ""),
    }


def build_append_payload(*, payload: Dict[str, Any], source: str, history_path: str, out_json: str, max_history: int = 0) -> Dict[str, Any]:
    record = _snapshot_record(payload, source=source)
    return {
        "history_path": str(history_path),
        "max_history": int(max_history),
        "appended": dict(record),
        "run_summary": build_run_summary(
            run_type="event_watchboard_snapshot_append",
            inputs={"source": str(source), "history_path": str(history_path), "max_history": int(max_history)},
            metrics={"top_lane": str(record.get("top_lane") or ""), "severe_count": int((record.get("state_counts") or {}).get("severe", 0))},
            artifacts={"json": str(out_json), "history_jsonl": str(history_path)},
        ),
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Append a research event watchboard snapshot to a JSONL history.")
    p.add_argument("--source", required=True)
    p.add_argument("--history-jsonl", default="reports/RESEARCH_EVENT_WATCHBOARD_HISTORY.jsonl")
    p.add_argument("--max-history", type=int, default=0)
    p.add_argument("--out-json", default="reports/RESEARCH_EVENT_WATCHBOARD_SNAPSHOT_APPEND.json")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    payload = _load_json(Path(str(args.source)))
    append_payload = build_append_payload(
        payload=payload,
        source=str(args.source),
        history_path=str(args.history_jsonl),
        out_json=str(args.out_json),
        max_history=int(args.max_history),
    )
    history_path = Path(str(args.history_jsonl))
    out_json = Path(str(args.out_json))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    history_stats = append_history_record(
        history_path=history_path,
        record=append_payload["appended"],
        max_history=int(args.max_history),
    )
    append_payload["history_stats"] = dict(history_stats)
    append_payload["run_summary"]["metrics"].update(history_stats)
    out_json.write_text(json.dumps(append_payload, indent=2), encoding="utf-8")
    print(f"appended {history_path}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
