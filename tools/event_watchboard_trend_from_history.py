from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.event_watchboard_trend import build_trend_payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _load_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def _snapshot_from_history(row: Dict[str, Any]) -> Dict[str, Any]:
    top_event = dict(row.get("top_event") or {})
    banner = dict(row.get("banner") or {})
    top_lane = str(row.get("top_lane") or top_event.get("lane") or "")
    state_counts = dict(row.get("state_counts") or {})
    top_level = str(top_event.get("level") or banner.get("top_level") or "quiet")
    recommended_action = str(top_event.get("recommended_action") or banner.get("recommended_action") or "monitor_only")
    headline = str(top_event.get("headline") or banner.get("headline") or "")
    severity_score = {"quiet": 0.0, "elevated": 100.0, "severe": 200.0}.get(top_level, 0.0)
    return {
        "summary": {"top_lane": top_lane, "state_counts": state_counts},
        "top_event": {
            "lane": top_lane,
            "level": top_level,
            "recommended_action": recommended_action,
            "headline": headline,
            "freshness_status": "unknown",
        },
        "lanes": [
            {
                "lane": top_lane,
                "priority_score": severity_score,
            }
        ],
    }


def build_trend_from_history_payload(*, history_rows: List[Dict[str, Any]], history_path: str, last_n: int, out_json: str, out_md: str) -> Dict[str, Any]:
    rows = history_rows[-max(1, int(last_n)) :]
    snapshots = [_snapshot_from_history(row) for row in rows]
    sources = [str(row.get("source") or history_path) for row in rows]
    payload = build_trend_payload(
        snapshots=snapshots,
        source_paths=sources,
        out_json=out_json,
        out_md=out_md,
    )
    payload["history"] = {
        "history_path": str(history_path),
        "last_n": int(last_n),
        "available_rows": int(len(history_rows)),
        "used_rows": int(len(rows)),
    }
    payload["run_summary"]["run_type"] = "event_watchboard_trend_from_history"
    payload["run_summary"]["inputs"] = {
        "history_path": str(history_path),
        "last_n": int(last_n),
    }
    payload["run_summary"]["metrics"] = {
        "available_rows": int(len(history_rows)),
        "used_rows": int(len(rows)),
        "delta_priority_score": _safe_float(payload.get("summary", {}).get("delta_priority_score")),
        "trend": str(payload.get("summary", {}).get("trend") or "flat"),
    }
    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build watchboard trend from JSONL history.")
    p.add_argument("--history-jsonl", default="reports/RESEARCH_EVENT_WATCHBOARD_HISTORY.jsonl")
    p.add_argument("--last-n", type=int, default=24)
    p.add_argument("--out-json", default="reports/RESEARCH_EVENT_WATCHBOARD_TREND_FROM_HISTORY.json")
    p.add_argument("--out-md", default="reports/RESEARCH_EVENT_WATCHBOARD_TREND_FROM_HISTORY.md")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    history_path = Path(str(args.history_jsonl))
    history_rows = _load_history(history_path)
    payload = build_trend_from_history_payload(
        history_rows=history_rows,
        history_path=str(history_path),
        last_n=int(args.last_n),
        out_json=str(args.out_json),
        out_md=str(args.out_md),
    )
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# RESEARCH EVENT WATCHBOARD TREND FROM HISTORY",
        "",
        f"available_rows={payload['history']['available_rows']}",
        f"used_rows={payload['history']['used_rows']}",
        f"delta_priority_score={float(payload['summary']['delta_priority_score']):.2f}",
        f"trend={payload['summary']['trend']}",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
