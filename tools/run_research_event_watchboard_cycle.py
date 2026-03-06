from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.event_watchboard_snapshot_append import append_history_record, build_append_payload
from tools.event_watchboard_trend_from_history import build_trend_from_history_payload, _load_history
from tools.research_event_operator_brief import build_operator_brief_payload
from tools.research_event_watchboard import build_watchboard_payload
from tools.run_summary import build_run_summary


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in str(raw).split(",") if s.strip()]


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_lines(path: str, lines: List[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_cycle_payload(
    *,
    micro_db: str,
    trade_source: str,
    symbols: List[str],
    lookback_min: int,
    bucket_sec: int,
    recent_limit: int,
    top_n: int,
    watchboard_json: str,
    watchboard_md: str,
    history_jsonl: str,
    max_history: int,
    append_json: str,
    trend_json: str,
    trend_md: str,
    brief_json: str,
    brief_md: str,
    out_json: str,
    out_md: str,
) -> Dict[str, Any]:
    watchboard = build_watchboard_payload(
        micro_db=micro_db,
        trade_source=trade_source,
        symbols=symbols,
        lookback_min=lookback_min,
        bucket_sec=bucket_sec,
        recent_limit=recent_limit,
        top_n=top_n,
        out_json=watchboard_json,
        out_md=watchboard_md,
    )
    _write_json(watchboard_json, watchboard)
    _write_lines(
        watchboard_md,
        [
            "# RESEARCH EVENT WATCHBOARD",
            "",
            str((watchboard.get("banner") or {}).get("headline") or ""),
        ],
    )

    history_path = Path(history_jsonl)
    append_payload = build_append_payload(
        payload=watchboard,
        source=watchboard_json,
        history_path=history_jsonl,
        out_json=append_json,
        max_history=max_history,
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_stats = append_history_record(
        history_path=history_path,
        record=append_payload["appended"],
        max_history=max_history,
    )
    append_payload["history_stats"] = dict(history_stats)
    append_payload["run_summary"]["metrics"].update(history_stats)
    _write_json(append_json, append_payload)

    history_rows = _load_history(history_path)
    trend_payload = build_trend_from_history_payload(
        history_rows=history_rows,
        history_path=history_jsonl,
        last_n=max(1, min(24, len(history_rows))),
        out_json=trend_json,
        out_md=trend_md,
    )
    _write_json(trend_json, trend_payload)
    _write_lines(
        trend_md,
        [
            "# RESEARCH EVENT WATCHBOARD TREND FROM HISTORY",
            "",
            f"trend={str((trend_payload.get('summary') or {}).get('trend') or 'flat')}",
            f"end_top_lane={str((trend_payload.get('summary') or {}).get('end_top_lane') or '')}",
        ],
    )
    brief_payload = build_operator_brief_payload(
        watchboard_json=watchboard_json,
        trend_json=trend_json,
        out_json=brief_json,
        out_md=brief_md,
    )
    _write_json(brief_json, brief_payload)
    _write_lines(
        brief_md,
        [
            "# RESEARCH EVENT OPERATOR BRIEF",
            "",
            str((brief_payload.get("brief") or {}).get("headline") or ""),
            "",
            str((brief_payload.get("brief") or {}).get("operator_note") or ""),
        ],
    )

    payload = {
        "watchboard_json": str(watchboard_json),
        "append_json": str(append_json),
        "trend_json": str(trend_json),
        "brief_json": str(brief_json),
        "history_jsonl": str(history_jsonl),
        "summary": {
            "top_lane": str((watchboard.get("summary") or {}).get("top_lane") or ""),
            "top_action": str((watchboard.get("top_event") or {}).get("recommended_action") or "monitor_only"),
            "history_rows": int((trend_payload.get("history") or {}).get("available_rows", 0)),
            "trend": str((trend_payload.get("summary") or {}).get("trend") or "flat"),
            "trimmed_rows": int(history_stats.get("trimmed_rows", 0)),
        },
    }
    payload["run_summary"] = build_run_summary(
        run_type="run_research_event_watchboard_cycle",
        inputs={
            "micro_db": micro_db,
            "trade_source": trade_source,
            "symbols": symbols,
            "lookback_min": lookback_min,
            "bucket_sec": bucket_sec,
            "recent_limit": recent_limit,
            "top_n": top_n,
            "max_history": max_history,
        },
        metrics={
            "top_lane": payload["summary"]["top_lane"],
            "history_rows": payload["summary"]["history_rows"],
            "trend": payload["summary"]["trend"],
            "trimmed_rows": payload["summary"]["trimmed_rows"],
        },
        artifacts={
            "json": out_json,
            "md": out_md,
            "watchboard_json": watchboard_json,
            "append_json": append_json,
            "trend_json": trend_json,
            "brief_json": brief_json,
            "history_jsonl": history_jsonl,
        },
    )
    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run watchboard -> snapshot append -> trend-from-history in one command.")
    p.add_argument("--micro-db", default="data/microstructure.db")
    p.add_argument("--trade-source", default="data/live/papertrades_live.parquet")
    p.add_argument("--symbols", default="ETHUSDT,BTCUSDT")
    p.add_argument("--lookback-min", type=int, default=240)
    p.add_argument("--bucket-sec", type=int, default=5)
    p.add_argument("--recent-limit", type=int, default=20)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--watchboard-json", default="reports/RESEARCH_EVENT_WATCHBOARD.json")
    p.add_argument("--watchboard-md", default="reports/RESEARCH_EVENT_WATCHBOARD.md")
    p.add_argument("--history-jsonl", default="reports/RESEARCH_EVENT_WATCHBOARD_HISTORY.jsonl")
    p.add_argument("--max-history", type=int, default=288)
    p.add_argument("--append-json", default="reports/RESEARCH_EVENT_WATCHBOARD_SNAPSHOT_APPEND.json")
    p.add_argument("--trend-json", default="reports/RESEARCH_EVENT_WATCHBOARD_TREND_FROM_HISTORY.json")
    p.add_argument("--trend-md", default="reports/RESEARCH_EVENT_WATCHBOARD_TREND_FROM_HISTORY.md")
    p.add_argument("--brief-json", default="reports/RESEARCH_EVENT_OPERATOR_BRIEF.json")
    p.add_argument("--brief-md", default="reports/RESEARCH_EVENT_OPERATOR_BRIEF.md")
    p.add_argument("--out-json", default="reports/RESEARCH_EVENT_WATCHBOARD_CYCLE.json")
    p.add_argument("--out-md", default="reports/RESEARCH_EVENT_WATCHBOARD_CYCLE.md")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    payload = build_cycle_payload(
        micro_db=str(args.micro_db),
        trade_source=str(args.trade_source),
        symbols=_parse_symbols(str(args.symbols)),
        lookback_min=int(args.lookback_min),
        bucket_sec=int(args.bucket_sec),
        recent_limit=int(args.recent_limit),
        top_n=int(args.top_n),
        watchboard_json=str(args.watchboard_json),
        watchboard_md=str(args.watchboard_md),
        history_jsonl=str(args.history_jsonl),
        max_history=int(args.max_history),
        append_json=str(args.append_json),
        trend_json=str(args.trend_json),
        trend_md=str(args.trend_md),
        brief_json=str(args.brief_json),
        brief_md=str(args.brief_md),
        out_json=str(args.out_json),
        out_md=str(args.out_md),
    )
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# RESEARCH EVENT WATCHBOARD CYCLE",
        "",
        f"top_lane={payload['summary']['top_lane']}",
        f"top_action={payload['summary']['top_action']}",
        f"history_rows={payload['summary']['history_rows']}",
        f"trend={payload['summary']['trend']}",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
