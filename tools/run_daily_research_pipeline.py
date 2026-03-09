from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from tools import daily_research_report as drr
from tools import refresh_dashboard_research_events as rdr


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run daily research report and dashboard artifact refresh in one command.")
    p.add_argument("--date", default="", help="Report date in YYYY-MM-DD. Defaults to local date.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--symbols", default="ETHUSDT,BTCUSDT")
    p.add_argument("--trade-source", default="data/live/papertrades_live.parquet")
    p.add_argument("--telemetry-path", default="logs/telemetry.jsonl")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--recovery-lookback-min", type=int, default=180)
    p.add_argument("--event-lookback-min", type=int, default=60)
    p.add_argument("--lookback-min", type=int, default=240)
    p.add_argument("--bucket-sec", type=int, default=5)
    p.add_argument("--recent-limit", type=int, default=20)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--out-json", default="reports/DAILY_RESEARCH_PIPELINE.json")
    p.add_argument("--out-md", default="reports/DAILY_RESEARCH_PIPELINE.md")
    return p.parse_args(argv)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    headline = payload.get("daily_report", {}).get("headline", {})
    refresh = payload.get("refresh", {}).get("summary", {})
    lines = [
        "# DAILY RESEARCH PIPELINE",
        "",
        f"report_date={payload.get('report_date')}",
        f"event_lanes={headline.get('event_lanes')}",
        f"regime_recovery_prep={headline.get('regime_recovery_prep')}",
        f"pocket_promotion_checklist={headline.get('pocket_promotion_checklist')}",
        f"watchboard_top_lane={refresh.get('watchboard_top_lane')}",
        f"artifact_count={refresh.get('artifact_count')}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    report_date = drr._default_report_date(str(args.date))
    daily_out = Path(str(args.reports_dir)) / f"DAILY_{report_date}.md"
    daily_args = type(
        "DailyArgs",
        (),
        {
            "date": report_date,
            "out": str(daily_out),
            "db": str(args.db),
            "symbol": str(args.symbol),
            "telemetry_path": str(args.telemetry_path),
            "recovery_lookback_min": int(args.recovery_lookback_min),
            "event_lookback_min": int(args.event_lookback_min),
            "event_bucket_sec": int(args.bucket_sec),
            "event_stale_after_sec": 60,
        },
    )()
    daily_rc = drr.run_once(daily_args)
    if daily_rc != 0:
      raise RuntimeError(f"daily_research_report failed rc={daily_rc}")

    daily_json = daily_out.with_suffix(".json")
    daily_payload = json.loads(daily_json.read_text(encoding="utf-8"))
    refresh_payload = rdr.build_refresh_payload(
        micro_db=str(args.db),
        trade_source=str(args.trade_source),
        primary_symbol=str(args.symbol).upper(),
        symbols=rdr._parse_symbols(str(args.symbols)),
        lookback_min=int(args.lookback_min),
        bucket_sec=int(args.bucket_sec),
        recent_limit=int(args.recent_limit),
        top_n=int(args.top_n),
        reports_dir=str(args.reports_dir),
    )
    return {
        "report_date": report_date,
        "daily_report": daily_payload,
        "refresh": refresh_payload,
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    payload = run_pipeline(args)
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    _write_json(out_json, payload)
    _write_md(out_md, payload)
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
