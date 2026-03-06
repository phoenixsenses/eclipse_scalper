from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.fill_toxicity_state import build_state_payload as build_fill_toxicity_state
from tools.latency_stress_state import build_state_payload as build_latency_stress_state
from tools.liquidation_watchlist import build_watchlist_payload as build_liquidation_watchlist
from tools.return_shock_watchlist import build_watchlist_payload as build_return_shock_watchlist
from tools.run_summary import build_run_summary
from tools.spread_stress_watchlist import build_watchlist_payload as build_spread_stress_watchlist
from tools.volume_vacuum_watchlist import build_watchlist_payload as build_volume_vacuum_watchlist
from tools.toxicity_report import build_toxicity_report, _load as load_toxicity_rows
from tools.execution_diagnostics import compute_execution_diagnostics, _load_rows as load_execution_rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _severity_score(level: str) -> float:
    return {"quiet": 0.0, "elevated": 100.0, "severe": 200.0}.get(str(level), 0.0)


def _freshness_bonus(status: str) -> float:
    return 25.0 if str(status) == "fresh" else 0.0


def _liquidation_entry(payload: Dict[str, Any]) -> Dict[str, Any]:
    top = dict(payload.get("top_summary") or {})
    banner = dict(payload.get("banner") or {})
    return {
        "lane": "liquidation",
        "level": str(top.get("state_level") or "quiet"),
        "freshness_status": str(top.get("freshness_status") or "stale"),
        "recommended_action": str(top.get("recommended_action") or "monitor_only"),
        "headline": str(banner.get("headline") or ""),
        "detail": str(top.get("dashboard_summary") or ""),
        "priority_score": _severity_score(top.get("state_level")) + _freshness_bonus(top.get("freshness_status")),
    }


def _spread_entry(payload: Dict[str, Any]) -> Dict[str, Any]:
    top = dict(payload.get("top_summary") or {})
    banner = dict(payload.get("banner") or {})
    return {
        "lane": "spread_stress",
        "level": str(top.get("state_level") or "quiet"),
        "freshness_status": str(top.get("freshness_status") or "stale"),
        "recommended_action": str(top.get("recommended_action") or "monitor_only"),
        "headline": str(banner.get("headline") or ""),
        "detail": str(top.get("dashboard_summary") or ""),
        "priority_score": _severity_score(top.get("state_level")) + _freshness_bonus(top.get("freshness_status")),
    }


def _return_shock_entry(payload: Dict[str, Any]) -> Dict[str, Any]:
    top = dict(payload.get("top_summary") or {})
    banner = dict(payload.get("banner") or {})
    return {
        "lane": "return_shock",
        "level": str(top.get("state_level") or "quiet"),
        "freshness_status": str(top.get("freshness_status") or "stale"),
        "recommended_action": str(top.get("recommended_action") or "monitor_only"),
        "headline": str(banner.get("headline") or ""),
        "detail": str(top.get("dashboard_summary") or ""),
        "priority_score": _severity_score(top.get("state_level")) + _freshness_bonus(top.get("freshness_status")),
    }


def _volume_vacuum_entry(payload: Dict[str, Any]) -> Dict[str, Any]:
    top = dict(payload.get("top_summary") or {})
    banner = dict(payload.get("banner") or {})
    return {
        "lane": "volume_vacuum",
        "level": str(top.get("state_level") or "quiet"),
        "freshness_status": str(top.get("freshness_status") or "stale"),
        "recommended_action": str(top.get("recommended_action") or "monitor_only"),
        "headline": str(banner.get("headline") or ""),
        "detail": str(top.get("dashboard_summary") or ""),
        "priority_score": _severity_score(top.get("state_level")) + _freshness_bonus(top.get("freshness_status")),
    }


def _state_entry(lane: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    state = dict(payload.get("state") or {})
    return {
        "lane": lane,
        "level": str(state.get("level") or "quiet"),
        "freshness_status": "fresh" if int(payload.get("rows", payload.get("card", {}).get("rows", 0)) or 0) > 0 else "stale",
        "recommended_action": str(payload.get("recommended_action") or "monitor_only"),
        "headline": str((payload.get("card") or {}).get("headline") or ""),
        "detail": str(payload.get("dashboard_summary") or ""),
        "priority_score": _severity_score(state.get("level")),
    }


def build_watchboard_payload(
    *,
    micro_db: str,
    trade_source: str,
    symbols: List[str],
    lookback_min: int,
    bucket_sec: int,
    recent_limit: int,
    top_n: int,
    out_json: str,
    out_md: str,
) -> Dict[str, Any]:
    liq = build_liquidation_watchlist(
        db=micro_db,
        symbols=symbols,
        lookback_min=lookback_min,
        bucket_sec=bucket_sec,
        rule="high_liq_reversal_regime",
        recent_limit=recent_limit,
        min_liq_rate=0.0,
        top_n=top_n,
        out_json="reports/LIQUIDATION_WATCHLIST.json",
        out_md="reports/LIQUIDATION_WATCHLIST.md",
    )
    spread = build_spread_stress_watchlist(
        db=micro_db,
        symbols=symbols,
        lookback_min=lookback_min,
        bucket_sec=bucket_sec,
        recent_limit=recent_limit,
        top_n=top_n,
        out_json="reports/SPREAD_STRESS_WATCHLIST.json",
        out_md="reports/SPREAD_STRESS_WATCHLIST.md",
    )
    return_shock = build_return_shock_watchlist(
        db=micro_db,
        symbols=symbols,
        lookback_min=lookback_min,
        bucket_sec=bucket_sec,
        recent_limit=recent_limit,
        top_n=top_n,
        out_json="reports/RETURN_SHOCK_WATCHLIST.json",
        out_md="reports/RETURN_SHOCK_WATCHLIST.md",
    )
    volume_vacuum = build_volume_vacuum_watchlist(
        db=micro_db,
        symbols=symbols,
        lookback_min=lookback_min,
        bucket_sec=bucket_sec,
        recent_limit=recent_limit,
        top_n=top_n,
        out_json="reports/VOLUME_VACUUM_WATCHLIST.json",
        out_md="reports/VOLUME_VACUUM_WATCHLIST.md",
    )
    fill = build_fill_toxicity_state(
        source=trade_source,
        report_payload=build_toxicity_report(load_toxicity_rows(Path(trade_source))),
        out_json="reports/FILL_TOXICITY_STATE.json",
        out_md="reports/FILL_TOXICITY_STATE.md",
    )
    latency = build_latency_stress_state(
        source=trade_source,
        diag=compute_execution_diagnostics(load_execution_rows(Path(trade_source))),
        out_json="reports/LATENCY_STRESS_STATE.json",
        out_md="reports/LATENCY_STRESS_STATE.md",
    )

    lanes = [
        _liquidation_entry(liq),
        _spread_entry(spread),
        _return_shock_entry(return_shock),
        _volume_vacuum_entry(volume_vacuum),
        _state_entry("fill_toxicity", fill),
        _state_entry("latency_stress", latency),
    ]
    lanes.sort(key=lambda row: float(row["priority_score"]), reverse=True)
    top_event = lanes[0] if lanes else {}
    state_counts: Dict[str, int] = {}
    for lane in lanes:
        level = str(lane.get("level") or "quiet")
        state_counts[level] = state_counts.get(level, 0) + 1
    payload = {
        "summary": {
            "lane_count": len(lanes),
            "state_counts": state_counts,
            "top_lane": str(top_event.get("lane") or ""),
        },
        "top_event": {
            "lane": str(top_event.get("lane") or ""),
            "level": str(top_event.get("level") or "quiet"),
            "recommended_action": str(top_event.get("recommended_action") or "monitor_only"),
            "headline": str(top_event.get("headline") or ""),
            "detail": str(top_event.get("detail") or ""),
        },
        "banner": {
            "headline": str(top_event.get("headline") or "Research event watchboard quiet"),
            "recommended_action": str(top_event.get("recommended_action") or "monitor_only"),
            "top_lane": str(top_event.get("lane") or ""),
            "top_level": str(top_event.get("level") or "quiet"),
        },
        "lanes": lanes,
    }
    payload["run_summary"] = build_run_summary(
        run_type="research_event_watchboard",
        inputs={
            "micro_db": micro_db,
            "trade_source": trade_source,
            "symbols": symbols,
            "lookback_min": lookback_min,
            "bucket_sec": bucket_sec,
            "recent_limit": recent_limit,
            "top_n": top_n,
        },
        metrics={
            "lane_count": len(lanes),
            "severe_count": state_counts.get("severe", 0),
            "elevated_count": state_counts.get("elevated", 0),
            "quiet_count": state_counts.get("quiet", 0),
        },
        artifacts={"json": out_json, "md": out_md},
    )
    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a single research event watchboard payload from active lanes.")
    p.add_argument("--micro-db", default="data/microstructure.db")
    p.add_argument("--trade-source", default="data/live/papertrades_live.parquet")
    p.add_argument("--symbols", default="ETHUSDT,BTCUSDT")
    p.add_argument("--lookback-min", type=int, default=240)
    p.add_argument("--bucket-sec", type=int, default=5)
    p.add_argument("--recent-limit", type=int, default=20)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--out-json", default="reports/RESEARCH_EVENT_WATCHBOARD.json")
    p.add_argument("--out-md", default="reports/RESEARCH_EVENT_WATCHBOARD.md")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    payload = build_watchboard_payload(
        micro_db=str(args.micro_db),
        trade_source=str(args.trade_source),
        symbols=symbols,
        lookback_min=int(args.lookback_min),
        bucket_sec=int(args.bucket_sec),
        recent_limit=int(args.recent_limit),
        top_n=int(args.top_n),
        out_json=str(args.out_json),
        out_md=str(args.out_md),
    )
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# RESEARCH EVENT WATCHBOARD",
        "",
        f"top_lane={payload['summary']['top_lane']}",
        f"state_counts={json.dumps(payload['summary']['state_counts'], ensure_ascii=True, sort_keys=True)}",
        f"top_event={json.dumps(payload['top_event'], ensure_ascii=True, sort_keys=True)}",
        f"banner={json.dumps(payload['banner'], ensure_ascii=True, sort_keys=True)}",
        "",
        "| lane | level | freshness_status | recommended_action | priority_score |",
        "|---|---|---|---|---:|",
    ]
    for row in payload["lanes"]:
        lines.append(
            f"| {row['lane']} | {row['level']} | {row['freshness_status']} | {row['recommended_action']} | {float(row['priority_score']):.2f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
