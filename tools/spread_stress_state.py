from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from tools.run_summary import build_run_summary


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _load_alert_payload(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _compute_state(summary: Dict[str, Any]) -> Tuple[str, List[str]]:
    recent_alert_count = _safe_int(summary.get("recent_alert_count"))
    high_count = _safe_int(summary.get("high_count"))
    medium_count = _safe_int(summary.get("medium_count"))
    tagged_rate = _safe_float(summary.get("tagged_rate"))
    avg_spread = _safe_float(summary.get("avg_spread_tagged"))
    reasons: List[str] = []

    if high_count >= 2:
        reasons.append("multiple_high_spread_stress")
    if recent_alert_count >= 8:
        reasons.append("dense_spread_stress_cluster")
    if tagged_rate >= 0.05:
        reasons.append("elevated_spread_stress_rate")
    if avg_spread >= 0.0002:
        reasons.append("extreme_avg_spread")
    if reasons:
        return ("severe", reasons)

    if medium_count >= 2:
        reasons.append("multiple_medium_spread_stress")
    if recent_alert_count >= 3:
        reasons.append("recent_spread_stress_cluster")
    if tagged_rate >= 0.02:
        reasons.append("moderate_spread_stress_rate")
    if avg_spread >= 0.0001:
        reasons.append("high_avg_spread")
    if reasons:
        return ("elevated", reasons)

    return ("quiet", ["no_runtime_threshold_hit"])


def _compute_freshness(*, latest_alert_ts_ms: int, now_ts_ms: int, bucket_sec: int) -> Dict[str, Any]:
    if latest_alert_ts_ms <= 0:
        return {"status": "stale", "age_sec": None, "stale_after_sec": max(60, int(bucket_sec) * 6)}
    age_sec = max(0.0, (float(now_ts_ms) - float(latest_alert_ts_ms)) / 1000.0)
    stale_after_sec = max(60, int(bucket_sec) * 6)
    return {
        "status": "fresh" if age_sec <= float(stale_after_sec) else "stale",
        "age_sec": age_sec,
        "stale_after_sec": int(stale_after_sec),
    }


def _recommended_action(*, state: str, freshness_status: str) -> str:
    if freshness_status != "fresh":
        return "monitor_only"
    if state == "severe":
        return "reduce_passive_aggression"
    if state == "elevated":
        return "show_caution"
    return "monitor_only"


def build_state_payload(
    *,
    alert_payload: Dict[str, Any],
    source_json: str,
    out_json: str,
    out_md: str,
    now_ts_ms: Optional[int] = None,
) -> Dict[str, Any]:
    summary = dict(alert_payload.get("summary") or {})
    alerts = list(alert_payload.get("alerts") or [])
    state, reasons = _compute_state(summary)
    recent_top = alerts[0] if alerts else {}
    latest_alert_ts_ms = _safe_int(recent_top.get("ts_ms"))
    effective_now_ts_ms = _safe_int(now_ts_ms, int(time.time() * 1000.0))
    freshness = _compute_freshness(
        latest_alert_ts_ms=latest_alert_ts_ms,
        now_ts_ms=effective_now_ts_ms,
        bucket_sec=_safe_int(alert_payload.get("bucket_sec"), 5),
    )
    recommended_action = _recommended_action(state=state, freshness_status=str(freshness["status"]))
    headline = (
        f"{str(alert_payload.get('symbol') or '').upper()} spread stress {state}: "
        f"{_safe_int(summary.get('recent_alert_count'))} recent alerts"
    )
    operator_note = {
        "quiet": "No spread stress action needed.",
        "elevated": "Show caution for passive execution quality.",
        "severe": "Escalate monitoring and consider reducing passive aggression.",
    }[state]
    dashboard_summary = (
        f"{str(alert_payload.get('symbol') or '').upper()} {state} spread stress, "
        f"{int(_safe_int(summary.get('recent_alert_count')))} recent alerts, freshness {str(freshness['status'])}."
    )
    notification_text = (
        f"[spread-stress] symbol={str(alert_payload.get('symbol') or '').upper()} "
        f"level={state} freshness={str(freshness['status'])} recent_alerts={int(_safe_int(summary.get('recent_alert_count')))} "
        f"avg_spread={_safe_float(summary.get('avg_spread_tagged')):.6f} action={recommended_action}"
    )
    payload = {
        "source_json": str(source_json),
        "symbol": str(alert_payload.get("symbol") or "").upper(),
        "state": {
            "level": state,
            "reasons": reasons,
            "freshness": dict(freshness),
        },
        "dashboard_summary": dashboard_summary,
        "notification_text": notification_text,
        "recommended_action": recommended_action,
        "card": {
            "headline": headline,
            "operator_note": operator_note,
            "recent_alert_count": _safe_int(summary.get("recent_alert_count")),
            "tagged_rate": _safe_float(summary.get("tagged_rate")),
            "high_count": _safe_int(summary.get("high_count")),
            "medium_count": _safe_int(summary.get("medium_count")),
            "avg_spread_tagged": _safe_float(summary.get("avg_spread_tagged")),
            "avg_trade_intensity_tagged": _safe_float(summary.get("avg_trade_intensity_tagged")),
            "latest_alert_ts_ms": latest_alert_ts_ms,
            "freshness_status": str(freshness["status"]),
            "age_sec": freshness["age_sec"],
        },
        "summary_snapshot": {
            "rows_total": _safe_int(summary.get("rows_total")),
            "tagged_count": _safe_int(summary.get("tagged_count")),
            "tagged_rate": _safe_float(summary.get("tagged_rate")),
            "recent_alert_count": _safe_int(summary.get("recent_alert_count")),
            "high_count": _safe_int(summary.get("high_count")),
            "medium_count": _safe_int(summary.get("medium_count")),
            "avg_spread_tagged": _safe_float(summary.get("avg_spread_tagged")),
            "avg_trade_intensity_tagged": _safe_float(summary.get("avg_trade_intensity_tagged")),
        },
    }
    payload["run_summary"] = build_run_summary(
        run_type="spread_stress_state",
        inputs={"source_json": str(source_json)},
        metrics={
            "state_level": state,
            "recent_alert_count": _safe_int(summary.get("recent_alert_count")),
            "tagged_rate": _safe_float(summary.get("tagged_rate")),
            "avg_spread_tagged": _safe_float(summary.get("avg_spread_tagged")),
            "freshness_status": str(freshness["status"]),
            "recommended_action": recommended_action,
        },
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compress spread stress alerts into a runtime-friendly state/card payload.")
    p.add_argument("--alerts-json", default="reports/SPREAD_STRESS_ALERTS.json")
    p.add_argument("--out-json", default="reports/SPREAD_STRESS_STATE.json")
    p.add_argument("--out-md", default="reports/SPREAD_STRESS_STATE.md")
    p.add_argument("--now-ts-ms", type=int, default=None)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    payload = build_state_payload(
        alert_payload=_load_alert_payload(str(args.alerts_json)),
        source_json=str(args.alerts_json),
        out_json=str(args.out_json),
        out_md=str(args.out_md),
        now_ts_ms=args.now_ts_ms,
    )
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# SPREAD STRESS STATE",
        "",
        f"symbol={payload['symbol']}",
        f"state={payload['state']['level']} reasons={json.dumps(payload['state']['reasons'], ensure_ascii=True)}",
        f"headline={payload['card']['headline']}",
        f"dashboard_summary={payload['dashboard_summary']}",
        f"notification_text={payload['notification_text']}",
        f"recommended_action={payload['recommended_action']}",
        f"freshness_status={payload['card']['freshness_status']} age_sec={payload['card']['age_sec']}",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
