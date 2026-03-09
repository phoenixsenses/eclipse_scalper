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


def _argmax_count(counts: Dict[str, Any], default: str) -> str:
    best_key = str(default)
    best_val = -1
    for key, value in (counts or {}).items():
        iv = _safe_int(value, 0)
        if iv > best_val:
            best_key = str(key)
            best_val = iv
    return best_key


def _compute_state(summary: Dict[str, Any]) -> Tuple[str, List[str]]:
    recent_alert_count = _safe_int(summary.get("recent_alert_count"))
    max_consecutive = _safe_int(summary.get("max_consecutive_tagged"))
    max_liq_rate = _safe_float(summary.get("max_liq_rate_recent"))
    tagged_rate = _safe_float(summary.get("tagged_rate"))
    severity_counts = summary.get("severity_counts") or {}
    high_count = _safe_int(severity_counts.get("high"))
    medium_count = _safe_int(severity_counts.get("medium"))
    reasons: List[str] = []

    if high_count >= 2:
        reasons.append("multiple_high_severity_alerts")
    if max_liq_rate >= 8.0:
        reasons.append("extreme_liquidation_rate")
    if recent_alert_count >= 8:
        reasons.append("dense_recent_alerts")
    if max_consecutive >= 3:
        reasons.append("persistent_regime")
    if tagged_rate >= 0.05:
        reasons.append("elevated_tagged_rate")
    if reasons:
        return ("severe", reasons)

    if medium_count >= 2:
        reasons.append("multiple_medium_alerts")
    if max_liq_rate >= 5.0:
        reasons.append("high_liquidation_rate")
    if recent_alert_count >= 3:
        reasons.append("recent_alert_cluster")
    if max_consecutive >= 2:
        reasons.append("repeated_regime")
    if tagged_rate >= 0.025:
        reasons.append("moderate_tagged_rate")
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
        return "escalate_monitoring"
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
    side_bias_counts = summary.get("side_bias_counts") or {}
    severity_counts = summary.get("severity_counts") or {}
    state, reasons = _compute_state(summary)
    primary_side_bias = _argmax_count(side_bias_counts, "NEUTRAL")
    dominant_severity = _argmax_count(severity_counts, "none")
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
        f"{str(alert_payload.get('symbol') or '').upper()} liquidation regime {state}: "
        f"{_safe_int(summary.get('recent_alert_count'))} recent alerts, "
        f"bias={primary_side_bias}, top_severity={dominant_severity}"
    )
    operator_note = {
        "quiet": "Monitor only. No execution change implied.",
        "elevated": "Show on dashboard and raise soft caution for runtime operators.",
        "severe": "Escalate on dashboard and monitoring. Use as caution context, not auto-trade.",
    }[state]
    dashboard_summary = (
        f"{str(alert_payload.get('symbol') or '').upper()} {state} liquidation regime, "
        f"bias {primary_side_bias}, {int(_safe_int(summary.get('recent_alert_count')))} recent alerts, "
        f"freshness {str(freshness['status'])}."
    )
    notification_text = (
        f"[liq-regime] symbol={str(alert_payload.get('symbol') or '').upper()} "
        f"level={state} freshness={str(freshness['status'])} bias={primary_side_bias} "
        f"recent_alerts={int(_safe_int(summary.get('recent_alert_count')))} "
        f"max_liq_rate={_safe_float(summary.get('max_liq_rate_recent')):.4f} "
        f"action={recommended_action}"
    )
    payload = {
        "source_json": str(source_json),
        "symbol": str(alert_payload.get("symbol") or "").upper(),
        "rule": str(alert_payload.get("rule") or ""),
        "state": {
            "level": state,
            "reasons": reasons,
            "primary_side_bias": primary_side_bias,
            "dominant_severity": dominant_severity,
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
            "max_consecutive_tagged": _safe_int(summary.get("max_consecutive_tagged")),
            "max_liq_rate_recent": _safe_float(summary.get("max_liq_rate_recent")),
            "primary_side_bias": primary_side_bias,
            "dominant_severity": dominant_severity,
            "latest_alert_ts_ms": latest_alert_ts_ms,
            "freshness_status": str(freshness["status"]),
            "age_sec": freshness["age_sec"],
        },
        "summary_snapshot": {
            "rows_total": _safe_int(summary.get("rows_total")),
            "tagged_count": _safe_int(summary.get("tagged_count")),
            "tagged_rate": _safe_float(summary.get("tagged_rate")),
            "recent_alert_count": _safe_int(summary.get("recent_alert_count")),
            "max_consecutive_tagged": _safe_int(summary.get("max_consecutive_tagged")),
            "max_liq_rate_recent": _safe_float(summary.get("max_liq_rate_recent")),
            "side_bias_counts": dict(side_bias_counts),
            "severity_counts": dict(severity_counts),
        },
    }
    payload["run_summary"] = build_run_summary(
        run_type="liquidation_alert_state",
        inputs={"source_json": str(source_json)},
        metrics={
            "state_level": state,
            "recent_alert_count": _safe_int(summary.get("recent_alert_count")),
            "tagged_rate": _safe_float(summary.get("tagged_rate")),
            "max_liq_rate_recent": _safe_float(summary.get("max_liq_rate_recent")),
            "freshness_status": str(freshness["status"]),
            "recommended_action": recommended_action,
        },
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compress liquidation alerts into a runtime-friendly state/card payload.")
    p.add_argument("--alerts-json", default="reports/LIQUIDATION_REGIME_ALERTS.json")
    p.add_argument("--out-json", default="reports/LIQUIDATION_ALERT_STATE.json")
    p.add_argument("--out-md", default="reports/LIQUIDATION_ALERT_STATE.md")
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
        "# LIQUIDATION ALERT STATE",
        "",
        f"symbol={payload['symbol']} rule={payload['rule']}",
        f"state={payload['state']['level']} reasons={json.dumps(payload['state']['reasons'], ensure_ascii=True)}",
        f"headline={payload['card']['headline']}",
        f"dashboard_summary={payload['dashboard_summary']}",
        f"notification_text={payload['notification_text']}",
        f"recommended_action={payload['recommended_action']}",
        f"operator_note={payload['card']['operator_note']}",
        f"recent_alert_count={payload['card']['recent_alert_count']} tagged_rate={payload['card']['tagged_rate']:.2%}",
        f"max_consecutive_tagged={payload['card']['max_consecutive_tagged']} max_liq_rate_recent={payload['card']['max_liq_rate_recent']:.4f}",
        f"primary_side_bias={payload['card']['primary_side_bias']} dominant_severity={payload['card']['dominant_severity']}",
        f"freshness_status={payload['card']['freshness_status']} age_sec={payload['card']['age_sec']}",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
