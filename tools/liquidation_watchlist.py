from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.liquidation_alert_state import build_state_payload
from tools.liquidation_regime_alerts import build_alert_payload
from tools.run_summary import build_run_summary


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _priority(state_level: str, freshness_status: str, recent_alert_count: int, max_liq_rate_recent: float) -> float:
    base = {"quiet": 0.0, "elevated": 100.0, "severe": 200.0}.get(str(state_level), 0.0)
    if str(freshness_status) != "fresh":
        base -= 75.0
    return base + min(50.0, float(recent_alert_count)) + min(50.0, float(max_liq_rate_recent))


def build_watchlist_payload(
    *,
    db: str,
    symbols: List[str],
    lookback_min: int,
    bucket_sec: int,
    rule: str,
    recent_limit: int,
    min_liq_rate: float,
    top_n: int,
    out_json: str,
    out_md: str,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    state_counts: Dict[str, int] = {}
    for raw_symbol in symbols:
        symbol = str(raw_symbol).upper()
        alert_payload = build_alert_payload(
            db=db,
            symbol=symbol,
            lookback_min=lookback_min,
            bucket_sec=bucket_sec,
            rule=rule,
            recent_limit=recent_limit,
            min_liq_rate=min_liq_rate,
            out_json=f"reports/{symbol}_LIQUIDATION_REGIME_ALERTS.json",
            out_md=f"reports/{symbol}_LIQUIDATION_REGIME_ALERTS.md",
        )
        state_payload = build_state_payload(
            alert_payload=alert_payload,
            source_json=f"reports/{symbol}_LIQUIDATION_REGIME_ALERTS.json",
            out_json=f"reports/{symbol}_LIQUIDATION_ALERT_STATE.json",
            out_md=f"reports/{symbol}_LIQUIDATION_ALERT_STATE.md",
        )
        state = state_payload["state"]
        card = state_payload["card"]
        summary = state_payload["summary_snapshot"]
        level = str(state.get("level") or "quiet")
        freshness = str((state.get("freshness") or {}).get("status") or "stale")
        state_counts[level] = state_counts.get(level, 0) + 1
        rows.append(
            {
                "symbol": symbol,
                "state_level": level,
                "freshness_status": freshness,
                "recommended_action": str(state_payload.get("recommended_action") or "monitor_only"),
                "primary_side_bias": str(state.get("primary_side_bias") or "NEUTRAL"),
                "dominant_severity": str(state.get("dominant_severity") or "none"),
                "recent_alert_count": _safe_int(card.get("recent_alert_count")),
                "max_liq_rate_recent": _safe_float(card.get("max_liq_rate_recent")),
                "tagged_rate": _safe_float(summary.get("tagged_rate")),
                "age_sec": (state.get("freshness") or {}).get("age_sec"),
                "dashboard_summary": str(state_payload.get("dashboard_summary") or ""),
                "priority_score": _priority(
                    level,
                    freshness,
                    _safe_int(card.get("recent_alert_count")),
                    _safe_float(card.get("max_liq_rate_recent")),
                ),
            }
        )
    rows.sort(key=lambda row: (float(row["priority_score"]), float(row["max_liq_rate_recent"])), reverse=True)
    top_rows = rows[: max(1, int(top_n))]
    summary = {
        "symbol_count": int(len(rows)),
        "top_n": int(top_n),
        "state_counts": dict(state_counts),
        "top_symbol": str(top_rows[0]["symbol"]) if top_rows else "",
    }
    payload = {
        "rule": str(rule),
        "lookback_min": int(lookback_min),
        "bucket_sec": int(bucket_sec),
        "recent_limit": int(recent_limit),
        "min_liq_rate": float(min_liq_rate),
        "summary": summary,
        "rows": top_rows,
    }
    payload["run_summary"] = build_run_summary(
        run_type="liquidation_watchlist",
        inputs={
            "db": str(db),
            "symbols": [str(s).upper() for s in symbols],
            "lookback_min": int(lookback_min),
            "bucket_sec": int(bucket_sec),
            "rule": str(rule),
            "recent_limit": int(recent_limit),
            "min_liq_rate": float(min_liq_rate),
            "top_n": int(top_n),
        },
        metrics={
            "symbol_count": int(len(rows)),
            "top_n": int(top_n),
            "quiet_count": int(state_counts.get("quiet", 0)),
            "elevated_count": int(state_counts.get("elevated", 0)),
            "severe_count": int(state_counts.get("severe", 0)),
        },
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a multi-symbol liquidation regime watchlist.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="ETHUSDT,BTCUSDT")
    p.add_argument("--lookback-min", type=int, default=240)
    p.add_argument("--bucket-sec", type=int, default=5)
    p.add_argument("--rule", default="high_liq_reversal_regime")
    p.add_argument("--recent-limit", type=int, default=20)
    p.add_argument("--min-liq-rate", type=float, default=0.0)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--out-json", default="reports/LIQUIDATION_WATCHLIST.json")
    p.add_argument("--out-md", default="reports/LIQUIDATION_WATCHLIST.md")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    payload = build_watchlist_payload(
        db=str(args.db),
        symbols=symbols,
        lookback_min=int(args.lookback_min),
        bucket_sec=int(args.bucket_sec),
        rule=str(args.rule),
        recent_limit=int(args.recent_limit),
        min_liq_rate=float(args.min_liq_rate),
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
        "# LIQUIDATION WATCHLIST",
        "",
        f"symbols={payload['summary']['symbol_count']} top_n={payload['summary']['top_n']} top_symbol={payload['summary']['top_symbol']}",
        f"state_counts={json.dumps(payload['summary']['state_counts'], ensure_ascii=True, sort_keys=True)}",
        "",
        "| symbol | state_level | freshness_status | recommended_action | primary_side_bias | dominant_severity | recent_alert_count | max_liq_rate_recent | priority_score |",
        "|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['symbol']} | {row['state_level']} | {row['freshness_status']} | {row['recommended_action']} | "
            f"{row['primary_side_bias']} | {row['dominant_severity']} | {int(row['recent_alert_count'])} | "
            f"{float(row['max_liq_rate_recent']):.4f} | {float(row['priority_score']):.2f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
