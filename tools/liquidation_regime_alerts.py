from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.liquidation_regime_tagger import _load_rows, _tag_rows
from tools.run_summary import build_run_summary


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _recent_alerts(tags: List[Dict[str, Any]], recent_limit: int, min_liq_rate: float) -> List[Dict[str, Any]]:
    fired = [row for row in tags if bool(row.get("rule_fired")) and _safe_float(row.get("liq_rate_per_sec")) >= float(min_liq_rate)]
    fired.sort(key=lambda r: int(r.get("ts_ms", 0) or 0), reverse=True)
    alerts: List[Dict[str, Any]] = []
    for row in fired[: max(1, int(recent_limit))]:
        liq_imb = _safe_float(row.get("liq_imbalance"))
        side_bias = "LONG" if liq_imb > 0.0 else "SHORT"
        alerts.append(
            {
                "ts_ms": int(row.get("ts_ms", 0) or 0),
                "tag": str(row.get("tag") or ""),
                "side_bias": side_bias,
                "liq_rate_per_sec": _safe_float(row.get("liq_rate_per_sec")),
                "liq_imbalance": liq_imb,
                "spread": _safe_float(row.get("spread")),
                "trade_intensity": _safe_float(row.get("trade_intensity")),
                "ret_1": _safe_float(row.get("ret_1")),
            }
        )
    return alerts


def _max_consecutive_tagged(tags: List[Dict[str, Any]]) -> int:
    cur = 0
    mx = 0
    for row in sorted(tags, key=lambda r: int(r.get("ts_ms", 0) or 0)):
        if bool(row.get("rule_fired")):
            cur += 1
            if cur > mx:
                mx = cur
        else:
            cur = 0
    return int(mx)


def build_alert_payload(
    *,
    db: str,
    symbol: str,
    lookback_min: int,
    bucket_sec: int,
    rule: str,
    recent_limit: int,
    min_liq_rate: float,
    out_json: str,
    out_md: str,
) -> Dict[str, Any]:
    rows = _load_rows(db, symbol, lookback_min, bucket_sec)
    tags = _tag_rows(rows, rule)
    alerts = _recent_alerts(tags, recent_limit=recent_limit, min_liq_rate=min_liq_rate)
    tagged_count = sum(1 for row in tags if bool(row.get("rule_fired")))
    summary = {
        "rows_total": int(len(tags)),
        "tagged_count": int(tagged_count),
        "tagged_rate": (float(tagged_count) / float(len(tags))) if tags else 0.0,
        "recent_alert_count": int(len(alerts)),
        "max_consecutive_tagged": _max_consecutive_tagged(tags),
        "max_liq_rate_recent": max((_safe_float(a.get("liq_rate_per_sec")) for a in alerts), default=0.0),
    }
    payload = {
        "symbol": str(symbol).upper(),
        "rule": str(rule),
        "lookback_min": int(lookback_min),
        "bucket_sec": int(bucket_sec),
        "recent_limit": int(recent_limit),
        "min_liq_rate": float(min_liq_rate),
        "summary": summary,
        "alerts": alerts,
    }
    payload["run_summary"] = build_run_summary(
        run_type="liquidation_regime_alerts",
        inputs={
            "db": str(db),
            "symbol": str(symbol).upper(),
            "lookback_min": int(lookback_min),
            "bucket_sec": int(bucket_sec),
            "rule": str(rule),
            "recent_limit": int(recent_limit),
            "min_liq_rate": float(min_liq_rate),
        },
        metrics=summary,
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build recent liquidation regime alerts from microstructure DB.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--lookback-min", type=int, default=240)
    p.add_argument("--bucket-sec", type=int, default=5)
    p.add_argument("--rule", default="high_liq_reversal_regime")
    p.add_argument("--recent-limit", type=int, default=20)
    p.add_argument("--min-liq-rate", type=float, default=0.0)
    p.add_argument("--out-json", default="reports/LIQUIDATION_REGIME_ALERTS.json")
    p.add_argument("--out-md", default="reports/LIQUIDATION_REGIME_ALERTS.md")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    payload = build_alert_payload(
        db=str(args.db),
        symbol=str(args.symbol),
        lookback_min=int(args.lookback_min),
        bucket_sec=int(args.bucket_sec),
        rule=str(args.rule),
        recent_limit=int(args.recent_limit),
        min_liq_rate=float(args.min_liq_rate),
        out_json=str(args.out_json),
        out_md=str(args.out_md),
    )
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = payload["summary"]
    lines = [
        "# LIQUIDATION REGIME ALERTS",
        "",
        f"symbol={payload['symbol']} rule={payload['rule']} lookback_min={payload['lookback_min']} bucket_sec={payload['bucket_sec']}",
        f"rows_total={summary['rows_total']} tagged_count={summary['tagged_count']} tagged_rate={float(summary['tagged_rate']):.2%}",
        f"recent_alert_count={summary['recent_alert_count']} max_consecutive_tagged={summary['max_consecutive_tagged']}",
        "",
        "| ts_ms | side_bias | liq_rate_per_sec | liq_imbalance | spread | trade_intensity | ret_1 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for alert in payload["alerts"]:
        lines.append(
            f"| {int(alert['ts_ms'])} | {alert['side_bias']} | {float(alert['liq_rate_per_sec']):.4f} | "
            f"{float(alert['liq_imbalance']):+.4f} | {float(alert['spread']):.6f} | {float(alert['trade_intensity']):.2f} | "
            f"{float(alert['ret_1']):+.6f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
