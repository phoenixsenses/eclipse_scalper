from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.liquidation_regime_tagger import _load_rows
from tools.run_summary import build_run_summary


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    pos = (len(xs) - 1) * max(0.0, min(1.0, float(q)))
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _tag_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    abs_ret = [abs(_safe_float(r.get("ret_1"))) for r in rows if r.get("ret_1") is not None]
    spreads = [_safe_float(r.get("spread")) for r in rows if r.get("spread") is not None]
    intensity = [_safe_float(r.get("trade_intensity")) for r in rows if r.get("trade_intensity") is not None]
    ret_q90 = _quantile(abs_ret, 0.90)
    ret_q75 = _quantile(abs_ret, 0.75)
    spread_q75 = _quantile(spreads, 0.75)
    intensity_q75 = _quantile(intensity, 0.75)
    tagged: List[Dict[str, Any]] = []
    for row in rows:
        ret_1 = _safe_float(row.get("ret_1"))
        abs_move = abs(ret_1)
        spread = _safe_float(row.get("spread"))
        trade_intensity = _safe_float(row.get("trade_intensity"))
        direction = "UP" if ret_1 > 0 else "DOWN" if ret_1 < 0 else "FLAT"
        if abs_move >= ret_q90 and trade_intensity >= intensity_q75:
            severity = "high"
            fired = True
        elif abs_move >= ret_q75 and spread <= max(spread_q75, 0.0):
            severity = "medium"
            fired = True
        else:
            severity = "none"
            fired = False
        tagged.append(
            {
                "ts_ms": int(float(row.get("ts_ms", 0.0) or 0.0)),
                "tag": "return_shock" if fired else "normal",
                "rule_fired": fired,
                "severity": severity,
                "direction": direction,
                "ret_1": ret_1,
                "abs_ret_1": abs_move,
                "spread": spread,
                "trade_intensity": trade_intensity,
            }
        )
    return tagged


def build_payload(
    *,
    db: str,
    symbol: str,
    lookback_min: int,
    bucket_sec: int,
    recent_limit: int,
    out_json: str,
    out_md: str,
) -> Dict[str, Any]:
    rows = _load_rows(db, symbol, lookback_min, bucket_sec)
    tags = _tag_rows(rows)
    fired = [r for r in tags if bool(r.get("rule_fired"))]
    recent = sorted(fired, key=lambda r: int(r.get("ts_ms", 0) or 0), reverse=True)[: max(1, int(recent_limit))]
    direction_counts = {
        "UP": sum(1 for r in recent if str(r.get("direction")) == "UP"),
        "DOWN": sum(1 for r in recent if str(r.get("direction")) == "DOWN"),
        "FLAT": sum(1 for r in recent if str(r.get("direction")) == "FLAT"),
    }
    summary = {
        "rows_total": int(len(tags)),
        "tagged_count": int(len(fired)),
        "tagged_rate": (float(len(fired)) / float(len(tags))) if tags else 0.0,
        "recent_alert_count": int(len(recent)),
        "high_count": int(sum(1 for r in recent if str(r.get("severity")) == "high")),
        "medium_count": int(sum(1 for r in recent if str(r.get("severity")) == "medium")),
        "avg_abs_ret_1_tagged": (
            sum(_safe_float(r.get("abs_ret_1")) for r in fired) / float(len(fired))
            if fired
            else 0.0
        ),
        "avg_trade_intensity_tagged": (
            sum(_safe_float(r.get("trade_intensity")) for r in fired) / float(len(fired))
            if fired
            else 0.0
        ),
        "direction_counts": direction_counts,
    }
    payload = {
        "symbol": str(symbol).upper(),
        "lookback_min": int(lookback_min),
        "bucket_sec": int(bucket_sec),
        "summary": summary,
        "alerts": recent,
    }
    payload["run_summary"] = build_run_summary(
        run_type="return_shock_alerts",
        inputs={
            "db": str(db),
            "symbol": str(symbol).upper(),
            "lookback_min": int(lookback_min),
            "bucket_sec": int(bucket_sec),
            "recent_limit": int(recent_limit),
        },
        metrics=summary,
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build return shock alerts from microstructure DB.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--lookback-min", type=int, default=240)
    p.add_argument("--bucket-sec", type=int, default=5)
    p.add_argument("--recent-limit", type=int, default=20)
    p.add_argument("--out-json", default="reports/RETURN_SHOCK_ALERTS.json")
    p.add_argument("--out-md", default="reports/RETURN_SHOCK_ALERTS.md")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    payload = build_payload(
        db=str(args.db),
        symbol=str(args.symbol),
        lookback_min=int(args.lookback_min),
        bucket_sec=int(args.bucket_sec),
        recent_limit=int(args.recent_limit),
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
        "# RETURN SHOCK ALERTS",
        "",
        f"symbol={payload['symbol']} lookback_min={payload['lookback_min']} bucket_sec={payload['bucket_sec']}",
        f"rows_total={summary['rows_total']} tagged_count={summary['tagged_count']} tagged_rate={float(summary['tagged_rate']):.2%}",
        f"recent_alert_count={summary['recent_alert_count']} high_count={summary['high_count']} medium_count={summary['medium_count']}",
        f"avg_abs_ret_1_tagged={float(summary['avg_abs_ret_1_tagged']):.6f} avg_trade_intensity_tagged={float(summary['avg_trade_intensity_tagged']):.2f}",
        "",
        "| ts_ms | severity | direction | ret_1 | abs_ret_1 | spread | trade_intensity |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["alerts"]:
        lines.append(
            f"| {int(row['ts_ms'])} | {row['severity']} | {row['direction']} | {float(row['ret_1']):+.6f} | "
            f"{float(row['abs_ret_1']):.6f} | {float(row['spread']):.6f} | {float(row['trade_intensity']):.2f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
