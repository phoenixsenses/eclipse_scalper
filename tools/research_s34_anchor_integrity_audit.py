from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRADES_PATH = ROOT / "reports" / "research" / "s34" / "S34_SHADOW_PAPER_TRADES.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_ANCHOR_INTEGRITY_AUDIT.md"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_ANCHOR_INTEGRITY_AUDIT.json"

RULES = [
    "ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30",
    "ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30",
]


def r1(value: float | None) -> float | None:
    return round(float(value), 1) if value is not None and math.isfinite(float(value)) else None


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "median": r1(statistics.median(values)),
        "mean": r1(statistics.mean(values)),
        "cum": r1(sum(values)),
        "wr": round(sum(v > 0 for v in values) / len(values), 3),
    }


def classify_trade(trade: dict) -> tuple[str, dict]:
    signal = trade.get("signal") or {}
    sig = trade.get("signal_ts_ms") or signal.get("ts_ms")
    start = signal.get("cluster_start_ts_ms")
    end = signal.get("cluster_end_ts_ms")
    threshold_cross = signal.get("threshold_cross_ts_ms")
    detail = {
        "trade_id": trade.get("trade_id"),
        "signal_ts_utc": trade.get("signal_ts_utc"),
        "entry_ts_utc": trade.get("entry_ts_utc"),
        "exit_reason": trade.get("exit_reason"),
        "net_bps": trade.get("net_bps"),
        "cluster_start_ts_ms": start,
        "cluster_end_ts_ms": end,
        "threshold_cross_ts_ms": threshold_cross,
    }
    if sig is None or start is None or end is None:
        return "missing_old_format", detail
    start_lag = (int(sig) - int(start)) / 1000.0
    end_lag = (int(end) - int(sig)) / 1000.0
    threshold_lag = None if threshold_cross is None else (int(sig) - int(threshold_cross)) / 1000.0
    detail.update(
        {
            "start_to_signal_sec": start_lag,
            "signal_to_end_sec": end_lag,
            "signal_minus_threshold_cross_sec": threshold_lag,
        }
    )
    if abs(start_lag) <= 1.0 and end_lag > 1.0:
        return "lookahead_like_cluster_start_entry", detail
    return "knowable_like_threshold_or_cluster_end", detail


def main() -> None:
    data = json.loads(TRADES_PATH.read_text(encoding="utf-8"))
    trades = data.get("trades", data if isinstance(data, list) else [])
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {"generated_at_utc": generated_at, "rules": {}}

    for rule in RULES:
        rows = [
            trade
            for trade in trades
            if (trade.get("rule", {}).get("name") or trade.get("rule_name")) == rule
            and trade.get("status") == "CLOSED"
            and trade.get("net_bps") is not None
        ]
        buckets: dict[str, list[dict]] = {}
        for trade in rows:
            label, detail = classify_trade(trade)
            buckets.setdefault(label, []).append(detail)
        payload["rules"][rule] = {
            "closed_n": len(rows),
            "overall": summarize([float(row["net_bps"]) for row in rows]),
            "classes": {
                label: {
                    "summary": summarize([float(row["net_bps"]) for row in bucket]),
                    "trades": bucket,
                }
                for label, bucket in sorted(buckets.items())
            },
        }

    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S34 Anchor Integrity Audit",
        "",
        f"Generated: `{generated_at}`",
        "",
        "Classifies closed forward paper trades by whether the recorded signal anchor is compatible with a knowable threshold-cross anchor.",
        "",
        "Definitions:",
        "- `lookahead_like_cluster_start_entry`: `signal_ts` is within 1s of `cluster_start`, while `cluster_end` is later. This records full cluster notional while entering at the cluster start.",
        "- `knowable_like_threshold_or_cluster_end`: signal is not at cluster start, or equals stored threshold/cluster end.",
        "- `missing_old_format`: older trade snapshot lacks cluster start/end fields.",
        "",
    ]
    for rule, section in payload["rules"].items():
        overall = section["overall"]
        lines += [
            f"## {rule}",
            "",
            f"Overall closed N={overall['n']}, median={overall.get('median')}, mean={overall.get('mean')}, cum={overall.get('cum')}, WR={overall.get('wr')}",
            "",
            "| Anchor Class | N | Median | Mean | Cum | WR |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for label, item in section["classes"].items():
            st = item["summary"]
            lines.append(f"| {label} | {st['n']} | {st.get('median')} | {st.get('mean')} | {st.get('cum')} | {st.get('wr')} |")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"MD: {OUT_MD}")
    print(f"JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
