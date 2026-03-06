from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List

from tools.micro_edge_lib import build_bucket_features, compute_rule_thresholds, rule_fires
from tools.micro_edge_smoke import _load_symbol_trades_marks_and_liqs
from tools.run_summary import build_run_summary


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure coverage of liquidation-driven rules across lookback windows.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--lookbacks-min", default="1440,10080,30240")
    p.add_argument("--bucket-sec", type=int, default=5)
    p.add_argument("--rule", default="high_liq_reversal_regime")
    p.add_argument("--out-json", default="reports/LIQUIDATION_RULE_COVERAGE.json")
    p.add_argument("--out-md", default="reports/LIQUIDATION_RULE_COVERAGE.md")
    return p.parse_args()


def _parse_int_list(raw: str) -> List[int]:
    vals: List[int] = []
    for tok in str(raw or "").replace(";", ",").split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    return vals


def _load_rows(db: str, symbol: str, lookback_min: int, bucket_sec: int) -> List[Dict[str, Any]]:
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(max(1, lookback_min) * 60 * 1000)
    conn = sqlite3.connect(str(db))
    try:
        trades, marks, liqs = _load_symbol_trades_marks_and_liqs(
            conn,
            str(symbol).upper(),
            start_ms=start_ms,
            end_ms=now_ms,
        )
    finally:
        conn.close()
    return build_bucket_features(
        trades=trades,
        marks=marks,
        liqs=liqs,
        bucket_sec=max(1, int(bucket_sec)),
        vol_window=max(4, int(60 / max(1, bucket_sec))),
    )


def _measure_rule(rows: List[Dict[str, Any]], rule_name: str) -> Dict[str, Any]:
    thr = compute_rule_thresholds(rows)
    fired = [r for r in rows if rule_fires(rule_name, r, thr)]
    liq_nonzero = [r for r in rows if float(r.get("liq_rate_per_sec") or 0.0) > 0.0]
    return {
        "bucket_rows": int(len(rows)),
        "liq_rows": int(len(liq_nonzero)),
        "rule_fire_count": int(len(fired)),
        "rule_fire_rate": (float(len(fired)) / float(len(rows))) if rows else 0.0,
        "rule_given_liq_rate": (float(len(fired)) / float(len(liq_nonzero))) if liq_nonzero else 0.0,
    }


def main() -> int:
    args = _args()
    lookbacks = _parse_int_list(args.lookbacks_min)
    results: List[Dict[str, Any]] = []
    for lb in lookbacks:
        rows = _load_rows(args.db, args.symbol, lb, args.bucket_sec)
        measure = _measure_rule(rows, args.rule)
        measure["lookback_min"] = int(lb)
        results.append(measure)

    payload = {
        "symbol": str(args.symbol).upper(),
        "rule": str(args.rule),
        "bucket_sec": int(args.bucket_sec),
        "results": results,
    }
    payload["run_summary"] = build_run_summary(
        run_type="liquidation_rule_coverage",
        inputs={
            "db": str(args.db),
            "symbol": str(args.symbol).upper(),
            "lookbacks_min": lookbacks,
            "bucket_sec": int(args.bucket_sec),
            "rule": str(args.rule),
        },
        metrics={
            "windows": int(len(results)),
            "max_rule_fire_count": int(max((r["rule_fire_count"] for r in results), default=0)),
        },
        artifacts={"json": str(args.out_json), "md": str(args.out_md)},
    )

    out_json = Path(str(args.out_json))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LIQUIDATION_RULE_COVERAGE",
        "",
        f"symbol={payload['symbol']} rule={payload['rule']} bucket_sec={payload['bucket_sec']}",
        "",
        "| lookback_min | bucket_rows | liq_rows | rule_fire_count | rule_fire_rate | rule_given_liq_rate |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {int(row['lookback_min'])} | {int(row['bucket_rows'])} | {int(row['liq_rows'])} | "
            f"{int(row['rule_fire_count'])} | {float(row['rule_fire_rate']):.4%} | {float(row['rule_given_liq_rate']):.4%} |"
        )
    out_md = Path(str(args.out_md))
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
