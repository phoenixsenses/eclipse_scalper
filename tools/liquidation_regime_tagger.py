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


def _load_rows(db: str, symbol: str, lookback_min: int, bucket_sec: int) -> List[Dict[str, Any]]:
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(max(1, lookback_min) * 60 * 1000)
    conn = sqlite3.connect(str(db))
    try:
        # root points at the target db's own directory (no real archive
        # catalog_index.json there) so plan_read() falls through to a
        # direct sqlite read; source_db_path is pinned explicitly so
        # execute_read() never falls back to the real data/microstructure.db.
        trades, marks, liqs = _load_symbol_trades_marks_and_liqs(
            conn, str(symbol).upper(), start_ms=start_ms, end_ms=now_ms,
            root=str(Path(db).resolve().parent), source_db_path=str(db),
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


def _tag_rows(rows: List[Dict[str, Any]], rule_name: str) -> List[Dict[str, Any]]:
    thresholds = compute_rule_thresholds(rows)
    tagged: List[Dict[str, Any]] = []
    for row in rows:
        fired = bool(rule_fires(rule_name, row, thresholds))
        tag = "high_liq_reversal" if fired else "normal"
        tagged.append(
            {
                "ts_ms": int(float(row.get("ts_ms", 0.0) or 0.0)),
                "tag": tag,
                "rule_fired": fired,
                "mid": row.get("mid"),
                "spread": row.get("spread"),
                "imbalance": row.get("imbalance"),
                "trade_intensity": row.get("trade_intensity"),
                "ret_1": row.get("ret_1"),
                "liq_count": row.get("liq_count"),
                "liq_qty": row.get("liq_qty"),
                "liq_imbalance": row.get("liq_imbalance"),
                "liq_rate_per_sec": row.get("liq_rate_per_sec"),
            }
        )
    return tagged


def _summarize(tags: List[Dict[str, Any]]) -> Dict[str, Any]:
    fired = [r for r in tags if bool(r.get("rule_fired"))]
    return {
        "rows_total": int(len(tags)),
        "tagged_count": int(len(fired)),
        "tagged_rate": (float(len(fired)) / float(len(tags))) if tags else 0.0,
        "avg_liq_rate_tagged": (
            sum(float(r.get("liq_rate_per_sec", 0.0) or 0.0) for r in fired) / float(len(fired))
            if fired
            else 0.0
        ),
        "avg_abs_imbalance_tagged": (
            sum(abs(float(r.get("liq_imbalance", 0.0) or 0.0)) for r in fired) / float(len(fired))
            if fired
            else 0.0
        ),
    }


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tag liquidation reversal regimes from microstructure DB.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--lookback-min", type=int, default=1440)
    p.add_argument("--bucket-sec", type=int, default=5)
    p.add_argument("--rule", default="high_liq_reversal_regime")
    p.add_argument("--out-json", default="reports/LIQUIDATION_REGIME_TAGGER.json")
    p.add_argument("--out-md", default="reports/LIQUIDATION_REGIME_TAGGER.md")
    return p.parse_args()


def main() -> int:
    args = _args()
    rows = _load_rows(str(args.db), str(args.symbol), int(args.lookback_min), int(args.bucket_sec))
    tags = _tag_rows(rows, str(args.rule))
    summary = _summarize(tags)

    payload = {
        "symbol": str(args.symbol).upper(),
        "rule": str(args.rule),
        "lookback_min": int(args.lookback_min),
        "bucket_sec": int(args.bucket_sec),
        "summary": summary,
        "tags": tags,
    }
    payload["run_summary"] = build_run_summary(
        run_type="liquidation_regime_tagger",
        inputs={
            "db": str(args.db),
            "symbol": str(args.symbol).upper(),
            "lookback_min": int(args.lookback_min),
            "bucket_sec": int(args.bucket_sec),
            "rule": str(args.rule),
        },
        metrics={
            "rows_total": int(summary["rows_total"]),
            "tagged_count": int(summary["tagged_count"]),
            "tagged_rate": float(summary["tagged_rate"]),
        },
        artifacts={"json": str(args.out_json), "md": str(args.out_md)},
    )

    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LIQUIDATION REGIME TAGGER",
        "",
        f"symbol={payload['symbol']} rule={payload['rule']} lookback_min={payload['lookback_min']} bucket_sec={payload['bucket_sec']}",
        f"rows_total={summary['rows_total']} tagged_count={summary['tagged_count']} tagged_rate={float(summary['tagged_rate']):.2%}",
        f"avg_liq_rate_tagged={float(summary['avg_liq_rate_tagged']):.4f}",
        f"avg_abs_imbalance_tagged={float(summary['avg_abs_imbalance_tagged']):.4f}",
        "",
        "| ts_ms | tag | liq_rate_per_sec | liq_imbalance | trade_intensity | spread | ret_1 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in tags[:50]:
        lines.append(
            f"| {int(row['ts_ms'])} | {row['tag']} | {float(row.get('liq_rate_per_sec', 0.0) or 0.0):.4f} | "
            f"{float(row.get('liq_imbalance', 0.0) or 0.0):+.4f} | {float(row.get('trade_intensity', 0.0) or 0.0):.2f} | "
            f"{float(row.get('spread', 0.0) or 0.0):.6f} | {float(row.get('ret_1', 0.0) or 0.0):+.6f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
