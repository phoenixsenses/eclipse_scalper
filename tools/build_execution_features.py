from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.microphys.execution.features import build_execution_features
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build execution micro-features from micro bars.")
    p.add_argument("--micro-bars", default="data/derived/micro_bars")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--out", default="data/derived/execution_sim")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        symbol = canonical_symbol(args.symbol)
        src = Path(str(args.micro_bars)) / f"interval_ms={int(args.interval_ms)}" / f"symbol={symbol}"
        files = sorted(src.glob("date=*/bars.parquet"))
        if not files:
            raise RuntimeError("micro_bars_missing")
        out_base = Path(str(args.out)) / f"interval_ms={int(args.interval_ms)}" / f"symbol={symbol}"
        out_base.mkdir(parents=True, exist_ok=True)
        parts = []
        for f in files:
            day = f.parent.name
            df = pd.read_parquet(f)
            ex = build_execution_features(df)
            out_day = out_base / day
            out_day.mkdir(parents=True, exist_ok=True)
            out_pq = out_day / "exec_features.parquet"
            ex.to_parquet(out_pq, index=False)
            parts.append({"date": day.split("=", 1)[-1], "rows": int(len(ex)), "output": str(out_pq)})
        (out_base / "manifest.json").write_text(
            json.dumps({"symbol": symbol, "interval_ms": int(args.interval_ms), "partitions": parts}, ensure_ascii=True, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(f"build_execution_features ok symbol={symbol} partitions={len(parts)}")
        return 0
    except Exception as e:
        print(f"build_execution_features error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
