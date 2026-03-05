from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.microphys.execution.calibration import load_execution_params
from src.microphys.execution.fill_models import HazardParams, simulate_maker_hazard_fill
from src.microphys.execution.queue_sim import QueueSimParams, simulate_maker_queue_fill
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate maker fills from execution features.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--exec-features", default="data/derived/execution_sim")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--model", choices=["maker_queue", "maker_hazard"], default="maker_queue")
    p.add_argument("--params", default="")
    p.add_argument("--ttl-bars", type=int, default=10)
    p.add_argument("--out", default="data/derived/execution_sim")
    return p.parse_args()


def _load_partitioned(root: Path, symbol: str, interval_ms: int, fname: str) -> pd.DataFrame:
    base = root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    files = sorted(base.glob(f"date=*/{fname}"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).sort_values("ts_ms").reset_index(drop=True)


def main() -> int:
    args = _parse_args()
    try:
        symbol = canonical_symbol(args.symbol)
        physics = _load_partitioned(Path(str(args.physics)), symbol, int(args.interval_ms), "physics.parquet")
        exf = _load_partitioned(Path(str(args.exec_features)), symbol, int(args.interval_ms), "exec_features.parquet")
        if physics.empty:
            raise RuntimeError("physics_missing")
        frame = physics.merge(exf, on=["ts_ms", "ts_utc", "symbol"], how="left")
        params = load_execution_params(Path(str(args.params))) if str(args.params).strip() else {}

        entries = pd.to_numeric(frame.get("F_ofi_z"), errors="coerce").fillna(0.0).abs() > 1.0
        out_rows = []
        for i in frame.index[entries]:
            side = "buy" if float(pd.to_numeric(frame.get("F_ofi_z"), errors="coerce").iloc[i]) >= 0 else "sell"
            if args.model == "maker_hazard":
                hp = HazardParams(**{**{"ttl_bars": int(args.ttl_bars)}, **params.get("maker_hazard", {})})
                sim = simulate_maker_hazard_fill(frame, entry_idx=int(i), side=side, params=hp)
            else:
                qp = QueueSimParams(**{**{"ttl_bars": int(args.ttl_bars)}, **params.get("maker_queue", {})})
                sim = simulate_maker_queue_fill(frame, entry_idx=int(i), side=side, params=qp)
            out_rows.append(
                {
                    "entry_idx": int(i),
                    "ts_ms": int(frame.iloc[i]["ts_ms"]),
                    "ts_utc": str(frame.iloc[i]["ts_utc"]),
                    "side": side,
                    **sim,
                }
            )
        out = pd.DataFrame(out_rows)
        out_base = Path(str(args.out)) / f"interval_ms={int(args.interval_ms)}" / f"symbol={symbol}"
        out_base.mkdir(parents=True, exist_ok=True)
        out_path = out_base / "maker_fill_sims.parquet"
        out.to_parquet(out_path, index=False)
        print(f"simulate_maker_fills ok rows={len(out)} out={out_path}")
        return 0
    except Exception as e:
        print(f"simulate_maker_fills error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
