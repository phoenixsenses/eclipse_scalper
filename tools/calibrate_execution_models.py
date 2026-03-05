from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.microphys.execution.calibration import calibrate_execution_models, save_execution_params
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate execution fill/adverse model params.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--out", default="data/derived/execution_calibration")
    p.add_argument("--out-params", default="")
    p.add_argument("--report", default="")
    p.add_argument("--days", type=int, default=0, help="use only last N date partitions if >0")
    return p.parse_args()


def _load_partitioned(root: Path, symbol: str, interval_ms: int) -> pd.DataFrame:
    base = root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    files = sorted(base.glob("date=*/physics.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).sort_values("ts_ms").reset_index(drop=True)


def main() -> int:
    args = _parse_args()
    try:
        symbol = canonical_symbol(args.symbol)
        physics = _load_partitioned(Path(str(args.physics)), symbol, int(args.interval_ms))
        if int(args.days) > 0 and not physics.empty and "ts_utc" in physics.columns:
            t = pd.to_datetime(physics["ts_utc"], utc=True, errors="coerce")
            cutoff = t.max() - pd.Timedelta(days=int(args.days))
            physics = physics[t >= cutoff].copy()
        if physics.empty:
            raise RuntimeError("physics_missing")
        params = calibrate_execution_models(physics)
        if str(args.out_params).strip():
            out_path = Path(str(args.out_params))
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_base = Path(str(args.out)) / f"interval_ms={int(args.interval_ms)}" / f"symbol={symbol}"
            out_base.mkdir(parents=True, exist_ok=True)
            out_path = out_base / "params.json"
        save_execution_params(out_path, params)
        report = Path(str(args.report)) if str(args.report).strip() else Path(f"reports/execution_realism_{symbol}_{int(args.interval_ms)}ms.md")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "\n".join(
                [
                    f"# Execution Calibration - {symbol} ({int(args.interval_ms)}ms)",
                    "",
                    f"- rows: `{len(physics)}`",
                    f"- params: `{out_path}`",
                    f"- maker_queue.queue_frac: `{float(params['maker_queue']['queue_frac']):.4f}`",
                    f"- maker_hazard.a: `{float(params['maker_hazard']['a']):.4f}`",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"calibrate_execution_models ok out={out_path} report={report}")
        return 0
    except Exception as e:
        print(f"calibrate_execution_models error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
