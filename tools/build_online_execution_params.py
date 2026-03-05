from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.microphys.execution.calibration import calibrate_execution_models, save_execution_params
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build online execution params artifact from recent physics data.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--out", default="data/live/artifacts/execution")
    p.add_argument("--report", default="")
    return p.parse_args()


def _load_recent(physics_root: Path, symbol: str, interval_ms: int, days: int) -> pd.DataFrame:
    base = physics_root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    files = sorted(base.glob("date=*/physics.parquet"))
    if days > 0:
        files = files[-int(days) :]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def main() -> int:
    args = _parse_args()
    try:
        symbol = canonical_symbol(args.symbol)
        frame = _load_recent(Path(str(args.physics)), symbol, int(args.interval_ms), int(args.days))
        if frame.empty:
            raise RuntimeError("physics_missing")
        params = calibrate_execution_models(frame)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d")
        out = Path(str(args.out))
        out.mkdir(parents=True, exist_ok=True)
        params_path = out / f"params_{ts}.json"
        save_execution_params(params_path, params)
        if str(args.report).strip():
            r = Path(str(args.report))
            r.parent.mkdir(parents=True, exist_ok=True)
            r.write_text(
                "\n".join(
                    [
                        f"# Online Execution Params - {symbol}",
                        "",
                        f"- rows: `{len(frame)}`",
                        f"- params: `{params_path}`",
                        f"- maker_queue.queue_frac: `{float(params.get('maker_queue', {}).get('queue_frac', 0.0)):.4f}`",
                        f"- maker_hazard.fill_threshold: `{float(params.get('maker_hazard', {}).get('fill_threshold', 0.0)):.4f}`",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        print(f"build_online_execution_params ok path={params_path}")
        return 0
    except Exception as e:
        print(f"build_online_execution_params error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

