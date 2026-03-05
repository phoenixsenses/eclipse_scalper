from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.microphys.alpha.calibration import compute_calibration, save_calibration
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build online calibration artifact from recent physics data.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--out", default="data/live/artifacts/calibration")
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
        cols = [c for c in ("F_ofi_z", "F_intensity_z", "spread_z", "rv_short", "rv_z", "top_depth_imbalance", "liq_rate_z") if c in frame.columns]
        ctx = compute_calibration(frame, columns=cols)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d")
        out = Path(str(args.out))
        out.mkdir(parents=True, exist_ok=True)
        cal_path = out / f"calibration_{ts}.json"
        save_calibration(ctx, cal_path)
        print(f"build_online_calibration ok path={cal_path} sample_count={ctx.sample_count}")
        return 0
    except Exception as e:
        print(f"build_online_calibration error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

