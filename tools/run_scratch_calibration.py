from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run scratch calibration sweeps and write markdown summaries.")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--out-sell", default="reports/SCRATCH_CALIBRATION_SELL_UP.md")
    p.add_argument("--out-buy", default="reports/SCRATCH_CALIBRATION_BUY_UP.md")
    p.add_argument("--adverse-sweep", default="2.0:10.0")
    p.add_argument("--trail-sweep", default="2.0,3.0,4.0,5.0")
    p.add_argument("--fee-bps", type=float, default=0.5)
    p.add_argument("--exec-model", default="passive_realistic", choices=["passive_realistic", "taker"])
    return p.parse_args()


def _run_side(
    side: str,
    db: str,
    symbol: str,
    out_md: Path,
    *,
    adverse_sweep: str,
    trail_sweep: str,
    fee_bps: float,
    exec_model: str,
) -> int:
    out_json = out_md.with_suffix(".json")
    cmd = [
        sys.executable,
        "-m",
        "tools.backtest_scratch",
        "--db",
        db,
        "--symbol",
        symbol,
        "--side",
        side,
        "--adverse-sweep",
        str(adverse_sweep),
        "--trail-sweep",
        str(trail_sweep),
        "--fee-bps",
        str(float(fee_bps)),
        "--exec-model",
        str(exec_model),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    return int(subprocess.call(cmd))


def main() -> int:
    args = _args()
    out_sell = Path(args.out_sell)
    out_buy = Path(args.out_buy)
    out_sell.parent.mkdir(parents=True, exist_ok=True)
    rc1 = _run_side(
        "sell",
        db=str(args.db),
        symbol=str(args.symbol),
        out_md=out_sell,
        adverse_sweep=str(args.adverse_sweep),
        trail_sweep=str(args.trail_sweep),
        fee_bps=float(args.fee_bps),
        exec_model=str(args.exec_model),
    )
    rc2 = _run_side(
        "buy",
        db=str(args.db),
        symbol=str(args.symbol),
        out_md=out_buy,
        adverse_sweep=str(args.adverse_sweep),
        trail_sweep=str(args.trail_sweep),
        fee_bps=float(args.fee_bps),
        exec_model=str(args.exec_model),
    )
    if rc1 != 0 and not out_sell.exists():
        out_sell.write_text("# SCRATCH CALIBRATION SELL_UP\n\nbacktest_scratch execution failed.\n", encoding="utf-8")
    if rc2 != 0 and not out_buy.exists():
        out_buy.write_text("# SCRATCH CALIBRATION BUY_UP\n\nbacktest_scratch execution failed.\n", encoding="utf-8")
    print(f"run_scratch_calibration: rc_sell={rc1} rc_buy={rc2}")
    return 0 if (rc1 == 0 and rc2 == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
