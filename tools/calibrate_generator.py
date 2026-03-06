from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.microphys.alpha.calibration import compute_calibration, save_calibration
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate quantile thresholds for alpha generator.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--out", default="")
    p.add_argument("--report", default="")
    return p.parse_args()


def _load_recent(root: Path, symbol: str, interval_ms: int, days: int) -> pd.DataFrame:
    base = root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    parts = sorted(base.glob("date=*/physics.parquet"))
    if not parts:
        return pd.DataFrame()
    files = parts[-max(1, int(days)) :]
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def main() -> int:
    args = _parse_args()
    try:
        sym = canonical_symbol(args.symbol)
        frame = _load_recent(Path(str(args.physics)), sym, int(args.interval_ms), int(args.days))
        if frame.empty:
            raise RuntimeError("physics_missing")
        cols = [
            "F_ofi_z",
            "F_intensity_z",
            "spread_z",
            "rv_short",
            "rv_z",
            "top_depth_imbalance",
            "liq_rate_z",
            "micro_trend",
        ]
        present = [c for c in cols if c in frame.columns]
        ctx = compute_calibration(frame, columns=present)
        out_dir = (
            Path(str(args.out))
            if str(args.out).strip()
            else Path(f"data/derived/alpha_candidates/symbol={sym}/interval_ms={int(args.interval_ms)}")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        cal_path = out_dir / "calibration.json"
        save_calibration(ctx, cal_path)
        report = (
            Path(str(args.report))
            if str(args.report).strip()
            else Path(f"reports/generator_calibration_{sym}_{int(args.interval_ms)}ms.md")
        )
        lines = [
            f"# Generator Calibration - {sym} ({int(args.interval_ms)}ms)",
            "",
            f"- sample_count: `{ctx.sample_count}`",
            f"- days: `{int(args.days)}`",
            "",
            "| column | nan_ratio | q10 | q50 | q90 | q95 | q99 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for col in sorted(present):
            lines.append(
                f"| {col} | {float(ctx.nan_ratio.get(col, 1.0)):.4f} | {ctx.q(col, 0.10):.8f} | {ctx.q(col, 0.50):.8f} | {ctx.q(col, 0.90):.8f} | {ctx.q(col, 0.95):.8f} | {ctx.q(col, 0.99):.8f} |"
            )
        lines.append("")
        lines.append(f"- calibration: `{cal_path}`")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"calibrate_generator ok calibration={cal_path} report={report}")
        return 0
    except Exception as e:
        print(f"calibrate_generator error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
