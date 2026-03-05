from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.microphys.impact.models import bucket_impact, fit_impact_models
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate linear vs sqrt price impact from physics parquet.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--out", default="data/derived/impact")
    p.add_argument("--out-report", default=None)
    return p.parse_args()


def _load_partitions(physics_root: Path, symbol: str, interval_ms: int) -> list[tuple[str, pd.DataFrame]]:
    base = physics_root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    out: list[tuple[str, pd.DataFrame]] = []
    for p in sorted(base.glob("date=*/physics.parquet")):
        day = p.parent.name.split("=", 1)[1]
        out.append((day, pd.read_parquet(p)))
    return out


def estimate_impact(physics_root: Path, out_root: Path, symbol: str, interval_ms: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sym = canonical_symbol(symbol)
    parts = _load_partitions(physics_root, sym, interval_ms)
    if not parts:
        raise RuntimeError(f"no_physics_partitions symbol={sym} interval_ms={interval_ms}")

    daily_rows: list[dict[str, Any]] = []
    bucket_rows: list[pd.DataFrame] = []
    for day, df in parts:
        vol = pd.to_numeric(df.get("volume_proxy"), errors="coerce").fillna(0.0)
        abs_ret = pd.to_numeric(df.get("r_1"), errors="coerce").abs()
        fits = fit_impact_models(vol, abs_ret)
        buck = bucket_impact(vol, abs_ret, q=10)
        if not buck.empty:
            buck = buck.copy()
            buck["date"] = day
            bucket_rows.append(buck)
        daily_rows.append(
            {
                "date": day,
                "symbol": sym,
                "interval_ms": int(interval_ms),
                "n": int(fits["linear"].n),
                "linear_alpha": float(fits["linear"].alpha),
                "linear_beta": float(fits["linear"].beta),
                "linear_r2": float(fits["linear"].r2),
                "sqrt_alpha": float(fits["sqrt"].alpha),
                "sqrt_beta": float(fits["sqrt"].beta),
                "sqrt_r2": float(fits["sqrt"].r2),
                "winner": "sqrt" if fits["sqrt"].r2 >= fits["linear"].r2 else "linear",
            }
        )

    daily_df = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    bucket_df = pd.concat(bucket_rows, ignore_index=True) if bucket_rows else pd.DataFrame(columns=["date", "bucket", "count", "volume_mean", "abs_return_mean"])

    out_base = out_root / f"interval_ms={int(interval_ms)}" / f"symbol={sym}"
    out_base.mkdir(parents=True, exist_ok=True)
    daily_df.to_parquet(out_base / "impact_daily.parquet", index=False)
    bucket_df.to_parquet(out_base / "impact_buckets.parquet", index=False)
    manifest = {
        "symbol": sym,
        "interval_ms": int(interval_ms),
        "rows_daily": int(len(daily_df)),
        "rows_buckets": int(len(bucket_df)),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_base / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return daily_df, bucket_df


def _write_report(daily_df: pd.DataFrame, bucket_df: pd.DataFrame, symbol: str, interval_ms: int, report_path: Path) -> None:
    plots_dir = report_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # daily r2 comparison
    p1 = plots_dir / f"impact_r2_{symbol}_{interval_ms}ms.png"
    plt.figure(figsize=(8, 4))
    if not daily_df.empty:
        x = np.arange(len(daily_df))
        plt.plot(x, daily_df["linear_r2"], label="linear_r2")
        plt.plot(x, daily_df["sqrt_r2"], label="sqrt_r2")
        plt.legend()
    plt.title(f"Impact Model R2 - {symbol} {interval_ms}ms")
    plt.xlabel("day index")
    plt.ylabel("R2")
    plt.tight_layout()
    plt.savefig(p1)
    plt.close()

    # bucket curve
    p2 = plots_dir / f"impact_bucket_curve_{symbol}_{interval_ms}ms.png"
    plt.figure(figsize=(8, 4))
    if not bucket_df.empty:
        gb = bucket_df.groupby("bucket", as_index=False).agg(volume_mean=("volume_mean", "mean"), abs_return_mean=("abs_return_mean", "mean"))
        plt.plot(gb["volume_mean"], gb["abs_return_mean"], marker="o")
    plt.title(f"Impact Bucket Curve - {symbol}")
    plt.xlabel("volume_mean")
    plt.ylabel("mean |r_1|")
    plt.tight_layout()
    plt.savefig(p2)
    plt.close()

    lines: list[str] = []
    lines.append(f"# Impact Report - {symbol} ({interval_ms}ms)")
    lines.append("")
    lines.append(f"- days: `{len(daily_df)}`")
    lines.append(f"- sqrt beats linear: `{int((daily_df['winner'] == 'sqrt').sum()) if not daily_df.empty else 0}` days")
    if not daily_df.empty:
        lines.append(
            f"- mean linear_r2=`{float(daily_df['linear_r2'].mean()):.6f}` mean sqrt_r2=`{float(daily_df['sqrt_r2'].mean()):.6f}`"
        )
        lines.append(
            f"- mean linear_beta=`{float(daily_df['linear_beta'].mean()):.6e}` mean sqrt_beta=`{float(daily_df['sqrt_beta'].mean()):.6e}`"
        )
    lines.append("")
    lines.append("## Daily Regression Summary")
    lines.append("")
    lines.append("| date | n | linear_r2 | sqrt_r2 | winner | linear_beta | sqrt_beta |")
    lines.append("|---|---:|---:|---:|---|---:|---:|")
    for _, r in daily_df.iterrows():
        lines.append(
            f"| {r['date']} | {int(r['n'])} | {float(r['linear_r2']):.6f} | {float(r['sqrt_r2']):.6f} | {r['winner']} | {float(r['linear_beta']):.6e} | {float(r['sqrt_beta']):.6e} |"
        )
    lines.append("")
    lines.append("## Bucket Summary (all days)")
    lines.append("")
    if bucket_df.empty:
        lines.append("- no bucket rows")
    else:
        gb = bucket_df.groupby("bucket", as_index=False).agg(count=("count", "sum"), volume_mean=("volume_mean", "mean"), abs_return_mean=("abs_return_mean", "mean"))
        lines.append("| bucket | count | volume_mean | abs_return_mean |")
        lines.append("|---:|---:|---:|---:|")
        for _, r in gb.iterrows():
            lines.append(f"| {int(r['bucket'])} | {int(r['count'])} | {float(r['volume_mean']):.6f} | {float(r['abs_return_mean']):.8f} |")
    lines.append("")
    lines.append(f"- plot r2: `{p1}`")
    lines.append(f"- plot buckets: `{p2}`")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    try:
        daily_df, bucket_df = estimate_impact(
            physics_root=Path(str(args.physics)),
            out_root=Path(str(args.out)),
            symbol=str(args.symbol),
            interval_ms=int(args.interval_ms),
        )
        report = Path(str(args.out_report)) if args.out_report else Path(f"reports/impact_{canonical_symbol(args.symbol)}_{int(args.interval_ms)}ms.md")
        _write_report(daily_df, bucket_df, canonical_symbol(args.symbol), int(args.interval_ms), report)
        print(f"estimate_impact ok rows_daily={len(daily_df)} rows_buckets={len(bucket_df)} report={report}")
        return 0
    except Exception as e:
        print(f"estimate_impact error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
