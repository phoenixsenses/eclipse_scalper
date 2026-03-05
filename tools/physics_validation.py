from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate empirical microstructure physics relationships.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--impact", default="data/derived/impact")
    p.add_argument("--propagator", default="data/derived/propagator")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _load_all_physics(root: Path, symbol: str, interval_ms: int) -> pd.DataFrame:
    base = root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    files = sorted(base.glob("date=*/physics.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def _ofi_decile_stats(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["F_ofi_z", "r_1"]].copy().replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return pd.DataFrame(columns=["decile", "count", "mean_next_ret", "direction_acc"])
    try:
        d["decile"] = pd.qcut(d["F_ofi_z"], q=10, labels=False, duplicates="drop")
    except Exception:
        return pd.DataFrame(columns=["decile", "count", "mean_next_ret", "direction_acc"])
    d["correct"] = (np.sign(d["F_ofi_z"]) == np.sign(d["r_1"]))
    g = (
        d.groupby("decile", as_index=False)
        .agg(count=("r_1", "size"), mean_next_ret=("r_1", "mean"), direction_acc=("correct", "mean"))
        .sort_values("decile")
    )
    return g


def _kernel_decay_quality(kernel: pd.DataFrame) -> dict[str, float]:
    if kernel.empty:
        return {"smooth_decay_score": 0.0, "positive_lag1": 0.0}
    abs_resp = pd.to_numeric(kernel["abs_response"], errors="coerce").fillna(0.0)
    # Higher score => smoother decay (fewer upward jumps)
    diff = abs_resp.diff().fillna(0.0)
    jumps = float((diff > 0).sum())
    score = 1.0 - (jumps / max(1.0, float(len(abs_resp))))
    lag1 = float(kernel.iloc[0]["response"]) if len(kernel) else 0.0
    return {"smooth_decay_score": score, "positive_lag1": lag1}


def main() -> int:
    args = _parse_args()
    try:
        sym = canonical_symbol(args.symbol)
        interval = int(args.interval_ms)

        physics = _load_all_physics(Path(str(args.physics)), sym, interval)
        if physics.empty:
            raise RuntimeError("physics_missing")

        impact_path = Path(str(args.impact)) / f"interval_ms={interval}" / f"symbol={sym}" / "impact_daily.parquet"
        prop_path = Path(str(args.propagator)) / f"interval_ms={interval}" / f"symbol={sym}" / "kernel.parquet"

        impact = pd.read_parquet(impact_path) if impact_path.exists() else pd.DataFrame()
        kernel = pd.read_parquet(prop_path) if prop_path.exists() else pd.DataFrame()

        dec = _ofi_decile_stats(physics)

        sqrt_better = float((impact["sqrt_r2"] > impact["linear_r2"]).mean()) if not impact.empty else 0.0
        decay = _kernel_decay_quality(kernel)

        out_path = Path(str(args.out)) if args.out else Path(f"reports/physics_validation_{sym}_{interval}ms.md")
        plots_dir = out_path.parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        p_dec = plots_dir / f"physics_ofi_decile_{sym}_{interval}ms.png"
        plt.figure(figsize=(8, 4))
        if not dec.empty:
            plt.bar(dec["decile"].astype(int).astype(str), dec["mean_next_ret"])
        plt.title(f"OFI z-decile vs next return {sym}")
        plt.xlabel("decile")
        plt.ylabel("mean r_1")
        plt.tight_layout()
        plt.savefig(p_dec)
        plt.close()

        lines: list[str] = []
        lines.append(f"# Physics Validation - {sym} ({interval}ms)")
        lines.append("")
        lines.append("## OFI Predictive Power")
        lines.append("")
        lines.append(f"- samples: `{len(physics)}`")
        if dec.empty:
            lines.append("- decile table unavailable")
        else:
            lines.append("| decile | count | mean_next_ret | direction_acc |")
            lines.append("|---:|---:|---:|---:|")
            for _, r in dec.iterrows():
                lines.append(f"| {int(r['decile'])} | {int(r['count'])} | {float(r['mean_next_ret']):.10f} | {float(r['direction_acc']):.4f} |")
        lines.append(f"- plot: `{p_dec}`")
        lines.append("")
        lines.append("## Impact Law")
        lines.append("")
        lines.append(f"- impact rows: `{len(impact)}`")
        lines.append(f"- sqrt beats linear ratio: `{sqrt_better:.4f}`")
        if not impact.empty:
            lines.append(
                f"- mean linear_r2=`{float(impact['linear_r2'].mean()):.6f}` mean sqrt_r2=`{float(impact['sqrt_r2'].mean()):.6f}`"
            )
        lines.append("")
        lines.append("## Propagator")
        lines.append("")
        lines.append(f"- kernel rows: `{len(kernel)}`")
        lines.append(f"- lag1 response: `{decay['positive_lag1']:.8e}`")
        lines.append(f"- smooth decay score: `{decay['smooth_decay_score']:.4f}`")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"physics_validation ok out={out_path}")
        return 0
    except Exception as e:
        print(f"physics_validation error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
