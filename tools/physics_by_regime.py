from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.microphys.analysis.regime_metrics import compute_regime_metrics
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute physics metrics by regime.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--regimes", default="data/derived/regimes")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--tau-max", type=int, default=200)
    p.add_argument("--out", default="data/derived/physics_regime_metrics")
    p.add_argument("--report", default=None)
    return p.parse_args()


def _iter_pairs(physics_root: Path, regime_root: Path, symbol: str, interval_ms: int) -> list[tuple[str, Path, Path]]:
    pbase = physics_root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    rbase = regime_root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    out: list[tuple[str, Path, Path]] = []
    if not pbase.exists() or not rbase.exists():
        return out
    for p in sorted(pbase.glob("date=*/physics.parquet")):
        day = p.parent.name.split("=", 1)[1]
        rp = rbase / f"date={day}" / "regimes.parquet"
        if rp.exists():
            out.append((day, p, rp))
    return out


def run_metrics(physics_root: Path, regime_root: Path, out_root: Path, symbol: str, interval_ms: int, tau_max: int) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    sym = canonical_symbol(symbol)
    pairs = _iter_pairs(physics_root, regime_root, sym, interval_ms)
    if not pairs:
        raise RuntimeError(f"no_matching_partitions symbol={sym} interval_ms={interval_ms}")

    merged_all = []
    for day, pp, rp in pairs:
        p = pd.read_parquet(pp)
        r = pd.read_parquet(rp)
        m = p.merge(r[["ts_ms", "regime_id", "regime_name", "regime_prob"]], on="ts_ms", how="inner")
        m["date"] = day
        merged_all.append(m)
    merged = pd.concat(merged_all, ignore_index=True).sort_values("ts_ms").reset_index(drop=True)

    metrics, kernels = compute_regime_metrics(merged, tau_max=tau_max)
    metrics["symbol"] = sym
    metrics["interval_ms"] = int(interval_ms)

    out_base = out_root / f"interval_ms={int(interval_ms)}" / f"symbol={sym}"
    out_base.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(out_base / "metrics.parquet", index=False)

    k_rows = []
    for rid, kdf in kernels.items():
        t = kdf.copy()
        t["regime_id"] = int(rid)
        k_rows.append(t)
    kernel_all = pd.concat(k_rows, ignore_index=True) if k_rows else pd.DataFrame(columns=["tau", "response", "abs_response", "cumulative_response", "count", "regime_id"])
    kernel_all.to_parquet(out_base / "kernel_by_regime.parquet", index=False)

    manifest = {
        "symbol": sym,
        "interval_ms": int(interval_ms),
        "tau_max": int(tau_max),
        "rows_metrics": int(len(metrics)),
        "rows_kernel": int(len(kernel_all)),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_base / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return metrics, kernels


def _write_report(metrics: pd.DataFrame, kernels: dict[int, pd.DataFrame], symbol: str, interval_ms: int, report_path: Path) -> None:
    plots = report_path.parent / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    p_kernel = plots / f"physics_by_regime_kernel_{symbol}_{interval_ms}ms.png"
    plt.figure(figsize=(8, 4))
    for rid, k in sorted(kernels.items()):
        plt.plot(k["tau"], k["response"], label=f"R{rid}")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title(f"Regime Kernels {symbol}")
    plt.xlabel("tau")
    plt.ylabel("response")
    if kernels:
        plt.legend()
    plt.tight_layout()
    plt.savefig(p_kernel)
    plt.close()

    lines: list[str] = []
    lines.append(f"# Physics By Regime - {symbol} ({interval_ms}ms)")
    lines.append("")
    lines.append(f"- regimes: `{len(metrics)}`")
    lines.append("")
    lines.append("| regime_id | count | ofi_lift_r1 | ofi_lift_r5 | ofi_dir_acc_r1 | linear_r2 | sqrt_r2 | kernel_lag1 | kernel_auc |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in metrics.sort_values("regime_id").iterrows():
        lines.append(
            f"| {int(r['regime_id'])} | {int(r['count'])} | {float(r['ofi_lift_r1']):.8e} | {float(r['ofi_lift_r5']):.8e} | {float(r['ofi_dir_acc_r1']):.4f} | {float(r['linear_r2']):.6f} | {float(r['sqrt_r2']):.6f} | {float(r['kernel_lag1']):.8e} | {float(r['kernel_auc']):.8e} |"
        )
    lines.append("")
    lines.append(f"- plot kernels: `{p_kernel}`")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    try:
        sym = canonical_symbol(args.symbol)
        metrics, kernels = run_metrics(
            physics_root=Path(str(args.physics)),
            regime_root=Path(str(args.regimes)),
            out_root=Path(str(args.out)),
            symbol=sym,
            interval_ms=int(args.interval_ms),
            tau_max=int(args.tau_max),
        )
        report = Path(str(args.report)) if args.report else Path(f"reports/physics_by_regime_{sym}_{int(args.interval_ms)}ms.md")
        _write_report(metrics, kernels, sym, int(args.interval_ms), report)
        print(f"physics_by_regime ok regimes={len(metrics)} report={report}")
        return 0
    except Exception as e:
        print(f"physics_by_regime error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
