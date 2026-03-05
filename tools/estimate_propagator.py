from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.microphys.propagator.kernel import compute_response_kernel, summarize_kernel
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate propagator response kernel from physics parquet.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--max-lag", type=int, default=200)
    p.add_argument("--out", default="data/derived/propagator")
    p.add_argument("--out-report", default=None)
    return p.parse_args()


def _load_all(physics_root: Path, symbol: str, interval_ms: int) -> pd.DataFrame:
    base = physics_root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    files = sorted(base.glob("date=*/physics.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in files]
    return pd.concat(frames, ignore_index=True).sort_values("ts_ms").reset_index(drop=True)


def estimate_propagator(physics_root: Path, out_root: Path, symbol: str, interval_ms: int, max_lag: int) -> pd.DataFrame:
    sym = canonical_symbol(symbol)
    df = _load_all(physics_root, sym, interval_ms)
    if df.empty:
        raise RuntimeError(f"no_physics_data symbol={sym} interval_ms={interval_ms}")

    kernel = compute_response_kernel(df["mid"], df["ofi"], max_lag=max_lag)
    out_base = out_root / f"interval_ms={int(interval_ms)}" / f"symbol={sym}"
    out_base.mkdir(parents=True, exist_ok=True)
    out_pq = out_base / "kernel.parquet"
    kernel.to_parquet(out_pq, index=False)

    summary = summarize_kernel(kernel)
    manifest = {
        "symbol": sym,
        "interval_ms": int(interval_ms),
        "max_lag": int(max_lag),
        "rows": int(len(kernel)),
        "summary": summary,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_base / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return kernel


def _write_report(kernel: pd.DataFrame, symbol: str, interval_ms: int, report_path: Path) -> None:
    plots_dir = report_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    p1 = plots_dir / f"propagator_response_{symbol}_{interval_ms}ms.png"
    plt.figure(figsize=(8, 4))
    plt.plot(kernel["tau"], kernel["response"], label="response")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title(f"Response Kernel R(tau) - {symbol}")
    plt.xlabel("tau")
    plt.ylabel("response")
    plt.tight_layout()
    plt.savefig(p1)
    plt.close()

    p2 = plots_dir / f"propagator_abs_decay_{symbol}_{interval_ms}ms.png"
    plt.figure(figsize=(8, 4))
    plt.plot(kernel["tau"], kernel["abs_response"], label="abs_response")
    plt.title(f"Absolute Response Decay - {symbol}")
    plt.xlabel("tau")
    plt.ylabel("abs response")
    plt.tight_layout()
    plt.savefig(p2)
    plt.close()

    p3 = plots_dir / f"propagator_cumulative_{symbol}_{interval_ms}ms.png"
    plt.figure(figsize=(8, 4))
    plt.plot(kernel["tau"], kernel["cumulative_response"], label="cumulative_response")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title(f"Cumulative Response - {symbol}")
    plt.xlabel("tau")
    plt.ylabel("cumulative")
    plt.tight_layout()
    plt.savefig(p3)
    plt.close()

    lines: list[str] = []
    lines.append(f"# Propagator Report - {symbol} ({interval_ms}ms)")
    lines.append("")
    lines.append(f"- taus: `{len(kernel)}`")
    lines.append(f"- response max: `{float(kernel['response'].max()):.8e}`")
    lines.append(f"- response min: `{float(kernel['response'].min()):.8e}`")
    lines.append(f"- cumulative final: `{float(kernel['cumulative_response'].iloc[-1]):.8e}`")
    lines.append("")
    lines.append("## Kernel Table")
    lines.append("")
    lines.append("| tau | response | abs_response | cumulative_response | count |")
    lines.append("|---:|---:|---:|---:|---:|")
    for _, r in kernel.iterrows():
        lines.append(
            f"| {int(r['tau'])} | {float(r['response']):.10e} | {float(r['abs_response']):.10e} | {float(r['cumulative_response']):.10e} | {int(r['count'])} |"
        )
    lines.append("")
    lines.append(f"- plot response: `{p1}`")
    lines.append(f"- plot abs decay: `{p2}`")
    lines.append(f"- plot cumulative: `{p3}`")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    try:
        kernel = estimate_propagator(
            physics_root=Path(str(args.physics)),
            out_root=Path(str(args.out)),
            symbol=str(args.symbol),
            interval_ms=int(args.interval_ms),
            max_lag=int(args.max_lag),
        )
        report = Path(str(args.out_report)) if args.out_report else Path(f"reports/propagator_{canonical_symbol(args.symbol)}_{int(args.interval_ms)}ms.md")
        _write_report(kernel, canonical_symbol(args.symbol), int(args.interval_ms), report)
        print(f"estimate_propagator ok rows={len(kernel)} report={report}")
        return 0
    except Exception as e:
        print(f"estimate_propagator error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
