from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate sanity report from derived state parquet.")
    p.add_argument("--state", default="data/derived/state")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--out", default="reports/state_sanity_ETHUSDT_100ms.md")
    return p.parse_args()


def _load_state(state_root: Path, symbol: str, interval_ms: int) -> pd.DataFrame:
    base = state_root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    files = sorted(base.glob("date=*/state.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("ts_ms").reset_index(drop=True)
    return df


def _coverage_stats(df: pd.DataFrame, interval_ms: int) -> dict[str, Any]:
    if df.empty:
        return {"rows": 0, "missing_pct": None}
    ts = df["ts_ms"].astype("int64")
    min_ts = int(ts.min())
    max_ts = int(ts.max())
    expected = int(((max_ts - min_ts) // int(interval_ms)) + 1)
    actual = int(len(df))
    missing = max(0, expected - actual)
    return {
        "rows": actual,
        "min_ts": min_ts,
        "max_ts": max_ts,
        "missing_pct": (100.0 * missing / expected) if expected > 0 else 0.0,
    }


def _ofi_decile_next_ret(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["ofi", "mid"]].copy()
    d["next_ret"] = d["mid"].shift(-1) / d["mid"] - 1.0
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["ofi", "next_ret"])
    if d.empty:
        return pd.DataFrame(columns=["decile", "count", "mean_next_ret", "median_next_ret"])
    try:
        d["decile"] = pd.qcut(d["ofi"], q=10, labels=False, duplicates="drop")
    except Exception:
        return pd.DataFrame(columns=["decile", "count", "mean_next_ret", "median_next_ret"])
    g = (
        d.groupby("decile", as_index=False)
        .agg(count=("next_ret", "size"), mean_next_ret=("next_ret", "mean"), median_next_ret=("next_ret", "median"))
        .sort_values("decile")
    )
    return g


def _safe_autocorr(series: pd.Series, lag: int) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= lag:
        return 0.0
    if float(s.std()) == 0.0:
        return 0.0
    val = s.autocorr(lag=lag)
    if val is None or np.isnan(val):
        return 0.0
    return float(val)


def _save_plots(df: pd.DataFrame, deciles: pd.DataFrame, symbol: str, interval_ms: int, plot_dir: Path) -> dict[str, str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}

    # spread histogram
    plt.figure(figsize=(8, 4))
    plt.hist(df["spread"].dropna().to_numpy(), bins=80)
    plt.title(f"Spread Distribution {symbol} {interval_ms}ms")
    plt.xlabel("spread")
    plt.ylabel("count")
    p1 = plot_dir / f"state_spread_hist_{symbol}_{interval_ms}ms.png"
    plt.tight_layout()
    plt.savefig(p1)
    plt.close()
    files["spread_hist"] = str(p1)

    # intensity histogram
    plt.figure(figsize=(8, 4))
    plt.hist(df["trade_intensity"].dropna().to_numpy(), bins=80)
    plt.title(f"Trade Intensity {symbol} {interval_ms}ms")
    plt.xlabel("trade_intensity")
    plt.ylabel("count")
    p2 = plot_dir / f"state_intensity_hist_{symbol}_{interval_ms}ms.png"
    plt.tight_layout()
    plt.savefig(p2)
    plt.close()
    files["intensity_hist"] = str(p2)

    # OFI decile predictive plot
    plt.figure(figsize=(8, 4))
    if not deciles.empty:
        plt.bar(deciles["decile"].astype(int).astype(str), deciles["mean_next_ret"])
    plt.title(f"OFI Decile vs Mean Next Return {symbol}")
    plt.xlabel("OFI decile")
    plt.ylabel("mean next return")
    p3 = plot_dir / f"state_ofi_decile_nextret_{symbol}_{interval_ms}ms.png"
    plt.tight_layout()
    plt.savefig(p3)
    plt.close()
    files["ofi_decile"] = str(p3)

    # OFI autocorr
    lags = list(range(1, 51))
    ac_vals = [_safe_autocorr(df["ofi"], lag=l) for l in lags]
    plt.figure(figsize=(8, 4))
    plt.plot(lags, ac_vals)
    plt.title(f"OFI Autocorrelation {symbol}")
    plt.xlabel("lag")
    plt.ylabel("autocorr")
    p4 = plot_dir / f"state_ofi_autocorr_{symbol}_{interval_ms}ms.png"
    plt.tight_layout()
    plt.savefig(p4)
    plt.close()
    files["ofi_autocorr"] = str(p4)

    return files


def generate_report(state_root: Path, symbol: str, interval_ms: int, out_md: Path) -> str:
    sym = canonical_symbol(symbol)
    df = _load_state(state_root, sym, interval_ms)
    if df.empty:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        text = f"# State Sanity Report\n\nNo state data found for `{sym}` interval `{interval_ms}`ms.\n"
        out_md.write_text(text, encoding="utf-8")
        return text

    coverage = _coverage_stats(df, interval_ms)
    spread = df["spread"].dropna().to_numpy()
    spread_pct = np.percentile(spread, [1, 5, 50, 95, 99]) if spread.size else [0, 0, 0, 0, 0]

    ofi = df["ofi"].dropna()
    ofi_stats = {
        "mean": float(ofi.mean()) if len(ofi) else 0.0,
        "std": float(ofi.std()) if len(ofi) else 0.0,
        "p01": float(ofi.quantile(0.01)) if len(ofi) else 0.0,
        "p99": float(ofi.quantile(0.99)) if len(ofi) else 0.0,
    }
    ofi_autocorr = [_safe_autocorr(ofi, lag=i) for i in range(1, 51)] if len(ofi) else []

    intensity = df["trade_intensity"].dropna()
    deciles = _ofi_decile_next_ret(df)

    plot_dir = out_md.parent / "plots"
    plot_files = _save_plots(df, deciles, sym, interval_ms, plot_dir)

    liq_txt = ""
    if "liq_rate" in df.columns:
        burst_thresh = float(df["liq_rate"].quantile(0.99)) if len(df) else 0.0
        burst = df[df["liq_rate"] >= burst_thresh].copy()
        burst_next = (burst["mid"].shift(-1) / burst["mid"] - 1.0).dropna()
        liq_txt = (
            f"- liquidation burst threshold (p99 liq_rate): `{burst_thresh:.6f}`\n"
            f"- mean next return after burst: `{(burst_next.mean() if len(burst_next) else 0.0):.8f}`\n"
        )

    lines: list[str] = []
    lines.append(f"# State Sanity Report - {sym} ({interval_ms}ms)")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- rows: `{coverage['rows']}`")
    lines.append(f"- min_ts_ms: `{coverage.get('min_ts')}`")
    lines.append(f"- max_ts_ms: `{coverage.get('max_ts')}`")
    lines.append(f"- missing interval %: `{coverage.get('missing_pct'):.4f}`")
    lines.append("")
    lines.append("## Spread Distribution")
    lines.append("")
    lines.append(
        f"- p01=`{spread_pct[0]:.8f}` p05=`{spread_pct[1]:.8f}` p50=`{spread_pct[2]:.8f}` p95=`{spread_pct[3]:.8f}` p99=`{spread_pct[4]:.8f}`"
    )
    lines.append(f"- plot: `{plot_files['spread_hist']}`")
    lines.append("")
    lines.append("## OFI")
    lines.append("")
    lines.append(
        f"- mean=`{ofi_stats['mean']:.8f}` std=`{ofi_stats['std']:.8f}` p01=`{ofi_stats['p01']:.8f}` p99=`{ofi_stats['p99']:.8f}`"
    )
    if ofi_autocorr:
        lines.append(f"- autocorr lag1=`{ofi_autocorr[0]:.8f}` lag10=`{ofi_autocorr[9]:.8f}` lag50=`{ofi_autocorr[49]:.8f}`")
    lines.append(f"- plot: `{plot_files['ofi_autocorr']}`")
    lines.append("")
    lines.append("## Trade Intensity")
    lines.append("")
    lines.append(
        f"- mean=`{(float(intensity.mean()) if len(intensity) else 0.0):.8f}` median=`{(float(intensity.median()) if len(intensity) else 0.0):.8f}`"
    )
    lines.append(f"- plot: `{plot_files['intensity_hist']}`")
    lines.append("")
    lines.append("## OFI -> Next Return (Deciles)")
    lines.append("")
    if deciles.empty:
        lines.append("- insufficient data")
    else:
        lines.append("| decile | count | mean_next_ret | median_next_ret |")
        lines.append("|---:|---:|---:|---:|")
        for _, r in deciles.iterrows():
            lines.append(
                f"| {int(r['decile'])} | {int(r['count'])} | {float(r['mean_next_ret']):.10f} | {float(r['median_next_ret']):.10f} |"
            )
    lines.append(f"- plot: `{plot_files['ofi_decile']}`")
    lines.append("")
    lines.append("## Liquidation Bursts")
    lines.append("")
    lines.append(liq_txt or "- liquidation fields not available")

    text = "\n".join(lines).rstrip() + "\n"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text, encoding="utf-8")
    return text


def main() -> int:
    args = _parse_args()
    try:
        out = Path(str(args.out))
        generate_report(
            state_root=Path(str(args.state)),
            symbol=str(args.symbol),
            interval_ms=int(args.interval_ms),
            out_md=out,
        )
        print(f"report_state_sanity ok out={out}")
        return 0
    except Exception as e:
        print(f"report_state_sanity error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
