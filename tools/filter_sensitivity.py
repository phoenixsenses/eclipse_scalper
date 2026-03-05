from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.microphys.alpha.filter_sweep import default_settings, run_filter_sweep
from src.microphys.alpha.selection import summarize_signals
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter sensitivity sweep for alpha selection.")
    p.add_argument("--eval", required=True)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--out", default="data/derived/alpha_diag")
    p.add_argument("--report", default="")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        eval_df = pd.read_parquet(Path(str(args.eval)))
        sweep = run_filter_sweep(eval_df, default_settings())
        summary = summarize_signals(eval_df)
        symbol = canonical_symbol(args.symbol)
        out_base = Path(str(args.out)) / f"interval_ms={int(args.interval_ms)}" / f"symbol={symbol}"
        out_base.mkdir(parents=True, exist_ok=True)
        out_pq = out_base / "filter_sweep.parquet"
        sweep.to_parquet(out_pq, index=False)
        report = Path(str(args.report)) if str(args.report).strip() else Path(f"reports/filter_sensitivity_{symbol}_{int(args.interval_ms)}ms.md")
        lines = [
            f"# Filter Sensitivity - {symbol} ({int(args.interval_ms)}ms)",
            "",
            f"- eval rows: `{len(eval_df)}`",
            f"- settings: `{len(sweep)}`",
            "",
            "## Selected count by setting",
            "",
            "| min_trades | require_positive_all_folds | stability_max_cv | allow_one_fold_negative | selected_count | top_signal | top_score |",
            "|---:|---:|---:|---:|---:|---|---:|",
        ]
        for _, r in sweep.head(50).iterrows():
            lines.append(
                f"| {int(r['min_trades_per_split'])} | {int(r['require_positive_all_folds'])} | {float(r['stability_max_cv']):.2f} | "
                f"{int(r['allow_one_fold_negative'])} | {int(r['selected_count'])} | {r['top_signal']} | {float(r['top_score']):.6f} |"
            )
        lines.extend(["", "## Top baseline signals", "", "| signal | test_net_mean | test_sharpe | stability |", "|---|---:|---:|---:|"])
        for _, r in summary.head(20).iterrows():
            lines.append(
                f"| {r['signal']} | {float(r['test_net_mean']):.8f} | {float(r['test_sharpe']):.6f} | {float(r['stability_score']):.6f} |"
            )
        lines.append("")
        lines.append(f"- output parquet: `{out_pq}`")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"filter_sensitivity ok out={out_pq} report={report}")
        return 0
    except Exception as e:
        print(f"filter_sensitivity error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
