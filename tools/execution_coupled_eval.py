from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.microphys.execution.cost_models import CostConfig
from src.microphys.execution.eval import evaluate_conditions
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Execution-coupled fast eval for physics conditions.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--regimes", default="data/derived/regimes")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--fee-bps", type=float, default=0.5)
    p.add_argument("--latency-bars", type=int, default=2)
    p.add_argument("--mode", choices=["taker", "maker"], default="taker")
    p.add_argument("--fill-prob", type=float, default=0.3)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _load_physics(root: Path, symbol: str, interval_ms: int) -> pd.DataFrame:
    base = root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    files = sorted(base.glob("date=*/physics.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).sort_values("ts_ms").reset_index(drop=True)


def main() -> int:
    args = _parse_args()
    try:
        sym = canonical_symbol(args.symbol)
        df = _load_physics(Path(str(args.physics)), sym, int(args.interval_ms))
        if df.empty:
            raise RuntimeError("physics_missing")

        cfg = CostConfig(
            fee_bps=float(args.fee_bps),
            latency_bars=int(args.latency_bars),
            mode=str(args.mode),
            fill_prob=float(args.fill_prob),
        )
        res = evaluate_conditions(df, horizon=int(args.horizon), cfg=cfg)

        out = Path(str(args.out)) if args.out else Path(f"reports/execution_coupled_eval_{sym}_{int(args.interval_ms)}ms.md")
        out.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        lines.append(f"# Execution Coupled Eval - {sym} ({int(args.interval_ms)}ms)")
        lines.append("")
        lines.append(f"- horizon: `{int(args.horizon)}` bars")
        lines.append(f"- mode: `{cfg.mode}` fee_bps=`{cfg.fee_bps}` latency_bars=`{cfg.latency_bars}` fill_prob=`{cfg.fill_prob}`")
        lines.append("")
        lines.append("| condition | side | count | gross_mean | net_mean | net_median | worst_5pct_day | t_stat | ci_low | ci_high |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in res.iterrows():
            lines.append(
                f"| {r['condition']} | {r['side']} | {int(r['count'])} | {float(r['gross_mean']):.8f} | {float(r['net_mean']):.8f} | {float(r['net_median']):.8f} | {float(r['worst_5pct_day']):.8f} | {float(r['t_stat']):.4f} | {float(r['bootstrap_ci_low']):.8f} | {float(r['bootstrap_ci_high']):.8f} |"
            )

        out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"execution_coupled_eval ok rows={len(res)} out={out}")
        return 0
    except Exception as e:
        print(f"execution_coupled_eval error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
