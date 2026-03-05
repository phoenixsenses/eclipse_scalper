from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from src.microphys.alpha.eval import evaluate_walkforward
from src.microphys.execution.calibration import load_execution_params
from src.microphys.alpha.spec import SignalSpec, signal_from_dict
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward evaluation for alpha candidates.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--regimes", default="data/derived/regimes")
    p.add_argument("--candidates", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--splits", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["taker", "maker"], default="taker")
    p.add_argument("--fee-bps", type=float, default=0.5)
    p.add_argument("--latency-bars", type=int, default=2)
    p.add_argument("--fill-prob", type=float, default=0.3)
    p.add_argument("--max-trades-per-day", type=int, default=500)
    p.add_argument("--execution-model", choices=["simple", "maker_queue", "maker_hazard"], default="simple")
    p.add_argument("--execution-params", default="")
    p.add_argument("--ttl-bars", type=int, default=10)
    p.add_argument("--out", default="data/derived/alpha_eval")
    p.add_argument("--report", default="")
    return p.parse_args()


def _load_physics(root: Path, symbol: str, interval_ms: int) -> pd.DataFrame:
    base = root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    files = sorted(base.glob("date=*/physics.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).sort_values("ts_ms").reset_index(drop=True)


def _load_regimes(root: Path, symbol: str, interval_ms: int) -> pd.DataFrame:
    base = root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    files = sorted(base.glob("date=*/regimes.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).sort_values("ts_ms").reset_index(drop=True)


def _load_specs(path: Path) -> List[SignalSpec]:
    rows: List[SignalSpec] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        rows.append(signal_from_dict(json.loads(s)))
    return rows


def _merge(physics: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    if physics.empty:
        return physics
    if regimes.empty:
        out = physics.copy()
        if "regime_id" not in out.columns:
            out["regime_id"] = -1
        return out
    cols = [c for c in ("ts_ms", "regime_id", "regime_name", "regime_prob") if c in regimes.columns]
    reg = regimes[cols].drop_duplicates(subset=["ts_ms"], keep="last")
    out = physics.merge(reg, on="ts_ms", how="left")
    out["regime_id"] = pd.to_numeric(out.get("regime_id"), errors="coerce").fillna(-1).astype(int)
    return out


def _write_report(
    path: Path,
    eval_df: pd.DataFrame,
    symbol: str,
    interval_ms: int,
    mode: str,
    fee_bps: float,
    latency: int,
    max_trades_per_day: int,
) -> None:
    lines = [
        f"# Walkforward Alpha Eval - {symbol} ({interval_ms}ms)",
        "",
        f"- mode: `{mode}` fee_bps=`{fee_bps}` latency_bars=`{latency}`",
        f"- max_trades_per_day: `{int(max_trades_per_day)}`",
        f"- rows: `{len(eval_df)}`",
        "",
        "| signal | split_id | test_trades | test_net_mean | test_sharpe | stability | overfit_gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in eval_df.sort_values(["signal", "split_id"]).iterrows():
        lines.append(
            f"| {r['signal']} | {int(r['split_id'])} | {int(r['test_trade_count'])} | "
            f"{float(r['test_net_mean']):.8f} | {float(r['test_sharpe']):.6f} | "
            f"{float(r['stability_score']):.6f} | {float(r['overfit_gap']):.8f} |"
        )
    warn = eval_df[
        (pd.to_numeric(eval_df.get("trade_density"), errors="coerce") > 0.30)
        | (pd.to_numeric(eval_df.get("capped_trades"), errors="coerce") > 0)
    ]
    if not warn.empty:
        lines.extend(
            [
                "",
                "## Reality Warnings",
                "",
                f"- high density or capped rows: `{len(warn)}`",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    _ = int(args.seed)
    try:
        symbol = canonical_symbol(args.symbol)
        physics = _load_physics(Path(str(args.physics)), symbol, int(args.interval_ms))
        if physics.empty:
            raise RuntimeError("physics_missing")
        regimes = _load_regimes(Path(str(args.regimes)), symbol, int(args.interval_ms))
        frame = _merge(physics, regimes)
        specs = _load_specs(Path(str(args.candidates)))
        if not specs:
            raise RuntimeError("no_candidates")
        eval_df, trades_df = evaluate_walkforward(
            frame,
            specs,
            splits=int(args.splits),
            fee_bps=float(args.fee_bps),
            latency_bars=int(args.latency_bars),
            mode=str(args.mode),
            fill_prob=float(args.fill_prob),
            max_trades_per_day=int(args.max_trades_per_day),
            execution_model=str(args.execution_model),
            execution_params=(load_execution_params(Path(str(args.execution_params))) if str(args.execution_params).strip() else None),
            ttl_bars=int(args.ttl_bars),
        )
        out_base = Path(str(args.out)) / f"interval_ms={int(args.interval_ms)}" / f"symbol={symbol}"
        out_base.mkdir(parents=True, exist_ok=True)
        eval_path = out_base / "eval.parquet"
        trades_path = out_base / "trades.parquet"
        eval_df.to_parquet(eval_path, index=False)
        trades_df.to_parquet(trades_path, index=False)
        (out_base / "manifest.json").write_text(
            json.dumps(
                {
                    "symbol": symbol,
                    "interval_ms": int(args.interval_ms),
                    "eval_rows": int(len(eval_df)),
                    "trade_rows": int(len(trades_df)),
                    "splits": int(args.splits),
                    "mode": str(args.mode),
                    "fee_bps": float(args.fee_bps),
                    "latency_bars": int(args.latency_bars),
                    "max_trades_per_day": int(args.max_trades_per_day),
                    "execution_model": str(args.execution_model),
                    "execution_params": str(args.execution_params),
                    "ttl_bars": int(args.ttl_bars),
                },
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        report = Path(str(args.report)) if str(args.report).strip() else Path(f"reports/walkforward_{symbol}_{int(args.interval_ms)}ms.md")
        _write_report(
            report,
            eval_df,
            symbol,
            int(args.interval_ms),
            str(args.mode),
            float(args.fee_bps),
            int(args.latency_bars),
            int(args.max_trades_per_day),
        )
        print(f"eval_alpha_walkforward ok eval={eval_path} trades={trades_path} report={report}")
        return 0
    except Exception as e:
        print(f"eval_alpha_walkforward error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
