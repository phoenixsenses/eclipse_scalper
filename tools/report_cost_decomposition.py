from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from src.microphys.alpha.diagnostics import cost_decomposition
from src.microphys.alpha.spec import SignalSpec, signal_from_dict
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cost decomposition diagnostics for alpha candidates.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--regimes", default="data/derived/regimes")
    p.add_argument("--candidates", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--splits", type=int, default=3)
    p.add_argument("--fee-bps", type=float, default=0.5)
    p.add_argument("--latency-bars", type=int, default=2)
    p.add_argument("--fill-prob", type=float, default=0.3)
    p.add_argument("--out", default="data/derived/alpha_diag")
    p.add_argument("--report", default="")
    return p.parse_args()


def _load_specs(path: Path) -> List[SignalSpec]:
    out: List[SignalSpec] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(signal_from_dict(json.loads(s)))
    return out


def _load(root: Path, symbol: str, interval_ms: int, name: str) -> pd.DataFrame:
    base = root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    files = sorted(base.glob(f"date=*/{name}.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).sort_values("ts_ms").reset_index(drop=True)


def main() -> int:
    args = _parse_args()
    try:
        symbol = canonical_symbol(args.symbol)
        physics = _load(Path(str(args.physics)), symbol, int(args.interval_ms), "physics")
        regimes = _load(Path(str(args.regimes)), symbol, int(args.interval_ms), "regimes")
        if physics.empty:
            raise RuntimeError("physics_missing")
        if not regimes.empty and "regime_id" in regimes.columns:
            physics = physics.merge(regimes[["ts_ms", "regime_id"]].drop_duplicates(subset=["ts_ms"], keep="last"), on="ts_ms", how="left")
        if "regime_id" not in physics.columns:
            physics["regime_id"] = -1
        specs = _load_specs(Path(str(args.candidates)))
        taker = cost_decomposition(
            physics,
            specs,
            fee_bps=float(args.fee_bps),
            latency_bars=int(args.latency_bars),
            mode="taker",
            fill_prob=float(args.fill_prob),
        )
        maker = cost_decomposition(
            physics,
            specs,
            fee_bps=float(args.fee_bps),
            latency_bars=int(args.latency_bars),
            mode="maker",
            fill_prob=float(args.fill_prob),
        )
        joined = pd.concat([taker, maker], ignore_index=True).sort_values(["signal", "mode"]).reset_index(drop=True)
        out_base = Path(str(args.out)) / f"interval_ms={int(args.interval_ms)}" / f"symbol={symbol}"
        out_base.mkdir(parents=True, exist_ok=True)
        out_pq = out_base / "costs.parquet"
        joined.to_parquet(out_pq, index=False)

        report = Path(str(args.report)) if str(args.report).strip() else Path(f"reports/alpha_diagnostics_{symbol}_{int(args.interval_ms)}ms.md")
        almost = joined[(joined["trade_count"] > 0) & (joined["gross_mean"] > 0) & (joined["net_mean"] <= 0)].copy()
        almost["spread_kill"] = almost["spread_cost_mean"] / (almost["gross_mean"].abs() + 1e-12)
        almost["adverse_kill"] = almost["adverse_cost_mean"] / (almost["gross_mean"].abs() + 1e-12)
        spread_top = almost.sort_values(["spread_kill", "signal"], ascending=[False, True]).head(20)
        adverse_top = almost.sort_values(["adverse_kill", "signal"], ascending=[False, True]).head(20)
        lines = [
            f"# Alpha Diagnostics - {symbol} ({int(args.interval_ms)}ms)",
            "",
            f"- fee_bps: `{float(args.fee_bps)}` latency_bars: `{int(args.latency_bars)}` fill_prob: `{float(args.fill_prob)}`",
            f"- rows: `{len(joined)}`",
            "",
            "## Compare taker vs maker (top by net_mean)",
            "",
            "| signal | mode | trades | gross_mean | fee_cost | spread_cost | adverse_cost | net_mean |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        top = joined.sort_values(["net_mean", "trade_count", "signal"], ascending=[False, False, True]).head(30)
        for _, r in top.iterrows():
            lines.append(
                f"| {r['signal']} | {r['mode']} | {int(r['trade_count'])} | {float(r['gross_mean']):.8f} | "
                f"{float(r['fee_cost_mean']):.8f} | {float(r['spread_cost_mean']):.8f} | {float(r['adverse_cost_mean']):.8f} | {float(r['net_mean']):.8f} |"
            )
        lines.extend(["", "## Almost works but dies by spread", "", "| signal | mode | gross_mean | spread_kill | net_mean |", "|---|---|---:|---:|---:|"])
        for _, r in spread_top.iterrows():
            lines.append(f"| {r['signal']} | {r['mode']} | {float(r['gross_mean']):.8f} | {float(r['spread_kill']):.4f} | {float(r['net_mean']):.8f} |")
        lines.extend(["", "## Almost works but dies by adverse", "", "| signal | mode | gross_mean | adverse_kill | net_mean |", "|---|---|---:|---:|---:|"])
        for _, r in adverse_top.iterrows():
            lines.append(f"| {r['signal']} | {r['mode']} | {float(r['gross_mean']):.8f} | {float(r['adverse_kill']):.4f} | {float(r['net_mean']):.8f} |")
        lines.append("")
        lines.append(f"- output parquet: `{out_pq}`")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"report_cost_decomposition ok out={out_pq} report={report}")
        return 0
    except Exception as e:
        print(f"report_cost_decomposition error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
