from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from src.microphys.alpha.eval import evaluate_spec_on_frame
from src.microphys.alpha.spec import SignalSpec, signal_from_dict
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Horizon scan diagnostics for alpha candidates.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--regimes", default="data/derived/regimes")
    p.add_argument("--candidates", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--horizons", default="1,2,3,5,10,20,50")
    p.add_argument("--fee-bps", type=float, default=0.5)
    p.add_argument("--latency-bars", type=int, default=2)
    p.add_argument("--mode", choices=["taker", "maker"], default="taker")
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
        horizons = [int(x.strip()) for x in str(args.horizons).split(",") if x.strip()]
        physics = _load(Path(str(args.physics)), symbol, int(args.interval_ms), "physics")
        regimes = _load(Path(str(args.regimes)), symbol, int(args.interval_ms), "regimes")
        if physics.empty:
            raise RuntimeError("physics_missing")
        if not regimes.empty and "regime_id" in regimes.columns:
            physics = physics.merge(regimes[["ts_ms", "regime_id"]].drop_duplicates(subset=["ts_ms"], keep="last"), on="ts_ms", how="left")
        if "regime_id" not in physics.columns:
            physics["regime_id"] = -1
        specs = _load_specs(Path(str(args.candidates)))

        rows = []
        for spec in specs:
            for h in horizons:
                s2 = SignalSpec(
                    name=spec.name,
                    side=spec.side,
                    condition=spec.condition,
                    entry=spec.entry,
                    horizon_bars=int(h),
                    cooldown_bars=spec.cooldown_bars,
                    regime_filter=spec.regime_filter,
                    entry_mode_preference=spec.entry_mode_preference,
                    meta=spec.meta,
                )
                _, stats = evaluate_spec_on_frame(
                    physics,
                    s2,
                    fee_bps=float(args.fee_bps),
                    latency_bars=int(args.latency_bars),
                    mode=str(args.mode),
                    fill_prob=float(args.fill_prob),
                )
                rows.append(
                    {
                        "signal": spec.name,
                        "horizon": int(h),
                        "trade_count": int(stats["trade_count"]),
                        "net_mean": float(stats["net_mean"]),
                        "net_median": float(stats["net_median"]),
                        "stability_score": float(stats["stability_score"]),
                    }
                )
        scan = pd.DataFrame(rows).sort_values(["signal", "horizon"]).reset_index(drop=True)
        best = scan.sort_values(["signal", "net_mean", "trade_count"], ascending=[True, False, False]).groupby("signal", as_index=False).first()

        out_base = Path(str(args.out)) / f"interval_ms={int(args.interval_ms)}" / f"symbol={symbol}"
        out_base.mkdir(parents=True, exist_ok=True)
        out_pq = out_base / "horizon_scan.parquet"
        scan.to_parquet(out_pq, index=False)
        best.to_parquet(out_base / "horizon_best.parquet", index=False)

        report = Path(str(args.report)) if str(args.report).strip() else Path(f"reports/horizon_scan_{symbol}_{int(args.interval_ms)}ms.md")
        lines = [
            f"# Horizon Scan - {symbol} ({int(args.interval_ms)}ms)",
            "",
            f"- candidates: `{scan['signal'].nunique() if not scan.empty else 0}`",
            f"- horizons: `{','.join(str(x) for x in horizons)}`",
            "",
            "## Best horizon per signal",
            "",
            "| signal | horizon | trade_count | net_mean | net_median | stability_score |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for _, r in best.head(40).iterrows():
            lines.append(
                f"| {r['signal']} | {int(r['horizon'])} | {int(r['trade_count'])} | {float(r['net_mean']):.8f} | {float(r['net_median']):.8f} | {float(r['stability_score']):.6f} |"
            )
        lines.append("")
        lines.append(f"- output parquet: `{out_pq}`")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"horizon_scan ok out={out_pq} report={report}")
        return 0
    except Exception as e:
        print(f"horizon_scan error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
