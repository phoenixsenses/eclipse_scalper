from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from src.microphys.alpha.diagnostics import candidate_coverage
from src.microphys.alpha.spec import SignalSpec, signal_from_dict
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Candidate coverage diagnostics.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--regimes", default="data/derived/regimes")
    p.add_argument("--candidates", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--splits", type=int, default=3)
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
        cov = candidate_coverage(physics, specs, splits=int(args.splits))
        out_base = Path(str(args.out)) / f"interval_ms={int(args.interval_ms)}" / f"symbol={symbol}"
        out_base.mkdir(parents=True, exist_ok=True)
        out_pq = out_base / "coverage.parquet"
        cov.to_parquet(out_pq, index=False)
        report = Path(str(args.report)) if str(args.report).strip() else Path(f"reports/candidate_coverage_{symbol}_{int(args.interval_ms)}ms.md")
        agg = (
            cov.groupby("signal", as_index=False)
            .agg(
                triggered_events=("triggered_events", "sum"),
                after_cooldown=("after_cooldown", "sum"),
                effective_trades=("effective_trades", "sum"),
                cooldown_drop_pct=("cooldown_drop_pct", "mean"),
                missing_horizon_pct=("missing_horizon_pct", "mean"),
                regime_concentration_top3=("regime_concentration_top3", "first"),
            )
            .sort_values(["effective_trades", "triggered_events", "signal"], ascending=[True, True, True])
        )
        lines = [
            f"# Candidate Coverage - {symbol} ({int(args.interval_ms)}ms)",
            "",
            f"- candidates: `{len(agg)}`",
            f"- splits: `{int(args.splits)}`",
            "",
            "## Zero trigger candidates",
            "",
            "| signal | triggered_events | after_cooldown | effective_trades |",
            "|---|---:|---:|---:|",
        ]
        zero = agg[agg["triggered_events"] == 0].head(30)
        for _, r in zero.iterrows():
            lines.append(f"| {r['signal']} | {int(r['triggered_events'])} | {int(r['after_cooldown'])} | {int(r['effective_trades'])} |")
        lines.extend(
            [
                "",
                "## Triggered but no effective trades",
                "",
                "| signal | triggered_events | after_cooldown | effective_trades | cooldown_drop_pct | missing_horizon_pct | regime_concentration_top3 |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        bad = agg[(agg["triggered_events"] > 0) & (agg["effective_trades"] == 0)].head(30)
        for _, r in bad.iterrows():
            lines.append(
                f"| {r['signal']} | {int(r['triggered_events'])} | {int(r['after_cooldown'])} | {int(r['effective_trades'])} | "
                f"{float(r['cooldown_drop_pct']):.2f} | {float(r['missing_horizon_pct']):.2f} | {r['regime_concentration_top3']} |"
            )
        lines.extend(
            [
                "",
                "## Trigger distribution",
                "",
                f"- min triggered: `{int(agg['triggered_events'].min()) if not agg.empty else 0}`",
                f"- median triggered: `{float(agg['triggered_events'].median()) if not agg.empty else 0.0:.2f}`",
                f"- max triggered: `{int(agg['triggered_events'].max()) if not agg.empty else 0}`",
                f"- output parquet: `{out_pq}`",
            ]
        )
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"report_candidate_coverage ok out={out_pq} report={report}")
        return 0
    except Exception as e:
        print(f"report_candidate_coverage error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
