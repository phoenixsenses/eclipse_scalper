from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.microphys.alpha.ensemble import build_ensemble_scores, pick_topk_by_regime
from src.microphys.alpha.selection import select_robust_signals, summarize_signals
from src.microphys.alpha.spec import SignalSpec, signal_from_dict
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select robust alpha signals and build regime-aware ensemble.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--regimes", default="data/derived/regimes")
    p.add_argument("--eval", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--min-trades-per-split", type=int, default=10)
    p.add_argument("--min-stability", type=float, default=0.2)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--out", default="data/derived/alpha_eval")
    p.add_argument("--report-alpha", default="")
    p.add_argument("--report-ensemble", default="")
    return p.parse_args()


def _load_specs(path: Path) -> Dict[str, SignalSpec]:
    out: Dict[str, SignalSpec] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        spec = signal_from_dict(json.loads(s))
        out[spec.name] = spec
    return out


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


def _write_alpha_report(path: Path, summary: pd.DataFrame, selected: pd.DataFrame) -> None:
    lines = [
        "# Alpha Discovery Summary",
        "",
        f"- total signals: `{len(summary)}`",
        f"- selected robust: `{len(selected)}`",
        "",
        "## Top Signals",
        "",
        "| rank | signal | composite | sharpe | net_mean | stability | overfit_gap |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, (_, r) in enumerate(selected.head(20).iterrows(), start=1):
        lines.append(
            f"| {i} | {r['signal']} | {float(r['composite_score']):.6f} | {float(r['test_sharpe']):.6f} | "
            f"{float(r['test_net_mean']):.8f} | {float(r['stability_score']):.6f} | {float(r['overfit_gap']):.8f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_ensemble_report(path: Path, picked: Dict[int, List[str]], ensemble: pd.DataFrame) -> None:
    lines = [
        "# Regime-Aware Ensemble",
        "",
        f"- rows: `{len(ensemble)}`",
        "",
        "## Picked per regime",
        "",
        "| regime_id | signals |",
        "|---:|---|",
    ]
    for rid in sorted(picked):
        lines.append(f"| {rid} | `{','.join(picked[rid])}` |")
    lines.extend(
        [
            "",
            "## Ensemble diagnostics",
            "",
            f"- mean score: `{float(pd.to_numeric(ensemble.get('ensemble_score'), errors='coerce').mean() if not ensemble.empty else 0.0):.8f}`",
            f"- active bars: `{int((pd.to_numeric(ensemble.get('signal_count'), errors='coerce') > 0).sum() if not ensemble.empty else 0)}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    try:
        symbol = canonical_symbol(args.symbol)
        eval_df = pd.read_parquet(Path(str(args.eval)))
        specs = _load_specs(Path(str(args.candidates)))
        summary = summarize_signals(eval_df)
        selected = select_robust_signals(
            summary,
            min_trades_per_split=int(args.min_trades_per_split),
            min_stability=float(args.min_stability),
        )
        top_names = selected.head(max(1, int(args.top_k)))["signal"].astype(str).tolist()
        top_specs = [specs[n] for n in top_names if n in specs]
        physics = _load_physics(Path(str(args.physics)), symbol, int(args.interval_ms))
        regimes = _load_regimes(Path(str(args.regimes)), symbol, int(args.interval_ms))
        if not regimes.empty:
            reg = regimes[[c for c in ("ts_ms", "regime_id") if c in regimes.columns]].drop_duplicates(subset=["ts_ms"], keep="last")
            physics = physics.merge(reg, on="ts_ms", how="left")
        if "regime_id" not in physics.columns:
            physics["regime_id"] = -1
        picked = pick_topk_by_regime(selected, specs, top_k=max(1, int(args.top_k)))
        ensemble = build_ensemble_scores(physics, top_specs)

        out_base = Path(str(args.out)) / f"interval_ms={int(args.interval_ms)}" / f"symbol={symbol}"
        out_base.mkdir(parents=True, exist_ok=True)
        summary.to_parquet(out_base / "selection_summary.parquet", index=False)
        selected.to_parquet(out_base / "selected.parquet", index=False)
        ensemble.to_parquet(out_base / "ensemble.parquet", index=False)
        (out_base / "ensemble_selected.json").write_text(
            json.dumps({str(k): v for k, v in sorted(picked.items())}, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (out_base / "selection_manifest.json").write_text(
            json.dumps(
                {
                    "symbol": symbol,
                    "interval_ms": int(args.interval_ms),
                    "summary_rows": int(len(summary)),
                    "selected_rows": int(len(selected)),
                    "ensemble_rows": int(len(ensemble)),
                    "top_k": int(args.top_k),
                },
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        rpt_alpha = Path(str(args.report_alpha)) if str(args.report_alpha).strip() else Path(f"reports/alpha_discovery_{symbol}_{int(args.interval_ms)}ms.md")
        rpt_ens = Path(str(args.report_ensemble)) if str(args.report_ensemble).strip() else Path(f"reports/ensemble_{symbol}_{int(args.interval_ms)}ms.md")
        _write_alpha_report(rpt_alpha, summary, selected)
        _write_ensemble_report(rpt_ens, picked, ensemble)
        print(f"select_alpha ok selected={len(selected)} ensemble_rows={len(ensemble)} out={out_base}")
        return 0
    except Exception as e:
        print(f"select_alpha error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
