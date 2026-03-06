from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.microphys.alpha.ensemble_experts import ExpertBuildConfig, build_regime_experts
from src.microphys.alpha.gating import build_gating_decisions
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build regime-specialized experts and gating artifacts.")
    p.add_argument("--eval", required=True)
    p.add_argument("--trades", required=True)
    p.add_argument("--aligned-regimes", required=True)
    p.add_argument("--transfer-by-regime", default="")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--min-trades", type=int, default=10)
    p.add_argument("--min-regime-rows", type=int, default=50)
    p.add_argument("--out", default="data/derived/alpha_eval")
    p.add_argument("--report", default="reports/ensemble_regime_experts.md")
    return p.parse_args()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_report(path: Path, experts: pd.DataFrame, gating: pd.DataFrame, symbol: str, interval_ms: int) -> None:
    lines = [
        f"# Regime Experts - {symbol} ({interval_ms}ms)",
        "",
        f"- expert_rows: `{len(experts)}`",
        f"- gating_rows: `{len(gating)}`",
        "",
        "| aligned_regime_id | signal | weight | penalty | trade_count | mean_net_ret | expected_trigger_rate | expected_fill_rate |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in experts.sort_values(["aligned_regime_id", "weight", "signal"], ascending=[True, False, True]).iterrows():
        lines.append(
            f"| {int(r['aligned_regime_id'])} | {r['signal']} | {float(r['weight']):.6f} | {float(r['penalty']):.3f} | "
            f"{int(r['trade_count'])} | {float(r['mean_net_ret']):.8f} | {float(r['expected_trigger_rate']):.6f} | {float(r['expected_fill_rate']):.4f} |"
        )
    if not gating.empty:
        lines += [
            "",
            "## Gating Summary",
            "",
            f"- fallback_rate: `{float(pd.to_numeric(gating.get('fallback_used'), errors='coerce').fillna(0.0).mean()):.4f}`",
            f"- mean_confidence: `{float(pd.to_numeric(gating.get('confidence_score'), errors='coerce').fillna(0.0).mean()):.4f}`",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    try:
        symbol = canonical_symbol(args.symbol)
        eval_df = pd.read_parquet(Path(str(args.eval)))
        trades_df = pd.read_parquet(Path(str(args.trades)))
        aligned_df = pd.read_parquet(Path(str(args.aligned_regimes)))
        transfer_df = pd.read_parquet(Path(str(args.transfer_by_regime))) if str(args.transfer_by_regime).strip() and Path(str(args.transfer_by_regime)).exists() else pd.DataFrame()
        experts = build_regime_experts(
            eval_df=eval_df,
            trades_df=trades_df,
            aligned_regimes_df=aligned_df,
            symbol=symbol,
            transfer_by_regime_df=transfer_df,
            cfg=ExpertBuildConfig(
                top_k_per_regime=int(args.top_k),
                min_trades_per_signal=int(args.min_trades),
                min_regime_rows=int(args.min_regime_rows),
            ),
        )
        sym_aligned = aligned_df[aligned_df["symbol"] == symbol].copy() if "symbol" in aligned_df.columns else aligned_df.copy()
        gating = build_gating_decisions(sym_aligned, experts, regime_col="aligned_regime_id", data_quality_ok=True)
        out_base = Path(str(args.out)) / f"interval_ms={int(args.interval_ms)}" / f"symbol={symbol}"
        out_base.mkdir(parents=True, exist_ok=True)
        experts_path = out_base / "ensemble_regime_experts.parquet"
        gating_path = out_base / "ensemble_gating.parquet"
        experts.to_parquet(experts_path, index=False)
        gating.to_parquet(gating_path, index=False)
        _write_json(
            out_base / "ensemble_regime_experts_manifest.json",
            {
                "symbol": symbol,
                "interval_ms": int(args.interval_ms),
                "expert_rows": int(len(experts)),
                "gating_rows": int(len(gating)),
                "experts_parquet": str(experts_path),
                "gating_parquet": str(gating_path),
                "transfer_by_regime_path": str(args.transfer_by_regime),
            },
        )
        _write_report(Path(str(args.report)), experts, gating, symbol, int(args.interval_ms))
        print(f"build_ensemble_regime_experts ok experts={len(experts)} gating={len(gating)} out={out_base}")
        return 0
    except Exception as e:
        print(f"build_ensemble_regime_experts error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

