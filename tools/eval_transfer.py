from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.microphys.alpha.calibration import CalibrationContext, load_calibration
from src.microphys.alpha.eval import evaluate_walkforward
from src.microphys.alpha.transfer import (
    load_partitioned_parquet,
    load_specs_jsonl,
    merge_physics_regimes,
)
from src.microphys.execution.calibration import load_execution_params
from src.microphys.live.registry import get_active_artifacts
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-asset transfer evaluation without refit.")
    p.add_argument("--exported", required=True, help="exported_specs.jsonl from export_selected_specs")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--regimes", default="data/derived/regimes")
    p.add_argument("--source-symbol", required=True)
    p.add_argument("--target-symbol", required=True)
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
    p.add_argument("--calibration-mode", choices=["source", "target"], default="source")
    p.add_argument("--source-calibration", default="")
    p.add_argument("--target-calibration", default="")
    p.add_argument("--live-root", default="data/live")
    p.add_argument("--out", default="data/derived/transfer")
    p.add_argument("--report", default="")
    return p.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _resolve_calibration(
    *,
    args: argparse.Namespace,
    export_manifest: Dict[str, Any],
) -> tuple[CalibrationContext | None, str]:
    mode = str(args.calibration_mode)
    if mode == "source":
        src = str(args.source_calibration).strip()
        if not src:
            src = str((export_manifest.get("pointers", {}) or {}).get("calibration_json", "")).strip()
        if not src:
            return None, ""
        p = Path(src)
        if not p.exists():
            raise RuntimeError("source_calibration_missing")
        return load_calibration(p), str(p)

    tgt = str(args.target_calibration).strip()
    if not tgt:
        active = get_active_artifacts(Path(str(args.live_root)))
        tgt = str(active.get("calibration_json_path", "")).strip()
    if not tgt:
        raise RuntimeError("leakage_guard:target_calibration_required")
    p = Path(tgt)
    if not p.exists():
        raise RuntimeError("target_calibration_missing")
    return load_calibration(p), str(p)


def _aggregate_eval(eval_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame(columns=["signal", f"{prefix}_test_net_mean", f"{prefix}_test_sharpe", f"{prefix}_test_trade_count", f"{prefix}_fill_rate"])
    out = (
        eval_df.groupby("signal", as_index=False)
        .agg(
            test_net_mean=("test_net_mean", "mean"),
            test_sharpe=("test_sharpe", "mean"),
            test_trade_count=("test_trade_count", "sum"),
            fill_rate=("fill_rate", "mean"),
        )
        .sort_values("signal")
        .reset_index(drop=True)
    )
    return out.rename(
        columns={
            "test_net_mean": f"{prefix}_test_net_mean",
            "test_sharpe": f"{prefix}_test_sharpe",
            "test_trade_count": f"{prefix}_test_trade_count",
            "fill_rate": f"{prefix}_fill_rate",
        }
    )


def _write_report(
    path: Path,
    merged: pd.DataFrame,
    *,
    source_symbol: str,
    target_symbol: str,
    calibration_mode: str,
    calibration_path_used: str,
    execution_params_path: str,
) -> None:
    m = merged.copy()
    if not m.empty:
        m["delta_net_mean"] = pd.to_numeric(m["target_test_net_mean"], errors="coerce") - pd.to_numeric(m["source_test_net_mean"], errors="coerce")
        m["delta_sharpe"] = pd.to_numeric(m["target_test_sharpe"], errors="coerce") - pd.to_numeric(m["source_test_sharpe"], errors="coerce")
        m["trigger_ratio"] = (
            pd.to_numeric(m["target_test_trade_count"], errors="coerce")
            / pd.to_numeric(m["source_test_trade_count"], errors="coerce").replace(0.0, pd.NA)
        ).fillna(0.0)
    pos_frac = float((pd.to_numeric(m.get("target_test_net_mean"), errors="coerce") > 0).mean()) if not m.empty else 0.0
    med_drop = float(pd.to_numeric(m.get("delta_net_mean"), errors="coerce").median()) if not m.empty else 0.0
    lines = [
        f"# Transfer Report - {source_symbol} -> {target_symbol}",
        "",
        f"- calibration_mode: `{calibration_mode}`",
        f"- calibration_path_used: `{calibration_path_used or 'N/A'}`",
        f"- execution_params: `{execution_params_path or 'N/A'}`",
        f"- specs: `{len(m)}`",
        f"- median_performance_drop: `{med_drop:.8f}`",
        f"- target_positive_frac: `{pos_frac:.4f}`",
        "",
        "| signal | source_net_mean | target_net_mean | delta | source_sharpe | target_sharpe | source_trades | target_trades | source_fill | target_fill |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in m.sort_values(["delta_net_mean", "signal"], ascending=[False, True]).iterrows():
        lines.append(
            f"| {r['signal']} | {float(r['source_test_net_mean']):.8f} | {float(r['target_test_net_mean']):.8f} | {float(r['delta_net_mean']):.8f} | "
            f"{float(r['source_test_sharpe']):.6f} | {float(r['target_test_sharpe']):.6f} | "
            f"{int(r['source_test_trade_count'])} | {int(r['target_test_trade_count'])} | "
            f"{float(r['source_fill_rate']):.4f} | {float(r['target_fill_rate']):.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    _ = int(args.seed)
    try:
        exported = Path(str(args.exported))
        if not exported.exists():
            raise RuntimeError("exported_specs_missing")
        export_manifest_path = exported.parent / "manifest.json"
        if not export_manifest_path.exists():
            raise RuntimeError("export_manifest_missing")
        export_manifest = _read_json(export_manifest_path)
        specs = load_specs_jsonl(exported)
        if not specs:
            raise RuntimeError("exported_specs_empty")

        source_symbol = canonical_symbol(args.source_symbol)
        target_symbol = canonical_symbol(args.target_symbol)
        interval_ms = int(args.interval_ms)
        source_eval_path = Path(str((export_manifest.get("pointers", {}) or {}).get("eval_parquet", "")).strip())
        if not source_eval_path.exists():
            raise RuntimeError("source_eval_missing")
        source_eval = pd.read_parquet(source_eval_path)
        source_eval_agg = _aggregate_eval(source_eval, "source")

        target_physics = load_partitioned_parquet(Path(str(args.physics)), symbol=target_symbol, interval_ms=interval_ms, name="physics")
        if target_physics.empty:
            raise RuntimeError("target_physics_missing")
        target_reg = load_partitioned_parquet(Path(str(args.regimes)), symbol=target_symbol, interval_ms=interval_ms, name="regimes")
        target_frame = merge_physics_regimes(target_physics, target_reg)

        calibration, calibration_path = _resolve_calibration(args=args, export_manifest=export_manifest)

        exec_params = None
        exec_params_path = str(args.execution_params).strip()
        if exec_params_path:
            exec_params = load_execution_params(Path(exec_params_path))

        target_eval, target_trades = evaluate_walkforward(
            target_frame,
            specs,
            calibration=calibration,
            splits=int(args.splits),
            fee_bps=float(args.fee_bps),
            latency_bars=int(args.latency_bars),
            mode=str(args.mode),
            fill_prob=float(args.fill_prob),
            max_trades_per_day=int(args.max_trades_per_day),
            execution_model=str(args.execution_model),
            execution_params=exec_params,
            ttl_bars=int(args.ttl_bars),
        )
        target_eval_agg = _aggregate_eval(target_eval, "target")
        merged = source_eval_agg.merge(target_eval_agg, on="signal", how="outer").fillna(0.0)
        merged["delta_net_mean"] = pd.to_numeric(merged["target_test_net_mean"], errors="coerce") - pd.to_numeric(
            merged["source_test_net_mean"], errors="coerce"
        )
        merged["delta_sharpe"] = pd.to_numeric(merged["target_test_sharpe"], errors="coerce") - pd.to_numeric(
            merged["source_test_sharpe"], errors="coerce"
        )

        out_dir = Path(str(args.out)) / f"source={source_symbol}" / f"target={target_symbol}" / f"interval_ms={interval_ms}"
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_out = out_dir / "eval_transfer.parquet"
        trades_out = out_dir / "trades_transfer.parquet"
        manifest_out = out_dir / "manifest.json"
        merged.sort_values("signal").reset_index(drop=True).to_parquet(eval_out, index=False)
        target_trades.to_parquet(trades_out, index=False)
        _write_json(
            manifest_out,
            {
                "source_symbol": source_symbol,
                "target_symbol": target_symbol,
                "source_exported_specs": str(exported),
                "source_eval_path": str(source_eval_path),
                "target_eval_transfer_parquet": str(eval_out),
                "target_trades_transfer_parquet": str(trades_out),
                "calibration_mode": str(args.calibration_mode),
                "calibration_path_used": calibration_path,
                "execution_params_path": exec_params_path,
                "splits": int(args.splits),
                "mode": str(args.mode),
                "fee_bps": float(args.fee_bps),
                "latency_bars": int(args.latency_bars),
                "execution_model": str(args.execution_model),
            },
        )
        report = (
            Path(str(args.report))
            if str(args.report).strip()
            else Path("reports/transfer") / f"transfer_{source_symbol}_to_{target_symbol}.md"
        )
        _write_report(
            report,
            merged,
            source_symbol=source_symbol,
            target_symbol=target_symbol,
            calibration_mode=str(args.calibration_mode),
            calibration_path_used=calibration_path,
            execution_params_path=exec_params_path,
        )
        print(f"eval_transfer ok eval={eval_out} report={report}")
        return 0
    except Exception as e:
        print(f"eval_transfer error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

