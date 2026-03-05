from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.microphys.alpha.calibration import compute_calibration, save_calibration
from src.microphys.alpha.ensemble import build_ensemble_scores, pick_topk_by_regime
from src.microphys.alpha.ensemble_experts import ExpertBuildConfig, build_regime_experts
from src.microphys.alpha.eval import evaluate_walkforward
from src.microphys.alpha.gating import build_gating_decisions
from src.microphys.alpha.generator import generate_candidates
from src.microphys.alpha.overlap import dedupe_specs, pairwise_overlap
from src.microphys.alpha.runlog import RunLog
from src.microphys.alpha.selection import select_robust_signals, summarize_signals
from src.microphys.alpha.spec import SignalSpec, signal_from_dict, specs_to_jsonl
from src.microphys.execution.calibration import calibrate_execution_models, save_execution_params
from src.microphys.execution.features import build_execution_features
from src.microphys.sim.papertrade import PaperTradeConfig, generate_papertrades
from utils.symbols import canonical_symbol


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _run_id(symbol: str, interval_ms: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_symbol={symbol}_interval={int(interval_ms)}ms"


def _load_partitioned(root: Path, name: str, symbol: str, interval_ms: int, tail_days: int | None = None) -> pd.DataFrame:
    base = root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    files = sorted(base.glob(f"date=*/{name}.parquet"))
    if not files:
        return pd.DataFrame()
    if tail_days and tail_days > 0:
        files = files[-int(tail_days) :]
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).sort_values("ts_ms").reset_index(drop=True)


def _write_md(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _selection_diagnostics(summary: pd.DataFrame, min_trades_per_split: int, min_stability: float) -> Dict[str, int]:
    if summary.empty:
        return {"empty_summary": 1}
    req = pd.to_numeric(summary["splits"], errors="coerce").fillna(0).clip(lower=1) * int(min_trades_per_split)
    low_trades = int((pd.to_numeric(summary["test_trade_count"], errors="coerce") < req).sum())
    fold_fail = int(
        (pd.to_numeric(summary["positive_test_folds"], errors="coerce") < pd.to_numeric(summary["splits"], errors="coerce")).sum()
    )
    stability_fail = int((pd.to_numeric(summary["stability_score"], errors="coerce") < float(min_stability)).sum())
    return {
        "low_trades": low_trades,
        "fold_consistency_fail": fold_fail,
        "stability_fail": stability_fail,
    }


def _resolve_latest_file(candidates: List[Path]) -> Path | None:
    files = [p for p in candidates if p.exists() and p.is_file()]
    if not files:
        return None
    files.sort(key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
    return files[0]


def _auto_resolve_alignment(interval_ms: int) -> Path | None:
    latest = Path("data/derived/regime_alignment/LATEST.json")
    if latest.exists():
        try:
            payload = json.loads(latest.read_text(encoding="utf-8"))
            p = Path(str(payload.get("aligned_regimes_parquet", "")).strip())
            if p.exists() and p.is_file():
                return p
        except Exception:
            pass
    return _resolve_latest_file(list(Path("data/derived/regime_alignment").glob(f"**/interval_ms={int(interval_ms)}/aligned_regimes.parquet")))


def _auto_resolve_transfer_by_regime() -> Path | None:
    stable = Path("data/derived/regime_alignment/transfer_by_regime.parquet")
    if stable.exists() and stable.is_file():
        return stable
    return _resolve_latest_file(list(Path("data/derived/regime_alignment").glob("**/transfer_by_regime.parquet")))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Phase 4.4 alpha pipeline end-to-end.")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--regimes", default="data/derived/regimes")
    p.add_argument("--out-root", default="data/runs/alpha")
    p.add_argument("--run-dir", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quick", action="store_true")

    p.add_argument("--days-calibration", type=int, default=14)
    p.add_argument("--limit", type=int, default=600)
    p.add_argument("--horizons", default="5,10,20")
    p.add_argument("--target-triggers-per-day", type=float, default=200.0)
    p.add_argument("--target-trigger-band", type=float, default=0.5)
    p.add_argument("--min-triggered", type=int, default=50)
    p.add_argument("--min-triggers-per-day", type=float, default=50.0)
    p.add_argument("--max-triggers-per-day", type=float, default=500.0)
    p.add_argument("--jaccard-thr", type=float, default=0.90)

    p.add_argument("--splits", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["taker", "maker"], default="taker")
    p.add_argument("--fee-bps", type=float, default=0.5)
    p.add_argument("--latency-bars", type=int, default=2)
    p.add_argument("--fill-prob", type=float, default=0.3)
    p.add_argument("--max-trades-per-day", type=int, default=500)

    p.add_argument("--min-trades-per-split", type=int, default=10)
    p.add_argument("--min-stability", type=float, default=0.2)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--allow-empty-selection", action="store_true", default=True)
    p.add_argument("--calibrate-execution", action="store_true")
    p.add_argument("--execution-calibration-days", type=int, default=14)
    p.add_argument("--execution-model", choices=["simple", "maker_queue", "maker_hazard"], default="maker_hazard")
    p.add_argument("--execution-params-mode", choices=["taker", "maker"], default="maker")
    p.add_argument("--require-execution-params", action="store_true")
    p.add_argument("--build-regime-experts", action="store_true")
    p.add_argument("--aligned-regimes", default="")
    p.add_argument("--transfer-by-regime", default="")
    p.add_argument("--min-expert-rows", type=int, default=50)
    p.add_argument("--experts-max-per-regime", type=int, default=5)
    p.add_argument("--experts-fallback-global", action="store_true", default=True)
    p.add_argument("--require-regime-experts", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _ = int(args.seed)
    try:
        symbol = canonical_symbol(args.symbol)
        interval_ms = int(args.interval_ms)
        if bool(args.quick):
            args.limit = min(int(args.limit), 100)
            args.days_calibration = min(int(args.days_calibration), 7)
        if args.run_dir:
            run_dir = Path(str(args.run_dir))
        else:
            run_dir = Path(str(args.out_root)) / _run_id(symbol, interval_ms)

        run = RunLog(
            run_dir=run_dir,
            params={
                "symbol": symbol,
                "interval_ms": interval_ms,
                "physics": str(args.physics),
                "regimes": str(args.regimes),
                "days_calibration": int(args.days_calibration),
                "limit": int(args.limit),
                "horizons": str(args.horizons),
                "target_triggers_per_day": float(args.target_triggers_per_day),
                "target_trigger_band": float(args.target_trigger_band),
                "jaccard_thr": float(args.jaccard_thr),
                "splits": int(args.splits),
                "mode": str(args.mode),
                "fee_bps": float(args.fee_bps),
                "latency_bars": int(args.latency_bars),
                "max_trades_per_day": int(args.max_trades_per_day),
                "quick": bool(args.quick),
                "resume": bool(args.resume),
                "calibrate_execution": bool(args.calibrate_execution),
                "execution_calibration_days": int(args.execution_calibration_days),
                "execution_model": str(args.execution_model),
                "execution_params_mode": str(args.execution_params_mode),
                "require_execution_params": bool(args.require_execution_params),
                "build_regime_experts": bool(args.build_regime_experts),
                "aligned_regimes": str(args.aligned_regimes),
                "transfer_by_regime": str(args.transfer_by_regime),
                "min_expert_rows": int(args.min_expert_rows),
                "experts_max_per_regime": int(args.experts_max_per_regime),
                "experts_fallback_global": bool(args.experts_fallback_global),
                "require_regime_experts": bool(args.require_regime_experts),
            },
        )
        reports_dir = run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_base = run_dir / "artifacts" / f"interval_ms={interval_ms}" / f"symbol={symbol}"
        out_base.mkdir(parents=True, exist_ok=True)
        exec_out_base = run_dir / "artifacts" / "execution"
        exec_out_base.mkdir(parents=True, exist_ok=True)
        ensemble_out_base = run_dir / "artifacts" / "ensemble"
        ensemble_out_base.mkdir(parents=True, exist_ok=True)

        aligned_path = Path(str(args.aligned_regimes)) if str(args.aligned_regimes).strip() else _auto_resolve_alignment(interval_ms)
        transfer_regime_path = (
            Path(str(args.transfer_by_regime)) if str(args.transfer_by_regime).strip() else _auto_resolve_transfer_by_regime()
        )
        run.update_pointers(
            aligned_regimes_path=(str(aligned_path) if aligned_path is not None and aligned_path.exists() else ""),
            transfer_by_regime_path=(str(transfer_regime_path) if transfer_regime_path is not None and transfer_regime_path.exists() else ""),
        )

        physics = _load_partitioned(Path(str(args.physics)), "physics", symbol, interval_ms)
        if physics.empty:
            raise RuntimeError("fail_fast:physics_missing")
        regimes = _load_partitioned(Path(str(args.regimes)), "regimes", symbol, interval_ms)
        if not regimes.empty and "regime_id" in regimes.columns:
            reg = regimes[[c for c in ("ts_ms", "regime_id", "regime_name", "regime_prob") if c in regimes.columns]].drop_duplicates(
                subset=["ts_ms"], keep="last"
            )
            physics = physics.merge(reg, on="ts_ms", how="left")
        if "regime_id" not in physics.columns:
            physics["regime_id"] = -1

        # Optional: execution calibration step
        if bool(args.calibrate_execution):
            step = "execution_calibration"
            exec_params_path = exec_out_base / "params.json"
            exec_report_path = reports_dir / "execution_realism.md"
            if not (
                bool(args.resume)
                and run.read_manifest().get("steps", {}).get(step, {}).get("status") == "completed"
                and exec_params_path.exists()
            ):
                run.set_step(step, "running")
                try:
                    pcal = physics.copy()
                    if int(args.execution_calibration_days) > 0 and "ts_utc" in pcal.columns:
                        t = pd.to_datetime(pcal["ts_utc"], utc=True, errors="coerce")
                        cutoff = t.max() - pd.Timedelta(days=int(args.execution_calibration_days))
                        pcal = pcal[t >= cutoff].copy()
                    exec_features = build_execution_features(
                        pcal[
                            [
                                c
                                for c in (
                                    "ts_ms",
                                    "ts_utc",
                                    "symbol",
                                    "mid",
                                    "spread",
                                    "bid_qty",
                                    "ask_qty",
                                    "trade_intensity",
                                    "trade_intensity_qty_per_sec",
                                )
                                if c in pcal.columns
                            ]
                        ].copy()
                    )
                    exec_feat_path = exec_out_base / "exec_features.parquet"
                    exec_features.to_parquet(exec_feat_path, index=False)
                    params = calibrate_execution_models(pcal)
                    save_execution_params(exec_params_path, params)
                    exec_report_path.write_text(
                        "\n".join(
                            [
                                f"# Execution Realism - {symbol}",
                                "",
                                f"- calibration_rows: `{len(pcal)}`",
                                f"- execution_model_target: `{str(args.execution_model)}`",
                                f"- mode: `{str(args.execution_params_mode)}`",
                                f"- params_path: `{exec_params_path}`",
                                f"- maker_queue.queue_frac: `{float(params.get('maker_queue', {}).get('queue_frac', 0.0)):.4f}`",
                                f"- maker_hazard.a: `{float(params.get('maker_hazard', {}).get('a', 0.0)):.4f}`",
                            ]
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    run.update_pointers(
                        execution_params_json=str(exec_params_path),
                        execution_realism_report_md=str(exec_report_path),
                        execution_features_parquet=str(exec_feat_path),
                    )
                    run.log(
                        "step_completed",
                        step=step,
                        maker_queue_queue_frac=float(params.get("maker_queue", {}).get("queue_frac", 0.0) or 0.0),
                        maker_hazard_a=float(params.get("maker_hazard", {}).get("a", 0.0) or 0.0),
                    )
                    run.set_step(step, "completed")
                except Exception as e:
                    run.set_step(step, "failed", error=f"{type(e).__name__}:{e}")
                    run.log("step_failed", step=step, error=f"{type(e).__name__}:{e}")
                    if bool(args.require_execution_params):
                        raise

        # Step 1: generate + calibration artifacts
        step = "generate"
        candidates_path = out_base / "candidates.jsonl"
        calibration_path = out_base / "calibration.json"
        selectivity_path = out_base / "selectivity.parquet"
        if not (bool(args.resume) and run.read_manifest().get("steps", {}).get(step, {}).get("status") == "completed" and candidates_path.exists()):
            run.set_step(step, "running")
            cols = [c for c in ("F_ofi_z", "F_intensity_z", "spread_z", "rv_short", "rv_z", "top_depth_imbalance", "liq_rate_z") if c in physics.columns]
            calibration = compute_calibration(physics.tail(max(1000, min(len(physics), 200_000))), columns=cols or ["ts_ms"])
            save_calibration(calibration, calibration_path)
            _write_md(
                reports_dir / "generator_calibration.md",
                [
                    f"# Generator Calibration - {symbol}",
                    "",
                    f"- sample_count: `{int(calibration.sample_count)}`",
                    f"- calibrated_columns: `{','.join(sorted(calibration.quantiles.keys()))}`",
                ],
            )
            target = float(args.target_triggers_per_day)
            band = max(0.0, float(args.target_trigger_band))
            min_tpd = max(float(args.min_triggers_per_day), target * (1.0 - band))
            max_tpd = min(float(args.max_triggers_per_day), target * (1.0 + band))
            specs = generate_candidates(
                horizons=_parse_int_list(args.horizons),
                compression_options=[False, True],
                vacuum_options=[False, True],
                limit=int(args.limit),
                calibration=calibration,
                frame=physics.tail(max(10_000, min(len(physics), 400_000))).reset_index(drop=True),
                coverage_guarantee=True,
                min_triggered=int(args.min_triggered),
                max_tries=30,
                target_triggers_per_day=target,
                min_triggers_per_day=min_tpd,
                max_triggers_per_day=max_tpd,
                available_columns=physics.columns.tolist(),
            )
            if not specs:
                run.set_step(step, "failed", error="fail_fast:candidates_empty")
                raise RuntimeError("fail_fast:candidates_empty")
            trig_vals = [int((s.meta or {}).get("calibration_triggered", 0) or 0) for s in specs]
            if max(trig_vals) <= 0:
                run.set_step(step, "failed", error="fail_fast:triggered_zero")
                raise RuntimeError("fail_fast:triggered_zero")
            candidates_path.write_text(specs_to_jsonl(specs), encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "signal": s.name,
                        "calibration_triggered": int((s.meta or {}).get("calibration_triggered", 0) or 0),
                        "trigger_rate_per_day": float((s.meta or {}).get("trigger_rate_per_day", 0.0) or 0.0),
                        "relax_steps": int((s.meta or {}).get("relax_steps", 0) or 0),
                        "tighten_steps": int((s.meta or {}).get("tighten_steps", 0) or 0),
                    }
                    for s in specs
                ]
            ).sort_values("signal").to_parquet(selectivity_path, index=False)
            _write_md(
                reports_dir / "generator_coverage_guard.md",
                [
                    f"# Generator Coverage Guard - {symbol}",
                    "",
                    f"- candidates: `{len(specs)}`",
                    f"- triggered min/median/max: `{min(trig_vals)}/{sorted(trig_vals)[len(trig_vals)//2]}/{max(trig_vals)}`",
                ],
            )
            run.update_pointers(candidates_jsonl=str(candidates_path), selectivity_parquet=str(selectivity_path), calibration_json=str(calibration_path))
            run.log("step_completed", step=step, count=len(specs))
            run.set_step(step, "completed")

        specs = [signal_from_dict(json.loads(line)) for line in candidates_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        calibration = compute_calibration(physics.tail(max(1000, min(len(physics), 200_000))), columns=[c for c in ("F_ofi_z", "F_intensity_z", "spread_z") if c in physics.columns])

        # Step 2: overlap report
        step = "overlap"
        overlap_path = out_base / "overlap_pairs.parquet"
        if not (bool(args.resume) and run.read_manifest().get("steps", {}).get(step, {}).get("status") == "completed" and overlap_path.exists()):
            run.set_step(step, "running")
            pairs = pairwise_overlap(physics.tail(max(20_000, min(len(physics), 300_000))), specs, calibration=calibration)
            pairs.to_parquet(overlap_path, index=False)
            _write_md(
                reports_dir / "candidate_overlap.md",
                [
                    f"# Candidate Overlap - {symbol}",
                    "",
                    f"- pairs: `{len(pairs)}`",
                    f"- jaccard>=0.90: `{int((pd.to_numeric(pairs.get('jaccard'), errors='coerce') >= 0.90).sum()) if not pairs.empty else 0}`",
                ],
            )
            run.update_pointers(overlap_pairs_parquet=str(overlap_path))
            run.log("step_completed", step=step, pair_rows=len(pairs))
            run.set_step(step, "completed")

        pairs = pd.read_parquet(overlap_path) if overlap_path.exists() else pd.DataFrame()

        # Step 3: dedup
        step = "dedup"
        dedup_path = out_base / "candidates_deduped.jsonl"
        if not (bool(args.resume) and run.read_manifest().get("steps", {}).get(step, {}).get("status") == "completed" and dedup_path.exists()):
            run.set_step(step, "running")
            dres = dedupe_specs(
                specs,
                pairs,
                jaccard_thr=float(args.jaccard_thr),
                target_triggers_per_day=float(args.target_triggers_per_day),
            )
            if not dres.selected:
                run.set_step(step, "failed", error="fail_fast:dedup_empty")
                raise RuntimeError("fail_fast:dedup_empty")
            dedup_path.write_text(specs_to_jsonl(dres.selected), encoding="utf-8")
            _write_md(
                reports_dir / "dedup_summary.md",
                [
                    f"# Dedup Summary - {symbol}",
                    "",
                    f"- input: `{len(specs)}`",
                    f"- selected: `{len(dres.selected)}`",
                    f"- dropped: `{len(dres.dropped)}`",
                ],
            )
            run.update_pointers(candidates_deduped_jsonl=str(dedup_path))
            run.log("step_completed", step=step, selected=len(dres.selected), dropped=len(dres.dropped))
            run.set_step(step, "completed")

        dedup_specs = [signal_from_dict(json.loads(line)) for line in dedup_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        # Step 4: walkforward eval
        step = "eval"
        eval_path = out_base / "eval.parquet"
        trades_path = out_base / "trades.parquet"
        if not (bool(args.resume) and run.read_manifest().get("steps", {}).get(step, {}).get("status") == "completed" and eval_path.exists()):
            run.set_step(step, "running")
            eval_df, trades_df = evaluate_walkforward(
                physics,
                dedup_specs,
                splits=int(args.splits),
                fee_bps=float(args.fee_bps),
                latency_bars=int(args.latency_bars),
                mode=str(args.mode),
                fill_prob=float(args.fill_prob),
                max_trades_per_day=int(args.max_trades_per_day),
            )
            eval_df.to_parquet(eval_path, index=False)
            trades_df.to_parquet(trades_path, index=False)
            if eval_df.empty or int(pd.to_numeric(eval_df.get("test_trade_count"), errors="coerce").fillna(0).sum()) <= 0:
                run.set_step(step, "failed", error="fail_fast:eval_zero_trades")
                raise RuntimeError("fail_fast:eval_zero_trades")
            _write_md(
                reports_dir / "walkforward.md",
                [
                    f"# Walkforward - {symbol}",
                    "",
                    f"- eval rows: `{len(eval_df)}`",
                    f"- trades rows: `{len(trades_df)}`",
                    f"- total test trades: `{int(pd.to_numeric(eval_df['test_trade_count'], errors='coerce').sum())}`",
                ],
            )
            run.update_pointers(eval_parquet=str(eval_path), trades_parquet=str(trades_path))
            run.log("step_completed", step=step, eval_rows=len(eval_df), trade_rows=len(trades_df))
            run.set_step(step, "completed")

        eval_df = pd.read_parquet(eval_path)

        # Step 5: select + ensemble
        step = "select"
        selected_path = out_base / "selected.parquet"
        ensemble_path = out_base / "ensemble.parquet"
        summary_path = out_base / "selection_summary.parquet"
        if not (bool(args.resume) and run.read_manifest().get("steps", {}).get(step, {}).get("status") == "completed" and selected_path.exists()):
            run.set_step(step, "running")
            summary = summarize_signals(eval_df)
            selected = select_robust_signals(
                summary,
                min_trades_per_split=int(args.min_trades_per_split),
                min_stability=float(args.min_stability),
            )
            spec_map = {s.name: s for s in dedup_specs}
            top_names = selected.head(max(1, int(args.top_k)))["signal"].astype(str).tolist()
            top_specs = [spec_map[n] for n in top_names if n in spec_map]
            picked = pick_topk_by_regime(selected, spec_map, top_k=max(1, int(args.top_k)))
            ensemble = build_ensemble_scores(physics, top_specs)

            summary.to_parquet(summary_path, index=False)
            selected.to_parquet(selected_path, index=False)
            ensemble.to_parquet(ensemble_path, index=False)
            run.update_pointers(selection_summary_parquet=str(summary_path), selected_parquet=str(selected_path), ensemble_parquet=str(ensemble_path))

            diag = _selection_diagnostics(summary, int(args.min_trades_per_split), float(args.min_stability))
            lines = [
                f"# Alpha Discovery - {symbol}",
                "",
                f"- summary rows: `{len(summary)}`",
                f"- selected rows: `{len(selected)}`",
                "",
                "## Filter Diagnostics",
                "",
                f"- low_trades: `{int(diag.get('low_trades', 0))}`",
                f"- fold_consistency_fail: `{int(diag.get('fold_consistency_fail', 0))}`",
                f"- stability_fail: `{int(diag.get('stability_fail', 0))}`",
                "",
                "## Top 10 by test_net_mean",
                "",
                "| signal | test_net_mean | test_sharpe | trade_count |",
                "|---|---:|---:|---:|",
            ]
            for _, r in summary.sort_values(["test_net_mean", "signal"], ascending=[False, True]).head(10).iterrows():
                lines.append(
                    f"| {r['signal']} | {float(r['test_net_mean']):.8f} | {float(r['test_sharpe']):.6f} | {int(r['test_trade_count'])} |"
                )
            _write_md(reports_dir / "alpha_discovery.md", lines)
            _write_md(
                reports_dir / "ensemble.md",
                [
                    f"# Ensemble - {symbol}",
                    "",
                    f"- ensemble rows: `{len(ensemble)}`",
                    f"- active bars: `{int((pd.to_numeric(ensemble.get('signal_count'), errors='coerce') > 0).sum()) if not ensemble.empty else 0}`",
                    f"- picked regimes: `{len(picked)}`",
                ],
            )
            run.set_step(step, "completed", detail={"selected_count": int(len(selected))})
            run.log("step_completed", step=step, selected=len(selected), ensemble_rows=len(ensemble))
            if selected.empty and not bool(args.allow_empty_selection):
                run.set_step(step, "failed", error="fail_fast:selection_empty")
                raise RuntimeError("fail_fast:selection_empty")

        ensemble = pd.read_parquet(ensemble_path)

        # Optional Step 5.1: regime experts + gating artifacts
        if bool(args.build_regime_experts):
            step = "regime_experts"
            experts_path = ensemble_out_base / "ensemble_regime_experts.parquet"
            gating_path = ensemble_out_base / "ensemble_gating.parquet"
            experts_manifest_path = ensemble_out_base / "ensemble_regime_experts_manifest.json"
            experts_report_path = reports_dir / "ensemble_regime_experts.md"
            gating_report_path = reports_dir / "ensemble_gating.md"
            if not (
                bool(args.resume)
                and run.read_manifest().get("steps", {}).get(step, {}).get("status") == "completed"
                and experts_path.exists()
                and gating_path.exists()
            ):
                run.set_step(step, "running")
                try:
                    trades_for_experts = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
                    if aligned_path is not None and aligned_path.exists():
                        aligned_df = pd.read_parquet(aligned_path)
                    else:
                        aligned_df = physics[["ts_ms", "ts_utc", "symbol", "regime_id"]].copy()
                        aligned_df = aligned_df.rename(columns={"regime_id": "aligned_regime_id"})
                    transfer_df = (
                        pd.read_parquet(transfer_regime_path)
                        if (transfer_regime_path is not None and transfer_regime_path.exists())
                        else pd.DataFrame()
                    )
                    experts_df = build_regime_experts(
                        eval_df=eval_df,
                        trades_df=trades_for_experts,
                        aligned_regimes_df=aligned_df,
                        symbol=symbol,
                        transfer_by_regime_df=transfer_df,
                        cfg=ExpertBuildConfig(
                            top_k_per_regime=max(1, int(args.experts_max_per_regime)),
                            min_trades_per_signal=max(1, int(args.min_trades_per_split)),
                            min_regime_rows=max(1, int(args.min_expert_rows)),
                        ),
                    )
                    sym_aligned = aligned_df[aligned_df["symbol"] == symbol].copy() if "symbol" in aligned_df.columns else aligned_df.copy()
                    gating_df = build_gating_decisions(
                        sym_aligned if not sym_aligned.empty else physics[["ts_ms"]].copy(),
                        experts_df,
                        regime_col=("aligned_regime_id" if "aligned_regime_id" in sym_aligned.columns else "regime_id"),
                        data_quality_ok=True,
                    )
                    experts_df.to_parquet(experts_path, index=False)
                    gating_df.to_parquet(gating_path, index=False)
                    experts_manifest = {
                        "symbol": symbol,
                        "interval_ms": int(interval_ms),
                        "expert_rows": int(len(experts_df)),
                        "gating_rows": int(len(gating_df)),
                        "aligned_regimes_path": str(aligned_path) if aligned_path is not None else "",
                        "transfer_by_regime_path": str(transfer_regime_path) if transfer_regime_path is not None else "",
                        "experts_parquet": str(experts_path),
                        "gating_parquet": str(gating_path),
                    }
                    experts_manifest_path.write_text(json.dumps(experts_manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                    _write_md(
                        experts_report_path,
                        [
                            f"# Ensemble Regime Experts - {symbol}",
                            "",
                            f"- experts rows: `{len(experts_df)}`",
                            f"- gating rows: `{len(gating_df)}`",
                            f"- aligned path: `{str(aligned_path) if aligned_path is not None else 'N/A'}`",
                            f"- transfer-by-regime path: `{str(transfer_regime_path) if transfer_regime_path is not None else 'N/A'}`",
                        ],
                    )
                    _write_md(
                        gating_report_path,
                        [
                            f"# Ensemble Gating - {symbol}",
                            "",
                            f"- fallback_rate: `{float(pd.to_numeric(gating_df.get('fallback_used'), errors='coerce').fillna(0.0).mean()) if not gating_df.empty else 0.0:.4f}`",
                            f"- mean_confidence: `{float(pd.to_numeric(gating_df.get('confidence_score'), errors='coerce').fillna(0.0).mean()) if not gating_df.empty else 0.0:.4f}`",
                        ],
                    )
                    run.update_pointers(
                        ensemble_regime_experts_parquet=str(experts_path),
                        ensemble_gating_parquet=str(gating_path),
                        ensemble_regime_experts_manifest_json=str(experts_manifest_path),
                        ensemble_regime_experts_report_md=str(experts_report_path),
                        ensemble_gating_report_md=str(gating_report_path),
                        aligned_regimes_path=(str(aligned_path) if aligned_path is not None and aligned_path.exists() else ""),
                        transfer_by_regime_path=(
                            str(transfer_regime_path) if transfer_regime_path is not None and transfer_regime_path.exists() else ""
                        ),
                    )
                    run.log("step_completed", step=step, experts_rows=len(experts_df), gating_rows=len(gating_df))
                    run.set_step(step, "completed")
                except Exception as e:
                    run.set_step(step, "failed", error=f"{type(e).__name__}:{e}")
                    run.log("step_failed", step=step, error=f"{type(e).__name__}:{e}")
                    if bool(args.require_regime_experts):
                        raise

        # Step 6: papertrades
        step = "papertrades"
        paper_path = out_base / "papertrades.parquet"
        if not (bool(args.resume) and run.read_manifest().get("steps", {}).get(step, {}).get("status") == "completed" and paper_path.exists()):
            run.set_step(step, "running")
            frame = physics.merge(ensemble[["ts_ms", "ensemble_side", "signal_count"]], on="ts_ms", how="left")
            frame["ensemble_side"] = pd.to_numeric(frame.get("ensemble_side"), errors="coerce").fillna(0.0)
            frame["signal_count"] = pd.to_numeric(frame.get("signal_count"), errors="coerce").fillna(0).astype(int)
            exec_params: dict | None = None
            ptr_json = json.loads(run.pointers_path.read_text(encoding="utf-8")) if run.pointers_path.exists() else {}
            ep_raw = str(ptr_json.get("execution_params_json", "")).strip()
            ep = Path(ep_raw) if ep_raw else None
            if ep is not None and ep.exists() and ep.is_file():
                exec_params = json.loads(ep.read_text(encoding="utf-8"))
            trades = generate_papertrades(
                frame,
                horizon_bars=max(_parse_int_list(args.horizons) or [10]),
                cfg=PaperTradeConfig(
                    mode=str(args.mode),
                    fee_bps=float(args.fee_bps),
                    execution_model=str(args.execution_model),
                    execution_params=exec_params,
                    ttl_bars=10,
                ),
            )
            trades.to_parquet(paper_path, index=False)
            _write_md(
                reports_dir / "papertrades.md",
                [
                    f"# Papertrades - {symbol}",
                    "",
                    f"- rows: `{len(trades)}`",
                    f"- mean pnl_net: `{float(pd.to_numeric(trades.get('pnl_net'), errors='coerce').mean() if not trades.empty else 0.0):.8f}`",
                ],
            )
            run.update_pointers(papertrades_parquet=str(paper_path))
            run.log("step_completed", step=step, rows=len(trades))
            run.set_step(step, "completed")

        run.set_status("completed")
        print(f"run_alpha_pipeline ok run_dir={run_dir}")
        print(f"manifest={run.manifest_path}")
        print(f"pointers={run.pointers_path}")
        return 0
    except Exception as e:
        try:
            run.set_status("failed")  # type: ignore[name-defined]
            run.log("pipeline_failed", error=f"{type(e).__name__}:{e}")  # type: ignore[name-defined]
        except Exception:
            pass
        print(f"run_alpha_pipeline error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
