from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from tools import report_generalization, report_multi_symbol_rollup
from tools import run_alpha_pipeline as alpha_pipeline
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run alpha pipeline for multiple symbols with shared settings.")
    p.add_argument("--symbols", required=True, help="comma-separated, e.g. ETHUSDT,BTCUSDT")
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--regimes", default="data/derived/regimes")
    p.add_argument("--out-root", default="data/runs/alpha_multi")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--splits", type=int, default=3)
    p.add_argument("--mode", choices=["taker", "maker"], default="taker")
    p.add_argument("--fee-bps", type=float, default=0.5)
    p.add_argument("--latency-bars", type=int, default=2)
    p.add_argument("--target-triggers-per-day", type=float, default=200.0)
    p.add_argument("--jaccard-thr", type=float, default=0.9)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--calibrate-execution", action="store_true")
    p.add_argument("--with-reports", action="store_true")
    p.add_argument("--reports-out", default="reports/multi_symbol")
    p.add_argument("--metrics-out", default="data/derived/multi_symbol_metrics")
    p.add_argument("--require-all-ok", action="store_true")
    return p.parse_args()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("multi_%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        return str(out).strip()
    except Exception:
        return ""


def run_alpha_multi(args: argparse.Namespace) -> int:
    symbols = [canonical_symbol(x) for x in str(args.symbols).split(",") if str(x).strip()]
    if not symbols:
        raise RuntimeError("symbols_empty")
    out_root = Path(str(args.out_root))
    run_root = out_root / _run_id()
    run_root.mkdir(parents=True, exist_ok=True)
    runs: List[Dict[str, Any]] = []
    ok_all = True
    started = time.perf_counter()
    for sym in symbols:
        run_dir = run_root / f"symbol={sym}"
        t0 = time.perf_counter()
        argv = [
            "run_alpha_pipeline",
            "--symbol",
            sym,
            "--interval-ms",
            str(int(args.interval_ms)),
            "--physics",
            str(args.physics),
            "--regimes",
            str(args.regimes),
            "--run-dir",
            str(run_dir),
            "--seed",
            str(int(args.seed)),
            "--splits",
            str(int(args.splits)),
            "--mode",
            str(args.mode),
            "--fee-bps",
            str(float(args.fee_bps)),
            "--latency-bars",
            str(int(args.latency_bars)),
            "--target-triggers-per-day",
            str(float(args.target_triggers_per_day)),
            "--jaccard-thr",
            str(float(args.jaccard_thr)),
        ]
        if bool(args.quick):
            argv.append("--quick")
        if bool(args.resume):
            argv.append("--resume")
        if bool(args.calibrate_execution):
            argv.append("--calibrate-execution")

        old_argv = sys.argv
        try:
            sys.argv = argv
            rc = int(alpha_pipeline.main())
        finally:
            sys.argv = old_argv
        manifest_path = run_dir / "manifest.json"
        pointers_path = run_dir / "pointers.json"
        ok = rc == 0 and manifest_path.exists() and pointers_path.exists()
        ok_all = bool(ok_all and ok)
        runs.append(
            {
                "symbol": sym,
                "run_dir": str(run_dir),
                "manifest_path": str(manifest_path),
                "pointers_path": str(pointers_path),
                "exit_code": int(rc),
                "ok": bool(ok),
                "duration_sec": float(max(0.0, time.perf_counter() - t0)),
            }
        )

    payload = {
        "run_id": run_root.name,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params": {
            "symbols": symbols,
            "interval_ms": int(args.interval_ms),
            "physics": str(args.physics),
            "regimes": str(args.regimes),
            "seed": int(args.seed),
            "splits": int(args.splits),
            "mode": str(args.mode),
            "fee_bps": float(args.fee_bps),
            "latency_bars": int(args.latency_bars),
            "target_triggers_per_day": float(args.target_triggers_per_day),
            "jaccard_thr": float(args.jaccard_thr),
            "quick": bool(args.quick),
            "resume": bool(args.resume),
            "with_reports": bool(args.with_reports),
            "require_all_ok": bool(args.require_all_ok),
        },
        "runtime": {
            "duration_sec": float(max(0.0, time.perf_counter() - started)),
            "python_version": platform.python_version(),
            "git_commit": _git_commit(),
        },
        "runs": runs,
        "ok": bool(ok_all),
    }
    _write_json(run_root / "manifest.json", payload)
    _write_json(run_root / "index.json", {"runs": runs})
    outputs: Dict[str, str] = {}
    if bool(args.with_reports):
        rollup_md = Path(str(args.reports_out)) / "rollup.md"
        rollup_pq = Path(str(args.metrics_out)) / "rollup.parquet"
        gen_md = Path(str(args.reports_out)) / "generalization.md"
        gen_pq = Path(str(args.metrics_out)) / "generalization.parquet"
        old_argv = sys.argv
        try:
            sys.argv = [
                "report_multi_symbol_rollup",
                "--multi-manifest",
                str(run_root / "manifest.json"),
                "--out-md",
                str(rollup_md),
                "--out-parquet",
                str(rollup_pq),
            ]
            _ = report_multi_symbol_rollup.main()
            sys.argv = [
                "report_generalization",
                "--multi-manifest",
                str(run_root / "manifest.json"),
                "--out-md",
                str(gen_md),
                "--out-parquet",
                str(gen_pq),
            ]
            _ = report_generalization.main()
        finally:
            sys.argv = old_argv
        outputs = {
            "rollup_md": str(rollup_md),
            "rollup_parquet": str(rollup_pq),
            "generalization_md": str(gen_md),
            "generalization_parquet": str(gen_pq),
        }
        _write_json(run_root / "reports.json", outputs)

    latest_payload = {
        "multi_run_dir": str(run_root),
        "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbols": symbols,
        "ok": bool(ok_all),
        "outputs": outputs,
    }
    _write_json(out_root / "LATEST.json", latest_payload)
    print(f"run_alpha_multi ok={int(bool(ok_all))} run_root={run_root}")
    if bool(args.require_all_ok) and not ok_all:
        return 2
    return 0


def main() -> int:
    try:
        return run_alpha_multi(_parse_args())
    except Exception as e:
        print(f"run_alpha_multi error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
