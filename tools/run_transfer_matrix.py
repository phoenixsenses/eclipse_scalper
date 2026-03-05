from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tools import (
    build_regime_alignment,
    eval_transfer,
    export_selected_specs,
    report_transfer_by_aligned_regime,
    report_transfer_matrix,
)
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run directed cross-asset transfer matrix.")
    p.add_argument("--symbols", required=True, help="comma-separated symbols, e.g. ETHUSDT,BTCUSDT")
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
    p.add_argument("--calibration-modes", default="", help="optional comma list, e.g. source,target")
    p.add_argument("--target-calibration", default="")
    p.add_argument("--live-root", default="data/live")
    p.add_argument("--from", dest="from_mode", choices=["selected", "topk"], default="topk")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--score-col", default="test_sharpe")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--regimes", default="data/derived/regimes")
    p.add_argument("--alpha-multi-latest", default="data/runs/alpha_multi/LATEST.json")
    p.add_argument("--alpha-runs-root", default="data/runs/alpha")
    p.add_argument("--out", default="data/derived/transfer_matrix")
    p.add_argument("--reports-out", default="reports/transfer")
    p.add_argument("--require-all-ok", action="store_true")
    p.add_argument("--with-regime-alignment", action="store_true")
    p.add_argument("--alignment-method", choices=["quantile_buckets", "kmeans_global", "gmm_global"], default="kmeans_global")
    p.add_argument("--alignment-k", type=int, default=6)
    p.add_argument("--alignment-sample-rows", type=int, default=500000)
    p.add_argument("--alignment-out", default="data/derived/regime_alignment")
    return p.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("matrix_%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_symbols(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in str(raw).split(","):
        s = canonical_symbol(item)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _resolve_modes(args: argparse.Namespace) -> List[str]:
    raw = str(args.calibration_modes).strip()
    if not raw:
        return [str(args.calibration_mode)]
    out: List[str] = []
    seen = set()
    for item in raw.split(","):
        s = str(item).strip().lower()
        if s in {"source", "target"} and s not in seen:
            out.append(s)
            seen.add(s)
    return out or [str(args.calibration_mode)]


def _latest_runs_from_multi(path: Path) -> Dict[str, Path]:
    if not path.exists():
        return {}
    latest = _read_json(path)
    multi_run_dir = Path(str(latest.get("multi_run_dir", "")).strip())
    if not multi_run_dir.exists():
        return {}
    manifest = multi_run_dir / "manifest.json"
    if not manifest.exists():
        return {}
    m = _read_json(manifest)
    out: Dict[str, Path] = {}
    for r in list(m.get("runs", []) or []):
        sym = str(r.get("symbol", "")).strip().upper()
        run_dir = Path(str(r.get("run_dir", "")).strip())
        if sym and run_dir.exists():
            out[sym] = run_dir
    return out


def _latest_runs_from_alpha(root: Path, symbols: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not root.exists():
        return out
    dirs = [d for d in root.iterdir() if d.is_dir()]
    for sym in symbols:
        matches = [d for d in dirs if f"symbol={sym}" in d.name]
        if not matches:
            continue
        matches.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
        out[sym] = matches[0]
    return out


def _run_main_with_argv(module_main, argv: List[str]) -> int:
    old = sys.argv
    try:
        sys.argv = argv
        return int(module_main())
    finally:
        sys.argv = old


def run_transfer_matrix(args: argparse.Namespace) -> int:
    symbols = _resolve_symbols(args.symbols)
    if len(symbols) < 2:
        raise RuntimeError("need_at_least_two_symbols")
    modes = _resolve_modes(args)
    out_root = Path(str(args.out))
    matrix_dir = out_root / _run_id()
    matrix_dir.mkdir(parents=True, exist_ok=True)

    runs = _latest_runs_from_multi(Path(str(args.alpha_multi_latest)))
    if not runs:
        runs = _latest_runs_from_alpha(Path(str(args.alpha_runs_root)), symbols)
    missing = [s for s in symbols if s not in runs]
    if missing:
        raise RuntimeError(f"missing_source_runs:{','.join(missing)}")

    pair_rows: List[Dict[str, Any]] = []
    all_ok = True
    for src in symbols:
        for tgt in symbols:
            if src == tgt:
                continue
            for mode in modes:
                pair_id = f"{src}_to_{tgt}__cal={mode}"
                pair_dir = matrix_dir / "pairs" / pair_id
                pair_dir.mkdir(parents=True, exist_ok=True)
                export_out = pair_dir / "export"
                export_rc = _run_main_with_argv(
                    export_selected_specs.main,
                    [
                        "export_selected_specs",
                        "--run-dir",
                        str(runs[src]),
                        "--from",
                        str(args.from_mode),
                        "--k",
                        str(int(args.k)),
                        "--score-col",
                        str(args.score_col),
                        "--source-symbol",
                        src,
                        "--out",
                        str(export_out),
                    ],
                )
                exported = export_out / f"source={src}" / f"run={runs[src].name}" / "exported_specs.jsonl"
                transfer_out = pair_dir / "transfer"
                pair_report = pair_dir / "transfer.md"
                eval_argv = [
                    "eval_transfer",
                    "--exported",
                    str(exported),
                    "--physics",
                    str(args.physics),
                    "--regimes",
                    str(args.regimes),
                    "--source-symbol",
                    src,
                    "--target-symbol",
                    tgt,
                    "--interval-ms",
                    str(int(args.interval_ms)),
                    "--splits",
                    str(int(args.splits)),
                    "--seed",
                    str(int(args.seed)),
                    "--mode",
                    str(args.mode),
                    "--fee-bps",
                    str(float(args.fee_bps)),
                    "--latency-bars",
                    str(int(args.latency_bars)),
                    "--fill-prob",
                    str(float(args.fill_prob)),
                    "--max-trades-per-day",
                    str(int(args.max_trades_per_day)),
                    "--execution-model",
                    str(args.execution_model),
                    "--ttl-bars",
                    str(int(args.ttl_bars)),
                    "--calibration-mode",
                    str(mode),
                    "--live-root",
                    str(args.live_root),
                    "--out",
                    str(transfer_out),
                    "--report",
                    str(pair_report),
                ]
                if str(args.execution_params).strip():
                    eval_argv += ["--execution-params", str(args.execution_params)]
                if str(args.target_calibration).strip():
                    eval_argv += ["--target-calibration", str(args.target_calibration)]
                eval_rc = _run_main_with_argv(eval_transfer.main, eval_argv)
                eval_parquet = transfer_out / f"source={src}" / f"target={tgt}" / f"interval_ms={int(args.interval_ms)}" / "eval_transfer.parquet"
                transfer_manifest = transfer_out / f"source={src}" / f"target={tgt}" / f"interval_ms={int(args.interval_ms)}" / "manifest.json"
                ok = bool(export_rc == 0 and eval_rc == 0 and eval_parquet.exists())
                all_ok = bool(all_ok and ok)
                row = {
                    "pair_id": pair_id,
                    "source_symbol": src,
                    "target_symbol": tgt,
                    "calibration_mode": mode,
                    "source_run_dir": str(runs[src]),
                    "target_run_dir": str(runs[tgt]),
                    "exported_specs_jsonl": str(exported),
                    "eval_transfer_parquet": str(eval_parquet),
                    "transfer_manifest_json": str(transfer_manifest),
                    "report_md": str(pair_report),
                    "export_exit_code": int(export_rc),
                    "eval_exit_code": int(eval_rc),
                    "ok": bool(ok),
                }
                pair_rows.append(row)

    manifest = {
        "run_id": matrix_dir.name,
        "created_utc": _utc_now(),
        "symbols": symbols,
        "calibration_modes": modes,
        "params": {
            "interval_ms": int(args.interval_ms),
            "splits": int(args.splits),
            "seed": int(args.seed),
            "mode": str(args.mode),
            "fee_bps": float(args.fee_bps),
            "latency_bars": int(args.latency_bars),
            "execution_model": str(args.execution_model),
            "from_mode": str(args.from_mode),
            "k": int(args.k),
            "score_col": str(args.score_col),
        },
        "pairs": pair_rows,
        "ok": bool(all_ok),
    }
    manifest_path = matrix_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    matrix_report_md = Path(str(args.reports_out)) / "transfer_matrix.md"
    matrix_parquet = out_root / "transfer_matrix.parquet"
    report_rc = _run_main_with_argv(
        report_transfer_matrix.main,
        [
            "report_transfer_matrix",
            "--matrix-manifest",
            str(manifest_path),
            "--out-md",
            str(matrix_report_md),
            "--out-parquet",
            str(matrix_parquet),
        ],
    )
    align_rc = 0
    aligned_path = ""
    aligned_report = ""
    transfer_by_regime_md = ""
    transfer_by_regime_pq = ""
    if bool(args.with_regime_alignment):
        aligned_report = str(Path(str(args.reports_out)) / "regime_alignment.md")
        align_rc = _run_main_with_argv(
            build_regime_alignment.main,
            [
                "build_regime_alignment",
                "--physics",
                str(args.physics),
                "--regimes",
                str(args.regimes),
                "--symbols",
                ",".join(symbols),
                "--interval-ms",
                str(int(args.interval_ms)),
                "--method",
                str(args.alignment_method),
                "--k",
                str(int(args.alignment_k)),
                "--sample-rows",
                str(int(args.alignment_sample_rows)),
                "--out",
                str(args.alignment_out),
                "--report",
                aligned_report,
            ],
        )
        aligned_path = str(Path(str(args.alignment_out)) / f"interval_ms={int(args.interval_ms)}" / "aligned_regimes.parquet")
        transfer_by_regime_md = str(Path(str(args.reports_out)) / "transfer_by_aligned_regime.md")
        transfer_by_regime_pq = str(Path(str(args.alignment_out)) / "transfer_by_regime.parquet")
        if align_rc == 0:
            align_rc = _run_main_with_argv(
                report_transfer_by_aligned_regime.main,
                [
                    "report_transfer_by_aligned_regime",
                    "--matrix-manifest",
                    str(manifest_path),
                    "--aligned-regimes",
                    aligned_path,
                    "--out-parquet",
                    transfer_by_regime_pq,
                    "--out-md",
                    transfer_by_regime_md,
                ],
            )

    latest = {
        "matrix_run_dir": str(matrix_dir),
        "ts_utc": _utc_now(),
        "pairs": [{"source": r["source_symbol"], "target": r["target_symbol"], "calibration_mode": r["calibration_mode"], "ok": r["ok"]} for r in pair_rows],
        "report_md": str(matrix_report_md),
        "parquet": str(matrix_parquet),
        "aligned_regimes_parquet": aligned_path,
        "regime_alignment_report_md": aligned_report,
        "transfer_by_aligned_regime_md": transfer_by_regime_md,
        "transfer_by_aligned_regime_parquet": transfer_by_regime_pq,
        "ok": bool(all_ok and report_rc == 0 and align_rc == 0),
    }
    _write_json(out_root / "LATEST.json", latest)
    print(f"run_transfer_matrix ok={int(bool(all_ok))} run_dir={matrix_dir}")
    if bool(args.require_all_ok) and (not all_ok or report_rc != 0 or align_rc != 0):
        return 2
    return 0 if (report_rc == 0 and align_rc == 0) else 2


def main() -> int:
    try:
        return run_transfer_matrix(_parse_args())
    except Exception as e:
        print(f"run_transfer_matrix error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
