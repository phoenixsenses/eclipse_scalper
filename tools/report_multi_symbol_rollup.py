from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build multi-symbol rollup from run_alpha_multi manifest.")
    p.add_argument("--multi-manifest", required=True)
    p.add_argument("--out-md", default="reports/multi_symbol/rollup.md")
    p.add_argument("--out-parquet", default="data/derived/multi_symbol_metrics/rollup.parquet")
    return p.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not str(path).strip() or str(path) == ".":
        return pd.DataFrame()
    if not path.exists() or not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _path_or_empty(raw: Any) -> Path:
    s = str(raw or "").strip()
    if not s:
        return Path("")
    return Path(s)


def _collect_run_row(run_dir: Path, symbol: str, *, ok: bool, exit_code: int) -> Dict[str, Any]:
    ptr = _read_json(run_dir / "pointers.json") if (run_dir / "pointers.json").exists() else {}
    cand_path = _path_or_empty(ptr.get("candidates_jsonl", ""))
    dedup_path = _path_or_empty(ptr.get("candidates_deduped_jsonl", ""))
    sel_path = _path_or_empty(ptr.get("selected_parquet", ""))
    sum_path = _path_or_empty(ptr.get("selection_summary_parquet", ""))
    eval_path = _path_or_empty(ptr.get("eval_parquet", ""))
    cal_rollup_path = run_dir / "artifacts" / "execution" / "params.json"

    candidates_count = 0
    if str(cand_path).strip() and cand_path.exists() and cand_path.is_file():
        candidates_count = len([x for x in cand_path.read_text(encoding="utf-8").splitlines() if x.strip()])
    deduped_count = 0
    if str(dedup_path).strip() and dedup_path.exists() and dedup_path.is_file():
        deduped_count = len([x for x in dedup_path.read_text(encoding="utf-8").splitlines() if x.strip()])

    selected = _safe_read_parquet(sel_path)
    summary = _safe_read_parquet(sum_path)
    ev = _safe_read_parquet(eval_path)
    s_trade = pd.to_numeric(summary["test_trade_count"], errors="coerce") if ("test_trade_count" in summary.columns) else pd.Series([], dtype=float)
    e_net = pd.to_numeric(ev["test_net_mean"], errors="coerce") if ("test_net_mean" in ev.columns) else pd.Series([], dtype=float)
    e_sh = pd.to_numeric(ev["test_sharpe"], errors="coerce") if ("test_sharpe" in ev.columns) else pd.Series([], dtype=float)
    e_fill = pd.to_numeric(ev["fill_rate"], errors="coerce") if ("fill_rate" in ev.columns) else pd.Series([], dtype=float)
    e_reg = pd.to_numeric(ev["regime_concentration"], errors="coerce") if ("regime_concentration" in ev.columns) else pd.Series([], dtype=float)

    row = {
        "symbol": symbol,
        "run_dir": str(run_dir),
        "ok": bool(ok),
        "exit_code": int(exit_code),
        "candidates_count": int(candidates_count),
        "deduped_count": int(deduped_count),
        "selected_count": int(len(selected)),
        "median_trigger_rate_per_day": float(s_trade.median() if not s_trade.empty else 0.0),
        "walkforward_test_net_mean": float(e_net.mean() if not e_net.empty else 0.0),
        "walkforward_test_sharpe_mean": float(e_sh.mean() if not e_sh.empty else 0.0),
        "fill_rate_mean": float(e_fill.mean() if not e_fill.empty else 0.0),
        "regime_concentration_mean": float(e_reg.mean() if not e_reg.empty else 0.0),
        "execution_params_present": bool(cal_rollup_path.exists()),
    }
    missing = []
    for name, pth in (
        ("candidates_deduped_jsonl", dedup_path),
        ("selected_parquet", sel_path),
        ("selection_summary_parquet", sum_path),
        ("eval_parquet", eval_path),
    ):
        if not str(pth).strip() or not pth.exists():
            missing.append(name)
    row["missing_artifacts"] = ",".join(missing)
    return row


def main() -> int:
    args = _parse_args()
    try:
        manifest = _read_json(Path(str(args.multi_manifest)))
        runs = list(manifest.get("runs", []) or [])
        rows: List[Dict[str, Any]] = []
        for r in runs:
            sym = str(r.get("symbol", ""))
            run_dir = Path(str(r.get("run_dir", "")))
            if not run_dir.exists():
                continue
            rows.append(_collect_run_row(run_dir, sym, ok=bool(r.get("ok", False)), exit_code=int(r.get("exit_code", 0) or 0)))
        df = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True) if rows else pd.DataFrame()
        out_pq = Path(str(args.out_parquet))
        out_pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_pq, index=False)

        lines = [
            "# Multi-Symbol Rollup",
            "",
            f"- symbols: `{len(df)}`",
            f"- source_manifest: `{args.multi_manifest}`",
            "",
            "| symbol | ok | exit_code | candidates | deduped | selected | test_net_mean | test_sharpe_mean | fill_rate_mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _, x in df.iterrows():
            def _na(v: Any, fmt: str = "{:.4f}") -> str:
                try:
                    f = float(v)
                    if pd.isna(f):
                        return "N/A"
                    return fmt.format(f)
                except Exception:
                    return "N/A"
            lines.append(
                f"| {x['symbol']} | {int(bool(x.get('ok', False)))} | {int(x.get('exit_code', 0) or 0)} | "
                f"{int(x['candidates_count'])} | {int(x['deduped_count'])} | {int(x['selected_count'])} | "
                f"{_na(x['walkforward_test_net_mean'], '{:.8f}')} | {_na(x['walkforward_test_sharpe_mean'], '{:.6f}')} | {_na(x['fill_rate_mean'], '{:.4f}')} |"
            )
        failed = df.loc[df["ok"] == False].copy() if (not df.empty and "ok" in df.columns) else pd.DataFrame()
        if not failed.empty:
            lines += [
                "",
                "## Failed Runs",
                "",
                "| symbol | exit_code | run_dir | missing_artifacts |",
                "|---|---:|---|---|",
            ]
            for _, x in failed.iterrows():
                lines.append(
                    f"| {x['symbol']} | {int(x.get('exit_code', 0) or 0)} | {x['run_dir']} | {str(x.get('missing_artifacts', '')) or 'N/A'} |"
                )
        out_md = Path(str(args.out_md))
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"report_multi_symbol_rollup ok rows={len(df)} out={out_md}")
        return 0
    except Exception as e:
        print(f"report_multi_symbol_rollup error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
