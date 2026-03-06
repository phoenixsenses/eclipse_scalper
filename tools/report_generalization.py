from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.microphys.alpha.generalization import compute_family_generalization, infer_family


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build cross-symbol generalization report from multi manifest.")
    p.add_argument("--multi-manifest", required=True)
    p.add_argument("--out-md", default="reports/multi_symbol/generalization.md")
    p.add_argument("--out-parquet", default="data/derived/multi_symbol_metrics/generalization.parquet")
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


def _load_symbol_frames(run_dir: Path) -> Dict[str, pd.DataFrame]:
    ptr = _read_json(run_dir / "pointers.json") if (run_dir / "pointers.json").exists() else {}
    cand_path = _path_or_empty(ptr.get("candidates_deduped_jsonl", ""))
    sel_path = _path_or_empty(ptr.get("selected_parquet", ""))
    sum_path = _path_or_empty(ptr.get("selection_summary_parquet", ""))

    cand_rows = []
    if str(cand_path).strip() and cand_path.exists() and cand_path.is_file():
        for line in cand_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                j = json.loads(s)
                nm = str(j.get("name", ""))
                cand_rows.append({"signal": nm, "family": infer_family(nm)})
            except Exception:
                continue
    candidates = pd.DataFrame(cand_rows).drop_duplicates() if cand_rows else pd.DataFrame(columns=["signal", "family"])
    selected = _safe_read_parquet(sel_path)
    if not selected.empty and "signal" in selected.columns:
        selected = selected.copy()
        selected["signal"] = selected["signal"].astype(str)
        selected["family"] = selected["signal"].map(infer_family)
    else:
        selected = pd.DataFrame(columns=["signal", "family"])

    summary = _safe_read_parquet(sum_path)
    if not summary.empty and "signal" in summary.columns:
        summary = summary.copy()
        summary["signal"] = summary["signal"].astype(str)
        summary["family"] = summary["signal"].map(infer_family)
    else:
        summary = pd.DataFrame(columns=["signal", "family", "test_net_mean", "regime_concentration"])
    return {"candidates": candidates, "selected": selected, "summary": summary}


def main() -> int:
    args = _parse_args()
    try:
        manifest = _read_json(Path(str(args.multi_manifest)))
        per_symbol: Dict[str, Dict[str, pd.DataFrame]] = {}
        runs = list(manifest.get("runs", []) or [])
        failed_runs = []
        for r in runs:
            sym = str(r.get("symbol", ""))
            run_dir = Path(str(r.get("run_dir", "")))
            ok = bool(r.get("ok", False))
            if not ok:
                failed_runs.append(
                    {
                        "symbol": sym or "N/A",
                        "run_dir": str(run_dir) if str(run_dir) else "N/A",
                        "exit_code": int(r.get("exit_code", 0) or 0),
                    }
                )
                continue
            if not sym or not run_dir.exists():
                continue
            per_symbol[sym] = _load_symbol_frames(run_dir)

        g = compute_family_generalization(per_symbol=per_symbol)
        out_pq = Path(str(args.out_parquet))
        out_pq.parent.mkdir(parents=True, exist_ok=True)
        g.to_parquet(out_pq, index=False)

        lines = [
            "# Multi-Symbol Generalization",
            "",
            f"- symbols_ok: `{','.join(sorted(per_symbol.keys())) if per_symbol else 'N/A'}`",
            f"- symbols_total: `{len(runs)}`",
            f"- symbols_failed: `{len(failed_runs)}`",
            f"- source_manifest: `{args.multi_manifest}`",
            "",
            "| family | survival | rank_consistency | directional_consistency | regime_similarity | generalization_score |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for _, r in g.iterrows():
            lines.append(
                f"| {r['family']} | {float(r['survival_frac_mean']):.4f} | {float(r['rank_consistency']):.4f} | "
                f"{float(r['directional_consistency']):.4f} | {float(r['regime_similarity']):.4f} | {float(r['generalization_score']):.4f} |"
            )
        if not failed_runs:
            pass
        else:
            lines += [
                "",
                "## Failed Runs",
                "",
                "| symbol | exit_code | run_dir |",
                "|---|---:|---|",
            ]
            for fr in failed_runs:
                lines.append(f"| {fr['symbol']} | {fr['exit_code']} | {fr['run_dir']} |")
        out_md = Path(str(args.out_md))
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"report_generalization ok rows={len(g)} out={out_md}")
        return 0
    except Exception as e:
        print(f"report_generalization error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
