from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.microphys.alpha.transfer import (
    load_run_pointers,
    load_specs_jsonl,
    rank_source_signals,
    write_specs_jsonl,
)
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export source signal specs for cross-asset transfer evaluation.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--from", dest="from_mode", choices=["selected", "topk"], default="selected")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--score-col", default="test_sharpe")
    p.add_argument("--source-symbol", default="")
    p.add_argument("--out", default="data/derived/transfer")
    return p.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    try:
        run_dir = Path(str(args.run_dir))
        if not run_dir.exists():
            raise RuntimeError("run_dir_missing")
        pointers = load_run_pointers(run_dir)
        cands_path = Path(str(pointers.get("candidates_deduped_jsonl", "")).strip())
        if not cands_path.exists():
            raise RuntimeError("candidates_deduped_missing")
        specs = load_specs_jsonl(cands_path)
        if not specs:
            raise RuntimeError("candidates_deduped_empty")
        spec_map = {s.name: s for s in specs}

        source_symbol = str(args.source_symbol).strip()
        if not source_symbol:
            source_symbol = run_dir.name.split("symbol=")[-1].split("_")[0]
        source_symbol = canonical_symbol(source_symbol)

        selected_names: List[str] = []
        mode = str(args.from_mode)
        if mode == "selected":
            sel_path = Path(str(pointers.get("selected_parquet", "")).strip())
            if not sel_path.exists():
                raise RuntimeError("selected_parquet_missing")
            sel = pd.read_parquet(sel_path)
            selected_names = sorted(sel.get("signal", pd.Series([], dtype=str)).astype(str).dropna().unique().tolist())
            if not selected_names:
                raise RuntimeError("selected_parquet_empty")
        else:
            eval_path = Path(str(pointers.get("eval_parquet", "")).strip())
            if not eval_path.exists():
                raise RuntimeError("eval_parquet_missing")
            eval_df = pd.read_parquet(eval_path)
            selected_names = rank_source_signals(eval_df, score_col=str(args.score_col), top_k=max(1, int(args.k)))
            if not selected_names:
                raise RuntimeError("topk_empty")

        picked = [spec_map[n] for n in selected_names if n in spec_map]
        if not picked:
            raise RuntimeError("picked_specs_empty")

        run_id = run_dir.name
        out_dir = Path(str(args.out)) / f"source={source_symbol}" / f"run={run_id}"
        out_specs = out_dir / "exported_specs.jsonl"
        specs_sha = write_specs_jsonl(out_specs, picked)
        manifest = {
            "source_run_dir": str(run_dir),
            "source_symbol": source_symbol,
            "run_id": run_id,
            "export_mode": mode,
            "k": int(args.k),
            "score_col": str(args.score_col),
            "count": int(len(picked)),
            "specs_sha256": specs_sha,
            "pointers": {k: str(v) for k, v in sorted(pointers.items())},
            "exported_specs_jsonl": str(out_specs),
        }
        _write_json(out_dir / "manifest.json", manifest)
        print(f"export_selected_specs ok count={len(picked)} out={out_specs}")
        return 0
    except Exception as e:
        print(f"export_selected_specs error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

