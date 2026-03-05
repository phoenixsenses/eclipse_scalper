from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.microphys.alpha.transfer import load_partitioned_parquet, merge_physics_regimes
from src.microphys.regime.alignment import (
    AlignmentConfig,
    SHARED_FEATURES,
    assign_aligned_regimes,
    build_shared_alignment_frame,
    describe_aligned_regimes,
)
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build cross-symbol regime alignment in feature-space.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--regimes", default="data/derived/regimes")
    p.add_argument("--symbols", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--method", choices=["quantile_buckets", "kmeans_global", "gmm_global"], default="quantile_buckets")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-rows", type=int, default=500000)
    p.add_argument("--out", default="data/derived/regime_alignment")
    p.add_argument("--report", default="reports/transfer/regime_alignment.md")
    return p.parse_args()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _resolve_symbols(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in str(raw).split(","):
        s = canonical_symbol(x)
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _write_report(path: Path, summary: pd.DataFrame, manifest: Dict[str, Any]) -> None:
    lines = [
        "# Regime Alignment",
        "",
        f"- method: `{manifest['method']}`",
        f"- k: `{manifest['k']}`",
        f"- symbols: `{','.join(manifest['symbols'])}`",
        f"- rows: `{manifest['rows']}`",
        f"- warnings: `{len(manifest.get('warnings', []))}`",
        "",
        "| aligned_regime_id | rows | eth_frac | btc_frac | rv_z | spread_z | intensity_z | liq_rate_z | impact_proxy | micro_trend | of_flow_persistence |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {int(r['aligned_regime_id'])} | {int(r['rows'])} | {float(r['eth_frac']):.3f} | {float(r['btc_frac']):.3f} | "
            f"{float(r['rv_z_mean']):.3f} | {float(r['spread_z_mean']):.3f} | {float(r['intensity_z_mean']):.3f} | "
            f"{float(r['liq_rate_z_mean']):.3f} | {float(r['impact_proxy_mean']):.3f} | {float(r['micro_trend_mean']):.3f} | "
            f"{float(r['of_flow_persistence_mean']):.3f} |"
        )
    if manifest.get("warnings"):
        lines += ["", "## Warnings", ""]
        for w in list(manifest.get("warnings", [])):
            lines.append(f"- `{w}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    try:
        symbols = _resolve_symbols(args.symbols)
        if len(symbols) < 2:
            raise RuntimeError("need_at_least_two_symbols")
        frames: Dict[str, pd.DataFrame] = {}
        warnings: List[str] = []
        for s in symbols:
            p = load_partitioned_parquet(Path(str(args.physics)), symbol=s, interval_ms=int(args.interval_ms), name="physics")
            r = load_partitioned_parquet(Path(str(args.regimes)), symbol=s, interval_ms=int(args.interval_ms), name="regimes")
            m = merge_physics_regimes(p, r)
            if int(args.sample_rows) > 0 and len(m) > int(args.sample_rows):
                m = m.tail(int(args.sample_rows)).copy()
            frames[s] = m
            if m.empty:
                warnings.append(f"{s}:empty")
        shared, warn2 = build_shared_alignment_frame(frames)
        warnings.extend(warn2)
        if shared.empty:
            raise RuntimeError("alignment_empty")
        aligned = assign_aligned_regimes(
            shared,
            AlignmentConfig(method=str(args.method), k=int(args.k), seed=int(args.seed), sample_rows=int(args.sample_rows)),
        )
        summary = describe_aligned_regimes(aligned)
        out_base = Path(str(args.out)) / f"interval_ms={int(args.interval_ms)}"
        out_base.mkdir(parents=True, exist_ok=True)
        aligned_path = out_base / "aligned_regimes.parquet"
        summary_path = out_base / "summary.parquet"
        manifest_path = out_base / "manifest.json"
        aligned.to_parquet(aligned_path, index=False)
        summary.to_parquet(summary_path, index=False)
        manifest = {
            "symbols": symbols,
            "interval_ms": int(args.interval_ms),
            "method": str(args.method),
            "k": int(args.k),
            "seed": int(args.seed),
            "sample_rows": int(args.sample_rows),
            "rows": int(len(aligned)),
            "features": list(SHARED_FEATURES),
            "warnings": sorted(set(str(x) for x in warnings)),
            "aligned_regimes_parquet": str(aligned_path),
            "summary_parquet": str(summary_path),
        }
        _write_json(manifest_path, manifest)
        _write_report(Path(str(args.report)), summary, manifest)
        print(f"build_regime_alignment ok rows={len(aligned)} out={aligned_path}")
        return 0
    except Exception as e:
        print(f"build_regime_alignment error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

