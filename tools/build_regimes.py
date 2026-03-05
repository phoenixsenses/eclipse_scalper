from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.microphys.regime.features import build_regime_features
from src.microphys.regime.models import RegimeFitConfig, fit_regimes
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build regime labels from physics parquet.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--method", default="hmm", choices=["hmm", "gmm", "kmeans"])
    p.add_argument("--n-regimes", type=int, default=4)
    p.add_argument("--rolling", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/derived/regimes")
    p.add_argument("--report", default=None)
    return p.parse_args()


def _iter_physics(physics_root: Path, symbol: str, interval_ms: int) -> list[Path]:
    base = physics_root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    return sorted(base.glob("date=*/physics.parquet")) if base.exists() else []


def build_regimes(
    physics_root: Path,
    out_root: Path,
    symbol: str,
    interval_ms: int,
    method: str,
    n_regimes: int,
    rolling: int,
    seed: int,
) -> dict[str, Any]:
    sym = canonical_symbol(symbol)
    files = _iter_physics(physics_root, sym, interval_ms)
    if not files:
        raise RuntimeError(f"no_physics_partitions symbol={sym} interval_ms={interval_ms}")

    out_base = out_root / f"interval_ms={int(interval_ms)}" / f"symbol={sym}"
    out_base.mkdir(parents=True, exist_ok=True)

    part_stats: list[dict[str, Any]] = []
    for fp in files:
        day = fp.parent.name.split("=", 1)[1]
        df = pd.read_parquet(fp)
        feats = build_regime_features(df, rolling=rolling)
        labeled = fit_regimes(feats, RegimeFitConfig(method=method, n_regimes=n_regimes, seed=seed))

        day_dir = out_base / f"date={day}"
        day_dir.mkdir(parents=True, exist_ok=True)
        out_pq = day_dir / "regimes.parquet"
        labeled.to_parquet(out_pq, index=False)

        counts = labeled["regime_id"].value_counts().sort_index().to_dict()
        pman = {
            "date": day,
            "symbol": sym,
            "interval_ms": int(interval_ms),
            "rows": int(len(labeled)),
            "regime_counts": {str(k): int(v) for k, v in counts.items()},
            "input": str(fp),
            "output": str(out_pq),
        }
        (day_dir / "manifest.json").write_text(json.dumps(pman, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        part_stats.append(pman)

    run_manifest = {
        "symbol": sym,
        "interval_ms": int(interval_ms),
        "method": method,
        "n_regimes": int(n_regimes),
        "rolling": int(rolling),
        "seed": int(seed),
        "partitions": part_stats,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_base / "manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return run_manifest


def _write_report(manifest: dict[str, Any], out_path: Path) -> None:
    lines: list[str] = []
    lines.append(f"# Regime Report - {manifest['symbol']} ({manifest['interval_ms']}ms)")
    lines.append("")
    lines.append(f"- method: `{manifest['method']}`")
    lines.append(f"- n_regimes: `{manifest['n_regimes']}`")
    lines.append(f"- partitions: `{len(manifest['partitions'])}`")
    lines.append("")
    lines.append("## Partition Counts")
    lines.append("")
    lines.append("| date | rows | regime_counts |")
    lines.append("|---|---:|---|")
    for p in manifest["partitions"]:
        lines.append(f"| {p['date']} | {p['rows']} | `{json.dumps(p['regime_counts'], sort_keys=True)}` |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    try:
        manifest = build_regimes(
            physics_root=Path(str(args.physics)),
            out_root=Path(str(args.out)),
            symbol=str(args.symbol),
            interval_ms=int(args.interval_ms),
            method=str(args.method),
            n_regimes=int(args.n_regimes),
            rolling=int(args.rolling),
            seed=int(args.seed),
        )
        report = Path(str(args.report)) if args.report else Path(f"reports/regimes_{canonical_symbol(args.symbol)}_{int(args.interval_ms)}ms.md")
        _write_report(manifest, report)
        rows = sum(int(x["rows"]) for x in manifest["partitions"])
        print(f"build_regimes ok symbol={manifest['symbol']} partitions={len(manifest['partitions'])} rows={rows} report={report}")
        return 0
    except Exception as e:
        print(f"build_regimes error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
