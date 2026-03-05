from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from src.microphys.alpha.calibration import load_calibration
from src.microphys.alpha.overlap import dedupe_specs, pairwise_overlap
from src.microphys.alpha.spec import SignalSpec, signal_from_dict, specs_to_jsonl
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deduplicate high-overlap alpha candidates.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--candidates", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--out", required=True)
    p.add_argument("--jaccard-thr", type=float, default=0.90)
    p.add_argument("--target-triggers-per-day", type=float, default=200.0)
    p.add_argument("--report", default="")
    return p.parse_args()


def _load_specs(path: Path) -> List[SignalSpec]:
    out: List[SignalSpec] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(signal_from_dict(json.loads(s)))
    return out


def _load_physics(root: Path, symbol: str, interval_ms: int) -> pd.DataFrame:
    base = root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    files = sorted(base.glob("date=*/physics.parquet"))
    if not files:
        return pd.DataFrame()
    chosen = files[-14:]
    return pd.concat([pd.read_parquet(p) for p in chosen], ignore_index=True).sort_values("ts_ms").reset_index(drop=True)


def main() -> int:
    args = _parse_args()
    try:
        symbol = canonical_symbol(args.symbol)
        cand_path = Path(str(args.candidates))
        specs = _load_specs(cand_path)
        physics = _load_physics(Path(str(args.physics)), symbol, int(args.interval_ms))
        if physics.empty:
            raise RuntimeError("physics_missing_for_dedup")
        cal_path = cand_path.parent / "calibration.json"
        calibration = load_calibration(cal_path) if cal_path.exists() else None
        pairs = pairwise_overlap(physics, specs, calibration=calibration)
        res = dedupe_specs(
            specs,
            pairs,
            jaccard_thr=float(args.jaccard_thr),
            target_triggers_per_day=float(args.target_triggers_per_day),
        )

        out_path = Path(str(args.out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(specs_to_jsonl(res.selected), encoding="utf-8")
        manifest_path = out_path.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(
                {
                    "symbol": symbol,
                    "interval_ms": int(args.interval_ms),
                    "input_count": len(specs),
                    "selected_count": len(res.selected),
                    "dropped_count": len(res.dropped),
                    "clusters": len(res.clusters),
                    "jaccard_thr": float(args.jaccard_thr),
                },
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        report = Path(str(args.report)) if str(args.report).strip() else Path(
            f"reports/dedup_summary_{symbol}_{int(args.interval_ms)}ms.md"
        )
        lines = [
            f"# Dedup Summary - {symbol} ({int(args.interval_ms)}ms)",
            "",
            f"- input candidates: `{len(specs)}`",
            f"- selected candidates: `{len(res.selected)}`",
            f"- dropped candidates: `{len(res.dropped)}`",
            f"- clusters: `{len(res.clusters)}`",
            f"- jaccard threshold: `{float(args.jaccard_thr):.2f}`",
            "",
            "## Largest clusters",
            "",
            "| size | representative | members |",
            "|---:|---|---|",
        ]
        sel_names = {s.name for s in res.selected}
        for cluster in res.clusters[:30]:
            rep = sorted([x for x in cluster if x in sel_names])[0] if any(x in sel_names for x in cluster) else cluster[0]
            lines.append(f"| {len(cluster)} | {rep} | `{','.join(cluster[:8])}` |")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"dedup_candidates ok out={out_path} selected={len(res.selected)} dropped={len(res.dropped)}")
        return 0
    except Exception as e:
        print(f"dedup_candidates error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
