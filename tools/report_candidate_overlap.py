from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from src.microphys.alpha.calibration import load_calibration
from src.microphys.alpha.overlap import pairwise_overlap
from src.microphys.alpha.spec import SignalSpec, signal_from_dict
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pairwise overlap diagnostics for alpha candidates.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--candidates", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--out", default="data/derived/alpha_diag")
    p.add_argument("--report", default="")
    p.add_argument("--top-pairs", type=int, default=20_000)
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
            raise RuntimeError("physics_missing_for_overlap")
        cal_path = cand_path.parent / "calibration.json"
        calibration = load_calibration(cal_path) if cal_path.exists() else None

        out_base = Path(str(args.out)) / f"interval_ms={int(args.interval_ms)}" / f"symbol={symbol}"
        out_base.mkdir(parents=True, exist_ok=True)
        pairs = pairwise_overlap(
            physics,
            specs,
            calibration=calibration,
            top_pairs=int(args.top_pairs),
        )
        out_pq = out_base / "overlap_pairs.parquet"
        pairs.to_parquet(out_pq, index=False)

        report = Path(str(args.report)) if str(args.report).strip() else Path(
            f"reports/dedup_summary_{symbol}_{int(args.interval_ms)}ms.md"
        )
        hi = pairs[pairs["jaccard"] >= 0.90] if not pairs.empty else pd.DataFrame()
        lines = [
            f"# Candidate Overlap - {symbol} ({int(args.interval_ms)}ms)",
            "",
            f"- candidates: `{len(specs)}`",
            f"- pair rows: `{len(pairs)}`",
            f"- high-overlap pairs (jaccard>=0.90): `{len(hi)}`",
            "",
            "## Top overlap pairs",
            "",
            "| a | b | jaccard | phi | intersect | union |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for _, r in pairs.head(50).iterrows():
            lines.append(
                f"| {r['a']} | {r['b']} | {float(r['jaccard']):.4f} | {float(r['phi']):.4f} | "
                f"{int(r['intersect'])} | {int(r['union'])} |"
            )
        lines.extend(["", f"- output parquet: `{out_pq}`"])
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"report_candidate_overlap ok out={out_pq} report={report}")
        return 0
    except Exception as e:
        print(f"report_candidate_overlap error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
