from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect physics columns and distribution stats.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--out", default="")
    p.add_argument("--out-parquet", default="")
    return p.parse_args()


def _load_recent(root: Path, symbol: str, interval_ms: int, days: int) -> pd.DataFrame:
    base = root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    parts = sorted(base.glob("date=*/physics.parquet"))
    if not parts:
        return pd.DataFrame()
    take = max(1, int(days))
    files = parts[-take:]
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def main() -> int:
    args = _parse_args()
    try:
        sym = canonical_symbol(args.symbol)
        df = _load_recent(Path(str(args.physics)), sym, int(args.interval_ms), int(args.days))
        if df.empty:
            raise RuntimeError("physics_missing")
        rows = []
        for c in sorted(df.columns):
            s = pd.to_numeric(df[c], errors="coerce").astype(float)
            non_na = int(s.notna().sum())
            n = int(len(df))
            nan_pct = float((1.0 - (non_na / max(1, n))) * 100.0)
            if non_na > 0:
                q = s.quantile([0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]).to_dict()
                rows.append(
                    {
                        "column": c,
                        "dtype": str(df[c].dtype),
                        "nan_pct": nan_pct,
                        "min": float(s.min()),
                        "p01": float(q.get(0.01, 0.0)),
                        "p05": float(q.get(0.05, 0.0)),
                        "p10": float(q.get(0.10, 0.0)),
                        "p50": float(q.get(0.50, 0.0)),
                        "p90": float(q.get(0.90, 0.0)),
                        "p95": float(q.get(0.95, 0.0)),
                        "p99": float(q.get(0.99, 0.0)),
                        "max": float(s.max()),
                    }
                )
            else:
                rows.append(
                    {
                        "column": c,
                        "dtype": str(df[c].dtype),
                        "nan_pct": 100.0,
                        "min": 0.0,
                        "p01": 0.0,
                        "p05": 0.0,
                        "p10": 0.0,
                        "p50": 0.0,
                        "p90": 0.0,
                        "p95": 0.0,
                        "p99": 0.0,
                        "max": 0.0,
                    }
                )
        out_df = pd.DataFrame(rows).sort_values("column").reset_index(drop=True)
        out_parquet = (
            Path(str(args.out_parquet))
            if str(args.out_parquet).strip()
            else Path(f"data/derived/alpha_diag/interval_ms={int(args.interval_ms)}/symbol={sym}/column_inventory.parquet")
        )
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out_parquet, index=False)
        out_md = Path(str(args.out)) if str(args.out).strip() else Path(f"reports/physics_column_inventory_{sym}_{int(args.interval_ms)}ms.md")
        lines = [
            f"# Physics Column Inventory - {sym} ({int(args.interval_ms)}ms)",
            "",
            f"- rows: `{len(df)}` (last `{int(args.days)}` partition days)",
            "",
            "| column | dtype | nan_pct | min | p01 | p05 | p10 | p50 | p90 | p95 | p99 | max |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in out_df.iterrows():
            lines.append(
                f"| {r['column']} | {r['dtype']} | {float(r['nan_pct']):.2f} | {float(r['min']):.8f} | {float(r['p01']):.8f} | "
                f"{float(r['p05']):.8f} | {float(r['p10']):.8f} | {float(r['p50']):.8f} | {float(r['p90']):.8f} | {float(r['p95']):.8f} | {float(r['p99']):.8f} | {float(r['max']):.8f} |"
            )
        lines.append("")
        lines.append(f"- parquet: `{out_parquet}`")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"inspect_physics_columns ok out={out_md} parquet={out_parquet}")
        return 0
    except Exception as e:
        print(f"inspect_physics_columns error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
