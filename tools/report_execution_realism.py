from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare simple vs realistic execution outputs.")
    p.add_argument("--simple", required=True, help="papertrades/eval trades from simple model")
    p.add_argument("--realistic", required=True, help="papertrades/eval trades from maker model")
    p.add_argument("--out", default="reports/execution_realism_comparison_ETHUSDT_100ms.md")
    return p.parse_args()


def _stats(df: pd.DataFrame, label: str) -> dict:
    d = df.copy()
    pnl = pd.to_numeric(d.get("pnl_net", d.get("net_ret")), errors="coerce")
    filled = d.get("filled")
    if filled is None:
        fill_rate = 1.0
    else:
        fill_rate = float(pd.to_numeric(filled, errors="coerce").fillna(0).mean())
    return {
        "label": label,
        "rows": int(len(d)),
        "fill_rate": float(fill_rate),
        "pnl_mean": float(pnl.mean()) if not pnl.dropna().empty else 0.0,
        "pnl_median": float(pnl.median()) if not pnl.dropna().empty else 0.0,
    }


def main() -> int:
    args = _parse_args()
    try:
        s = pd.read_parquet(Path(str(args.simple)))
        r = pd.read_parquet(Path(str(args.realistic)))
        ss = _stats(s, "simple")
        rr = _stats(r, "realistic")
        out = Path(str(args.out))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "\n".join(
                [
                    "# Execution Realism Comparison",
                    "",
                    "| model | rows | fill_rate | pnl_mean | pnl_median |",
                    "|---|---:|---:|---:|---:|",
                    f"| simple | {ss['rows']} | {ss['fill_rate']:.4f} | {ss['pnl_mean']:.8f} | {ss['pnl_median']:.8f} |",
                    f"| realistic | {rr['rows']} | {rr['fill_rate']:.4f} | {rr['pnl_mean']:.8f} | {rr['pnl_median']:.8f} |",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"report_execution_realism ok out={out}")
        return 0
    except Exception as e:
        print(f"report_execution_realism error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
