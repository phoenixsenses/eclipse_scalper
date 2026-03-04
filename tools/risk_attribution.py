from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.microphys.risk.attribution import build_risk_attribution


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build risk attribution report from live paper-trade outputs.")
    p.add_argument("--live-root", default="data/live")
    p.add_argument("--out-md", default="reports/risk_attribution.md")
    p.add_argument("--out-dir", default="data/derived/risk_attribution")
    return p.parse_args()


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _render_table(df: pd.DataFrame, *, group_label: str = "group") -> list[str]:
    if df.empty:
        return ["_empty_"]
    lines = [
        f"| {group_label} | count | net_sum | net_mean | win_rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['group']} | {int(r['count'])} | {float(r['net_sum']):.6f} | {float(r['net_mean']):.6f} | {float(r['win_rate']):.4f} |"
        )
    return lines


def main() -> int:
    args = _parse_args()
    live_root = Path(str(args.live_root))
    out_md = Path(str(args.out_md))
    out_dir = Path(str(args.out_dir))
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = _read_parquet(live_root / "papertrades_live.parquet")
    gating = _read_parquet(live_root / "gating_live.parquet")
    attr = build_risk_attribution(trades, gating_df=gating)

    for name, df in attr.items():
        p = out_dir / f"{name}.parquet"
        df.to_parquet(p, index=False)

    total = 0.0
    if not trades.empty:
        total = float(pd.to_numeric(trades.get("pnl_net_notional"), errors="coerce").fillna(0.0).sum())
        if abs(total) <= 0:
            total = float(pd.to_numeric(trades.get("pnl_net"), errors="coerce").fillna(0.0).sum())
    lines = [
        "# Risk Attribution",
        "",
        f"- live_root: `{live_root}`",
        f"- trades_rows: `{len(trades)}`",
        f"- gating_rows: `{len(gating)}`",
        f"- total_net: `{total:.6f}`",
        "",
        "## By Side",
        "",
        *_render_table(attr["by_side"]),
        "",
        "## By Fill State",
        "",
        *_render_table(attr["by_fill"]),
        "",
        "## By Risk Reason",
        "",
        *_render_table(attr["by_reason"]),
        "",
        "## By Active Expert",
        "",
        *_render_table(attr["by_expert"]),
    ]
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"risk_attribution ok out={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

