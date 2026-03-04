from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path


def _q(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    xs = sorted(vals)
    pos = (len(xs) - 1) * max(0.0, min(1.0, float(q)))
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature distribution analysis from microstructure DB.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--lookback-hours", type=int, default=24)
    p.add_argument("--out", default="reports/FEATURE_STATIONARITY.md")
    p.add_argument("--plots-dir", default="reports/plots")
    return p.parse_args()


def _write_hist(values: list[float], *, title: str, out_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7, 3))
        plt.hist(values, bins=60)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        return True
    except Exception:
        return False


def main() -> int:
    args = _args()
    db = Path(args.db)
    if not db.exists():
        print(f"feature_distribution_analysis: missing db {db}")
        return 2
    conn = sqlite3.connect(str(db), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - int(args.lookback_hours) * 3600 * 1000
        rows = conn.execute(
            "SELECT CAST(ts_ms/1000 AS INTEGER) ts_sec, "
            "SUM(CASE WHEN is_buyer_maker=0 THEN quantity ELSE 0 END) buy_qty, "
            "SUM(CASE WHEN is_buyer_maker=1 THEN quantity ELSE 0 END) sell_qty, "
            "COUNT(*) n "
            "FROM agg_trades WHERE symbol=? AND ts_ms>=? "
            "GROUP BY CAST(ts_ms/1000 AS INTEGER) ORDER BY ts_sec",
            (args.symbol, start_ms),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        print("feature_distribution_analysis: no rows")
        return 1
    imbs: list[float] = []
    ints: list[float] = []
    for r in rows:
        buy = float(r["buy_qty"] or 0.0)
        sell = float(r["sell_qty"] or 0.0)
        tot = buy + sell
        imbs.append(((buy - sell) / tot) if tot > 0 else 0.0)
        ints.append(float(r["n"] or 0.0) * 60.0)  # per-minute equivalent
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(args.plots_dir)
    imb_png = plots_dir / f"feature_dist_{args.symbol}_imbalance.png"
    int_png = plots_dir / f"feature_dist_{args.symbol}_trade_intensity.png"
    ok_imb_plot = _write_hist(imbs, title=f"{args.symbol} imbalance_signed", out_path=imb_png)
    ok_int_plot = _write_hist(ints, title=f"{args.symbol} trade_intensity_per_min", out_path=int_png)
    md = "\n".join(
        [
            "# Feature Distribution Analysis",
            "",
            f"- symbol: {args.symbol}",
            f"- lookback_hours: {args.lookback_hours}",
            f"- buckets(sec): {len(rows)}",
            "",
            "## Imbalance",
            f"- q05: {_q(imbs, 0.05):.4f}",
            f"- q50: {_q(imbs, 0.50):.4f}",
            f"- q95: {_q(imbs, 0.95):.4f}",
            "",
            "## Trade Intensity (per-minute equivalent)",
            f"- q05: {_q(ints, 0.05):.2f}",
            f"- q50: {_q(ints, 0.50):.2f}",
            f"- q95: {_q(ints, 0.95):.2f}",
            "",
            "## Plots",
            f"- imbalance_hist: {imb_png if ok_imb_plot else 'unavailable'}",
            f"- intensity_hist: {int_png if ok_int_plot else 'unavailable'}",
            "",
        ]
    )
    out.write_text(md, encoding="utf-8")
    print(f"feature_distribution_analysis: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
