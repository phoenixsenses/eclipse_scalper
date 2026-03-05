from __future__ import annotations

import argparse
import math
import sqlite3
import time
from datetime import datetime, timezone
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


def _corr(x: list[float], y: list[float]) -> float:
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    xs = x[:n]
    ys = y[:n]
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    denx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    deny = math.sqrt(sum((b - my) ** 2 for b in ys))
    if denx <= 0 or deny <= 0:
        return 0.0
    return float(num / (denx * deny))


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature distribution analysis from microstructure DB.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--trades-db", default="data/paper_trades.db")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--lookback-hours", type=int, default=24)
    p.add_argument("--out", default="reports/FEATURE_STATIONARITY.md")
    p.add_argument("--tod-out", default="reports/TIME_OF_DAY_ANALYSIS.md")
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


def _load_feature_rows(db: Path, symbol: str, start_ms: int) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT CAST(ts_ms/1000 AS INTEGER) ts_sec, "
            "SUM(CASE WHEN is_buyer_maker=0 THEN quantity ELSE 0 END) buy_qty, "
            "SUM(CASE WHEN is_buyer_maker=1 THEN quantity ELSE 0 END) sell_qty, "
            "COUNT(*) n "
            "FROM agg_trades WHERE symbol=? AND ts_ms>=? "
            "GROUP BY CAST(ts_ms/1000 AS INTEGER) ORDER BY ts_sec",
            (symbol, start_ms),
        ).fetchall()
    finally:
        conn.close()
    return rows


def _time_of_day_stats(ts_sec: list[int], flags: list[bool]) -> dict[int, float]:
    totals = {h: 0 for h in range(24)}
    hits = {h: 0 for h in range(24)}
    for ts, flg in zip(ts_sec, flags):
        h = datetime.fromtimestamp(float(ts), tz=timezone.utc).hour
        totals[h] += 1
        if flg:
            hits[h] += 1
    out: dict[int, float] = {}
    for h in range(24):
        out[h] = (float(hits[h]) / float(totals[h])) if totals[h] > 0 else 0.0
    return out


def _entry_feature_outcome_corr(trades_db: Path, symbol: str, imbs: list[float], ints: list[float]) -> dict[str, float]:
    if not trades_db.exists():
        return {"imb_vs_pnl": 0.0, "int_vs_pnl": 0.0, "n": 0.0}
    conn = sqlite3.connect(str(trades_db), check_same_thread=False)
    try:
        # best-effort: use latest N completed trades for symbol
        rows = conn.execute(
            "SELECT pnl_bps FROM trades WHERE symbol=? ORDER BY exit_time DESC LIMIT 500",
            (symbol,),
        ).fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()
    pnls = [float(r[0] or 0.0) for r in rows]
    n = min(len(pnls), len(imbs), len(ints))
    if n < 2:
        return {"imb_vs_pnl": 0.0, "int_vs_pnl": 0.0, "n": float(n)}
    return {
        "imb_vs_pnl": _corr(imbs[-n:], pnls[:n]),
        "int_vs_pnl": _corr(ints[-n:], pnls[:n]),
        "n": float(n),
    }


def main() -> int:
    args = _args()
    db = Path(args.db)
    if not db.exists():
        print(f"feature_distribution_analysis: missing db {db}")
        return 2
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(args.lookback_hours) * 3600 * 1000
    rows = _load_feature_rows(db, str(args.symbol), start_ms)
    if not rows:
        print("feature_distribution_analysis: no rows")
        return 1

    ts_sec: list[int] = []
    imbs: list[float] = []
    ints: list[float] = []
    pocket_like: list[bool] = []
    for r in rows:
        buy = float(r["buy_qty"] or 0.0)
        sell = float(r["sell_qty"] or 0.0)
        tot = buy + sell
        imb = ((buy - sell) / tot) if tot > 0 else 0.0
        intensity = float(r["n"] or 0.0) * 60.0  # per-minute equivalent
        tsv = int(r["ts_sec"] or 0)
        ts_sec.append(tsv)
        imbs.append(imb)
        ints.append(intensity)
        pocket_like.append(abs(imb) >= 0.5 and intensity >= 2500.0)

    plots_dir = Path(args.plots_dir)
    imb_png = plots_dir / f"feature_dist_{args.symbol}_imbalance.png"
    int_png = plots_dir / f"feature_dist_{args.symbol}_trade_intensity.png"
    ok_imb_plot = _write_hist(imbs, title=f"{args.symbol} imbalance_signed", out_path=imb_png)
    ok_int_plot = _write_hist(ints, title=f"{args.symbol} trade_intensity_per_min", out_path=int_png)

    # Stationarity summary (split halves drift)
    m = len(imbs)
    half = max(1, m // 2)
    imb_drift = (sum(imbs[half:]) / max(1, len(imbs[half:]))) - (sum(imbs[:half]) / max(1, len(imbs[:half])))
    int_drift = (sum(ints[half:]) / max(1, len(ints[half:]))) - (sum(ints[:half]) / max(1, len(ints[:half])))
    corr_stats = _entry_feature_outcome_corr(Path(args.trades_db), str(args.symbol), imbs, ints)

    stationarity_md = "\n".join(
        [
            "# Feature Stationarity",
            "",
            "_Feature Distribution Analysis_",
            "",
            f"- symbol: {args.symbol}",
            f"- lookback_hours: {args.lookback_hours}",
            f"- buckets(sec): {len(rows)}",
            "",
            "## Imbalance",
            f"- q05: {_q(imbs, 0.05):.4f}",
            f"- q50: {_q(imbs, 0.50):.4f}",
            f"- q95: {_q(imbs, 0.95):.4f}",
            f"- drift_half_window: {imb_drift:+.5f}",
            f"- frac_abs_gt_0.50: {sum(1 for x in imbs if abs(x)>=0.5)/max(1,len(imbs))*100.0:.2f}%",
            "",
            "## Trade Intensity (per-minute equivalent)",
            f"- q05: {_q(ints, 0.05):.2f}",
            f"- q50: {_q(ints, 0.50):.2f}",
            f"- q95: {_q(ints, 0.95):.2f}",
            f"- drift_half_window: {int_drift:+.2f}",
            f"- frac_ge_2500: {sum(1 for x in ints if x>=2500)/max(1,len(ints))*100.0:.2f}%",
            "",
            "## Feature vs PnL Correlation (paper trades)",
            f"- n_used: {int(corr_stats.get('n', 0.0))}",
            f"- corr(imbalance, pnl_bps): {corr_stats.get('imb_vs_pnl', 0.0):+.4f}",
            f"- corr(intensity, pnl_bps): {corr_stats.get('int_vs_pnl', 0.0):+.4f}",
            "",
            "## Plots",
            f"- imbalance_hist: {imb_png if ok_imb_plot else 'unavailable'}",
            f"- intensity_hist: {int_png if ok_int_plot else 'unavailable'}",
            "",
        ]
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(stationarity_md, encoding="utf-8")

    tod = _time_of_day_stats(ts_sec, pocket_like)
    tod_lines = [
        "# Time-of-Day Analysis",
        "",
        f"- symbol: {args.symbol}",
        f"- lookback_hours: {args.lookback_hours}",
        "",
        "## Pocket-like Opportunity Rate by UTC Hour",
    ]
    for h in range(24):
        tod_lines.append(f"- {h:02d}:00 -> {tod.get(h, 0.0)*100.0:.2f}%")
    Path(args.tod_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.tod_out).write_text("\n".join(tod_lines) + "\n", encoding="utf-8")

    print(f"feature_distribution_analysis: wrote {out}")
    print(f"feature_distribution_analysis: wrote {args.tod_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
