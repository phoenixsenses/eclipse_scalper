from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "microstructure.db"
OUT = ROOT / "reports" / "research" / "s34" / "S34_24H_PRICE_CHART_2026-06-06.png"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.execute("pragma query_only=on")
    latest_ms = int(con.execute("select max(ts_ms) from mark_prices").fetchone()[0])
    start_ms = latest_ms - 24 * 3600 * 1000

    rows = con.execute(
        """
        select symbol, cast(ts_ms / 300000 as integer) as bucket, max(ts_ms), mark_price
        from mark_prices
        where ts_ms between ? and ? and symbol in ('BTCUSDT','ETHUSDT','SOLUSDT')
        group by symbol, bucket
        order by bucket
        """,
        (start_ms, latest_ms),
    ).fetchall()
    con.close()

    data = {s: [] for s in SYMBOLS}
    for sym, bucket, _ts, price in rows:
        if sym in data:
            data[sym].append((int(bucket) * 300000, float(price)))

    import matplotlib.pyplot as plt  # type: ignore

    fig, ax = plt.subplots(figsize=(15, 8), facecolor="#f7f7f2")
    ax.set_facecolor("#f7f7f2")
    colors = {"BTCUSDT": "#1f77b4", "ETHUSDT": "#2ca02c", "SOLUSDT": "#9467bd"}

    for sym in SYMBOLS:
        series = data[sym]
        if not series:
            continue
        base = series[0][1]
        xs = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in series]
        ys = [(p / base - 1.0) * 100.0 for _, p in series]
        ax.plot(xs, ys, label=sym, linewidth=2.4, color=colors[sym])

    ax.axhline(0, color="#333333", linewidth=1)
    ax.axhspan(-4, -1.5, color="#f2b8a2", alpha=0.28, label="cascade-prone zone")
    ax.axhspan(0.5, 2.0, color="#b9e6c6", alpha=0.20, label="reclaim/squeeze risk")
    ax.set_title("S34 24H Price Chart - BTC / ETH / SOL", fontsize=18, weight="bold", pad=18)
    ax.set_ylabel("Normalized return from 24h start (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    note = (
        "Base read: cascade-prone, but NOT a trade signal.\n"
        "S34 execution gate remains closed until liquidation transport is restored."
    )
    ax.text(
        0.02,
        0.04,
        note,
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#8a1f11", alpha=0.9),
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT, dpi=170)
    plt.close(fig)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
