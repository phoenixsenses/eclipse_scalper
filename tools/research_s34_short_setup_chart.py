from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "microstructure.db"
OUT = ROOT / "reports" / "research" / "s34" / "S34_ETH_SHORT_SETUP_2026-06-06.png"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.execute("pragma query_only=on")
    latest_ms = int(con.execute("select max(ts_ms) from mark_prices").fetchone()[0])
    start_ms = latest_ms - 24 * 3600 * 1000
    rows = con.execute(
        """
        select cast(ts_ms / 300000 as integer) as bucket, max(ts_ms), mark_price
        from mark_prices
        where symbol='ETHUSDT' and ts_ms between ? and ?
        group by bucket
        order by bucket
        """,
        (start_ms, latest_ms),
    ).fetchall()
    con.close()

    xs = [datetime.fromtimestamp(int(r[1]) / 1000, tz=timezone.utc) for r in rows]
    ys = [float(r[2]) for r in rows]
    last = ys[-1]

    entry_low, entry_high = 1560.0, 1566.0
    stop = 1574.0
    tp1 = 1535.0
    tp2 = 1505.0

    import matplotlib.pyplot as plt  # type: ignore

    fig, ax = plt.subplots(figsize=(15, 8.5), facecolor="#f7f7f2")
    ax.set_facecolor("#f7f7f2")
    ax.plot(xs, ys, color="#2ca02c", linewidth=2.2, label="ETHUSDT mark price")

    ax.axhspan(entry_low, entry_high, color="#ffd166", alpha=0.45, label="SHORT ZONE 1560-1566")
    ax.axhline(stop, color="#d62828", linewidth=2.2, linestyle="--", label="STOP / invalidation 1574")
    ax.axhline(tp1, color="#1d3557", linewidth=2.0, linestyle="--", label="TP1 1535")
    ax.axhline(tp2, color="#457b9d", linewidth=2.0, linestyle="--", label="TP2 1505")
    ax.axhline(last, color="#111111", linewidth=1.2, alpha=0.7)

    ax.annotate(
        "Wait for rejection here\nSHORT only if BTC/SOL also fail reclaim",
        xy=(xs[-1], entry_high),
        xytext=(xs[max(5, len(xs) - 90)], entry_high + 18),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#8a1f11"),
        fontsize=11,
        color="#8a1f11",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#8a1f11"),
    )
    ax.annotate(
        "No S34 execution while liquidation feed is missing",
        xy=(xs[-1], last),
        xytext=(xs[max(5, len(xs) - 120)], tp2 - 14),
        arrowprops=dict(arrowstyle="->", lw=1.6, color="#d62828"),
        fontsize=11,
        color="#d62828",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#d62828"),
    )

    ax.set_title("S34 ETH Short Setup Map - Entry Zone / Stop / Targets", fontsize=18, weight="bold", pad=18)
    ax.set_ylabel("ETHUSDT mark price")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    ax.text(
        0.02,
        0.035,
        "Plan: ETH short only on 1560-1566 rejection. Stop 1574. TP1 1535, TP2 1505.\n"
        "Gate: liquidation confirmation is required. Without it this is watchlist, not S34 entry.",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#ffffff", edgecolor="#333333", alpha=0.92),
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT, dpi=170)
    plt.close(fig)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
