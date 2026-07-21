from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "microstructure.db"
OUT = ROOT / "reports" / "research" / "s34" / "S34_TRADE_EXPLANATION_2026-06-06.png"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.execute("pragma query_only=on")
    latest_ms = int(con.execute("select max(ts_ms) from mark_prices").fetchone()[0])
    start_ms = latest_ms - 18 * 3600 * 1000
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

    # Values visible in the user's TradingView screenshot.
    user_entry = 1560.44
    user_stop = 1563.80
    user_tp = 1549.11

    # Wider S34 plan from the local frame.
    s34_entry_low = 1560.0
    s34_entry_high = 1566.0
    s34_stop = 1574.0
    s34_tp1 = 1535.0
    s34_tp2 = 1505.0

    import matplotlib.pyplot as plt  # type: ignore

    fig, ax = plt.subplots(figsize=(16, 9), facecolor="#f7f7f2")
    ax.set_facecolor("#f7f7f2")
    ax.plot(xs, ys, color="#2ca02c", linewidth=2.2, label="ETHUSDT mark price")

    # S34 structure.
    ax.axhspan(s34_entry_low, s34_entry_high, color="#ffd166", alpha=0.35, label="S34 short zone 1560-1566")
    ax.axhline(s34_stop, color="#d62828", linewidth=2.0, linestyle="--", label="S34 stop 1574")
    ax.axhline(s34_tp1, color="#1d3557", linewidth=2.0, linestyle="--", label="S34 TP1 1535")
    ax.axhline(s34_tp2, color="#457b9d", linewidth=2.0, linestyle="--", label="S34 TP2 1505")

    # User scalp structure.
    ax.axhline(user_entry, color="#111111", linewidth=2.0, label="Your entry approx 1560.44")
    ax.axhline(user_stop, color="#ff006e", linewidth=2.3, linestyle="-.", label="Your tight stop 1563.80")
    ax.axhline(user_tp, color="#00a896", linewidth=2.3, linestyle="-.", label="Your TP 1549.11")

    ax.annotate(
        "Your paper short is directionally aligned\nwith the S34 short-bias zone.",
        xy=(xs[-1], user_entry),
        xytext=(xs[max(4, len(xs) - 120)], user_entry + 24),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#111111"),
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#111111"),
    )
    ax.annotate(
        "Main mismatch:\nyour stop is inside the S34 noise zone.\nIt can get wicked out before the setup fails.",
        xy=(xs[-1], user_stop),
        xytext=(xs[max(4, len(xs) - 88)], user_stop + 15),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#ff006e"),
        fontsize=11,
        color="#8a1f11",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#ff006e"),
    )
    ax.annotate(
        "S34 invalidation is higher: 1574.\nAbove this, short idea is wrong.",
        xy=(xs[-1], s34_stop),
        xytext=(xs[max(4, len(xs) - 160)], s34_stop + 14),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#d62828"),
        fontsize=11,
        color="#8a1f11",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#d62828"),
    )
    ax.annotate(
        "Your TP is a scalp target.\nS34 targets are deeper: 1535 then 1505.",
        xy=(xs[-1], user_tp),
        xytext=(xs[max(4, len(xs) - 115)], user_tp - 20),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#00a896"),
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#00a896"),
    )

    ax.set_title("Your ETH Paper Short vs S34 Setup - What Is Right / What Is Different", fontsize=18, weight="bold", pad=18)
    ax.set_ylabel("ETHUSDT mark price")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(
        0.02,
        0.035,
        "Verdict: direction is correct for short-bias watchlist. Structure is a tight scalp, not full S34.\n"
        "Full S34 still requires liquidation confirmation; current liquidation feed is unavailable.",
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
