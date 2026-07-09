from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.storage import production as PR
from ami.storage import research_reader as RR

DB = ROOT / "data" / "microstructure.db"
OUT = ROOT / "reports" / "research" / "s34" / "S34_PREDICTION_MAP_2026-06-06.png"
MD = ROOT / "reports" / "research" / "s34" / "S34_PREDICTION_MAP_2026-06-06.md"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


@dataclass
class Snap:
    symbol: str
    price: float
    ret_1h: float
    ret_6h: float
    ret_24h: float
    last_mark_ms: int
    last_agg_ms: Optional[int]


def iso(ms: Optional[int]) -> str:
    if not ms:
        return "-"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def pct(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None or a == 0:
        return 0.0
    return (b / a - 1.0) * 100.0


def nearest_price(con: sqlite3.Connection, symbol: str, target_ms: int) -> Optional[float]:
    """Direct-SQL oracle -- kept as the parity reference for
    `nearest_price_v2` (BATCH-STORAGE-ROTATION-RETENTION-ASOF-LOOKUP-
    CONSUMER-MIGRATION-V4). No longer called by load(); the reader-backed
    path is used instead."""
    row = con.execute(
        """
        select mark_price from mark_prices
        where symbol=? and ts_ms<=?
        order by ts_ms desc limit 1
        """,
        (symbol, target_ms),
    ).fetchone()
    return float(row[0]) if row else None


def nearest_price_v2(root, symbol: str, target_ms: int, source_db_path=None) -> Optional[float]:
    """Reader-backed replacement for `nearest_price` (latest mark_price
    at-or-before `target_ms`), via lookup_latest_at_or_before. ETHUSDT
    resolves through the real mark_prices/ETHUSDT/2026-05 archive for
    historical timestamps; BTCUSDT/SOLUSDT have no mark_prices archive."""
    result = RR.lookup_latest_at_or_before(
        root, table="mark_prices", symbol=symbol, ts_ms=target_ms, columns=("mark_price",),
        source_db_path=source_db_path)
    return float(result.row[0]) if result.found else None


def load() -> Dict[str, Snap]:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.execute("pragma query_only=on")
    root, _root_source = PR.resolve_production_root()
    latest = int(con.execute("select max(ts_ms) from mark_prices").fetchone()[0])
    out: Dict[str, Snap] = {}
    for sym in SYMBOLS:
        # OUT OF SCOPE for this ASOF gate: "latest row overall for symbol"
        # (no `ts_ms <= ?` bound) is a different query shape than the
        # helper's at-or-before contract; and the agg_trades latest-ts
        # lookup has no safely pre-computed bound (agg max ts != mark max
        # ts). Both left on direct SQL, per the partial-migration policy.
        row = con.execute(
            "select ts_ms, mark_price from mark_prices where symbol=? order by ts_ms desc limit 1",
            (sym,),
        ).fetchone()
        last_agg = con.execute(
            "select ts_ms from agg_trades where symbol=? order by ts_ms desc limit 1",
            (sym,),
        ).fetchone()
        price = float(row[1])
        out[sym] = Snap(
            symbol=sym,
            price=price,
            ret_1h=pct(nearest_price_v2(root, sym, latest - 3600_000, source_db_path=str(DB)), price),
            ret_6h=pct(nearest_price_v2(root, sym, latest - 6 * 3600_000, source_db_path=str(DB)), price),
            ret_24h=pct(nearest_price_v2(root, sym, latest - 24 * 3600_000, source_db_path=str(DB)), price),
            last_mark_ms=int(row[0]),
            last_agg_ms=int(last_agg[0]) if last_agg else None,
        )
    con.close()
    return out


def bias(snaps: Dict[str, Snap]) -> str:
    btc = snaps["BTCUSDT"]
    eth = snaps["ETHUSDT"]
    sol = snaps["SOLUSDT"]
    risk_off_votes = 0
    if btc.ret_6h < -0.25 and btc.ret_24h < 1.0:
        risk_off_votes += 1
    if eth.ret_24h < 0:
        risk_off_votes += 1
    if sol.ret_24h < -2.0:
        risk_off_votes += 1
    if risk_off_votes >= 2:
        return "BASE CASE: downside continuation / cascade-prone"
    if btc.ret_1h > 0.25 and eth.ret_1h > 0.25:
        return "RECLAIM CASE: short squeeze risk, avoid fresh S34 shorts"
    return "NEUTRAL CASE: wait for liquidation confirmation"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    snaps = load()
    current_bias = bias(snaps)

    import matplotlib.pyplot as plt  # type: ignore

    fig = plt.figure(figsize=(15, 9), facecolor="#f6f7f3")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.04, 0.94, "S34 Prediction Map - BTC / ETH / SOL", fontsize=21, weight="bold")
    ax.text(0.04, 0.90, current_bias, fontsize=15, color="#8a1f11", weight="bold")
    ax.text(0.04, 0.865, "Not a trade signal. S34 remains blocked until liquidation transport is restored.", fontsize=11)

    xs = [0.06, 0.37, 0.68]
    colors = {"BTCUSDT": "#1f77b4", "ETHUSDT": "#2ca02c", "SOLUSDT": "#9467bd"}
    for x, sym in zip(xs, SYMBOLS):
        s = snaps[sym]
        c = colors[sym]
        ax.add_patch(plt.Rectangle((x, 0.58), 0.25, 0.24, fill=False, linewidth=2.2, edgecolor=c))
        ax.text(x + 0.015, 0.785, sym, fontsize=16, color=c, weight="bold")
        ax.text(x + 0.015, 0.745, f"Mark: {s.price:,.4f}", fontsize=11)
        ax.text(x + 0.015, 0.710, f"1h: {s.ret_1h:+.2f}%   6h: {s.ret_6h:+.2f}%", fontsize=10)
        ax.text(x + 0.015, 0.675, f"24h: {s.ret_24h:+.2f}%", fontsize=10)
        ax.text(x + 0.015, 0.640, f"Last mark: {iso(s.last_mark_ms)}", fontsize=8.5)
        ax.text(x + 0.015, 0.610, f"Last agg: {iso(s.last_agg_ms)}", fontsize=8.5)

    ax.annotate("", xy=(0.37, 0.70), xytext=(0.31, 0.70), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.68, 0.70), xytext=(0.62, 0.70), arrowprops=dict(arrowstyle="->", lw=2))

    ax.text(0.06, 0.50, "Scenario tree", fontsize=16, weight="bold")
    scenarios = [
        ("Bear / cascade-prone", "BTC fails to reclaim, ETH/SOL stay weaker, funding/liquidations remain stressed", "S34 wants liq confirmation; without it, only watchlist."),
        ("Base / chop", "BTC stabilizes but ETH/SOL lag; correlations high, no forced-flow proof", "No entry; preserve data collection."),
        ("Reclaim / squeeze", "BTC + ETH reclaim together; SOL stops bleeding", "Avoid short-bias S34; wait for new cascade setup."),
    ]
    y = 0.455
    for title, condition, action in scenarios:
        ax.add_patch(plt.Rectangle((0.07, y - 0.055), 0.86, 0.075, fill=False, linewidth=1.4, edgecolor="#555555"))
        ax.text(0.09, y, title, fontsize=12, weight="bold")
        ax.text(0.29, y, condition, fontsize=10)
        ax.text(0.29, y - 0.028, action, fontsize=10, color="#8a1f11")
        y -= 0.105

    ax.text(0.06, 0.16, "Execution gate", fontsize=15, weight="bold")
    gates = [
        "GREEN data collection is not enough for S34.",
        "Required before S34 signal trust: current liquidation feed OR validated liquidation substitute.",
        "BTC is fuel/regime; ETH is primary cascade target; SOL is confirmation/high-beta stress gauge.",
    ]
    for i, g in enumerate(gates):
        ax.text(0.08, 0.125 - i * 0.032, f"- {g}", fontsize=10.5)

    fig.savefig(OUT, dpi=160)
    plt.close(fig)

    lines = [
        "# S34 Prediction Map - 2026-06-06",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Bias: **{current_bias}**",
        "",
        "This is a scenario map, not a trade signal. The S34 execution gate remains closed until liquidation transport is restored.",
        "",
        "| Symbol | Mark | 1h | 6h | 24h | Last mark | Last agg |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for s in snaps.values():
        lines.append(
            f"| {s.symbol} | {s.price:,.4f} | {s.ret_1h:+.2f}% | {s.ret_6h:+.2f}% | {s.ret_24h:+.2f}% | {iso(s.last_mark_ms)} | {iso(s.last_agg_ms)} |"
        )
    lines += [
        "",
        f"Image: `{OUT.relative_to(ROOT)}`",
    ]
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)
    print(MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
