from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def iso(ms: Optional[int]) -> str:
    if not ms:
        return "-"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or a == 0:
        return None
    return (b / a - 1.0) * 100.0


def fmt(x: Optional[float], nd: int = 2) -> str:
    if x is None or not math.isfinite(float(x)):
        return "-"
    return f"{float(x):.{nd}f}"


def scalar(con: sqlite3.Connection, sql: str, params: Iterable = ()) -> Optional[float]:
    row = con.execute(sql, tuple(params)).fetchone()
    if not row:
        return None
    return row[0]


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    return bool(con.execute("select 1 from sqlite_master where type='table' and name=?", (table,)).fetchone())


@dataclass
class SymbolFrame:
    symbol: str
    last_price: Optional[float]
    ret_1h: Optional[float]
    ret_6h: Optional[float]
    ret_24h: Optional[float]
    agg_count_1h: int
    agg_notional_1h: float
    agg_signed_1h: float
    agg_count_24h: int
    agg_notional_24h: float
    agg_signed_24h: float
    mark_count_24h: int
    liq_count_30d: int
    last_liq_ts: Optional[int]
    last_agg_ts: Optional[int]


def nearest_price(con: sqlite3.Connection, symbol: str, target_ms: int, max_back_ms: int) -> Optional[float]:
    row = con.execute(
        """
        select mark_price from mark_prices
        where symbol=? and ts_ms between ? and ?
        order by ts_ms desc limit 1
        """,
        (symbol, target_ms - max_back_ms, target_ms),
    ).fetchone()
    return float(row[0]) if row else None


def flow_stats(con: sqlite3.Connection, symbol: str, start_ms: int) -> Tuple[int, float, float]:
    row = con.execute(
        """
        select
          count(*),
          coalesce(sum(notional), 0.0),
          coalesce(sum(case when is_buyer_maker=0 then notional else -notional end), 0.0)
        from agg_trades
        where symbol=? and ts_ms>=?
        """,
        (symbol, start_ms),
    ).fetchone()
    return int(row[0] or 0), float(row[1] or 0.0), float(row[2] or 0.0)


def load_minute_series(con: sqlite3.Connection, start_ms: int, end_ms: int) -> Dict[str, Dict[int, float]]:
    series: Dict[str, Dict[int, float]] = {s: {} for s in SYMBOLS}
    rows = con.execute(
        """
        select symbol, cast(ts_ms / 60000 as integer) as minute, mark_price, ts_ms
        from mark_prices
        where ts_ms between ? and ? and symbol in ('BTCUSDT','ETHUSDT','SOLUSDT')
        order by symbol, minute, ts_ms
        """,
        (start_ms, end_ms),
    )
    for sym, minute, price, _ts in rows:
        if sym in series:
            series[sym][int(minute)] = float(price)
    return series


def align_returns(series: Dict[int, float], minutes: List[int]) -> List[Optional[float]]:
    vals: List[Optional[float]] = []
    last: Optional[float] = None
    prev: Optional[float] = None
    for m in minutes:
        if m in series:
            last = series[m]
        if last is None or prev is None or prev == 0:
            vals.append(None)
        else:
            vals.append((last / prev) - 1.0)
        if last is not None:
            prev = last
    return vals


def corr(xs: List[Optional[float]], ys: List[Optional[float]]) -> Optional[float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 30:
        return None
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def lead_lag(series: Dict[str, Dict[int, float]], start_ms: int, end_ms: int) -> Dict[str, Dict[str, Optional[float]]]:
    minutes = list(range(start_ms // 60000, end_ms // 60000 + 1))
    returns = {sym: align_returns(series[sym], minutes) for sym in SYMBOLS}
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for leader, follower in [("BTCUSDT", "ETHUSDT"), ("BTCUSDT", "SOLUSDT"), ("ETHUSDT", "SOLUSDT")]:
        best_lag = None
        best_corr = None
        zero_corr = corr(returns[leader], returns[follower])
        for lag in range(1, 11):
            c = corr(returns[leader][:-lag], returns[follower][lag:])
            if c is None:
                continue
            if best_corr is None or abs(c) > abs(best_corr):
                best_corr = c
                best_lag = lag
        out[f"{leader}->{follower}"] = {"zero_corr": zero_corr, "best_lag_min": best_lag, "best_corr": best_corr}
    return out


def write_picture(path: Path, frames: List[SymbolFrame], lags: Dict[str, Dict[str, Optional[float]]], readiness: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        path.with_suffix(".txt").write_text("matplotlib unavailable; see markdown report.\n", encoding="utf-8")
        return

    fig = plt.figure(figsize=(14, 9), facecolor="#f7f7f2")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.04, 0.94, "S34 Frame Picture: BTC fuel -> ETH/SOL cascade -> execution gate", fontsize=18, weight="bold")
    ax.text(0.04, 0.90, f"Readiness: {readiness}", fontsize=12, color="#8a1f11", weight="bold")

    colors = ["#1f77b4", "#2ca02c", "#9467bd"]
    x0s = [0.05, 0.37, 0.69]
    for x0, frame, color in zip(x0s, frames, colors):
        ax.add_patch(plt.Rectangle((x0, 0.58), 0.26, 0.25, fill=False, linewidth=2, edgecolor=color))
        ax.text(x0 + 0.015, 0.79, frame.symbol, fontsize=15, weight="bold", color=color)
        lines = [
            f"Price: {fmt(frame.last_price, 4)}",
            f"1h / 6h / 24h: {fmt(frame.ret_1h)}% / {fmt(frame.ret_6h)}% / {fmt(frame.ret_24h)}%",
            f"24h agg notional: ${frame.agg_notional_24h/1e9:.2f}B",
            f"24h taker imbalance: ${frame.agg_signed_24h/1e6:.1f}M",
            f"Last agg: {iso(frame.last_agg_ts)}",
            f"30d local liq rows: {frame.liq_count_30d}",
            f"Last local liq: {iso(frame.last_liq_ts)}",
        ]
        for i, line in enumerate(lines):
            ax.text(x0 + 0.015, 0.755 - i * 0.03, line, fontsize=9)

    ax.annotate("", xy=(0.37, 0.70), xytext=(0.31, 0.70), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.69, 0.70), xytext=(0.63, 0.70), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.36, 0.51, "Lead-lag snapshot, last 24h minute returns", fontsize=13, weight="bold")
    y = 0.47
    for pair, vals in lags.items():
        ax.text(0.08, y, pair, fontsize=11, weight="bold")
        ax.text(0.28, y, f"same-minute corr: {fmt(vals.get('zero_corr'), 3)}", fontsize=10)
        ax.text(0.50, y, f"best BTC/leader lead: {vals.get('best_lag_min')}m", fontsize=10)
        ax.text(0.68, y, f"best corr: {fmt(vals.get('best_corr'), 3)}", fontsize=10)
        y -= 0.045

    ax.add_patch(plt.Rectangle((0.06, 0.12), 0.88, 0.21, fill=False, linewidth=2, edgecolor="#333333"))
    ax.text(0.08, 0.29, "Best S34 operating frame", fontsize=14, weight="bold")
    bullets = [
        "Use BTC as regime/fuel leader, not as sole trigger.",
        "ETH remains primary cascade target; SOL is secondary only when ETH/SOL both confirm.",
        "Current agg/mark data is usable via REST fallback; WebSocket transport is still suppressed.",
        "S34 execution is NOT green until liquidation transport is restored or a validated signed REST liq poller is active.",
        "Next productive step: liquidation-only restoration, then detector replay against this frame.",
    ]
    for i, line in enumerate(bullets):
        ax.text(0.09, 0.255 - i * 0.032, f"- {line}", fontsize=10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "S34_FRAME_PICTURE_2026-06-06.md"
    picture_path = OUT_DIR / "S34_FRAME_PICTURE_2026-06-06.png"

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.execute("pragma query_only=on")
    latest_ms = int(scalar(con, "select max(ts_ms) from mark_prices") or 0)
    h1 = latest_ms - 3600 * 1000
    h6 = latest_ms - 6 * 3600 * 1000
    h24 = latest_ms - 24 * 3600 * 1000
    d30 = latest_ms - 30 * 86400 * 1000

    frames: List[SymbolFrame] = []
    for sym in SYMBOLS:
        last_price = scalar(con, "select mark_price from mark_prices where symbol=? order by ts_ms desc limit 1", (sym,))
        p1 = nearest_price(con, sym, h1, 2 * 3600 * 1000)
        p6 = nearest_price(con, sym, h6, 8 * 3600 * 1000)
        p24 = nearest_price(con, sym, h24, 30 * 3600 * 1000)
        a1 = flow_stats(con, sym, h1)
        a24 = flow_stats(con, sym, h24)
        mark_count_24h = int(scalar(con, "select count(*) from mark_prices where symbol=? and ts_ms>=?", (sym, h24)) or 0)
        liq_count_30d = int(scalar(con, "select count(*) from liquidations where symbol=? and ts_ms>=?", (sym, d30)) or 0)
        last_liq_ts = scalar(con, "select max(ts_ms) from liquidations where symbol=?", (sym,))
        last_agg_ts = scalar(con, "select ts_ms from agg_trades where symbol=? order by ts_ms desc limit 1", (sym,))
        frames.append(
            SymbolFrame(
                symbol=sym,
                last_price=float(last_price) if last_price is not None else None,
                ret_1h=pct(p1, float(last_price) if last_price is not None else None),
                ret_6h=pct(p6, float(last_price) if last_price is not None else None),
                ret_24h=pct(p24, float(last_price) if last_price is not None else None),
                agg_count_1h=a1[0],
                agg_notional_1h=a1[1],
                agg_signed_1h=a1[2],
                agg_count_24h=a24[0],
                agg_notional_24h=a24[1],
                agg_signed_24h=a24[2],
                mark_count_24h=mark_count_24h,
                liq_count_30d=liq_count_30d,
                last_liq_ts=int(last_liq_ts) if last_liq_ts else None,
                last_agg_ts=int(last_agg_ts) if last_agg_ts else None,
            )
        )

    series = load_minute_series(con, h24, latest_ms)
    lags = lead_lag(series, h24, latest_ms)
    detector_latest = scalar(con, "select max(signal_ts_ms) from detector_signals") if table_exists(con, "detector_signals") else None
    detector_count_30d = int(scalar(con, "select count(*) from detector_signals where signal_ts_ms>=?", (d30,)) or 0)
    health = {}
    for p in [ROOT / "logs" / "collector_heartbeat.json", ROOT / "logs" / "health" / "collector.json", ROOT / "reports" / "WATCHDOG_STATUS.json"]:
        try:
            health[p.name] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            health[p.name] = {}

    readiness = "YELLOW for data collection, RED for S34 execution until liquidation transport is restored"
    write_picture(picture_path, frames, lags, readiness)

    lines: List[str] = []
    lines += [
        "# S34 Frame Picture - BTC / ETH / SOL",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Local latest mark timestamp: {iso(latest_ms)}",
        "",
        "## Executive Frame",
        "",
        f"Verdict: **{readiness}.**",
        "",
        "S34 should be framed as a cascade strategy with BTC as the market-state leader, ETH as the primary execution target, and SOL as a secondary/high-beta confirmation leg. The current data layer is no longer dead: REST fallback is writing fresh mark prices for BTC/ETH/SOL and fresh SOL agg trades, but BTC/ETH aggTrade freshness is not yet as clean as markPrice. The hard blocker is still liquidation transport: local liquidation rows are stale, so S34 cannot honestly confirm forced-flow cascades from current local data.",
        "",
        f"Picture artifact: `{picture_path.relative_to(ROOT)}`",
        "",
        "## External Market Context",
        "",
        "- CoinDesk reported a fresh liquidation-heavy regime on 2026-06-03: roughly $1.84B liquidated in 24h, with long liquidations led by BTC, ETH and SOL, and Binance handling a large share of the cascade.",
        "- Coinalyze currently shows large derivatives surface activity: BTC, ETH and SOL all appear among the largest 24h liquidation / OI names, with BTC 24h liquidations around $197M, ETH around $145M, SOL around $31M on its public dashboard snapshot.",
        "",
        "Sources: CoinDesk liquidation report and Coinalyze futures dashboard.",
        "",
        "## Local BTC / ETH / SOL Snapshot",
        "",
        "| Symbol | Last mark | 1h % | 6h % | 24h % | 1h agg notional | 1h taker imbalance | 24h agg notional | 24h taker imbalance | 24h marks | last agg | 30d local liq rows | last local liq |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for f in frames:
        lines.append(
            f"| {f.symbol} | {fmt(f.last_price, 4)} | {fmt(f.ret_1h)} | {fmt(f.ret_6h)} | {fmt(f.ret_24h)} | "
            f"${f.agg_notional_1h/1e6:.1f}M | ${f.agg_signed_1h/1e6:.1f}M | "
            f"${f.agg_notional_24h/1e9:.2f}B | ${f.agg_signed_24h/1e6:.1f}M | "
            f"{f.mark_count_24h:,} | {iso(f.last_agg_ts)} | {f.liq_count_30d:,} | {iso(f.last_liq_ts)} |"
        )

    lines += [
        "",
        "Interpretation:",
        "",
        "- Positive taker imbalance means buyer-initiated notional dominates; negative means seller-initiated notional dominates.",
        "- mark rows are current and usable for regime framing; agg rows need per-symbol freshness checks because BTC/ETH agg are currently stale relative to SOL.",
        "- the liquidation column is the key S34 red flag.",
        "",
        "## Lead-Lag Frame",
        "",
        "| Pair | Same-minute corr | Best leader lag | Best lag corr | S34 meaning |",
        "|---|---:|---:|---:|---|",
    ]
    for pair, vals in lags.items():
        meaning = "core S34 premise" if pair == "BTCUSDT->ETHUSDT" else "confirmation / secondary propagation"
        lines.append(
            f"| {pair} | {fmt(vals.get('zero_corr'), 3)} | {vals.get('best_lag_min')}m | {fmt(vals.get('best_corr'), 3)} | {meaning} |"
        )

    lines += [
        "",
        "The lead-lag table should not be read as a standalone alpha verdict. It is a frame check: if BTC->ETH correlation/lag is weak or unstable, S34's BTC_WATCH premise needs caution; if BTC->ETH remains the strongest propagation lane, S34 architecture is directionally supported.",
        "",
        "## Current System Truth",
        "",
        f"- Collector health: `{(health.get('collector.json') or {}).get('status')}`",
        f"- Watchdog overall: `{(health.get('WATCHDOG_STATUS.json') or {}).get('overall')}`",
        f"- REST fallback active: `{(health.get('collector_heartbeat.json') or {}).get('rest_fallback_active')}`",
        f"- Rows since current collector start: `{json.dumps((health.get('collector_heartbeat.json') or {}).get('rows_written_since_start'), separators=(',', ':'))}`",
        f"- Liquidation transport available: `{(health.get('collector.json') or {}).get('liquidation_transport_available')}`",
        f"- Latest detector signal: {iso(int(detector_latest) if detector_latest else None)}",
        f"- Detector signals in last 30d: {detector_count_30d}",
        "",
        "## Best Frame For S34",
        "",
        "1. BTC is the regime/fuel instrument, not the final confirmation. Watch BTC return shock, BTC taker sell/buy pressure, and BTC->ETH propagation.",
        "2. ETH is the primary S34 execution instrument. It should require BTC lead plus ETH local confirmation, not ETH alone.",
        "3. SOL is a high-beta secondary leg. Use it as confirmation when BTC shock propagates broadly; do not let SOL override BTC/ETH unless a separate SOL-specific edge is validated.",
        "4. Liquidation flow remains the missing forced-flow sensor. Without it, S34 can frame risk but cannot honestly claim cascade confirmation.",
        "5. The next engineering step should be liquidation-only restoration, isolated from agg/mark collector stability.",
        "",
        "## Actionable Decision",
        "",
        "- Data collection: **continue running**. mark data is useful and fresh; agg freshness needs a focused follow-up for BTC/ETH.",
        "- S34 live/paper signal generation: **do not trust yet** until liquidation transport or validated liquidation substitute is online.",
        "- Next PR: **liquidation-only restoration**. Do not stop the working agg/mark fallback while doing it.",
        "",
        "## Honest Limits",
        "",
        "- This report uses local mark/agg data and public derivatives context. It does not repair liquidation transport.",
        "- The local DB has historical liquidation rows, but current liquidation feed is unavailable.",
        "- Lead-lag computed here is a framing diagnostic, not a full walk-forward alpha proof.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)
    print(picture_path)
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
