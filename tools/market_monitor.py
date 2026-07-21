"""
tools/market_monitor.py — Eclipse Scalp Monitor

Terminal tabanlı manuel scalp trading ekranı.
Kolektörler (bookticker_collector + microstructure_collector) arka planda
çalışırken bu ekran SQLite'ı okuyarak canlı güncellenir.

Kullanım:
    python tools/market_monitor.py
    python tools/market_monitor.py --symbols BTCUSDT,ETHUSDT --threshold 100000
    python tools/market_monitor.py --threshold 50000 --refresh 0.3
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from rich import box
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    raise SystemExit("rich gerekli: pip install rich")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "microstructure.db"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_THRESHOLD = 100_000
REFRESH_SEC = 0.5
DELTA_WINDOW_SEC = 30
TAPE_WINDOW_SEC = 300
LIQ_WINDOW_SEC = 300
TAPE_MAX_ROWS = 14
LIQ_MAX_ROWS = 12


# ── Yardımcı ──────────────────────────────────────────────────────────────────

def _now_ms() -> int:
    return int(time.time() * 1000)


def _fmt_usd(n: float) -> str:
    if n >= 1_000_000:
        return f"${n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"${n / 1_000:.0f}k"
    return f"${n:.0f}"


def _fmt_price(p: float, symbol: str) -> str:
    if "BTC" in symbol:
        return f"{p:,.1f}"
    if "ETH" in symbol:
        return f"{p:,.2f}"
    return f"{p:,.3f}"


def _fmt_time(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%H:%M:%S")


# ── Veritabanı sorguları ───────────────────────────────────────────────────────

def _ticker(conn: sqlite3.Connection, symbol: str) -> dict | None:
    row = conn.execute(
        """
        SELECT bid_price, ask_price, mid_price, spread_pct, book_imbalance
        FROM book_ticker
        WHERE symbol = ? AND ts_ms > ?
        ORDER BY ts_ms DESC LIMIT 1
        """,
        (symbol, _now_ms() - 3_000),
    ).fetchone()
    if not row:
        return None
    return {"bid": row[0], "ask": row[1], "mid": row[2],
            "spread_bps": row[3] * 10_000, "imbalance": row[4]}


def _price_30s_ago(conn: sqlite3.Connection, symbol: str) -> float | None:
    row = conn.execute(
        """
        SELECT mid_price FROM book_ticker
        WHERE symbol = ? AND ts_ms BETWEEN ? AND ?
        ORDER BY ts_ms ASC LIMIT 1
        """,
        (symbol, _now_ms() - 35_000, _now_ms() - 25_000),
    ).fetchone()
    return row[0] if row else None


def _delta(conn: sqlite3.Connection, symbol: str) -> float:
    row = conn.execute(
        """
        SELECT SUM(CASE WHEN is_buyer_maker = 0 THEN notional ELSE -notional END)
        FROM agg_trades
        WHERE symbol = ? AND ts_ms > ?
        """,
        (symbol, _now_ms() - DELTA_WINDOW_SEC * 1_000),
    ).fetchone()
    return row[0] or 0.0


def _big_trades(conn: sqlite3.Connection, symbols: list[str], threshold: float) -> list[dict]:
    ph = ",".join("?" * len(symbols))
    rows = conn.execute(
        f"""
        SELECT ts_ms, symbol, notional, is_buyer_maker
        FROM agg_trades
        WHERE symbol IN ({ph}) AND ts_ms > ? AND notional >= ?
        ORDER BY ts_ms DESC LIMIT ?
        """,
        (*symbols, _now_ms() - TAPE_WINDOW_SEC * 1_000, threshold, TAPE_MAX_ROWS),
    ).fetchall()
    return [{"ts_ms": r[0], "symbol": r[1], "notional": r[2], "is_buy": r[3] == 0}
            for r in rows]


def _liquidations(conn: sqlite3.Connection, symbols: list[str]) -> list[dict]:
    ph = ",".join("?" * len(symbols))
    rows = conn.execute(
        f"""
        SELECT ts_ms, symbol, side, notional
        FROM liquidations
        WHERE symbol IN ({ph}) AND ts_ms > ?
        ORDER BY ts_ms DESC LIMIT ?
        """,
        (*symbols, _now_ms() - LIQ_WINDOW_SEC * 1_000, LIQ_MAX_ROWS),
    ).fetchall()
    return [{"ts_ms": r[0], "symbol": r[1], "side": r[2], "notional": r[3]}
            for r in rows]


# ── UI bileşenleri ─────────────────────────────────────────────────────────────

def _imbalance_bar(imb: float, width: int = 10) -> Text:
    filled = int((imb + 1) / 2 * width)
    filled = max(0, min(width, filled))
    pct = int((imb + 1) / 2 * 100)
    bar = "█" * filled + "░" * (width - filled)
    color = "green" if imb > 0.15 else "red" if imb < -0.15 else "yellow"
    t = Text()
    t.append(bar, style=color)
    t.append(f" {pct:2d}%", style=f"{color} bold")
    return t


def _pressure_label(imb: float, delta: float) -> Text:
    score = imb * 0.4 + (1 if delta > 0 else -1) * 0.6
    if score > 0.3:
        return Text("▲ ALICI  ", style="green bold")
    if score < -0.3:
        return Text("▼ SATICI ", style="red bold")
    return Text("◆ NÖTR  ", style="yellow bold")


def _symbol_panel(conn: sqlite3.Connection, symbol: str) -> Panel:
    ticker = _ticker(conn, symbol)
    delta = _delta(conn, symbol)
    old_mid = _price_30s_ago(conn, symbol)

    lines = Table.grid(padding=(0, 0))
    lines.add_column(min_width=26)

    if ticker:
        # Fiyat + 30s değişim
        mid = ticker["mid"]
        price_t = Text(f" {_fmt_price(mid, symbol)}", style="white bold")
        if old_mid and old_mid > 0:
            chg_pct = (mid - old_mid) / old_mid * 100
            color = "green" if chg_pct > 0 else "red" if chg_pct < 0 else "white"
            sign = "+" if chg_pct >= 0 else ""
            price_t.append(f"  {sign}{chg_pct:.3f}%", style=f"{color}")
        lines.add_row(price_t)

        # Bid / Ask
        ba = Text(f" Bid {_fmt_price(ticker['bid'], symbol)}  Ask {_fmt_price(ticker['ask'], symbol)}",
                  style="dim")
        lines.add_row(ba)

        # Spread
        lines.add_row(Text(f" Spread: {ticker['spread_bps']:.2f} bp", style="dim"))

        # İmbalance bar
        imb_row = Text(" Imb: ")
        imb_row.append_text(_imbalance_bar(ticker["imbalance"]))
        lines.add_row(imb_row)

        # Delta
        delta_color = "green" if delta > 0 else "red"
        sign = "+" if delta >= 0 else ""
        delta_row = Text(f" Δ30s: ")
        delta_row.append(f"{sign}{_fmt_usd(abs(delta))} ", style=f"{delta_color} bold")
        delta_row.append_text(_pressure_label(ticker["imbalance"], delta))
        lines.add_row(delta_row)
    else:
        lines.add_row(Text(" — veri yok —", style="dim red"))

    short = symbol.replace("USDT", "")
    return Panel(lines, title=f"[bold]{short}[/bold]", border_style="blue", padding=(0, 0))


def _tape_panel(trades: list[dict], threshold: float) -> Panel:
    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold dim",
              expand=True, padding=(0, 1))
    t.add_column("Saat", style="dim", width=8, no_wrap=True)
    t.add_column("Sembol", width=4, no_wrap=True)
    t.add_column("Yön", width=5, no_wrap=True)
    t.add_column("Tutar", width=9, no_wrap=True)
    t.add_column("", min_width=8)

    max_n = max((x["notional"] for x in trades), default=1.0)

    for tr in trades:
        color = "green" if tr["is_buy"] else "red"
        side = "BUY" if tr["is_buy"] else "SELL"
        bar_len = max(1, int(tr["notional"] / max_n * 22))
        bar = "█" * bar_len
        sym = tr["symbol"].replace("USDT", "")
        t.add_row(
            _fmt_time(tr["ts_ms"]),
            sym,
            Text(side, style=f"{color} bold"),
            Text(_fmt_usd(tr["notional"]), style=color),
            Text(bar, style=color),
        )

    if not trades:
        t.add_row("—", "—", "—", "—", Text("Henüz yok", style="dim"))

    title = f"[bold yellow]BÜYÜK İŞLEMLER ({_fmt_usd(threshold)}+)[/bold yellow]"
    return Panel(t, title=title, border_style="yellow", padding=(0, 0))


def _liq_panel(liqs: list[dict]) -> Panel:
    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold dim",
              expand=True, padding=(0, 1))
    t.add_column("Saat", style="dim", width=8, no_wrap=True)
    t.add_column("Sem.", width=4, no_wrap=True)
    t.add_column("Yön", width=5, no_wrap=True)
    t.add_column("Tutar", width=9, no_wrap=True)

    for liq in liqs:
        is_sell = liq["side"].upper() == "SELL"
        color = "red" if is_sell else "green"
        sym = liq["symbol"].replace("USDT", "")
        t.add_row(
            _fmt_time(liq["ts_ms"]),
            sym,
            Text(liq["side"], style=f"{color} bold"),
            Text(_fmt_usd(liq["notional"]), style=color),
        )

    if not liqs:
        t.add_row("—", "—", "—", Text("Temiz", style="dim green"))

    return Panel(t, title="[bold red]LİKİDASYONLAR[/bold red]",
                 border_style="red", padding=(0, 0))


# ── Ekran düzeni ───────────────────────────────────────────────────────────────

def _build_screen(conn: sqlite3.Connection, symbols: list[str], threshold: float) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=1),
        Layout(name="tickers", size=8),
        Layout(name="bottom"),
    )

    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    header_text = Text(
        f"  ECLIPSE SCALP MONITOR  │  {now_str}  │  Eşik: {_fmt_usd(threshold)}+  │  q: çıkış",
        style="bold white on navy_blue",
    )
    layout["header"].update(header_text)

    sym_layouts = [Layout(_symbol_panel(conn, s), name=s) for s in symbols]
    layout["tickers"].split_row(*sym_layouts)

    trades = _big_trades(conn, symbols, threshold)
    liqs = _liquidations(conn, symbols)

    layout["bottom"].split_row(
        Layout(_tape_panel(trades, threshold), ratio=3),
        Layout(_liq_panel(liqs), ratio=2),
    )

    return layout


# ── Ana döngü ─────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Eclipse Scalp Monitor")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                        help="Virgülle ayrılmış semboller")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Büyük işlem eşiği (USD)")
    parser.add_argument("--db-path", default=str(DEFAULT_DB))
    parser.add_argument("--refresh", type=float, default=REFRESH_SEC,
                        help="Yenileme süresi (saniye)")
    args = parser.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    db_path = Path(args.db_path)

    if not db_path.exists():
        print(f"Hata: DB bulunamadı: {db_path}", file=sys.stderr)
        print("Kolektörleri başlatın:", file=sys.stderr)
        print("  python -m data.bookticker_collector", file=sys.stderr)
        print("  python -m data.microstructure_collector", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=5.0)
    console = Console()

    try:
        with Live(console=console, refresh_per_second=max(1, int(1 / args.refresh)),
                  screen=True) as live:
            while True:
                try:
                    screen = _build_screen(conn, symbols, args.threshold)
                    live.update(screen)
                except Exception as exc:
                    live.update(Text(f"Hata: {exc}", style="bold red"))
                time.sleep(args.refresh)
    except KeyboardInterrupt:
        pass
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
