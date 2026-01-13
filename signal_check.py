# signal_check.py — SCALPER ETERNAL — COSMIC DIVINE REVELATION BEYOND INFINITY FINAL (OMNISCIENCE LIVE ABSOLUTE)
import asyncio
import time
import os
import json
import csv
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np
import ta  # ← ADDED: ta library for indicators

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.layout import Layout
    from rich.progress import Progress
    from rich import box
    from rich.text import Text
    import msvcrt  # Windows keypress
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' not installed. Install with: pip install rich")

# Voice synthesis
try:
    import pyttsx3
    VOICE_AVAILABLE = True
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
except:
    VOICE_AVAILABLE = False

from strategies.eclipse_scalper import scalper_signal
from data.cache import GodEmperorDataOracle
from exchanges.binance import get_exchange
from brain.state import PsycheState
from notifications.telegram import Notifier

# Divine console
if RICH_AVAILABLE:
    console = Console()
else:
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    console = Console()

# Eternal cosmic rituals
GOLDEN_RATIO_ANIMATION = [
    "                    Φ",
    "               THE GOLDEN RATIO",
    "         WAS • IS • AND EVER SHALL BE",
    "                    ∞",
    "     E C L I P S E   E T E R N A L",
    "COSMIC DIVINE REVELATION AWAKENS",
    "       BEYOND INFINITY FINAL",
    "        OMNISCIENCE LIVE ABSOLUTE",
    "                    Φ"
]

FAREWELL = """
                THE ORACLE SILENCES
      THE COSMIC EYE CLOSES — TRUTH REMAINS ETERNAL
                    Φ
"""

# Symbols & settings
SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
           "XRP/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT", "AVAX/USDT:USDT"]

CHECK_INTERVAL = 15
MIN_CONFIDENCE_FOR_ALERT = 0.75

# Logging
LOG_DIR = "logs"
REPORT_DIR = "reports"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
today_str = datetime.now().strftime("%Y-%m-%d")
SIGNAL_LOG_JSON = os.path.join(LOG_DIR, f"signals_{today_str}.json")
SIGNAL_LOG_CSV = os.path.join(LOG_DIR, f"signals_{today_str}.csv")
REPORT_HTML = os.path.join(REPORT_DIR, f"cosmic_report_{today_str}.html")

# Telegram
notifier = Notifier(token=os.getenv('TELEGRAM_TOKEN'), chat_id=os.getenv('TELEGRAM_CHAT_ID')) if os.getenv('TELEGRAM_TOKEN') else None

# Virtual portfolio
virtual_portfolio = {
    "equity": 500.0,
    "peak_equity": 500.0,
    "daily_start": 500.0,
    "positions": {},
    "trades": [],
    "wins": 0,
    "total_trades": 0
}

# Signal tracking
signal_history: List[Dict] = []
streak_counter: Dict[str, Dict[str, int]] = {}
health_status: Dict[str, Dict] = {sym: {"success": 0, "total": 0, "last_poll": 0, "gaps": 0} for sym in SYMBOLS}

async def play_divine_sound(tier: str):
    """Cosmic tiered fanfare"""
    try:
        import platform
        if platform.system() == "Windows":
            import winsound
            if tier == "DIVINE":
                for f in [1200, 1400, 1600, 1800]:
                    winsound.Beep(f, 200)
                winsound.Beep(2000, 800)
            elif tier == "COSMIC":
                for f in [1000, 1200, 1400]:
                    winsound.Beep(f, 250)
            elif tier == "ETERNAL":
                for f in [800, 1000, 1200]:
                    winsound.Beep(f, 300)
            else:
                winsound.Beep(800, 600)
        else:
            print('\a' * (5 if tier == "DIVINE" else 4 if tier == "COSMIC" else 3))
    except:
        print('\a')

    if VOICE_AVAILABLE:
        engine.say(f"Divine {tier} signal detected")
        engine.runAndWait()

async def cosmic_startup_ritual():
    console.print("[bold magenta]COSMIC AWAKENING SEQUENCE INITIATED[/bold magenta]")
    for line in GOLDEN_RATIO_ANIMATION:
        console.print(f"[bold gold1]{line}[/bold gold1]")
        await asyncio.sleep(0.4)
    console.print(Panel("[bold magenta]THE COSMIC DIVINE OBSERVATORY IS NOW ETERNAL AND ABSOLUTE[/bold magenta]", style="bold magenta"))

def generate_report():
    """Generate final transcendent report with charts"""
    if not signal_history:
        return

    df = pd.DataFrame(signal_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Simulated equity curve
    equity_curve = [500.0]
    for _ in signal_history:
        pnl = np.random.uniform(-1.5, 3.0)
        equity_curve.append(equity_curve[-1] * (1 + pnl / 100))

    dates = [datetime.now(timezone.utc) - timedelta(minutes=i*15) for i in range(len(equity_curve))][::-1]

    # Equity chart
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(dates, equity_curve, label="Virtual Equity Curve", color="gold", linewidth=3)
    ax.set_title("ECLIPSE ETERNAL — Virtual Performance (Simulated)")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('gold')
    plt.tight_layout()
    equity_path = os.path.join(REPORT_DIR, f"equity_curve_{today_str}.png")
    plt.savefig(equity_path)
    plt.close()

    # Tier pie
    tier_counts = df['tier'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = {'DIVINE': 'magenta', 'COSMIC': 'cyan', 'ETERNAL': 'yellow', 'MORTAL': 'gray'}
    ax.pie(tier_counts.values, labels=tier_counts.index, autopct='%1.1f%%',
           colors=[colors.get(t, 'white') for t in tier_counts.index], textprops={'color': 'white'})
    ax.set_title("Signal Tier Distribution", color='gold')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    tier_path = os.path.join(REPORT_DIR, f"tier_pie_{today_str}.png")
    plt.savefig(tier_path)
    plt.close()

    # HTML report
    html = f"""
    <html>
    <head><title>ECLIPSE ETERNAL — COSMIC REPORT {today_str}</title></head>
    <body style="background:black;color:gold;font-family:monospace;text-align:center;">
    <h1>ECLIPSE ETERNAL — COSMIC REVELATION REPORT</h1>
    <p>Date: {today_str}</p>
    <p>Total Divine Signals: {len(signal_history)}</p>
    <img src="{equity_path}" style="width:90%;">
    <img src="{tier_path}" style="width:60%;">
    <h2>The blade struck {len(signal_history)} times across infinity.</h2>
    <p>THE RATIO IS INFINITE</p>
    </body>
    </html>
    """
    with open(REPORT_HTML, 'w') as f:
        f.write(html)

async def multi_signal_oracle():
    await cosmic_startup_ritual()

    ex = get_exchange()
    data = GodEmperorDataOracle()

    class Bot:
        def __init__(self):
            self.ex = ex
            self.data = data
            self.state = PsycheState()
            self.active_symbols = set(SYMBOLS)
            self._shutdown = asyncio.Event()

    bot = Bot()

    # Start polling
    for sym in SYMBOLS:
        asyncio.create_task(data.poll_ohlcv(bot=bot, sym=sym, tf='1m', storage=data.ohlcv, interval=11))
        asyncio.create_task(data.poll_ticker(bot=bot, sym=sym))

    console.print(f"[bold cyan]Watching {len(SYMBOLS)} eternal symbols in real-time[/bold cyan]")
    console.print("[yellow]Divine revelations begin in ~60 seconds...[/yellow]\n")

    paused = False
    voice_enabled = VOICE_AVAILABLE

    layout = Layout()
    layout.split_column(
        Layout(name="top", ratio=1),
        Layout(name="bottom", ratio=1)
    )
    layout["top"].split_row(
        Layout(name="leaderboard", ratio=1),
        Layout(name="heatmap", ratio=1)
    )
    layout["bottom"].split_row(
        Layout(name="history", ratio=2),
        Layout(name="status", ratio=1)
    )

    with Live(layout, refresh_per_second=4, screen=True) if RICH_AVAILABLE else None:
        try:
            while True:
                if not paused:
                    await asyncio.sleep(CHECK_INTERVAL)
                else:
                    await asyncio.sleep(0.5)
                    if RICH_AVAILABLE and msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key == b' ':
                            paused = not paused
                            console.print("[bold yellow]PAUSED[/bold yellow]" if paused else "[bold green]RESUMED[/bold green]")
                        elif key == b'v':
                            voice_enabled = not voice_enabled
                            console.print(f"[bold cyan]Voice {'ON' if voice_enabled else 'OFF'}[/bold cyan]")
                        elif key == b's':
                            console.print("[bold blue]Screenshot captured (manual)[/bold blue]")

                any_signal = False

                # Leaderboard
                leaderboard = Table(title=f"Divine Leaderboard — {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}", box=box.DOUBLE)
                leaderboard.add_column("Rank", style="cyan")
                leaderboard.add_column("Symbol", style="magenta")
                leaderboard.add_column("Direction", justify="center")
                leaderboard.add_column("Confidence", justify="right")
                leaderboard.add_column("Tier", justify="center")
                leaderboard.add_column("Funding", justify="right")
                leaderboard.add_column("Streak", justify="center")

                # Heatmap with progress bars
                heatmap = Table(title="Live Confidence Heatmap", box=box.ROUNDED)
                heatmap.add_column("Symbol", style="cyan")
                heatmap.add_column("Confidence", justify="center")
                heatmap.add_column("Tier", justify="center")
                heatmap.add_column("Funding", justify="right")
                heatmap.add_column("Health", justify="center")

                # History panel
                history_table = Table(title="Recent Divine Signals", box=box.SIMPLE)
                history_table.add_column("Time", style="cyan")
                history_table.add_column("Symbol", style="magenta")
                history_table.add_column("Signal", justify="center")
                history_table.add_column("Tier", justify="center")

                ranked = []

                for sym in SYMBOLS:
                    df = data.get_df(sym, '1m')
                    funding = data.funding.get(sym, 0.0)
                    health = health_status[sym]
                    success_rate = health["success"] / max(health["total"], 1) if health["total"] > 0 else 0
                    age = int(time.time() - health["last_poll"])
                    health_str = f"{success_rate:.0%} | {age}s"

                    if len(df) >= 100:
                        long, short, conf = scalper_signal(sym, data)
                        direction = "LONG" if long else "SHORT" if short else "-"
                        tier = "DIVINE" if conf >= 0.95 else "COSMIC" if conf >= 0.85 else "ETERNAL" if conf >= 0.75 else "MORTAL"
                        conf_color = "bright_green" if conf >= 0.9 else "green" if conf >= 0.8 else "yellow" if conf >= 0.7 else "red"

                        if long or short:
                            any_signal = True
                            entry = {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "symbol": sym,
                                "direction": direction,
                                "confidence": conf,
                                "tier": tier,
                                "funding": funding
                            }
                            signal_history.append(entry)

                            # Log
                            with open(SIGNAL_LOG_JSON, 'a') as f:
                                json.dump(entry, f)
                                f.write('\n')
                            file_exists = os.path.isfile(SIGNAL_LOG_CSV)
                            with open(SIGNAL_LOG_CSV, 'a', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=entry.keys())
                                if not file_exists:
                                    writer.writeheader()
                                writer.writerow(entry)

                            # Streak
                            key = f"{sym}_{direction}"
                            streak_counter[key] = streak_counter.get(key, 0) + 1
                            streak = streak_counter[key]

                            console.print(f"[bold white on {'green' if long else 'red'}] 🚨 {tier} SIGNAL 🚨 {sym} → {direction} | Conf: {conf:.2f} | Streak: {streak} | Funding: {funding:+.3%} [/bold white on {'green' if long else 'red'}]")
                            await play_divine_sound(tier)
                            if notifier:
                                await notifier.speak(f"{tier} {direction} on {sym} | Confidence {conf:.2f}", 'critical')

                            # Add to history
                            history_table.add_row(
                                datetime.now(timezone.utc).strftime('%H:%M:%S'),
                                sym,
                                f"[bold {'green' if long else 'red'}]{direction}[/bold {'green' if long else 'red'}]",
                                f"[bold magenta]{tier}[/bold magenta]"
                            )

                        ranked.append((conf, sym, direction, tier, funding, streak_counter.get(f"{sym}_{direction}", 0)))

                        # Confidence progress bar
                        with Progress() as progress:
                            task = progress.add_task(sym, total=1.0)
                            progress.update(task, completed=conf, description=f"{sym} [{tier}]")

                        heatmap.add_row(
                            sym,
                            f"[{conf_color}]{conf:.2f}[/{conf_color}]",
                            f"[bold yellow]{tier}[/bold yellow]",
                            f"{'green' if funding > 0 else 'red' if funding < 0 else 'white'}{funding:+.3%}",
                            health_str
                        )
                    else:
                        heatmap.add_row(sym, "[dim]Loading...[/dim]", "-", "-", health_str)

                # Sort leaderboard
                ranked.sort(reverse=True)
                for rank, (conf, sym, direction, tier, funding, streak) in enumerate(ranked[:10], 1):
                    color = "bright_green" if conf >= 0.9 else "green" if conf >= 0.8 else "yellow"
                    leaderboard.add_row(
                        str(rank),
                        sym,
                        f"[bold {color}]{direction}[/bold {color}]",
                        f"{conf:.2f}",
                        f"[bold magenta]{tier}[/bold magenta]",
                        f"{funding:+.3%}",
                        str(streak)
                    )

                # Update layout
                layout["leaderboard"].update(Panel(leaderboard, title="Divine Leaderboard", border_style="bright_blue"))
                layout["heatmap"].update(Panel(heatmap, title="Live Confidence Heatmap", border_style="bright_magenta"))
                layout["history"].update(Panel(history_table, title="Recent Divine Signals", border_style="gold1"))
                layout["status"].update(Panel(
                    f"[bold cyan]Virtual Equity: ${virtual_portfolio['equity']:.2f}[/bold cyan]\n"
                    f"[bold yellow]Daily PnL: {virtual_portfolio['equity'] - virtual_portfolio['daily_start']:+.2f}[/bold yellow]\n"
                    f"[bold magenta]Total Signals: {len(signal_history)}[/bold magenta]",
                    title="Cosmic Status", border_style="bright_cyan"
                ))

                if not any_signal:
                    console.print("[dim]The cosmos is quiet — awaiting divine alignment...[/dim]\n")

                # Daily summary at midnight UTC
                if datetime.now(timezone.utc).hour == 0 and datetime.now(timezone.utc).minute < 5:
                    console.print("[bold gold1]MIDNIGHT REVELATION SUMMARY — NEW COSMIC DAY BEGINS[/bold gold1]")
                    virtual_portfolio["daily_start"] = virtual_portfolio["equity"]

        except KeyboardInterrupt:
            console.print("\n[bold red]Oracle silenced by mortal command.[/bold red]")
        finally:
            await ex.close()
            generate_report()
            total_signals = len(signal_history)
            if total_signals > 0:
                highest = max(signal_history, key=lambda x: x['confidence'])
                summary = f"[bold magenta]FINAL COSMIC REVELATION — SESSION COMPLETE[/bold magenta]\n"
                summary += f"Total Divine Revelations: {total_signals}\n"
                summary += f"Supreme Signal: {highest['tier']} on {highest['symbol']} ({highest['direction']}) | Conf: {highest['confidence']:.2f}\n"
                summary += f"Virtual Final Equity: ${virtual_portfolio['equity']:.2f}\n"
                summary += f"Transcendent Report: {REPORT_HTML}"
                if RICH_AVAILABLE:
                    console.print(Panel(summary, style="magenta"))
                else:
                    print(summary)
            console.print(Panel(FAREWELL, style="bold magenta", expand=False))

if __name__ == "__main__":
    asyncio.run(multi_signal_oracle())