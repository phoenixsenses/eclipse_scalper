# strategies/backtest_diagnostics.py — Deep diagnostic analysis of backtest results
#
# Investigations:
# 1. Examine losing trades — are entries reasonable? Are stops hit immediately?
# 2. Test inverted R:R (wider TP / tighter SL)
# 3. Test reduced frequency (min time between entries, higher confidence)
# 4. Debug trend-following zero-trade issue
# 5. Calm vs volatile week comparison

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Tuple, Dict, List
from dataclasses import dataclass

from strategies.backtester import DataLoader, TradeDirection, Trade
from strategies.comprehensive_backtest import (
    StrategyBacktester,
    StrategyResult,
    create_legacy_signal,
    create_trend_signal,
    create_reversion_signal,
    create_combined_signal,
)
from strategies.regime_detector import RegimeDetector
from strategies.trend_strategy import TrendFollowingStrategy
from strategies.reversion_strategy import MeanReversionStrategy


# ── Patched backtester that returns trade objects ──────────────────────

class DiagnosticBacktester(StrategyBacktester):
    """Extended backtester that returns full trade list for analysis."""

    def run_with_trades(self, name, df_1m, df_5m, signal_func, min_confidence=0.4,
                        min_bars_between=0):
        trades = []
        trade_id = 0
        equity = self.initial_capital
        peak_equity = equity
        max_drawdown_pct = 0.0
        position = None
        atr_period = 14
        lookback = 100
        last_entry_bar = -9999

        for i in range(lookback, len(df_1m)):
            bar = df_1m.iloc[i]
            current_time = bar.name if hasattr(bar.name, 'strftime') else datetime.now(timezone.utc)
            current_price = float(bar["c"])
            current_high = float(bar["h"])
            current_low = float(bar["l"])

            high = df_1m["h"].iloc[i - atr_period:i]
            low = df_1m["l"].iloc[i - atr_period:i]
            close_prev = df_1m["c"].iloc[i - atr_period - 1:i - 1]
            tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
            atr = float(tr.mean()) if len(tr) > 0 else 0.0

            if position is not None:
                exit_price = None
                exit_reason = ""

                if position["direction"] == TradeDirection.LONG:
                    if current_low <= position["stop_loss"]:
                        exit_price = position["stop_loss"]
                        exit_reason = "SL"
                    elif current_high >= position["take_profit"]:
                        exit_price = position["take_profit"]
                        exit_reason = "TP"
                else:
                    if current_high >= position["stop_loss"]:
                        exit_price = position["stop_loss"]
                        exit_reason = "SL"
                    elif current_low <= position["take_profit"]:
                        exit_price = position["take_profit"]
                        exit_reason = "TP"

                if exit_price is None and (i - position["entry_bar"]) >= self.max_hold_bars:
                    exit_price = current_price
                    exit_reason = "TIME"

                if exit_price is not None:
                    if position["direction"] == TradeDirection.LONG:
                        pnl = (exit_price - position["entry_price"]) * position["size"]
                    else:
                        pnl = (position["entry_price"] - exit_price) * position["size"]
                    fees = (position["entry_price"] + exit_price) * position["size"] * self.fee_rate
                    pnl -= fees

                    trade = Trade(
                        id=trade_id, symbol=self.symbol, direction=position["direction"],
                        entry_price=position["entry_price"], entry_time=position["entry_time"],
                        exit_price=exit_price, exit_time=current_time,
                        size=position["size"], stop_loss=position["stop_loss"],
                        take_profit=position["take_profit"], pnl=pnl,
                        pnl_pct=pnl / (position["entry_price"] * position["size"]),
                        exit_reason=exit_reason, strategy=name,
                    )
                    # Store hold duration
                    trade._hold_bars = i - position["entry_bar"]
                    trade._atr_at_entry = position.get("atr", 0)
                    trades.append(trade)
                    trade_id += 1
                    equity += pnl
                    peak_equity = max(peak_equity, equity)
                    dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                    max_drawdown_pct = max(max_drawdown_pct, dd)
                    position = None

            if position is None and atr > 0 and equity > 0:
                if (i - last_entry_bar) < min_bars_between:
                    continue

                df_1m_window = df_1m.iloc[max(0, i - 300):i + 1]
                df_5m_window = df_5m.loc[:df_1m.index[i]]

                try:
                    is_long, is_short, confidence = signal_func(df_1m_window, df_5m_window, i)
                except Exception:
                    is_long, is_short, confidence = False, False, 0.0

                if (is_long or is_short) and confidence >= min_confidence:
                    slippage = current_price * 0.0001
                    entry_price = current_price + slippage if is_long else current_price - slippage
                    if is_long:
                        stop_loss = entry_price - (atr * self.sl_mult)
                        take_profit = entry_price + (atr * self.tp_mult)
                    else:
                        stop_loss = entry_price + (atr * self.sl_mult)
                        take_profit = entry_price - (atr * self.tp_mult)

                    risk_amount = equity * self.risk_per_trade
                    stop_distance = abs(entry_price - stop_loss)
                    size = risk_amount / stop_distance if stop_distance > 0 else 0
                    max_size = (equity * 5) / entry_price
                    size = min(size, max_size)

                    if size > 0:
                        position = {
                            "direction": TradeDirection.LONG if is_long else TradeDirection.SHORT,
                            "entry_price": entry_price, "entry_time": current_time,
                            "entry_bar": i, "stop_loss": stop_loss,
                            "take_profit": take_profit, "size": size, "atr": atr,
                        }
                        last_entry_bar = i

        total = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / total if total > 0 else 0.0
        gross_profit = sum(t.pnl for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
        pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

        result = StrategyResult(
            name=name, total_trades=total, winning_trades=len(wins),
            losing_trades=len(losses), win_rate=win_rate, profit_factor=pf,
            total_return_pct=(equity - self.initial_capital) / self.initial_capital,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=0.0, final_equity=equity,
        )
        return result, trades


def load_data(symbol="BTCUSDT", days=7):
    loader = DataLoader()
    df_1m = loader.download_binance(symbol, "1m", days=days)
    if not isinstance(df_1m.index, pd.DatetimeIndex):
        if 'ts' in df_1m.columns:
            df_1m['datetime'] = pd.to_datetime(df_1m['ts'], unit='ms')
            df_1m = df_1m.set_index('datetime')
    df_5m = df_1m.resample("5min").agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna()
    return df_1m, df_5m


# ══════════════════════════════════════════════════════════════════════
# INVESTIGATION 1: Examine losing trades
# ══════════════════════════════════════════════════════════════════════

def investigate_losing_trades(df_1m, df_5m):
    print("\n" + "=" * 90)
    print("INVESTIGATION 1: LOSING TRADE ANATOMY")
    print("=" * 90)

    bt = DiagnosticBacktester(sl_mult=1.5, tp_mult=3.0)
    result, trades = bt.run_with_trades("Legacy", df_1m, df_5m, create_legacy_signal, 0.3)

    if not trades:
        print("No trades to analyze")
        return

    losses = [t for t in trades if t.pnl <= 0]
    wins = [t for t in trades if t.pnl > 0]

    # Exit reason breakdown
    sl_hits = [t for t in trades if t.exit_reason == "SL"]
    tp_hits = [t for t in trades if t.exit_reason == "TP"]
    time_exits = [t for t in trades if t.exit_reason == "TIME"]

    print(f"\nTotal trades: {len(trades)}")
    print(f"  SL hits: {len(sl_hits)} ({len(sl_hits)/len(trades)*100:.1f}%)")
    print(f"  TP hits: {len(tp_hits)} ({len(tp_hits)/len(trades)*100:.1f}%)")
    print(f"  TIME exits: {len(time_exits)} ({len(time_exits)/len(trades)*100:.1f}%)")

    # Hold duration analysis
    hold_bars = [getattr(t, '_hold_bars', 0) for t in trades]
    sl_hold = [getattr(t, '_hold_bars', 0) for t in sl_hits]
    tp_hold = [getattr(t, '_hold_bars', 0) for t in tp_hits]

    print(f"\nHold duration (bars = minutes):")
    print(f"  All trades:  median={np.median(hold_bars):.0f}, mean={np.mean(hold_bars):.0f}")
    if sl_hold:
        print(f"  SL trades:   median={np.median(sl_hold):.0f}, mean={np.mean(sl_hold):.0f}")
    if tp_hold:
        print(f"  TP trades:   median={np.median(tp_hold):.0f}, mean={np.mean(tp_hold):.0f}")

    # Immediate SL hits (within 1-3 bars)
    immediate_sl = [t for t in sl_hits if getattr(t, '_hold_bars', 999) <= 3]
    print(f"\n  IMMEDIATE SL (<=3 bars): {len(immediate_sl)} ({len(immediate_sl)/max(1,len(sl_hits))*100:.1f}% of SL hits)")

    # Time exit P&L direction
    if time_exits:
        time_wins = [t for t in time_exits if t.pnl > 0]
        time_losses = [t for t in time_exits if t.pnl <= 0]
        print(f"\n  TIME exits winning: {len(time_wins)}, losing: {len(time_losses)}")
        if time_exits:
            avg_time_pnl_pct = np.mean([t.pnl_pct for t in time_exits]) * 100
            print(f"  TIME exit avg P&L: {avg_time_pnl_pct:+.3f}%")

    # Direction analysis
    longs = [t for t in trades if t.direction == TradeDirection.LONG]
    shorts = [t for t in trades if t.direction == TradeDirection.SHORT]
    long_wr = len([t for t in longs if t.pnl > 0]) / max(1, len(longs))
    short_wr = len([t for t in shorts if t.pnl > 0]) / max(1, len(shorts))
    print(f"\nDirection split:")
    print(f"  LONG:  {len(longs)} trades, WR={long_wr*100:.1f}%")
    print(f"  SHORT: {len(shorts)} trades, WR={short_wr*100:.1f}%")

    # Sample 5 worst losses
    worst = sorted(trades, key=lambda t: t.pnl)[:5]
    print(f"\n--- 5 worst trades ---")
    for t in worst:
        atr = getattr(t, '_atr_at_entry', 0)
        sl_dist_atr = abs(t.entry_price - t.stop_loss) / atr if atr > 0 else 0
        tp_dist_atr = abs(t.take_profit - t.entry_price) / atr if atr > 0 else 0
        hold = getattr(t, '_hold_bars', 0)
        print(f"  {t.direction.value:5s} entry=${t.entry_price:.2f} exit=${t.exit_price:.2f} "
              f"SL=${t.stop_loss:.2f} TP=${t.take_profit:.2f} "
              f"hold={hold}bars exit={t.exit_reason} pnl={t.pnl_pct*100:+.2f}%")


# ══════════════════════════════════════════════════════════════════════
# INVESTIGATION 2: Test inverted R:R
# ══════════════════════════════════════════════════════════════════════

def investigate_rr_ratios(df_1m, df_5m):
    print("\n" + "=" * 90)
    print("INVESTIGATION 2: R:R RATIO SWEEP")
    print("=" * 90)

    configs = [
        ("Original (SL=1.5 TP=3.0)", 1.5, 3.0),
        ("Tight SL (SL=0.75 TP=3.0)", 0.75, 3.0),
        ("Wide TP (SL=1.5 TP=6.0)", 1.5, 6.0),
        ("Inverted (SL=3.0 TP=1.5)", 3.0, 1.5),
        ("Scalp (SL=1.0 TP=1.0)", 1.0, 1.0),
        ("Tight both (SL=0.5 TP=1.0)", 0.5, 1.0),
        ("Wide SL (SL=3.0 TP=3.0)", 3.0, 3.0),
        ("Extreme inv (SL=4.0 TP=1.0)", 4.0, 1.0),
    ]

    print(f"\n{'Config':<32} {'Trades':>7} {'WR':>8} {'PF':>7} {'Return':>9} {'MaxDD':>8}")
    print("-" * 75)

    for label, sl, tp in configs:
        bt = DiagnosticBacktester(sl_mult=sl, tp_mult=tp)
        result, trades = bt.run_with_trades("Legacy", df_1m, df_5m, create_legacy_signal, 0.3)
        pf_str = f"{result.profit_factor:.2f}" if result.profit_factor < 100 else "inf"
        print(f"{label:<32} {result.total_trades:>7} {result.win_rate*100:>7.1f}% {pf_str:>7} "
              f"{result.total_return_pct*100:>8.1f}% {result.max_drawdown_pct*100:>7.1f}%")


# ══════════════════════════════════════════════════════════════════════
# INVESTIGATION 3: Reduced frequency
# ══════════════════════════════════════════════════════════════════════

def investigate_frequency(df_1m, df_5m):
    print("\n" + "=" * 90)
    print("INVESTIGATION 3: TRADE FREQUENCY / CONFIDENCE FILTER")
    print("=" * 90)

    configs = [
        ("Baseline (gap=0, conf=0.3)", 0, 0.3),
        ("Gap=15min, conf=0.3", 15, 0.3),
        ("Gap=30min, conf=0.3", 30, 0.3),
        ("Gap=60min, conf=0.3", 60, 0.3),
        ("Gap=0, conf=0.5", 0, 0.5),
        ("Gap=0, conf=0.7", 0, 0.7),
        ("Gap=30, conf=0.5", 30, 0.5),
        ("Gap=60, conf=0.7", 60, 0.7),
    ]

    print(f"\n{'Config':<32} {'Trades':>7} {'WR':>8} {'PF':>7} {'Return':>9} {'MaxDD':>8}")
    print("-" * 75)

    for label, gap, conf in configs:
        bt = DiagnosticBacktester(sl_mult=1.5, tp_mult=3.0)
        result, trades = bt.run_with_trades("Legacy", df_1m, df_5m, create_legacy_signal, conf,
                                            min_bars_between=gap)
        pf_str = f"{result.profit_factor:.2f}" if result.profit_factor < 100 else "inf"
        print(f"{label:<32} {result.total_trades:>7} {result.win_rate*100:>7.1f}% {pf_str:>7} "
              f"{result.total_return_pct*100:>8.1f}% {result.max_drawdown_pct*100:>7.1f}%")


# ══════════════════════════════════════════════════════════════════════
# INVESTIGATION 4: Debug trend-following
# ══════════════════════════════════════════════════════════════════════

def investigate_trend_following(df_1m, df_5m):
    print("\n" + "=" * 90)
    print("INVESTIGATION 4: TREND-FOLLOWING SIGNAL DEBUG")
    print("=" * 90)

    strategy = TrendFollowingStrategy()

    # Sample signals at regular intervals
    lookback = 100
    signals_long = 0
    signals_short = 0
    signals_none = 0
    confidences = []

    sample_points = range(lookback, len(df_1m), 60)  # every 60 bars
    for i in sample_points:
        df_1m_window = df_1m.iloc[max(0, i - 300):i + 1]
        df_5m_window = df_5m.loc[:df_1m.index[i]]

        try:
            is_long, is_short, confidence = strategy.check_entry(
                df_1m_window, df_5m_window if len(df_5m_window) > 0 else None
            )
        except Exception as e:
            print(f"  ERROR at bar {i}: {e}")
            continue

        if is_long:
            signals_long += 1
        elif is_short:
            signals_short += 1
        else:
            signals_none += 1
        if confidence > 0:
            confidences.append(confidence)

    total_samples = len(list(sample_points))
    print(f"\nSampled {total_samples} points (every 60 bars):")
    print(f"  LONG signals:  {signals_long} ({signals_long/max(1,total_samples)*100:.1f}%)")
    print(f"  SHORT signals: {signals_short} ({signals_short/max(1,total_samples)*100:.1f}%)")
    print(f"  No signal:     {signals_none} ({signals_none/max(1,total_samples)*100:.1f}%)")

    if confidences:
        print(f"\nConfidence when signal fires:")
        print(f"  min={min(confidences):.3f} median={np.median(confidences):.3f} max={max(confidences):.3f}")
        above_04 = len([c for c in confidences if c >= 0.4])
        print(f"  Above 0.4 threshold: {above_04}/{len(confidences)}")
    else:
        print("\n  ** NO SIGNALS FIRED AT ALL **")

    # Check regime detection
    print(f"\nRegime detection across data:")
    detector = RegimeDetector()
    regimes = {"TRENDING": 0, "RANGING": 0, "VOLATILE": 0, "UNKNOWN": 0}
    for i in range(lookback, len(df_1m), 60):
        df_1m_window = df_1m.iloc[max(0, i - 300):i + 1]
        try:
            regime = detector.detect(df_1m_window)
            r = regime.regime if hasattr(regime, 'regime') else str(regime)
            regimes[r] = regimes.get(r, 0) + 1
        except Exception:
            regimes["UNKNOWN"] += 1

    for r, count in sorted(regimes.items(), key=lambda x: -x[1]):
        print(f"  {r}: {count} ({count/max(1,total_samples)*100:.1f}%)")

    # Try with relaxed confidence threshold
    print(f"\nTrend-following with min_confidence=0.1:")
    bt = DiagnosticBacktester(sl_mult=1.5, tp_mult=3.0)
    result, trades = bt.run_with_trades("Trend-Relaxed", df_1m, df_5m,
                                        create_trend_signal(strategy), 0.1)
    print(f"  Trades: {result.total_trades}, WR: {result.win_rate*100:.1f}%, PF: {result.profit_factor:.2f}")


# ══════════════════════════════════════════════════════════════════════
# INVESTIGATION 5: Calm vs volatile periods
# ══════════════════════════════════════════════════════════════════════

def investigate_regimes(df_1m, df_5m):
    print("\n" + "=" * 90)
    print("INVESTIGATION 5: CALM vs VOLATILE PERIODS")
    print("=" * 90)

    # Split data into daily chunks and measure volatility
    daily_vol = {}
    for date, group in df_1m.groupby(df_1m.index.date):
        if len(group) < 100:
            continue
        returns = group["c"].pct_change().dropna()
        vol = float(returns.std() * 100)  # daily vol as pct
        daily_vol[date] = vol

    if len(daily_vol) < 4:
        print("Not enough daily data for regime split")
        return

    sorted_days = sorted(daily_vol.items(), key=lambda x: x[1])
    mid = len(sorted_days) // 2
    calm_days = set(d for d, _ in sorted_days[:mid])
    volatile_days = set(d for d, _ in sorted_days[mid:])

    calm_vol = np.mean([v for d, v in sorted_days[:mid]])
    volatile_vol = np.mean([v for d, v in sorted_days[mid:]])
    print(f"\nCalm days ({len(calm_days)}):    avg daily vol = {calm_vol:.4f}%")
    print(f"Volatile days ({len(volatile_days)}): avg daily vol = {volatile_vol:.4f}%")

    # Filter data
    df_1m_calm = df_1m[df_1m.index.map(lambda x: x.date() in calm_days)]
    df_1m_volatile = df_1m[df_1m.index.map(lambda x: x.date() in volatile_days)]

    for label, df_subset in [("CALM", df_1m_calm), ("VOLATILE", df_1m_volatile)]:
        if len(df_subset) < 200:
            print(f"\n{label}: insufficient data ({len(df_subset)} bars)")
            continue

        df_5m_sub = df_subset.resample("5min").agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna()

        print(f"\n--- {label} ({len(df_subset)} bars) ---")
        for name, sl, tp in [("Original R:R", 1.5, 3.0), ("Inverted R:R", 3.0, 1.5)]:
            bt = DiagnosticBacktester(sl_mult=sl, tp_mult=tp)
            result, trades = bt.run_with_trades("Legacy", df_subset, df_5m_sub, create_legacy_signal, 0.3)
            sl_count = len([t for t in trades if t.exit_reason == "SL"])
            tp_count = len([t for t in trades if t.exit_reason == "TP"])
            time_count = len([t for t in trades if t.exit_reason == "TIME"])
            print(f"  {name:<18} trades={result.total_trades:>4} WR={result.win_rate*100:.1f}% "
                  f"PF={result.profit_factor:.2f} ret={result.total_return_pct*100:+.1f}% "
                  f"[SL={sl_count} TP={tp_count} TIME={time_count}]")


# ══════════════════════════════════════════════════════════════════════

def main():
    print("Loading 7 days BTCUSDT data...")
    df_1m, df_5m = load_data("BTCUSDT", 7)
    print(f"Loaded {len(df_1m):,} bars")

    investigate_losing_trades(df_1m, df_5m)
    investigate_rr_ratios(df_1m, df_5m)
    investigate_frequency(df_1m, df_5m)
    investigate_trend_following(df_1m, df_5m)
    investigate_regimes(df_1m, df_5m)


if __name__ == "__main__":
    main()
