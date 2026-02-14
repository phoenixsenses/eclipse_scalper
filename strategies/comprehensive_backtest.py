# strategies/comprehensive_backtest.py — SCALPER ETERNAL — COMPREHENSIVE BACKTEST — 2026 v1.0
# Compare all strategies: Legacy, Enhanced, Trend-Following, Mean-Reversion, Combined
#
# Usage:
#     python -m strategies.comprehensive_backtest --symbol BTCUSDT --days 30

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from strategies.backtester import (
    DataLoader,
    BacktestConfig,
    Trade,
    TradeDirection,
)
from strategies.regime_detector import RegimeDetector
from strategies.trend_strategy import TrendFollowingStrategy
from strategies.reversion_strategy import MeanReversionStrategy


@dataclass
class StrategyResult:
    """Result for a single strategy backtest."""
    name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    final_equity: float


class StrategyBacktester:
    """Backtest a strategy with custom signal function."""

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        initial_capital: float = 10000.0,
        sl_mult: float = 1.5,
        tp_mult: float = 3.0,  # Better ratio based on analysis
        risk_per_trade: float = 0.01,
        fee_rate: float = 0.0004,
        max_hold_bars: int = 60,
    ):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult
        self.risk_per_trade = risk_per_trade
        self.fee_rate = fee_rate
        self.max_hold_bars = max_hold_bars

    def run(
        self,
        name: str,
        df_1m: pd.DataFrame,
        df_5m: pd.DataFrame,
        signal_func: Callable[[pd.DataFrame, pd.DataFrame, int], Tuple[bool, bool, float]],
        min_confidence: float = 0.4,
    ) -> StrategyResult:
        """
        Run backtest with custom signal function.

        Args:
            name: Strategy name
            df_1m: 1-minute data
            df_5m: 5-minute data
            signal_func: Function(df_1m_window, df_5m_window, bar_idx) -> (is_long, is_short, confidence)
            min_confidence: Minimum confidence for entry

        Returns:
            StrategyResult with metrics
        """
        trades = []
        trade_id = 0
        equity = self.initial_capital
        peak_equity = equity
        max_drawdown_pct = 0.0

        position = None
        atr_period = 14
        lookback = 100

        for i in range(lookback, len(df_1m)):
            bar = df_1m.iloc[i]
            current_time = bar.name if hasattr(bar.name, 'strftime') else datetime.now(timezone.utc)
            current_price = float(bar["c"])
            current_high = float(bar["h"])
            current_low = float(bar["l"])

            # ATR
            high = df_1m["h"].iloc[i - atr_period:i]
            low = df_1m["l"].iloc[i - atr_period:i]
            close_prev = df_1m["c"].iloc[i - atr_period - 1:i - 1]
            tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
            atr = float(tr.mean()) if len(tr) > 0 else 0.0

            # Exit check
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
                        id=trade_id,
                        symbol=self.symbol,
                        direction=position["direction"],
                        entry_price=position["entry_price"],
                        entry_time=position["entry_time"],
                        exit_price=exit_price,
                        exit_time=current_time,
                        size=position["size"],
                        stop_loss=position["stop_loss"],
                        take_profit=position["take_profit"],
                        pnl=pnl,
                        pnl_pct=pnl / (position["entry_price"] * position["size"]),
                        exit_reason=exit_reason,
                        strategy=name,
                    )
                    trades.append(trade)
                    trade_id += 1

                    equity += pnl
                    peak_equity = max(peak_equity, equity)
                    dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                    max_drawdown_pct = max(max_drawdown_pct, dd)

                    position = None

            # Entry check
            if position is None and atr > 0 and equity > 0:
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
                            "entry_price": entry_price,
                            "entry_time": current_time,
                            "entry_bar": i,
                            "size": size,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                        }

        # Metrics
        total = len(trades)
        wins = [t for t in trades if t.is_winner]
        losses = [t for t in trades if not t.is_winner]
        win_rate = len(wins) / total if total > 0 else 0.0

        gross_profit = sum(t.pnl for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
        pf = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)

        total_return = (equity - self.initial_capital) / self.initial_capital

        # Sharpe
        if trades:
            returns = [t.pnl_pct for t in trades]
            mean_ret = np.mean(returns)
            std_ret = np.std(returns) if len(returns) > 1 else 0.0
            sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        return StrategyResult(
            name=name,
            total_trades=total,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            profit_factor=pf,
            total_return_pct=total_return,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe,
            final_equity=equity,
        )


class TimeExitBacktester(StrategyBacktester):
    """Backtest using time-based exits only (no SL/TP). Tests pure directional edge."""

    def __init__(self, symbol="BTCUSDT", initial_capital=10000.0, exit_bars=30,
                 risk_per_trade=0.01, fee_rate=0.0004, emergency_sl_mult=5.0):
        super().__init__(symbol=symbol, initial_capital=initial_capital,
                         sl_mult=emergency_sl_mult, tp_mult=999.0,
                         risk_per_trade=risk_per_trade, fee_rate=fee_rate,
                         max_hold_bars=exit_bars)


class TrailingExitBacktester(StrategyBacktester):
    """Time exit + trailing stop once in profit. Captures winning runs, limits losers."""

    def __init__(self, symbol="BTCUSDT", initial_capital=10000.0, exit_bars=30,
                 risk_per_trade=0.01, fee_rate=0.0004, trail_atr_mult=2.0,
                 emergency_sl_mult=4.0):
        super().__init__(symbol=symbol, initial_capital=initial_capital,
                         sl_mult=emergency_sl_mult, tp_mult=999.0,
                         risk_per_trade=risk_per_trade, fee_rate=fee_rate,
                         max_hold_bars=exit_bars)
        self.trail_atr_mult = trail_atr_mult

    def run(self, name, df_1m, df_5m, signal_func, min_confidence=0.4):
        trades = []
        trade_id = 0
        equity = self.initial_capital
        peak_equity = equity
        max_drawdown_pct = 0.0
        position = None
        atr_period = 14
        lookback = 100

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

                # Update trailing stop if in profit
                if position["direction"] == TradeDirection.LONG:
                    new_trail = current_price - atr * self.trail_atr_mult
                    if current_price > position["entry_price"]:
                        position["stop_loss"] = max(position["stop_loss"], new_trail)
                    if current_low <= position["stop_loss"]:
                        exit_price = position["stop_loss"]
                        exit_reason = "TRAIL" if position["stop_loss"] > position["entry_price"] - atr * self.sl_mult else "SL"
                else:
                    new_trail = current_price + atr * self.trail_atr_mult
                    if current_price < position["entry_price"]:
                        position["stop_loss"] = min(position["stop_loss"], new_trail)
                    if current_high >= position["stop_loss"]:
                        exit_price = position["stop_loss"]
                        exit_reason = "TRAIL" if position["stop_loss"] < position["entry_price"] + atr * self.sl_mult else "SL"

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
                        exit_price=exit_price, exit_time=current_time, size=position["size"],
                        stop_loss=position["stop_loss"], take_profit=position.get("take_profit", 0),
                        pnl=pnl, pnl_pct=pnl / (position["entry_price"] * position["size"]),
                        exit_reason=exit_reason, strategy=name,
                    )
                    trades.append(trade)
                    trade_id += 1
                    equity += pnl
                    peak_equity = max(peak_equity, equity)
                    dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                    max_drawdown_pct = max(max_drawdown_pct, dd)
                    position = None

            if position is None and atr > 0 and equity > 0:
                df_1m_window = df_1m.iloc[max(0, i - 300):i + 1]
                df_5m_window = df_5m.loc[:df_1m.index[i]]
                try:
                    is_long, is_short, confidence = signal_func(df_1m_window, df_5m_window, i)
                except Exception:
                    is_long, is_short, confidence = False, False, 0.0

                if (is_long or is_short) and confidence >= min_confidence:
                    slippage = current_price * 0.0001
                    entry_price = current_price + slippage if is_long else current_price - slippage
                    stop_loss = entry_price - (atr * self.sl_mult) if is_long else entry_price + (atr * self.sl_mult)
                    risk_amount = equity * self.risk_per_trade
                    stop_distance = abs(entry_price - stop_loss)
                    size = risk_amount / stop_distance if stop_distance > 0 else 0
                    max_size = (equity * 5) / entry_price
                    size = min(size, max_size)
                    if size > 0:
                        position = {
                            "direction": TradeDirection.LONG if is_long else TradeDirection.SHORT,
                            "entry_price": entry_price, "entry_time": current_time,
                            "entry_bar": i, "size": size, "stop_loss": stop_loss,
                        }

        total = len(trades)
        wins = [t for t in trades if t.is_winner]
        losses = [t for t in trades if not t.is_winner]
        win_rate = len(wins) / total if total > 0 else 0.0
        gross_profit = sum(t.pnl for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
        pf = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)
        total_return = (equity - self.initial_capital) / self.initial_capital
        if trades:
            returns = [t.pnl_pct for t in trades]
            mean_ret = np.mean(returns)
            std_ret = np.std(returns) if len(returns) > 1 else 0.0
            sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0
        return StrategyResult(
            name=name, total_trades=total, winning_trades=len(wins),
            losing_trades=len(losses), win_rate=win_rate, profit_factor=pf,
            total_return_pct=total_return, max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe, final_equity=equity,
        )


def create_legacy_signal(df_1m: pd.DataFrame, df_5m: pd.DataFrame, i: int) -> Tuple[bool, bool, float]:
    """Legacy EMA crossover signal."""
    if len(df_1m) < 60:
        return False, False, 0.0

    close = df_1m["c"]
    ema20 = close.ewm(span=20).mean().iloc[-1]
    ema50 = close.ewm(span=50).mean().iloc[-1]
    mom = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] if close.iloc[-6] != 0 else 0

    is_long = ema20 > ema50 and mom > 0.001
    is_short = ema20 < ema50 and mom < -0.001

    confidence = 0.5 if (is_long or is_short) else 0.0
    return is_long, is_short, confidence


def create_trend_signal(strategy: TrendFollowingStrategy):
    """Create trend-following signal function."""
    def signal_func(df_1m: pd.DataFrame, df_5m: pd.DataFrame, i: int) -> Tuple[bool, bool, float]:
        return strategy.check_entry(df_1m, df_5m if len(df_5m) > 0 else None)
    return signal_func


def create_reversion_signal(strategy: MeanReversionStrategy):
    """Create mean-reversion signal function."""
    def signal_func(df_1m: pd.DataFrame, df_5m: pd.DataFrame, i: int) -> Tuple[bool, bool, float]:
        return strategy.check_entry(df_1m, df_5m if len(df_5m) > 0 else None)
    return signal_func


def create_combined_signal(
    detector: RegimeDetector,
    trend_strategy: TrendFollowingStrategy,
    reversion_strategy: MeanReversionStrategy,
):
    """Create combined regime-based signal function."""
    def signal_func(df_1m: pd.DataFrame, df_5m: pd.DataFrame, i: int) -> Tuple[bool, bool, float]:
        # Detect regime
        regime = detector.detect(df_1m)

        if regime.regime == "TRENDING" and regime.confidence >= 0.3:
            return trend_strategy.check_entry(df_1m, df_5m if len(df_5m) > 0 else None)
        elif regime.regime == "RANGING" and regime.confidence >= 0.3:
            return reversion_strategy.check_entry(df_1m, df_5m if len(df_5m) > 0 else None)
        else:
            # VOLATILE or low confidence - no trade
            return False, False, 0.0
    return signal_func


def create_enhanced_strict_signal(enhanced):
    """Create enhanced signal function with strict MTF filtering."""
    agg = {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}

    def signal_func(df_1m: pd.DataFrame, df_5m: pd.DataFrame, i: int) -> Tuple[bool, bool, float]:
        if len(df_1m) < 50:
            return False, False, 0.0
        _5m = df_1m.resample("5min").agg(agg).dropna() if len(df_1m) >= 10 else None
        _15m = df_1m.resample("15min").agg(agg).dropna() if len(df_1m) >= 30 else None
        _1h = df_1m.resample("1h").agg(agg).dropna() if len(df_1m) >= 120 else None
        result = enhanced.analyze(df_1m=df_1m, df_5m=_5m, df_15m=_15m, df_1h=_1h)
        return result.long_signal, result.short_signal, result.confidence
    return signal_func


def create_regime_filtered_signal(
    enhanced,
    detector: RegimeDetector,
    allowed_regimes: Tuple[str, ...] = ("TRENDING", "TRANSITIONING"),
):
    """Create enhanced signal filtered by regime — only trade when regime matches."""
    agg = {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}

    def signal_func(df_1m: pd.DataFrame, df_5m: pd.DataFrame, i: int) -> Tuple[bool, bool, float]:
        if len(df_1m) < 50:
            return False, False, 0.0
        # Check regime first (cheap)
        regime = detector.detect(df_1m)
        if regime.regime not in allowed_regimes:
            return False, False, 0.0
        # Run full enhanced analysis
        _5m = df_1m.resample("5min").agg(agg).dropna() if len(df_1m) >= 10 else None
        _15m = df_1m.resample("15min").agg(agg).dropna() if len(df_1m) >= 30 else None
        _1h = df_1m.resample("1h").agg(agg).dropna() if len(df_1m) >= 120 else None
        result = enhanced.analyze(df_1m=df_1m, df_5m=_5m, df_15m=_15m, df_1h=_1h)
        # Boost confidence by regime confidence
        conf = result.confidence * (0.8 + 0.2 * regime.confidence)
        return result.long_signal, result.short_signal, conf
    return signal_func


def run_comprehensive_backtest(symbol: str, days: int) -> Dict[str, StrategyResult]:
    """Run backtest on all strategies and compare."""
    print(f"\nLoading {days} days of {symbol} data...")
    loader = DataLoader()
    df_1m = loader.download_binance(symbol, "1m", days=days)

    if df_1m.empty:
        raise ValueError("No data loaded")

    print(f"Loaded {len(df_1m):,} bars")

    # Ensure datetime index
    if not isinstance(df_1m.index, pd.DatetimeIndex):
        if 'ts' in df_1m.columns:
            df_1m['datetime'] = pd.to_datetime(df_1m['ts'], unit='ms')
            df_1m = df_1m.set_index('datetime')
        elif 't' in df_1m.columns:
            df_1m['datetime'] = pd.to_datetime(df_1m['t'], unit='ms')
            df_1m = df_1m.set_index('datetime')

    # Aggregate to 5m
    df_5m = df_1m.resample("5min").agg({
        "o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"
    }).dropna()

    # Aggregate higher timeframes
    df_15m = df_1m.resample("15min").agg({
        "o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"
    }).dropna()
    df_1h = df_1m.resample("1h").agg({
        "o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"
    }).dropna()

    # Initialize strategies
    detector = RegimeDetector()
    trend_strategy = TrendFollowingStrategy()
    reversion_strategy = MeanReversionStrategy()

    from strategies.enhanced_signal import EnhancedSignal
    enhanced_strict = EnhancedSignal(
        min_confluence=0.3,
        min_mtf_alignment=0.3,
        require_mtf_trade=True,
    )

    backtester = StrategyBacktester(symbol=symbol)
    # Wider stops to let directional edge play out
    backtester_wide = StrategyBacktester(symbol=symbol, sl_mult=3.0, tp_mult=2.0, max_hold_bars=90)
    # Time-based exit (30 bars) — tests pure directional edge
    bt_time30 = TimeExitBacktester(symbol=symbol, exit_bars=30)
    bt_time60 = TimeExitBacktester(symbol=symbol, exit_bars=60)
    # Trailing stop + time exit
    bt_trail30 = TrailingExitBacktester(symbol=symbol, exit_bars=30, trail_atr_mult=2.0)
    bt_trail60 = TrailingExitBacktester(symbol=symbol, exit_bars=60, trail_atr_mult=2.0)

    results = {}

    # Test each strategy — focused on most promising configurations
    strategies = [
        ("Legacy", create_legacy_signal, 0.3, backtester),
        ("Enhanced-Strict", create_enhanced_strict_signal(enhanced_strict), 0.2, backtester),
        ("Reg+Time30", create_regime_filtered_signal(enhanced_strict, detector), 0.2, bt_time30),
        ("Reg+Time60", create_regime_filtered_signal(enhanced_strict, detector), 0.2, bt_time60),
        ("Reg+Trail30", create_regime_filtered_signal(enhanced_strict, detector), 0.2, bt_trail30),
        ("Reg+Trail60", create_regime_filtered_signal(enhanced_strict, detector), 0.2, bt_trail60),
        ("Reg+Time30hi", create_regime_filtered_signal(enhanced_strict, detector), 0.3, bt_time30),
        ("Reg+Trail30hi", create_regime_filtered_signal(enhanced_strict, detector), 0.3, bt_trail30),
    ]

    for name, signal_func, min_conf, bt in strategies:
        print(f"\nTesting {name}...")
        result = bt.run(name, df_1m, df_5m, signal_func, min_confidence=min_conf)
        results[name] = result
        print(f"  Trades: {result.total_trades}, WR: {result.win_rate * 100:.1f}%, PF: {result.profit_factor:.2f}")

    return results


def print_comparison(results: Dict[str, StrategyResult]):
    """Print formatted comparison table."""
    print("\n" + "=" * 95)
    print("COMPREHENSIVE BACKTEST RESULTS")
    print("=" * 95)
    print(f"{'Strategy':<18} {'Trades':>8} {'WinRate':>10} {'PF':>8} {'Return':>10} {'MaxDD':>8} {'Sharpe':>8} {'Equity':>12}")
    print("-" * 95)

    for name, r in sorted(results.items(), key=lambda x: -x[1].profit_factor):
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor < 100 else "inf"
        print(
            f"{name:<18} "
            f"{r.total_trades:>8} "
            f"{r.win_rate * 100:>9.1f}% "
            f"{pf_str:>8} "
            f"{r.total_return_pct * 100:>9.1f}% "
            f"{r.max_drawdown_pct * 100:>7.1f}% "
            f"{r.sharpe_ratio:>8.2f} "
            f"${r.final_equity:>10,.2f}"
        )

    print("-" * 95)

    # Find best
    best = max(results.values(), key=lambda x: x.profit_factor)
    print(f"\nBEST STRATEGY: {best.name}")
    print(f"  Profit Factor: {best.profit_factor:.2f}")
    print(f"  Win Rate: {best.win_rate * 100:.1f}%")
    print(f"  Total Return: {best.total_return_pct * 100:.1f}%")

    # Check if any profitable
    profitable = [r for r in results.values() if r.profit_factor > 1.0]
    if profitable:
        print(f"\nPROFITABLE STRATEGIES: {len(profitable)}")
        for r in profitable:
            print(f"  - {r.name}: PF={r.profit_factor:.2f}")
    else:
        print("\n** NO PROFITABLE STRATEGIES FOUND **")
        print("The signal quality needs fundamental improvement.")

    print("=" * 95)


def main():
    """Run comprehensive backtest."""
    parser = argparse.ArgumentParser(description="Comprehensive strategy backtest")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--days", type=int, default=30, help="Days of data")
    args = parser.parse_args()

    results = run_comprehensive_backtest(args.symbol, args.days)
    print_comparison(results)

    # Also test ETH
    if args.days >= 14:
        print("\n\nTesting ETHUSDT...")
        eth_results = run_comprehensive_backtest("ETHUSDT", args.days)
        print_comparison(eth_results)

    return results


if __name__ == "__main__":
    main()
