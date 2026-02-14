# strategies/sltp_optimizer.py — SCALPER ETERNAL — SL/TP OPTIMIZER — 2026 v1.0
# Test different stop-loss and take-profit ATR multipliers.
#
# Purpose:
# - Find optimal SL/TP ratio for profitability
# - Test different risk-reward combinations
# - Generate recommendations based on backtest evidence
#
# Usage:
#     python -m strategies.sltp_optimizer --symbol BTCUSDT --days 30

from __future__ import annotations

import argparse
import itertools
from typing import Dict, List, Tuple, Optional
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


@dataclass
class SLTPTestResult:
    """Result of testing a specific SL/TP combination."""
    sl_mult: float
    tp_mult: float
    ratio: str  # e.g., "1:2.0"
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_return_pct: float
    max_drawdown_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    expectancy: float  # (win_rate * avg_win) - (loss_rate * avg_loss)


class SLTPOptimizer:
    """Test different SL/TP combinations."""

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.loader = DataLoader()

    def run_single_test(
        self,
        df: pd.DataFrame,
        sl_mult: float,
        tp_mult: float,
        signal_func=None,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.01,
        fee_rate: float = 0.0004,
    ) -> SLTPTestResult:
        """
        Run backtest with specific SL/TP multipliers.

        Args:
            df: OHLCV DataFrame
            sl_mult: Stop-loss ATR multiplier
            tp_mult: Take-profit ATR multiplier
            signal_func: Optional custom signal function
            initial_capital: Starting capital
            risk_per_trade: Risk per trade as decimal
            fee_rate: Trading fee rate

        Returns:
            SLTPTestResult with performance metrics
        """
        ratio = f"1:{tp_mult / sl_mult:.1f}"

        trades = []
        trade_id = 0
        equity = initial_capital
        peak_equity = equity
        max_drawdown_pct = 0.0

        position = None
        atr_period = 14
        lookback = 100

        for i in range(lookback, len(df)):
            bar = df.iloc[i]
            current_time = bar.name if hasattr(bar.name, 'strftime') else datetime.now(timezone.utc)
            current_price = float(bar["c"])
            current_high = float(bar["h"])
            current_low = float(bar["l"])

            # Calculate ATR
            high = df["h"].iloc[i - atr_period:i]
            low = df["l"].iloc[i - atr_period:i]
            close_prev = df["c"].iloc[i - atr_period - 1:i - 1]

            tr = pd.concat([
                high - low,
                (high - close_prev).abs(),
                (low - close_prev).abs()
            ], axis=1).max(axis=1)
            atr = float(tr.mean()) if len(tr) > 0 else 0.0

            # Exit check
            if position is not None:
                exit_price = None
                exit_reason = ""

                if position["direction"] == TradeDirection.LONG:
                    if current_low <= position["stop_loss"]:
                        exit_price = position["stop_loss"]
                        exit_reason = "stop_loss"
                    elif current_high >= position["take_profit"]:
                        exit_price = position["take_profit"]
                        exit_reason = "take_profit"
                else:
                    if current_high >= position["stop_loss"]:
                        exit_price = position["stop_loss"]
                        exit_reason = "stop_loss"
                    elif current_low <= position["take_profit"]:
                        exit_price = position["take_profit"]
                        exit_reason = "take_profit"

                # Time exit (60 bars max)
                if exit_price is None:
                    bars_held = i - position["entry_bar"]
                    if bars_held >= 60:
                        exit_price = current_price
                        exit_reason = "time_exit"

                if exit_price is not None:
                    if position["direction"] == TradeDirection.LONG:
                        pnl = (exit_price - position["entry_price"]) * position["size"]
                    else:
                        pnl = (position["entry_price"] - exit_price) * position["size"]

                    fees = (position["entry_price"] + exit_price) * position["size"] * fee_rate
                    pnl -= fees

                    pnl_pct = pnl / (position["entry_price"] * position["size"])

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
                        pnl_pct=pnl_pct,
                        fees=fees,
                        exit_reason=exit_reason,
                    )
                    trades.append(trade)
                    trade_id += 1

                    equity += pnl
                    peak_equity = max(peak_equity, equity)
                    dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                    max_drawdown_pct = max(max_drawdown_pct, dd)

                    position = None

            # Entry signal
            if position is None and atr > 0 and equity > 0:
                # Use signal function or simple EMA crossover
                if signal_func:
                    is_long, is_short = signal_func(df, i)
                else:
                    is_long, is_short = self._simple_signal(df, i)

                if is_long or is_short:
                    slippage = current_price * 0.0001
                    entry_price = current_price + slippage if is_long else current_price - slippage

                    if is_long:
                        stop_loss = entry_price - (atr * sl_mult)
                        take_profit = entry_price + (atr * tp_mult)
                    else:
                        stop_loss = entry_price + (atr * sl_mult)
                        take_profit = entry_price - (atr * tp_mult)

                    risk_amount = equity * risk_per_trade
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

        # Calculate metrics
        total = len(trades)
        wins = [t for t in trades if t.is_winner]
        losses = [t for t in trades if not t.is_winner]

        winning = len(wins)
        losing = len(losses)
        win_rate = winning / total if total > 0 else 0.0

        gross_profit = sum(t.pnl for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
        pf = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)

        total_return = (equity - initial_capital) / initial_capital

        avg_win_pct = np.mean([t.pnl_pct for t in wins]) if wins else 0.0
        avg_loss_pct = np.mean([abs(t.pnl_pct) for t in losses]) if losses else 0.0

        # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        expectancy = (win_rate * avg_win_pct) - ((1 - win_rate) * avg_loss_pct)

        return SLTPTestResult(
            sl_mult=sl_mult,
            tp_mult=tp_mult,
            ratio=ratio,
            total_trades=total,
            winning_trades=winning,
            losing_trades=losing,
            win_rate=win_rate,
            profit_factor=pf,
            total_return_pct=total_return,
            max_drawdown_pct=max_drawdown_pct,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            expectancy=expectancy,
        )

    def _simple_signal(self, df: pd.DataFrame, i: int) -> Tuple[bool, bool]:
        """Simple EMA crossover signal for testing."""
        if i < 50:
            return False, False

        ema20 = df["c"].iloc[i - 20:i].ewm(span=20).mean().iloc[-1]
        ema50 = df["c"].iloc[i - 50:i].ewm(span=50).mean().iloc[-1]

        # Momentum filter
        mom = (df["c"].iloc[i] - df["c"].iloc[i - 5]) / df["c"].iloc[i - 5]

        is_long = ema20 > ema50 and mom > 0.001
        is_short = ema20 < ema50 and mom < -0.001

        return is_long, is_short

    def test_matrix(
        self,
        days: int = 30,
        sl_range: List[float] = None,
        tp_range: List[float] = None,
    ) -> List[SLTPTestResult]:
        """
        Test a matrix of SL/TP combinations.

        Args:
            days: Days of data
            sl_range: List of SL multipliers to test
            tp_range: List of TP multipliers to test

        Returns:
            List of results sorted by profit factor
        """
        if sl_range is None:
            sl_range = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        if tp_range is None:
            tp_range = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

        print(f"\nLoading {days} days of {self.symbol} 1m data...")
        df = self.loader.download_binance(self.symbol, "1m", days=days)

        if df.empty:
            raise ValueError("No data loaded")

        print(f"Loaded {len(df):,} bars")

        # Test all combinations
        results = []
        combinations = list(itertools.product(sl_range, tp_range))
        total = len(combinations)

        print(f"Testing {total} SL/TP combinations...")

        for idx, (sl, tp) in enumerate(combinations):
            if idx % 10 == 0:
                print(f"  Progress: {idx}/{total}...")
            result = self.run_single_test(df, sl, tp)
            results.append(result)

        # Sort by profit factor
        results.sort(key=lambda x: x.profit_factor, reverse=True)

        return results

    def print_results(self, results: List[SLTPTestResult], top_n: int = 20):
        """Print formatted results table."""
        print("\n" + "=" * 100)
        print("SL/TP OPTIMIZATION RESULTS")
        print("=" * 100)
        print(f"{'SL':>6} {'TP':>6} {'Ratio':>8} {'Trades':>8} {'WinRate':>10} {'PF':>8} {'Return':>10} {'AvgWin':>8} {'AvgLoss':>8} {'Expect':>8}")
        print("-" * 100)

        for r in results[:top_n]:
            pf_str = f"{r.profit_factor:.2f}" if r.profit_factor < 100 else "inf"
            print(
                f"{r.sl_mult:>6.2f} "
                f"{r.tp_mult:>6.2f} "
                f"{r.ratio:>8} "
                f"{r.total_trades:>8} "
                f"{r.win_rate * 100:>9.1f}% "
                f"{pf_str:>8} "
                f"{r.total_return_pct * 100:>9.1f}% "
                f"{r.avg_win_pct * 100:>7.2f}% "
                f"{r.avg_loss_pct * 100:>7.2f}% "
                f"{r.expectancy * 100:>7.3f}%"
            )

        print("-" * 100)

        # Best result analysis
        if results:
            best = results[0]
            print(f"\nBEST CONFIGURATION:")
            print(f"  SL Multiplier: {best.sl_mult:.2f}x ATR")
            print(f"  TP Multiplier: {best.tp_mult:.2f}x ATR")
            print(f"  Risk:Reward Ratio: {best.ratio}")
            print(f"  Win Rate: {best.win_rate * 100:.1f}%")
            print(f"  Profit Factor: {best.profit_factor:.2f}")
            print(f"  Total Return: {best.total_return_pct * 100:.1f}%")
            print(f"  Expectancy: {best.expectancy * 100:.3f}% per trade")

            # Find profitable configs
            profitable = [r for r in results if r.profit_factor > 1.0]
            if profitable:
                print(f"\nPROFITABLE CONFIGURATIONS: {len(profitable)}")
                for r in profitable[:5]:
                    print(f"  SL={r.sl_mult:.2f}, TP={r.tp_mult:.2f} -> PF={r.profit_factor:.2f}, WR={r.win_rate * 100:.1f}%")
            else:
                print("\n** NO PROFITABLE CONFIGURATIONS FOUND **")
                print("This suggests the signal logic itself needs improvement.")

        print("=" * 100)


def main():
    """Run SL/TP optimization."""
    parser = argparse.ArgumentParser(description="Optimize SL/TP ratios")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--days", type=int, default=30, help="Days of data")
    parser.add_argument("--fast", action="store_true", help="Fast mode with fewer combinations")
    args = parser.parse_args()

    optimizer = SLTPOptimizer(symbol=args.symbol)

    if args.fast:
        # Fast mode: test key combinations only
        sl_range = [0.5, 1.0, 1.5, 2.0]
        tp_range = [1.0, 1.5, 2.0, 3.0]
    else:
        # Full mode
        sl_range = [0.5, 0.75, 1.0, 1.5, 2.0]
        tp_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    results = optimizer.test_matrix(
        days=args.days,
        sl_range=sl_range,
        tp_range=tp_range,
    )

    optimizer.print_results(results)
    return results


if __name__ == "__main__":
    main()
