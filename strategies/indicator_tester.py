# strategies/indicator_tester.py — SCALPER ETERNAL — INDIVIDUAL INDICATOR TESTER — 2026 v1.0
# Test each indicator independently to find which ones have predictive value.
#
# Purpose:
# - Run backtest using only ONE indicator for signals
# - Rank indicators by standalone win rate and profit factor
# - Generate weight recommendations based on actual performance
#
# Usage:
#     python -m strategies.indicator_tester --symbol BTCUSDT --days 30

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Import backtester components
from strategies.backtester import (
    DataLoader,
    BacktestConfig,
    BacktestMetrics,
    BacktestResult,
    Trade,
    TradeDirection,
)

# Import indicators
from strategies.indicators import (
    SignalDirection,
    SignalStrength,
    IndicatorSignal,
)


@dataclass
class IndicatorTestResult:
    """Result of testing a single indicator."""
    name: str
    weight: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_confidence: float
    signal_count: int
    signal_rate: float  # % of bars with non-NEUTRAL signal


@dataclass
class IndicatorRanking:
    """Ranking of all indicators by profitability."""
    rankings: List[IndicatorTestResult]
    recommended_weights: Dict[str, float]
    analysis: str


class SingleIndicatorBacktester:
    """Run backtest using only one indicator for signals."""

    def __init__(
        self,
        config: BacktestConfig,
        min_confidence: float = 0.3,
    ):
        self.config = config
        self.min_confidence = min_confidence

    def run(
        self,
        indicator,
        df_1m: pd.DataFrame,
        df_5m: Optional[pd.DataFrame] = None,
        name_override: str = "",
    ) -> IndicatorTestResult:
        """
        Run backtest with a single indicator.

        Args:
            indicator: Indicator instance with calculate() method
            df_1m: 1-minute OHLCV data
            df_5m: 5-minute data (optional)
            name_override: Override indicator name

        Returns:
            IndicatorTestResult with performance metrics
        """
        name = name_override or indicator.name
        weight = getattr(indicator, "weight", 1.0)

        # Track signals
        signals = []
        confidences = []

        # Track trades
        trades = []
        trade_id = 0

        # Capital tracking
        equity = self.config.initial_capital
        peak_equity = equity
        max_drawdown_pct = 0.0

        # Current position
        position = None

        # Calculate ATR for stops
        atr_period = 14
        lookback = max(100, atr_period * 2)

        for i in range(lookback, len(df_1m)):
            # Get current bar data
            bar = df_1m.iloc[i]
            current_time = bar.name if hasattr(bar.name, 'strftime') else datetime.now(timezone.utc)
            current_price = float(bar["c"])
            current_high = float(bar["h"])
            current_low = float(bar["l"])

            # Calculate ATR
            high = df_1m["h"].iloc[i - atr_period:i]
            low = df_1m["l"].iloc[i - atr_period:i]
            close_prev = df_1m["c"].iloc[i - atr_period - 1:i - 1]

            tr = pd.concat([
                high - low,
                (high - close_prev).abs(),
                (low - close_prev).abs()
            ], axis=1).max(axis=1)
            atr = float(tr.mean()) if len(tr) > 0 else 0.0

            # Exit check if in position
            if position is not None:
                exit_price = None
                exit_reason = ""

                # Check stop loss
                if position["direction"] == TradeDirection.LONG:
                    if current_low <= position["stop_loss"]:
                        exit_price = position["stop_loss"]
                        exit_reason = "stop_loss"
                    elif current_high >= position["take_profit"]:
                        exit_price = position["take_profit"]
                        exit_reason = "take_profit"
                else:  # SHORT
                    if current_high >= position["stop_loss"]:
                        exit_price = position["stop_loss"]
                        exit_reason = "stop_loss"
                    elif current_low <= position["take_profit"]:
                        exit_price = position["take_profit"]
                        exit_reason = "take_profit"

                # Time-based exit
                if exit_price is None:
                    bars_held = i - position["entry_bar"]
                    if bars_held >= self.config.max_hold_bars:
                        exit_price = current_price
                        exit_reason = "time_exit"

                # Close position
                if exit_price is not None:
                    if position["direction"] == TradeDirection.LONG:
                        pnl = (exit_price - position["entry_price"]) * position["size"]
                    else:
                        pnl = (position["entry_price"] - exit_price) * position["size"]

                    # Apply fees
                    fees = (position["entry_price"] + exit_price) * position["size"] * self.config.fee_rate
                    pnl -= fees

                    trade = Trade(
                        id=trade_id,
                        symbol=self.config.symbol,
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
                        fees=fees,
                        exit_reason=exit_reason,
                        confidence=position["confidence"],
                        strategy=name,
                    )
                    trades.append(trade)
                    trade_id += 1

                    equity += pnl
                    peak_equity = max(peak_equity, equity)
                    dd_pct = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                    max_drawdown_pct = max(max_drawdown_pct, dd_pct)

                    position = None

            # Generate signal if no position
            if position is None and atr > 0 and equity > 0:
                # Get indicator signal
                df_window = df_1m.iloc[max(0, i - 300):i + 1].copy()

                try:
                    signal = indicator.calculate(df_window)
                except Exception:
                    signal = IndicatorSignal(name=name)

                # Track signal stats
                if signal.direction != SignalDirection.NEUTRAL:
                    signals.append(signal.direction)
                    confidences.append(signal.confidence)

                # Check for entry
                if signal.direction != SignalDirection.NEUTRAL and signal.confidence >= self.min_confidence:
                    # Determine direction
                    is_long = signal.direction == SignalDirection.LONG

                    # Entry with slippage
                    slippage = current_price * self.config.slippage_pct
                    entry_price = current_price + slippage if is_long else current_price - slippage

                    # Calculate stops
                    if is_long:
                        stop_loss = entry_price - (atr * self.config.stop_atr_mult)
                        take_profit = entry_price + (atr * self.config.tp_atr_mult)
                    else:
                        stop_loss = entry_price + (atr * self.config.stop_atr_mult)
                        take_profit = entry_price - (atr * self.config.tp_atr_mult)

                    # Position size based on risk
                    risk_amount = equity * self.config.risk_per_trade
                    stop_distance = abs(entry_price - stop_loss)
                    size = risk_amount / stop_distance if stop_distance > 0 else 0

                    # Max leverage limit
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
                            "confidence": signal.confidence,
                        }

        # Calculate metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.is_winner)
        losing_trades = total_trades - winning_trades

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Profit factor
        wins = [t.pnl for t in trades if t.is_winner]
        losses = [abs(t.pnl) for t in trades if not t.is_winner]
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = sum(losses) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)

        # Returns
        total_return_pct = (equity - self.config.initial_capital) / self.config.initial_capital

        # Sharpe ratio (simplified)
        if trades:
            returns = [t.pnl_pct for t in trades]
            mean_return = np.mean(returns) if returns else 0.0
            std_return = np.std(returns) if len(returns) > 1 else 0.0
            sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0
        else:
            sharpe = 0.0

        # Signal stats
        signal_count = len(signals)
        signal_rate = signal_count / (len(df_1m) - lookback) if len(df_1m) > lookback else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return IndicatorTestResult(
            name=name,
            weight=weight,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe,
            avg_confidence=avg_confidence,
            signal_count=signal_count,
            signal_rate=signal_rate,
        )


class IndicatorTester:
    """Test all indicators and rank by profitability."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.loader = DataLoader()

    def load_indicators(self) -> Dict[str, Any]:
        """Load all available indicators."""
        indicators = {}

        # Try to load each indicator
        try:
            from strategies.indicators.adx import ADXIndicator
            indicators["ADX"] = ADXIndicator()
        except ImportError:
            pass

        try:
            from strategies.indicators.macd import MACDIndicator
            indicators["MACD"] = MACDIndicator()
        except ImportError:
            pass

        try:
            from strategies.indicators.bollinger import BollingerIndicator
            indicators["Bollinger"] = BollingerIndicator()
        except ImportError:
            pass

        try:
            from strategies.indicators.rsi_enhanced import RSIEnhancedIndicator
            indicators["RSI"] = RSIEnhancedIndicator()
        except ImportError:
            pass

        try:
            from strategies.indicators.stochastic import StochasticIndicator
            indicators["Stochastic"] = StochasticIndicator()
        except ImportError:
            pass

        try:
            from strategies.indicators.momentum import MomentumIndicator
            indicators["Momentum"] = MomentumIndicator()
        except ImportError:
            pass

        try:
            from strategies.indicators.vwap_enhanced import VWAPEnhancedIndicator
            indicators["VWAP"] = VWAPEnhancedIndicator()
        except ImportError:
            pass

        return indicators

    def test_all(
        self,
        symbol: str = "BTCUSDT",
        days: int = 30,
        min_confidence: float = 0.3,
    ) -> IndicatorRanking:
        """
        Test all indicators independently.

        Args:
            symbol: Trading symbol
            days: Days of historical data
            min_confidence: Minimum confidence for entry

        Returns:
            IndicatorRanking with all results
        """
        print(f"\nLoading {days} days of {symbol} 1m data...")
        df = self.loader.download_binance(symbol, "1m", days=days)

        if df.empty:
            raise ValueError("No data loaded")

        print(f"Loaded {len(df):,} bars")

        # Load indicators
        indicators = self.load_indicators()
        print(f"Testing {len(indicators)} indicators: {list(indicators.keys())}")

        # Test each indicator
        results = []
        config = BacktestConfig(symbol=symbol)
        backtester = SingleIndicatorBacktester(config, min_confidence=min_confidence)

        for name, indicator in indicators.items():
            print(f"\nTesting {name}...")
            result = backtester.run(indicator, df, name_override=name)
            results.append(result)
            print(f"  Trades: {result.total_trades}, WR: {result.win_rate * 100:.1f}%, PF: {result.profit_factor:.2f}")

        # Sort by profit factor (>1 is profitable)
        results.sort(key=lambda x: x.profit_factor, reverse=True)

        # Generate weight recommendations
        recommended_weights = self._generate_weights(results)

        # Generate analysis text
        analysis = self._generate_analysis(results)

        return IndicatorRanking(
            rankings=results,
            recommended_weights=recommended_weights,
            analysis=analysis,
        )

    def _generate_weights(self, results: List[IndicatorTestResult]) -> Dict[str, float]:
        """Generate weight recommendations based on performance."""
        weights = {}

        for r in results:
            if r.profit_factor >= 1.5:
                # Very profitable - high weight
                weights[r.name] = 2.0
            elif r.profit_factor >= 1.2:
                # Profitable - above average weight
                weights[r.name] = 1.5
            elif r.profit_factor >= 1.0:
                # Break-even - normal weight
                weights[r.name] = 1.0
            elif r.profit_factor >= 0.8:
                # Slightly unprofitable - low weight
                weights[r.name] = 0.5
            else:
                # Unprofitable - very low weight or disable
                weights[r.name] = 0.2

        return weights

    def _generate_analysis(self, results: List[IndicatorTestResult]) -> str:
        """Generate text analysis of results."""
        lines = []
        lines.append("INDICATOR PERFORMANCE ANALYSIS")
        lines.append("=" * 60)

        profitable = [r for r in results if r.profit_factor > 1.0]
        unprofitable = [r for r in results if r.profit_factor <= 1.0]

        if profitable:
            lines.append(f"\nPROFITABLE ({len(profitable)}):")
            for r in profitable:
                lines.append(f"  {r.name}: PF={r.profit_factor:.2f}, WR={r.win_rate * 100:.1f}%")
        else:
            lines.append("\nNo profitable indicators found!")

        if unprofitable:
            lines.append(f"\nUNPROFITABLE ({len(unprofitable)}):")
            for r in unprofitable:
                lines.append(f"  {r.name}: PF={r.profit_factor:.2f}, WR={r.win_rate * 100:.1f}%")

        # Best and worst
        if results:
            best = results[0]
            worst = results[-1]
            lines.append(f"\nBEST: {best.name} (PF={best.profit_factor:.2f})")
            lines.append(f"WORST: {worst.name} (PF={worst.profit_factor:.2f})")

        return "\n".join(lines)

    def print_ranking(self, ranking: IndicatorRanking):
        """Print formatted ranking table."""
        print("\n" + "=" * 90)
        print("INDIVIDUAL INDICATOR BACKTEST RESULTS")
        print("=" * 90)
        print(f"{'Indicator':<12} {'Trades':>8} {'WinRate':>10} {'PF':>8} {'Return':>10} {'MaxDD':>8} {'Sharpe':>8} {'SigRate':>8}")
        print("-" * 90)

        for r in ranking.rankings:
            pf_str = f"{r.profit_factor:.2f}" if r.profit_factor < 100 else "inf"
            print(
                f"{r.name:<12} "
                f"{r.total_trades:>8} "
                f"{r.win_rate * 100:>9.1f}% "
                f"{pf_str:>8} "
                f"{r.total_return_pct * 100:>9.1f}% "
                f"{r.max_drawdown_pct * 100:>7.1f}% "
                f"{r.sharpe_ratio:>8.2f} "
                f"{r.signal_rate * 100:>7.1f}%"
            )

        print("-" * 90)

        print("\nRECOMMENDED WEIGHTS:")
        print("-" * 40)
        for name, weight in sorted(ranking.recommended_weights.items(), key=lambda x: -x[1]):
            print(f"  {name:<12}: {weight:.1f}")

        print("\n" + ranking.analysis)
        print("=" * 90)


def main():
    """Run individual indicator testing."""
    parser = argparse.ArgumentParser(description="Test individual indicator performance")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--days", type=int, default=30, help="Days of data")
    parser.add_argument("--min-confidence", type=float, default=0.3, help="Minimum signal confidence")
    args = parser.parse_args()

    tester = IndicatorTester()

    try:
        ranking = tester.test_all(
            symbol=args.symbol,
            days=args.days,
            min_confidence=args.min_confidence,
        )
        tester.print_ranking(ranking)
        return ranking
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
