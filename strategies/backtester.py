# strategies/backtester.py — SCALPER ETERNAL — BACKTESTING FRAMEWORK — 2026 v1.0
# Comprehensive backtesting for comparing legacy vs enhanced signal strategies.
#
# Features:
# - Download historical klines from Binance REST API
# - Load data from CSV files
# - Multi-timeframe data aggregation
# - Simulate trades with ATR-based stops
# - Calculate performance metrics (win rate, profit factor, Sharpe, drawdown)
# - Compare baseline vs enhanced strategy
# - Generate detailed reports
#
# Usage:
#     python -m strategies.backtester --symbol BTCUSDT --days 30 --compare
#
# Or programmatically:
#     from strategies.backtester import Backtester
#     bt = Backtester()
#     results = bt.run("BTCUSDT", days=30)
#     bt.print_report(results)

from __future__ import annotations

import os
import sys
import json
import time
import csv
import argparse
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd

# Binance API
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

# TA library for ATR
try:
    from ta.volatility import AverageTrueRange
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False


class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Trade:
    """Single trade record."""
    id: int
    symbol: str
    direction: TradeDirection
    entry_price: float
    entry_time: datetime
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    size: float = 1.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    exit_reason: str = ""
    confidence: float = 0.0
    strategy: str = "legacy"

    @property
    def duration_seconds(self) -> float:
        if self.exit_time and self.entry_time:
            return (self.exit_time - self.entry_time).total_seconds()
        return 0.0

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    symbol: str = "BTCUSDT"
    timeframe: str = "1m"
    initial_capital: float = 10000.0
    risk_per_trade: float = 0.01  # 1% risk per trade
    max_positions: int = 1
    stop_atr_mult: float = 1.5
    tp_atr_mult: float = 2.5
    fee_rate: float = 0.0004  # 0.04% taker fee
    slippage_pct: float = 0.0001  # 0.01% slippage
    min_confidence: float = 0.5
    max_hold_bars: int = 60  # Max bars to hold position
    use_enhanced: bool = True
    use_legacy: bool = True


@dataclass
class BacktestMetrics:
    """Performance metrics for a backtest run."""
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    total_bars: int

    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    long_trades: int = 0
    short_trades: int = 0

    # Returns
    total_return: float = 0.0
    total_return_pct: float = 0.0
    avg_trade_return: float = 0.0
    avg_trade_return_pct: float = 0.0

    # Win/Loss metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Time metrics
    avg_trade_duration_min: float = 0.0
    avg_bars_in_trade: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Equity
    final_equity: float = 0.0
    peak_equity: float = 0.0

    # Fees
    total_fees: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest result."""
    config: BacktestConfig
    metrics: BacktestMetrics
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)


class DataLoader:
    """Load historical kline data from Binance or CSV."""

    BINANCE_API = "https://api.binance.com/api/v3/klines"
    CACHE_DIR = Path("data/backtest_cache")

    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        if cache_enabled:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> Path:
        """Get cache file path for given parameters."""
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        return self.CACHE_DIR / f"{symbol}_{timeframe}_{start_str}_{end_str}.csv"

    def download_binance(
        self,
        symbol: str,
        timeframe: str = "1m",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Download kline data from Binance.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            timeframe: Candle interval (1m, 5m, 15m, 1h, etc.)
            start: Start datetime (UTC)
            end: End datetime (UTC)
            days: Number of days to download (if start/end not specified)

        Returns:
            DataFrame with columns: ts, o, h, l, c, v
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library not available. Install with: pip install requests")

        # Set default date range
        if end is None:
            end = datetime.now(timezone.utc)
        if start is None:
            start = end - timedelta(days=days)

        # Check cache
        cache_path = self._get_cache_path(symbol, timeframe, start, end)
        if self.cache_enabled and cache_path.exists():
            print(f"Loading cached data from {cache_path}")
            return self.load_csv(str(cache_path))

        print(f"Downloading {symbol} {timeframe} data from {start.date()} to {end.date()}...")

        # Convert to milliseconds
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        all_data = []
        current_start = start_ms

        # Binance limit is 1000 candles per request
        limit = 1000

        while current_start < end_ms:
            params = {
                "symbol": symbol.upper(),
                "interval": timeframe,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": limit,
            }

            try:
                response = requests.get(self.BINANCE_API, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if not data:
                    break

                all_data.extend(data)

                # Move to next batch
                last_ts = data[-1][0]
                current_start = last_ts + 1

                # Rate limiting
                time.sleep(0.1)

                print(f"  Downloaded {len(all_data)} candles...")

            except Exception as e:
                print(f"  Error downloading: {e}")
                break

        if not all_data:
            raise ValueError(f"No data downloaded for {symbol}")

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            "ts", "o", "h", "l", "c", "v",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])

        # Keep only needed columns and convert types
        df = df[["ts", "o", "h", "l", "c", "v"]].copy()
        df["ts"] = df["ts"].astype(int)
        for col in ["o", "h", "l", "c", "v"]:
            df[col] = df[col].astype(float)

        df = df.sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)

        # Cache the data
        if self.cache_enabled:
            df.to_csv(cache_path, index=False)
            print(f"  Cached to {cache_path}")

        print(f"  Total: {len(df)} candles")
        return df

    def load_csv(self, path: str) -> pd.DataFrame:
        """
        Load kline data from CSV file.

        Expected columns: ts (or timestamp), o (or open), h (or high),
                         l (or low), c (or close), v (or volume)
        """
        df = pd.read_csv(path)

        # Normalize column names
        col_map = {
            "timestamp": "ts", "time": "ts", "date": "ts",
            "open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Ensure required columns
        required = ["ts", "o", "h", "l", "c", "v"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Convert types
        df["ts"] = df["ts"].astype(int)
        for col in ["o", "h", "l", "c", "v"]:
            df[col] = df[col].astype(float)

        return df.sort_values("ts").reset_index(drop=True)

    def aggregate_timeframe(self, df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """
        Aggregate 1m data to higher timeframe.

        Args:
            df: 1-minute OHLCV DataFrame
            target_tf: Target timeframe (5m, 15m, 1h)

        Returns:
            Aggregated DataFrame
        """
        tf_minutes = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440
        }

        minutes = tf_minutes.get(target_tf, 1)
        if minutes <= 1:
            return df.copy()

        # Create time buckets
        bucket = (df["ts"] // (minutes * 60_000)) * (minutes * 60_000)

        agg = df.groupby(bucket).agg({
            "ts": "last",
            "o": "first",
            "h": "max",
            "l": "min",
            "c": "last",
            "v": "sum"
        }).reset_index(drop=True)

        return agg


class SignalGenerator:
    """Generate trading signals using legacy or enhanced strategy."""

    def __init__(self, use_enhanced: bool = True):
        self.use_enhanced = use_enhanced
        self._legacy_available = False
        self._enhanced_available = False

        # Try to import strategies
        try:
            from strategies.eclipse_scalper import scalper_signal
            self._scalper_signal = scalper_signal
            self._legacy_available = True
        except ImportError:
            self._scalper_signal = None

        try:
            from strategies.enhanced_signal import enhanced_signal_check
            self._enhanced_signal = enhanced_signal_check
            self._enhanced_available = True
        except ImportError:
            self._enhanced_signal = None

    def generate(
        self,
        df_1m: pd.DataFrame,
        df_5m: Optional[pd.DataFrame] = None,
        df_15m: Optional[pd.DataFrame] = None,
        df_1h: Optional[pd.DataFrame] = None,
        symbol: str = "",
        use_enhanced: Optional[bool] = None,
    ) -> Tuple[bool, bool, float]:
        """
        Generate trading signal.

        Returns: (long_signal, short_signal, confidence)
        """
        use_enh = use_enhanced if use_enhanced is not None else self.use_enhanced

        if use_enh and self._enhanced_available:
            try:
                return self._enhanced_signal(
                    df_1m=df_1m,
                    df_5m=df_5m,
                    df_15m=df_15m,
                    df_1h=df_1h,
                    symbol=symbol,
                )
            except Exception:
                pass

        # Fallback to simple momentum-based signal for backtest
        return self._simple_signal(df_1m)

    def _simple_signal(self, df: pd.DataFrame) -> Tuple[bool, bool, float]:
        """Simple momentum-based signal for backtesting when strategies unavailable."""
        if df is None or len(df) < 50:
            return False, False, 0.0

        try:
            close = df["c"]

            # EMAs
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()

            c = float(close.iloc[-1])
            ema20_curr = float(ema20.iloc[-1])
            ema50_curr = float(ema50.iloc[-1])

            # Momentum
            mom = close.pct_change(5).iloc[-1]

            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            rsi_curr = float(rsi.iloc[-1])

            # Signals
            long_signal = (
                c > ema20_curr > ema50_curr and
                mom > 0.001 and
                rsi_curr < 70 and rsi_curr > 30
            )

            short_signal = (
                c < ema20_curr < ema50_curr and
                mom < -0.001 and
                rsi_curr > 30 and rsi_curr < 70
            )

            # Confidence based on alignment
            confidence = 0.0
            if long_signal or short_signal:
                conf_factors = []
                conf_factors.append(0.3)  # Base
                if abs(mom) > 0.002:
                    conf_factors.append(0.2)
                if (long_signal and rsi_curr < 50) or (short_signal and rsi_curr > 50):
                    conf_factors.append(0.2)
                if abs(ema20_curr - ema50_curr) / ema50_curr > 0.002:
                    conf_factors.append(0.2)
                confidence = sum(conf_factors)

            return long_signal, short_signal, min(1.0, confidence)

        except Exception:
            return False, False, 0.0


class BacktestEngine:
    """Core backtesting simulation engine."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.signal_gen = SignalGenerator(use_enhanced=config.use_enhanced)
        self.loader = DataLoader()

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        if TA_AVAILABLE:
            atr = AverageTrueRange(df["h"], df["l"], df["c"], window=period)
            return atr.average_true_range()
        else:
            # Manual ATR calculation
            high = df["h"]
            low = df["l"]
            close = df["c"]

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            return tr.rolling(period).mean()

    def _calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
    ) -> float:
        """Calculate position size based on risk."""
        risk_amount = equity * self.config.risk_per_trade
        stop_distance = abs(entry_price - stop_price)

        if stop_distance <= 0:
            return 0.0

        # Position size in base currency
        size = risk_amount / stop_distance

        # Cap at reasonable leverage
        max_size = (equity * 5) / entry_price  # Max 5x leverage
        return min(size, max_size)

    def run(
        self,
        df_1m: pd.DataFrame,
        df_5m: Optional[pd.DataFrame] = None,
        df_15m: Optional[pd.DataFrame] = None,
        df_1h: Optional[pd.DataFrame] = None,
        strategy_name: str = "backtest",
    ) -> BacktestResult:
        """
        Run backtest simulation.

        Args:
            df_1m: 1-minute OHLCV data
            df_5m: 5-minute data (optional)
            df_15m: 15-minute data (optional)
            df_1h: 1-hour data (optional)
            strategy_name: Name for this strategy

        Returns:
            BacktestResult with trades and metrics
        """
        # Initialize
        equity = self.config.initial_capital
        peak_equity = equity
        position: Optional[Trade] = None
        trades: List[Trade] = []
        equity_curve: List[Tuple[datetime, float]] = []
        trade_id = 0

        # Calculate ATR for the entire dataset
        atr_series = self._calculate_atr(df_1m)

        # Minimum lookback for indicators
        lookback = 100

        print(f"Running backtest: {strategy_name} on {self.config.symbol}...")
        print(f"  Bars: {len(df_1m)}, Starting capital: ${self.config.initial_capital:,.2f}")

        # Main simulation loop
        for i in range(lookback, len(df_1m)):
            # Current bar data
            row = df_1m.iloc[i]
            ts = int(row["ts"])
            current_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            current_price = float(row["c"])
            current_high = float(row["h"])
            current_low = float(row["l"])
            atr = float(atr_series.iloc[i]) if i < len(atr_series) and not np.isnan(atr_series.iloc[i]) else 0.0

            # Get historical data for signal generation
            hist_1m = df_1m.iloc[max(0, i-lookback):i+1].copy()
            hist_5m = df_5m.iloc[:i//5+1].copy() if df_5m is not None and len(df_5m) > i//5 else None
            hist_15m = df_15m.iloc[:i//15+1].copy() if df_15m is not None and len(df_15m) > i//15 else None
            hist_1h = df_1h.iloc[:i//60+1].copy() if df_1h is not None and len(df_1h) > i//60 else None

            # Check existing position
            if position is not None:
                bars_held = i - position.id

                # Check stop loss
                stop_hit = False
                tp_hit = False

                if position.direction == TradeDirection.LONG:
                    if current_low <= position.stop_loss:
                        stop_hit = True
                        exit_price = position.stop_loss
                    elif current_high >= position.take_profit:
                        tp_hit = True
                        exit_price = position.take_profit
                else:  # SHORT
                    if current_high >= position.stop_loss:
                        stop_hit = True
                        exit_price = position.stop_loss
                    elif current_low <= position.take_profit:
                        tp_hit = True
                        exit_price = position.take_profit

                # Check max hold time
                time_exit = bars_held >= self.config.max_hold_bars

                # Exit position
                if stop_hit or tp_hit or time_exit:
                    if time_exit and not (stop_hit or tp_hit):
                        exit_price = current_price

                    # Apply slippage
                    if position.direction == TradeDirection.LONG:
                        exit_price *= (1 - self.config.slippage_pct)
                    else:
                        exit_price *= (1 + self.config.slippage_pct)

                    # Calculate PnL
                    if position.direction == TradeDirection.LONG:
                        pnl = (exit_price - position.entry_price) * position.size
                    else:
                        pnl = (position.entry_price - exit_price) * position.size

                    # Fees
                    fees = (position.entry_price + exit_price) * position.size * self.config.fee_rate
                    pnl -= fees

                    # Update trade
                    position.exit_price = exit_price
                    position.exit_time = current_time
                    position.pnl = pnl
                    position.pnl_pct = pnl / (position.entry_price * position.size)
                    position.fees = fees
                    position.exit_reason = "stop_loss" if stop_hit else ("take_profit" if tp_hit else "time_exit")

                    trades.append(position)
                    equity += pnl
                    peak_equity = max(peak_equity, equity)
                    position = None

            # Generate signal if no position
            if position is None and atr > 0:
                long_sig, short_sig, confidence = self.signal_gen.generate(
                    df_1m=hist_1m,
                    df_5m=hist_5m,
                    df_15m=hist_15m,
                    df_1h=hist_1h,
                    symbol=self.config.symbol,
                    use_enhanced=self.config.use_enhanced,
                )

                if confidence >= self.config.min_confidence and (long_sig or short_sig):
                    direction = TradeDirection.LONG if long_sig else TradeDirection.SHORT

                    # Calculate entry with slippage
                    if direction == TradeDirection.LONG:
                        entry_price = current_price * (1 + self.config.slippage_pct)
                        stop_loss = entry_price - (atr * self.config.stop_atr_mult)
                        take_profit = entry_price + (atr * self.config.tp_atr_mult)
                    else:
                        entry_price = current_price * (1 - self.config.slippage_pct)
                        stop_loss = entry_price + (atr * self.config.stop_atr_mult)
                        take_profit = entry_price - (atr * self.config.tp_atr_mult)

                    # Calculate position size
                    size = self._calculate_position_size(equity, entry_price, stop_loss)

                    if size > 0:
                        trade_id += 1
                        position = Trade(
                            id=i,
                            symbol=self.config.symbol,
                            direction=direction,
                            entry_price=entry_price,
                            entry_time=current_time,
                            size=size,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            confidence=confidence,
                            strategy=strategy_name,
                        )

            # Record equity curve (every 60 bars = 1 hour)
            if i % 60 == 0:
                # Mark-to-market if in position
                mtm_equity = equity
                if position is not None:
                    if position.direction == TradeDirection.LONG:
                        unrealized = (current_price - position.entry_price) * position.size
                    else:
                        unrealized = (position.entry_price - current_price) * position.size
                    mtm_equity += unrealized

                equity_curve.append((current_time, mtm_equity))

        # Close any remaining position at last price
        if position is not None:
            last_row = df_1m.iloc[-1]
            exit_price = float(last_row["c"])
            current_time = datetime.fromtimestamp(int(last_row["ts"]) / 1000, tz=timezone.utc)

            if position.direction == TradeDirection.LONG:
                pnl = (exit_price - position.entry_price) * position.size
            else:
                pnl = (position.entry_price - exit_price) * position.size

            fees = (position.entry_price + exit_price) * position.size * self.config.fee_rate
            pnl -= fees

            position.exit_price = exit_price
            position.exit_time = current_time
            position.pnl = pnl
            position.pnl_pct = pnl / (position.entry_price * position.size)
            position.fees = fees
            position.exit_reason = "end_of_data"

            trades.append(position)
            equity += pnl

        # Calculate metrics
        metrics = self._calculate_metrics(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=self.config.initial_capital,
            final_equity=equity,
            strategy_name=strategy_name,
            symbol=self.config.symbol,
            df=df_1m,
        )

        return BacktestResult(
            config=self.config,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
        )

    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: List[Tuple[datetime, float]],
        initial_capital: float,
        final_equity: float,
        strategy_name: str,
        symbol: str,
        df: pd.DataFrame,
    ) -> BacktestMetrics:
        """Calculate performance metrics."""
        # Basic counts
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.is_winner)
        losing_trades = total_trades - winning_trades
        long_trades = sum(1 for t in trades if t.direction == TradeDirection.LONG)
        short_trades = total_trades - long_trades

        # Returns
        total_return = final_equity - initial_capital
        total_return_pct = total_return / initial_capital if initial_capital > 0 else 0.0

        # Win/Loss metrics
        wins = [t.pnl for t in trades if t.is_winner]
        losses = [t.pnl for t in trades if not t.is_winner]

        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

        # Drawdown
        peak = initial_capital
        max_dd = 0.0
        max_dd_pct = 0.0
        peak_equity = initial_capital

        for _, eq in equity_curve:
            peak = max(peak, eq)
            dd = peak - eq
            dd_pct = dd / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
            peak_equity = max(peak_equity, eq)

        # Daily returns for Sharpe/Sortino
        daily_returns = []
        if len(equity_curve) >= 2:
            # Group by day
            daily_eq = {}
            for dt, eq in equity_curve:
                day = dt.date()
                daily_eq[day] = eq

            days = sorted(daily_eq.keys())
            for i in range(1, len(days)):
                prev_eq = daily_eq[days[i-1]]
                curr_eq = daily_eq[days[i]]
                if prev_eq > 0:
                    daily_returns.append((curr_eq - prev_eq) / prev_eq)

        # Sharpe ratio (annualized, assuming 365 trading days for crypto)
        if daily_returns:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            sharpe = (mean_return * np.sqrt(365)) / std_return if std_return > 0 else 0.0

            # Sortino (downside deviation)
            downside = [r for r in daily_returns if r < 0]
            downside_std = np.std(downside) if downside else 0.0
            sortino = (mean_return * np.sqrt(365)) / downside_std if downside_std > 0 else 0.0
        else:
            sharpe = 0.0
            sortino = 0.0

        # Calmar ratio
        annual_return = total_return_pct * (365 / max(1, len(df) / 1440))  # Approx annualized
        calmar = annual_return / max_dd_pct if max_dd_pct > 0 else 0.0

        # Trade duration
        durations = [t.duration_minutes for t in trades if t.duration_minutes > 0]
        avg_duration = np.mean(durations) if durations else 0.0

        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        curr_wins = 0
        curr_losses = 0

        for t in trades:
            if t.is_winner:
                curr_wins += 1
                curr_losses = 0
                max_consec_wins = max(max_consec_wins, curr_wins)
            else:
                curr_losses += 1
                curr_wins = 0
                max_consec_losses = max(max_consec_losses, curr_losses)

        # Total fees
        total_fees = sum(t.fees for t in trades)

        # Date range
        start_ts = int(df["ts"].iloc[0])
        end_ts = int(df["ts"].iloc[-1])
        start_date = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        end_date = datetime.fromtimestamp(end_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

        return BacktestMetrics(
            strategy=strategy_name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            total_bars=len(df),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            long_trades=long_trades,
            short_trades=short_trades,
            total_return=total_return,
            total_return_pct=total_return_pct,
            avg_trade_return=total_return / total_trades if total_trades > 0 else 0.0,
            avg_trade_return_pct=total_return_pct / total_trades if total_trades > 0 else 0.0,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            avg_trade_duration_min=avg_duration,
            avg_bars_in_trade=avg_duration,  # 1m bars = minutes
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            final_equity=final_equity,
            peak_equity=peak_equity,
            total_fees=total_fees,
        )


class Backtester:
    """
    Main backtesting interface.

    Usage:
        bt = Backtester()

        # Single strategy test
        result = bt.run("BTCUSDT", days=30, use_enhanced=True)
        bt.print_report(result)

        # Compare strategies
        comparison = bt.compare("BTCUSDT", days=30)
        bt.print_comparison(comparison)
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.loader = DataLoader()

    def run(
        self,
        symbol: str,
        days: int = 30,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        csv_path: Optional[str] = None,
        use_enhanced: bool = True,
        **config_overrides,
    ) -> BacktestResult:
        """
        Run backtest on a single strategy.

        Args:
            symbol: Trading pair
            days: Number of days to test
            start: Start date (optional)
            end: End date (optional)
            csv_path: Path to CSV file (instead of downloading)
            use_enhanced: Use enhanced signal engine
            **config_overrides: Override config parameters

        Returns:
            BacktestResult
        """
        # Update config
        cfg = BacktestConfig(
            symbol=symbol,
            use_enhanced=use_enhanced,
            **{k: v for k, v in config_overrides.items() if hasattr(BacktestConfig, k)}
        )

        # Load data
        if csv_path:
            df_1m = self.loader.load_csv(csv_path)
        else:
            df_1m = self.loader.download_binance(symbol, "1m", start, end, days)

        # Aggregate to higher timeframes
        df_5m = self.loader.aggregate_timeframe(df_1m, "5m")
        df_15m = self.loader.aggregate_timeframe(df_1m, "15m")
        df_1h = self.loader.aggregate_timeframe(df_1m, "1h")

        # Run backtest
        engine = BacktestEngine(cfg)
        strategy_name = "enhanced" if use_enhanced else "legacy"

        return engine.run(df_1m, df_5m, df_15m, df_1h, strategy_name)

    def compare(
        self,
        symbol: str,
        days: int = 30,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        csv_path: Optional[str] = None,
        **config_overrides,
    ) -> Dict[str, BacktestResult]:
        """
        Compare enhanced vs legacy strategy.

        Returns:
            Dict with "enhanced" and "legacy" BacktestResults
        """
        # Load data once
        if csv_path:
            df_1m = self.loader.load_csv(csv_path)
        else:
            df_1m = self.loader.download_binance(symbol, "1m", start, end, days)

        df_5m = self.loader.aggregate_timeframe(df_1m, "5m")
        df_15m = self.loader.aggregate_timeframe(df_1m, "15m")
        df_1h = self.loader.aggregate_timeframe(df_1m, "1h")

        results = {}

        # Run enhanced
        print("\n" + "=" * 60)
        cfg_enh = BacktestConfig(symbol=symbol, use_enhanced=True, **config_overrides)
        engine_enh = BacktestEngine(cfg_enh)
        results["enhanced"] = engine_enh.run(df_1m, df_5m, df_15m, df_1h, "enhanced")

        # Run legacy
        print("\n" + "=" * 60)
        cfg_leg = BacktestConfig(symbol=symbol, use_enhanced=False, **config_overrides)
        engine_leg = BacktestEngine(cfg_leg)
        results["legacy"] = engine_leg.run(df_1m, df_5m, df_15m, df_1h, "legacy")

        return results

    def print_report(self, result: BacktestResult) -> None:
        """Print detailed backtest report."""
        m = result.metrics

        print("\n" + "=" * 60)
        print(f"BACKTEST REPORT: {m.strategy.upper()}")
        print("=" * 60)

        print(f"\nSymbol: {m.symbol}")
        print(f"Period: {m.start_date} to {m.end_date}")
        print(f"Total Bars: {m.total_bars:,}")

        print("\n--- PERFORMANCE ---")
        print(f"Total Return: ${m.total_return:,.2f} ({m.total_return_pct:+.2%})")
        print(f"Final Equity: ${m.final_equity:,.2f}")
        print(f"Peak Equity: ${m.peak_equity:,.2f}")

        print("\n--- TRADES ---")
        print(f"Total Trades: {m.total_trades}")
        print(f"  Long: {m.long_trades}, Short: {m.short_trades}")
        print(f"  Winners: {m.winning_trades}, Losers: {m.losing_trades}")
        print(f"Win Rate: {m.win_rate:.1%}")
        print(f"Profit Factor: {m.profit_factor:.2f}")

        print("\n--- WIN/LOSS ---")
        print(f"Avg Win: ${m.avg_win:,.2f}")
        print(f"Avg Loss: ${m.avg_loss:,.2f}")
        print(f"Largest Win: ${m.largest_win:,.2f}")
        print(f"Largest Loss: ${m.largest_loss:,.2f}")

        print("\n--- RISK METRICS ---")
        print(f"Max Drawdown: ${m.max_drawdown:,.2f} ({m.max_drawdown_pct:.2%})")
        print(f"Sharpe Ratio: {m.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {m.sortino_ratio:.2f}")
        print(f"Calmar Ratio: {m.calmar_ratio:.2f}")

        print("\n--- TIME ---")
        print(f"Avg Trade Duration: {m.avg_trade_duration_min:.1f} min")
        print(f"Max Consecutive Wins: {m.max_consecutive_wins}")
        print(f"Max Consecutive Losses: {m.max_consecutive_losses}")

        print("\n--- COSTS ---")
        print(f"Total Fees: ${m.total_fees:,.2f}")

        print("=" * 60)

    def print_comparison(self, results: Dict[str, BacktestResult]) -> None:
        """Print comparison of multiple strategies."""
        print("\n" + "=" * 70)
        print("STRATEGY COMPARISON")
        print("=" * 70)

        # Header
        strategies = list(results.keys())
        header = f"{'Metric':<30}"
        for s in strategies:
            header += f"{s.upper():>18}"
        print(header)
        print("-" * 70)

        # Get metrics
        metrics_list = [results[s].metrics for s in strategies]

        # Rows
        rows = [
            ("Total Return", lambda m: f"${m.total_return:,.0f}"),
            ("Return %", lambda m: f"{m.total_return_pct:+.2%}"),
            ("Total Trades", lambda m: f"{m.total_trades}"),
            ("Win Rate", lambda m: f"{m.win_rate:.1%}"),
            ("Profit Factor", lambda m: f"{m.profit_factor:.2f}"),
            ("Avg Win", lambda m: f"${m.avg_win:,.0f}"),
            ("Avg Loss", lambda m: f"${m.avg_loss:,.0f}"),
            ("Max Drawdown", lambda m: f"{m.max_drawdown_pct:.2%}"),
            ("Sharpe Ratio", lambda m: f"{m.sharpe_ratio:.2f}"),
            ("Sortino Ratio", lambda m: f"{m.sortino_ratio:.2f}"),
            ("Avg Duration (min)", lambda m: f"{m.avg_trade_duration_min:.1f}"),
            ("Total Fees", lambda m: f"${m.total_fees:,.0f}"),
        ]

        for label, formatter in rows:
            row = f"{label:<30}"
            for m in metrics_list:
                row += f"{formatter(m):>18}"
            print(row)

        print("=" * 70)

        # Winner determination
        enh = results.get("enhanced")
        leg = results.get("legacy")
        if enh and leg:
            print("\n--- WINNER ---")
            if enh.metrics.total_return > leg.metrics.total_return:
                diff = enh.metrics.total_return - leg.metrics.total_return
                print(f"ENHANCED wins by ${diff:,.2f}")
            elif leg.metrics.total_return > enh.metrics.total_return:
                diff = leg.metrics.total_return - enh.metrics.total_return
                print(f"LEGACY wins by ${diff:,.2f}")
            else:
                print("TIE")

    def save_report(
        self,
        result: BacktestResult,
        path: Optional[str] = None,
        format: str = "json",
    ) -> str:
        """
        Save backtest report to file.

        Args:
            result: BacktestResult to save
            path: Output path (auto-generated if not specified)
            format: "json" or "csv"

        Returns:
            Path to saved file
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_path = Path("reports/backtest")
            dir_path.mkdir(parents=True, exist_ok=True)
            path = str(dir_path / f"backtest_{result.metrics.symbol}_{result.metrics.strategy}_{timestamp}.{format}")

        if format == "json":
            # Convert to serializable format
            data = {
                "config": asdict(result.config),
                "metrics": asdict(result.metrics),
                "trades": [
                    {
                        **{k: v for k, v in asdict(t).items() if k != "direction"},
                        "direction": t.direction.value,
                        "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                        "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                    }
                    for t in result.trades
                ],
                "equity_curve": [
                    {"time": dt.isoformat(), "equity": eq}
                    for dt, eq in result.equity_curve
                ],
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

        elif format == "csv":
            # Save trades to CSV
            with open(path, "w", newline="", encoding="utf-8") as f:
                if result.trades:
                    writer = csv.DictWriter(f, fieldnames=[
                        "id", "symbol", "direction", "entry_price", "entry_time",
                        "exit_price", "exit_time", "size", "stop_loss", "take_profit",
                        "pnl", "pnl_pct", "fees", "exit_reason", "confidence", "strategy"
                    ])
                    writer.writeheader()
                    for t in result.trades:
                        writer.writerow({
                            "id": t.id,
                            "symbol": t.symbol,
                            "direction": t.direction.value,
                            "entry_price": t.entry_price,
                            "entry_time": t.entry_time.isoformat() if t.entry_time else "",
                            "exit_price": t.exit_price,
                            "exit_time": t.exit_time.isoformat() if t.exit_time else "",
                            "size": t.size,
                            "stop_loss": t.stop_loss,
                            "take_profit": t.take_profit,
                            "pnl": t.pnl,
                            "pnl_pct": t.pnl_pct,
                            "fees": t.fees,
                            "exit_reason": t.exit_reason,
                            "confidence": t.confidence,
                            "strategy": t.strategy,
                        })

        print(f"Report saved to: {path}")
        return path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Eclipse Scalper Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test enhanced strategy on BTCUSDT for 30 days
  python -m strategies.backtester --symbol BTCUSDT --days 30

  # Compare enhanced vs legacy
  python -m strategies.backtester --symbol BTCUSDT --days 30 --compare

  # Load from CSV file
  python -m strategies.backtester --csv data/btcusdt_1m.csv --compare

  # Save report
  python -m strategies.backtester --symbol ETHUSDT --days 14 --save
        """
    )

    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair (default: BTCUSDT)")
    parser.add_argument("--days", type=int, default=30, help="Days to backtest (default: 30)")
    parser.add_argument("--csv", help="Load data from CSV file instead of downloading")
    parser.add_argument("--compare", action="store_true", help="Compare enhanced vs legacy")
    parser.add_argument("--enhanced", action="store_true", default=True, help="Use enhanced strategy (default)")
    parser.add_argument("--legacy", action="store_true", help="Use legacy strategy")
    parser.add_argument("--save", action="store_true", help="Save report to file")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital (default: 10000)")
    parser.add_argument("--risk", type=float, default=0.01, help="Risk per trade (default: 0.01 = 1%%)")
    parser.add_argument("--min-conf", type=float, default=0.5, help="Minimum confidence (default: 0.5)")

    args = parser.parse_args()

    # Create backtester
    bt = Backtester()

    config_overrides = {
        "initial_capital": args.capital,
        "risk_per_trade": args.risk,
        "min_confidence": args.min_conf,
    }

    if args.compare:
        # Compare both strategies
        results = bt.compare(
            symbol=args.symbol,
            days=args.days,
            csv_path=args.csv,
            **config_overrides,
        )
        bt.print_comparison(results)

        if args.save:
            for name, result in results.items():
                bt.save_report(result)
    else:
        # Single strategy
        use_enhanced = not args.legacy

        result = bt.run(
            symbol=args.symbol,
            days=args.days,
            csv_path=args.csv,
            use_enhanced=use_enhanced,
            **config_overrides,
        )
        bt.print_report(result)

        if args.save:
            bt.save_report(result)


if __name__ == "__main__":
    main()
