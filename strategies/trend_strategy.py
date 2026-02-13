# strategies/trend_strategy.py — SCALPER ETERNAL — TREND FOLLOWING — 2026 v1.0
# Trend-following entry strategy for use when regime is TRENDING.
#
# Entry conditions:
# - ADX > 25 (trending market)
# - +DI > -DI for LONG, -DI > +DI for SHORT
# - MACD > Signal line for LONG, MACD < Signal for SHORT
# - Price > EMA20 > EMA50 for LONG (inverse for SHORT)
#
# Usage:
#     from strategies.trend_strategy import TrendFollowingStrategy
#     strategy = TrendFollowingStrategy()
#     is_long, is_short, confidence = strategy.check_entry(df_1m, df_5m)

from __future__ import annotations

from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TrendSignal:
    """Trend following signal result."""
    is_long: bool
    is_short: bool
    confidence: float
    adx: float
    plus_di: float
    minus_di: float
    macd_histogram: float
    ema_alignment: str
    reason: str


class TrendFollowingStrategy:
    """
    Trend-following entry strategy.

    Uses ADX, MACD, and EMA alignment to confirm trend entries.
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        ema_short: int = 20,
        ema_long: int = 50,
        require_ema_alignment: bool = True,
        require_macd_confirmation: bool = True,
    ):
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.require_ema_alignment = require_ema_alignment
        self.require_macd_confirmation = require_macd_confirmation

    def calculate_adx(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate ADX, +DI, -DI."""
        if len(df) < self.adx_period + 5:
            return 0.0, 50.0, 50.0

        high = df["h"]
        low = df["l"]
        close = df["c"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed
        atr = tr.rolling(self.adx_period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(self.adx_period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(self.adx_period).mean() / atr

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(self.adx_period).mean()

        return (
            float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0,
            float(plus_di.iloc[-1]) if not np.isnan(plus_di.iloc[-1]) else 50.0,
            float(minus_di.iloc[-1]) if not np.isnan(minus_di.iloc[-1]) else 50.0,
        )

    def calculate_macd(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate MACD, Signal, Histogram."""
        if len(df) < self.macd_slow + self.macd_signal:
            return 0.0, 0.0, 0.0

        close = df["c"]
        ema_fast = close.ewm(span=self.macd_fast).mean()
        ema_slow = close.ewm(span=self.macd_slow).mean()

        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.macd_signal).mean()
        histogram = macd - signal

        return (
            float(macd.iloc[-1]),
            float(signal.iloc[-1]),
            float(histogram.iloc[-1]),
        )

    def check_ema_alignment(self, df: pd.DataFrame) -> Tuple[str, bool, bool]:
        """
        Check EMA alignment for trend direction.

        Returns:
            Tuple of (alignment_type, is_bullish_aligned, is_bearish_aligned)
        """
        if len(df) < self.ema_long:
            return "NEUTRAL", False, False

        close = df["c"]
        current = float(close.iloc[-1])
        ema20 = float(close.ewm(span=self.ema_short).mean().iloc[-1])
        ema50 = float(close.ewm(span=self.ema_long).mean().iloc[-1])

        # Bullish: Price > EMA20 > EMA50
        if current > ema20 > ema50:
            return "BULLISH", True, False
        # Bearish: Price < EMA20 < EMA50
        elif current < ema20 < ema50:
            return "BEARISH", False, True
        else:
            return "NEUTRAL", False, False

    def check_entry(
        self,
        df_1m: pd.DataFrame,
        df_5m: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, bool, float]:
        """
        Check for trend-following entry signal.

        Args:
            df_1m: 1-minute OHLCV data
            df_5m: 5-minute data (optional, for confirmation)

        Returns:
            Tuple of (is_long, is_short, confidence)
        """
        if len(df_1m) < 50:
            return False, False, 0.0

        # Calculate indicators
        adx, plus_di, minus_di = self.calculate_adx(df_1m)
        macd, signal, histogram = self.calculate_macd(df_1m)
        alignment, bullish_align, bearish_align = self.check_ema_alignment(df_1m)

        # Check trend strength
        if adx < self.adx_threshold:
            return False, False, 0.0

        # Determine direction
        is_long = False
        is_short = False
        confidence = 0.0
        scores = []

        # ADX direction
        if plus_di > minus_di:
            adx_bullish = True
            scores.append(0.3)  # ADX confirms bullish
        else:
            adx_bullish = False
            scores.append(0.3)  # ADX confirms bearish

        # MACD confirmation
        macd_bullish = histogram > 0
        if self.require_macd_confirmation:
            if (adx_bullish and not macd_bullish) or (not adx_bullish and macd_bullish):
                return False, False, 0.0
        if macd_bullish == adx_bullish:
            scores.append(0.3)  # MACD aligns with ADX

        # EMA alignment
        if self.require_ema_alignment:
            if adx_bullish and not bullish_align:
                return False, False, 0.0
            if not adx_bullish and not bearish_align:
                return False, False, 0.0
        if (adx_bullish and bullish_align) or (not adx_bullish and bearish_align):
            scores.append(0.2)  # EMA alignment bonus

        # ADX strength bonus
        if adx > 40:
            scores.append(0.2)  # Strong trend

        # Calculate confidence
        confidence = min(1.0, sum(scores))

        # Set direction
        if adx_bullish:
            is_long = True
        else:
            is_short = True

        # 5m confirmation (optional)
        if df_5m is not None and len(df_5m) >= 50:
            adx_5m, plus_di_5m, minus_di_5m = self.calculate_adx(df_5m)
            macd_5m, signal_5m, hist_5m = self.calculate_macd(df_5m)

            # Check 5m alignment
            mtf_aligned = False
            if is_long and plus_di_5m > minus_di_5m and hist_5m > 0:
                mtf_aligned = True
            elif is_short and minus_di_5m > plus_di_5m and hist_5m < 0:
                mtf_aligned = True

            if mtf_aligned:
                confidence = min(1.0, confidence + 0.1)
            else:
                confidence *= 0.8  # Reduce confidence if MTF doesn't align

        return is_long, is_short, confidence

    def get_signal(
        self,
        df_1m: pd.DataFrame,
        df_5m: Optional[pd.DataFrame] = None,
    ) -> TrendSignal:
        """
        Get detailed trend signal with all indicator values.

        Returns:
            TrendSignal with full details
        """
        if len(df_1m) < 50:
            return TrendSignal(
                is_long=False,
                is_short=False,
                confidence=0.0,
                adx=0.0,
                plus_di=0.0,
                minus_di=0.0,
                macd_histogram=0.0,
                ema_alignment="NEUTRAL",
                reason="Insufficient data",
            )

        adx, plus_di, minus_di = self.calculate_adx(df_1m)
        macd, signal, histogram = self.calculate_macd(df_1m)
        alignment, _, _ = self.check_ema_alignment(df_1m)

        is_long, is_short, confidence = self.check_entry(df_1m, df_5m)

        reason = ""
        if not is_long and not is_short:
            if adx < self.adx_threshold:
                reason = f"ADX too low ({adx:.1f} < {self.adx_threshold})"
            elif alignment == "NEUTRAL":
                reason = "EMA not aligned"
            else:
                reason = "Indicators conflicting"
        else:
            direction = "LONG" if is_long else "SHORT"
            reason = f"{direction}: ADX={adx:.1f}, Hist={histogram:.2f}, EMA={alignment}"

        return TrendSignal(
            is_long=is_long,
            is_short=is_short,
            confidence=confidence,
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
            macd_histogram=histogram,
            ema_alignment=alignment,
            reason=reason,
        )


def main():
    """Test trend strategy on historical data."""
    import argparse

    parser = argparse.ArgumentParser(description="Test trend-following strategy")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--days", type=int, default=7, help="Days of data")
    args = parser.parse_args()

    from strategies.backtester import DataLoader

    print(f"\nLoading {args.days} days of {args.symbol} 1m data...")
    loader = DataLoader()
    df = loader.download_binance(args.symbol, "1m", days=args.days)

    if df.empty:
        print("No data loaded")
        return

    print(f"Loaded {len(df):,} bars")

    # Aggregate to 5m
    df_5m = df.resample("5min").agg({
        "o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"
    }).dropna()

    strategy = TrendFollowingStrategy()

    # Count signals
    long_signals = 0
    short_signals = 0
    no_signals = 0

    print("\nAnalyzing signals...")
    for i in range(100, len(df), 60):  # Check every hour
        df_window = df.iloc[:i + 1]
        df_5m_window = df_5m.loc[:df_window.index[-1]]

        signal = strategy.get_signal(df_window, df_5m_window)

        if signal.is_long:
            long_signals += 1
        elif signal.is_short:
            short_signals += 1
        else:
            no_signals += 1

    total = long_signals + short_signals + no_signals
    print("\nSIGNAL DISTRIBUTION:")
    print("-" * 40)
    print(f"  LONG:     {long_signals:>5} ({long_signals / total * 100:>5.1f}%)")
    print(f"  SHORT:    {short_signals:>5} ({short_signals / total * 100:>5.1f}%)")
    print(f"  NO SIGNAL:{no_signals:>5} ({no_signals / total * 100:>5.1f}%)")

    # Current signal
    current = strategy.get_signal(df, df_5m)
    print(f"\nCURRENT SIGNAL:")
    print(f"  Direction: {'LONG' if current.is_long else 'SHORT' if current.is_short else 'NONE'}")
    print(f"  Confidence: {current.confidence:.2f}")
    print(f"  ADX: {current.adx:.1f}")
    print(f"  +DI: {current.plus_di:.1f}, -DI: {current.minus_di:.1f}")
    print(f"  MACD Histogram: {current.macd_histogram:.4f}")
    print(f"  EMA Alignment: {current.ema_alignment}")
    print(f"  Reason: {current.reason}")


if __name__ == "__main__":
    main()
