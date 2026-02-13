# strategies/reversion_strategy.py — SCALPER ETERNAL — MEAN REVERSION — 2026 v1.0
# Mean-reversion entry strategy for use when regime is RANGING.
#
# Entry conditions:
# - BB %B < 0.1 for LONG (near lower band), %B > 0.9 for SHORT
# - RSI < 30 for LONG (oversold), RSI > 70 for SHORT
# - Stochastic %K < 20 for LONG, %K > 80 for SHORT
# - Price within 0.3% of VWAP bands (optional)
#
# Usage:
#     from strategies.reversion_strategy import MeanReversionStrategy
#     strategy = MeanReversionStrategy()
#     is_long, is_short, confidence = strategy.check_entry(df_1m, df_5m)

from __future__ import annotations

from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ReversionSignal:
    """Mean reversion signal result."""
    is_long: bool
    is_short: bool
    confidence: float
    rsi: float
    stoch_k: float
    bb_pct_b: float
    bb_width: float
    distance_from_mean: float
    reason: str


class MeanReversionStrategy:
    """
    Mean-reversion entry strategy.

    Uses RSI, Stochastic, and Bollinger Bands to identify oversold/overbought conditions
    in ranging markets.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        stoch_k: int = 14,
        stoch_d: int = 3,
        stoch_oversold: float = 20.0,
        stoch_overbought: float = 80.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        bb_oversold: float = 0.1,
        bb_overbought: float = 0.9,
        require_multiple_confirms: int = 2,
    ):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.stoch_oversold = stoch_oversold
        self.stoch_overbought = stoch_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_oversold = bb_oversold
        self.bb_overbought = bb_overbought
        self.require_multiple_confirms = require_multiple_confirms

    def calculate_rsi(self, df: pd.DataFrame) -> float:
        """Calculate RSI."""
        if len(df) < self.rsi_period + 1:
            return 50.0

        close = df["c"]
        delta = close.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(self.rsi_period).mean()
        avg_loss = loss.rolling(self.rsi_period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

    def calculate_stochastic(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate Stochastic %K and %D."""
        if len(df) < self.stoch_k:
            return 50.0, 50.0

        high = df["h"]
        low = df["l"]
        close = df["c"]

        lowest = low.rolling(self.stoch_k).min()
        highest = high.rolling(self.stoch_k).max()

        k = 100 * (close - lowest) / (highest - lowest + 1e-10)
        d = k.rolling(self.stoch_d).mean()

        return (
            float(k.iloc[-1]) if not np.isnan(k.iloc[-1]) else 50.0,
            float(d.iloc[-1]) if not np.isnan(d.iloc[-1]) else 50.0,
        )

    def calculate_bollinger(self, df: pd.DataFrame) -> Tuple[float, float, float, float]:
        """
        Calculate Bollinger Band indicators.

        Returns:
            Tuple of (%B, width, upper, lower)
        """
        if len(df) < self.bb_period:
            return 0.5, 0.02, 0.0, 0.0

        close = df["c"]
        sma = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()

        upper = sma + (std * self.bb_std)
        lower = sma - (std * self.bb_std)

        # %B: position within bands (0 = lower, 1 = upper)
        pct_b = (close - lower) / (upper - lower + 1e-10)

        # Width as percentage
        width = (upper - lower) / sma

        return (
            float(pct_b.iloc[-1]) if not np.isnan(pct_b.iloc[-1]) else 0.5,
            float(width.iloc[-1]) if not np.isnan(width.iloc[-1]) else 0.02,
            float(upper.iloc[-1]),
            float(lower.iloc[-1]),
        )

    def check_entry(
        self,
        df_1m: pd.DataFrame,
        df_5m: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, bool, float]:
        """
        Check for mean-reversion entry signal.

        Args:
            df_1m: 1-minute OHLCV data
            df_5m: 5-minute data (optional, for confirmation)

        Returns:
            Tuple of (is_long, is_short, confidence)
        """
        if len(df_1m) < 30:
            return False, False, 0.0

        # Calculate indicators
        rsi = self.calculate_rsi(df_1m)
        stoch_k, stoch_d = self.calculate_stochastic(df_1m)
        pct_b, bb_width, _, _ = self.calculate_bollinger(df_1m)

        # Count oversold/overbought confirmations
        oversold_count = 0
        overbought_count = 0
        confidence_scores = []

        # RSI check
        if rsi < self.rsi_oversold:
            oversold_count += 1
            # Stronger signal for more extreme RSI
            score = 0.3 + (self.rsi_oversold - rsi) / 30 * 0.2
            confidence_scores.append(("RSI_oversold", min(0.5, score)))
        elif rsi > self.rsi_overbought:
            overbought_count += 1
            score = 0.3 + (rsi - self.rsi_overbought) / 30 * 0.2
            confidence_scores.append(("RSI_overbought", min(0.5, score)))

        # Stochastic check
        if stoch_k < self.stoch_oversold:
            oversold_count += 1
            score = 0.25 + (self.stoch_oversold - stoch_k) / 20 * 0.15
            confidence_scores.append(("Stoch_oversold", min(0.4, score)))
        elif stoch_k > self.stoch_overbought:
            overbought_count += 1
            score = 0.25 + (stoch_k - self.stoch_overbought) / 20 * 0.15
            confidence_scores.append(("Stoch_overbought", min(0.4, score)))

        # Bollinger %B check
        if pct_b < self.bb_oversold:
            oversold_count += 1
            score = 0.25 + (self.bb_oversold - pct_b) / 0.1 * 0.15
            confidence_scores.append(("BB_oversold", min(0.4, score)))
        elif pct_b > self.bb_overbought:
            overbought_count += 1
            score = 0.25 + (pct_b - self.bb_overbought) / 0.1 * 0.15
            confidence_scores.append(("BB_overbought", min(0.4, score)))

        # Require multiple confirmations
        is_long = oversold_count >= self.require_multiple_confirms
        is_short = overbought_count >= self.require_multiple_confirms

        # Can't be both
        if is_long and is_short:
            return False, False, 0.0

        # Calculate confidence
        if is_long or is_short:
            scores = [s[1] for s in confidence_scores]
            confidence = min(1.0, sum(scores))

            # Bonus for all three confirming
            if (is_long and oversold_count == 3) or (is_short and overbought_count == 3):
                confidence = min(1.0, confidence + 0.1)

            # 5m confirmation
            if df_5m is not None and len(df_5m) >= 20:
                rsi_5m = self.calculate_rsi(df_5m)

                if is_long and rsi_5m < 40:  # 5m also oversold-ish
                    confidence = min(1.0, confidence + 0.1)
                elif is_short and rsi_5m > 60:  # 5m also overbought-ish
                    confidence = min(1.0, confidence + 0.1)
                else:
                    confidence *= 0.9  # Slight penalty for no MTF alignment

            return is_long, is_short, confidence
        else:
            return False, False, 0.0

    def get_signal(
        self,
        df_1m: pd.DataFrame,
        df_5m: Optional[pd.DataFrame] = None,
    ) -> ReversionSignal:
        """
        Get detailed mean-reversion signal.

        Returns:
            ReversionSignal with full details
        """
        if len(df_1m) < 30:
            return ReversionSignal(
                is_long=False,
                is_short=False,
                confidence=0.0,
                rsi=50.0,
                stoch_k=50.0,
                bb_pct_b=0.5,
                bb_width=0.02,
                distance_from_mean=0.0,
                reason="Insufficient data",
            )

        rsi = self.calculate_rsi(df_1m)
        stoch_k, stoch_d = self.calculate_stochastic(df_1m)
        pct_b, bb_width, upper, lower = self.calculate_bollinger(df_1m)

        # Distance from middle band (mean)
        close = float(df_1m["c"].iloc[-1])
        sma = float(df_1m["c"].rolling(self.bb_period).mean().iloc[-1])
        distance = (close - sma) / sma if sma > 0 else 0.0

        is_long, is_short, confidence = self.check_entry(df_1m, df_5m)

        # Generate reason
        if is_long:
            confirms = []
            if rsi < self.rsi_oversold:
                confirms.append(f"RSI={rsi:.1f}")
            if stoch_k < self.stoch_oversold:
                confirms.append(f"Stoch={stoch_k:.1f}")
            if pct_b < self.bb_oversold:
                confirms.append(f"%B={pct_b:.2f}")
            reason = f"LONG (oversold): {', '.join(confirms)}"
        elif is_short:
            confirms = []
            if rsi > self.rsi_overbought:
                confirms.append(f"RSI={rsi:.1f}")
            if stoch_k > self.stoch_overbought:
                confirms.append(f"Stoch={stoch_k:.1f}")
            if pct_b > self.bb_overbought:
                confirms.append(f"%B={pct_b:.2f}")
            reason = f"SHORT (overbought): {', '.join(confirms)}"
        else:
            reason = f"No extreme: RSI={rsi:.1f}, Stoch={stoch_k:.1f}, %B={pct_b:.2f}"

        return ReversionSignal(
            is_long=is_long,
            is_short=is_short,
            confidence=confidence,
            rsi=rsi,
            stoch_k=stoch_k,
            bb_pct_b=pct_b,
            bb_width=bb_width,
            distance_from_mean=distance,
            reason=reason,
        )


def main():
    """Test mean-reversion strategy on historical data."""
    import argparse

    parser = argparse.ArgumentParser(description="Test mean-reversion strategy")
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

    strategy = MeanReversionStrategy()

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
    print(f"  RSI: {current.rsi:.1f}")
    print(f"  Stochastic %K: {current.stoch_k:.1f}")
    print(f"  BB %B: {current.bb_pct_b:.2f}")
    print(f"  BB Width: {current.bb_width * 100:.2f}%")
    print(f"  Distance from Mean: {current.distance_from_mean * 100:.2f}%")
    print(f"  Reason: {current.reason}")


if __name__ == "__main__":
    main()
