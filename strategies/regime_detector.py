# strategies/regime_detector.py — SCALPER ETERNAL — REGIME DETECTION — 2026 v1.0
# Detect market regime: TRENDING, RANGING, or VOLATILE
#
# Purpose:
# - Classify current market conditions
# - Route to appropriate strategy (trend-following vs mean-reversion)
# - Avoid trading in unsuitable conditions
#
# Usage:
#     from strategies.regime_detector import RegimeDetector
#     regime = RegimeDetector().detect(df)

from __future__ import annotations

from typing import Literal, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd


RegimeType = Literal["TRENDING", "RANGING", "VOLATILE", "UNKNOWN"]


@dataclass
class RegimeResult:
    """Regime detection result."""
    regime: RegimeType
    confidence: float
    adx: float
    atr_ratio: float
    bb_width: float
    trend_direction: Literal["UP", "DOWN", "NEUTRAL"]
    details: str


class RegimeDetector:
    """
    Detect market regime using ADX, ATR, and Bollinger Band width.

    Regimes:
    - TRENDING: Strong directional move (ADX > 25, price above/below EMA consistently)
    - RANGING: Sideways market (ADX < 20, BB width < 2%)
    - VOLATILE: High volatility with no clear direction (ATR > 1.5x average)
    """

    def __init__(
        self,
        adx_period: int = 14,
        ema_period: int = 20,
        bb_period: int = 20,
        atr_period: int = 14,
        trend_threshold: float = 25.0,
        range_threshold: float = 20.0,
        volatility_mult: float = 1.5,
    ):
        self.adx_period = adx_period
        self.ema_period = ema_period
        self.bb_period = bb_period
        self.atr_period = atr_period
        self.trend_threshold = trend_threshold
        self.range_threshold = range_threshold
        self.volatility_mult = volatility_mult

    def calculate_adx(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Calculate ADX, +DI, -DI.

        Returns:
            Tuple of (ADX, +DI, -DI)
        """
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

        # Smoothed averages
        atr = tr.rolling(self.adx_period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(self.adx_period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(self.adx_period).mean() / atr

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(self.adx_period).mean()

        return (
            float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0,
            float(plus_di.iloc[-1]) if not np.isnan(plus_di.iloc[-1]) else 50.0,
            float(minus_di.iloc[-1]) if not np.isnan(minus_di.iloc[-1]) else 50.0,
        )

    def calculate_atr_ratio(self, df: pd.DataFrame) -> float:
        """
        Calculate ATR ratio (current ATR vs 20-period average ATR).

        Returns:
            Ratio of current ATR to average ATR
        """
        if len(df) < self.atr_period + 20:
            return 1.0

        high = df["h"]
        low = df["l"]
        close = df["c"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Current ATR
        current_atr = tr.iloc[-self.atr_period:].mean()

        # Average ATR (longer period)
        avg_atr = tr.iloc[-self.atr_period * 2:-self.atr_period].mean()

        if avg_atr > 0:
            return float(current_atr / avg_atr)
        return 1.0

    def calculate_bb_width(self, df: pd.DataFrame) -> float:
        """
        Calculate Bollinger Band width as percentage.

        Returns:
            BB width as percentage (0.02 = 2%)
        """
        if len(df) < self.bb_period:
            return 0.02

        close = df["c"]
        sma = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()

        upper = sma + (std * 2)
        lower = sma - (std * 2)

        width = (upper - lower) / sma
        return float(width.iloc[-1]) if not np.isnan(width.iloc[-1]) else 0.02

    def get_trend_direction(self, df: pd.DataFrame) -> Literal["UP", "DOWN", "NEUTRAL"]:
        """
        Determine trend direction from EMA relationship.

        Returns:
            "UP", "DOWN", or "NEUTRAL"
        """
        if len(df) < 50:
            return "NEUTRAL"

        close = df["c"]
        ema20 = close.ewm(span=20).mean().iloc[-1]
        ema50 = close.ewm(span=50).mean().iloc[-1]
        current = close.iloc[-1]

        # Price above both EMAs and EMA20 > EMA50
        if current > ema20 > ema50:
            return "UP"
        # Price below both EMAs and EMA20 < EMA50
        elif current < ema20 < ema50:
            return "DOWN"
        else:
            return "NEUTRAL"

    def detect(self, df: pd.DataFrame) -> RegimeResult:
        """
        Detect current market regime.

        Args:
            df: OHLCV DataFrame with at least 50 bars

        Returns:
            RegimeResult with regime type and confidence
        """
        if len(df) < 50:
            return RegimeResult(
                regime="UNKNOWN",
                confidence=0.0,
                adx=0.0,
                atr_ratio=1.0,
                bb_width=0.02,
                trend_direction="NEUTRAL",
                details="Insufficient data",
            )

        # Calculate indicators
        adx, plus_di, minus_di = self.calculate_adx(df)
        atr_ratio = self.calculate_atr_ratio(df)
        bb_width = self.calculate_bb_width(df)
        trend_dir = self.get_trend_direction(df)

        # Regime detection logic
        regime: RegimeType = "UNKNOWN"
        confidence = 0.0
        details = ""

        # Check VOLATILE first (highest priority safety check)
        if atr_ratio > self.volatility_mult:
            regime = "VOLATILE"
            confidence = min(1.0, (atr_ratio - 1.0) / 0.5)
            details = f"High volatility: ATR ratio {atr_ratio:.2f}x"

        # Check TRENDING
        elif adx >= self.trend_threshold:
            regime = "TRENDING"
            # Confidence based on ADX strength
            confidence = min(1.0, (adx - 25) / 25)  # 25-50 ADX -> 0-1 confidence
            if plus_di > minus_di:
                details = f"Uptrend: ADX={adx:.1f}, +DI={plus_di:.1f}>{minus_di:.1f}"
            else:
                details = f"Downtrend: ADX={adx:.1f}, -DI={minus_di:.1f}>{plus_di:.1f}"

        # Check RANGING
        elif adx < self.range_threshold and bb_width < 0.02:
            regime = "RANGING"
            # Confidence based on how low ADX is and how tight bands are
            adx_factor = (self.range_threshold - adx) / self.range_threshold
            bb_factor = (0.02 - bb_width) / 0.02 if bb_width < 0.02 else 0.0
            confidence = (adx_factor + bb_factor) / 2
            details = f"Range-bound: ADX={adx:.1f}, BB width={bb_width * 100:.2f}%"

        # Default to RANGING with low confidence
        else:
            regime = "RANGING"
            confidence = 0.3
            details = f"Unclear regime: ADX={adx:.1f}, BB width={bb_width * 100:.2f}%"

        return RegimeResult(
            regime=regime,
            confidence=confidence,
            adx=adx,
            atr_ratio=atr_ratio,
            bb_width=bb_width,
            trend_direction=trend_dir,
            details=details,
        )


def main():
    """Test regime detector on historical data."""
    import argparse

    parser = argparse.ArgumentParser(description="Test regime detection")
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

    detector = RegimeDetector()

    # Analyze regime distribution
    regimes = {"TRENDING": 0, "RANGING": 0, "VOLATILE": 0, "UNKNOWN": 0}
    window = 100

    print("\nAnalyzing regime distribution...")
    for i in range(window, len(df), 60):  # Check every hour
        df_window = df.iloc[:i + 1]
        result = detector.detect(df_window)
        regimes[result.regime] += 1

    total = sum(regimes.values())
    print("\nREGIME DISTRIBUTION:")
    print("-" * 40)
    for regime, count in sorted(regimes.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        print(f"  {regime:<12}: {count:>5} ({pct:>5.1f}%)")

    # Current regime
    current = detector.detect(df)
    print(f"\nCURRENT REGIME: {current.regime}")
    print(f"  Confidence: {current.confidence:.2f}")
    print(f"  ADX: {current.adx:.1f}")
    print(f"  ATR Ratio: {current.atr_ratio:.2f}x")
    print(f"  BB Width: {current.bb_width * 100:.2f}%")
    print(f"  Trend: {current.trend_direction}")
    print(f"  Details: {current.details}")


if __name__ == "__main__":
    main()
