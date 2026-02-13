# strategies/indicators/rsi_enhanced.py — SCALPER ETERNAL — RSI ENHANCED — 2026 v1.0
# Enhanced RSI with divergence, failure swings, and momentum analysis.
#
# Signals:
# - Oversold/Overbought conditions
# - Regular and hidden divergence
# - Failure swings (strong reversal signal)
# - RSI momentum (rising/falling)

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, List

from strategies.indicators import (
    BaseIndicator,
    IndicatorSignal,
    SignalDirection,
    SignalStrength,
)


@dataclass
class RSIResult:
    """RSI calculation result."""
    rsi: pd.Series
    rsi_current: float
    rsi_prev: float
    rsi_avg: float


class RSIEnhancedIndicator(BaseIndicator):
    """
    Enhanced RSI with divergence and failure swing detection.

    Usage:
        rsi = RSIEnhancedIndicator(period=14, oversold=30, overbought=70)
        signal = rsi.calculate(df)

        if signal.metadata.get("divergence") == "regular":
            print("RSI divergence detected - strong reversal signal")
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        extreme_oversold: float = 20.0,
        extreme_overbought: float = 80.0,
        weight: float = 1.0,
    ):
        super().__init__("RSI", weight)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.extreme_oversold = extreme_oversold
        self.extreme_overbought = extreme_overbought

    def _calculate_rsi(self, df: pd.DataFrame) -> Optional[RSIResult]:
        """Calculate RSI."""
        try:
            close = df["c"]
            delta = close.diff()

            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)

            avg_gain = gain.ewm(span=self.period, adjust=False).mean()
            avg_loss = loss.ewm(span=self.period, adjust=False).mean()

            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(50)

            return RSIResult(
                rsi=rsi,
                rsi_current=self.safe_float(rsi.iloc[-1], 50.0),
                rsi_prev=self.safe_float(rsi.iloc[-2], 50.0) if len(rsi) > 1 else 50.0,
                rsi_avg=self.safe_float(rsi.iloc[-20:].mean(), 50.0) if len(rsi) >= 20 else 50.0,
            )
        except Exception:
            return None

    def _detect_extremes(self, result: RSIResult) -> Tuple[SignalDirection, float, str]:
        """Detect oversold/overbought conditions."""
        rsi = result.rsi_current

        if rsi <= self.extreme_oversold:
            return SignalDirection.LONG, 0.9, "extreme_oversold"
        elif rsi <= self.oversold:
            return SignalDirection.LONG, 0.6, "oversold"
        elif rsi >= self.extreme_overbought:
            return SignalDirection.SHORT, 0.9, "extreme_overbought"
        elif rsi >= self.overbought:
            return SignalDirection.SHORT, 0.6, "overbought"
        else:
            return SignalDirection.NEUTRAL, 0.0, "neutral"

    def _find_pivots(self, series: pd.Series, lookback: int = 5) -> Tuple[List[int], List[int]]:
        """Find local highs and lows in series."""
        highs = []
        lows = []

        values = series.values

        for i in range(lookback, len(values) - 1):
            # Local high
            if values[i] > max(values[i-lookback:i]) and values[i] > values[i+1]:
                highs.append(i)
            # Local low
            if values[i] < min(values[i-lookback:i]) and values[i] < values[i+1]:
                lows.append(i)

        return highs, lows

    def _detect_divergence(
        self,
        df: pd.DataFrame,
        result: RSIResult,
        lookback: int = 30,
    ) -> Tuple[SignalDirection, str]:
        """
        Detect RSI divergence.

        Regular bullish: Price lower low, RSI higher low
        Regular bearish: Price higher high, RSI lower high
        Hidden bullish: Price higher low, RSI lower low (trend continuation)
        Hidden bearish: Price lower high, RSI higher high (trend continuation)
        """
        try:
            if len(df) < lookback + 10:
                return SignalDirection.NEUTRAL, "none"

            price = df["c"].iloc[-lookback:]
            rsi = result.rsi.iloc[-lookback:]

            price_highs, price_lows = self._find_pivots(price)
            rsi_highs, rsi_lows = self._find_pivots(rsi)

            # Need at least 2 pivots for divergence
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                # Get last two lows
                p1, p2 = price_lows[-2], price_lows[-1]
                r1, r2 = rsi_lows[-2], rsi_lows[-1]

                price_vals = price.values
                rsi_vals = rsi.values

                # Regular bullish: price lower low, RSI higher low
                if price_vals[p2] < price_vals[p1] and rsi_vals[r2] > rsi_vals[r1]:
                    return SignalDirection.LONG, "regular"

                # Hidden bullish: price higher low, RSI lower low
                if price_vals[p2] > price_vals[p1] and rsi_vals[r2] < rsi_vals[r1]:
                    return SignalDirection.LONG, "hidden"

            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                # Get last two highs
                p1, p2 = price_highs[-2], price_highs[-1]
                r1, r2 = rsi_highs[-2], rsi_highs[-1]

                price_vals = price.values
                rsi_vals = rsi.values

                # Regular bearish: price higher high, RSI lower high
                if price_vals[p2] > price_vals[p1] and rsi_vals[r2] < rsi_vals[r1]:
                    return SignalDirection.SHORT, "regular"

                # Hidden bearish: price lower high, RSI higher high
                if price_vals[p2] < price_vals[p1] and rsi_vals[r2] > rsi_vals[r1]:
                    return SignalDirection.SHORT, "hidden"

            return SignalDirection.NEUTRAL, "none"

        except Exception:
            return SignalDirection.NEUTRAL, "none"

    def _detect_failure_swing(self, result: RSIResult) -> Tuple[SignalDirection, bool]:
        """
        Detect RSI failure swing (strong reversal signal).

        Bullish failure swing:
        1. RSI falls below 30 (oversold)
        2. RSI bounces above 30
        3. RSI pulls back but stays above 30
        4. RSI breaks above the prior peak

        Bearish failure swing:
        1. RSI rises above 70 (overbought)
        2. RSI falls below 70
        3. RSI rallies but stays below 70
        4. RSI breaks below the prior trough
        """
        try:
            rsi = result.rsi

            if len(rsi) < 10:
                return SignalDirection.NEUTRAL, False

            # Look at last 10 bars
            recent = rsi.iloc[-10:].values

            # Check for bullish failure swing
            # Find if RSI was oversold, bounced, pulled back, and broke higher
            was_oversold = any(v < self.oversold for v in recent[:5])
            if was_oversold:
                # Find the bounce peak
                peak_idx = np.argmax(recent[:7])
                peak_val = recent[peak_idx]

                # Check if pulled back and then broke above peak
                if peak_idx < len(recent) - 2:
                    pullback = min(recent[peak_idx+1:-1]) if peak_idx < len(recent) - 2 else recent[-1]
                    current = recent[-1]

                    if pullback > self.oversold and current > peak_val:
                        return SignalDirection.LONG, True

            # Check for bearish failure swing
            was_overbought = any(v > self.overbought for v in recent[:5])
            if was_overbought:
                # Find the trough
                trough_idx = np.argmin(recent[:7])
                trough_val = recent[trough_idx]

                # Check if rallied and then broke below trough
                if trough_idx < len(recent) - 2:
                    rally = max(recent[trough_idx+1:-1]) if trough_idx < len(recent) - 2 else recent[-1]
                    current = recent[-1]

                    if rally < self.overbought and current < trough_val:
                        return SignalDirection.SHORT, True

            return SignalDirection.NEUTRAL, False

        except Exception:
            return SignalDirection.NEUTRAL, False

    def _analyze_momentum(self, result: RSIResult) -> Tuple[SignalDirection, float]:
        """Analyze RSI momentum (direction of RSI movement)."""
        try:
            rsi = result.rsi

            if len(rsi) < 5:
                return SignalDirection.NEUTRAL, 0.0

            # Calculate RSI slope
            recent = rsi.iloc[-5:].values
            slope = (recent[-1] - recent[0]) / 4  # Avg change per bar

            if slope > 2:
                return SignalDirection.LONG, min(1.0, slope / 10)
            elif slope < -2:
                return SignalDirection.SHORT, min(1.0, abs(slope) / 10)
            else:
                return SignalDirection.NEUTRAL, 0.0

        except Exception:
            return SignalDirection.NEUTRAL, 0.0

    def calculate(self, df: pd.DataFrame, **kwargs) -> IndicatorSignal:
        """Calculate enhanced RSI signal."""
        if not self.validate_df(df, min_bars=self.period + 20):
            return IndicatorSignal(name=self.name)

        result = self._calculate_rsi(df)
        if result is None:
            return IndicatorSignal(name=self.name)

        # Analyze components
        extreme_dir, extreme_conf, extreme_type = self._detect_extremes(result)
        div_dir, div_type = self._detect_divergence(df, result)
        fs_dir, is_failure_swing = self._detect_failure_swing(result)
        mom_dir, mom_strength = self._analyze_momentum(result)

        # Combine signals
        signals = []
        confidences = []

        # Failure swing (strongest signal)
        if is_failure_swing:
            signals.append(fs_dir)
            confidences.append(0.9)

        # Divergence (strong signal)
        if div_type == "regular":
            signals.append(div_dir)
            confidences.append(0.85)
        elif div_type == "hidden":
            signals.append(div_dir)
            confidences.append(0.7)

        # Extreme readings
        if extreme_dir != SignalDirection.NEUTRAL:
            signals.append(extreme_dir)
            confidences.append(extreme_conf)

        # Momentum
        if mom_dir != SignalDirection.NEUTRAL:
            signals.append(mom_dir)
            confidences.append(0.3 + mom_strength * 0.3)

        # Determine final direction
        if not signals:
            direction = SignalDirection.NEUTRAL
            confidence = 0.0
            strength = SignalStrength.NONE
        else:
            long_count = sum(1 for s in signals if s == SignalDirection.LONG)
            short_count = sum(1 for s in signals if s == SignalDirection.SHORT)

            if long_count > short_count:
                direction = SignalDirection.LONG
            elif short_count > long_count:
                direction = SignalDirection.SHORT
            else:
                direction = SignalDirection.NEUTRAL

            confidence = sum(confidences) / len(confidences)

            if is_failure_swing or div_type == "regular":
                strength = SignalStrength.STRONG
            elif div_type == "hidden" or extreme_type.startswith("extreme"):
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK

        return IndicatorSignal(
            name=self.name,
            direction=direction,
            strength=strength,
            value=result.rsi_current,
            confidence=min(1.0, confidence),
            raw_values={
                "rsi": result.rsi_current,
                "rsi_prev": result.rsi_prev,
                "rsi_avg": result.rsi_avg,
            },
            metadata={
                "extreme": extreme_type,
                "divergence": div_type,
                "failure_swing": is_failure_swing,
                "momentum": mom_dir.value,
            },
        )
