# strategies/indicators/stochastic.py — SCALPER ETERNAL — STOCHASTIC OSCILLATOR — 2026 v1.0
# Stochastic Oscillator with crossover and divergence signals.

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple

from strategies.indicators import (
    BaseIndicator,
    IndicatorSignal,
    SignalDirection,
    SignalStrength,
)


@dataclass
class StochasticResult:
    """Stochastic calculation result."""
    k: pd.Series
    d: pd.Series
    k_current: float
    d_current: float
    k_prev: float
    d_prev: float


class StochasticIndicator(BaseIndicator):
    """
    Stochastic Oscillator with %K/%D crossover and extreme signals.

    Usage:
        stoch = StochasticIndicator(k_period=14, d_period=3)
        signal = stoch.calculate(df)
    """

    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3,
        oversold: float = 20.0,
        overbought: float = 80.0,
        weight: float = 1.0,
    ):
        super().__init__("Stochastic", weight)
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
        self.oversold = oversold
        self.overbought = overbought

    def _calculate_stochastic(self, df: pd.DataFrame) -> Optional[StochasticResult]:
        """Calculate Stochastic Oscillator."""
        try:
            high = df["h"]
            low = df["l"]
            close = df["c"]

            # Lowest low and highest high
            lowest_low = low.rolling(self.k_period).min()
            highest_high = high.rolling(self.k_period).max()

            # Fast %K
            fast_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)

            # Slow %K (smoothed)
            k = fast_k.rolling(self.smooth_k).mean()

            # %D (signal line)
            d = k.rolling(self.d_period).mean()

            return StochasticResult(
                k=k,
                d=d,
                k_current=self.safe_float(k.iloc[-1], 50.0),
                d_current=self.safe_float(d.iloc[-1], 50.0),
                k_prev=self.safe_float(k.iloc[-2], 50.0) if len(k) > 1 else 50.0,
                d_prev=self.safe_float(d.iloc[-2], 50.0) if len(d) > 1 else 50.0,
            )
        except Exception:
            return None

    def _detect_crossover(self, result: StochasticResult) -> Tuple[SignalDirection, bool]:
        """Detect %K/%D crossover."""
        k_curr, d_curr = result.k_current, result.d_current
        k_prev, d_prev = result.k_prev, result.d_prev

        # Bullish crossover: %K crosses above %D
        if k_curr > d_curr and k_prev <= d_prev:
            return SignalDirection.LONG, True

        # Bearish crossover: %K crosses below %D
        if k_curr < d_curr and k_prev >= d_prev:
            return SignalDirection.SHORT, True

        return SignalDirection.NEUTRAL, False

    def _detect_extremes(self, result: StochasticResult) -> Tuple[SignalDirection, float, str]:
        """Detect oversold/overbought conditions."""
        k = result.k_current

        if k <= self.oversold:
            return SignalDirection.LONG, 0.7, "oversold"
        elif k >= self.overbought:
            return SignalDirection.SHORT, 0.7, "overbought"
        else:
            return SignalDirection.NEUTRAL, 0.0, "neutral"

    def calculate(self, df: pd.DataFrame, **kwargs) -> IndicatorSignal:
        """Calculate Stochastic signal."""
        if not self.validate_df(df, min_bars=self.k_period + 10):
            return IndicatorSignal(name=self.name)

        result = self._calculate_stochastic(df)
        if result is None:
            return IndicatorSignal(name=self.name)

        cross_dir, is_crossover = self._detect_crossover(result)
        extreme_dir, extreme_conf, extreme_type = self._detect_extremes(result)

        # Combine signals
        signals = []
        confidences = []

        # Crossover in extreme zone is strongest
        if is_crossover:
            if extreme_type == "oversold" and cross_dir == SignalDirection.LONG:
                signals.append(SignalDirection.LONG)
                confidences.append(0.9)
            elif extreme_type == "overbought" and cross_dir == SignalDirection.SHORT:
                signals.append(SignalDirection.SHORT)
                confidences.append(0.9)
            else:
                signals.append(cross_dir)
                confidences.append(0.6)

        # Extreme without crossover
        if extreme_dir != SignalDirection.NEUTRAL and not is_crossover:
            signals.append(extreme_dir)
            confidences.append(extreme_conf * 0.5)

        if not signals:
            direction = SignalDirection.NEUTRAL
            confidence = 0.0
            strength = SignalStrength.NONE
        else:
            long_count = sum(1 for s in signals if s == SignalDirection.LONG)
            short_count = sum(1 for s in signals if s == SignalDirection.SHORT)

            direction = SignalDirection.LONG if long_count > short_count else (
                SignalDirection.SHORT if short_count > long_count else SignalDirection.NEUTRAL
            )
            confidence = max(confidences)
            strength = SignalStrength.STRONG if is_crossover and extreme_type != "neutral" else (
                SignalStrength.MODERATE if is_crossover else SignalStrength.WEAK
            )

        return IndicatorSignal(
            name=self.name,
            direction=direction,
            strength=strength,
            value=result.k_current,
            confidence=min(1.0, confidence),
            raw_values={
                "k": result.k_current,
                "d": result.d_current,
            },
            metadata={
                "crossover": is_crossover,
                "extreme": extreme_type,
            },
        )
