# strategies/indicators/macd.py — SCALPER ETERNAL — MACD INDICATOR — 2026 v1.0
# MACD (Moving Average Convergence Divergence) indicator module.
#
# Signals:
# - Crossover (MACD line crosses signal line)
# - Histogram momentum (increasing/decreasing)
# - Zero line cross
# - Divergence (price vs MACD)

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
class MACDResult:
    """MACD calculation result."""
    macd_line: pd.Series
    signal_line: pd.Series
    histogram: pd.Series
    macd_current: float
    signal_current: float
    histogram_current: float
    histogram_prev: float


class MACDIndicator(BaseIndicator):
    """
    MACD Indicator with crossover, divergence, and momentum signals.

    Usage:
        macd = MACDIndicator(fast=12, slow=26, signal=9)
        signal = macd.calculate(df)

        if signal.is_bullish():
            print(f"MACD bullish with confidence {signal.confidence}")
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        weight: float = 1.0,
    ):
        super().__init__("MACD", weight)
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def _calculate_macd(self, df: pd.DataFrame) -> Optional[MACDResult]:
        """Calculate MACD components."""
        try:
            close = df["c"]

            # EMA calculations
            ema_fast = close.ewm(span=self.fast, adjust=False).mean()
            ema_slow = close.ewm(span=self.slow, adjust=False).mean()

            # MACD line
            macd_line = ema_fast - ema_slow

            # Signal line
            signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

            # Histogram
            histogram = macd_line - signal_line

            return MACDResult(
                macd_line=macd_line,
                signal_line=signal_line,
                histogram=histogram,
                macd_current=self.safe_float(macd_line.iloc[-1]),
                signal_current=self.safe_float(signal_line.iloc[-1]),
                histogram_current=self.safe_float(histogram.iloc[-1]),
                histogram_prev=self.safe_float(histogram.iloc[-2]) if len(histogram) > 1 else 0.0,
            )
        except Exception:
            return None

    def _detect_crossover(self, result: MACDResult) -> Tuple[SignalDirection, bool]:
        """Detect MACD/Signal line crossover."""
        try:
            macd = result.macd_line
            signal = result.signal_line

            if len(macd) < 3:
                return SignalDirection.NEUTRAL, False

            # Current and previous relationship
            curr_above = macd.iloc[-1] > signal.iloc[-1]
            prev_above = macd.iloc[-2] > signal.iloc[-2]

            # Bullish crossover: was below, now above
            if curr_above and not prev_above:
                return SignalDirection.LONG, True

            # Bearish crossover: was above, now below
            if not curr_above and prev_above:
                return SignalDirection.SHORT, True

            # No crossover, but maintain direction
            if curr_above:
                return SignalDirection.LONG, False
            else:
                return SignalDirection.SHORT, False

        except Exception:
            return SignalDirection.NEUTRAL, False

    def _detect_zero_cross(self, result: MACDResult) -> Tuple[SignalDirection, bool]:
        """Detect zero line cross."""
        try:
            macd = result.macd_line

            if len(macd) < 2:
                return SignalDirection.NEUTRAL, False

            curr = macd.iloc[-1]
            prev = macd.iloc[-2]

            # Bullish: crossed above zero
            if curr > 0 and prev <= 0:
                return SignalDirection.LONG, True

            # Bearish: crossed below zero
            if curr < 0 and prev >= 0:
                return SignalDirection.SHORT, True

            return SignalDirection.NEUTRAL, False

        except Exception:
            return SignalDirection.NEUTRAL, False

    def _analyze_histogram(self, result: MACDResult) -> Tuple[SignalDirection, float]:
        """Analyze histogram momentum."""
        try:
            hist = result.histogram

            if len(hist) < 5:
                return SignalDirection.NEUTRAL, 0.0

            # Current histogram values
            h0 = hist.iloc[-1]
            h1 = hist.iloc[-2]
            h2 = hist.iloc[-3]

            # Histogram increasing (bullish momentum)
            if h0 > h1 > h2 and h0 > 0:
                strength = min(1.0, abs(h0 - h2) / abs(h0) if h0 != 0 else 0.5)
                return SignalDirection.LONG, strength

            # Histogram decreasing (bearish momentum)
            if h0 < h1 < h2 and h0 < 0:
                strength = min(1.0, abs(h0 - h2) / abs(h0) if h0 != 0 else 0.5)
                return SignalDirection.SHORT, strength

            # Histogram turning (potential reversal)
            if h0 > h1 and h1 < h2 and h0 < 0:
                # Bullish divergence in histogram
                return SignalDirection.LONG, 0.3

            if h0 < h1 and h1 > h2 and h0 > 0:
                # Bearish divergence in histogram
                return SignalDirection.SHORT, 0.3

            return SignalDirection.NEUTRAL, 0.0

        except Exception:
            return SignalDirection.NEUTRAL, 0.0

    def _detect_divergence(
        self,
        df: pd.DataFrame,
        result: MACDResult,
        lookback: int = 30,
    ) -> Tuple[SignalDirection, str]:
        """
        Detect price vs MACD divergence.

        Regular bullish: Price lower low, MACD higher low
        Regular bearish: Price higher high, MACD lower high
        Hidden bullish: Price higher low, MACD lower low
        Hidden bearish: Price lower high, MACD higher high
        """
        try:
            if len(df) < lookback:
                return SignalDirection.NEUTRAL, "none"

            price = df["c"].iloc[-lookback:].values
            macd = result.macd_line.iloc[-lookback:].values

            # Find local extrema (simple approach)
            price_max_idx = np.argmax(price)
            price_min_idx = np.argmin(price)
            macd_max_idx = np.argmax(macd)
            macd_min_idx = np.argmin(macd)

            # Current values vs lookback extrema
            price_curr = price[-1]
            macd_curr = macd[-1]

            # Regular bullish divergence
            if price_curr <= price[price_min_idx] and macd_curr > macd[macd_min_idx]:
                return SignalDirection.LONG, "regular"

            # Regular bearish divergence
            if price_curr >= price[price_max_idx] and macd_curr < macd[macd_max_idx]:
                return SignalDirection.SHORT, "regular"

            # Hidden bullish (trend continuation)
            if price_curr > price[price_min_idx] and macd_curr < macd[macd_min_idx]:
                return SignalDirection.LONG, "hidden"

            # Hidden bearish (trend continuation)
            if price_curr < price[price_max_idx] and macd_curr > macd[macd_max_idx]:
                return SignalDirection.SHORT, "hidden"

            return SignalDirection.NEUTRAL, "none"

        except Exception:
            return SignalDirection.NEUTRAL, "none"

    def calculate(self, df: pd.DataFrame, **kwargs) -> IndicatorSignal:
        """Calculate MACD signal."""
        if not self.validate_df(df, min_bars=self.slow + self.signal_period + 10):
            return IndicatorSignal(name=self.name)

        result = self._calculate_macd(df)
        if result is None:
            return IndicatorSignal(name=self.name)

        # Analyze components
        crossover_dir, is_crossover = self._detect_crossover(result)
        zero_dir, is_zero_cross = self._detect_zero_cross(result)
        hist_dir, hist_strength = self._analyze_histogram(result)
        div_dir, div_type = self._detect_divergence(df, result)

        # Combine signals
        signals = []
        confidences = []

        # Crossover signal (strongest)
        if is_crossover:
            signals.append(crossover_dir)
            confidences.append(0.8)

        # Zero cross
        if is_zero_cross:
            signals.append(zero_dir)
            confidences.append(0.6)

        # Histogram momentum
        if hist_dir != SignalDirection.NEUTRAL:
            signals.append(hist_dir)
            confidences.append(0.4 + hist_strength * 0.3)

        # Divergence (very strong)
        if div_dir != SignalDirection.NEUTRAL:
            signals.append(div_dir)
            conf = 0.9 if div_type == "regular" else 0.7
            confidences.append(conf)

        # Determine final direction
        if not signals:
            direction = crossover_dir  # Use underlying trend
            strength = SignalStrength.WEAK
            confidence = 0.2
        else:
            # Count votes
            long_votes = sum(1 for s in signals if s == SignalDirection.LONG)
            short_votes = sum(1 for s in signals if s == SignalDirection.SHORT)

            if long_votes > short_votes:
                direction = SignalDirection.LONG
            elif short_votes > long_votes:
                direction = SignalDirection.SHORT
            else:
                direction = SignalDirection.NEUTRAL

            # Calculate confidence
            confidence = sum(confidences) / len(confidences) if confidences else 0.0

            # Determine strength
            if is_crossover or div_type == "regular":
                strength = SignalStrength.STRONG
            elif is_zero_cross or div_type == "hidden":
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK

        return IndicatorSignal(
            name=self.name,
            direction=direction,
            strength=strength,
            value=result.macd_current,
            confidence=min(1.0, confidence),
            raw_values={
                "macd": result.macd_current,
                "signal": result.signal_current,
                "histogram": result.histogram_current,
                "histogram_prev": result.histogram_prev,
            },
            metadata={
                "crossover": is_crossover,
                "zero_cross": is_zero_cross,
                "divergence": div_type,
                "hist_direction": hist_dir.value,
            },
        )
