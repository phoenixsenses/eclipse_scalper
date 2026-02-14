# strategies/indicators/bollinger.py — SCALPER ETERNAL — BOLLINGER BANDS — 2026 v1.0
# Bollinger Bands indicator with squeeze, expansion, and band touch signals.
#
# Signals:
# - Squeeze (low volatility, potential breakout)
# - Expansion (high volatility, trend confirmation)
# - Band touch/walk
# - %B position (where price is within bands)

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
class BollingerResult:
    """Bollinger Bands calculation result."""
    upper: pd.Series
    middle: pd.Series
    lower: pd.Series
    width: pd.Series
    percent_b: pd.Series
    upper_current: float
    middle_current: float
    lower_current: float
    width_current: float
    width_pct: float
    percent_b_current: float


class BollingerIndicator(BaseIndicator):
    """
    Bollinger Bands indicator with squeeze and expansion detection.

    Usage:
        bb = BollingerIndicator(period=20, std_dev=2.0)
        signal = bb.calculate(df)

        if signal.metadata.get("squeeze"):
            print("Volatility squeeze detected - potential breakout")
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        squeeze_threshold: float = 0.015,  # BB width % for squeeze
        expansion_threshold: float = 0.04,  # BB width % for expansion
        weight: float = 1.0,
    ):
        super().__init__("Bollinger", weight)
        self.period = period
        self.std_dev = std_dev
        self.squeeze_threshold = squeeze_threshold
        self.expansion_threshold = expansion_threshold

    def _calculate_bands(self, df: pd.DataFrame) -> Optional[BollingerResult]:
        """Calculate Bollinger Bands."""
        try:
            close = df["c"]

            # Middle band (SMA)
            middle = close.rolling(window=self.period).mean()

            # Standard deviation
            std = close.rolling(window=self.period).std()

            # Upper and lower bands
            upper = middle + (std * self.std_dev)
            lower = middle - (std * self.std_dev)

            # Band width
            width = upper - lower

            # Width as percentage of middle
            width_pct = (width / middle).fillna(0)

            # Percent B (%B) - where price is within the bands
            # 0 = at lower band, 1 = at upper band, 0.5 = at middle
            percent_b = (close - lower) / (upper - lower)
            percent_b = percent_b.fillna(0.5)

            return BollingerResult(
                upper=upper,
                middle=middle,
                lower=lower,
                width=width,
                percent_b=percent_b,
                upper_current=self.safe_float(upper.iloc[-1]),
                middle_current=self.safe_float(middle.iloc[-1]),
                lower_current=self.safe_float(lower.iloc[-1]),
                width_current=self.safe_float(width.iloc[-1]),
                width_pct=self.safe_float(width_pct.iloc[-1]),
                percent_b_current=self.safe_float(percent_b.iloc[-1]),
            )
        except Exception:
            return None

    def _detect_squeeze(self, result: BollingerResult) -> Tuple[bool, float]:
        """Detect Bollinger Band squeeze (low volatility)."""
        try:
            width_pct = result.width_pct

            if len(result.width) < 20:
                return False, 0.0

            # Current width vs recent average
            recent_avg = result.width.iloc[-20:].mean()
            current = result.width_current

            is_squeeze = width_pct < self.squeeze_threshold

            # Squeeze intensity (how tight compared to recent)
            if recent_avg > 0 and is_squeeze:
                intensity = 1.0 - (current / recent_avg)
                return True, max(0.0, min(1.0, intensity))

            return is_squeeze, 0.0

        except Exception:
            return False, 0.0

    def _detect_expansion(self, result: BollingerResult) -> Tuple[bool, SignalDirection]:
        """Detect band expansion (breakout confirmation)."""
        try:
            width = result.width

            if len(width) < 5:
                return False, SignalDirection.NEUTRAL

            # Check if width is expanding
            w0 = width.iloc[-1]
            w1 = width.iloc[-2]
            w2 = width.iloc[-3]

            is_expanding = w0 > w1 > w2

            if is_expanding and result.width_pct > self.expansion_threshold:
                # Determine direction based on %B
                if result.percent_b_current > 0.8:
                    return True, SignalDirection.LONG
                elif result.percent_b_current < 0.2:
                    return True, SignalDirection.SHORT
                return True, SignalDirection.NEUTRAL

            return False, SignalDirection.NEUTRAL

        except Exception:
            return False, SignalDirection.NEUTRAL

    def _detect_band_touch(self, df: pd.DataFrame, result: BollingerResult) -> Tuple[str, SignalDirection]:
        """Detect band touch or walk."""
        try:
            close = df["c"].iloc[-1]
            high = df["h"].iloc[-1]
            low = df["l"].iloc[-1]

            upper = result.upper_current
            lower = result.lower_current

            # Check for band touch
            if high >= upper:
                return "upper_touch", SignalDirection.SHORT  # Potential reversal
            if low <= lower:
                return "lower_touch", SignalDirection.LONG  # Potential reversal

            # Check for band walk (strong trend)
            if len(df) >= 3:
                last_3_highs = df["h"].iloc[-3:]
                last_3_lows = df["l"].iloc[-3:]
                upper_3 = result.upper.iloc[-3:]
                lower_3 = result.lower.iloc[-3:]

                # Walking upper band (bullish trend)
                if all(last_3_highs.values >= upper_3.values * 0.995):
                    return "upper_walk", SignalDirection.LONG

                # Walking lower band (bearish trend)
                if all(last_3_lows.values <= lower_3.values * 1.005):
                    return "lower_walk", SignalDirection.SHORT

            return "none", SignalDirection.NEUTRAL

        except Exception:
            return "none", SignalDirection.NEUTRAL

    def _analyze_percent_b(self, result: BollingerResult) -> Tuple[SignalDirection, float]:
        """Analyze %B for mean reversion or trend signals."""
        try:
            pb = result.percent_b_current

            # Extreme readings
            if pb > 1.0:
                # Above upper band - overbought
                return SignalDirection.SHORT, 0.7
            elif pb < 0.0:
                # Below lower band - oversold
                return SignalDirection.LONG, 0.7
            elif pb > 0.8:
                # Near upper band
                return SignalDirection.SHORT, 0.4
            elif pb < 0.2:
                # Near lower band
                return SignalDirection.LONG, 0.4
            else:
                # Middle zone
                return SignalDirection.NEUTRAL, 0.0

        except Exception:
            return SignalDirection.NEUTRAL, 0.0

    def calculate(self, df: pd.DataFrame, **kwargs) -> IndicatorSignal:
        """Calculate Bollinger Bands signal."""
        if not self.validate_df(df, min_bars=self.period + 10):
            return IndicatorSignal(name=self.name)

        result = self._calculate_bands(df)
        if result is None:
            return IndicatorSignal(name=self.name)

        # Analyze components
        is_squeeze, squeeze_intensity = self._detect_squeeze(result)
        is_expansion, expansion_dir = self._detect_expansion(result)
        band_touch, touch_dir = self._detect_band_touch(df, result)
        pb_dir, pb_confidence = self._analyze_percent_b(result)

        # Determine signal
        signals = []
        confidences = []

        # Squeeze (anticipation, no direction yet)
        if is_squeeze:
            # During squeeze, prepare for breakout but no direction
            signals.append(SignalDirection.NEUTRAL)
            confidences.append(0.3)

        # Expansion (breakout confirmation)
        if is_expansion and expansion_dir != SignalDirection.NEUTRAL:
            signals.append(expansion_dir)
            confidences.append(0.8)

        # Band touch/walk
        if band_touch == "upper_walk":
            signals.append(SignalDirection.LONG)
            confidences.append(0.7)
        elif band_touch == "lower_walk":
            signals.append(SignalDirection.SHORT)
            confidences.append(0.7)
        elif band_touch in ("upper_touch", "lower_touch"):
            signals.append(touch_dir)
            confidences.append(0.5)

        # %B analysis
        if pb_dir != SignalDirection.NEUTRAL:
            signals.append(pb_dir)
            confidences.append(pb_confidence)

        # Determine final direction
        if not signals or all(s == SignalDirection.NEUTRAL for s in signals):
            direction = SignalDirection.NEUTRAL
            confidence = 0.2 if is_squeeze else 0.0
            strength = SignalStrength.WEAK if is_squeeze else SignalStrength.NONE
        else:
            long_count = sum(1 for s in signals if s == SignalDirection.LONG)
            short_count = sum(1 for s in signals if s == SignalDirection.SHORT)

            if long_count > short_count:
                direction = SignalDirection.LONG
            elif short_count > long_count:
                direction = SignalDirection.SHORT
            else:
                direction = SignalDirection.NEUTRAL

            confidence = sum(confidences) / len(confidences) if confidences else 0.0

            if is_expansion or "walk" in band_touch:
                strength = SignalStrength.STRONG
            elif "touch" in band_touch:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK

        return IndicatorSignal(
            name=self.name,
            direction=direction,
            strength=strength,
            value=result.percent_b_current,
            confidence=min(1.0, confidence),
            raw_values={
                "upper": result.upper_current,
                "middle": result.middle_current,
                "lower": result.lower_current,
                "width": result.width_current,
                "width_pct": result.width_pct,
                "percent_b": result.percent_b_current,
            },
            metadata={
                "squeeze": is_squeeze,
                "squeeze_intensity": squeeze_intensity,
                "expansion": is_expansion,
                "band_touch": band_touch,
            },
        )
