# strategies/indicators/vwap_enhanced.py — SCALPER ETERNAL — VWAP ENHANCED — 2026 v1.0
# Enhanced VWAP with bands, deviation analysis, and price location signals.
#
# Signals:
# - Price location (above/below VWAP)
# - Standard deviation bands
# - VWAP bounce/rejection
# - Distance-based entry signals

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
class VWAPResult:
    """VWAP calculation result."""
    vwap: pd.Series
    upper_1: pd.Series  # VWAP + 1σ
    lower_1: pd.Series  # VWAP - 1σ
    upper_2: pd.Series  # VWAP + 2σ
    lower_2: pd.Series  # VWAP - 2σ
    vwap_current: float
    upper_1_current: float
    lower_1_current: float
    upper_2_current: float
    lower_2_current: float
    deviation: float  # Current σ from VWAP
    distance_pct: float  # Distance as percentage


class VWAPEnhancedIndicator(BaseIndicator):
    """
    Enhanced VWAP with standard deviation bands.

    Usage:
        vwap = VWAPEnhancedIndicator(window=240)
        signal = vwap.calculate(df)

        if signal.metadata.get("bounce"):
            print("VWAP bounce detected")
    """

    def __init__(
        self,
        window: int = 240,
        min_distance_pct: float = 0.005,  # Minimum distance for signal
        band_touch_tolerance: float = 0.001,  # Tolerance for band touch
        weight: float = 1.0,
    ):
        super().__init__("VWAP", weight)
        self.window = window
        self.min_distance_pct = min_distance_pct
        self.band_touch_tolerance = band_touch_tolerance

    def _calculate_vwap(self, df: pd.DataFrame) -> Optional[VWAPResult]:
        """Calculate VWAP with bands."""
        try:
            # Typical price
            tp = (df["h"] + df["l"] + df["c"]) / 3.0

            # Price * Volume
            pv = tp * df["v"]

            # Rolling VWAP
            cum_pv = pv.rolling(self.window, min_periods=max(10, self.window // 5)).sum()
            cum_v = df["v"].rolling(self.window, min_periods=max(10, self.window // 5)).sum()

            vwap = cum_pv / cum_v.replace(0, np.nan)
            vwap = vwap.fillna(method="ffill")

            # Standard deviation for bands
            squared_diff = ((tp - vwap) ** 2) * df["v"]
            cum_sq_diff = squared_diff.rolling(self.window, min_periods=max(10, self.window // 5)).sum()
            variance = cum_sq_diff / cum_v.replace(0, np.nan)
            std_dev = np.sqrt(variance).fillna(0)

            # Bands
            upper_1 = vwap + std_dev
            lower_1 = vwap - std_dev
            upper_2 = vwap + (std_dev * 2)
            lower_2 = vwap - (std_dev * 2)

            # Current values
            vwap_curr = self.safe_float(vwap.iloc[-1])
            close_curr = self.safe_float(df["c"].iloc[-1])
            std_curr = self.safe_float(std_dev.iloc[-1])

            # Deviation from VWAP in σ units
            deviation = (close_curr - vwap_curr) / std_curr if std_curr > 0 else 0.0

            # Distance as percentage
            distance_pct = abs(close_curr - vwap_curr) / vwap_curr if vwap_curr > 0 else 0.0

            return VWAPResult(
                vwap=vwap,
                upper_1=upper_1,
                lower_1=lower_1,
                upper_2=upper_2,
                lower_2=lower_2,
                vwap_current=vwap_curr,
                upper_1_current=self.safe_float(upper_1.iloc[-1]),
                lower_1_current=self.safe_float(lower_1.iloc[-1]),
                upper_2_current=self.safe_float(upper_2.iloc[-1]),
                lower_2_current=self.safe_float(lower_2.iloc[-1]),
                deviation=deviation,
                distance_pct=distance_pct,
            )
        except Exception:
            return None

    def _analyze_location(self, df: pd.DataFrame, result: VWAPResult) -> Tuple[SignalDirection, str]:
        """Analyze price location relative to VWAP."""
        close = df["c"].iloc[-1]
        vwap = result.vwap_current

        if close > result.upper_2_current:
            return SignalDirection.SHORT, "above_2sigma"  # Extended, potential pullback
        elif close > result.upper_1_current:
            return SignalDirection.LONG, "above_1sigma"  # Strong trend
        elif close > vwap:
            return SignalDirection.LONG, "above_vwap"
        elif close < result.lower_2_current:
            return SignalDirection.LONG, "below_2sigma"  # Extended, potential bounce
        elif close < result.lower_1_current:
            return SignalDirection.SHORT, "below_1sigma"  # Weak trend
        else:
            return SignalDirection.SHORT, "below_vwap"

    def _detect_bounce(self, df: pd.DataFrame, result: VWAPResult) -> Tuple[bool, SignalDirection]:
        """Detect VWAP bounce pattern."""
        try:
            if len(df) < 5:
                return False, SignalDirection.NEUTRAL

            close = df["c"].iloc[-5:].values
            low = df["l"].iloc[-5:].values
            high = df["h"].iloc[-5:].values
            vwap = result.vwap.iloc[-5:].values

            # Bullish bounce: price touched VWAP from above and bounced
            touched_from_above = any(low[i] <= vwap[i] * 1.001 for i in range(len(low) - 1))
            bounced_up = close[-1] > vwap[-1] and close[-1] > close[-2]

            if touched_from_above and bounced_up:
                return True, SignalDirection.LONG

            # Bearish rejection: price touched VWAP from below and rejected
            touched_from_below = any(high[i] >= vwap[i] * 0.999 for i in range(len(high) - 1))
            rejected_down = close[-1] < vwap[-1] and close[-1] < close[-2]

            if touched_from_below and rejected_down:
                return True, SignalDirection.SHORT

            return False, SignalDirection.NEUTRAL

        except Exception:
            return False, SignalDirection.NEUTRAL

    def _detect_cross(self, df: pd.DataFrame, result: VWAPResult) -> Tuple[bool, SignalDirection]:
        """Detect VWAP cross."""
        try:
            if len(df) < 3:
                return False, SignalDirection.NEUTRAL

            close = df["c"].iloc[-3:].values
            vwap = result.vwap.iloc[-3:].values

            # Bullish cross: was below, now above
            was_below = close[-2] < vwap[-2]
            now_above = close[-1] > vwap[-1]

            if was_below and now_above:
                return True, SignalDirection.LONG

            # Bearish cross: was above, now below
            was_above = close[-2] > vwap[-2]
            now_below = close[-1] < vwap[-1]

            if was_above and now_below:
                return True, SignalDirection.SHORT

            return False, SignalDirection.NEUTRAL

        except Exception:
            return False, SignalDirection.NEUTRAL

    def _analyze_distance(self, result: VWAPResult) -> Tuple[bool, float]:
        """Analyze distance from VWAP for entry signal."""
        distance = result.distance_pct

        if distance >= self.min_distance_pct:
            # Normalize confidence based on distance
            # Higher distance = higher confidence (up to a point)
            confidence = min(1.0, distance / (self.min_distance_pct * 3))
            return True, confidence
        return False, 0.0

    def calculate(self, df: pd.DataFrame, **kwargs) -> IndicatorSignal:
        """Calculate enhanced VWAP signal."""
        if not self.validate_df(df, min_bars=max(50, self.window // 2)):
            return IndicatorSignal(name=self.name)

        result = self._calculate_vwap(df)
        if result is None:
            return IndicatorSignal(name=self.name)

        # Analyze components
        location_dir, location_zone = self._analyze_location(df, result)
        is_bounce, bounce_dir = self._detect_bounce(df, result)
        is_cross, cross_dir = self._detect_cross(df, result)
        has_distance, distance_conf = self._analyze_distance(result)

        # Combine signals
        signals = []
        confidences = []

        # VWAP cross (strong)
        if is_cross:
            signals.append(cross_dir)
            confidences.append(0.8)

        # VWAP bounce (strong)
        if is_bounce:
            signals.append(bounce_dir)
            confidences.append(0.75)

        # Location-based
        if location_zone in ("above_2sigma", "below_2sigma"):
            # Mean reversion opportunity
            signals.append(location_dir)
            confidences.append(0.6)
        elif has_distance:
            signals.append(location_dir)
            confidences.append(0.4 + distance_conf * 0.3)

        # Determine final direction
        if not signals:
            direction = location_dir
            confidence = 0.2
            strength = SignalStrength.WEAK
        else:
            long_count = sum(1 for s in signals if s == SignalDirection.LONG)
            short_count = sum(1 for s in signals if s == SignalDirection.SHORT)

            if long_count > short_count:
                direction = SignalDirection.LONG
            elif short_count > long_count:
                direction = SignalDirection.SHORT
            else:
                direction = location_dir

            confidence = sum(confidences) / len(confidences)

            if is_cross or is_bounce:
                strength = SignalStrength.STRONG
            elif has_distance:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK

        return IndicatorSignal(
            name=self.name,
            direction=direction,
            strength=strength,
            value=result.vwap_current,
            confidence=min(1.0, confidence),
            raw_values={
                "vwap": result.vwap_current,
                "upper_1": result.upper_1_current,
                "lower_1": result.lower_1_current,
                "upper_2": result.upper_2_current,
                "lower_2": result.lower_2_current,
                "deviation": result.deviation,
                "distance_pct": result.distance_pct,
            },
            metadata={
                "location": location_zone,
                "bounce": is_bounce,
                "cross": is_cross,
                "has_distance": has_distance,
            },
        )
