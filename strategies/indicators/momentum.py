# strategies/indicators/momentum.py — SCALPER ETERNAL — MOMENTUM INDICATOR — 2026 v1.0
# Multi-period momentum analysis with Heikin-Ashi smoothing.

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
class MomentumResult:
    """Momentum calculation result."""
    momentum: pd.Series
    ha_momentum: pd.Series
    momentum_current: float
    ha_momentum_current: float
    momentum_pct: float
    momentum_direction: str
    consecutive_bars: int


class MomentumIndicator(BaseIndicator):
    """
    Multi-period momentum with Heikin-Ashi smoothing.

    Usage:
        mom = MomentumIndicator(periods=[2, 5, 10])
        signal = mom.calculate(df)
    """

    def __init__(
        self,
        periods: List[int] = None,
        min_momentum_pct: float = 0.001,
        strong_momentum_pct: float = 0.003,
        weight: float = 1.0,
    ):
        super().__init__("Momentum", weight)
        self.periods = periods or [2, 5, 10]
        self.min_momentum_pct = min_momentum_pct
        self.strong_momentum_pct = strong_momentum_pct

    def _calculate_ha_close(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Heikin-Ashi close."""
        return (df["o"] + df["h"] + df["l"] + df["c"]) / 4.0

    def _calculate_momentum(self, df: pd.DataFrame) -> Optional[MomentumResult]:
        """Calculate multi-period momentum."""
        try:
            close = df["c"]
            ha_close = self._calculate_ha_close(df)

            # Primary momentum (2-bar)
            momentum = close.pct_change(2)
            ha_momentum = ha_close.pct_change(2)

            # Current values
            mom_curr = self.safe_float(momentum.iloc[-1])
            ha_mom_curr = self.safe_float(ha_momentum.iloc[-1])
            mom_pct = abs(mom_curr)

            # Direction
            if mom_curr > self.min_momentum_pct:
                direction = "bullish"
            elif mom_curr < -self.min_momentum_pct:
                direction = "bearish"
            else:
                direction = "neutral"

            # Consecutive bars in same direction
            consecutive = 0
            for i in range(2, min(10, len(momentum))):
                prev_mom = momentum.iloc[-i]
                if direction == "bullish" and prev_mom > 0:
                    consecutive += 1
                elif direction == "bearish" and prev_mom < 0:
                    consecutive += 1
                else:
                    break

            return MomentumResult(
                momentum=momentum,
                ha_momentum=ha_momentum,
                momentum_current=mom_curr,
                ha_momentum_current=ha_mom_curr,
                momentum_pct=mom_pct,
                momentum_direction=direction,
                consecutive_bars=consecutive,
            )
        except Exception:
            return None

    def _analyze_multi_period(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Analyze momentum across multiple periods."""
        try:
            close = df["c"]
            scores = []

            for period in self.periods:
                if len(close) <= period:
                    continue
                mom = close.pct_change(period).iloc[-1]
                if mom > self.min_momentum_pct:
                    scores.append(1.0)
                elif mom < -self.min_momentum_pct:
                    scores.append(-1.0)
                else:
                    scores.append(0.0)

            if not scores:
                return 0.0, 0.0

            # Average score (-1 to 1)
            avg_score = sum(scores) / len(scores)

            # Alignment (how many periods agree)
            alignment = abs(avg_score)

            return avg_score, alignment

        except Exception:
            return 0.0, 0.0

    def _detect_acceleration(self, result: MomentumResult) -> Tuple[bool, SignalDirection]:
        """Detect momentum acceleration."""
        try:
            momentum = result.momentum

            if len(momentum) < 5:
                return False, SignalDirection.NEUTRAL

            mom = momentum.iloc[-5:].values

            # Accelerating up
            if all(mom[i] > mom[i-1] for i in range(1, len(mom))) and mom[-1] > 0:
                return True, SignalDirection.LONG

            # Accelerating down
            if all(mom[i] < mom[i-1] for i in range(1, len(mom))) and mom[-1] < 0:
                return True, SignalDirection.SHORT

            return False, SignalDirection.NEUTRAL

        except Exception:
            return False, SignalDirection.NEUTRAL

    def calculate(self, df: pd.DataFrame, **kwargs) -> IndicatorSignal:
        """Calculate momentum signal."""
        if not self.validate_df(df, min_bars=max(self.periods) + 10):
            return IndicatorSignal(name=self.name)

        result = self._calculate_momentum(df)
        if result is None:
            return IndicatorSignal(name=self.name)

        multi_score, alignment = self._analyze_multi_period(df)
        is_accelerating, accel_dir = self._detect_acceleration(result)

        # Determine direction
        if result.momentum_direction == "bullish":
            direction = SignalDirection.LONG
        elif result.momentum_direction == "bearish":
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.NEUTRAL

        # Calculate confidence
        confidences = []

        # Base momentum confidence
        if result.momentum_pct >= self.strong_momentum_pct:
            confidences.append(0.8)
        elif result.momentum_pct >= self.min_momentum_pct:
            confidences.append(0.5)

        # Multi-period alignment
        if alignment >= 0.8:
            confidences.append(0.7)
        elif alignment >= 0.5:
            confidences.append(0.4)

        # Consecutive bars
        if result.consecutive_bars >= 3:
            confidences.append(0.6)
        elif result.consecutive_bars >= 2:
            confidences.append(0.3)

        # Acceleration
        if is_accelerating:
            confidences.append(0.8)
            direction = accel_dir

        confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Strength
        if is_accelerating and result.momentum_pct >= self.strong_momentum_pct:
            strength = SignalStrength.STRONG
        elif result.momentum_pct >= self.min_momentum_pct and alignment >= 0.5:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        return IndicatorSignal(
            name=self.name,
            direction=direction,
            strength=strength,
            value=result.momentum_current,
            confidence=min(1.0, confidence),
            raw_values={
                "momentum": result.momentum_current,
                "ha_momentum": result.ha_momentum_current,
                "momentum_pct": result.momentum_pct,
            },
            metadata={
                "direction": result.momentum_direction,
                "consecutive_bars": result.consecutive_bars,
                "multi_period_score": multi_score,
                "alignment": alignment,
                "accelerating": is_accelerating,
            },
        )
