# strategies/indicators/adx.py — SCALPER ETERNAL — ADX INDICATOR — 2026 v1.0
# Average Directional Index for trend strength measurement.

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
class ADXResult:
    """ADX calculation result."""
    adx: pd.Series
    plus_di: pd.Series
    minus_di: pd.Series
    adx_current: float
    plus_di_current: float
    minus_di_current: float
    adx_prev: float


class ADXIndicator(BaseIndicator):
    """
    ADX indicator for trend strength and direction.

    Usage:
        adx = ADXIndicator(period=14, trend_threshold=25)
        signal = adx.calculate(df)

        if signal.metadata.get("trending"):
            print("Strong trend detected")
    """

    def __init__(
        self,
        period: int = 14,
        trend_threshold: float = 25.0,
        strong_trend: float = 40.0,
        weight: float = 1.0,
    ):
        super().__init__("ADX", weight)
        self.period = period
        self.trend_threshold = trend_threshold
        self.strong_trend = strong_trend

    def _calculate_adx(self, df: pd.DataFrame) -> Optional[ADXResult]:
        """Calculate ADX with +DI and -DI."""
        try:
            high = df["h"]
            low = df["l"]
            close = df["c"]

            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Directional Movement
            plus_dm = high.diff()
            minus_dm = -low.diff()

            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

            # Smoothed values
            atr = tr.ewm(span=self.period, adjust=False).mean()
            plus_di = 100 * (plus_dm.ewm(span=self.period, adjust=False).mean() / atr.replace(0, np.nan))
            minus_di = 100 * (minus_dm.ewm(span=self.period, adjust=False).mean() / atr.replace(0, np.nan))

            # DX and ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx = dx.ewm(span=self.period, adjust=False).mean()

            return ADXResult(
                adx=adx,
                plus_di=plus_di,
                minus_di=minus_di,
                adx_current=self.safe_float(adx.iloc[-1]),
                plus_di_current=self.safe_float(plus_di.iloc[-1]),
                minus_di_current=self.safe_float(minus_di.iloc[-1]),
                adx_prev=self.safe_float(adx.iloc[-2]) if len(adx) > 1 else 0.0,
            )
        except Exception:
            return None

    def _analyze_trend(self, result: ADXResult) -> Tuple[bool, str]:
        """Analyze trend strength."""
        adx = result.adx_current

        if adx >= self.strong_trend:
            return True, "strong"
        elif adx >= self.trend_threshold:
            return True, "moderate"
        else:
            return False, "weak"

    def _get_direction(self, result: ADXResult) -> SignalDirection:
        """Get trend direction from +DI/-DI."""
        if result.plus_di_current > result.minus_di_current:
            return SignalDirection.LONG
        elif result.minus_di_current > result.plus_di_current:
            return SignalDirection.SHORT
        else:
            return SignalDirection.NEUTRAL

    def _detect_di_crossover(self, result: ADXResult) -> Tuple[bool, SignalDirection]:
        """Detect DI crossover."""
        try:
            plus_di = result.plus_di
            minus_di = result.minus_di

            if len(plus_di) < 2:
                return False, SignalDirection.NEUTRAL

            plus_curr, minus_curr = result.plus_di_current, result.minus_di_current
            plus_prev = self.safe_float(plus_di.iloc[-2])
            minus_prev = self.safe_float(minus_di.iloc[-2])

            # Bullish crossover: +DI crosses above -DI
            if plus_curr > minus_curr and plus_prev <= minus_prev:
                return True, SignalDirection.LONG

            # Bearish crossover: -DI crosses above +DI
            if minus_curr > plus_curr and minus_prev <= plus_prev:
                return True, SignalDirection.SHORT

            return False, SignalDirection.NEUTRAL
        except Exception:
            return False, SignalDirection.NEUTRAL

    def calculate(self, df: pd.DataFrame, **kwargs) -> IndicatorSignal:
        """Calculate ADX signal."""
        if not self.validate_df(df, min_bars=self.period * 2 + 10):
            return IndicatorSignal(name=self.name)

        result = self._calculate_adx(df)
        if result is None:
            return IndicatorSignal(name=self.name)

        is_trending, trend_strength = self._analyze_trend(result)
        direction = self._get_direction(result)
        is_crossover, cross_dir = self._detect_di_crossover(result)

        # ADX rising (trend strengthening)
        adx_rising = result.adx_current > result.adx_prev

        # Confidence based on trend strength and ADX value
        if trend_strength == "strong":
            confidence = 0.9
            strength = SignalStrength.STRONG
        elif trend_strength == "moderate":
            confidence = 0.6
            strength = SignalStrength.MODERATE
        else:
            confidence = 0.3
            strength = SignalStrength.WEAK

        # Boost confidence for DI crossover
        if is_crossover:
            confidence = min(1.0, confidence + 0.2)
            direction = cross_dir

        # If not trending, reduce confidence
        if not is_trending:
            confidence *= 0.5
            direction = SignalDirection.NEUTRAL

        return IndicatorSignal(
            name=self.name,
            direction=direction,
            strength=strength,
            value=result.adx_current,
            confidence=confidence,
            raw_values={
                "adx": result.adx_current,
                "plus_di": result.plus_di_current,
                "minus_di": result.minus_di_current,
            },
            metadata={
                "trending": is_trending,
                "trend_strength": trend_strength,
                "adx_rising": adx_rising,
                "di_crossover": is_crossover,
            },
        )
