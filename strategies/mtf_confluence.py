# strategies/mtf_confluence.py — SCALPER ETERNAL — MTF CONFLUENCE — 2026 v1.0
# Multi-timeframe analysis with weighted scoring and conflict resolution.
#
# Timeframe roles:
# - 1m: Entry timing, micro structure
# - 5m: Trend confirmation, momentum
# - 15m: Bias direction, swing structure
# - 1h: Major trend, support/resistance
#
# Conflict handling:
# - All aligned: Full weight
# - 1m vs higher TF: Reduce confidence 30%
# - 5m vs 15m/1h: Reduce confidence 50%
# - Complete conflict: No trade

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

from utils.logging import log


class TrendDirection(Enum):
    """Trend direction."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class MarketStructure(Enum):
    """Market structure type."""
    HIGHER_HIGH = "HIGHER_HIGH"
    LOWER_LOW = "LOWER_LOW"
    HIGHER_LOW = "HIGHER_LOW"
    LOWER_HIGH = "LOWER_HIGH"
    RANGE = "RANGE"
    UNKNOWN = "UNKNOWN"


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe."""
    timeframe: str
    trend: TrendDirection
    momentum: float  # -1 to 1
    structure: MarketStructure
    ema_fast: float
    ema_slow: float
    close: float
    atr: float
    atr_pct: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MTFResult:
    """Multi-timeframe confluence result."""
    direction: TrendDirection
    confluence_score: float  # 0-1
    alignment: float  # 0-1, how aligned are timeframes
    conflict_level: float  # 0-1, how much disagreement
    signals: Dict[str, TimeframeSignal]
    entry_timing_ok: bool
    bias_direction: TrendDirection
    major_trend: TrendDirection
    recommended_action: str  # "TRADE", "WAIT", "SKIP"
    confidence_multiplier: float  # 0-1, apply to final confidence
    reasons: List[str] = field(default_factory=list)


class MTFConfluence:
    """
    Multi-timeframe confluence analyzer.

    Usage:
        mtf = MTFConfluence()

        # Analyze with multiple DataFrames
        result = mtf.analyze({
            "1m": df_1m,
            "5m": df_5m,
            "15m": df_15m,
            "1h": df_1h,
        })

        if result.recommended_action == "TRADE":
            print(f"Trade {result.direction.value} with {result.confluence_score:.2f} confluence")
    """

    # Timeframe weights (importance for signal generation)
    WEIGHTS = {
        "1m": 0.15,   # Entry timing
        "5m": 0.30,   # Trend confirmation
        "15m": 0.35,  # Bias direction
        "1h": 0.20,   # Major trend
    }

    # EMA periods for each timeframe
    EMA_FAST = {"1m": 20, "5m": 20, "15m": 20, "1h": 20}
    EMA_SLOW = {"1m": 50, "5m": 50, "15m": 50, "1h": 50}

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        require_alignment: int = 2,  # Minimum TFs that must agree
        conflict_threshold: float = 0.5,  # Max conflict before skipping
    ):
        self.weights = weights or self.WEIGHTS
        self.require_alignment = require_alignment
        self.conflict_threshold = conflict_threshold

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert to float."""
        try:
            v = float(value)
            return default if not np.isfinite(v) else v
        except (ValueError, TypeError):
            return default

    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate EMA."""
        return series.ewm(span=period, adjust=False).mean()

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        high = df["h"]
        low = df["l"]
        close = df["c"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.rolling(period).mean()

    def _detect_structure(self, df: pd.DataFrame, lookback: int = 20) -> MarketStructure:
        """Detect market structure from swing highs/lows."""
        try:
            if len(df) < lookback + 5:
                return MarketStructure.UNKNOWN

            highs = df["h"].iloc[-lookback:]
            lows = df["l"].iloc[-lookback:]

            # Find swing points (simplified)
            swing_high_1 = highs.iloc[-10:].max()
            swing_high_2 = highs.iloc[-20:-10].max()
            swing_low_1 = lows.iloc[-10:].min()
            swing_low_2 = lows.iloc[-20:-10].min()

            # Determine structure
            if swing_high_1 > swing_high_2 and swing_low_1 > swing_low_2:
                return MarketStructure.HIGHER_HIGH  # Uptrend
            elif swing_high_1 < swing_high_2 and swing_low_1 < swing_low_2:
                return MarketStructure.LOWER_LOW  # Downtrend
            elif swing_high_1 < swing_high_2 and swing_low_1 > swing_low_2:
                return MarketStructure.RANGE  # Compression
            elif swing_high_1 > swing_high_2 and swing_low_1 < swing_low_2:
                return MarketStructure.RANGE  # Expansion
            else:
                return MarketStructure.UNKNOWN

        except Exception:
            return MarketStructure.UNKNOWN

    def _calculate_momentum(self, df: pd.DataFrame, period: int = 10) -> float:
        """Calculate normalized momentum (-1 to 1)."""
        try:
            close = df["c"]
            if len(close) < period + 1:
                return 0.0

            pct_change = close.pct_change(period).iloc[-1]
            # Normalize to -1 to 1 (assuming max 5% move)
            normalized = max(-1.0, min(1.0, pct_change / 0.05))
            return self._safe_float(normalized)

        except Exception:
            return 0.0

    def _analyze_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str,
    ) -> TimeframeSignal:
        """Analyze a single timeframe."""
        try:
            if df is None or len(df) < 50:
                return TimeframeSignal(
                    timeframe=timeframe,
                    trend=TrendDirection.NEUTRAL,
                    momentum=0.0,
                    structure=MarketStructure.UNKNOWN,
                    ema_fast=0.0,
                    ema_slow=0.0,
                    close=0.0,
                    atr=0.0,
                    atr_pct=0.0,
                    confidence=0.0,
                )

            close = df["c"]
            close_current = self._safe_float(close.iloc[-1])

            # EMAs
            fast_period = self.EMA_FAST.get(timeframe, 20)
            slow_period = self.EMA_SLOW.get(timeframe, 50)

            ema_fast = self._calculate_ema(close, fast_period)
            ema_slow = self._calculate_ema(close, slow_period)

            ema_fast_curr = self._safe_float(ema_fast.iloc[-1])
            ema_slow_curr = self._safe_float(ema_slow.iloc[-1])

            # Trend direction
            if ema_fast_curr > ema_slow_curr and close_current > ema_fast_curr:
                trend = TrendDirection.BULLISH
            elif ema_fast_curr < ema_slow_curr and close_current < ema_fast_curr:
                trend = TrendDirection.BEARISH
            else:
                trend = TrendDirection.NEUTRAL

            # Momentum
            momentum = self._calculate_momentum(df)

            # Structure
            structure = self._detect_structure(df)

            # ATR
            atr_series = self._calculate_atr(df)
            atr = self._safe_float(atr_series.iloc[-1])
            atr_pct = atr / close_current if close_current > 0 else 0.0

            # Confidence based on alignment
            confidence = 0.0
            if trend == TrendDirection.BULLISH:
                if momentum > 0.1 and structure in (MarketStructure.HIGHER_HIGH, MarketStructure.HIGHER_LOW):
                    confidence = 0.9
                elif momentum > 0:
                    confidence = 0.6
                else:
                    confidence = 0.3
            elif trend == TrendDirection.BEARISH:
                if momentum < -0.1 and structure in (MarketStructure.LOWER_LOW, MarketStructure.LOWER_HIGH):
                    confidence = 0.9
                elif momentum < 0:
                    confidence = 0.6
                else:
                    confidence = 0.3

            return TimeframeSignal(
                timeframe=timeframe,
                trend=trend,
                momentum=momentum,
                structure=structure,
                ema_fast=ema_fast_curr,
                ema_slow=ema_slow_curr,
                close=close_current,
                atr=atr,
                atr_pct=atr_pct,
                confidence=confidence,
            )

        except Exception as e:
            log.warning(f"[mtf] Error analyzing {timeframe}: {e}")
            return TimeframeSignal(
                timeframe=timeframe,
                trend=TrendDirection.NEUTRAL,
                momentum=0.0,
                structure=MarketStructure.UNKNOWN,
                ema_fast=0.0,
                ema_slow=0.0,
                close=0.0,
                atr=0.0,
                atr_pct=0.0,
                confidence=0.0,
            )

    def _calculate_alignment(
        self,
        signals: Dict[str, TimeframeSignal],
    ) -> Tuple[float, TrendDirection]:
        """Calculate how aligned the timeframes are."""
        bullish_weight = 0.0
        bearish_weight = 0.0
        total_weight = 0.0

        for tf, signal in signals.items():
            weight = self.weights.get(tf, 0.25)
            total_weight += weight

            if signal.trend == TrendDirection.BULLISH:
                bullish_weight += weight * signal.confidence
            elif signal.trend == TrendDirection.BEARISH:
                bearish_weight += weight * signal.confidence

        if total_weight == 0:
            return 0.0, TrendDirection.NEUTRAL

        bullish_score = bullish_weight / total_weight
        bearish_score = bearish_weight / total_weight

        alignment = abs(bullish_score - bearish_score)

        if bullish_score > bearish_score:
            direction = TrendDirection.BULLISH
        elif bearish_score > bullish_score:
            direction = TrendDirection.BEARISH
        else:
            direction = TrendDirection.NEUTRAL

        return alignment, direction

    def _calculate_conflict(
        self,
        signals: Dict[str, TimeframeSignal],
    ) -> float:
        """Calculate conflict level between timeframes."""
        if len(signals) < 2:
            return 0.0

        # Check for direct conflicts
        trends = [s.trend for s in signals.values() if s.trend != TrendDirection.NEUTRAL]

        if not trends:
            return 0.0

        bullish = sum(1 for t in trends if t == TrendDirection.BULLISH)
        bearish = sum(1 for t in trends if t == TrendDirection.BEARISH)

        if bullish == 0 or bearish == 0:
            return 0.0

        # Conflict ratio
        conflict = min(bullish, bearish) / max(bullish, bearish)
        return conflict

    def _determine_action(
        self,
        signals: Dict[str, TimeframeSignal],
        alignment: float,
        conflict: float,
    ) -> Tuple[str, float, List[str]]:
        """Determine recommended action and confidence multiplier."""
        reasons = []

        # Get key signals
        signal_1m = signals.get("1m")
        signal_5m = signals.get("5m")
        signal_15m = signals.get("15m")
        signal_1h = signals.get("1h")

        # Check entry timing (1m)
        entry_timing_ok = True
        if signal_1m:
            if signal_1m.trend == TrendDirection.NEUTRAL:
                entry_timing_ok = False
                reasons.append("1m_no_momentum")

        # Check trend confirmation (5m)
        trend_confirmed = True
        if signal_5m and signal_5m.confidence < 0.4:
            trend_confirmed = False
            reasons.append("5m_weak_trend")

        # Conflict check
        if conflict > self.conflict_threshold:
            reasons.append(f"conflict_high_{conflict:.2f}")
            return "SKIP", 0.0, reasons

        # Alignment check
        if alignment < 0.3:
            reasons.append(f"alignment_low_{alignment:.2f}")
            return "WAIT", 0.5, reasons

        # Calculate confidence multiplier
        conf_mult = 1.0

        # 1m vs higher TF conflict
        if signal_1m and signal_5m:
            if signal_1m.trend != signal_5m.trend and signal_1m.trend != TrendDirection.NEUTRAL:
                conf_mult *= 0.7
                reasons.append("1m_vs_5m")

        # 5m vs 15m conflict
        if signal_5m and signal_15m:
            if signal_5m.trend != signal_15m.trend and signal_5m.trend != TrendDirection.NEUTRAL:
                conf_mult *= 0.5
                reasons.append("5m_vs_15m")

        # 15m vs 1h conflict
        if signal_15m and signal_1h:
            if signal_15m.trend != signal_1h.trend and signal_15m.trend != TrendDirection.NEUTRAL:
                conf_mult *= 0.6
                reasons.append("15m_vs_1h")

        # Perfect alignment bonus
        non_neutral = [s for s in signals.values() if s.trend != TrendDirection.NEUTRAL]
        if len(non_neutral) >= 3:
            trends = [s.trend for s in non_neutral]
            if all(t == trends[0] for t in trends):
                conf_mult = min(1.2, conf_mult * 1.2)
                reasons.append("perfect_alignment")

        if entry_timing_ok and trend_confirmed and conf_mult >= 0.5:
            return "TRADE", conf_mult, reasons
        elif conf_mult >= 0.3:
            return "WAIT", conf_mult, reasons
        else:
            return "SKIP", conf_mult, reasons

    def analyze(self, dfs: Dict[str, pd.DataFrame]) -> MTFResult:
        """
        Analyze multiple timeframes for confluence.

        Args:
            dfs: Dict of timeframe -> DataFrame
                 Keys should be "1m", "5m", "15m", "1h"

        Returns:
            MTFResult with confluence analysis
        """
        # Analyze each timeframe
        signals = {}
        for tf, df in dfs.items():
            signals[tf] = self._analyze_timeframe(df, tf)

        # Calculate alignment and direction
        alignment, direction = self._calculate_alignment(signals)

        # Calculate conflict
        conflict = self._calculate_conflict(signals)

        # Determine action
        action, conf_mult, reasons = self._determine_action(signals, alignment, conflict)

        # Get bias and major trend
        bias = signals.get("15m", signals.get("5m", TimeframeSignal(
            timeframe="", trend=TrendDirection.NEUTRAL, momentum=0.0,
            structure=MarketStructure.UNKNOWN, ema_fast=0.0, ema_slow=0.0,
            close=0.0, atr=0.0, atr_pct=0.0, confidence=0.0,
        ))).trend

        major = signals.get("1h", signals.get("15m", TimeframeSignal(
            timeframe="", trend=TrendDirection.NEUTRAL, momentum=0.0,
            structure=MarketStructure.UNKNOWN, ema_fast=0.0, ema_slow=0.0,
            close=0.0, atr=0.0, atr_pct=0.0, confidence=0.0,
        ))).trend

        # Entry timing check
        signal_1m = signals.get("1m")
        entry_ok = signal_1m is not None and signal_1m.trend != TrendDirection.NEUTRAL

        # Calculate confluence score
        total_conf = sum(s.confidence * self.weights.get(tf, 0.25) for tf, s in signals.items())
        confluence_score = total_conf * alignment * (1 - conflict) * conf_mult

        return MTFResult(
            direction=direction,
            confluence_score=min(1.0, confluence_score),
            alignment=alignment,
            conflict_level=conflict,
            signals=signals,
            entry_timing_ok=entry_ok,
            bias_direction=bias,
            major_trend=major,
            recommended_action=action,
            confidence_multiplier=conf_mult,
            reasons=reasons,
        )


# Singleton
_mtf_instance: Optional[MTFConfluence] = None


def get_mtf_confluence() -> MTFConfluence:
    """Get or create MTF confluence analyzer."""
    global _mtf_instance
    if _mtf_instance is None:
        _mtf_instance = MTFConfluence()
    return _mtf_instance
