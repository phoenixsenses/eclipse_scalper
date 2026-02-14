# strategies/signal_engine.py — SCALPER ETERNAL — SIGNAL ENGINE — 2026 v1.0
# Pluggable signal engine that combines multiple indicators with weighted confluence.
#
# Features:
# - Register/unregister indicators dynamically
# - Weighted confluence scoring
# - Conflict detection and resolution
# - Multi-timeframe support
# - Signal caching for performance

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

from utils.logging import log

from strategies.indicators import (
    BaseIndicator,
    IndicatorSignal,
    SignalDirection,
    SignalStrength,
    MACDIndicator,
    BollingerIndicator,
    RSIEnhancedIndicator,
    VWAPEnhancedIndicator,
    StochasticIndicator,
    ADXIndicator,
    MomentumIndicator,
)


class ConflictResolution(Enum):
    """How to handle conflicting signals."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    STRONGEST_SIGNAL = "strongest_signal"
    REQUIRE_CONSENSUS = "require_consensus"


@dataclass
class ConfluenceResult:
    """Result of confluence analysis."""
    direction: SignalDirection
    confidence: float
    strength: SignalStrength
    signals: Dict[str, IndicatorSignal]
    long_votes: float
    short_votes: float
    neutral_votes: float
    conflict_level: float  # 0-1, how much disagreement
    contributing_indicators: List[str]
    blocking_indicators: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid_signal(self, min_confidence: float = 0.5) -> bool:
        """Check if signal meets minimum criteria."""
        return (
            self.direction != SignalDirection.NEUTRAL
            and self.confidence >= min_confidence
            and self.conflict_level < 0.5
        )


@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    min_confidence: float = 0.5
    min_indicators: int = 2
    conflict_resolution: ConflictResolution = ConflictResolution.WEIGHTED_VOTE
    require_trend_alignment: bool = True
    require_momentum_confirmation: bool = True
    max_conflict_level: float = 0.5


class SignalEngine:
    """
    Modular signal engine with pluggable indicators.

    Usage:
        engine = SignalEngine()

        # Register indicators with weights
        engine.register(MACDIndicator(), weight=1.5)
        engine.register(RSIEnhancedIndicator(), weight=1.0)
        engine.register(BollingerIndicator(), weight=1.0)

        # Calculate confluence
        result = engine.calculate(df)

        if result.is_valid_signal():
            print(f"{result.direction.value} signal with {result.confidence:.2f} confidence")
    """

    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self.indicators: Dict[str, BaseIndicator] = {}
        self.weights: Dict[str, float] = {}
        self._cache: Dict[str, Tuple[float, ConfluenceResult]] = {}
        self._cache_ttl = 1.0  # seconds

    def register(
        self,
        indicator: BaseIndicator,
        weight: float = 1.0,
        enabled: bool = True,
    ) -> 'SignalEngine':
        """Register an indicator with optional weight."""
        name = indicator.name
        self.indicators[name] = indicator
        self.weights[name] = weight
        indicator.enabled = enabled
        return self

    def unregister(self, name: str) -> 'SignalEngine':
        """Unregister an indicator."""
        self.indicators.pop(name, None)
        self.weights.pop(name, None)
        return self

    def enable(self, name: str) -> 'SignalEngine':
        """Enable an indicator."""
        if name in self.indicators:
            self.indicators[name].enabled = True
        return self

    def disable(self, name: str) -> 'SignalEngine':
        """Disable an indicator."""
        if name in self.indicators:
            self.indicators[name].enabled = False
        return self

    def set_weight(self, name: str, weight: float) -> 'SignalEngine':
        """Set indicator weight."""
        if name in self.indicators:
            self.weights[name] = weight
        return self

    def _calculate_indicators(
        self,
        df: pd.DataFrame,
        **kwargs,
    ) -> Dict[str, IndicatorSignal]:
        """Calculate all enabled indicators."""
        signals = {}

        for name, indicator in self.indicators.items():
            if not indicator.enabled:
                continue

            try:
                signal = indicator.calculate(df, **kwargs)
                signals[name] = signal
            except Exception as e:
                log.warning(f"[signal_engine] {name} calculation failed: {e}")
                signals[name] = IndicatorSignal(name=name)

        return signals

    def _resolve_conflicts(
        self,
        signals: Dict[str, IndicatorSignal],
    ) -> Tuple[SignalDirection, float, float, float, float]:
        """
        Resolve conflicting signals using configured resolution method.

        Returns: (direction, confidence, long_votes, short_votes, conflict_level)
        """
        long_votes = 0.0
        short_votes = 0.0
        neutral_votes = 0.0
        total_weight = 0.0
        confidences = []

        for name, signal in signals.items():
            weight = self.weights.get(name, 1.0)
            total_weight += weight

            if signal.direction == SignalDirection.LONG:
                long_votes += weight * signal.confidence
                confidences.append((SignalDirection.LONG, signal.confidence * weight))
            elif signal.direction == SignalDirection.SHORT:
                short_votes += weight * signal.confidence
                confidences.append((SignalDirection.SHORT, signal.confidence * weight))
            else:
                neutral_votes += weight

        # Normalize votes
        if total_weight > 0:
            long_votes /= total_weight
            short_votes /= total_weight
            neutral_votes /= total_weight

        # Calculate conflict level
        max_votes = max(long_votes, short_votes)
        min_votes = min(long_votes, short_votes)
        conflict_level = min_votes / max_votes if max_votes > 0 else 0.0

        # Resolve based on method
        if self.config.conflict_resolution == ConflictResolution.MAJORITY_VOTE:
            if long_votes > short_votes and long_votes > neutral_votes:
                direction = SignalDirection.LONG
                confidence = long_votes
            elif short_votes > long_votes and short_votes > neutral_votes:
                direction = SignalDirection.SHORT
                confidence = short_votes
            else:
                direction = SignalDirection.NEUTRAL
                confidence = 0.0

        elif self.config.conflict_resolution == ConflictResolution.WEIGHTED_VOTE:
            if long_votes > short_votes * 1.2:  # Need 20% more weight
                direction = SignalDirection.LONG
                confidence = long_votes * (1 - conflict_level)
            elif short_votes > long_votes * 1.2:
                direction = SignalDirection.SHORT
                confidence = short_votes * (1 - conflict_level)
            else:
                direction = SignalDirection.NEUTRAL
                confidence = 0.0

        elif self.config.conflict_resolution == ConflictResolution.STRONGEST_SIGNAL:
            if confidences:
                best_dir, best_conf = max(confidences, key=lambda x: x[1])
                direction = best_dir
                confidence = best_conf / total_weight if total_weight > 0 else 0.0
            else:
                direction = SignalDirection.NEUTRAL
                confidence = 0.0

        elif self.config.conflict_resolution == ConflictResolution.REQUIRE_CONSENSUS:
            # All non-neutral signals must agree
            directions = [
                s.direction for s in signals.values()
                if s.direction != SignalDirection.NEUTRAL
            ]
            if directions and all(d == directions[0] for d in directions):
                direction = directions[0]
                confidence = (long_votes + short_votes) / 2
            else:
                direction = SignalDirection.NEUTRAL
                confidence = 0.0

        else:
            direction = SignalDirection.NEUTRAL
            confidence = 0.0

        return direction, confidence, long_votes, short_votes, conflict_level

    def _determine_strength(
        self,
        signals: Dict[str, IndicatorSignal],
        direction: SignalDirection,
    ) -> SignalStrength:
        """Determine overall signal strength."""
        if direction == SignalDirection.NEUTRAL:
            return SignalStrength.NONE

        # Count strong signals in same direction
        strong_count = sum(
            1 for s in signals.values()
            if s.direction == direction and s.strength == SignalStrength.STRONG
        )
        moderate_count = sum(
            1 for s in signals.values()
            if s.direction == direction and s.strength == SignalStrength.MODERATE
        )

        if strong_count >= 2:
            return SignalStrength.STRONG
        elif strong_count >= 1 or moderate_count >= 2:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK

    def _identify_contributors(
        self,
        signals: Dict[str, IndicatorSignal],
        direction: SignalDirection,
    ) -> Tuple[List[str], List[str]]:
        """Identify contributing and blocking indicators."""
        contributors = []
        blockers = []

        for name, signal in signals.items():
            if signal.direction == direction:
                contributors.append(name)
            elif signal.direction != SignalDirection.NEUTRAL:
                blockers.append(name)

        return contributors, blockers

    def calculate(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        **kwargs,
    ) -> ConfluenceResult:
        """
        Calculate confluence signal from all indicators.

        Args:
            df: OHLCV DataFrame
            symbol: Trading symbol (for caching)
            **kwargs: Additional parameters passed to indicators

        Returns:
            ConfluenceResult with direction, confidence, and metadata
        """
        # Check cache
        cache_key = f"{symbol}:{len(df)}"
        if cache_key in self._cache:
            cache_time, cached_result = self._cache[cache_key]
            if time.time() - cache_time < self._cache_ttl:
                return cached_result

        # Calculate all indicators
        signals = self._calculate_indicators(df, **kwargs)

        # Check minimum indicators
        active_signals = {
            k: v for k, v in signals.items()
            if v.direction != SignalDirection.NEUTRAL or v.confidence > 0
        }

        if len(active_signals) < self.config.min_indicators:
            result = ConfluenceResult(
                direction=SignalDirection.NEUTRAL,
                confidence=0.0,
                strength=SignalStrength.NONE,
                signals=signals,
                long_votes=0.0,
                short_votes=0.0,
                neutral_votes=1.0,
                conflict_level=0.0,
                contributing_indicators=[],
                blocking_indicators=[],
                metadata={"reason": "insufficient_indicators"},
            )
            self._cache[cache_key] = (time.time(), result)
            return result

        # Resolve conflicts
        direction, confidence, long_votes, short_votes, conflict_level = (
            self._resolve_conflicts(signals)
        )

        # Check conflict level
        if conflict_level > self.config.max_conflict_level:
            direction = SignalDirection.NEUTRAL
            confidence = 0.0

        # Determine strength
        strength = self._determine_strength(signals, direction)

        # Identify contributors/blockers
        contributors, blockers = self._identify_contributors(signals, direction)

        # Apply additional filters
        if self.config.require_trend_alignment:
            adx_signal = signals.get("ADX")
            if adx_signal and not adx_signal.metadata.get("trending", True):
                confidence *= 0.7

        if self.config.require_momentum_confirmation:
            mom_signal = signals.get("Momentum")
            if mom_signal and mom_signal.direction != direction:
                confidence *= 0.6

        result = ConfluenceResult(
            direction=direction,
            confidence=min(1.0, confidence),
            strength=strength,
            signals=signals,
            long_votes=long_votes,
            short_votes=short_votes,
            neutral_votes=1.0 - long_votes - short_votes,
            conflict_level=conflict_level,
            contributing_indicators=contributors,
            blocking_indicators=blockers,
            metadata={
                "indicator_count": len(active_signals),
                "total_indicators": len(self.indicators),
            },
        )

        self._cache[cache_key] = (time.time(), result)
        return result

    def get_signal_summary(self, result: ConfluenceResult) -> str:
        """Get human-readable signal summary."""
        lines = [
            f"Signal: {result.direction.value}",
            f"Confidence: {result.confidence:.2f}",
            f"Strength: {result.strength.value}",
            f"Conflict: {result.conflict_level:.2f}",
            f"",
            f"Votes: L={result.long_votes:.2f} S={result.short_votes:.2f}",
            f"",
            f"Contributors: {', '.join(result.contributing_indicators)}",
            f"Blockers: {', '.join(result.blocking_indicators)}",
        ]
        return "\n".join(lines)


# Default engine factory
def create_default_engine(config: Optional[SignalConfig] = None) -> SignalEngine:
    """Create signal engine with default indicators."""
    engine = SignalEngine(config)

    # Register indicators with tuned weights
    engine.register(MACDIndicator(), weight=1.5)
    engine.register(RSIEnhancedIndicator(), weight=1.2)
    engine.register(BollingerIndicator(), weight=1.0)
    engine.register(VWAPEnhancedIndicator(), weight=1.3)
    engine.register(StochasticIndicator(), weight=0.8)
    engine.register(ADXIndicator(), weight=1.0)
    engine.register(MomentumIndicator(), weight=1.4)

    return engine


# Singleton instance
_default_engine: Optional[SignalEngine] = None


def get_engine(config: Optional[SignalConfig] = None) -> SignalEngine:
    """Get or create default signal engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = create_default_engine(config)
    return _default_engine
