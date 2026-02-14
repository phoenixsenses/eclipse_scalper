# strategies/enhanced_signal.py — SCALPER ETERNAL — ENHANCED SIGNAL — 2026 v1.0
# Integration layer combining Signal Engine + MTF Confluence for improved signal generation.
#
# Features:
# - Pluggable indicator framework (7 indicators)
# - Multi-timeframe confluence analysis
# - Weighted confidence scoring
# - Backward compatible with existing signal function

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from utils.logging import log

# Import new modules
from strategies.signal_engine import (
    SignalEngine,
    SignalConfig,
    ConfluenceResult,
    ConflictResolution,
    create_default_engine,
    SignalDirection,
    SignalStrength,
)
from strategies.mtf_confluence import (
    MTFConfluence,
    MTFResult,
    TrendDirection,
    get_mtf_confluence,
)


@dataclass
class EnhancedSignalResult:
    """Result from enhanced signal analysis."""
    long_signal: bool
    short_signal: bool
    confidence: float

    # Component results
    indicator_result: Optional[ConfluenceResult] = None
    mtf_result: Optional[MTFResult] = None

    # Metadata
    direction: str = "NEUTRAL"
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedSignal:
    """
    Enhanced signal generator combining multiple analysis systems.

    Usage:
        enhanced = EnhancedSignal()

        result = enhanced.analyze(
            df_1m=df_1m,
            df_5m=df_5m,
            df_15m=df_15m,
            df_1h=df_1h,
            symbol="BTCUSDT",
        )

        if result.long_signal:
            print(f"LONG signal with {result.confidence:.2f} confidence")
    """

    def __init__(
        self,
        indicator_config: Optional[SignalConfig] = None,
        mtf_weights: Optional[Dict[str, float]] = None,
        min_confluence: float = 0.5,
        min_mtf_alignment: float = 0.3,
        require_mtf_trade: bool = True,
    ):
        # Initialize signal engine
        self.engine = create_default_engine(indicator_config)

        # Initialize MTF analyzer
        self.mtf = MTFConfluence(weights=mtf_weights)

        # Thresholds
        self.min_confluence = min_confluence
        self.min_mtf_alignment = min_mtf_alignment
        self.require_mtf_trade = require_mtf_trade

    def _direction_to_signal(
        self,
        ind_direction: SignalDirection,
        mtf_direction: TrendDirection,
    ) -> Tuple[bool, bool]:
        """Convert directions to long/short signals."""
        # Both must agree for signal
        if ind_direction == SignalDirection.LONG and mtf_direction == TrendDirection.BULLISH:
            return True, False
        elif ind_direction == SignalDirection.SHORT and mtf_direction == TrendDirection.BEARISH:
            return False, True
        else:
            return False, False

    def _combine_confidence(
        self,
        ind_result: ConfluenceResult,
        mtf_result: MTFResult,
    ) -> float:
        """Combine confidence scores from indicator and MTF analysis."""
        # Base confidence from indicators
        ind_conf = ind_result.confidence

        # MTF factors
        mtf_conf_mult = mtf_result.confidence_multiplier
        mtf_alignment = mtf_result.alignment
        mtf_confluence = mtf_result.confluence_score

        # Combined confidence
        # Weight: 60% indicators, 40% MTF
        combined = (ind_conf * 0.6) + (mtf_confluence * 0.4)

        # Apply MTF multiplier (for conflicts)
        combined *= mtf_conf_mult

        # Boost for strong alignment
        if mtf_alignment > 0.8:
            combined *= 1.1

        # Penalty for high conflict
        if ind_result.conflict_level > 0.3:
            combined *= (1 - ind_result.conflict_level * 0.3)

        return min(1.0, max(0.0, combined))

    def analyze(
        self,
        df_1m: Optional[pd.DataFrame] = None,
        df_5m: Optional[pd.DataFrame] = None,
        df_15m: Optional[pd.DataFrame] = None,
        df_1h: Optional[pd.DataFrame] = None,
        symbol: str = "",
    ) -> EnhancedSignalResult:
        """
        Analyze price data using enhanced signal generation.

        Args:
            df_1m: 1-minute OHLCV DataFrame
            df_5m: 5-minute OHLCV DataFrame
            df_15m: 15-minute OHLCV DataFrame
            df_1h: 1-hour OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            EnhancedSignalResult with signals and confidence
        """
        reasons = []

        # Validate minimum data
        if df_1m is None or len(df_1m) < 50:
            return EnhancedSignalResult(
                long_signal=False,
                short_signal=False,
                confidence=0.0,
                reasons=["insufficient_1m_data"],
            )

        # Run indicator analysis on 1m data
        try:
            ind_result = self.engine.calculate(df_1m, symbol=symbol)
        except Exception as e:
            log.warning(f"[enhanced_signal] Indicator error: {e}")
            ind_result = ConfluenceResult(
                direction=SignalDirection.NEUTRAL,
                confidence=0.0,
                strength=SignalStrength.NONE,
                signals={},
                long_votes=0.0,
                short_votes=0.0,
                neutral_votes=1.0,
                conflict_level=0.0,
                contributing_indicators=[],
                blocking_indicators=[],
            )

        # Run MTF analysis
        mtf_dfs = {}
        if df_1m is not None and len(df_1m) >= 50:
            mtf_dfs["1m"] = df_1m
        if df_5m is not None and len(df_5m) >= 50:
            mtf_dfs["5m"] = df_5m
        if df_15m is not None and len(df_15m) >= 50:
            mtf_dfs["15m"] = df_15m
        if df_1h is not None and len(df_1h) >= 50:
            mtf_dfs["1h"] = df_1h

        try:
            mtf_result = self.mtf.analyze(mtf_dfs)
        except Exception as e:
            log.warning(f"[enhanced_signal] MTF error: {e}")
            mtf_result = MTFResult(
                direction=TrendDirection.NEUTRAL,
                confluence_score=0.0,
                alignment=0.0,
                conflict_level=0.0,
                signals={},
                entry_timing_ok=False,
                bias_direction=TrendDirection.NEUTRAL,
                major_trend=TrendDirection.NEUTRAL,
                recommended_action="SKIP",
                confidence_multiplier=0.0,
            )

        # Check MTF recommendation
        if self.require_mtf_trade and mtf_result.recommended_action != "TRADE":
            reasons.append(f"mtf_{mtf_result.recommended_action.lower()}")
            reasons.extend(mtf_result.reasons)
            return EnhancedSignalResult(
                long_signal=False,
                short_signal=False,
                confidence=0.0,
                indicator_result=ind_result,
                mtf_result=mtf_result,
                reasons=reasons,
            )

        # Check minimum alignment
        if mtf_result.alignment < self.min_mtf_alignment:
            reasons.append(f"alignment_low_{mtf_result.alignment:.2f}")
            return EnhancedSignalResult(
                long_signal=False,
                short_signal=False,
                confidence=0.0,
                indicator_result=ind_result,
                mtf_result=mtf_result,
                reasons=reasons,
            )

        # Check indicator confidence
        if ind_result.confidence < self.min_confluence:
            reasons.append(f"confluence_low_{ind_result.confidence:.2f}")
            return EnhancedSignalResult(
                long_signal=False,
                short_signal=False,
                confidence=0.0,
                indicator_result=ind_result,
                mtf_result=mtf_result,
                reasons=reasons,
            )

        # Convert directions to signals
        long_signal, short_signal = self._direction_to_signal(
            ind_result.direction,
            mtf_result.direction,
        )

        # If no agreement, no signal
        if not long_signal and not short_signal:
            # Check for partial agreement
            if ind_result.direction == SignalDirection.LONG:
                reasons.append("ind_long_mtf_not_bullish")
            elif ind_result.direction == SignalDirection.SHORT:
                reasons.append("ind_short_mtf_not_bearish")
            else:
                reasons.append("ind_neutral")

            return EnhancedSignalResult(
                long_signal=False,
                short_signal=False,
                confidence=0.0,
                indicator_result=ind_result,
                mtf_result=mtf_result,
                reasons=reasons,
            )

        # Calculate combined confidence
        confidence = self._combine_confidence(ind_result, mtf_result)

        # Build result
        direction = "LONG" if long_signal else "SHORT"
        reasons.append(f"signal_{direction.lower()}")
        reasons.extend([f"ind_{x}" for x in ind_result.contributing_indicators[:3]])
        reasons.extend(mtf_result.reasons[:2])

        return EnhancedSignalResult(
            long_signal=long_signal,
            short_signal=short_signal,
            confidence=round(confidence, 2),
            indicator_result=ind_result,
            mtf_result=mtf_result,
            direction=direction,
            reasons=reasons,
            metadata={
                "ind_confidence": ind_result.confidence,
                "ind_strength": ind_result.strength.value,
                "mtf_confluence": mtf_result.confluence_score,
                "mtf_alignment": mtf_result.alignment,
                "mtf_action": mtf_result.recommended_action,
            },
        )

    def get_summary(self, result: EnhancedSignalResult) -> str:
        """Get human-readable summary."""
        lines = [
            f"Signal: {result.direction}",
            f"Long: {result.long_signal}, Short: {result.short_signal}",
            f"Confidence: {result.confidence:.2f}",
            f"",
        ]

        if result.indicator_result:
            ir = result.indicator_result
            lines.extend([
                f"Indicators:",
                f"  Direction: {ir.direction.value}",
                f"  Confidence: {ir.confidence:.2f}",
                f"  Strength: {ir.strength.value}",
                f"  Contributors: {', '.join(ir.contributing_indicators)}",
                f"",
            ])

        if result.mtf_result:
            mr = result.mtf_result
            lines.extend([
                f"MTF:",
                f"  Direction: {mr.direction.value}",
                f"  Confluence: {mr.confluence_score:.2f}",
                f"  Alignment: {mr.alignment:.2f}",
                f"  Action: {mr.recommended_action}",
                f"",
            ])

        lines.append(f"Reasons: {', '.join(result.reasons)}")
        return "\n".join(lines)


# Singleton instance
_enhanced_signal: Optional[EnhancedSignal] = None


def get_enhanced_signal(**kwargs) -> EnhancedSignal:
    """Get or create enhanced signal instance."""
    global _enhanced_signal
    if _enhanced_signal is None:
        _enhanced_signal = EnhancedSignal(**kwargs)
    return _enhanced_signal


def enhanced_signal_check(
    df_1m: Optional[pd.DataFrame],
    df_5m: Optional[pd.DataFrame] = None,
    df_15m: Optional[pd.DataFrame] = None,
    df_1h: Optional[pd.DataFrame] = None,
    symbol: str = "",
) -> Tuple[bool, bool, float]:
    """
    Quick function to get enhanced signal.

    Returns: (long_signal, short_signal, confidence)
    """
    enhanced = get_enhanced_signal()
    result = enhanced.analyze(
        df_1m=df_1m,
        df_5m=df_5m,
        df_15m=df_15m,
        df_1h=df_1h,
        symbol=symbol,
    )
    return result.long_signal, result.short_signal, result.confidence
