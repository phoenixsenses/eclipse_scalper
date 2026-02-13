# strategies/indicators/__init__.py — SCALPER ETERNAL — INDICATOR MODULES — 2026 v1.0
# Pluggable indicator architecture for modular signal generation.
#
# Design:
# - Each indicator is a standalone module
# - Implements BaseIndicator interface
# - Returns standardized signal dict
# - Can be enabled/disabled via config

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum

import pandas as pd
import numpy as np


class SignalDirection(Enum):
    """Signal direction enum."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class SignalStrength(Enum):
    """Signal strength enum."""
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    NONE = "NONE"


@dataclass
class IndicatorSignal:
    """Standardized indicator signal output."""
    name: str
    direction: SignalDirection = SignalDirection.NEUTRAL
    strength: SignalStrength = SignalStrength.NONE
    value: float = 0.0
    confidence: float = 0.0  # 0-1 normalized
    raw_values: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_bullish(self) -> bool:
        return self.direction == SignalDirection.LONG

    def is_bearish(self) -> bool:
        return self.direction == SignalDirection.SHORT

    def is_neutral(self) -> bool:
        return self.direction == SignalDirection.NEUTRAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction.value,
            "strength": self.strength.value,
            "value": self.value,
            "confidence": self.confidence,
            "raw_values": self.raw_values,
            "metadata": self.metadata,
        }


class BaseIndicator(ABC):
    """
    Base class for all indicators.

    Implement calculate() to return an IndicatorSignal.
    """

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.enabled = True

    @abstractmethod
    def calculate(self, df: pd.DataFrame, **kwargs) -> IndicatorSignal:
        """
        Calculate indicator and return signal.

        Args:
            df: OHLCV DataFrame with columns [o, h, l, c, v]
            **kwargs: Additional parameters (symbol, config, etc.)

        Returns:
            IndicatorSignal with direction, strength, and confidence
        """
        pass

    def validate_df(self, df: pd.DataFrame, min_bars: int = 50) -> bool:
        """Validate DataFrame has required columns and length."""
        if df is None or not isinstance(df, pd.DataFrame):
            return False
        if len(df) < min_bars:
            return False
        required = ["o", "h", "l", "c", "v"]
        for col in required:
            if col not in df.columns:
                return False
        return True

    def safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float."""
        try:
            v = float(value)
            return default if not np.isfinite(v) else v
        except (ValueError, TypeError):
            return default


# Export all indicator classes
from strategies.indicators.macd import MACDIndicator
from strategies.indicators.bollinger import BollingerIndicator
from strategies.indicators.rsi_enhanced import RSIEnhancedIndicator
from strategies.indicators.vwap_enhanced import VWAPEnhancedIndicator
from strategies.indicators.stochastic import StochasticIndicator
from strategies.indicators.adx import ADXIndicator
from strategies.indicators.momentum import MomentumIndicator

__all__ = [
    "BaseIndicator",
    "IndicatorSignal",
    "SignalDirection",
    "SignalStrength",
    "MACDIndicator",
    "BollingerIndicator",
    "RSIEnhancedIndicator",
    "VWAPEnhancedIndicator",
    "StochasticIndicator",
    "ADXIndicator",
    "MomentumIndicator",
]
