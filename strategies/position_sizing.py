# strategies/position_sizing.py — SCALPER ETERNAL — POSITION SIZING — 2026 v1.0
# Dynamic position sizing with Kelly criterion, volatility adjustment, and risk management.
#
# Features:
# - Kelly criterion with half-Kelly safety
# - ATR-based volatility scaling
# - Win rate adaptive sizing
# - Drawdown reduction
# - Max position limits

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np

from utils.logging import log


class SizingMode(Enum):
    """Position sizing mode."""
    FIXED = "FIXED"              # Fixed position size
    FIXED_RISK = "FIXED_RISK"    # Fixed risk per trade (e.g., 1% of equity)
    KELLY = "KELLY"              # Kelly criterion based
    VOLATILITY = "VOLATILITY"    # ATR-based sizing
    ADAPTIVE = "ADAPTIVE"        # Combination of all


@dataclass
class TradeStats:
    """Trading statistics for adaptive sizing."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    current_streak: int = 0  # Positive = wins, negative = losses

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_trades == 0:
            return 0.5  # Assume 50% if no data
        return self.winning_trades / self.total_trades

    @property
    def avg_win(self) -> float:
        """Calculate average win."""
        if self.winning_trades == 0:
            return 0.0
        return self.total_profit / self.winning_trades

    @property
    def avg_loss(self) -> float:
        """Calculate average loss (absolute value)."""
        if self.losing_trades == 0:
            return 0.0
        return abs(self.total_loss) / self.losing_trades

    @property
    def profit_factor(self) -> float:
        """Calculate profit factor."""
        if self.total_loss == 0:
            return float('inf') if self.total_profit > 0 else 0.0
        return abs(self.total_profit / self.total_loss)

    @property
    def expectancy(self) -> float:
        """Calculate trade expectancy."""
        return (self.win_rate * self.avg_win) - ((1 - self.win_rate) * self.avg_loss)


@dataclass
class SizingResult:
    """Position sizing result."""
    position_size: float          # Final position size in base currency
    position_value: float         # Position value in quote currency
    risk_amount: float            # Dollar risk
    risk_pct: float               # Risk as percentage of equity
    leverage_used: float          # Effective leverage
    sizing_mode: SizingMode
    adjustments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PositionSizer:
    """
    Dynamic position sizing with multiple strategies.

    Usage:
        sizer = PositionSizer(
            mode=SizingMode.ADAPTIVE,
            base_risk_pct=0.01,  # 1% risk per trade
            max_position_pct=0.10,  # Max 10% of equity per position
        )

        result = sizer.calculate(
            equity=10000,
            entry_price=50000,
            stop_price=49500,
            confidence=0.75,
            atr=250,
            stats=trade_stats,
        )

        print(f"Size: {result.position_size:.6f} BTC")
        print(f"Risk: ${result.risk_amount:.2f} ({result.risk_pct:.2%})")
    """

    def __init__(
        self,
        mode: SizingMode = SizingMode.ADAPTIVE,
        base_risk_pct: float = 0.01,        # 1% base risk
        max_risk_pct: float = 0.02,         # 2% max risk
        min_risk_pct: float = 0.005,        # 0.5% min risk
        max_position_pct: float = 0.10,     # 10% max position
        min_position_value: float = 10.0,   # $10 minimum
        kelly_fraction: float = 0.5,        # Half Kelly
        volatility_target: float = 0.02,    # 2% daily vol target
        drawdown_threshold: float = 0.10,   # 10% drawdown triggers reduction
        max_leverage: float = 5.0,          # Max leverage allowed
    ):
        self.mode = mode
        self.base_risk_pct = base_risk_pct
        self.max_risk_pct = max_risk_pct
        self.min_risk_pct = min_risk_pct
        self.max_position_pct = max_position_pct
        self.min_position_value = min_position_value
        self.kelly_fraction = kelly_fraction
        self.volatility_target = volatility_target
        self.drawdown_threshold = drawdown_threshold
        self.max_leverage = max_leverage

        # Track equity high water mark for drawdown
        self._equity_hwm: float = 0.0

    def _calculate_kelly(
        self,
        stats: TradeStats,
    ) -> float:
        """
        Calculate Kelly criterion sizing.

        Kelly % = W - [(1-W) / R]
        Where:
            W = Win rate
            R = Win/Loss ratio (avg_win / avg_loss)
        """
        if stats.total_trades < 20:
            # Not enough data, use conservative sizing
            return self.base_risk_pct

        win_rate = stats.win_rate

        if stats.avg_loss == 0:
            return self.base_risk_pct

        win_loss_ratio = stats.avg_win / stats.avg_loss

        # Kelly formula
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Apply fractional Kelly for safety
        kelly = kelly * self.kelly_fraction

        # Clamp to reasonable range
        kelly = max(self.min_risk_pct, min(self.max_risk_pct, kelly))

        return kelly

    def _calculate_volatility_adjustment(
        self,
        atr: float,
        entry_price: float,
    ) -> float:
        """
        Calculate volatility-based position adjustment.

        Higher volatility = smaller position
        Lower volatility = larger position (up to limit)
        """
        if entry_price <= 0 or atr <= 0:
            return 1.0

        # ATR as percentage of price
        atr_pct = atr / entry_price

        # Compare to target volatility
        # If ATR is 2x target, position is 0.5x
        # If ATR is 0.5x target, position is 2x (capped)
        vol_adjustment = self.volatility_target / atr_pct

        # Clamp adjustment
        vol_adjustment = max(0.5, min(2.0, vol_adjustment))

        return vol_adjustment

    def _calculate_drawdown_adjustment(
        self,
        equity: float,
    ) -> float:
        """
        Calculate drawdown-based position reduction.

        Reduces position size when in drawdown to preserve capital.
        """
        # Update high water mark
        if equity > self._equity_hwm:
            self._equity_hwm = equity

        if self._equity_hwm <= 0:
            return 1.0

        # Current drawdown
        drawdown = (self._equity_hwm - equity) / self._equity_hwm

        if drawdown <= 0:
            return 1.0

        if drawdown < self.drawdown_threshold:
            # Linear reduction up to threshold
            return 1.0 - (drawdown / self.drawdown_threshold) * 0.3
        else:
            # More aggressive reduction beyond threshold
            excess = drawdown - self.drawdown_threshold
            return 0.7 - min(0.4, excess * 2)  # Min 30% of normal size

    def _calculate_streak_adjustment(
        self,
        stats: TradeStats,
    ) -> float:
        """
        Adjust sizing based on win/loss streaks.

        Reduce size during losing streaks.
        Slight increase during winning streaks (capped).
        """
        streak = stats.current_streak

        if streak >= 3:
            # Winning streak - slight increase (max 1.2x)
            return 1.0 + min(0.2, streak * 0.05)
        elif streak <= -3:
            # Losing streak - reduce size
            return max(0.5, 1.0 + streak * 0.1)
        else:
            return 1.0

    def _calculate_confidence_adjustment(
        self,
        confidence: float,
    ) -> float:
        """
        Adjust sizing based on signal confidence.

        Higher confidence = larger position (up to 1.5x)
        Lower confidence = smaller position (down to 0.5x)
        """
        # Confidence 0.5 = 1.0x (baseline)
        # Confidence 1.0 = 1.5x
        # Confidence 0.0 = 0.5x
        return 0.5 + confidence

    def calculate(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        confidence: float = 0.5,
        atr: float = 0.0,
        stats: Optional[TradeStats] = None,
        current_exposure: float = 0.0,  # Current total exposure
    ) -> SizingResult:
        """
        Calculate position size with all adjustments.

        Args:
            equity: Current account equity
            entry_price: Planned entry price
            stop_price: Stop loss price
            confidence: Signal confidence (0-1)
            atr: Average True Range
            stats: Trading statistics
            current_exposure: Current position exposure

        Returns:
            SizingResult with position size and metadata
        """
        adjustments = []

        # Validate inputs
        if equity <= 0 or entry_price <= 0:
            return SizingResult(
                position_size=0.0,
                position_value=0.0,
                risk_amount=0.0,
                risk_pct=0.0,
                leverage_used=0.0,
                sizing_mode=self.mode,
                adjustments=["invalid_inputs"],
            )

        # Calculate stop distance
        stop_distance = abs(entry_price - stop_price)
        stop_pct = stop_distance / entry_price if entry_price > 0 else 0.01

        # Minimum stop distance
        if stop_pct < 0.001:  # Less than 0.1%
            stop_pct = 0.005  # Default to 0.5%
            adjustments.append("min_stop_applied")

        # Base risk percentage
        if self.mode == SizingMode.FIXED:
            risk_pct = self.base_risk_pct

        elif self.mode == SizingMode.FIXED_RISK:
            risk_pct = self.base_risk_pct

        elif self.mode == SizingMode.KELLY:
            if stats:
                risk_pct = self._calculate_kelly(stats)
                adjustments.append(f"kelly_{risk_pct:.3f}")
            else:
                risk_pct = self.base_risk_pct
                adjustments.append("kelly_no_stats")

        elif self.mode == SizingMode.VOLATILITY:
            risk_pct = self.base_risk_pct
            if atr > 0:
                vol_adj = self._calculate_volatility_adjustment(atr, entry_price)
                risk_pct *= vol_adj
                adjustments.append(f"vol_adj_{vol_adj:.2f}")

        else:  # ADAPTIVE
            # Start with Kelly if we have stats
            if stats and stats.total_trades >= 20:
                risk_pct = self._calculate_kelly(stats)
                adjustments.append(f"kelly_{risk_pct:.3f}")
            else:
                risk_pct = self.base_risk_pct

            # Apply volatility adjustment
            if atr > 0:
                vol_adj = self._calculate_volatility_adjustment(atr, entry_price)
                risk_pct *= vol_adj
                adjustments.append(f"vol_{vol_adj:.2f}")

            # Apply drawdown adjustment
            dd_adj = self._calculate_drawdown_adjustment(equity)
            if dd_adj < 1.0:
                risk_pct *= dd_adj
                adjustments.append(f"dd_{dd_adj:.2f}")

            # Apply streak adjustment
            if stats:
                streak_adj = self._calculate_streak_adjustment(stats)
                if streak_adj != 1.0:
                    risk_pct *= streak_adj
                    adjustments.append(f"streak_{streak_adj:.2f}")

            # Apply confidence adjustment
            conf_adj = self._calculate_confidence_adjustment(confidence)
            risk_pct *= conf_adj
            adjustments.append(f"conf_{conf_adj:.2f}")

        # Clamp risk percentage
        risk_pct = max(self.min_risk_pct, min(self.max_risk_pct, risk_pct))

        # Calculate dollar risk
        risk_amount = equity * risk_pct

        # Calculate position size based on stop distance
        # Position Size = Risk Amount / Stop Distance
        position_value = risk_amount / stop_pct
        position_size = position_value / entry_price

        # Apply maximum position limit
        max_position_value = equity * self.max_position_pct
        if position_value > max_position_value:
            position_value = max_position_value
            position_size = position_value / entry_price
            # Recalculate actual risk with capped position
            risk_amount = position_value * stop_pct
            risk_pct = risk_amount / equity
            adjustments.append("max_pos_cap")

        # Check leverage limit
        effective_leverage = position_value / equity
        if effective_leverage > self.max_leverage:
            position_value = equity * self.max_leverage
            position_size = position_value / entry_price
            risk_amount = position_value * stop_pct
            risk_pct = risk_amount / equity
            adjustments.append(f"lev_cap_{self.max_leverage}")

        # Check available margin (considering current exposure)
        available_margin = equity - current_exposure
        if position_value > available_margin * self.max_leverage:
            position_value = available_margin * self.max_leverage
            position_size = position_value / entry_price
            risk_amount = position_value * stop_pct
            risk_pct = risk_amount / equity
            adjustments.append("margin_limited")

        # Apply minimum position value
        if position_value < self.min_position_value:
            return SizingResult(
                position_size=0.0,
                position_value=0.0,
                risk_amount=0.0,
                risk_pct=0.0,
                leverage_used=0.0,
                sizing_mode=self.mode,
                adjustments=["below_minimum"],
            )

        return SizingResult(
            position_size=position_size,
            position_value=position_value,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            leverage_used=position_value / equity,
            sizing_mode=self.mode,
            adjustments=adjustments,
            metadata={
                "stop_pct": stop_pct,
                "confidence": confidence,
                "equity": equity,
            },
        )

    def update_stats(
        self,
        stats: TradeStats,
        pnl: float,
    ) -> TradeStats:
        """Update trade statistics with a completed trade."""
        stats.total_trades += 1

        if pnl > 0:
            stats.winning_trades += 1
            stats.total_profit += pnl
            stats.largest_win = max(stats.largest_win, pnl)

            if stats.current_streak >= 0:
                stats.current_streak += 1
            else:
                stats.current_streak = 1

            stats.consecutive_wins = max(stats.consecutive_wins, stats.current_streak)

        else:
            stats.losing_trades += 1
            stats.total_loss += pnl  # pnl is negative
            stats.largest_loss = min(stats.largest_loss, pnl)

            if stats.current_streak <= 0:
                stats.current_streak -= 1
            else:
                stats.current_streak = -1

            stats.consecutive_losses = max(
                stats.consecutive_losses,
                abs(stats.current_streak)
            )

        return stats

    def get_sizing_summary(self, result: SizingResult) -> str:
        """Get human-readable sizing summary."""
        lines = [
            f"Position Size: {result.position_size:.6f}",
            f"Position Value: ${result.position_value:.2f}",
            f"Risk Amount: ${result.risk_amount:.2f}",
            f"Risk %: {result.risk_pct:.2%}",
            f"Leverage: {result.leverage_used:.2f}x",
            f"Mode: {result.sizing_mode.value}",
            f"Adjustments: {', '.join(result.adjustments) or 'none'}",
        ]
        return "\n".join(lines)


# Singleton instance
_sizer_instance: Optional[PositionSizer] = None


def get_position_sizer(
    mode: SizingMode = SizingMode.ADAPTIVE,
    **kwargs,
) -> PositionSizer:
    """Get or create position sizer instance."""
    global _sizer_instance
    if _sizer_instance is None:
        _sizer_instance = PositionSizer(mode=mode, **kwargs)
    return _sizer_instance


# Convenience function for quick sizing
def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    confidence: float = 0.5,
    atr: float = 0.0,
    risk_pct: float = 0.01,
) -> Tuple[float, float]:
    """
    Quick position size calculation.

    Returns: (position_size, risk_amount)
    """
    sizer = PositionSizer(
        mode=SizingMode.FIXED_RISK,
        base_risk_pct=risk_pct,
    )
    result = sizer.calculate(
        equity=equity,
        entry_price=entry_price,
        stop_price=stop_price,
        confidence=confidence,
        atr=atr,
    )
    return result.position_size, result.risk_amount
