# strategies/exit_strategy.py — SCALPER ETERNAL — EXIT STRATEGY — 2026 v1.0
# Advanced exit management with partial TPs, adaptive trailing, and time-based exits.
#
# Features:
# - Multiple partial take-profit levels
# - ATR-based adaptive trailing stops
# - Breakeven management
# - Time-based exits
# - Momentum-based exit acceleration

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

from utils.logging import log


class ExitType(Enum):
    """Exit type classification."""
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    PARTIAL_TP = "PARTIAL_TP"
    TRAILING_STOP = "TRAILING_STOP"
    BREAKEVEN = "BREAKEVEN"
    TIME_EXIT = "TIME_EXIT"
    SIGNAL_EXIT = "SIGNAL_EXIT"
    MANUAL = "MANUAL"
    EMERGENCY = "EMERGENCY"


class TrailMode(Enum):
    """Trailing stop mode."""
    FIXED = "FIXED"          # Fixed percentage trail
    ATR = "ATR"              # ATR-based trail
    CHANDELIER = "CHANDELIER"  # Chandelier exit
    PARABOLIC = "PARABOLIC"  # Tightening trail
    ADAPTIVE = "ADAPTIVE"    # Momentum-adaptive


@dataclass
class PartialTPLevel:
    """Single partial take-profit level."""
    target_pct: float        # Price target as % gain
    close_pct: float         # % of position to close
    triggered: bool = False
    trigger_price: float = 0.0
    trigger_time: float = 0.0


@dataclass
class ExitPlan:
    """Complete exit plan for a position."""
    symbol: str
    entry_price: float
    direction: str  # "LONG" or "SHORT"

    # Stop loss
    initial_stop: float
    current_stop: float
    stop_type: str = "fixed"

    # Take profits
    partial_tps: List[PartialTPLevel] = field(default_factory=list)
    final_tp: float = 0.0

    # Trailing
    trail_mode: TrailMode = TrailMode.ATR
    trail_activation: float = 0.0  # Price where trailing activates
    trail_distance: float = 0.0    # Current trail distance
    trailing_active: bool = False
    highest_price: float = 0.0     # For LONG
    lowest_price: float = 0.0      # For SHORT

    # Breakeven
    breakeven_trigger: float = 0.0  # Price to trigger BE
    breakeven_active: bool = False
    breakeven_buffer: float = 0.0   # Buffer above entry

    # Time
    entry_time: float = 0.0
    max_hold_seconds: int = 0      # 0 = no limit

    # State
    remaining_pct: float = 1.0     # Remaining position %
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExitSignal:
    """Exit signal from strategy."""
    should_exit: bool
    exit_type: ExitType
    exit_pct: float          # % of position to exit
    exit_price: float        # Suggested exit price
    reason: str
    urgency: float = 0.5     # 0-1, higher = more urgent
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExitStrategy:
    """
    Advanced exit strategy manager.

    Usage:
        exit_mgr = ExitStrategy()

        # Create exit plan
        plan = exit_mgr.create_plan(
            symbol="BTCUSDT",
            entry_price=50000,
            direction="LONG",
            stop_loss=49500,
            atr=250,
        )

        # Check exits on each tick
        signal = exit_mgr.check_exit(
            plan=plan,
            current_price=50500,
            df=df,  # For momentum analysis
        )

        if signal.should_exit:
            print(f"Exit {signal.exit_pct:.0%} at {signal.exit_price}")
    """

    # Default partial TP levels for scalping
    DEFAULT_PARTIAL_TPS = [
        {"target_pct": 0.003, "close_pct": 0.25},  # 0.3% gain -> close 25%
        {"target_pct": 0.006, "close_pct": 0.25},  # 0.6% gain -> close 25%
        {"target_pct": 0.010, "close_pct": 0.25},  # 1.0% gain -> close 25%
        # Remaining 25% trails
    ]

    def __init__(
        self,
        partial_tps: Optional[List[dict]] = None,
        trail_mode: TrailMode = TrailMode.ATR,
        trail_atr_mult: float = 2.0,        # ATR multiplier for trail
        trail_activation_pct: float = 0.005,  # 0.5% profit to activate trail
        breakeven_trigger_pct: float = 0.004,  # 0.4% profit to move to BE
        breakeven_buffer_pct: float = 0.001,   # 0.1% buffer above entry
        max_hold_minutes: int = 60,            # Max hold time
        use_momentum_exit: bool = True,
        momentum_exit_threshold: float = -0.3,  # Exit if momentum reverses
    ):
        self.partial_tps = partial_tps or self.DEFAULT_PARTIAL_TPS
        self.trail_mode = trail_mode
        self.trail_atr_mult = trail_atr_mult
        self.trail_activation_pct = trail_activation_pct
        self.breakeven_trigger_pct = breakeven_trigger_pct
        self.breakeven_buffer_pct = breakeven_buffer_pct
        self.max_hold_seconds = max_hold_minutes * 60
        self.use_momentum_exit = use_momentum_exit
        self.momentum_exit_threshold = momentum_exit_threshold

    def create_plan(
        self,
        symbol: str,
        entry_price: float,
        direction: str,
        stop_loss: float,
        atr: float = 0.0,
        custom_tps: Optional[List[dict]] = None,
    ) -> ExitPlan:
        """Create exit plan for a new position."""
        # Create partial TP levels
        tp_config = custom_tps or self.partial_tps
        partial_tps = []

        for tp in tp_config:
            target_pct = tp["target_pct"]
            close_pct = tp["close_pct"]

            if direction == "LONG":
                trigger_price = entry_price * (1 + target_pct)
            else:
                trigger_price = entry_price * (1 - target_pct)

            partial_tps.append(PartialTPLevel(
                target_pct=target_pct,
                close_pct=close_pct,
                triggered=False,
                trigger_price=trigger_price,
            ))

        # Calculate final TP (beyond partials)
        max_partial = max(tp["target_pct"] for tp in tp_config) if tp_config else 0.01
        final_target = max_partial * 1.5

        if direction == "LONG":
            final_tp = entry_price * (1 + final_target)
        else:
            final_tp = entry_price * (1 - final_target)

        # Trail activation price
        if direction == "LONG":
            trail_activation = entry_price * (1 + self.trail_activation_pct)
        else:
            trail_activation = entry_price * (1 - self.trail_activation_pct)

        # Initial trail distance (ATR-based)
        trail_distance = atr * self.trail_atr_mult if atr > 0 else entry_price * 0.005

        # Breakeven trigger
        if direction == "LONG":
            be_trigger = entry_price * (1 + self.breakeven_trigger_pct)
        else:
            be_trigger = entry_price * (1 - self.breakeven_trigger_pct)

        # Breakeven with buffer
        if direction == "LONG":
            be_buffer = entry_price * self.breakeven_buffer_pct
        else:
            be_buffer = -entry_price * self.breakeven_buffer_pct

        return ExitPlan(
            symbol=symbol,
            entry_price=entry_price,
            direction=direction,
            initial_stop=stop_loss,
            current_stop=stop_loss,
            partial_tps=partial_tps,
            final_tp=final_tp,
            trail_mode=self.trail_mode,
            trail_activation=trail_activation,
            trail_distance=trail_distance,
            trailing_active=False,
            highest_price=entry_price,
            lowest_price=entry_price,
            breakeven_trigger=be_trigger,
            breakeven_active=False,
            breakeven_buffer=be_buffer,
            entry_time=time.time(),
            max_hold_seconds=self.max_hold_seconds,
            remaining_pct=1.0,
            metadata={"atr": atr},
        )

    def _update_trailing_stop(
        self,
        plan: ExitPlan,
        current_price: float,
        atr: float = 0.0,
        momentum: float = 0.0,
    ) -> float:
        """Update trailing stop based on mode."""
        if plan.direction == "LONG":
            # Update highest price
            if current_price > plan.highest_price:
                plan.highest_price = current_price

            if self.trail_mode == TrailMode.FIXED:
                new_stop = plan.highest_price * (1 - plan.trail_distance / plan.entry_price)

            elif self.trail_mode == TrailMode.ATR:
                trail_dist = atr * self.trail_atr_mult if atr > 0 else plan.trail_distance
                new_stop = plan.highest_price - trail_dist

            elif self.trail_mode == TrailMode.CHANDELIER:
                # Chandelier: 3 ATR from high
                trail_dist = atr * 3 if atr > 0 else plan.trail_distance * 1.5
                new_stop = plan.highest_price - trail_dist

            elif self.trail_mode == TrailMode.PARABOLIC:
                # Tighten trail as profit increases
                profit_pct = (plan.highest_price - plan.entry_price) / plan.entry_price
                tightness = min(0.5, profit_pct * 10)  # Max 50% tighter
                trail_dist = plan.trail_distance * (1 - tightness)
                new_stop = plan.highest_price - trail_dist

            elif self.trail_mode == TrailMode.ADAPTIVE:
                # Tighten on negative momentum
                if momentum < -0.2:
                    mult = 0.5  # Tight trail
                elif momentum > 0.2:
                    mult = 1.5  # Loose trail
                else:
                    mult = 1.0
                trail_dist = atr * self.trail_atr_mult * mult if atr > 0 else plan.trail_distance
                new_stop = plan.highest_price - trail_dist

            else:
                new_stop = plan.current_stop

            # Only move stop up, never down
            return max(plan.current_stop, new_stop)

        else:  # SHORT
            # Update lowest price
            if current_price < plan.lowest_price:
                plan.lowest_price = current_price

            if self.trail_mode == TrailMode.FIXED:
                new_stop = plan.lowest_price * (1 + plan.trail_distance / plan.entry_price)

            elif self.trail_mode == TrailMode.ATR:
                trail_dist = atr * self.trail_atr_mult if atr > 0 else plan.trail_distance
                new_stop = plan.lowest_price + trail_dist

            elif self.trail_mode == TrailMode.CHANDELIER:
                trail_dist = atr * 3 if atr > 0 else plan.trail_distance * 1.5
                new_stop = plan.lowest_price + trail_dist

            elif self.trail_mode == TrailMode.PARABOLIC:
                profit_pct = (plan.entry_price - plan.lowest_price) / plan.entry_price
                tightness = min(0.5, profit_pct * 10)
                trail_dist = plan.trail_distance * (1 - tightness)
                new_stop = plan.lowest_price + trail_dist

            elif self.trail_mode == TrailMode.ADAPTIVE:
                if momentum > 0.2:  # Momentum reversing against short
                    mult = 0.5
                elif momentum < -0.2:
                    mult = 1.5
                else:
                    mult = 1.0
                trail_dist = atr * self.trail_atr_mult * mult if atr > 0 else plan.trail_distance
                new_stop = plan.lowest_price + trail_dist

            else:
                new_stop = plan.current_stop

            # Only move stop down, never up
            return min(plan.current_stop, new_stop)

    def _calculate_momentum(self, df: pd.DataFrame, period: int = 5) -> float:
        """Calculate recent momentum (-1 to 1)."""
        try:
            if df is None or len(df) < period + 1:
                return 0.0

            close = df["c"]
            pct_change = close.pct_change(period).iloc[-1]

            # Normalize to -1 to 1
            return max(-1.0, min(1.0, pct_change / 0.02))
        except Exception:
            return 0.0

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate current ATR."""
        try:
            if df is None or len(df) < period + 1:
                return 0.0

            high = df["h"]
            low = df["l"]
            close = df["c"]

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            return float(tr.rolling(period).mean().iloc[-1])
        except Exception:
            return 0.0

    def check_exit(
        self,
        plan: ExitPlan,
        current_price: float,
        df: Optional[pd.DataFrame] = None,
    ) -> ExitSignal:
        """
        Check if any exit conditions are met.

        Args:
            plan: Current exit plan
            current_price: Current market price
            df: OHLCV DataFrame for momentum analysis

        Returns:
            ExitSignal with exit decision
        """
        # Calculate momentum and ATR if we have data
        momentum = self._calculate_momentum(df) if df is not None else 0.0
        atr = self._calculate_atr(df) if df is not None else plan.metadata.get("atr", 0.0)

        # Check stop loss
        if plan.direction == "LONG":
            if current_price <= plan.current_stop:
                exit_type = ExitType.TRAILING_STOP if plan.trailing_active else ExitType.STOP_LOSS
                return ExitSignal(
                    should_exit=True,
                    exit_type=exit_type,
                    exit_pct=plan.remaining_pct,
                    exit_price=plan.current_stop,
                    reason=f"stop_hit_{plan.current_stop:.2f}",
                    urgency=1.0,
                )
        else:  # SHORT
            if current_price >= plan.current_stop:
                exit_type = ExitType.TRAILING_STOP if plan.trailing_active else ExitType.STOP_LOSS
                return ExitSignal(
                    should_exit=True,
                    exit_type=exit_type,
                    exit_pct=plan.remaining_pct,
                    exit_price=plan.current_stop,
                    reason=f"stop_hit_{plan.current_stop:.2f}",
                    urgency=1.0,
                )

        # Check partial take-profits
        for tp in plan.partial_tps:
            if tp.triggered:
                continue

            if plan.direction == "LONG" and current_price >= tp.trigger_price:
                tp.triggered = True
                tp.trigger_time = time.time()
                plan.remaining_pct -= tp.close_pct
                return ExitSignal(
                    should_exit=True,
                    exit_type=ExitType.PARTIAL_TP,
                    exit_pct=tp.close_pct,
                    exit_price=current_price,
                    reason=f"partial_tp_{tp.target_pct:.3f}",
                    urgency=0.7,
                    metadata={"level": tp.target_pct},
                )

            elif plan.direction == "SHORT" and current_price <= tp.trigger_price:
                tp.triggered = True
                tp.trigger_time = time.time()
                plan.remaining_pct -= tp.close_pct
                return ExitSignal(
                    should_exit=True,
                    exit_type=ExitType.PARTIAL_TP,
                    exit_pct=tp.close_pct,
                    exit_price=current_price,
                    reason=f"partial_tp_{tp.target_pct:.3f}",
                    urgency=0.7,
                    metadata={"level": tp.target_pct},
                )

        # Check final TP
        if plan.direction == "LONG" and current_price >= plan.final_tp:
            return ExitSignal(
                should_exit=True,
                exit_type=ExitType.TAKE_PROFIT,
                exit_pct=plan.remaining_pct,
                exit_price=current_price,
                reason="final_tp_reached",
                urgency=0.8,
            )
        elif plan.direction == "SHORT" and current_price <= plan.final_tp:
            return ExitSignal(
                should_exit=True,
                exit_type=ExitType.TAKE_PROFIT,
                exit_pct=plan.remaining_pct,
                exit_price=current_price,
                reason="final_tp_reached",
                urgency=0.8,
            )

        # Check breakeven activation
        if not plan.breakeven_active:
            if plan.direction == "LONG" and current_price >= plan.breakeven_trigger:
                plan.breakeven_active = True
                plan.current_stop = max(
                    plan.current_stop,
                    plan.entry_price + plan.breakeven_buffer
                )
            elif plan.direction == "SHORT" and current_price <= plan.breakeven_trigger:
                plan.breakeven_active = True
                plan.current_stop = min(
                    plan.current_stop,
                    plan.entry_price + plan.breakeven_buffer  # buffer is negative for short
                )

        # Check trail activation
        if not plan.trailing_active:
            if plan.direction == "LONG" and current_price >= plan.trail_activation:
                plan.trailing_active = True
            elif plan.direction == "SHORT" and current_price <= plan.trail_activation:
                plan.trailing_active = True

        # Update trailing stop if active
        if plan.trailing_active:
            plan.current_stop = self._update_trailing_stop(
                plan=plan,
                current_price=current_price,
                atr=atr,
                momentum=momentum,
            )

        # Check time-based exit
        if plan.max_hold_seconds > 0:
            hold_time = time.time() - plan.entry_time
            if hold_time >= plan.max_hold_seconds:
                return ExitSignal(
                    should_exit=True,
                    exit_type=ExitType.TIME_EXIT,
                    exit_pct=plan.remaining_pct,
                    exit_price=current_price,
                    reason=f"max_hold_{hold_time/60:.0f}min",
                    urgency=0.6,
                )

        # Check momentum-based exit
        if self.use_momentum_exit:
            if plan.direction == "LONG" and momentum < self.momentum_exit_threshold:
                # Strong negative momentum, consider exiting
                return ExitSignal(
                    should_exit=True,
                    exit_type=ExitType.SIGNAL_EXIT,
                    exit_pct=plan.remaining_pct * 0.5,  # Exit half
                    exit_price=current_price,
                    reason=f"momentum_reversal_{momentum:.2f}",
                    urgency=0.5,
                )
            elif plan.direction == "SHORT" and momentum > -self.momentum_exit_threshold:
                return ExitSignal(
                    should_exit=True,
                    exit_type=ExitType.SIGNAL_EXIT,
                    exit_pct=plan.remaining_pct * 0.5,
                    exit_price=current_price,
                    reason=f"momentum_reversal_{momentum:.2f}",
                    urgency=0.5,
                )

        # No exit
        return ExitSignal(
            should_exit=False,
            exit_type=ExitType.MANUAL,
            exit_pct=0.0,
            exit_price=current_price,
            reason="no_exit_condition",
            urgency=0.0,
        )

    def get_plan_summary(self, plan: ExitPlan) -> str:
        """Get human-readable plan summary."""
        triggered = sum(1 for tp in plan.partial_tps if tp.triggered)
        lines = [
            f"Symbol: {plan.symbol} ({plan.direction})",
            f"Entry: {plan.entry_price:.2f}",
            f"Current Stop: {plan.current_stop:.2f}",
            f"Trailing: {'Active' if plan.trailing_active else 'Inactive'}",
            f"Breakeven: {'Active' if plan.breakeven_active else 'Inactive'}",
            f"Partial TPs: {triggered}/{len(plan.partial_tps)}",
            f"Remaining: {plan.remaining_pct:.0%}",
            f"Hold Time: {(time.time() - plan.entry_time)/60:.1f} min",
        ]
        return "\n".join(lines)


# Singleton instance
_exit_strategy: Optional[ExitStrategy] = None


def get_exit_strategy(**kwargs) -> ExitStrategy:
    """Get or create exit strategy instance."""
    global _exit_strategy
    if _exit_strategy is None:
        _exit_strategy = ExitStrategy(**kwargs)
    return _exit_strategy
