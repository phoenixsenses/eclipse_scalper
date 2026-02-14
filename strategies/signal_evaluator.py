# strategies/signal_evaluator.py — SCALPER ETERNAL — SIGNAL EVALUATION HARNESS — 2026 v1.0
# Measure directional accuracy of any signal function WITHOUT SL/TP.
#
# Purpose:
# - Answer "does this signal predict direction better than coin flip?"
# - Measure accuracy at multiple forward horizons (5, 15, 30, 60 bars)
# - Break down accuracy by confidence bucket
# - Compare signal generators head-to-head
#
# Usage:
#     python -m strategies.signal_evaluator --symbol BTCUSDT --days 7

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.backtester import DataLoader


SignalFunc = Callable[[pd.DataFrame, pd.DataFrame, int], Tuple[bool, bool, float]]


@dataclass
class DirectionOutcome:
    """Outcome for a single signal at one horizon."""
    bar_idx: int
    direction: str  # "long" or "short"
    confidence: float
    entry_price: float
    future_price: float
    move_pct: float  # signed: positive = direction was correct
    correct: bool


@dataclass
class HorizonStats:
    """Accuracy stats for one forward horizon."""
    horizon: int
    total_signals: int
    correct: int
    accuracy: float
    avg_move_pct: float  # average signed move (positive = profitable direction)
    avg_abs_move_pct: float  # average absolute move
    long_accuracy: float
    short_accuracy: float
    long_count: int
    short_count: int


@dataclass
class ConfidenceBucket:
    """Stats for a confidence range."""
    lo: float
    hi: float
    count: int
    accuracy: float
    avg_move_pct: float


@dataclass
class EvaluationResult:
    """Full evaluation result for one signal function."""
    name: str
    total_bars: int
    total_signals: int
    signal_rate: float  # signals per bar
    horizon_stats: Dict[int, HorizonStats]
    confidence_buckets: List[ConfidenceBucket]
    outcomes: Dict[int, List[DirectionOutcome]]  # keyed by horizon


class SignalEvaluator:
    """
    Evaluate signal directional accuracy without SL/TP.

    For each signal generated, look forward N bars and check whether
    price moved in the predicted direction.
    """

    def __init__(
        self,
        horizons: List[int] = None,
        lookback: int = 100,
        confidence_bins: List[float] = None,
        min_gap_bars: int = 0,
    ):
        self.horizons = horizons or [5, 15, 30, 60]
        self.lookback = lookback
        self.confidence_bins = confidence_bins or [0.0, 0.3, 0.5, 0.7, 0.9, 1.01]
        self.min_gap_bars = min_gap_bars

    def evaluate(
        self,
        name: str,
        df_1m: pd.DataFrame,
        df_5m: pd.DataFrame,
        signal_func: SignalFunc,
        min_confidence: float = 0.0,
    ) -> EvaluationResult:
        """
        Evaluate a signal function on historical data.

        Args:
            name: Name for this signal
            df_1m: 1-minute OHLCV data
            df_5m: 5-minute OHLCV data
            signal_func: Function(df_1m_window, df_5m_window, bar_idx) -> (is_long, is_short, confidence)
            min_confidence: Minimum confidence to count as signal

        Returns:
            EvaluationResult with accuracy metrics
        """
        max_horizon = max(self.horizons)
        end_idx = len(df_1m) - max_horizon  # need room for forward look

        outcomes: Dict[int, List[DirectionOutcome]] = {h: [] for h in self.horizons}
        last_signal_bar = -self.min_gap_bars - 1

        for i in range(self.lookback, end_idx):
            # Enforce minimum gap
            if self.min_gap_bars > 0 and (i - last_signal_bar) < self.min_gap_bars:
                continue

            df_1m_window = df_1m.iloc[max(0, i - 300):i + 1]
            df_5m_window = df_5m.loc[:df_1m.index[i]] if len(df_5m) > 0 else df_5m

            try:
                is_long, is_short, confidence = signal_func(df_1m_window, df_5m_window, i)
            except Exception:
                continue

            if not (is_long or is_short) or confidence < min_confidence:
                continue

            last_signal_bar = i
            direction = "long" if is_long else "short"
            entry_price = float(df_1m.iloc[i]["c"])

            # Evaluate at each horizon
            for h in self.horizons:
                future_idx = i + h
                if future_idx >= len(df_1m):
                    continue

                future_price = float(df_1m.iloc[future_idx]["c"])
                move_pct = (future_price - entry_price) / entry_price * 100

                # Flip sign for shorts (positive = correct direction)
                signed_move = move_pct if direction == "long" else -move_pct
                correct = signed_move > 0

                outcomes[h].append(DirectionOutcome(
                    bar_idx=i,
                    direction=direction,
                    confidence=confidence,
                    entry_price=entry_price,
                    future_price=future_price,
                    move_pct=signed_move,
                    correct=correct,
                ))

        # Compute horizon stats
        horizon_stats = {}
        for h in self.horizons:
            ocs = outcomes[h]
            if not ocs:
                horizon_stats[h] = HorizonStats(
                    horizon=h, total_signals=0, correct=0, accuracy=0.0,
                    avg_move_pct=0.0, avg_abs_move_pct=0.0,
                    long_accuracy=0.0, short_accuracy=0.0,
                    long_count=0, short_count=0,
                )
                continue

            correct_count = sum(1 for o in ocs if o.correct)
            longs = [o for o in ocs if o.direction == "long"]
            shorts = [o for o in ocs if o.direction == "short"]

            horizon_stats[h] = HorizonStats(
                horizon=h,
                total_signals=len(ocs),
                correct=correct_count,
                accuracy=correct_count / len(ocs),
                avg_move_pct=float(np.mean([o.move_pct for o in ocs])),
                avg_abs_move_pct=float(np.mean([abs(o.move_pct) for o in ocs])),
                long_accuracy=sum(1 for o in longs if o.correct) / len(longs) if longs else 0.0,
                short_accuracy=sum(1 for o in shorts if o.correct) / len(shorts) if shorts else 0.0,
                long_count=len(longs),
                short_count=len(shorts),
            )

        # Compute confidence buckets (using largest horizon for most data)
        ref_horizon = max(self.horizons)
        ref_outcomes = outcomes[ref_horizon]
        confidence_buckets = []

        for j in range(len(self.confidence_bins) - 1):
            lo = self.confidence_bins[j]
            hi = self.confidence_bins[j + 1]
            bucket = [o for o in ref_outcomes if lo <= o.confidence < hi]
            if bucket:
                confidence_buckets.append(ConfidenceBucket(
                    lo=lo, hi=hi,
                    count=len(bucket),
                    accuracy=sum(1 for o in bucket if o.correct) / len(bucket),
                    avg_move_pct=float(np.mean([o.move_pct for o in bucket])),
                ))

        # Signal rate
        scan_range = end_idx - self.lookback
        total_signals = len(outcomes[self.horizons[0]]) if self.horizons else 0

        return EvaluationResult(
            name=name,
            total_bars=scan_range,
            total_signals=total_signals,
            signal_rate=total_signals / scan_range if scan_range > 0 else 0.0,
            horizon_stats=horizon_stats,
            confidence_buckets=confidence_buckets,
            outcomes=outcomes,
        )


def print_evaluation(result: EvaluationResult):
    """Print formatted evaluation results."""
    print(f"\n{'=' * 70}")
    print(f"SIGNAL EVALUATION: {result.name}")
    print(f"{'=' * 70}")
    print(f"Total bars scanned: {result.total_bars:,}")
    print(f"Total signals: {result.total_signals:,}")
    print(f"Signal rate: {result.signal_rate:.4f} signals/bar ({result.signal_rate * 100:.1f}%)")

    print(f"\n{'DIRECTIONAL ACCURACY BY HORIZON':}")
    print(f"{'Horizon':>10} {'Signals':>10} {'Accuracy':>10} {'AvgMove':>10} {'LongAcc':>10} {'ShortAcc':>10} {'L/S':>8}")
    print("-" * 70)

    for h in sorted(result.horizon_stats.keys()):
        hs = result.horizon_stats[h]
        coin_flip = "COIN FLIP" if abs(hs.accuracy - 0.5) < 0.02 else ""
        edge = "EDGE" if hs.accuracy > 0.52 else ""
        anti = "ANTI-EDGE" if hs.accuracy < 0.48 else ""
        label = coin_flip or edge or anti
        print(
            f"{h:>8}m "
            f"{hs.total_signals:>10} "
            f"{hs.accuracy * 100:>9.1f}% "
            f"{hs.avg_move_pct:>+9.3f}% "
            f"{hs.long_accuracy * 100:>9.1f}% "
            f"{hs.short_accuracy * 100:>9.1f}% "
            f"{hs.long_count}/{hs.short_count}"
            f"  {label}"
        )

    if result.confidence_buckets:
        print(f"\nACCURACY BY CONFIDENCE BUCKET (horizon={max(result.horizon_stats.keys())}m):")
        print(f"{'Range':>12} {'Count':>8} {'Accuracy':>10} {'AvgMove':>10}")
        print("-" * 42)
        for b in result.confidence_buckets:
            print(
                f"[{b.lo:.1f}-{b.hi:.1f}) "
                f"{b.count:>8} "
                f"{b.accuracy * 100:>9.1f}% "
                f"{b.avg_move_pct:>+9.3f}%"
            )

    print(f"{'=' * 70}")


def compare_signals(results: List[EvaluationResult], horizon: int = 30):
    """Compare multiple signal functions side by side."""
    print(f"\n{'=' * 80}")
    print(f"SIGNAL COMPARISON (horizon={horizon}m)")
    print(f"{'=' * 80}")
    print(f"{'Signal':<25} {'Signals':>8} {'Accuracy':>10} {'AvgMove':>10} {'Rate':>10} {'Verdict':>12}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: -(x.horizon_stats.get(horizon, HorizonStats(0,0,0,0,0,0,0,0,0,0)).accuracy)):
        hs = r.horizon_stats.get(horizon)
        if hs is None or hs.total_signals == 0:
            print(f"{r.name:<25} {'NO SIGNALS':>8}")
            continue

        if hs.accuracy > 0.55:
            verdict = "PROMISING"
        elif hs.accuracy > 0.52:
            verdict = "SLIGHT EDGE"
        elif hs.accuracy > 0.48:
            verdict = "COIN FLIP"
        elif hs.accuracy > 0.45:
            verdict = "SLIGHT ANTI"
        else:
            verdict = "ANTI-EDGE"

        print(
            f"{r.name:<25} "
            f"{hs.total_signals:>8} "
            f"{hs.accuracy * 100:>9.1f}% "
            f"{hs.avg_move_pct:>+9.3f}% "
            f"{r.signal_rate * 100:>9.1f}% "
            f"{verdict:>12}"
        )

    print(f"{'=' * 80}")


def main():
    """Run signal evaluation on all available signal generators."""
    parser = argparse.ArgumentParser(description="Signal evaluation harness")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--days", type=int, default=7, help="Days of data")
    parser.add_argument("--horizons", type=str, default="5,15,30,60", help="Forward look horizons (bars)")
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]

    print(f"\nLoading {args.days} days of {args.symbol} data...")
    loader = DataLoader()
    df_1m = loader.download_binance(args.symbol, "1m", days=args.days)

    if df_1m.empty:
        print("No data loaded")
        return

    print(f"Loaded {len(df_1m):,} bars")

    # Ensure datetime index
    if not isinstance(df_1m.index, pd.DatetimeIndex):
        if 'ts' in df_1m.columns:
            df_1m['datetime'] = pd.to_datetime(df_1m['ts'], unit='ms')
            df_1m = df_1m.set_index('datetime')
        elif 't' in df_1m.columns:
            df_1m['datetime'] = pd.to_datetime(df_1m['t'], unit='ms')
            df_1m = df_1m.set_index('datetime')

    # Aggregate to higher timeframes
    agg = {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}
    df_5m = df_1m.resample("5min").agg(agg).dropna()
    df_15m = df_1m.resample("15min").agg(agg).dropna()
    df_1h = df_1m.resample("1h").agg(agg).dropna()

    evaluator = SignalEvaluator(horizons=horizons)
    results = []

    # ---- Signal 1: Legacy EMA crossover ----
    from strategies.comprehensive_backtest import create_legacy_signal
    print("\nEvaluating Legacy (EMA crossover)...")
    r_legacy = evaluator.evaluate("Legacy (EMA20/50)", df_1m, df_5m, create_legacy_signal)
    print_evaluation(r_legacy)
    results.append(r_legacy)

    # ---- Signal 2: Enhanced signal (7 indicators + MTF) ----
    try:
        from strategies.enhanced_signal import EnhancedSignal

        enhanced = EnhancedSignal(
            min_confluence=0.1,  # Low threshold to see raw accuracy
            min_mtf_alignment=0.1,
            require_mtf_trade=False,  # Don't filter by MTF action
        )

        def enhanced_signal_func(df_1m_w: pd.DataFrame, df_5m_w: pd.DataFrame, i: int):
            # Aggregate windows to higher timeframes
            if len(df_1m_w) < 50:
                return False, False, 0.0
            _5m = df_1m_w.resample("5min").agg(agg).dropna() if len(df_1m_w) >= 10 else None
            _15m = df_1m_w.resample("15min").agg(agg).dropna() if len(df_1m_w) >= 30 else None
            _1h = df_1m_w.resample("1h").agg(agg).dropna() if len(df_1m_w) >= 120 else None
            result = enhanced.analyze(df_1m=df_1m_w, df_5m=_5m, df_15m=_15m, df_1h=_1h)
            return result.long_signal, result.short_signal, result.confidence

        print("\nEvaluating Enhanced (7 indicators + MTF)...")
        r_enhanced = evaluator.evaluate("Enhanced (7ind+MTF)", df_1m, df_5m, enhanced_signal_func)
        print_evaluation(r_enhanced)
        results.append(r_enhanced)
    except Exception as e:
        print(f"Enhanced signal evaluation failed: {e}")

    # ---- Signal 3: Enhanced with strict MTF ----
    try:
        enhanced_strict = EnhancedSignal(
            min_confluence=0.3,
            min_mtf_alignment=0.3,
            require_mtf_trade=True,
        )

        def enhanced_strict_func(df_1m_w: pd.DataFrame, df_5m_w: pd.DataFrame, i: int):
            if len(df_1m_w) < 50:
                return False, False, 0.0
            _5m = df_1m_w.resample("5min").agg(agg).dropna() if len(df_1m_w) >= 10 else None
            _15m = df_1m_w.resample("15min").agg(agg).dropna() if len(df_1m_w) >= 30 else None
            _1h = df_1m_w.resample("1h").agg(agg).dropna() if len(df_1m_w) >= 120 else None
            result = enhanced_strict.analyze(df_1m=df_1m_w, df_5m=_5m, df_15m=_15m, df_1h=_1h)
            return result.long_signal, result.short_signal, result.confidence

        print("\nEvaluating Enhanced (strict MTF)...")
        r_strict = evaluator.evaluate("Enhanced (strict)", df_1m, df_5m, enhanced_strict_func)
        print_evaluation(r_strict)
        results.append(r_strict)
    except Exception as e:
        print(f"Enhanced strict evaluation failed: {e}")

    # ---- Signal 4: Signal Engine only (no MTF) ----
    try:
        from strategies.signal_engine import create_default_engine, SignalDirection

        engine = create_default_engine()

        def engine_only_func(df_1m_w: pd.DataFrame, df_5m_w: pd.DataFrame, i: int):
            if len(df_1m_w) < 50:
                return False, False, 0.0
            result = engine.calculate(df_1m_w)
            is_long = result.direction == SignalDirection.LONG
            is_short = result.direction == SignalDirection.SHORT
            return is_long, is_short, result.confidence

        print("\nEvaluating SignalEngine only (7 indicators, no MTF)...")
        r_engine = evaluator.evaluate("Engine (7ind only)", df_1m, df_5m, engine_only_func)
        print_evaluation(r_engine)
        results.append(r_engine)
    except Exception as e:
        print(f"SignalEngine evaluation failed: {e}")

    # ---- Comparison ----
    if len(results) > 1:
        compare_signals(results, horizon=30)

    return results


if __name__ == "__main__":
    main()
