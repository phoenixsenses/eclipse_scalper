# strategies/signal_diagnostics.py — Signal Engine Diagnostic Tool
# Analyzes why signals are/aren't being generated

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np

# Import signal components
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
    TimeframeSignal,
)
from strategies.enhanced_signal import EnhancedSignal


@dataclass
class IndicatorDiagnostic:
    """Diagnostic data for a single indicator."""
    name: str
    direction: str
    strength: str
    confidence: float
    value: float
    is_contributing: bool
    is_blocking: bool


@dataclass
class MTFDiagnostic:
    """Diagnostic data for MTF analysis."""
    timeframe: str
    trend: str
    momentum: float
    structure: str
    confidence: float
    ema_fast: float
    ema_slow: float


@dataclass
class SignalDiagnostic:
    """Complete diagnostic for a single bar."""
    bar_index: int
    timestamp: int
    close: float

    # Indicator analysis
    indicators: List[IndicatorDiagnostic] = field(default_factory=list)
    ind_direction: str = "NEUTRAL"
    ind_confidence: float = 0.0
    ind_long_votes: float = 0.0
    ind_short_votes: float = 0.0
    ind_conflict: float = 0.0

    # MTF analysis
    mtf_signals: List[MTFDiagnostic] = field(default_factory=list)
    mtf_direction: str = "NEUTRAL"
    mtf_confluence: float = 0.0
    mtf_alignment: float = 0.0
    mtf_action: str = "SKIP"
    mtf_conf_mult: float = 0.0
    mtf_reasons: List[str] = field(default_factory=list)

    # Final result
    long_signal: bool = False
    short_signal: bool = False
    final_confidence: float = 0.0
    block_reasons: List[str] = field(default_factory=list)


class SignalDiagnostics:
    """
    Diagnostic tool for analyzing signal generation.

    Usage:
        diag = SignalDiagnostics()
        results = diag.analyze_dataset(df_1m, df_5m, df_15m, df_1h)
        diag.print_summary(results)
    """

    def __init__(self):
        self.engine = create_default_engine()
        self.mtf = MTFConfluence()
        self.enhanced = EnhancedSignal()

    def analyze_bar(
        self,
        bar_index: int,
        df_1m: pd.DataFrame,
        df_5m: Optional[pd.DataFrame] = None,
        df_15m: Optional[pd.DataFrame] = None,
        df_1h: Optional[pd.DataFrame] = None,
    ) -> SignalDiagnostic:
        """Analyze a single bar with full diagnostics."""

        row = df_1m.iloc[bar_index]
        diag = SignalDiagnostic(
            bar_index=bar_index,
            timestamp=int(row["ts"]),
            close=float(row["c"]),
        )

        # Get historical slices
        lookback = min(100, bar_index)
        hist_1m = df_1m.iloc[max(0, bar_index-lookback):bar_index+1].copy()

        # Calculate 5m/15m/1h indices
        idx_5m = bar_index // 5
        idx_15m = bar_index // 15
        idx_1h = bar_index // 60

        hist_5m = df_5m.iloc[:idx_5m+1].copy() if df_5m is not None and len(df_5m) > idx_5m else None
        hist_15m = df_15m.iloc[:idx_15m+1].copy() if df_15m is not None and len(df_15m) > idx_15m else None
        hist_1h = df_1h.iloc[:idx_1h+1].copy() if df_1h is not None and len(df_1h) > idx_1h else None

        # 1. Analyze indicators
        try:
            ind_result = self.engine.calculate(hist_1m, symbol="DIAG")

            diag.ind_direction = ind_result.direction.value
            diag.ind_confidence = ind_result.confidence
            diag.ind_long_votes = ind_result.long_votes
            diag.ind_short_votes = ind_result.short_votes
            diag.ind_conflict = ind_result.conflict_level

            # Per-indicator details
            for name, signal in ind_result.signals.items():
                diag.indicators.append(IndicatorDiagnostic(
                    name=name,
                    direction=signal.direction.value,
                    strength=signal.strength.value,
                    confidence=signal.confidence,
                    value=signal.value,
                    is_contributing=name in ind_result.contributing_indicators,
                    is_blocking=name in ind_result.blocking_indicators,
                ))
        except Exception as e:
            diag.block_reasons.append(f"indicator_error: {e}")

        # 2. Analyze MTF
        try:
            mtf_dfs = {"1m": hist_1m}
            if hist_5m is not None and len(hist_5m) >= 50:
                mtf_dfs["5m"] = hist_5m
            if hist_15m is not None and len(hist_15m) >= 50:
                mtf_dfs["15m"] = hist_15m
            if hist_1h is not None and len(hist_1h) >= 50:
                mtf_dfs["1h"] = hist_1h

            mtf_result = self.mtf.analyze(mtf_dfs)

            diag.mtf_direction = mtf_result.direction.value
            diag.mtf_confluence = mtf_result.confluence_score
            diag.mtf_alignment = mtf_result.alignment
            diag.mtf_action = mtf_result.recommended_action
            diag.mtf_conf_mult = mtf_result.confidence_multiplier
            diag.mtf_reasons = list(mtf_result.reasons)

            # Per-timeframe details
            for tf, signal in mtf_result.signals.items():
                diag.mtf_signals.append(MTFDiagnostic(
                    timeframe=tf,
                    trend=signal.trend.value,
                    momentum=signal.momentum,
                    structure=signal.structure.value,
                    confidence=signal.confidence,
                    ema_fast=signal.ema_fast,
                    ema_slow=signal.ema_slow,
                ))
        except Exception as e:
            diag.block_reasons.append(f"mtf_error: {e}")

        # 3. Determine final signal
        # Check why signal might be blocked
        if diag.ind_confidence < 0.5:
            diag.block_reasons.append(f"ind_conf_low:{diag.ind_confidence:.2f}")

        if diag.mtf_action != "TRADE":
            diag.block_reasons.append(f"mtf_action:{diag.mtf_action}")

        if diag.mtf_alignment < 0.3:
            diag.block_reasons.append(f"mtf_align_low:{diag.mtf_alignment:.2f}")

        if diag.ind_direction == "NEUTRAL":
            diag.block_reasons.append("ind_neutral")

        if diag.mtf_direction == "NEUTRAL":
            diag.block_reasons.append("mtf_neutral")

        # Check direction agreement
        ind_bull = diag.ind_direction == "LONG"
        ind_bear = diag.ind_direction == "SHORT"
        mtf_bull = diag.mtf_direction == "BULLISH"
        mtf_bear = diag.mtf_direction == "BEARISH"

        if ind_bull and not mtf_bull:
            diag.block_reasons.append("direction_mismatch:ind_long_mtf_not_bull")
        if ind_bear and not mtf_bear:
            diag.block_reasons.append("direction_mismatch:ind_short_mtf_not_bear")

        # Final signal
        if len(diag.block_reasons) == 0:
            if ind_bull and mtf_bull:
                diag.long_signal = True
                diag.final_confidence = (diag.ind_confidence * 0.6) + (diag.mtf_confluence * 0.4)
            elif ind_bear and mtf_bear:
                diag.short_signal = True
                diag.final_confidence = (diag.ind_confidence * 0.6) + (diag.mtf_confluence * 0.4)

        return diag

    def analyze_dataset(
        self,
        df_1m: pd.DataFrame,
        df_5m: Optional[pd.DataFrame] = None,
        df_15m: Optional[pd.DataFrame] = None,
        df_1h: Optional[pd.DataFrame] = None,
        sample_every: int = 60,  # Analyze every N bars
        start_bar: int = 100,
    ) -> List[SignalDiagnostic]:
        """Analyze entire dataset with sampling."""
        results = []

        total_bars = len(df_1m)
        print(f"Analyzing {total_bars} bars (sampling every {sample_every})...")

        for i in range(start_bar, total_bars, sample_every):
            if i % 1000 == 0:
                print(f"  Progress: {i}/{total_bars} ({100*i/total_bars:.1f}%)")

            diag = self.analyze_bar(i, df_1m, df_5m, df_15m, df_1h)
            results.append(diag)

        return results

    def get_statistics(self, results: List[SignalDiagnostic]) -> Dict[str, Any]:
        """Calculate statistics from diagnostic results."""
        stats = {
            "total_bars": len(results),
            "signals": {
                "long": sum(1 for r in results if r.long_signal),
                "short": sum(1 for r in results if r.short_signal),
                "total": sum(1 for r in results if r.long_signal or r.short_signal),
            },
            "indicator_stats": defaultdict(lambda: {
                "long": 0, "short": 0, "neutral": 0,
                "contributing": 0, "blocking": 0,
                "avg_confidence": [],
            }),
            "mtf_stats": defaultdict(lambda: {
                "bullish": 0, "bearish": 0, "neutral": 0,
                "avg_confidence": [],
                "avg_momentum": [],
            }),
            "block_reasons": defaultdict(int),
            "mtf_actions": defaultdict(int),
            "direction_distribution": {
                "ind_long": 0, "ind_short": 0, "ind_neutral": 0,
                "mtf_bullish": 0, "mtf_bearish": 0, "mtf_neutral": 0,
            },
            "confidence_distribution": {
                "ind_conf_mean": 0.0,
                "ind_conf_max": 0.0,
                "mtf_confluence_mean": 0.0,
                "mtf_alignment_mean": 0.0,
            },
        }

        ind_confs = []
        mtf_confs = []
        mtf_aligns = []

        for r in results:
            # Block reasons
            for reason in r.block_reasons:
                # Extract category
                if ":" in reason:
                    cat = reason.split(":")[0]
                else:
                    cat = reason
                stats["block_reasons"][cat] += 1

            # MTF actions
            stats["mtf_actions"][r.mtf_action] += 1

            # Direction distribution
            if r.ind_direction == "LONG":
                stats["direction_distribution"]["ind_long"] += 1
            elif r.ind_direction == "SHORT":
                stats["direction_distribution"]["ind_short"] += 1
            else:
                stats["direction_distribution"]["ind_neutral"] += 1

            if r.mtf_direction == "BULLISH":
                stats["direction_distribution"]["mtf_bullish"] += 1
            elif r.mtf_direction == "BEARISH":
                stats["direction_distribution"]["mtf_bearish"] += 1
            else:
                stats["direction_distribution"]["mtf_neutral"] += 1

            # Confidence
            ind_confs.append(r.ind_confidence)
            mtf_confs.append(r.mtf_confluence)
            mtf_aligns.append(r.mtf_alignment)

            # Per-indicator stats
            for ind in r.indicators:
                ist = stats["indicator_stats"][ind.name]
                if ind.direction == "LONG":
                    ist["long"] += 1
                elif ind.direction == "SHORT":
                    ist["short"] += 1
                else:
                    ist["neutral"] += 1
                if ind.is_contributing:
                    ist["contributing"] += 1
                if ind.is_blocking:
                    ist["blocking"] += 1
                ist["avg_confidence"].append(ind.confidence)

            # Per-MTF stats
            for mtf in r.mtf_signals:
                mst = stats["mtf_stats"][mtf.timeframe]
                if mtf.trend == "BULLISH":
                    mst["bullish"] += 1
                elif mtf.trend == "BEARISH":
                    mst["bearish"] += 1
                else:
                    mst["neutral"] += 1
                mst["avg_confidence"].append(mtf.confidence)
                mst["avg_momentum"].append(mtf.momentum)

        # Finalize averages
        stats["confidence_distribution"]["ind_conf_mean"] = np.mean(ind_confs) if ind_confs else 0.0
        stats["confidence_distribution"]["ind_conf_max"] = np.max(ind_confs) if ind_confs else 0.0
        stats["confidence_distribution"]["mtf_confluence_mean"] = np.mean(mtf_confs) if mtf_confs else 0.0
        stats["confidence_distribution"]["mtf_alignment_mean"] = np.mean(mtf_aligns) if mtf_aligns else 0.0

        # Finalize indicator stats
        for name, ist in stats["indicator_stats"].items():
            confs = ist["avg_confidence"]
            ist["avg_confidence"] = np.mean(confs) if confs else 0.0

        # Finalize MTF stats
        for tf, mst in stats["mtf_stats"].items():
            confs = mst["avg_confidence"]
            moms = mst["avg_momentum"]
            mst["avg_confidence"] = np.mean(confs) if confs else 0.0
            mst["avg_momentum"] = np.mean(moms) if moms else 0.0

        return stats

    def print_summary(self, results: List[SignalDiagnostic]) -> None:
        """Print diagnostic summary."""
        stats = self.get_statistics(results)

        print("\n" + "=" * 70)
        print("SIGNAL DIAGNOSTIC SUMMARY")
        print("=" * 70)

        total = stats["total_bars"]
        sigs = stats["signals"]

        print(f"\nBars Analyzed: {total}")
        print(f"Signals Generated: {sigs['total']} ({100*sigs['total']/total:.2f}%)")
        print(f"  Long: {sigs['long']}, Short: {sigs['short']}")

        # Block reasons
        print("\n--- BLOCK REASONS (why no signal) ---")
        reasons = sorted(stats["block_reasons"].items(), key=lambda x: -x[1])
        for reason, count in reasons[:10]:
            pct = 100 * count / total
            print(f"  {reason:40} {count:5} ({pct:5.1f}%)")

        # MTF Actions
        print("\n--- MTF RECOMMENDED ACTIONS ---")
        for action, count in sorted(stats["mtf_actions"].items(), key=lambda x: -x[1]):
            pct = 100 * count / total
            print(f"  {action:15} {count:5} ({pct:5.1f}%)")

        # Direction distribution
        print("\n--- DIRECTION DISTRIBUTION ---")
        dd = stats["direction_distribution"]
        print(f"  Indicators:  LONG={dd['ind_long']} ({100*dd['ind_long']/total:.1f}%)  "
              f"SHORT={dd['ind_short']} ({100*dd['ind_short']/total:.1f}%)  "
              f"NEUTRAL={dd['ind_neutral']} ({100*dd['ind_neutral']/total:.1f}%)")
        print(f"  MTF:         BULL={dd['mtf_bullish']} ({100*dd['mtf_bullish']/total:.1f}%)  "
              f"BEAR={dd['mtf_bearish']} ({100*dd['mtf_bearish']/total:.1f}%)  "
              f"NEUT={dd['mtf_neutral']} ({100*dd['mtf_neutral']/total:.1f}%)")

        # Confidence stats
        print("\n--- CONFIDENCE STATISTICS ---")
        cd = stats["confidence_distribution"]
        print(f"  Indicator Confidence: mean={cd['ind_conf_mean']:.3f}, max={cd['ind_conf_max']:.3f}")
        print(f"  MTF Confluence: mean={cd['mtf_confluence_mean']:.3f}")
        print(f"  MTF Alignment: mean={cd['mtf_alignment_mean']:.3f}")

        # Per-indicator analysis
        print("\n--- PER-INDICATOR ANALYSIS ---")
        print(f"  {'Indicator':<15} {'LONG':>6} {'SHORT':>6} {'NEUT':>6} {'AvgConf':>8} {'Contrib':>7} {'Block':>6}")
        print("  " + "-" * 60)
        for name, ist in sorted(stats["indicator_stats"].items()):
            print(f"  {name:<15} {ist['long']:>6} {ist['short']:>6} {ist['neutral']:>6} "
                  f"{ist['avg_confidence']:>8.3f} {ist['contributing']:>7} {ist['blocking']:>6}")

        # Per-MTF analysis
        print("\n--- PER-TIMEFRAME MTF ANALYSIS ---")
        print(f"  {'TF':<6} {'BULL':>6} {'BEAR':>6} {'NEUT':>6} {'AvgConf':>8} {'AvgMom':>8}")
        print("  " + "-" * 45)
        for tf in ["1m", "5m", "15m", "1h"]:
            if tf in stats["mtf_stats"]:
                mst = stats["mtf_stats"][tf]
                print(f"  {tf:<6} {mst['bullish']:>6} {mst['bearish']:>6} {mst['neutral']:>6} "
                      f"{mst['avg_confidence']:>8.3f} {mst['avg_momentum']:>+8.4f}")

        print("\n" + "=" * 70)

    def find_near_signals(
        self,
        results: List[SignalDiagnostic],
        min_ind_conf: float = 0.3,
        top_n: int = 10,
    ) -> List[SignalDiagnostic]:
        """Find bars that were close to generating signals."""
        # Sort by combined confidence
        scored = []
        for r in results:
            if r.long_signal or r.short_signal:
                continue  # Skip actual signals

            score = r.ind_confidence * 0.6 + r.mtf_confluence * 0.4
            if r.ind_confidence >= min_ind_conf:
                scored.append((score, r))

        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:top_n]]

    def print_near_signals(self, results: List[SignalDiagnostic], top_n: int = 5) -> None:
        """Print analysis of near-miss signals."""
        near = self.find_near_signals(results, top_n=top_n)

        print("\n--- NEAR-MISS SIGNALS (closest to triggering) ---")
        for i, r in enumerate(near):
            print(f"\n  #{i+1} Bar {r.bar_index} @ {r.close:.2f}")
            print(f"     Ind: dir={r.ind_direction}, conf={r.ind_confidence:.3f}, "
                  f"L/S={r.ind_long_votes:.2f}/{r.ind_short_votes:.2f}")
            print(f"     MTF: dir={r.mtf_direction}, conf={r.mtf_confluence:.3f}, "
                  f"align={r.mtf_alignment:.3f}, action={r.mtf_action}")
            print(f"     Blocks: {', '.join(r.block_reasons[:3])}")


def run_diagnostics(
    symbol: str = "BTCUSDT",
    days: int = 30,
    sample_every: int = 60,
) -> Dict[str, Any]:
    """Run full diagnostic analysis."""
    from strategies.backtester import DataLoader

    # Load data
    loader = DataLoader()
    print(f"Loading {symbol} data for {days} days...")

    df_1m = loader.download_binance(symbol, "1m", days=days)
    df_5m = loader.aggregate_timeframe(df_1m, "5m")
    df_15m = loader.aggregate_timeframe(df_1m, "15m")
    df_1h = loader.aggregate_timeframe(df_1m, "1h")

    print(f"Data: 1m={len(df_1m)}, 5m={len(df_5m)}, 15m={len(df_15m)}, 1h={len(df_1h)}")

    # Run diagnostics
    diag = SignalDiagnostics()
    results = diag.analyze_dataset(df_1m, df_5m, df_15m, df_1h, sample_every=sample_every)

    # Print summary
    diag.print_summary(results)
    diag.print_near_signals(results)

    return diag.get_statistics(results)


if __name__ == "__main__":
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    run_diagnostics(symbol, days)
