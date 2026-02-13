# strategies/vwap_analysis.py — SCALPER ETERNAL — VWAP DISTANCE ANALYSIS — 2026 v1.0
# Analyze actual VWAP distance_pct distribution in historical data.
#
# Purpose:
# - Find what distance_pct values actually occur on 1m BTC candles
# - Recommend realistic threshold for min_distance_pct
# - Generate distribution statistics and histogram
#
# Usage:
#     python -m strategies.vwap_analysis --symbol BTCUSDT --days 30

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VWAPDistributionStats:
    """Statistics about VWAP distance distribution."""
    total_bars: int
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    # Signal rate at different thresholds
    signal_rates: Dict[float, float]


class VWAPDistanceAnalyzer:
    """Analyze VWAP distance_pct distribution in historical data."""

    def __init__(self, window: int = 240):
        """
        Args:
            window: VWAP rolling window (default 240 = 4 hours on 1m)
        """
        self.window = window

    def calculate_vwap_series(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP for entire DataFrame."""
        # Typical price
        tp = (df["h"] + df["l"] + df["c"]) / 3.0

        # Price * Volume
        pv = tp * df["v"]

        # Rolling VWAP
        cum_pv = pv.rolling(self.window, min_periods=max(10, self.window // 5)).sum()
        cum_v = df["v"].rolling(self.window, min_periods=max(10, self.window // 5)).sum()

        vwap = cum_pv / cum_v.replace(0, np.nan)
        vwap = vwap.ffill()

        return vwap

    def calculate_distance_series(self, df: pd.DataFrame, vwap: pd.Series) -> pd.Series:
        """Calculate distance_pct for entire DataFrame."""
        distance_pct = (df["c"] - vwap).abs() / vwap
        distance_pct = distance_pct.replace([np.inf, -np.inf], np.nan)
        return distance_pct

    def analyze_distribution(self, df: pd.DataFrame) -> VWAPDistributionStats:
        """
        Analyze the distribution of VWAP distance_pct values.

        Returns:
            VWAPDistributionStats with percentiles and signal rates
        """
        # Calculate VWAP
        vwap = self.calculate_vwap_series(df)

        # Calculate distance percentages
        distance_pct = self.calculate_distance_series(df, vwap)

        # Drop NaN values
        distances = distance_pct.dropna()

        # Calculate statistics
        stats = VWAPDistributionStats(
            total_bars=len(distances),
            mean=float(distances.mean()),
            median=float(distances.median()),
            std=float(distances.std()),
            min_val=float(distances.min()),
            max_val=float(distances.max()),
            p25=float(np.percentile(distances, 25)),
            p50=float(np.percentile(distances, 50)),
            p75=float(np.percentile(distances, 75)),
            p90=float(np.percentile(distances, 90)),
            p95=float(np.percentile(distances, 95)),
            p99=float(np.percentile(distances, 99)),
            signal_rates={},
        )

        # Calculate signal rates at different thresholds
        thresholds = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01]
        for threshold in thresholds:
            rate = (distances >= threshold).sum() / len(distances)
            stats.signal_rates[threshold] = float(rate)

        return stats

    def find_optimal_threshold(
        self,
        df: pd.DataFrame,
        target_signal_rate: float = 0.10
    ) -> Tuple[float, float]:
        """
        Find threshold that generates approximately the target signal rate.

        Args:
            df: Historical data
            target_signal_rate: Target percentage of bars with signal (default 10%)

        Returns:
            Tuple of (threshold, actual_rate)
        """
        vwap = self.calculate_vwap_series(df)
        distances = self.calculate_distance_series(df, vwap).dropna()

        # Find percentile that corresponds to target rate
        # If we want 10% signal rate, we need the 90th percentile
        target_percentile = (1.0 - target_signal_rate) * 100

        threshold = float(np.percentile(distances, target_percentile))
        actual_rate = (distances >= threshold).sum() / len(distances)

        return threshold, float(actual_rate)

    def print_analysis(self, stats: VWAPDistributionStats, symbol: str = "BTCUSDT"):
        """Print formatted analysis results."""
        print("=" * 70)
        print(f"VWAP DISTANCE ANALYSIS: {symbol}")
        print("=" * 70)
        print(f"Total bars analyzed: {stats.total_bars:,}")
        print()

        print("DISTRIBUTION STATISTICS:")
        print("-" * 70)
        print(f"  Mean:     {stats.mean * 100:.4f}%")
        print(f"  Median:   {stats.median * 100:.4f}%")
        print(f"  Std Dev:  {stats.std * 100:.4f}%")
        print(f"  Min:      {stats.min_val * 100:.4f}%")
        print(f"  Max:      {stats.max_val * 100:.4f}%")
        print()

        print("PERCENTILES:")
        print("-" * 70)
        print(f"  25th:     {stats.p25 * 100:.4f}%")
        print(f"  50th:     {stats.p50 * 100:.4f}%")
        print(f"  75th:     {stats.p75 * 100:.4f}%")
        print(f"  90th:     {stats.p90 * 100:.4f}%")
        print(f"  95th:     {stats.p95 * 100:.4f}%")
        print(f"  99th:     {stats.p99 * 100:.4f}%")
        print()

        print("SIGNAL RATES BY THRESHOLD:")
        print("-" * 70)
        print(f"  {'Threshold':<12} {'Signal Rate':<15} {'Signals/Day':<15}")
        print("-" * 70)
        for threshold, rate in sorted(stats.signal_rates.items()):
            signals_per_day = rate * 1440  # 1440 minutes per day
            print(f"  {threshold * 100:.3f}%{' ' * 7} {rate * 100:.1f}%{' ' * 10} ~{signals_per_day:.0f}")
        print()

        # Current default analysis
        print("CURRENT vs RECOMMENDED THRESHOLD:")
        print("-" * 70)
        current_threshold = 0.005  # 0.5%
        current_rate = stats.signal_rates.get(current_threshold, 0.0)
        print(f"  Current (0.5%):  {current_rate * 100:.1f}% signal rate ({current_rate * 1440:.0f} signals/day)")

        # Find threshold for ~10% signal rate
        recommended = None
        for th, rate in sorted(stats.signal_rates.items()):
            if rate <= 0.12 and rate >= 0.08:
                recommended = (th, rate)
                break

        if not recommended:
            # Estimate from percentiles
            recommended = (stats.p90, 0.10)

        print(f"  Recommended ({recommended[0] * 100:.3f}%):  ~{recommended[1] * 100:.1f}% signal rate")
        print()

        print("RECOMMENDATION:")
        print("-" * 70)
        if stats.p90 < 0.005:
            print(f"  min_distance_pct should be REDUCED from 0.5% to {stats.p90 * 100:.3f}%")
            print(f"  This will increase signal rate from {current_rate * 100:.1f}% to ~10%")
        else:
            print(f"  Current threshold of 0.5% may be appropriate")
            print(f"  Current signal rate: {current_rate * 100:.1f}%")
        print("=" * 70)


def main():
    """Run VWAP distance analysis."""
    parser = argparse.ArgumentParser(description="Analyze VWAP distance distribution")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--days", type=int, default=30, help="Days of data to analyze")
    parser.add_argument("--window", type=int, default=240, help="VWAP rolling window")
    parser.add_argument("--target-rate", type=float, default=0.10, help="Target signal rate")
    args = parser.parse_args()

    print(f"\nLoading {args.days} days of {args.symbol} 1m data from Binance...")

    # Import DataLoader from backtester
    try:
        from strategies.backtester import DataLoader
    except ImportError:
        print("Error: Could not import DataLoader from backtester")
        return

    # Load data
    loader = DataLoader()
    df = loader.download_binance(args.symbol, "1m", days=args.days)

    if df.empty:
        print("Error: No data loaded")
        return

    print(f"Loaded {len(df):,} bars\n")

    # Analyze
    analyzer = VWAPDistanceAnalyzer(window=args.window)
    stats = analyzer.analyze_distribution(df)

    # Print results
    analyzer.print_analysis(stats, args.symbol)

    # Find optimal threshold
    opt_threshold, opt_rate = analyzer.find_optimal_threshold(df, args.target_rate)
    print(f"\nOPTIMAL THRESHOLD FOR {args.target_rate * 100:.0f}% SIGNAL RATE:")
    print(f"  Threshold: {opt_threshold * 100:.4f}%")
    print(f"  Actual rate: {opt_rate * 100:.1f}%")

    return stats


if __name__ == "__main__":
    main()
