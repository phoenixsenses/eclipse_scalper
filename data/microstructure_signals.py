# data/microstructure_signals.py — Microstructure Signal Evaluation
#
# STUB: To be completed after 2+ weeks of data collection.
#
# Will evaluate whether microstructure events (liquidation cascades,
# aggression imbalances, large trade clusters) predict price direction
# better than OHLCV-based signals.
#
# Approach:
#   1. Load detected events from microstructure_analysis.py
#   2. For each event, look up 1m OHLCV price at event timestamp
#   3. Measure price movement at 5, 15, 30, 60 minutes forward
#   4. Report directional accuracy per event type
#   5. Gate: >60% accuracy with N >= 30 events
#
# Usage (after data collection):
#   python -m data.microstructure_signals --db-path data/microstructure.db

from __future__ import annotations

# TODO: Implement after collecting 2+ weeks of microstructure data.
# See data/microstructure_analysis.py for event detection.
# See strategies/signal_evaluator.py for the evaluation pattern to follow.

def main():
    print("Signal evaluation not yet implemented.")
    print("Collect microstructure data first:")
    print("  python -m data.microstructure_collector --symbols BTCUSDT,ETHUSDT")
    print("\nThen run analysis to verify events are detected:")
    print("  python -m data.microstructure_analysis --db-path data/microstructure.db")


if __name__ == "__main__":
    main()
