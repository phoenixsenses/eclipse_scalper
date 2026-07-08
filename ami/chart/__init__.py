"""Chart-native object foundation (Phase 4). Candle/Swing/Level/Push objects
per AMI_CHART_NATIVE_PRICE_STRUCTURE_INTELLIGENCE_EXTENSION_v1.0 §4/§6/§7.

Closed-candle-only, known_at_ts-disciplined. No observer is activated by
this package -- purely descriptive/historical computation (master protocol
§12: chart-native stays downstream of event/cycle/timing/evidence
foundations, per DOCUMENT_RECONCILIATION_MATRIX.md wave sequencing).
"""
from ami.chart.candle_builder import CANDLE_DEFINITION_VERSION, build_candles

__all__ = ["CANDLE_DEFINITION_VERSION", "build_candles"]
