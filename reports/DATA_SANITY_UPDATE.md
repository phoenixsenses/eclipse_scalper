# DATA_SANITY_UPDATE

source=logs\micro_edge_debug_trades.jsonl

- rows=1344, invalid_json=0
- side_coverage={'SHORT': 639, 'LONG': 705}
- event_counts={'trade': 1176, 'passive_toxicity_block': 168}
- label_distribution={'up': 369, 'down': 408, 'flat': 399, '': 168}
- direction_match_rate_excluding_flat=44.02%

## Alignment checks
- entry_idx <= signal_idx violations: 0
- exit_idx <= entry_idx violations: 0
- lookahead_structural_flag=False

## Missingness (feature dict)
- spread: missing=0 (0.00%)
- trade_intensity: missing=0 (0.00%)
- micro_volatility: missing=0 (0.00%)
- imbalance: missing=0 (0.00%)
- ret_1: missing=0 (0.00%)

## Duplicates
- duplicate_event_id_count=0
- duplicate_ts_bucket_count=40

## Horizon/timing strings
- "signal at t, entry at t+1 mark, exit at t+1+h mark": 1176
- "signal at t blocked pre-post": 168

## Findings
- No structural alignment violations detected in debug rows (entry after signal, exit after entry).
- No direct evidence of lookahead from debug-index invariants; label/timing remains event at t, entry at t+1, exit at t+1+h.
