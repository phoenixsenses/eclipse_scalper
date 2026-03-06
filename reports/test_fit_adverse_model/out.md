# ADVERSE SELECTION MODEL — DIAGNOSTICS

generated : 2026-03-05T00:00:00Z
git_hash  : abc1234
rule      : micro_edge_v3_passive_alpha
horizon   : 120s
lookback  : 1440min
bucket_sec: 1

## ETHUSDT

- Total buckets  : 10
- Signal-firing  : 5
- Global mean adverse_bps : `1.5000`
- Global std              : `0.4000`
- Implied passive_adverse_mult vs 1bps baseline: `1.5`

**Percentiles (adverse_bps):**

## Run Summary
- `{'version': 'v1', 'run_type': 'fit_adverse_model', 'inputs': {'db': 'data/microstructure.db', 'symbols': ['ETHUSDT'], 'lookback_min': 1440, 'bucket_sec': 1, 'rule': 'micro_edge_v3_passive_alpha', 'horizon_sec': 120, 'side': 'LONG'}, 'metrics': {'symbol_count': 1, 'error_count': 0}, 'artifacts': {'json': 'reports\\test_fit_adverse_model\\out.json', 'md': 'reports\\test_fit_adverse_model\\out.md'}}`

