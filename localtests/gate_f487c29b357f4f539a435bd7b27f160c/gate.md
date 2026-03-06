# Canary Expansion Gate

- ts_utc: 2026-03-05T19:41:36Z
- verdict: GO
- window_days: 7
- required_max_top_score: 0.500
- days_observed: 7
- coverage_ok: 1
- score_ok: 1

## Daily Top Causes
| date | top_cause | top_score |
|---|---|---:|
| 2026-03-01 | Latency Modeling Drift | 0.420 |
| 2026-03-02 | Latency Modeling Drift | 0.420 |
| 2026-03-03 | Latency Modeling Drift | 0.420 |
| 2026-03-04 | Latency Modeling Drift | 0.420 |
| 2026-03-05 | Latency Modeling Drift | 0.420 |
| 2026-03-06 | Latency Modeling Drift | 0.420 |
| 2026-03-07 | Latency Modeling Drift | 0.420 |

## Policy
- GO only if all observed days in window have top_score below threshold and coverage is full.
- HOLD otherwise; keep canary only and continue daily calibration.

## Run Summary
- `{'version': 'v1', 'run_type': 'evaluate_canary_expansion_gate', 'inputs': {'report_dir': 'localtests\\gate_f487c29b357f4f539a435bd7b27f160c', 'window_days': 7, 'max_top_score': 0.5}, 'metrics': {'passed': True, 'days_observed': 7}, 'artifacts': {'json': 'localtests\\gate_f487c29b357f4f539a435bd7b27f160c\\gate.json', 'md': 'localtests\\gate_f487c29b357f4f539a435bd7b27f160c\\gate.md'}}`
