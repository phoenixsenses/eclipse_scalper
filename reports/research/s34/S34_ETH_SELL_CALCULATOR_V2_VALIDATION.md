# ETH SELL Calculator v2 Validation

Generated: 2026-06-26T15:22:00+00:00
Combo: ETHUSDT SELL  |  Primary route: SHORT_DELAY0_TP60
N events: 222  |  Train: 155 (70%)  |  Test: 67 (30%)

## Config Comparison

| Config | K | Metric | Features |
|---|---:|---|---|
| Old (default) | 20 | euclidean | all 11 |
| v2 | 10 | manhattan | 6 (excluded: btc_pre_15m_bps, cluster_duration_sec, cluster_liq_count, day_trend_bps, symbol_pre_15m_bps) |

## Results

| Config | N_test | DirAcc | MAE | PredMedian | RealMedian | BaseRate | PredStd |
|---|---:|---:|---:|---:|---:|---:|---:|
| Old | 67 | 75% | 22.0 | 52.4 | 52.4 | 37.7 | 23.3 |
| **v2** | 67 | **72%** | **23.9** | 52.4 | 52.4 | 37.7 | 24.0 |

## Delta (v2 - old)

- dir_acc: `-0.030` (-3.0 pp)
- MAE: `+1.9` bps

## Verdict: V2_NEUTRAL

No significant difference. v2 is not harmful; deploy for consistency with research findings.

## Notes

- All evaluation is temporal OOS (test = last 30%, strictly after train).
- No forward-looking features; distance computed on train pool only per test event.
- v2 config is now the default for ETH SELL in `s34_liq_outcome_calculator.py`.
- Other combos are unaffected.
