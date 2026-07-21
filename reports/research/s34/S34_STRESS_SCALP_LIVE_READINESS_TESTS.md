# S34 Stress Scalp Live Readiness Tests

Generated: `2026-06-29T13:51:09.298706+00:00`

Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`

## 1. Feature Availability

- `btc4h_bps`: knowable=`True`; prior return ending at signal timestamp
- `vdepth_bps`: knowable=`True`; anchor-local V depth as computed at anchor in current research objects; must be recomputed from past/current marks in live
- `top_hit`: knowable=`True`; selected from prior calibration cells; live must use frozen cells only
- `train_state_density_tail`: knowable=`True`; computed from prior calibration/train rows only
- `hold_state_density_tail`: knowable=`False`; uses complete future holdout fold; research-only contamination
- `near_15m_thresholds`: knowable=`False`; includes future events after current timestamp
- `causal_15m_thresholds`: knowable=`True`; uses events with ts<=current only
- `prior_15m_thresholds`: knowable=`True`; uses events before current timestamp only

## 2. Causal Chain / Stress Score Rebuild

| Selector | All | Final hold | Positive T3R folds | 15m non-overlap first |
| --- | --- | --- | ---: | --- |
| `original_holdstate_near3` | N=132 sum=3081.6 med=5.4 T3R=2496.6 tail150=0 maxLoss=-45.0 | N=132 sum=3081.6 med=5.4 T3R=2496.6 tail150=0 maxLoss=-45.0 | 2/5 | N=46 sum=1378.4 med=10.1 T3R=793.4 tail150=0 maxLoss=-45.0 |
| `live_like_near3` | N=169 sum=285.7 med=-11.7 T3R=-299.3 tail150=0 maxLoss=-45.0 | N=48 sum=1182.4 med=-3.7 T3R=597.4 tail150=0 maxLoss=-45.0 | 2/5 | N=72 sum=707.0 med=-5.6 T3R=122.0 tail150=0 maxLoss=-45.0 |
| `live_like_causal3` | N=97 sum=-418.8 med=-11.7 T3R=-929.3 tail150=0 maxLoss=-45.0 | N=23 sum=54.1 med=-9.5 T3R=-343.4 tail150=0 maxLoss=-45.0 | 0/5 | N=60 sum=-259.5 med=-15.4 T3R=-770.0 tail150=0 maxLoss=-45.0 |
| `live_like_prior3` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None | 0/5 | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |
| `live_like_causal2` | N=188 sum=-971.2 med=-14.8 T3R=-1556.2 tail150=0 maxLoss=-45.0 | N=49 sum=658.5 med=-9.5 T3R=73.5 tail150=0 maxLoss=-45.0 | 0/5 | N=101 sum=-34.9 med=-14.5 T3R=-619.9 tail150=0 maxLoss=-45.0 |

## 3. Entry Delay (live_like_causal3)

| Delay | Summary | Exits |
| --- | --- | --- |
| `delay_0s` | N=97 sum=-418.8 med=-11.7 T3R=-929.3 tail150=0 maxLoss=-45.0 | `{'SL': 36, 'TIME': 60, 'TP': 1}` |
| `delay_5s` | N=97 sum=-414.5 med=-11.6 T3R=-927.5 tail150=0 maxLoss=-45.0 | `{'SL': 36, 'TIME': 60, 'TP': 1}` |
| `delay_15s` | N=97 sum=-417.3 med=-10.2 T3R=-935.9 tail150=0 maxLoss=-45.0 | `{'SL': 35, 'TIME': 61, 'TP': 1}` |
| `delay_30s` | N=97 sum=-472.7 med=-8.7 T3R=-946.3 tail150=0 maxLoss=-45.0 | `{'SL': 33, 'TIME': 63, 'TP': 1}` |
| `delay_60s` | N=97 sum=-464.8 med=-4.9 T3R=-921.2 tail150=0 maxLoss=-45.0 | `{'SL': 32, 'TIME': 64, 'TP': 1}` |

## 4. Exit Robustness (live_like_causal3)

| Exit | Summary | Exits |
| --- | --- | --- |
| `TP150_SL30_15M` | N=97 sum=-471.8 med=-17.7 T3R=-906.8 tail150=0 maxLoss=-35.0 | `{'SL': 42, 'TIME': 52, 'TP': 3}` |
| `TP200_SL40_20M` | N=97 sum=-418.8 med=-11.7 T3R=-929.3 tail150=0 maxLoss=-45.0 | `{'SL': 36, 'TIME': 60, 'TP': 1}` |
| `TP250_SL50_30M` | N=97 sum=-763.8 med=-22.2 T3R=-1325.8 tail150=0 maxLoss=-55.0 | `{'SL': 33, 'TIME': 64}` |
| `TP200_SL50_20M` | N=97 sum=-662.4 med=-11.7 T3R=-1172.9 tail150=0 maxLoss=-55.0 | `{'SL': 27, 'TIME': 69, 'TP': 1}` |
| `TRAIL_PROXY_TP150_SL40_20M` | N=97 sum=-494.3 med=-11.7 T3R=-929.3 tail150=0 maxLoss=-45.0 | `{'SL': 36, 'TIME': 58, 'TP': 3}` |

## 5. SHORT Execution Realism (live_like_causal3)

| Model | Fill rate | Summary | Fill kinds | No-fill counterfactual |
| --- | ---: | --- | --- | --- |
| `taker_immediate` | None | N=97 sum=-418.8 med=-11.7 T3R=-929.3 tail150=0 maxLoss=-45.0 | `{}` | N=None sum=None med=None T3R=None tail150=None maxLoss=None |
| `passive_only_off5_wait15s` | 0.34 | N=33 sum=-94.5 med=-7.3 T3R=-424.0 tail150=0 maxLoss=-45.0 | `{'PASSIVE_FILL': 33, 'NO_FILL': 64}` | N=64 sum=-215.8 med=-9.5 T3R=-667.7 tail150=0 maxLoss=-45.0 |
| `passive_then_taker_off5_wait15s` | 1.0 | N=97 sum=-451.9 med=-8.4 T3R=-969.5 tail150=0 maxLoss=-45.0 | `{'PASSIVE_FILL': 33, 'FALLBACK_TAKER': 64}` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |
| `passive_only_off5_wait30s` | 0.485 | N=47 sum=-230.4 med=-10.0 T3R=-559.9 tail150=0 maxLoss=-45.0 | `{'PASSIVE_FILL': 47, 'NO_FILL': 50}` | N=50 sum=-48.7 med=-8.3 T3R=-500.6 tail150=0 maxLoss=-45.0 |
| `passive_then_taker_off5_wait30s` | 1.0 | N=97 sum=-477.5 med=-8.7 T3R=-965.4 tail150=0 maxLoss=-45.0 | `{'PASSIVE_FILL': 47, 'FALLBACK_TAKER': 50}` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |
| `passive_only_off5_wait60s` | 0.66 | N=64 sum=-405.3 med=-10.0 T3R=-734.8 tail150=0 maxLoss=-45.0 | `{'PASSIVE_FILL': 64, 'NO_FILL': 33}` | N=33 sum=234.9 med=-4.2 T3R=-217.0 tail150=0 maxLoss=-45.0 |
| `passive_then_taker_off5_wait60s` | 1.0 | N=97 sum=-365.5 med=-7.3 T3R=-858.8 tail150=0 maxLoss=-45.0 | `{'PASSIVE_FILL': 64, 'FALLBACK_TAKER': 33}` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |
| `passive_only_off10_wait15s` | 0.124 | N=12 sum=144.8 med=12.3 T3R=-105.3 tail150=0 maxLoss=-45.0 | `{'NO_FILL': 85, 'PASSIVE_FILL': 12}` | N=85 sum=-466.9 med=-14.3 T3R=-977.4 tail150=0 maxLoss=-45.0 |
| `passive_then_taker_off10_wait15s` | 1.0 | N=97 sum=-410.8 med=-8.4 T3R=-927.6 tail150=0 maxLoss=-45.0 | `{'FALLBACK_TAKER': 85, 'PASSIVE_FILL': 12}` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |
| `passive_only_off10_wait30s` | 0.196 | N=19 sum=192.4 med=-3.9 T3R=-74.7 tail150=0 maxLoss=-45.0 | `{'NO_FILL': 78, 'PASSIVE_FILL': 19}` | N=78 sum=-449.1 med=-10.6 T3R=-959.6 tail150=0 maxLoss=-45.0 |
| `passive_then_taker_off10_wait30s` | 1.0 | N=97 sum=-454.2 med=-8.7 T3R=-926.9 tail150=0 maxLoss=-45.0 | `{'FALLBACK_TAKER': 78, 'PASSIVE_FILL': 19}` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |
| `passive_only_off10_wait60s` | 0.299 | N=29 sum=289.5 med=19.5 T3R=22.4 tail150=0 maxLoss=-45.0 | `{'NO_FILL': 68, 'PASSIVE_FILL': 29}` | N=68 sum=-459.3 med=-15.4 T3R=-969.8 tail150=0 maxLoss=-45.0 |
| `passive_then_taker_off10_wait60s` | 1.0 | N=97 sum=-328.5 med=-4.9 T3R=-784.9 tail150=0 maxLoss=-45.0 | `{'FALLBACK_TAKER': 68, 'PASSIVE_FILL': 29}` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |
| `passive_only_off20_wait15s` | 0.0 | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None | `{'NO_FILL': 97}` | N=97 sum=-418.8 med=-11.7 T3R=-929.3 tail150=0 maxLoss=-45.0 |
| `passive_then_taker_off20_wait15s` | 1.0 | N=97 sum=-416.1 med=-8.4 T3R=-933.0 tail150=0 maxLoss=-45.0 | `{'FALLBACK_TAKER': 97}` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |
| `passive_only_off20_wait30s` | 0.01 | N=1 sum=-45.0 med=-45.0 T3R=-45.0 tail150=0 maxLoss=-45.0 | `{'NO_FILL': 96, 'PASSIVE_FILL': 1}` | N=96 sum=-373.8 med=-10.6 T3R=-884.3 tail150=0 maxLoss=-45.0 |
| `passive_then_taker_off20_wait30s` | 1.0 | N=97 sum=-482.1 med=-8.7 T3R=-954.9 tail150=0 maxLoss=-45.0 | `{'FALLBACK_TAKER': 96, 'PASSIVE_FILL': 1}` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |
| `passive_only_off20_wait60s` | 0.021 | N=2 sum=-90.0 med=-45.0 T3R=-90.0 tail150=0 maxLoss=-45.0 | `{'NO_FILL': 95, 'PASSIVE_FILL': 2}` | N=95 sum=-328.8 med=-9.5 T3R=-839.3 tail150=0 maxLoss=-45.0 |
| `passive_then_taker_off20_wait60s` | 1.0 | N=97 sum=-463.5 med=-4.9 T3R=-919.9 tail150=0 maxLoss=-45.0 | `{'FALLBACK_TAKER': 95, 'PASSIVE_FILL': 2}` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |

## 6. Duplicate / Overlap Guard (live_like_causal3)

| Policy | Summary | Exits |
| --- | --- | --- |
| `raw` | N=97 sum=-418.8 med=-11.7 T3R=-929.3 tail150=0 maxLoss=-45.0 | `{'SL': 36, 'TIME': 60, 'TP': 1}` |
| `dedup_60s_first` | N=61 sum=-304.5 med=-15.5 T3R=-815.0 tail150=0 maxLoss=-45.0 | `{'SL': 25, 'TIME': 35, 'TP': 1}` |
| `dedup_300s_first` | N=61 sum=-304.5 med=-15.5 T3R=-815.0 tail150=0 maxLoss=-45.0 | `{'SL': 25, 'TIME': 35, 'TP': 1}` |
| `dedup_900s_first` | N=60 sum=-259.5 med=-15.4 T3R=-770.0 tail150=0 maxLoss=-45.0 | `{'SL': 24, 'TIME': 35, 'TP': 1}` |
| `dedup_1800s_first` | N=56 sum=-122.3 med=-15.4 T3R=-632.8 tail150=0 maxLoss=-45.0 | `{'SL': 21, 'TIME': 34, 'TP': 1}` |

## 7. Stress Bucket Kill Rule Simulation (live_like_causal3)

| Rule | Traded N | Pauses | Summary |
| --- | ---: | ---: | --- |
| `pause_after_1_sl` | 54 | 13 | N=54 sum=97.3 med=-8.3 T3R=-319.6 tail150=0 maxLoss=-45.0 |
| `pause_after_2_sl` | 65 | 8 | N=65 sum=-16.7 med=-9.5 T3R=-527.2 tail150=0 maxLoss=-45.0 |
| `rolling_3_sum_lt_-90` | 69 | 8 | N=69 sum=-196.7 med=-11.7 T3R=-707.2 tail150=0 maxLoss=-45.0 |
| `rolling_5_sum_lt_-120` | 72 | 6 | N=72 sum=-305.5 med=-15.4 T3R=-816.0 tail150=0 maxLoss=-45.0 |
| `daily_loss_lt_-90` | 97 | 0 | N=97 sum=-418.8 med=-11.7 T3R=-929.3 tail150=0 maxLoss=-45.0 |

## 8. v0.2 Conflict Matrix (live_like_causal3)

- stress N: `97`
- v0.2 N: `31`
- conflict within 15m N: `11`
- conflict stress summary: N=11 sum=-260.7 med=-45.0 T3R=-295.4 tail150=0 maxLoss=-45.0; exits `{'SL': 6, 'TIME': 5}`
