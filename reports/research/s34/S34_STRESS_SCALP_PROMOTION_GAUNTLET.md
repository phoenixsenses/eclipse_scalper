# S34 Stress Scalp Promotion Gauntlet

Generated: `2026-06-29T13:44:53.647717+00:00`

Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`

Primary: `S3_BTC75_VLT50_CHAIN3_REV_TP200_SL40_20M`

Definition: `{'selector': 'stress_score>=3 AND btc4h_bps<-75 AND vdepth_bps<50 AND chain_near_15m_thresholds>=3', 'direction': 'REVERSE_SHORT', 'exit': 'TP200_SL40_20M', 'fee_bps': 5.0}`

Rows: `1204`; candidate N: `132`

## Promotion Verdict

Verdict: `SHADOW_ONLY`

Hard fail reasons: `['walkforward', 'regime_concentration']`

| Check | Pass? |
| --- | ---: |
| `causal_holdout` | `True` |
| `walkforward` | `False` |
| `non_overlap_15m` | `True` |
| `permutation` | `True` |
| `exit_robustness` | `True` |
| `fee_sensitivity` | `True` |
| `execution_realism` | `True` |
| `big_winner_loser` | `True` |
| `regime_concentration` | `False` |

## 1. Causal Holdout

N=132 sum=3089.8 med=5.4 T3R=2504.8 tail150=0 maxLoss=-45.0; exits `{'SL': 38, 'TIME': 84, 'TP': 10}`

## 2. Walk-Forward Stability

Positive T3R folds: `2/5`; fold T3R total `2102.1`

| Fold | Summary | Exits |
| --- | --- | --- |
| `fold_1` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None | `{}` |
| `fold_2` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None | `{}` |
| `fold_3` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None | `{}` |
| `fold_4` | N=50 sum=690.5 med=6.7 T3R=287.8 tail150=0 maxLoss=-45.0 | `{'SL': 9, 'TIME': 40, 'TP': 1}` |
| `fold_5` | N=82 sum=2399.3 med=-0.2 T3R=1814.3 tail150=0 maxLoss=-45.0 | `{'TIME': 44, 'TP': 9, 'SL': 29}` |

## 3. Non-Overlap

| Policy | Summary | Exits |
| --- | --- | --- |
| `overlap` | N=132 sum=3089.8 med=5.4 T3R=2504.8 tail150=0 maxLoss=-45.0 | `{'SL': 38, 'TIME': 84, 'TP': 10}` |
| `nonoverlap_15m_first` | N=46 sum=1381.0 med=10.4 T3R=796.0 tail150=0 maxLoss=-45.0 | `{'SL': 12, 'TIME': 30, 'TP': 4}` |
| `nonoverlap_15m_max_threshold` | N=46 sum=1008.0 med=-2.6 T3R=423.0 tail150=0 maxLoss=-45.0 | `{'SL': 13, 'TIME': 30, 'TP': 3}` |
| `nonoverlap_15m_min_vdepth` | N=46 sum=1122.0 med=7.6 T3R=537.0 tail150=0 maxLoss=-45.0 | `{'SL': 12, 'TIME': 31, 'TP': 3}` |
| `nonoverlap_30m_first` | N=33 sum=859.3 med=7.4 T3R=288.8 tail150=0 maxLoss=-45.0 | `{'SL': 8, 'TIME': 23, 'TP': 2}` |
| `nonoverlap_30m_max_threshold` | N=33 sum=540.8 med=-5.3 T3R=9.5 tail150=0 maxLoss=-45.0 | `{'SL': 9, 'TIME': 23, 'TP': 1}` |
| `nonoverlap_30m_min_vdepth` | N=33 sum=496.5 med=6.6 T3R=54.7 tail150=0 maxLoss=-45.0 | `{'SL': 9, 'TIME': 24}` |
| `nonoverlap_60m_first` | N=29 sum=714.3 med=7.4 T3R=198.3 tail150=0 maxLoss=-45.0 | `{'SL': 7, 'TIME': 21, 'TP': 1}` |
| `nonoverlap_60m_max_threshold` | N=29 sum=474.2 med=-3.8 T3R=-22.6 tail150=0 maxLoss=-45.0 | `{'SL': 7, 'TIME': 21, 'TP': 1}` |
| `nonoverlap_60m_min_vdepth` | N=29 sum=409.7 med=6.6 T3R=-32.1 tail150=0 maxLoss=-45.0 | `{'SL': 8, 'TIME': 21}` |

## 4. Max-Statistic Permutation

95pct max-stat T3R: `590.7`
Primary p: `{'real_t3r': 2504.8, 'mc_p': 0.001}`

## 5. Exit Robustness

| Exit | Summary | Exits |
| --- | --- | --- |
| `TP150_SL30_15M` | N=132 sum=2208.1 med=0.1 T3R=1773.1 tail150=0 maxLoss=-35.0 | `{'SL': 52, 'TIME': 66, 'TP': 14}` |
| `TP200_SL40_20M` | N=132 sum=3089.8 med=5.4 T3R=2504.8 tail150=0 maxLoss=-45.0 | `{'SL': 38, 'TIME': 84, 'TP': 10}` |
| `TP250_SL50_30M` | N=132 sum=1896.4 med=-10.8 T3R=1161.4 tail150=0 maxLoss=-55.0 | `{'SL': 40, 'TIME': 87, 'TP': 5}` |
| `TP200_SL30_20M` | N=132 sum=2805.3 med=-7.9 T3R=2220.3 tail150=0 maxLoss=-35.0 | `{'SL': 53, 'TIME': 69, 'TP': 10}` |
| `TP150_SL40_20M` | N=132 sum=2545.6 med=5.4 T3R=2110.6 tail150=0 maxLoss=-45.0 | `{'SL': 38, 'TIME': 77, 'TP': 17}` |

## 6. Fee Sensitivity

| Fee | Summary | Exits |
| --- | --- | --- |
| `fee_0bps` | N=132 sum=3749.8 med=10.4 T3R=3149.8 tail150=0 maxLoss=-40.0 | `{'SL': 38, 'TIME': 84, 'TP': 10}` |
| `fee_2.5bps` | N=132 sum=3419.8 med=7.9 T3R=2827.3 tail150=0 maxLoss=-42.5 | `{'SL': 38, 'TIME': 84, 'TP': 10}` |
| `fee_5bps` | N=132 sum=3089.8 med=5.4 T3R=2504.8 tail150=0 maxLoss=-45.0 | `{'SL': 38, 'TIME': 84, 'TP': 10}` |
| `fee_8bps` | N=132 sum=2693.8 med=2.4 T3R=2117.8 tail150=0 maxLoss=-48.0 | `{'SL': 38, 'TIME': 84, 'TP': 10}` |
| `fee_10bps` | N=132 sum=2429.8 med=0.4 T3R=1859.8 tail150=0 maxLoss=-50.0 | `{'SL': 38, 'TIME': 84, 'TP': 10}` |

## 7. Execution Realism

| Model | Fill rate | Summary | Fill kinds | Exits | No-fill counterfactual |
| --- | ---: | --- | --- | --- | --- |
| `taker_immediate` | 1.0 | N=132 sum=3089.8 med=5.4 T3R=2504.8 tail150=0 maxLoss=-45.0 | `{}` | `{'SL': 38, 'TIME': 84, 'TP': 10}` | N=None sum=None med=None T3R=None tail150=None maxLoss=None |
| `passive_only_off5_wait15s` | 0.28 | N=37 sum=799.5 med=14.2 T3R=214.5 tail150=0 maxLoss=-45.0 | `{'NO_FILL': 95, 'PASSIVE_FILL': 37}` | `{'TIME': 21, 'SL': 12, 'TP': 4}` | N=95 sum=2436.4 med=4.3 T3R=1851.4 tail150=0 maxLoss=-45.0 |
| `passive_then_taker_off5_wait15s` | 1.0 | N=132 sum=2852.5 med=2.1 T3R=2267.5 tail150=0 maxLoss=-45.0 | `{'FALLBACK_TAKER': 95, 'PASSIVE_FILL': 37}` | `{'SL': 40, 'TIME': 83, 'TP': 9}` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |
| `passive_only_off5_wait30s` | 0.409 | N=54 sum=1413.9 med=14.2 T3R=828.9 tail150=0 maxLoss=-45.0 | `{'NO_FILL': 78, 'PASSIVE_FILL': 54}` | `{'TIME': 32, 'SL': 16, 'TP': 6}` | N=78 sum=1850.1 med=0.3 T3R=1265.1 tail150=0 maxLoss=-45.0 |
| `passive_then_taker_off5_wait30s` | 1.0 | N=132 sum=2581.8 med=-2.6 T3R=1996.8 tail150=0 maxLoss=-45.0 | `{'FALLBACK_TAKER': 78, 'PASSIVE_FILL': 54}` | `{'SL': 41, 'TIME': 82, 'TP': 9}` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |
| `passive_only_off5_wait60s` | 0.561 | N=74 sum=1223.8 med=2.1 T3R=638.8 tail150=0 maxLoss=-45.0 | `{'PASSIVE_FILL': 74, 'NO_FILL': 58}` | `{'SL': 25, 'TIME': 43, 'TP': 6}` | N=58 sum=2079.8 med=8.6 T3R=1494.8 tail150=0 maxLoss=-45.0 |
| `passive_then_taker_off5_wait60s` | 1.0 | N=132 sum=2364.1 med=-0.8 T3R=1779.1 tail150=0 maxLoss=-45.0 | `{'PASSIVE_FILL': 74, 'FALLBACK_TAKER': 58}` | `{'SL': 44, 'TIME': 79, 'TP': 9}` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |
| `passive_only_off10_wait15s` | 0.091 | N=12 sum=489.9 med=10.4 T3R=-95.1 tail150=0 maxLoss=-45.0 | `{'NO_FILL': 120, 'PASSIVE_FILL': 12}` | `{'TIME': 6, 'SL': 3, 'TP': 3}` | N=120 sum=2649.5 med=5.4 T3R=2064.5 tail150=0 maxLoss=-45.0 |
| `passive_then_taker_off10_wait15s` | 1.0 | N=132 sum=2822.9 med=2.1 T3R=2237.9 tail150=0 maxLoss=-45.0 | `{'FALLBACK_TAKER': 120, 'PASSIVE_FILL': 12}` | `{'SL': 40, 'TIME': 83, 'TP': 9}` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |
| `passive_only_off10_wait30s` | 0.182 | N=24 sum=786.1 med=-1.1 T3R=201.1 tail150=0 maxLoss=-45.0 | `{'NO_FILL': 108, 'PASSIVE_FILL': 24}` | `{'TIME': 13, 'SL': 7, 'TP': 4}` | N=108 sum=2398.3 med=6.6 T3R=1813.3 tail150=0 maxLoss=-45.0 |
| `passive_then_taker_off10_wait30s` | 1.0 | N=132 sum=2422.0 med=-4.4 T3R=1837.0 tail150=0 maxLoss=-45.0 | `{'FALLBACK_TAKER': 108, 'PASSIVE_FILL': 24}` | `{'SL': 42, 'TIME': 81, 'TP': 9}` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |
| `passive_only_off10_wait60s` | 0.311 | N=41 sum=642.7 med=-6.7 T3R=57.7 tail150=0 maxLoss=-45.0 | `{'NO_FILL': 91, 'PASSIVE_FILL': 41}` | `{'SL': 17, 'TIME': 20, 'TP': 4}` | N=91 sum=2657.4 med=10.0 T3R=2072.4 tail150=0 maxLoss=-45.0 |
| `passive_then_taker_off10_wait60s` | 1.0 | N=132 sum=2136.2 med=-4.3 T3R=1551.2 tail150=0 maxLoss=-45.0 | `{'FALLBACK_TAKER': 91, 'PASSIVE_FILL': 41}` | `{'SL': 45, 'TIME': 78, 'TP': 9}` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |
| `passive_only_off20_wait15s` | 0.008 | N=1 sum=-45.0 med=-45.0 T3R=-45.0 tail150=0 maxLoss=-45.0 | `{'NO_FILL': 131, 'PASSIVE_FILL': 1}` | `{'SL': 1}` | N=131 sum=3134.8 med=6.6 T3R=2549.8 tail150=0 maxLoss=-45.0 |
| `passive_then_taker_off20_wait15s` | 1.0 | N=132 sum=2845.3 med=2.8 T3R=2260.3 tail150=0 maxLoss=-45.0 | `{'FALLBACK_TAKER': 131, 'PASSIVE_FILL': 1}` | `{'SL': 40, 'TIME': 83, 'TP': 9}` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |
| `passive_only_off20_wait30s` | 0.015 | N=2 sum=-40.4 med=-20.2 T3R=-40.4 tail150=0 maxLoss=-45.0 | `{'NO_FILL': 130, 'PASSIVE_FILL': 2}` | `{'TIME': 1, 'SL': 1}` | N=130 sum=3150.4 med=6.6 T3R=2565.4 tail150=0 maxLoss=-45.0 |
| `passive_then_taker_off20_wait30s` | 1.0 | N=132 sum=2456.3 med=-2.5 T3R=1871.3 tail150=0 maxLoss=-45.0 | `{'FALLBACK_TAKER': 130, 'PASSIVE_FILL': 2}` | `{'SL': 42, 'TIME': 81, 'TP': 9}` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |
| `passive_only_off20_wait60s` | 0.061 | N=8 sum=226.3 med=-20.2 T3R=-175.4 tail150=0 maxLoss=-45.0 | `{'NO_FILL': 124, 'PASSIVE_FILL': 8}` | `{'TIME': 2, 'SL': 4, 'TP': 2}` | N=124 sum=2896.8 med=6.8 T3R=2311.8 tail150=0 maxLoss=-45.0 |
| `passive_then_taker_off20_wait60s` | 1.0 | N=132 sum=2025.6 med=-3.1 T3R=1440.6 tail150=0 maxLoss=-45.0 | `{'FALLBACK_TAKER': 124, 'PASSIVE_FILL': 8}` | `{'SL': 45, 'TIME': 78, 'TP': 9}` | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |

## 8. Big Winner / Big Loser

Summary: N=132 sum=3089.8 med=5.4 T3R=2504.8 tail150=0 maxLoss=-45.0
Top3 removed sum: `2504.8`; top10 removed sum: `1139.8`
SL summary: N=38 sum=-1710.0 med=-45.0 T3R=-1575.0 tail150=0 maxLoss=-45.0

Worst 10:
- `{'event_id': 'ETHUSDT_SELL_50000_1781119401712', 'signal_utc': '2026-06-10T19:23:21.712000+00:00', 'value_bps': -45.0, 'exit': 'SL', 'fold': 4, 'vdepth': 14.9, 'btc4h': -97.0, 'chain_thresholds': 3}`
- `{'event_id': 'ETHUSDT_SELL_100000_1781119418078', 'signal_utc': '2026-06-10T19:23:38.078000+00:00', 'value_bps': -45.0, 'exit': 'SL', 'fold': 4, 'vdepth': 15.2, 'btc4h': -93.7, 'chain_thresholds': 3}`
- `{'event_id': 'ETHUSDT_SELL_200000_1781119418078', 'signal_utc': '2026-06-10T19:23:38.078000+00:00', 'value_bps': -45.0, 'exit': 'SL', 'fold': 4, 'vdepth': 15.2, 'btc4h': -93.7, 'chain_thresholds': 3}`
- `{'event_id': 'ETHUSDT_SELL_200000_1781190088276', 'signal_utc': '2026-06-11T15:01:28.276000+00:00', 'value_bps': -45.0, 'exit': 'SL', 'fold': 4, 'vdepth': 27.3, 'btc4h': -93.7, 'chain_thresholds': 3}`
- `{'event_id': 'ETHUSDT_SELL_50000_1781562702507', 'signal_utc': '2026-06-15T22:31:42.507000+00:00', 'value_bps': -45.0, 'exit': 'SL', 'fold': 4, 'vdepth': 14.0, 'btc4h': -92.6, 'chain_thresholds': 3}`
- `{'event_id': 'ETHUSDT_SELL_50000_1781623376098', 'signal_utc': '2026-06-16T15:22:56.098000+00:00', 'value_bps': -45.0, 'exit': 'SL', 'fold': 4, 'vdepth': 20.1, 'btc4h': -120.7, 'chain_thresholds': 3}`
- `{'event_id': 'ETHUSDT_SELL_100000_1781623376098', 'signal_utc': '2026-06-16T15:22:56.098000+00:00', 'value_bps': -45.0, 'exit': 'SL', 'fold': 4, 'vdepth': 20.1, 'btc4h': -120.7, 'chain_thresholds': 3}`
- `{'event_id': 'ETHUSDT_SELL_200000_1781623376098', 'signal_utc': '2026-06-16T15:22:56.098000+00:00', 'value_bps': -45.0, 'exit': 'SL', 'fold': 4, 'vdepth': 20.1, 'btc4h': -120.7, 'chain_thresholds': 3}`
- `{'event_id': 'ETHUSDT_SELL_200000_1781798251634', 'signal_utc': '2026-06-18T15:57:31.634000+00:00', 'value_bps': -45.0, 'exit': 'SL', 'fold': 4, 'vdepth': 29.2, 'btc4h': -255.2, 'chain_thresholds': 3}`
- `{'event_id': 'ETHUSDT_SELL_50000_1782307339365', 'signal_utc': '2026-06-24T13:22:19.365000+00:00', 'value_bps': -45.0, 'exit': 'SL', 'fold': 5, 'vdepth': 12.0, 'btc4h': -111.6, 'chain_thresholds': 3}`
Best 10:
- `{'event_id': 'ETHUSDT_SELL_200000_1781797000262', 'signal_utc': '2026-06-18T15:36:40.262000+00:00', 'value_bps': 195.0, 'exit': 'TP', 'fold': 4, 'vdepth': 13.7, 'btc4h': -141.6, 'chain_thresholds': 3}`
- `{'event_id': 'ETHUSDT_SELL_50000_1782202432090', 'signal_utc': '2026-06-23T08:13:52.090000+00:00', 'value_bps': 195.0, 'exit': 'TP', 'fold': 5, 'vdepth': 28.8, 'btc4h': -202.5, 'chain_thresholds': 3}`
- `{'event_id': 'ETHUSDT_SELL_50000_1782319123478', 'signal_utc': '2026-06-24T16:38:43.478000+00:00', 'value_bps': 195.0, 'exit': 'TP', 'fold': 5, 'vdepth': 16.6, 'btc4h': -406.7, 'chain_thresholds': 3}`
- `{'event_id': 'ETHUSDT_SELL_100000_1782319424366', 'signal_utc': '2026-06-24T16:43:44.366000+00:00', 'value_bps': 195.0, 'exit': 'TP', 'fold': 5, 'vdepth': 20.2, 'btc4h': -425.3, 'chain_thresholds': 3}`
- `{'event_id': 'ETHUSDT_SELL_50000_1782394272242', 'signal_utc': '2026-06-25T13:31:12.242000+00:00', 'value_bps': 195.0, 'exit': 'TP', 'fold': 5, 'vdepth': 24.2, 'btc4h': -100.5, 'chain_thresholds': 3}`
- `{'event_id': 'ETHUSDT_SELL_100000_1782394284495', 'signal_utc': '2026-06-25T13:31:24.495000+00:00', 'value_bps': 195.0, 'exit': 'TP', 'fold': 5, 'vdepth': 39.4, 'btc4h': -109.2, 'chain_thresholds': 3}`
- `{'event_id': 'ETHUSDT_SELL_200000_1782394305283', 'signal_utc': '2026-06-25T13:31:45.283000+00:00', 'value_bps': 195.0, 'exit': 'TP', 'fold': 5, 'vdepth': 45.6, 'btc4h': -112.0, 'chain_thresholds': 3}`
- `{'event_id': 'ETHUSDT_SELL_50000_1782395474187', 'signal_utc': '2026-06-25T13:51:14.187000+00:00', 'value_bps': 195.0, 'exit': 'TP', 'fold': 5, 'vdepth': 41.9, 'btc4h': -396.2, 'chain_thresholds': 3}`
- `{'event_id': 'ETHUSDT_SELL_100000_1782395476217', 'signal_utc': '2026-06-25T13:51:16.217000+00:00', 'value_bps': 195.0, 'exit': 'TP', 'fold': 5, 'vdepth': 43.1, 'btc4h': -401.2, 'chain_thresholds': 3}`
- `{'event_id': 'ETHUSDT_SELL_200000_1782395476217', 'signal_utc': '2026-06-25T13:51:16.217000+00:00', 'value_bps': 195.0, 'exit': 'TP', 'fold': 5, 'vdepth': 43.1, 'btc4h': -401.2, 'chain_thresholds': 3}`

## 9. Regime Concentration

Summary: N=132 sum=3089.8 med=5.4 T3R=2504.8 tail150=0 maxLoss=-45.0
Top abs date sum share: `0.389`
Warning: `candidate appears only in folds with matching stress/BTC regime; needs forward OOS before live`

Top dates:
- `{'date': '2026-06-25', 'n': 17, 'sum_bps': 1201.5, 't3r_bps': 616.5}`
- `{'date': '2026-06-23', 'n': 6, 'sum_bps': 958.4, 't3r_bps': 421.6}`
- `{'date': '2026-06-24', 'n': 34, 'sum_bps': 681.8, 't3r_bps': 116.8}`
- `{'date': '2026-06-18', 'n': 16, 'sum_bps': 356.4, 't3r_bps': -9.4}`
- `{'date': '2026-06-17', 'n': 12, 'sum_bps': 236.9, 't3r_bps': 20.4}`
- `{'date': '2026-06-26', 'n': 12, 'sum_bps': -213.7, 't3r_bps': -294.5}`
- `{'date': '2026-06-16', 'n': 7, 'sum_bps': 117.6, 't3r_bps': -120.7}`
- `{'date': '2026-06-29', 'n': 3, 'sum_bps': -104.4, 't3r_bps': -104.4}`
- `{'date': '2026-06-15', 'n': 3, 'sum_bps': 78.2, 't3r_bps': 78.2}`
- `{'date': '2026-06-21', 'n': 6, 'sum_bps': -72.1, 't3r_bps': -42.6}`
