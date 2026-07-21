# S34 BUY Spike Live-Readiness Gauntlet

Generated: `2026-06-29T18:16:04.499976+00:00`
Scope: `{'symbol': 'ETHUSDT', 'days': 7, 'start_utc': '2026-06-22T18:16:01+00:00', 'end_utc': '2026-06-29T18:16:01+00:00', 'liq_rows': 3624, 'note': 'Research-only. No live executor/config/order logic touched.'}`

## 1. Knowable Running Threshold-Cross

| window | threshold | N | 15m fee-net sum | median | WR | T3R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10s | 380143.7 | 32 | -32.2 | -14.53 | 0.375 | -513.4 |
| 15s | 504526.4 | 28 | 53.5 | -14.53 | 0.357 | -415.4 |
| 30s | 775481.4 | 22 | 25.9 | -17.84 | 0.318 | -426.1 |
| 60s | 888646.1 | 21 | 87.7 | -15.57 | 0.381 | -338.7 |

## 2. Second-Level Entry Delay, 60s Running Anchor

- 0s: N `21`, sum `87.7`, median `-15.57`, WR `0.381`, T3R `-338.7`
- 2s: N `21`, sum `98.5`, median `-12.71`, WR `0.381`, T3R `-331.0`
- 5s: N `21`, sum `145.4`, median `-7.72`, WR `0.429`, T3R `-284.3`
- 10s: N `21`, sum `119.8`, median `-10.84`, WR `0.429`, T3R `-297.1`
- 30s: N `21`, sum `148.0`, median `-9.88`, WR `0.429`, T3R `-238.7`
- 60s: N `21`, sum `196.6`, median `-3.96`, WR `0.476`, T3R `-97.1`

## 3. Fee / Slippage Stress, 60s Running Anchor

- fee 6.1bps: sum `87.7`, median `-15.57`, WR `0.381`, T3R `-338.7`
- fee 10.0bps: sum `5.8`, median `-19.47`, WR `0.381`, T3R `-408.9`
- fee 15.0bps: sum `-99.2`, median `-24.47`, WR `0.381`, T3R `-498.9`
- fee 20.0bps: sum `-204.2`, median `-29.47`, WR `0.333`, T3R `-588.9`
- fee 30.0bps: sum `-414.2`, median `-39.47`, WR `0.286`, T3R `-768.9`

## 4. Exit / Stop Robustness

- fixed_5m_fee6.1: sum `-94.1`, median `-11.94`, WR `0.381`, T3R `-365.9`, min `-72.75`
- fixed_10m_fee6.1: sum `-20.5`, median `-9.73`, WR `0.238`, T3R `-376.3`, min `-94.32`
- fixed_15m_fee6.1: sum `134.9`, median `-14.59`, WR `0.381`, T3R `-300.9`, min `-95.78`
- fixed_20m_fee6.1: sum `180.7`, median `-19.74`, WR `0.381`, T3R `-238.4`, min `-71.35`
- fixed_30m_fee6.1: sum `143.7`, median `10.78`, WR `0.524`, T3R `-312.7`, min `-126.58`
- SL30_15m_fee6.1: sum `55.4`, median `-22.47`, WR `0.333`, T3R `-380.4`, min `-59.38`
- SL50_15m_fee6.1: sum `145.7`, median `-14.59`, WR `0.381`, T3R `-290.1`, min `-67.07`
- SL75_15m_fee6.1: sum `147.6`, median `-14.59`, WR `0.381`, T3R `-288.2`, min `-83.03`
- SL100_15m_fee6.1: sum `134.9`, median `-14.59`, WR `0.381`, T3R `-300.9`, min `-95.78`

## 5. Walk-Forward

- fold 1: N `4`, sum `11.7`, median `9.07`, T3R `-36.9`
- fold 2: N `4`, sum `-49.4`, median `-15.87`, T3R `-65.2`
- fold 3: N `4`, sum `-200.3`, median `-43.48`, T3R `-95.8`
- fold 4: N `4`, sum `77.4`, median `8.04`, T3R `-14.6`
- fold 5: N `5`, sum `295.4`, median `-14.59`, T3R `-49.8`

## 6. Permutation Null

- `{'permutations': 30, 'real_max_sum': 87.7, 'null_p95': 93.4, 'null_p99': 177.7, 'p_right': 0.0667}`

## 7. Regime Splits

- btc_1h_up: N `19`, sum `200.2`, median `-14.59`, WR `0.368`, T3R `-235.6`
- btc_1h_down: N `2`, sum `-65.3`, median `-32.64`, WR `0.5`, T3R `None`
- eth_pre15_up: N `21`, sum `134.9`, median `-14.59`, WR `0.381`, T3R `-300.9`
- eth_pre15_down: N `0`, sum `0.0`, median `None`, WR `None`, T3R `None`
- p99_running_notional: N `1`, sum `-65.2`, median `-65.2`, WR `0.0`, T3R `None`
- non_p99_running_notional: N `20`, sum `200.1`, median `-12.5`, WR `0.4`, T3R `-235.7`

## Verdict

RESEARCH_ONLY: pass=[]; fail=['60s running threshold fails fee-net/T3R', 'walk-forward is not uniformly positive', 'permutation-null does not clear p<=0.05', '10s delay does not robustly survive']