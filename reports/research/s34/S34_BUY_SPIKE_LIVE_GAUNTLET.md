# S34 BUY Spike Live-Readiness Gauntlet

Generated: `2026-06-29T18:16:24.641637+00:00`
Scope: `{'symbol': 'ETHUSDT', 'days': 30, 'start_utc': '2026-05-30T18:16:11+00:00', 'end_utc': '2026-06-29T18:16:11+00:00', 'liq_rows': 11696, 'note': 'Research-only. No live executor/config/order logic touched.'}`

## 1. Knowable Running Threshold-Cross

| window | threshold | N | 15m fee-net sum | median | WR | T3R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10s | 487346.2 | 83 | -308.6 | -6.16 | 0.277 | -847.6 |
| 15s | 635643.8 | 80 | -349.7 | -8.25 | 0.263 | -878.9 |
| 30s | 982599.8 | 62 | -137.6 | -6.16 | 0.306 | -657.1 |
| 60s | 1349717.4 | 54 | 10.4 | -6.16 | 0.333 | -494.4 |

## 2. Second-Level Entry Delay, 60s Running Anchor

- 0s: N `54`, sum `10.4`, median `-6.16`, WR `0.333`, T3R `-494.4`
- 2s: N `54`, sum `47.8`, median `-6.16`, WR `0.296`, T3R `-460.6`
- 5s: N `54`, sum `44.7`, median `-6.16`, WR `0.315`, T3R `-463.8`
- 10s: N `54`, sum `-24.3`, median `-6.16`, WR `0.352`, T3R `-525.6`
- 30s: N `54`, sum `-133.1`, median `-6.16`, WR `0.352`, T3R `-620.8`
- 60s: N `54`, sum `31.4`, median `-6.16`, WR `0.333`, T3R `-436.2`

## 3. Fee / Slippage Stress, 60s Running Anchor

- fee 6.1bps: sum `10.4`, median `-6.16`, WR `0.333`, T3R `-494.4`
- fee 10.0bps: sum `-200.2`, median `-10.06`, WR `0.296`, T3R `-693.3`
- fee 15.0bps: sum `-470.2`, median `-15.06`, WR `0.296`, T3R `-948.3`
- fee 20.0bps: sum `-740.2`, median `-20.06`, WR `0.278`, T3R `-1203.3`
- fee 30.0bps: sum `-1280.2`, median `-30.06`, WR `0.222`, T3R `-1713.3`

## 4. Exit / Stop Robustness

- fixed_5m_fee6.1: sum `-91.0`, median `-10.35`, WR `0.407`, T3R `-578.3`, min `-85.12`
- fixed_10m_fee6.1: sum `-54.1`, median `-15.75`, WR `0.333`, T3R `-594.5`, min `-133.05`
- fixed_15m_fee6.1: sum `83.3`, median `-16.58`, WR `0.389`, T3R `-480.6`, min `-136.91`
- fixed_20m_fee6.1: sum `-24.1`, median `-21.9`, WR `0.389`, T3R `-590.9`, min `-125.43`
- fixed_30m_fee6.1: sum `-31.3`, median `-24.08`, WR `0.407`, T3R `-670.8`, min `-140.81`
- SL30_15m_fee6.1: sum `130.0`, median `-23.58`, WR `0.37`, T3R `-433.9`, min `-80.67`
- SL50_15m_fee6.1: sum `53.3`, median `-19.35`, WR `0.389`, T3R `-510.6`, min `-80.67`
- SL75_15m_fee6.1: sum `162.8`, median `-16.58`, WR `0.389`, T3R `-401.1`, min `-99.86`
- SL100_15m_fee6.1: sum `93.1`, median `-16.58`, WR `0.389`, T3R `-470.8`, min `-133.05`

## 5. Walk-Forward

- fold 1: N `10`, sum `-37.6`, median `-21.54`, T3R `-264.2`
- fold 2: N `11`, sum `109.8`, median `-15.63`, T3R `-162.3`
- fold 3: N `11`, sum `-9.0`, median `-8.38`, T3R `-325.6`
- fold 4: N `11`, sum `-134.6`, median `-7.43`, T3R `-225.0`
- fold 5: N `11`, sum `154.6`, median `-20.79`, T3R `-251.7`

## 6. Permutation Null

- `{'permutations': 300, 'real_max_sum': 10.4, 'null_p95': 201.6, 'null_p99': 367.5, 'p_right': 0.19}`

## 7. Regime Splits

- btc_1h_up: N `53`, sum `196.2`, median `-15.63`, WR `0.396`, T3R `-367.7`
- btc_1h_down: N `1`, sum `-112.9`, median `-112.88`, WR `0.0`, T3R `None`
- eth_pre15_up: N `54`, sum `83.3`, median `-16.58`, WR `0.389`, T3R `-480.6`
- eth_pre15_down: N `0`, sum `0.0`, median `None`, WR `None`, T3R `None`
- p99_running_notional: N `2`, sum `-4.1`, median `-2.04`, WR `0.5`, T3R `None`
- non_p99_running_notional: N `52`, sum `87.4`, median `-16.58`, WR `0.385`, T3R `-476.5`

## Verdict

RESEARCH_ONLY: pass=[]; fail=['60s running threshold fails fee-net/T3R', 'walk-forward is not uniformly positive', 'permutation-null does not clear p<=0.05', '10s delay does not robustly survive']