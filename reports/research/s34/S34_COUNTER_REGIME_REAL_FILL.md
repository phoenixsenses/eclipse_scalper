# S34 Counter-Regime Real-Fill Test

- generated_at_utc: `2026-06-19T16:34:51.683136+00:00`
- route_id: `LONG_DELAY0_TP60`
- scope: ETH BUY liquidation, LONG delay0 TP60/SL40/BE30, day_trend_bps < 0 counter-regime candidates
- live runner/config changes: `none`

## Candidate Results
| candidate | total | real | no-fill | real median | real mean | real cum | real WR | test-half N | test-half median | test-half cum | fill penalty med |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 500K_daytrend_negative | 43 | 15 | 28 (65.12%) | -9.08 | 11.13 | 166.96 | 46.67% | 14 | -9.27 | 115.01 | 0.51 |
| 500K_neg_count_ge10 | 32 | 10 | 22 (68.75%) | 51.92 | 24.18 | 241.79 | 60.00% | 8 | 52.62 | 204.00 | 0.54 |
| 500K_neg_count_ge15 | 26 | 9 | 17 (65.38%) | 51.89 | 20.21 | 181.91 | 55.56% | 7 | 51.89 | 144.12 | 0.58 |
| 500K_neg_count_ge20 | 18 | 6 | 12 (66.67%) | 52.65 | 43.19 | 259.13 | 83.33% | 5 | 53.35 | 207.17 | -0.20 |
| 500K_neg_count_ge22 | 15 | 4 | 11 (73.33%) | 53.54 | 53.81 | 215.24 | 100.00% | 4 | 53.54 | 215.24 | -0.17 |
| 500K_neg_stretched | 19 | 7 | 12 (63.16%) | 53.35 | 54.47 | 381.29 | 100.00% | 4 | 54.24 | 217.57 | -0.91 |
| 500K_neg_stretched_count_ge15 | 16 | 5 | 11 (68.75%) | 53.35 | 53.72 | 268.59 | 100.00% | 2 | 54.81 | 109.62 | -0.91 |
| 500K_neg_stretched_count_ge22 | 12 | 4 | 8 (66.67%) | 53.54 | 53.81 | 215.24 | 100.00% | 3 | 55.13 | 163.29 | -0.17 |

## Read

This is still research. A candidate surviving real-fill here is eligible for a separate exploratory paper rule only after explicit pre-registration; it does not change the current pre-reg S34 sample.
