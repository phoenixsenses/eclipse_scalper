# S34 Reversal Backtest (fade large liquidation cascades, fixed horizon)

Generated: `2026-06-28T22:19:30.893294+00:00`  |  SOLUSDT 200K, fee 3.05bps/side, book staleness 10s, holdout 0.3

Entry fades the cascade at the threshold cross (SHORT after BUY-liq, LONG after SELL-liq); exit at fixed horizon. Spread paid via bid/ask fills; net = gross - 2*fee. `all` = every event; `sequential` = single-unit capital, no overlap.

Total fade events: 66

| Horizon | Filled | all cal med | all cal win | all hold med | all hold win | all hold sum | seq N | seq hold med | seq hold sum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4h | 53 | -33.2 | 43.2 | 77.5 | 81.2 | 1335.2 | 32 | 90.4 | 934.9 |

Read: a credible edge wants positive median on BOTH cal and hold in the `all` view, and a positive `sequential` holdout sum (realistic single-capital P&L).
