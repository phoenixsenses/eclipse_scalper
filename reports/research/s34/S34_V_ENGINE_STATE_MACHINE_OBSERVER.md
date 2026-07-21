# S34 V Engine State Machine Observer

Generated: `2026-06-28T19:29:22.639841+00:00`

Protocol: `S34_V_ENGINE_V0_1_ETH_SELL_MAKER_LONG_H2_O20_V28_40_P4D`

Observation only. This treats the V Engine as Cascade -> Capitulation -> Reclaim -> Acceptance.

## Counts

- ledger_rows: `47`
- closed_state_rows: `26`
- closed_filled_rows: `19`
- closed_no_fill_rows: `7`
- acceptance_rows: `21`
- acceptance_rate: `0.808`

## Summaries

- original filled: N=19 sum=876.1 med=37.0 T3R=348.4
- delayed entry after acceptance to original exit: N=14 sum=99.1 med=-5.9 T3R=-268.1
- wait cost vs original maker limit: N=14 sum=757.6 med=46.3 T3R=491.0
- opportunity cost vs original filled trade: N=14 sum=772.7 med=47.5 T3R=504.1
- no-fill counterfactual: N=7 sum=568.9 med=57.3 T3R=115.6

## Acceptance Timing

| Bucket | N | Delayed outcome | Wait cost | Opportunity cost |
| --- | ---: | --- | --- | --- |
| `confirm_15_30m` | 4 | N=4 sum=58.5 med=-9.6 T3R=-27.5 | N=4 sum=180.9 med=43.6 T3R=40.2 | N=4 sum=185.5 med=44.6 T3R=41.7 |
| `confirm_30_60m` | 5 | N=4 sum=-134.7 med=-35.1 T3R=-70.7 | N=4 sum=247.4 med=52.8 T3R=47.3 | N=4 sum=250.9 med=53.7 T3R=48.4 |
| `confirm_5_15m` | 12 | N=6 sum=175.3 med=-3.8 T3R=-84.7 | N=6 sum=329.3 med=44.0 T3R=111.9 | N=6 sum=336.3 med=45.7 T3R=116.3 |
| `no_confirm` | 5 | N=0 sum=0.0 med=None T3R=0.0 | N=0 sum=0.0 med=None T3R=0.0 | N=0 sum=0.0 med=None T3R=0.0 |

## State

| State | N | Delayed outcome | Wait cost | Opportunity cost |
| --- | ---: | --- | --- | --- |
| `acceptance` | 21 | N=14 sum=99.1 med=-5.9 T3R=-268.1 | N=14 sum=757.6 med=46.3 T3R=491.0 | N=14 sum=772.7 med=47.5 T3R=504.1 |
| `reclaim` | 5 | N=0 sum=0.0 med=None T3R=0.0 | N=0 sum=0.0 med=None T3R=0.0 | N=0 sum=0.0 med=None T3R=0.0 |

## Latest Rows

| UTC | Sim | State | Acceptance delay | Wait cost | Original | Delayed | Opp cost |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2026-06-17T01:17:01.753000+00:00 | FILLED | acceptance | 1800.0 | 53.5 | 41.7 | -12.9 | 54.6 |
| 2026-06-18T15:57:31.634000+00:00 | FILLED | acceptance | 900.0 | 72.8 | 71.9 | -2.0 | 73.9 |
| 2026-06-20T14:08:51.159000+00:00 | NO_MAKER_FILL | acceptance | 900.0 | None | None | None | None |
| 2026-06-21T11:18:26.629000+00:00 | FILLED | acceptance | 1800.0 | 42.3 | 37.0 | -6.3 | 43.3 |
| 2026-06-21T23:33:42.690000+00:00 | NO_MAKER_FILL | acceptance | 900.0 | None | None | None | None |
| 2026-06-23T07:59:44.477000+00:00 | FILLED | reclaim | None | None | -146.0 | None | None |
| 2026-06-25T15:03:23.104000+00:00 | FILLED | acceptance | 2700.0 | 51.7 | 59.1 | 6.2 | 52.9 |
| 2026-06-25T16:32:03.169000+00:00 | FILLED | acceptance | 2700.0 | 53.8 | -16.2 | -70.7 | 54.5 |
| 2026-06-26T02:48:30.475000+00:00 | FILLED | acceptance | 1800.0 | 40.2 | 146.9 | 105.2 | 41.7 |
| 2026-06-26T10:39:33.530000+00:00 | FILLED | acceptance | 900.0 | 36.8 | 32.3 | -5.6 | 37.9 |
| 2026-06-26T13:18:54.877000+00:00 | FILLED | acceptance | 900.0 | 42.7 | 299.7 | 254.8 | 44.9 |
| 2026-06-28T16:34:14.359000+00:00 | FILLED | acceptance | 1800.0 | 44.9 | 18.4 | -27.5 | 45.9 |
