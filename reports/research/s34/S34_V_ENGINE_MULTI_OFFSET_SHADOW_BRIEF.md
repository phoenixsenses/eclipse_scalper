# S34 V Engine Multi-Offset Shadow Brief

Generated: `2026-06-28T19:54:04.437093+00:00`

Protocol: `S34_V_ENGINE_V0_1_ETH_SELL_MAKER_LONG_H2_O20_V28_40_P4D`

Research-only parallel shadow ledger. It compares maker offsets on the same eligible V Engine events; no order is authorized.

## Ledger

- rows total: `282`
- rows added this run: `282`
- status counts: `{'CLOSED': 157, 'DATA_INCOMPLETE': 125}`

## Offset Configs

| Rank | Config | Fill% | Filled | No-fill CF | Missed CF sum | Median fill delay |
| ---: | --- | ---: | --- | --- | ---: | ---: |
| 1 | `O20_C1` | 40.4 | N=19 sum=887.7 med=37.0 T3R=357.0 | N=6 sum=542.3 med=66.2 T3R=89.0 | 542.3 | 290.0 |
| 2 | `O20_C2` | 40.4 | N=19 sum=876.1 med=37.0 T3R=348.4 | N=7 sum=568.9 med=57.3 T3R=115.6 | 568.9 | 338.0 |
| 3 | `O15_C1` | 40.4 | N=19 sum=794.8 med=32.0 T3R=274.5 | N=6 sum=542.3 med=66.2 T3R=89.0 | 542.3 | 164.0 |
| 4 | `O15_C2` | 40.4 | N=19 sum=787.3 med=32.0 T3R=273.9 | N=6 sum=542.3 med=66.2 T3R=89.0 | 542.3 | 191.0 |
| 5 | `O25_C1` | 36.2 | N=17 sum=816.4 med=49.0 T3R=273.2 | N=11 sum=866.1 med=57.3 T3R=345.4 | 866.1 | 571.0 |
| 6 | `O25_C2` | 36.2 | N=17 sum=802.0 med=49.0 T3R=264.7 | N=11 sum=866.1 med=57.3 T3R=345.4 | 866.1 | 571.0 |

## Read

- Best current T3R-ranked config: `O20_C1` with N=19 sum=887.7 med=37.0 T3R=357.0.
- Treat this as execution observation, not a new frozen rule. The decision remains observe-only until new forward rows accumulate.

## Latest Rows

| UTC | Config | Status | Sim | Fill delay | Net | CF mark net |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 2026-06-26T10:39:33.530000+00:00 | `O15_C2` | CLOSED | FILLED | 4285.0 | 27.2 | -12.7 |
| 2026-06-26T10:39:33.530000+00:00 | `O25_C1` | CLOSED | FILLED | 4444.0 | 64.5 | -12.7 |
| 2026-06-26T10:39:33.530000+00:00 | `O20_C1` | CLOSED | FILLED | 4285.0 | 32.3 | -12.7 |
| 2026-06-26T10:39:33.530000+00:00 | `O25_C2` | CLOSED | FILLED | 4444.0 | 64.5 | -12.7 |
| 2026-06-26T10:39:33.530000+00:00 | `O20_C2` | CLOSED | FILLED | 4285.0 | 32.3 | -12.7 |
| 2026-06-26T10:39:33.530000+00:00 | `O15_C1` | CLOSED | FILLED | 4285.0 | 27.2 | -12.7 |
| 2026-06-26T13:18:54.877000+00:00 | `O15_C2` | CLOSED | FILLED | 36.0 | 295.2 | 266.8 |
| 2026-06-26T13:18:54.877000+00:00 | `O25_C1` | CLOSED | FILLED | 39.0 | 304.8 | 266.8 |
| 2026-06-26T13:18:54.877000+00:00 | `O20_C1` | CLOSED | FILLED | 37.0 | 299.7 | 266.8 |
| 2026-06-26T13:18:54.877000+00:00 | `O15_C1` | CLOSED | FILLED | 36.0 | 295.2 | 266.8 |
| 2026-06-26T13:18:54.877000+00:00 | `O20_C2` | CLOSED | FILLED | 37.0 | 299.7 | 266.8 |
| 2026-06-26T13:18:54.877000+00:00 | `O25_C2` | CLOSED | FILLED | 39.0 | 304.8 | 266.8 |
| 2026-06-28T16:34:14.359000+00:00 | `O20_C1` | CLOSED | FILLED | 652.0 | 18.4 | -8.2 |
| 2026-06-28T16:34:14.359000+00:00 | `O25_C2` | CLOSED | FILLED | 3663.0 | 26.6 | -8.2 |
| 2026-06-28T16:34:14.359000+00:00 | `O15_C1` | CLOSED | FILLED | 22.0 | 6.3 | -8.2 |
| 2026-06-28T16:34:14.359000+00:00 | `O20_C2` | CLOSED | FILLED | 652.0 | 18.4 | -8.2 |
| 2026-06-28T16:34:14.359000+00:00 | `O15_C2` | CLOSED | FILLED | 22.0 | 6.3 | -8.2 |
| 2026-06-28T16:34:14.359000+00:00 | `O25_C1` | CLOSED | FILLED | 3663.0 | 26.6 | -8.2 |
