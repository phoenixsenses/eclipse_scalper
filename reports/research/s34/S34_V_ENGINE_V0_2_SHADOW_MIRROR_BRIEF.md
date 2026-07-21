# S34 V Engine v0.2 Shadow Mirror Brief

Generated: `2026-07-21T08:12:48.394675+00:00`

Protocol: `S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID`

Status: `EXPLORATORY_FROZEN` / `OBSERVE_ONLY_NO_ORDER`. This is paper/shadow mirror only; it never sends orders.

## Ledger

- rows total: `17`
- rows added this run: `0`
- observation counts: `{'CLOSED': 13, 'DATA_INCOMPLETE': 4}`
- sim counts: `{'FILLED': 13, 'NO_EXIT_BOOK': 4}`

## Performance Labels

- overall: signals `17`, closed fills `13`, fill rate `0.765`, N=13 sum=1089.6 med=46.3 T3R=410.6
- recent 60d: signals `13`, closed fills `9`, fill rate `0.692`, N=9 sum=942.1 med=79.9 T3R=263.1
- no-fill counterfactual: closed no-fill `0`, N=0 sum=0.0 med=None T3R=0.0
- kill check: `not triggered` (60-day forward T3R < 0 after at least 3 closed fills)

## Latest Observations

| UTC | Status | Sim | Leg | V-depth | Bid depth | Fill delay | Net | CF mark net |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21T23:33:42.690000+00:00 | CLOSED | FILLED | replacement | 29.8 | 136804.1 | 645.0 | 227.0 | 225.9 |
| 2026-06-26T02:48:30.475000+00:00 | CLOSED | FILLED | initial | 39.6 | 135918.4 | 290.0 | 149.9 | 144.7 |
| 2026-06-26T10:39:33.530000+00:00 | CLOSED | FILLED | replacement | 34.4 | 293221.2 | 4285.0 | 17.2 | -12.7 |
| 2026-06-26T13:18:54.877000+00:00 | CLOSED | FILLED | initial | 36.3 | 829882.1 | 37.0 | 299.7 | 266.8 |
| 2026-06-30T13:32:16.371000+00:00 | CLOSED | FILLED | initial | 33.2 | 193387.5 | 129.0 | 79.9 | 55.6 |
| 2026-07-04T23:14:40.186000+00:00 | CLOSED | FILLED | replacement | 37.5 | 267545.0 | 4042.0 | -71.9 | -23.3 |
| 2026-07-19T18:04:37.388000+00:00 | DATA_INCOMPLETE | NO_EXIT_BOOK | initial | 28.3 | 232844.2 | None | None | None |
| 2026-07-19T18:04:37.388000+00:00 | DATA_INCOMPLETE | NO_EXIT_BOOK | initial | 28.3 | 232844.2 | None | None | None |
| 2026-07-20T06:33:26.488000+00:00 | DATA_INCOMPLETE | NO_EXIT_BOOK | initial | 32.1 | 167366.3 | None | None | None |
| 2026-07-20T06:33:26.488000+00:00 | DATA_INCOMPLETE | NO_EXIT_BOOK | initial | 32.1 | 167366.3 | None | None | None |
