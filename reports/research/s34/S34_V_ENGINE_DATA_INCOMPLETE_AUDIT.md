# S34 V Engine Data-Incomplete Audit

Generated: `2026-06-28T19:53:08.904870+00:00`

Protocol: `S34_V_ENGINE_V0_1_ETH_SELL_MAKER_LONG_H2_O20_V28_40_P4D`

This audits observation rows that could not be closed because the simulated exit could not be priced from `book_ticker`.

## Summary

- ledger rows: `47`
- data incomplete rows: `21`
- max allowed book staleness: `10s`
- reason counts: `{'exit_before_book_history': 19, 'stale_exit_book_gap': 2}`
- sim status counts: `{'NO_EXIT_BOOK': 21}`

## Reasons

| Reason | N | First signal | Last signal | Staleness sec | Next book gap sec |
| --- | ---: | --- | --- | --- | --- |
| `exit_before_book_history` | 19 | 2026-02-16T13:38:46.298000+00:00 | 2026-03-31T07:18:20.133000+00:00 | {'n': 0, 'min': None, 'median': None, 'max': None} | {'n': 19, 'min': 973394.0, 'median': 3324871.0, 'max': 4670973.0} |
| `stale_exit_book_gap` | 2 | 2026-06-08T04:06:15.767000+00:00 | 2026-06-10T02:17:13.158000+00:00 | {'n': 2, 'min': 224409.1, 'median': 390703.1, 'max': 390703.1} | {'n': 2, 'min': 102070.0, 'median': 268364.0, 'max': 268364.0} |

## Incomplete Rows

| Signal UTC | Sim | Reason | Fill UTC | Expected exit | Nearest book | Stale sec | Next gap sec |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| 2026-02-16T13:38:46.298000+00:00 | NO_EXIT_BOOK | `exit_before_book_history` | 2026-02-16T13:39:09+00:00 | 2026-02-16T15:39:09+00:00 | None | None | 4670973.0 |
| 2026-02-19T15:11:27.885000+00:00 | NO_EXIT_BOOK | `exit_before_book_history` | 2026-02-19T15:12:39+00:00 | 2026-02-19T17:12:39+00:00 | None | None | 4406163.0 |
| 2026-02-23T14:41:42.510000+00:00 | NO_EXIT_BOOK | `exit_before_book_history` | 2026-02-23T15:52:41.005000+00:00 | 2026-02-23T17:52:41.005000+00:00 | None | None | 4058161.0 |
| 2026-02-23T15:52:41.609000+00:00 | NO_EXIT_BOOK | `exit_before_book_history` | 2026-02-23T16:03:29.005000+00:00 | 2026-02-23T18:03:29.005000+00:00 | None | None | 4057513.0 |
| 2026-02-23T17:34:42.615000+00:00 | NO_EXIT_BOOK | `exit_before_book_history` | 2026-02-23T17:36:15.001000+00:00 | 2026-02-23T19:36:15.001000+00:00 | None | None | 4051947.0 |
| 2026-02-27T14:47:34.111000+00:00 | NO_EXIT_BOOK | `exit_before_book_history` | 2026-02-27T16:05:21+00:00 | 2026-02-27T18:05:21+00:00 | None | None | 3711801.0 |
| 2026-03-01T19:44:17.120000+00:00 | NO_EXIT_BOOK | `exit_before_book_history` | 2026-03-01T20:07:28.008000+00:00 | 2026-03-01T22:07:28.008000+00:00 | None | None | 3524474.0 |
| 2026-03-01T20:11:27.271000+00:00 | NO_EXIT_BOOK | `exit_before_book_history` | 2026-03-01T20:13:23+00:00 | 2026-03-01T22:13:23+00:00 | None | None | 3524119.0 |
| 2026-03-02T07:09:35.615000+00:00 | NO_EXIT_BOOK | `exit_before_book_history` | 2026-03-02T08:12:29.001000+00:00 | 2026-03-02T10:12:29.001000+00:00 | None | None | 3480973.0 |
| 2026-03-04T03:33:45.657000+00:00 | NO_EXIT_BOOK | `exit_before_book_history` | 2026-03-04T03:34:11.004000+00:00 | 2026-03-04T05:34:11.004000+00:00 | None | None | 3324871.0 |
| 2026-03-04T22:44:54.668000+00:00 | NO_EXIT_BOOK | `exit_before_book_history` | 2026-03-04T22:45:46+00:00 | 2026-03-05T00:45:46+00:00 | None | None | 3255776.0 |
| 2026-03-06T14:36:19.070000+00:00 | NO_EXIT_BOOK | `exit_before_book_history` | 2026-03-06T14:37:56.001000+00:00 | 2026-03-06T16:37:56.001000+00:00 | None | None | 3112246.0 |
| 2026-03-06T15:51:07.139000+00:00 | NO_EXIT_BOOK | `exit_before_book_history` | 2026-03-06T15:51:27+00:00 | 2026-03-06T17:51:27+00:00 | None | None | 3107835.0 |
| 2026-03-17T04:22:13.713000+00:00 | NO_EXIT_BOOK | `exit_before_book_history` | 2026-03-17T04:39:51+00:00 | 2026-03-17T06:39:51+00:00 | None | None | 2197731.0 |
| 2026-03-18T13:01:44.487000+00:00 | NO_EXIT_BOOK | `exit_before_book_history` | 2026-03-18T13:01:52+00:00 | 2026-03-18T15:01:52+00:00 | None | None | 2081210.0 |
| 2026-03-20T07:01:18.235000+00:00 | NO_EXIT_BOOK | `exit_before_book_history` | 2026-03-20T07:01:30.011000+00:00 | 2026-03-20T09:01:30.011000+00:00 | None | None | 1930032.0 |
| 2026-03-23T15:43:52.147000+00:00 | NO_EXIT_BOOK | `exit_before_book_history` | 2026-03-23T15:46:23+00:00 | 2026-03-23T17:46:23+00:00 | None | None | 1639339.0 |
| 2026-03-30T17:24:19.103000+00:00 | NO_EXIT_BOOK | `exit_before_book_history` | 2026-03-30T17:26:30+00:00 | 2026-03-30T19:26:30+00:00 | None | None | 1028532.0 |
| 2026-03-31T07:18:20.133000+00:00 | NO_EXIT_BOOK | `exit_before_book_history` | 2026-03-31T08:45:28+00:00 | 2026-03-31T10:45:28+00:00 | None | None | 973394.0 |
| 2026-06-08T04:06:15.767000+00:00 | NO_EXIT_BOOK | `stale_exit_book_gap` | 2026-06-08T04:20:58.009000+00:00 | 2026-06-08T06:20:58.009000+00:00 | 2026-06-05T16:00:48.943000+00:00 | 224409.1 | 268364.0 |
| 2026-06-10T02:17:13.158000+00:00 | NO_EXIT_BOOK | `stale_exit_book_gap` | 2026-06-10T02:32:32+00:00 | 2026-06-10T04:32:32+00:00 | 2026-06-05T16:00:48.943000+00:00 | 390703.1 | 102070.0 |
