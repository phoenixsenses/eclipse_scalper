# S34 Bridge Backtest (mark-based + modeled spread, full Feb-Jun span)

Generated: `2026-06-28T22:02:22.770847+00:00`  |  ETHUSDT SELL deep-V>= 28.0bps 200K 4h, cost 8.1bps RT (2*fee+spread), cont SL 20.0bps

Events: 169  |  months present: 2026-02, 2026-03, 2026-04, 2026-06

Per-month P&L. The test of the bridge: is the edge positive across MANY months, or does it flip every regime?

| Month | N | FADE sum | FADE win | FADE maxL | CONT sum | CONT win | CONT maxL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-02 | 35 | -223.3 | 54.3 | -331.4 | 266.3 | 14.3 | -28.1 |
| 2026-03 | 61 | 1002.3 | 59.0 | -319.7 | -1035.9 | 4.9 | -28.1 |
| 2026-04 | 27 | 759.2 | 59.3 | -410.0 | -474.2 | 3.7 | -28.1 |
| 2026-06 | 46 | 19.8 | 50.0 | -342.3 | -323.9 | 15.2 | -28.1 |

FADE positive months: 3/4  |  CONT positive months: 1/4

## Chronological holdout (full span)
- FADE: cal sum=1223.8 (N=118, win 56.8%) | hold sum=334.1 (N=51, win 52.9%)
- CONT: cal sum=-1103.3 (N=118, win 7.6%) | hold sum=-464.4 (N=51, win 13.7%)
