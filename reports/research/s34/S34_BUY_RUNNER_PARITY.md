# S34 BUY Rule Runner Parity

Generated: `2026-06-27T09:53:55.436737+00:00`

Exact runner-parity for all active BUY rules. Tests timing-gap hypothesis:
does runner's late entry (at threshold cross vs feature factory's first_ts) hurt BUY rules?

| Rule | Group | FF Median | Runner Median | Delta | Cascade Move | WR | H1 | H2 | Prelim |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ETH_BUY_500K_DAYTREND0 | MAIN | +52.2 | -16.6 | -68.8 | +40.1 | 33% | -15.3 | -18.6 | yes |
| SOL_BUY_200K | MAIN | +52.3 | -9.4 | -61.7 | +39.1 | 40% | -6.4 | -49.4 |  |
| SOL_BUY_100K | MAIN | +52.1 | -12.1 | -64.2 | +35.5 | 38% | -11.7 | -12.3 |  |
| BTC_BUY_1M_DISTRIBUTED | MAIN | +22.2 | -11.9 | -34.1 | +29.1 | 29% | -12.3 | -11.5 |  |
| ETH_BUY_200K | EXPLORATORY | -5.1 | -10.6 | -5.5 | +29.6 | 34% | -10.6 | -10.5 | yes |
| ETH_BUY_200K_BTC_PRE15_DELAY60 | EXPLORATORY | -8.8 | -10.7 | -1.9 | +18.1 | 23% | -10.2 | -15.9 | yes |
| ETH_BUY_500K_NEGTREND_STRETCHED | EXPLORATORY | +52.2 | -5.6 | -57.8 | +40.1 | 33% | -4.6 | -6.5 | yes |

## Interpretation

- **Delta** = runner median − FF median. Negative delta = runner underperforms FF (timing gap penalty).
- **Cascade Move** = median price move from first_ts to cluster_end_ts during cascade. For BUY liq (SHORT liquidations → forced buying), price should RISE during cascade.
- If cascade_move is large AND delta is similarly large negative, timing gap is the cause.
- VIABLE = N>=30, NF<40%, runner median>0.

## Viable Rules

None.