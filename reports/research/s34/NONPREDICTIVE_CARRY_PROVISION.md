# Non-Predictive Carry + Provision Research

Generated: `2026-06-29T06:02:16.500045+00:00`

`RESEARCH_ONLY_NO_LIVE_NO_PAPER` - no live executor, paper state, or runtime state was touched.

## Scope Sanity

- v5 shuffles directional PnL labels/signs; it does not model exogenous funding cashflows or two-sided spread capture.
- Therefore v5 kills directional prediction; it does not by itself kill carry/provision mechanisms.

## Funding Carry Harvest

Paid side: positive funding -> SHORT receives; negative funding -> LONG receives. Net = price P&L + funding cashflow - round-trip cost.

| Rank | Config | All | Cal | Hold | Hold funding-only | Hold price component |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `carry_paid_side_min_abs_funding_0bps` | N=1014 sum=416.0 mean=0.4 med=-0.6 T3R=-1595.9 WR=0.499 maxL=-777.5 | N=762 sum=-2174.0 mean=-2.9 med=-1.9 T3R=-3973.3 WR=0.491 maxL=-777.5 | N=252 sum=2590.0 mean=10.3 med=2.7 T3R=732.8 WR=0.524 maxL=-584.9 | N=252 sum=-1899.2 mean=-7.5 med=-7.7 T3R=-1884.2 WR=0.0 maxL=-8.0 | N=252 sum=4489.2 mean=17.8 med=10.3 T3R=2609.4 WR=0.56 maxL=-577.2 |
| 2 | `carry_paid_side_min_abs_funding_1bps` | N=76 sum=-185.5 mean=-2.4 med=-17.0 T3R=-1043.6 WR=0.461 maxL=-343.5 | N=53 sum=-1281.1 mean=-24.2 med=-35.6 T3R=-1934.2 WR=0.377 maxL=-295.6 | N=23 sum=1095.7 mean=47.6 med=85.8 T3R=243.1 WR=0.652 maxL=-343.5 | N=23 sum=-147.2 mean=-6.4 med=-6.8 T3R=-132.2 WR=0.0 maxL=-7.0 | N=23 sum=1242.9 mean=54.0 med=92.8 T3R=372.2 WR=0.652 maxL=-336.7 |
| 3 | `carry_paid_side_min_abs_funding_2bps` | N=10 sum=306.5 mean=30.7 med=79.0 T3R=-338.5 WR=0.7 maxL=-298.7 | N=4 sum=173.7 mean=43.4 med=117.6 T3R=-295.6 WR=0.75 maxL=-295.6 | N=6 sum=132.8 mean=22.1 med=54.8 T3R=-357.1 WR=0.667 maxL=-298.7 | N=6 sum=-32.2 mean=-5.4 med=-5.5 T3R=-17.2 WR=0.0 maxL=-5.9 | N=6 sum=165.0 mean=27.5 med=60.5 T3R=-340.6 WR=0.667 maxL=-293.3 |
| 4 | `carry_paid_side_min_abs_funding_5bps` | N=0 sum=0.0 mean=None med=None T3R=0.0 WR=None maxL=None | N=0 sum=0.0 mean=None med=None T3R=0.0 WR=None maxL=None | N=0 sum=0.0 mean=None med=None T3R=0.0 WR=None maxL=None | N=0 sum=0.0 mean=None med=None T3R=0.0 WR=None maxL=None | N=0 sum=0.0 mean=None med=None T3R=0.0 WR=None maxL=None |

### Best Carry By Symbol

| Symbol | Hold |
| --- | --- |
| `BTCUSDT` | N=84 sum=1093.5 mean=13.0 med=-0.1 T3R=2.4 WR=0.5 maxL=-426.6 |
| `ETHUSDT` | N=84 sum=4007.9 mean=47.7 med=32.0 T3R=2241.2 WR=0.619 maxL=-584.9 |
| `SOLUSDT` | N=84 sum=-2511.4 mean=-29.9 med=-48.6 T3R=-4055.6 WR=0.452 maxL=-469.9 |

## Direction-Agnostic Maker Provision

At each cascade anchor, quote both sides around mid. Both-filled = spread capture; one-side-filled = inventory flattened at horizon.

| Rank | Config | Fill counts | All | Cal | Hold |
| ---: | --- | --- | --- | --- | --- |
| 1 | `provision_o2_h300s` | `{'BOTH_FILLED': 213, 'BID_ONLY_LONG_INVENTORY': 10, 'ASK_ONLY_SHORT_INVENTORY': 12}` | N=235 sum=-1061.4 mean=-4.5 med=-0.0 T3R=-1061.4 WR=0.362 maxL=-169.4 | N=68 sum=-96.7 mean=-1.4 med=-0.0 T3R=-96.7 WR=0.471 maxL=-35.2 | N=167 sum=-964.8 mean=-5.8 med=-0.0 T3R=-964.8 WR=0.317 maxL=-169.4 |

### Best Provision By Symbol

| Symbol | Hold |
| --- | --- |
| `BTCUSDT` | N=44 sum=-285.5 mean=-6.5 med=-0.0 T3R=-285.5 WR=0.295 maxL=-103.2 |
| `ETHUSDT` | N=60 sum=-167.9 mean=-2.8 med=-0.0 T3R=-167.9 WR=0.3 maxL=-124.8 |
| `SOLUSDT` | N=63 sum=-511.4 mean=-8.1 med=-0.0 T3R=-511.4 WR=0.349 maxL=-169.4 |

## Read

- Funding carry is not directional mean-reversion; price P&L and funding cashflow are reported separately.
- Maker provision is a first-pass touch-fill model and is optimistic about queue priority. If it fails here, the real version is unlikely to improve.
- A positive result here would still need forward shadow and exchange-fee/rebate-specific validation before any live decision.
