# Non-Predictive Carry + Provision Research

Generated: `2026-06-29T06:03:59.991209+00:00`

`RESEARCH_ONLY_NO_LIVE_NO_PAPER` - no live executor, paper state, or runtime state was touched.

## Scope Sanity

- v5 shuffles directional PnL labels/signs; it does not model exogenous funding cashflows or two-sided spread capture.
- Therefore v5 kills directional prediction; it does not by itself kill carry/provision mechanisms.

## Funding Carry Harvest

Paid side: positive funding -> SHORT receives; negative funding -> LONG receives. Net = price P&L + funding cashflow - round-trip cost.

| Rank | Config | All | Cal | Hold | Hold funding-only | Hold price component |
| ---: | --- | --- | --- | --- | --- | --- |

## Direction-Agnostic Maker Provision

At each cascade anchor, quote both sides around mid. Both-filled = spread capture; one-side-filled = inventory flattened at horizon.

| Rank | Config | Fill counts | All | Cal | Hold |
| ---: | --- | --- | --- | --- | --- |
| 1 | `provision_o2_h300s_fee0` | `{'BOTH_FILLED': 213, 'BID_ONLY_LONG_INVENTORY': 10, 'ASK_ONLY_SHORT_INVENTORY': 12}` | N=235 sum=-165.4 mean=-0.7 med=4.0 T3R=-177.4 WR=0.906 maxL=-167.4 | N=68 sum=165.3 mean=2.4 med=4.0 T3R=153.3 WR=0.926 maxL=-33.2 | N=167 sum=-330.8 mean=-2.0 med=4.0 T3R=-342.8 WR=0.898 maxL=-167.4 |

### Best Provision By Symbol

| Symbol | Hold |
| --- | --- |
| `BTCUSDT` | N=44 sum=-121.5 mean=-2.8 med=4.0 T3R=-133.5 WR=0.864 maxL=-101.2 |
| `ETHUSDT` | N=60 sum=64.1 mean=1.1 med=4.0 T3R=52.1 WR=0.933 maxL=-122.8 |
| `SOLUSDT` | N=63 sum=-273.4 mean=-4.3 med=4.0 T3R=-285.4 WR=0.889 maxL=-167.4 |

## Read

- Funding carry is not directional mean-reversion; price P&L and funding cashflow are reported separately.
- Maker provision is a first-pass touch-fill model and is optimistic about queue priority. If it fails here, the real version is unlikely to improve.
- A positive result here would still need forward shadow and exchange-fee/rebate-specific validation before any live decision.
