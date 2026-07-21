# Non-Predictive Carry + Provision Research

Generated: `2026-06-29T06:03:59.991377+00:00`

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
| 1 | `provision_o2_h300s_fee-0.5` | `{'BOTH_FILLED': 213, 'BID_ONLY_LONG_INVENTORY': 10, 'ASK_ONLY_SHORT_INVENTORY': 12}` | N=235 sum=58.6 mean=0.2 med=5.0 T3R=43.6 WR=0.906 maxL=-166.9 | N=68 sum=230.8 mean=3.4 med=5.0 T3R=215.8 WR=0.926 maxL=-32.7 | N=167 sum=-172.3 mean=-1.0 med=5.0 T3R=-187.3 WR=0.898 maxL=-166.9 |

### Best Provision By Symbol

| Symbol | Hold |
| --- | --- |
| `BTCUSDT` | N=44 sum=-80.5 mean=-1.8 med=5.0 T3R=-95.5 WR=0.864 maxL=-100.7 |
| `ETHUSDT` | N=60 sum=122.1 mean=2.0 med=5.0 T3R=107.1 WR=0.933 maxL=-122.3 |
| `SOLUSDT` | N=63 sum=-213.9 mean=-3.4 med=5.0 T3R=-228.9 WR=0.889 maxL=-166.9 |

## Read

- Funding carry is not directional mean-reversion; price P&L and funding cashflow are reported separately.
- Maker provision is a first-pass touch-fill model and is optimistic about queue priority. If it fails here, the real version is unlikely to improve.
- A positive result here would still need forward shadow and exchange-fee/rebate-specific validation before any live decision.
