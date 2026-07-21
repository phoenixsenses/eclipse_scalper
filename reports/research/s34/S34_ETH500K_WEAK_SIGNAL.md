# S34 ETH 500K Weak Signal — Shadow Tag Report
Generated: `2026-06-27 15:12 UTC`

**Shadow observe only. No runner change. No live blocking.**

## Criteria

| Tag | Condition |
| --- | --- |
| `ETH_WEAK_COUNT_SHADOW` | liq_count ≤ 7 |
| `ETH_HIGH_SHARE_SHADOW` | max_single_share ≥ 80% |

## Performance Summary

| Group | N | WR | Median bps | Cum bps | SL | SL% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cnt≤7 | 3 | 67% | +51.7 | +72.7 | 1 | 33% |
| share≥80% | 5 | 80% | +52.5 | +189.8 | 1 | 20% |
| Either tag | 8 | 75% | +52.5 | +262.5 | 2 | 25% |
| Clean (no tag) | 11 | 82% | +52.3 | +480.8 | 0 | 0% |
| All ETH 500K | 19 | 79% | +52.3 | +743.3 | 2 | 11% |

## Status

- `ETH_WEAK_COUNT_SHADOW` DB rows: 3
- `ETH_HIGH_SHARE_SHADOW` DB rows: 5

Minimum N=30 tagged trades needed before any live evaluation.