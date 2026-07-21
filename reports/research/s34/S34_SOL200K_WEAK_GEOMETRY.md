# S34 SOL 200K Weak Geometry — Shadow Tag Report

Generated: `2026-06-27T10:53:54.544504+00:00`

**Shadow observe only. No runner change. No live blocking.**

Criteria for `SOL_WEAK_GEOMETRY_SHADOW` tag (any one triggers):
- cluster_notional 500K–1M
- max_single_liq_share ≥ 80%
- cluster_liq_count ≤ 2

## Performance Split

| Group | N | Median | Cum | Top3R | WR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tagged (weak geometry) | 14 | 18.5 | 86.7 | 268.1 | 50% |
| Untagged (clean) | 10 | 67.4 | 553.7 | 528.3 | 80% |
| All SOL 200K | 24 | 48.5 | 640.4 | 821.8 | 62% |

## Tagged Trades (SOL_WEAK_GEOMETRY_SHADOW)

| Trade ID | Net bps | Exit | Cascade | Liq Count | Single Share | Reasons |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| P349 | -63.2 | SL | 690,696 | 4 | 97.2% | cascade_500K_1M (690696), single_share_gte80 (97.2%) |
| P352 | +52.6 | TP | 558,976 | 26 | 30.2% | cascade_500K_1M (558977) |
| P383 | +48.6 | TP | 647,125 | 13 | 78.5% | cascade_500K_1M (647125) |
| P466 | -62.9 | SL | 533,577 | 6 | 71.2% | cascade_500K_1M (533577) |
| P505 | -14.0 | BE | 777,935 | 9 | 92.8% | cascade_500K_1M (777935), single_share_gte80 (92.8%) |
| P507 | +48.3 | TP | 203,657 | 6 | 97.0% | single_share_gte80 (97.0%) |
| P516 | -5.1 | BE | 531,748 | 9 | 68.3% | cascade_500K_1M (531748) |
| P528 | +54.8 | TP | 418,697 | 4 | 97.6% | single_share_gte80 (97.6%) |
| P611 | +59.7 | TP | 247,571 | 1 | 100.0% | single_share_gte80 (100.0%), liq_count_lte2 (1) |
| P630 | +42.0 | TP | 374,230 | 16 | 83.1% | single_share_gte80 (83.1%) |
| P645 | -52.9 | SL | 1,008,072 | 3 | 100.0% | single_share_gte80 (100.0%) |
| P662 | -16.4 | BE | 634,664 | 7 | 83.6% | cascade_500K_1M (634664), single_share_gte80 (83.6%) |
| P688 | +50.6 | TP | 289,991 | 4 | 99.4% | single_share_gte80 (99.4%) |
| P690 | -55.4 | SL | 581,217 | 2 | 99.9% | cascade_500K_1M (581218), single_share_gte80 (99.9%), liq_count_lte2 (2) |

## DB Tag Table

Tags written to `s34_shadow_geometry_tags` in `s34_intelligence.db`.
Total rows: 14

Schema: `trade_id, rule_name, tag, reasons, cascade_usd, liq_count, single_share, net_bps, exit_reason`

## Interpretation

- **No block recommended** — N_tagged too small for live filter.
- Monitor: if tagged trades continue underperforming as N grows toward 20+, consider `min_liq_count >= 3` as an exploratory shadow rule.
- The 500K–1M cascade band and single-dominant spike may indicate a different market dynamic (concentrated forced sell vs distributed cascade).
- Revisit when total SOL 200K N ≥ 50.