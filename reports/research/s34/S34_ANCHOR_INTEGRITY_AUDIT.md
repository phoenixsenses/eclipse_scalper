# S34 Anchor Integrity Audit

Generated: `2026-06-28T10:59:22.399465+00:00`

Classifies closed forward paper trades by whether the recorded signal anchor is compatible with a knowable threshold-cross anchor.

Definitions:
- `lookahead_like_cluster_start_entry`: `signal_ts` is within 1s of `cluster_start`, while `cluster_end` is later. This records full cluster notional while entering at the cluster start.
- `knowable_like_threshold_or_cluster_end`: signal is not at cluster start, or equals stored threshold/cluster end.
- `missing_old_format`: older trade snapshot lacks cluster start/end fields.

## ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30

Overall closed N=21, median=52.3, mean=37.2, cum=780.9, WR=0.762

| Anchor Class | N | Median | Mean | Cum | WR |
| --- | ---: | ---: | ---: | ---: | ---: |
| knowable_like_threshold_or_cluster_end | 2 | 18.8 | 18.8 | 37.6 | 0.5 |
| lookahead_like_cluster_start_entry | 15 | 53.5 | 42.7 | 640.4 | 0.8 |
| missing_old_format | 4 | 49.6 | 25.7 | 102.8 | 0.75 |

## ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30

Overall closed N=25, median=-38.9, mean=-3.2, cum=-80.9, WR=0.28

| Anchor Class | N | Median | Mean | Cum | WR |
| --- | ---: | ---: | ---: | ---: | ---: |
| knowable_like_threshold_or_cluster_end | 1 | -43.5 | -43.5 | -43.5 | 0.0 |
| lookahead_like_cluster_start_entry | 9 | -38.9 | -24.4 | -219.3 | 0.222 |
| missing_old_format | 15 | -10.9 | 12.1 | 182.0 | 0.333 |
