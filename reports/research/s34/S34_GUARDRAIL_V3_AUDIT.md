# S34 Guardrail V3 Audit

Generated at: `2026-06-23T14:54:15.507737+00:00`

Scope: closed intelligence-ledger trades only. This is shadow/audit work. No runner, config, or live filter changed.

## Read

V3 tests whether the newer no-lookahead features can refine the V2 warning bucket. A row marked `too_early_*` is not a failed idea; it means the closed sample or feature coverage is not large enough to promote or reject it.

## Baseline

| Bucket | N | Cum | Mean | Median | WR % |
| --- | --- | --- | --- | --- | --- |
| all_closed | 71 | 1097.71 | 15.46 | 31.09 | 52.11 |
| warning_closed | 27 | 93.0 | 3.44 | -11.77 | 37.04 |

## Feature Coverage On Closed Trades

| Feature | Closed Rows | Coverage % | Warning Rows |
| --- | --- | --- | --- |
| day_trend_bps | 0 | 0.0 | 0 |
| day_range_bps | 0 | 0.0 | 0 |
| day_buy_liq_notional | 0 | 0.0 | 0 |
| day_agg_trade_count | 0 | 0.0 | 0 |
| cluster_duration_sec | 33 | 46.48 | 10 |
| max_single_liq_share | 33 | 46.48 | 10 |
| intensity_per_sec | 0 | 0.0 | 0 |
| inter_cluster_gap_sec | 0 | 0.0 | 0 |
| prev_liq_gap_sec | 0 | 0.0 | 0 |
| btc_pre_15m_bps | 2 | 2.82 | 1 |

## Warning By Rule

| Rule | N | Cum | Mean | Median | WR % |
| --- | --- | --- | --- | --- | --- |
| BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30 | 1 | 55.02 | 55.02 | 55.02 | 100.0 |
| ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60 | 1 | -24.58 | -24.58 | -24.58 | 0.0 |
| ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | 3 | 43.99 | 14.66 | 47.69 | 66.67 |
| ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 22 | 18.57 | 0.84 | -27.65 | 31.82 |

## Candidate Shadow Blocks

| Candidate | Status | Feature N | Coverage % | Block N | Block Cum | Block Median | Block WR % | Kept Delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_baseline_warning_100k_200k | too_early_n | 71 | 100.0 | 8 | -318.37 | -47.89 | 0.0 | 318.37 |
| warning_day_trend_negative | too_early_n | 0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| warning_100k_200k_day_trend_negative | too_early_n | 0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| warning_max_single_share_ge80 | too_early_n | 33 | 46.48 | 4 | -165.5 | -45.78 | 0.0 | 165.5 |
| warning_intensity_ge10k | too_early_n | 0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| warning_intensity_ge50k | too_early_n | 0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| warning_intensity_lt5k | too_early_n | 0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| warning_gap_le5s | too_early_n | 0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| warning_gap_le60s | too_early_n | 0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| warning_50k_rule_only | auditable_shadow_only | 71 | 100.0 | 22 | 18.57 | -27.65 | 31.82 | -18.57 |
| warning_50k_rule_day_trend_negative | too_early_n | 0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Candidate Examples

### v2_baseline_warning_100k_200k

Definition: `warning AND 100K <= cluster_notional < 200K`

Status: `too_early_n`

| Trade | Rule | Exit | Net | Cluster | Trend | Share | Intensity | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P056 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -53.45 | 146467.59 |  |  |  |  |
| P169 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -51.87 | 101877.85 |  |  |  |  |
| P419 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -49.36 | 151053.75 |  | 99.32 |  |  |
| P416 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -48.03 | 154940.25 |  | 99.97 |  |  |
| P063 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -47.76 | 153126.29 |  |  |  |  |
| P149 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -46.3 | 185562.3 |  |  |  |  |
| P394 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | BE | -11.77 | 177012.84 |  | 47.0 |  |  |
| P058 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | BE | -9.83 | 135373.6 |  |  |  |  |

### warning_day_trend_negative

Definition: `warning AND day_trend_bps < 0`

Status: `too_early_n`

No closed examples yet.

### warning_100k_200k_day_trend_negative

Definition: `warning AND 100K <= cluster_notional < 200K AND day_trend_bps < 0`

Status: `too_early_n`

No closed examples yet.

### warning_max_single_share_ge80

Definition: `warning AND max_single_liq_share >= 80`

Status: `too_early_n`

| Trade | Rule | Exit | Net | Cluster | Trend | Share | Intensity | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P419 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -49.36 | 151053.75 |  | 99.32 |  |  |
| P416 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -48.03 | 154940.25 |  | 99.97 |  |  |
| P361 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | TIME | -43.52 | 57157.49 |  | 100.0 |  |  |
| P357 | ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60 | TIME | -24.58 | 324826.4 |  | 80.92 |  |  |

### warning_intensity_ge10k

Definition: `warning AND intensity_per_sec >= 10K`

Status: `too_early_n`

No closed examples yet.

### warning_intensity_ge50k

Definition: `warning AND intensity_per_sec >= 50K`

Status: `too_early_n`

No closed examples yet.

### warning_intensity_lt5k

Definition: `warning AND intensity_per_sec < 5K`

Status: `too_early_n`

No closed examples yet.

### warning_gap_le5s

Definition: `warning AND inter_cluster_gap_sec <= 5`

Status: `too_early_n`

No closed examples yet.

### warning_gap_le60s

Definition: `warning AND inter_cluster_gap_sec <= 60`

Status: `too_early_n`

No closed examples yet.

### warning_50k_rule_only

Definition: `warning AND rule_name = 50K/TP120`

Status: `auditable_shadow_only`

| Trade | Rule | Exit | Net | Cluster | Trend | Share | Intensity | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P065 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -56.76 | 296517.07 |  |  |  |  |
| P150 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -55.59 | 76855.88 |  |  |  |  |
| P056 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -53.45 | 146467.59 |  |  |  |  |
| P418 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -53.37 | 90925.93 |  | 57.7 |  |  |
| P169 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -51.87 | 101877.85 |  |  |  |  |
| P419 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -49.36 | 151053.75 |  | 99.32 |  |  |
| P416 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -48.03 | 154940.25 |  | 99.97 |  |  |
| P063 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -47.76 | 153126.29 |  |  |  |  |
| P149 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -46.3 | 185562.3 |  |  |  |  |
| P116 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -45.82 | 58150.63 |  |  |  |  |
| P361 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | TIME | -43.52 | 57157.49 |  | 100.0 |  |  |
| P394 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | BE | -11.77 | 177012.84 |  | 47.0 |  |  |
| P062 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | BE | -10.86 | 1873291.01 |  |  |  |  |
| P058 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | BE | -9.83 | 135373.6 |  |  |  |  |
| P391 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | BE | -7.71 | 55976.47 |  | 63.95 |  |  |
| P353 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | TIME | 10.4 | 92406.31 |  | 67.89 |  |  |
| P351 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | TIME | 31.09 | 246140.41 |  | 66.33 |  |  |
| P060 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | TP | 99.67 | 751665.37 |  |  |  |  |
| P146 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | TP | 113.15 | 1661065.44 |  |  |  |  |
| P138 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | TP | 114.52 | 201207.01 |  |  |  |  |

### warning_50k_rule_day_trend_negative

Definition: `warning AND rule_name = 50K/TP120 AND day_trend_bps < 0`

Status: `too_early_n`

No closed examples yet.

## Verdict

V3 engine is ready, but the newer feature sample is still too sparse for a new hard-block rule. Keep V2 shadow running and rerun this audit after more feature-complete closed trades.
