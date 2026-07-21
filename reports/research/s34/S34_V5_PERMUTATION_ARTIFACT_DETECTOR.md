# S34 v5 Permutation Null + Artifact Detector

Generated: `2026-06-29T05:43:42.074995+00:00`

`RESEARCH_ONLY_NO_LIVE_NO_PAPER` - no live executor, runtime state, or paper state was touched.

- permutations per candidate: `1000`
- seed: `3405`

## Verdict Summary

| Candidate | Verdict | Null | Hold | Null p95 | p-right | Gauntlet failures |
| --- | --- | --- | --- | ---: | ---: | --- |
| `cascade_fade_all_signflip` | `ARTIFACT` | `sign_flip` | N=375 sum=-4367.5 T3R=-6250.2 | 6348.6 | 0.8721 | `non_overlap, holdout_positive_sum_t3r, cross_asset, beta_control, permutation_p_le_0_05` |
| `sync_gate_label_shuffle` | `ARTIFACT` | `label_shuffle` | N=286 sum=-4271.7 T3R=-6154.3 | -337.8 | 0.7283 | `non_overlap, holdout_positive_sum_t3r, cross_asset, beta_control, permutation_p_le_0_05` |
| `deep_bid_absorption_label_shuffle` | `ARTIFACT` | `label_shuffle` | N=171 sum=-7486.2 T3R=-8841.0 | 1060.3 | 0.997 | `non_overlap, holdout_positive_sum_t3r, cal_positive_sum_t3r, cross_asset, beta_control, permutation_p_le_0_05` |
| `sync_plus_deep_bid_label_shuffle` | `ARTIFACT` | `label_shuffle` | N=121 sum=-7587.2 T3R=-8942.0 | 1589.1 | 1.0 | `non_overlap, holdout_positive_sum_t3r, cal_positive_sum_t3r, cross_asset, beta_control, permutation_p_le_0_05` |
| `funding_nonoverlap_z1_all_signflip` | `ARTIFACT` | `sign_flip` | N=76 sum=-254.3 T3R=-1158.8 | 2547.1 | 0.5734 | `holdout_positive_sum_t3r, cal_positive_sum_t3r, beta_control, permutation_p_le_0_05` |
| `funding_nonoverlap_z1_eth_only_signflip` | `ARTIFACT` | `sign_flip` | N=26 sum=410.4 T3R=-406.0 | 1393.2 | 0.3337 | `holdout_positive_sum_t3r, cal_positive_sum_t3r, cross_asset, beta_control, n_ge_min, permutation_p_le_0_05` |
| `funding_nonoverlap_z1.5_all_signflip` | `ARTIFACT` | `sign_flip` | N=39 sum=-237.3 T3R=-1021.0 | 1765.9 | 0.5784 | `holdout_positive_sum_t3r, cal_positive_sum_t3r, cross_asset, beta_control, n_ge_min, permutation_p_le_0_05` |
| `funding_nonoverlap_z1.5_eth_only_signflip` | `ARTIFACT` | `sign_flip` | N=16 sum=-366.1 T3R=-847.9 | 1061.8 | 0.6773 | `holdout_positive_sum_t3r, cal_positive_sum_t3r, cross_asset, beta_control, n_ge_min, permutation_p_le_0_05` |
| `funding_nonoverlap_z2_all_signflip` | `ARTIFACT` | `sign_flip` | N=14 sum=311.9 T3R=-212.7 | 642.9 | 0.2118 | `holdout_positive_sum_t3r, cal_positive_sum_t3r, cross_asset, beta_control, n_ge_min, permutation_p_le_0_05` |
| `funding_nonoverlap_z2_eth_only_signflip` | `ARTIFACT` | `sign_flip` | N=5 sum=58.8 T3R=-132.2 | 308.5 | 0.4885 | `holdout_positive_sum_t3r, cal_positive_sum_t3r, cross_asset, beta_control, n_ge_min, permutation_p_le_0_05` |

## Detail

### cascade_fade_all_signflip

- verdict: `ARTIFACT`
- family: `cascade`
- cal: `{'n': 166, 'sum': 1752.9, 'mean': 10.56, 'median': -6.9, 'win_rate': 0.476, 't3r': 608.8, 'max_loss': -285.3}`
- hold: `{'n': 375, 'sum': -4367.5, 'mean': -11.65, 'median': 10.71, 'win_rate': 0.533, 't3r': -6250.2, 'max_loss': -507.2}`
- permutation: `{'n_perm': 1000, 'real_sum': -4367.5, 'null_p50': -47.4, 'null_p95': 6348.6, 'p_right': 0.8721}`
- gauntlet: `{'non_overlap': False, 'holdout_positive_sum_t3r': False, 'cal_positive_sum_t3r': True, 'real_cost': True, 'cross_asset': False, 'beta_control': 'FAIL:not_beta_controlled', 'n_ge_min': True, 'permutation_p_le_0_05': False}`
- by_symbol_hold: `{'BTCUSDT': {'n': 85, 'sum': -1916.5, 'mean': -22.55, 'median': 13.79, 'win_rate': 0.576, 't3r': -2665.6, 'max_loss': -417.1}, 'ETHUSDT': {'n': 157, 'sum': -2074.0, 'mean': -13.21, 'median': 15.75, 'win_rate': 0.541, 't3r': -3912.0, 'max_loss': -507.2}, 'SOLUSDT': {'n': 133, 'sum': -377.0, 'mean': -2.83, 'median': -0.13, 'win_rate': 0.496, 't3r': -2238.6, 'max_loss': -484.2}}`

### sync_gate_label_shuffle

- verdict: `ARTIFACT`
- family: `cascade_sync`
- cal: `{'n': 95, 'sum': 2316.5, 'mean': 24.38, 'median': 14.3, 'win_rate': 0.568, 't3r': 1172.3, 'max_loss': -271.1}`
- hold: `{'n': 286, 'sum': -4271.7, 'mean': -14.94, 'median': 10.47, 'win_rate': 0.524, 't3r': -6154.3, 'max_loss': -507.2}`
- permutation: `{'n_perm': 1000, 'real_sum': -4271.7, 'null_p50': -3217.6, 'null_p95': -337.8, 'p_right': 0.7283}`
- gauntlet: `{'non_overlap': False, 'holdout_positive_sum_t3r': False, 'cal_positive_sum_t3r': True, 'real_cost': True, 'cross_asset': False, 'beta_control': 'FAIL:not_beta_controlled', 'n_ge_min': True, 'permutation_p_le_0_05': False}`
- by_symbol_hold: `{'BTCUSDT': {'n': 71, 'sum': -2038.8, 'mean': -28.71, 'median': 7.74, 'win_rate': 0.535, 't3r': -2787.8, 'max_loss': -417.1}, 'ETHUSDT': {'n': 101, 'sum': -2096.2, 'mean': -20.75, 'median': 17.08, 'win_rate': 0.535, 't3r': -3934.2, 'max_loss': -507.2}, 'SOLUSDT': {'n': 114, 'sum': -136.7, 'mean': -1.2, 'median': 6.02, 'win_rate': 0.509, 't3r': -1998.3, 'max_loss': -484.2}}`

### deep_bid_absorption_label_shuffle

- verdict: `ARTIFACT`
- family: `cascade_absorption`
- cal: `{'n': 102, 'sum': -178.9, 'mean': -1.75, 'median': -14.68, 'win_rate': 0.412, 't3r': -831.8, 'max_loss': -271.1}`
- hold: `{'n': 171, 'sum': -7486.2, 'mean': -43.78, 'median': -14.9, 'win_rate': 0.474, 't3r': -8841.0, 'max_loss': -494.0}`
- permutation: `{'n_perm': 1000, 'real_sum': -7486.2, 'null_p50': -2055.3, 'null_p95': 1060.3, 'p_right': 0.997}`
- gauntlet: `{'non_overlap': False, 'holdout_positive_sum_t3r': False, 'cal_positive_sum_t3r': False, 'real_cost': True, 'cross_asset': False, 'beta_control': 'FAIL:not_beta_controlled', 'n_ge_min': True, 'permutation_p_le_0_05': False}`
- by_symbol_hold: `{'BTCUSDT': {'n': 35, 'sum': -292.2, 'mean': -8.35, 'median': 25.09, 'win_rate': 0.657, 't3r': -1041.2, 'max_loss': -417.1}, 'ETHUSDT': {'n': 70, 'sum': -4279.6, 'mean': -61.14, 'median': -60.23, 'win_rate': 0.443, 't3r': -5404.7, 'max_loss': -494.0}, 'SOLUSDT': {'n': 66, 'sum': -2914.5, 'mean': -44.16, 'median': -26.31, 'win_rate': 0.409, 't3r': -3837.4, 'max_loss': -484.2}}`

### sync_plus_deep_bid_label_shuffle

- verdict: `ARTIFACT`
- family: `cascade_confluence`
- cal: `{'n': 47, 'sum': -346.3, 'mean': -7.37, 'median': -7.43, 'win_rate': 0.426, 't3r': -830.7, 'max_loss': -271.1}`
- hold: `{'n': 121, 'sum': -7587.2, 'mean': -62.7, 'median': -16.78, 'win_rate': 0.463, 't3r': -8942.0, 'max_loss': -494.0}`
- permutation: `{'n_perm': 1000, 'real_sum': -7587.2, 'null_p50': -1329.3, 'null_p95': 1589.1, 'p_right': 1.0}`
- gauntlet: `{'non_overlap': False, 'holdout_positive_sum_t3r': False, 'cal_positive_sum_t3r': False, 'real_cost': True, 'cross_asset': False, 'beta_control': 'FAIL:not_beta_controlled', 'n_ge_min': True, 'permutation_p_le_0_05': False}`
- by_symbol_hold: `{'BTCUSDT': {'n': 30, 'sum': -424.0, 'mean': -14.13, 'median': 23.62, 'win_rate': 0.633, 't3r': -1173.1, 'max_loss': -417.1}, 'ETHUSDT': {'n': 37, 'sum': -4545.3, 'mean': -122.85, 'median': -147.08, 'win_rate': 0.351, 't3r': -5638.3, 'max_loss': -494.0}, 'SOLUSDT': {'n': 54, 'sum': -2617.9, 'mean': -48.48, 'median': -15.84, 'win_rate': 0.444, 't3r': -3540.8, 'max_loss': -484.2}}`

### funding_nonoverlap_z1_all_signflip

- verdict: `ARTIFACT`
- family: `funding_nonoverlap`
- cal: `{'n': 168, 'sum': -3334.2, 'mean': -19.85, 'median': -20.53, 'win_rate': 0.399, 't3r': -4239.2, 'max_loss': -560.7}`
- hold: `{'n': 76, 'sum': -254.3, 'mean': -3.35, 'median': 5.82, 'win_rate': 0.539, 't3r': -1158.8, 'max_loss': -585.2}`
- permutation: `{'n_perm': 1000, 'real_sum': -254.3, 'null_p50': 54.5, 'null_p95': 2547.1, 'p_right': 0.5734}`
- gauntlet: `{'non_overlap': True, 'holdout_positive_sum_t3r': False, 'cal_positive_sum_t3r': False, 'real_cost': True, 'cross_asset': True, 'beta_control': 'FAIL:side_split_not_consistent', 'n_ge_min': True, 'permutation_p_le_0_05': False}`
- by_symbol_hold: `{'BTCUSDT': {'n': 30, 'sum': 210.2, 'mean': 7.01, 'median': 10.16, 'win_rate': 0.567, 't3r': -582.8, 'max_loss': -426.7}, 'ETHUSDT': {'n': 26, 'sum': 410.4, 'mean': 15.78, 'median': 5.82, 'win_rate': 0.538, 't3r': -406.0, 'max_loss': -585.2}, 'SOLUSDT': {'n': 20, 'sum': -874.8, 'mean': -43.74, 'median': -30.85, 'win_rate': 0.5, 't3r': -1474.6, 'max_loss': -390.3}}`

### funding_nonoverlap_z1_eth_only_signflip

- verdict: `ARTIFACT`
- family: `funding_nonoverlap`
- cal: `{'n': 62, 'sum': -1140.7, 'mean': -18.4, 'median': -28.1, 'win_rate': 0.355, 't3r': -1858.5, 'max_loss': -400.8}`
- hold: `{'n': 26, 'sum': 410.4, 'mean': 15.78, 'median': 5.82, 'win_rate': 0.538, 't3r': -406.0, 'max_loss': -585.2}`
- permutation: `{'n_perm': 1000, 'real_sum': 410.4, 'null_p50': -15.6, 'null_p95': 1393.2, 'p_right': 0.3337}`
- gauntlet: `{'non_overlap': True, 'holdout_positive_sum_t3r': False, 'cal_positive_sum_t3r': False, 'real_cost': True, 'cross_asset': False, 'beta_control': 'FAIL:single_asset', 'n_ge_min': False, 'permutation_p_le_0_05': False}`
- by_symbol_hold: `{'ETHUSDT': {'n': 26, 'sum': 410.4, 'mean': 15.78, 'median': 5.82, 'win_rate': 0.538, 't3r': -406.0, 'max_loss': -585.2}}`

### funding_nonoverlap_z1.5_all_signflip

- verdict: `ARTIFACT`
- family: `funding_nonoverlap`
- cal: `{'n': 88, 'sum': -1988.8, 'mean': -22.6, 'median': -18.15, 'win_rate': 0.409, 't3r': -2770.7, 'max_loss': -560.7}`
- hold: `{'n': 39, 'sum': -237.3, 'mean': -6.08, 'median': -3.85, 'win_rate': 0.487, 't3r': -1021.0, 'max_loss': -585.2}`
- permutation: `{'n_perm': 1000, 'real_sum': -237.3, 'null_p50': -49.8, 'null_p95': 1765.9, 'p_right': 0.5784}`
- gauntlet: `{'non_overlap': True, 'holdout_positive_sum_t3r': False, 'cal_positive_sum_t3r': False, 'real_cost': True, 'cross_asset': False, 'beta_control': 'FAIL:side_split_not_consistent', 'n_ge_min': False, 'permutation_p_le_0_05': False}`
- by_symbol_hold: `{'BTCUSDT': {'n': 9, 'sum': 357.2, 'mean': 39.69, 'median': 9.89, 'win_rate': 0.556, 't3r': -289.8, 'max_loss': -150.1}, 'ETHUSDT': {'n': 16, 'sum': -366.1, 'mean': -22.88, 'median': -9.56, 'win_rate': 0.375, 't3r': -847.9, 'max_loss': -585.2}, 'SOLUSDT': {'n': 14, 'sum': -228.5, 'mean': -16.32, 'median': 11.44, 'win_rate': 0.571, 't3r': -791.7, 'max_loss': -344.7}}`

### funding_nonoverlap_z1.5_eth_only_signflip

- verdict: `ARTIFACT`
- family: `funding_nonoverlap`
- cal: `{'n': 26, 'sum': -957.8, 'mean': -36.84, 'median': -42.58, 'win_rate': 0.346, 't3r': -1485.0, 'max_loss': -400.8}`
- hold: `{'n': 16, 'sum': -366.1, 'mean': -22.88, 'median': -9.56, 'win_rate': 0.375, 't3r': -847.9, 'max_loss': -585.2}`
- permutation: `{'n_perm': 1000, 'real_sum': -366.1, 'null_p50': 53.4, 'null_p95': 1061.8, 'p_right': 0.6773}`
- gauntlet: `{'non_overlap': True, 'holdout_positive_sum_t3r': False, 'cal_positive_sum_t3r': False, 'real_cost': True, 'cross_asset': False, 'beta_control': 'FAIL:single_asset', 'n_ge_min': False, 'permutation_p_le_0_05': False}`
- by_symbol_hold: `{'ETHUSDT': {'n': 16, 'sum': -366.1, 'mean': -22.88, 'median': -9.56, 'win_rate': 0.375, 't3r': -847.9, 'max_loss': -585.2}}`

### funding_nonoverlap_z2_all_signflip

- verdict: `ARTIFACT`
- family: `funding_nonoverlap`
- cal: `{'n': 31, 'sum': -1312.4, 'mean': -42.33, 'median': -39.93, 'win_rate': 0.323, 't3r': -2000.2, 'max_loss': -560.7}`
- hold: `{'n': 14, 'sum': 311.9, 'mean': 22.28, 'median': 4.7, 'win_rate': 0.571, 't3r': -212.7, 'max_loss': -150.1}`
- permutation: `{'n_perm': 1000, 'real_sum': 311.9, 'null_p50': 7.1, 'null_p95': 642.9, 'p_right': 0.2118}`
- gauntlet: `{'non_overlap': True, 'holdout_positive_sum_t3r': False, 'cal_positive_sum_t3r': False, 'real_cost': True, 'cross_asset': False, 'beta_control': 'FAIL:side_split_not_consistent', 'n_ge_min': False, 'permutation_p_le_0_05': False}`
- by_symbol_hold: `{'BTCUSDT': {'n': 5, 'sum': -62.2, 'mean': -12.44, 'median': -11.96, 'win_rate': 0.4, 't3r': -219.8, 'max_loss': -150.1}, 'ETHUSDT': {'n': 5, 'sum': 58.8, 'mean': 11.75, 'median': 7.37, 'win_rate': 0.6, 't3r': -132.2, 'max_loss': -107.9}, 'SOLUSDT': {'n': 4, 'sum': 315.3, 'mean': 78.83, 'median': 77.99, 'win_rate': 0.75, 't3r': -65.3, 'max_loss': -65.3}}`

### funding_nonoverlap_z2_eth_only_signflip

- verdict: `ARTIFACT`
- family: `funding_nonoverlap`
- cal: `{'n': 13, 'sum': -208.2, 'mean': -16.02, 'median': -39.93, 'win_rate': 0.308, 't3r': -601.2, 'max_loss': -297.9}`
- hold: `{'n': 5, 'sum': 58.8, 'mean': 11.75, 'median': 7.37, 'win_rate': 0.6, 't3r': -132.2, 'max_loss': -107.9}`
- permutation: `{'n_perm': 1000, 'real_sum': 58.8, 'null_p50': 44.0, 'null_p95': 308.5, 'p_right': 0.4885}`
- gauntlet: `{'non_overlap': True, 'holdout_positive_sum_t3r': False, 'cal_positive_sum_t3r': False, 'real_cost': True, 'cross_asset': False, 'beta_control': 'FAIL:single_asset', 'n_ge_min': False, 'permutation_p_le_0_05': False}`
- by_symbol_hold: `{'ETHUSDT': {'n': 5, 'sum': 58.8, 'mean': 11.75, 'median': 7.37, 'win_rate': 0.6, 't3r': -132.2, 'max_loss': -107.9}}`

## Read

- `label_shuffle` asks whether a filter selects better outcomes than a same-size random subset of holdout events.
- `sign_flip` asks whether the selected P&L is larger than a no-directional-edge sign-randomized null.
- A permutation win alone is not enough: the gauntlet also requires non-overlap, holdout, cost, cross-asset, beta control, and N discipline.
