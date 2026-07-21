# S34 State Machine V8 Robustness Suite

- generated_at_utc: `2026-06-30T19:06:14.799523+00:00`
- research_only: `true`
- live_changes: `none`

## Ideas / Questions Tested

1. Does profit-lock 100/50 survive folds, fee stress, and side split?
2. Is confidence sizing robust or just larger risk?
3. Does execution latency destroy the state-machine edge?
4. Is early weak/adverse movement useful as a defensive cut?
5. Does partial scale-out improve skew?
6. Which conflict policy is safest?
7. Which BTC/score grid is best without overfitting?
8. Does top-winner removal break candidates?
9. Which candidates survive extra 2/5/10/20 bps cost?
10. Which candidates deserve shadow only vs live consideration?

## Candidate Robustness

- baseline: hold `N=30 WR=83.3% sum=3471.4 mean=115.7 med=106.8 T3R=2411.8 maxL=-52.0 DD=52.0`, folds_pos=5/5, fold_t3r_sum=1489.7, hold_top3_removed `N=27 WR=81.5% sum=2411.8 mean=89.3 med=76.7 T3R=1565.9 maxL=-52.0 DD=52.0`
- profit_lock_100_50: hold `N=30 WR=86.7% sum=3700.6 mean=123.4 med=106.8 T3R=2640.9 maxL=-52.0 DD=52.0`, folds_pos=5/5, fold_t3r_sum=2101.1, hold_top3_removed `N=27 WR=85.2% sum=2640.9 mean=97.8 med=76.7 T3R=1795.1 maxL=-52.0 DD=52.0`
- profit_lock_long_only: hold `N=30 WR=83.3% sum=3471.3 mean=115.7 med=106.8 T3R=2411.6 maxL=-52.0 DD=52.0`, folds_pos=5/5, fold_t3r_sum=1706.8, hold_top3_removed `N=27 WR=81.5% sum=2411.6 mean=89.3 med=76.7 T3R=1565.8 maxL=-52.0 DD=52.0`
- profit_lock_short_only: hold `N=30 WR=86.7% sum=3700.7 mean=123.4 med=106.8 T3R=2641.1 maxL=-52.0 DD=52.0`, folds_pos=5/5, fold_t3r_sum=1783.7, hold_top3_removed `N=27 WR=85.2% sum=2641.1 mean=97.8 med=76.7 T3R=1795.2 maxL=-52.0 DD=52.0`
- confidence_sized: hold `N=30 WR=80.0% sum=4419.4 mean=147.3 med=93.6 T3R=2853.0 maxL=-60.2 DD=60.2`, folds_pos=5/5, fold_t3r_sum=1530.8, hold_top3_removed `N=27 WR=77.8% sum=2853.0 mean=105.7 med=44.7 T3R=1806.3 maxL=-60.2 DD=72.2`
- early_cut_5m_weak20: hold `N=30 WR=73.3% sum=3094.4 mean=103.1 med=71.2 T3R=2034.7 maxL=-89.5 DD=89.5`, folds_pos=5/5, fold_t3r_sum=221.1, hold_top3_removed `N=27 WR=70.4% sum=2034.7 mean=75.4 med=15.2 T3R=1188.9 maxL=-89.5 DD=89.5`
- early_cut_5m_adverse20: hold `N=30 WR=66.7% sum=2238.2 mean=74.6 med=32.3 T3R=1234.0 maxL=-89.5 DD=89.5`, folds_pos=5/5, fold_t3r_sum=-10.9, hold_top3_removed `N=27 WR=63.0% sum=1234.0 mean=45.7 med=16.5 T3R=461.9 maxL=-89.5 DD=89.5`
- early_cut_5m_weak20_or_adverse20: hold `N=30 WR=63.3% sum=2063.2 mean=68.8 med=5.5 T3R=1059.0 maxL=-89.5 DD=89.5`, folds_pos=5/5, fold_t3r_sum=-745.1, hold_top3_removed `N=27 WR=59.3% sum=1059.0 mean=39.2 med=3.1 T3R=286.9 maxL=-89.5 DD=89.5`
- scaleout_100_half: hold `N=30 WR=86.7% sum=2791.3 mean=93.0 med=100.9 T3R=2119.0 maxL=-52.0 DD=52.0`, folds_pos=5/5, fold_t3r_sum=2036.9, hold_top3_removed `N=27 WR=85.2% sum=2119.0 mean=78.5 med=85.9 T3R=1553.5 maxL=-52.0 DD=52.0`
- scaleout_150_half: hold `N=30 WR=83.3% sum=3097.7 mean=103.3 med=118.5 T3R=2350.4 maxL=-52.0 DD=52.0`, folds_pos=5/5, fold_t3r_sum=1861.0, hold_top3_removed `N=27 WR=81.5% sum=2350.4 mean=87.1 med=106.5 T3R=1709.9 maxL=-52.0 DD=52.0`

## Top Config Grid By Holdout T3R

- btc1000_l3_s3: `N=30 WR=83.3% sum=3471.4 mean=115.7 med=106.8 T3R=2411.8 maxL=-52.0 DD=52.0` small_n=False
- btc750_l3_s3: `N=32 WR=78.1% sum=3359.0 mean=105.0 med=72.8 T3R=2299.4 maxL=-52.0 DD=70.5` small_n=False
- btc1000_l3_s4: `N=23 WR=91.3% sum=3284.2 mean=142.8 med=137.9 T3R=2224.6 maxL=-40.2 DD=40.2` small_n=False
- btc750_l3_s4: `N=24 WR=87.5% sum=3158.6 mean=131.6 med=122.2 T3R=2099.0 maxL=-40.2 DD=40.2` small_n=False
- btc1500_l3_s3: `N=26 WR=84.6% sum=3024.8 mean=116.3 med=91.6 T3R=1965.2 maxL=-52.0 DD=52.0` small_n=False
- btc1500_l3_s4: `N=21 WR=90.5% sum=2961.5 mean=141.0 med=107.1 T3R=1901.9 maxL=-49.3 DD=49.3` small_n=False
- btc1000_l4_s3: `N=24 WR=79.2% sum=2680.4 mean=111.7 med=111.2 T3R=1636.1 maxL=-52.0 DD=52.0` small_n=False
- btc1000_l5_s3: `N=22 WR=77.3% sum=2558.0 mean=116.3 med=126.5 T3R=1513.7 maxL=-52.0 DD=52.0` small_n=False
- btc750_l4_s3: `N=25 WR=72.0% sum=2556.0 mean=102.2 med=35.8 T3R=1511.7 maxL=-52.0 DD=82.5` small_n=False
- btc750_l5_s3: `N=24 WR=70.8% sum=2540.8 mean=105.9 med=75.5 T3R=1496.4 maxL=-52.0 DD=82.5` small_n=False

## Full JSON

- `D:\eclipse_scalper\reports\research\s34\S34_STATE_MACHINE_V8_ROBUSTNESS_SUITE.json`
