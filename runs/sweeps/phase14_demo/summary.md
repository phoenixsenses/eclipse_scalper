# Sweep Eval Summary

- db: `data\microstructure.db`
- symbols: `ETHUSDT`
- slice: `2026-03-01T22:41:00Z` -> `2026-03-01T22:41:20Z`
- strategy: `baseline`
- sort: `pnl_net_sum` (desc)
- grid: `fee_bps=0,0.6;spread_bps=0,2,5,10;horizon_sec=5,10,30`
- total_runs: 24

## Top N

| rank | run_dir | pnl_net_sum | pnl_net_per_fill | fills_count | spread_bps | fee_bps | horizon_sec | avg_adverse_samples |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `runs/baseline_cba2a05ea5` | 21.316232518000 | 0.049688187688 | 429 | 0.0 | 0.0 | 30 | 1082.934732 |
| 2 | `runs/baseline_f4dd9bb1e6` | 17.155390658000 | 0.039989255613 | 429 | 0.0 | 0.0 | 10 | 982.727273 |
| 3 | `runs/baseline_c943366216` | 16.354118771951 | 0.038121488979 | 429 | 0.0 | 0.6 | 30 | 1082.934732 |
| 4 | `runs/baseline_2d81e17f04` | 12.193276911951 | 0.028422556904 | 429 | 0.0 | 0.6 | 10 | 982.727273 |
| 5 | `runs/baseline_b08d5abde7` | 4.773721741251 | 0.011127556506 | 429 | 2.0 | 0.0 | 30 | 1082.934732 |
| 6 | `runs/baseline_cbea9ffa98` | 0.613295965437 | 0.001429594325 | 429 | 2.0 | 0.0 | 10 | 982.727273 |
| 7 | `runs/baseline_4e69d3f2a2` | -0.188888216172 | -0.000440298872 | 429 | 2.0 | 0.6 | 30 | 1082.934732 |
| 8 | `runs/baseline_9e7278d4f7` | -4.349313991986 | -0.010138261054 | 429 | 2.0 | 0.6 | 10 | 982.727273 |
| 9 | `runs/baseline_480826e74d` | -5.478079106000 | -0.012769415166 | 429 | 0.0 | 0.0 | 5 | 696.958042 |
| 10 | `runs/baseline_3a148fc8ca` | -10.440192852049 | -0.024336113874 | 429 | 0.0 | 0.6 | 5 | 696.958042 |
