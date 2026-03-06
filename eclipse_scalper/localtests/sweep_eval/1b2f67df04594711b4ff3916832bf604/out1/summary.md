# Sweep Eval Summary

- db: `eclipse_scalper\localtests\sweep_eval\1b2f67df04594711b4ff3916832bf604\db.sqlite`
- symbols: `ETHUSDT`
- slice: `2024-03-01T00:00:00Z` -> `2024-03-01T00:01:00Z`
- strategy: `baseline`
- sort: `pnl_net_sum` (desc)
- grid: `fee_bps=0,0.6;spread_bps=0,10;horizon_sec=5,10`
- total_runs: 8

## Top N

| rank | run_dir | pnl_net_sum | pnl_net_per_fill | fills_count | spread_bps | fee_bps | horizon_sec | avg_adverse_samples |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `eclipse_scalper/localtests/sweep_eval/1b2f67df04594711b4ff3916832bf604/out1/runs/baseline_b1b3735e4a` | 1.700000000000 | 0.085000000000 | 20 | 0.0 | 0.0 | 10 | 9.500000 |
| 2 | `eclipse_scalper/localtests/sweep_eval/1b2f67df04594711b4ff3916832bf604/out1/runs/baseline_b8d95c0bd1` | 1.687760000000 | 0.084388000000 | 20 | 0.0 | 0.6 | 10 | 9.500000 |
| 3 | `eclipse_scalper/localtests/sweep_eval/1b2f67df04594711b4ff3916832bf604/out1/runs/baseline_15a8a959f4` | 1.495150000000 | 0.074757500000 | 20 | 10.0 | 0.0 | 10 | 9.500000 |
