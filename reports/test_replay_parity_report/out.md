# REPLAY PARITY REPORT

## Summary
- sim_count: 3
- live_count: 2
- matched_count: 1
- match_rate_vs_sim: 33.00%
- sim_fill_rate: 50.00%
- live_fill_rate: 50.00%
- fill_rate_delta: +0.0000
- mean_abs_dt_sec: 1.000
- mean_fill_delay_delta_sec: +0.500
- mean_pnl_bps_delta: +0.1000
- mean_adverse_bps_delta: +0.2000

## Matched Sample (first 20)
| symbol | side | dt_sec | sim_pnl_bps | live_pnl_bps | pnl_delta | sim_adv | live_adv | adv_delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|

## Run Summary
- `{'version': 'v1', 'run_type': 'replay_parity_report', 'inputs': {'sim': 'logs/x.jsonl', 'live_db': 'data/paper_trades.db', 'live_table': 'trades', 'match_window_sec': 30.0}, 'metrics': {'sim_count': 3, 'live_count': 2, 'matched_count': 1, 'match_rate_vs_sim': 0.33}, 'artifacts': {'json': 'reports\\test_replay_parity_report\\out.json', 'md': 'reports\\test_replay_parity_report\\out.md'}}`

