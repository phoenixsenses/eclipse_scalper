# S34 State Machine V5 Development Suite

- generated_at_utc: `2026-06-30T18:49:07.284678+00:00`
- research_only: `true`
- primary_config: `btc1000_dow_score3`
- primary_hold: `{'n': 30, 'wr': 0.833, 'sum': 3471.4, 'mean': 115.7, 'median': 106.8, 't3r': 2411.8, 'max_loss': -52.0, 'max_win': 370.0, 'max_dd_bps': 52.0}`

## Executive Read

- Final live config remains strongest conservative lane: hold N=30, WR=0.833, sum=3471.4bps, T3R=2411.8bps.
- Best development leads are management/navigation, not immediate live filter changes.
- Frequency expansion candidates are research-only; no in-sample expansion should be promoted without a separate gauntlet.

## Top Tables

### BTC Dose Response
- btc1000k_dow_score3: N=30 WR=83.3% sum=3471.4 med=106.8 T3R=2411.8 maxLoss=-52.0 DD=52.0
- btc750k_dow_score3: N=32 WR=78.1% sum=3359.0 med=72.8 T3R=2299.4 maxLoss=-52.0 DD=70.5
- btc500k_dow_score3: N=35 WR=71.4% sum=3210.4 med=35.8 T3R=2150.7 maxLoss=-166.4 DD=178.4
- btc1500k_dow_score3: N=26 WR=84.6% sum=3024.8 med=91.6 T3R=1965.2 maxLoss=-52.0 DD=52.0
- btc300k_dow_score3: N=42 WR=66.7% sum=2935.2 med=35.5 T3R=1899.8 maxLoss=-184.8 DD=206.1
- btc1250k_dow_score3: N=11 WR=100.0% sum=1584.4 med=106.5 T3R=600.0 maxLoss=15.2 DD=0.0

### Frequency Expansion Counterfactuals
- final_btc1000_score3_dow: N=30 WR=83.3% sum=3471.4 med=106.8 T3R=2411.8 maxLoss=-52.0 DD=52.0
- looser_btc750_score3_dow: N=32 WR=78.1% sum=3359.0 med=72.8 T3R=2299.4 maxLoss=-52.0 DD=70.5
- looser_btc500_score3_dow: N=35 WR=71.4% sum=3210.4 med=35.8 T3R=2150.7 maxLoss=-166.4 DD=178.4
- include_europe_long: N=33 WR=81.8% sum=3113.7 med=106.5 T3R=2109.5 maxLoss=-452.3 DD=464.3
- include_noisy_short: N=61 WR=63.9% sum=3013.9 med=22.9 T3R=1939.1 maxLoss=-162.2 DD=176.6
- score2_btc1000_dow: N=36 WR=77.8% sum=2958.5 med=60.1 T3R=1914.1 maxLoss=-169.5 DD=169.5

### Score Ablation
- full_score: N=30 WR=83.3% sum=3471.4 med=106.8 T3R=2411.8 maxLoss=-52.0 DD=52.0
- drop_us_session: N=26 WR=88.5% sum=3342.9 med=122.5 T3R=2283.2 maxLoss=-52.0 DD=52.0
- drop_vdepth: N=27 WR=81.5% sum=3281.3 med=107.1 T3R=2221.7 maxLoss=-52.0 DD=52.0
- drop_sync200: N=24 WR=87.5% sum=3209.4 med=122.5 T3R=2149.8 maxLoss=-40.2 DD=40.2
- drop_btc4h_down: N=20 WR=90.0% sum=2754.0 med=126.5 T3R=1709.7 maxLoss=-40.2 DD=40.2
- drop_sil_eth: N=24 WR=79.2% sum=2680.4 med=111.2 T3R=1636.1 maxLoss=-52.0 DD=52.0
- drop_n2h: N=20 WR=85.0% sum=2625.9 med=126.5 T3R=1581.6 maxLoss=-51.4 DD=51.4

### DOW Robustness
- with_dow_filter: N=30 WR=83.3% sum=3471.4 med=106.8 T3R=2411.8 maxLoss=-52.0 DD=52.0
- without_dow_filter: N=36 WR=80.6% sum=3625.9 med=72.8 T3R=2566.3 maxLoss=-78.4 DD=98.2
- excluded_counterfactual: N=8 WR=75.0% sum=443.4 med=19.1 T3R=-57.6 maxLoss=-78.4 DD=98.2

## 15 Question Results

1. LONG/SHORT anatomy: see `side_anatomy`.
2. Follow-on sequence: see `sequence_model`.
3. Silence cause: see `silence_cause`.
4. SHORT replace: see `lifecycle.conflict_policy_duel`.
5. Position lifecycle transitions: see `lifecycle.transition_summaries`.
6. Exit timing: see `exit_timing`.
7. Early danger monitor: see `early_danger`.
8. Bull-run adaptation: see `bull_run_adaptation`.
9. Regime recovery: see `regime_recovery`.
10. Score ablation: see `score_ablation`.
11. BTC threshold dose response: see `btc_dose_response`.
12. DOW robustness: see `dow_robustness`.
13. Frequency expansion: see `frequency_expansion`.
14. Navigation permission: see `navigation_permission`.
15. Tail neighborhood: see `tail_neighborhood`.

## Full JSON

- `D:\eclipse_scalper\reports\research\s34\S34_STATE_MACHINE_V5_DEV_SUITE.json`
