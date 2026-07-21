# S34 Question Tests A1-A10

Generated: `2026-07-01T03:24:00.755769+00:00`

Baseline sync<200K: N=100 | WR=58.0% | avg=+26.5 bps | /mo=23.0

## [A1_dow_on_time_exit_sync200]
- Mon: N=18 | WR=66.7% | avg=+32.4 bps | /mo=4.1
- Tue: N=17 | WR=70.6% | avg=+43.8 bps | /mo=4.6
- Wed: N=17 | WR=35.3% | avg=-5.9 bps | /mo=4.1
- Thu: N=15 | WR=53.3% | avg=-4.8 bps | /mo=3.6
- Fri: N=17 | WR=64.7% | avg=+40.6 bps | /mo=4.3
- Sat: N=6 | WR=66.7% | avg=+54.0 bps | /mo=1.5
- Sun: N=10 | WR=50.0% | avg=+48.2 bps | /mo=2.6

## [A2_sync_threshold_curve]
- sync_lt_100k: N=70 | WR=57.1% | avg=+30.0 bps | /mo=16.1
- sync_lt_150k: N=81 | WR=59.3% | avg=+30.3 bps | /mo=18.6
- sync_lt_200k: N=100 | WR=58.0% | avg=+26.5 bps | /mo=23.0
- sync_lt_300k: N=124 | WR=59.7% | avg=+38.4 bps | /mo=28.5
- sync_lt_500k: N=148 | WR=61.5% | avg=+39.0 bps | /mo=34.0
- sync_lt_all: N=167 | WR=61.1% | avg=+38.0 bps | /mo=38.3

## [A3_btc7d_threshold_curve]
- btc7d_lt_0: N=9 | WR=100.0% | avg=+170.6 bps | /mo=2.3
- btc7d_lt_50: N=10 | WR=100.0% | avg=+164.5 bps | /mo=2.5
- btc7d_lt_100: N=10 | WR=100.0% | avg=+164.5 bps | /mo=2.5
- btc7d_lt_200: N=10 | WR=100.0% | avg=+164.5 bps | /mo=2.5
- btc7d_lt_500: N=14 | WR=85.7% | avg=+123.4 bps | /mo=3.6
- btc7d_lt_all: N=21 | WR=76.2% | avg=+82.0 bps | /mo=5.3

## [A4_us_13_14_block]
- exclude_13_only: N=11 | WR=90.9% | avg=+140.1 bps | /mo=2.8
- exclude_14_only: N=10 | WR=90.0% | avg=+151.9 bps | /mo=2.4
- exclude_13_14: N=9 | WR=100.0% | avg=+170.6 bps | /mo=2.3
- blocked_13_14_only: N=3 | WR=33.3% | avg=-3.8 bps | /mo=3.0

## [A5_score_relaxation]
- new_if_long_score_ge2_base_score1: N=6 | WR=83.3% | avg=+98.3 bps | /mo=1.5
- base_score2_silence_score3: N=7 | WR=100.0% | avg=+168.7 bps | /mo=1.8
- current_base_score_ge2: N=9 | WR=100.0% | avg=+170.6 bps | /mo=2.3

## [A6_n2h_relax_2]
- added_only: N=3 | WR=100.0% | avg=+27.8 bps | /mo=3.0
- all_after_relax: N=12 | WR=100.0% | avg=+134.9 bps | /mo=3.1

## [A7_vdepth_relax]
- vdepth_ge_20:
  - all: N=10 | WR=100.0% | avg=+156.7 bps | /mo=2.5
  - added_only: N=1 | WR=100.0% | avg=+31.1 bps | /mo=1.0
- vdepth_ge_25:
  - all: N=10 | WR=100.0% | avg=+156.7 bps | /mo=2.5
  - added_only: N=1 | WR=100.0% | avg=+31.1 bps | /mo=1.0
- vdepth_ge_30:
  - all: N=9 | WR=100.0% | avg=+170.6 bps | /mo=2.3
  - added_only: N=0 | WR=NA | avg=NA bps | /mo=0.0

## [A8_noisy_exit_cost]
- noisy_exit_realized: N=268 | WR=10.4% | avg=-28.4 bps | /mo=61.3
- counterfactual_hold_4h: N=268 | WR=50.7% | avg=-24.2 bps | /mo=61.3
- cost_distribution_bps:
  - early_exit_cost_bps: N=268 | WR=NA | avg=NA bps | /mo=0

## [A9_winner_loser_profiles]
- full_pipeline_current:
  - all: N=9 | WR=100.0% | avg=+170.6 bps | /mo=2.3
  - wins_n: `9`
  - losses_n: `0`
  - winner_profile: `{'hour': {'n': 9, 'mean': 9.89, 'median': 6.0, 'min': 1.0, 'max': 22.0}, 'n2h': {'n': 9, 'mean': 10.0, 'median': 4.0, 'min': 0.0, 'max': 49.0}, 'vdepth_bps': {'n': 9, 'mean': 31.29, 'median': 21.4, 'min': 6.4, 'max': 71.4}, 'btc4h_bps': {'n': 9, 'mean': -66.46, 'median': -67.0, 'min': -220.71, 'max': 23.87}, 'eth4h_bps': {'n': 9, 'mean': -88.19, 'median': -81.68, 'min': -332.22, 'max': 16.68}, 'sync_k': {'n': 9, 'mean': 87268.25, 'median': 79616.77, 'min': 118.77, 'max': 192256.33}, 'running_notional': {'n': 9, 'mean': 282845.0, 'median': 279260.23, 'min': 202140.37, 'max': 429070.88}, 'elapsed_since_first_sec': {'n': 9, 'mean': 90.3, 'median': 52.19, 'min': 6.16, 'max': 230.88}}`
  - loser_profile: `{'hour': {'n': 0, 'mean': None, 'median': None, 'min': None, 'max': None}, 'n2h': {'n': 0, 'mean': None, 'median': None, 'min': None, 'max': None}, 'vdepth_bps': {'n': 0, 'mean': None, 'median': None, 'min': None, 'max': None}, 'btc4h_bps': {'n': 0, 'mean': None, 'median': None, 'min': None, 'max': None}, 'eth4h_bps': {'n': 0, 'mean': None, 'median': None, 'min': None, 'max': None}, 'sync_k': {'n': 0, 'mean': None, 'median': None, 'min': None, 'max': None}, 'running_notional': {'n': 0, 'mean': None, 'median': None, 'min': None, 'max': None}, 'elapsed_since_first_sec': {'n': 0, 'mean': None, 'median': None, 'min': None, 'max': None}}`
  - winner_sessions: `{'ASIA': 5, 'OFF': 1, 'US': 3}`
  - loser_sessions: `{}`
  - winner_dow: `{'Fri': 2, 'Sat': 1, 'Sun': 3, 'Thu': 3}`
  - loser_dow: `{}`
- sync200_baseline:
  - all: N=100 | WR=58.0% | avg=+26.5 bps | /mo=23.0
  - wins_n: `58`
  - losses_n: `42`
  - winner_profile: `{'hour': {'n': 58, 'mean': 11.14, 'median': 11.5, 'min': 0.0, 'max': 23.0}, 'n2h': {'n': 58, 'mean': 5.24, 'median': 2.0, 'min': 0.0, 'max': 66.0}, 'vdepth_bps': {'n': 58, 'mean': 23.36, 'median': 17.85, 'min': 5.2, 'max': 71.4}, 'btc4h_bps': {'n': 58, 'mean': -50.11, 'median': -46.49, 'min': -322.91, 'max': 97.24}, 'eth4h_bps': {'n': 58, 'mean': -76.7, 'median': -71.09, 'min': -475.39, 'max': 84.69}, 'sync_k': {'n': 58, 'mean': 65418.97, 'median': 48139.85, 'min': 0.0, 'max': 194037.45}, 'running_notional': {'n': 58, 'mean': 541349.08, 'median': 298400.27, 'min': 200029.41, 'max': 11909513.02}, 'elapsed_since_first_sec': {'n': 58, 'mean': 95.84, 'median': 63.21, 'min': 6.16, 'max': 275.38}}`
  - loser_profile: `{'hour': {'n': 42, 'mean': 11.79, 'median': 12.5, 'min': 0.0, 'max': 23.0}, 'n2h': {'n': 42, 'mean': 2.76, 'median': 2.0, 'min': 0.0, 'max': 12.0}, 'vdepth_bps': {'n': 42, 'mean': 21.71, 'median': 18.85, 'min': 5.2, 'max': 69.6}, 'btc4h_bps': {'n': 42, 'mean': -13.55, 'median': -14.1, 'min': -343.66, 'max': 184.32}, 'eth4h_bps': {'n': 42, 'mean': -29.68, 'median': -8.29, 'min': -437.27, 'max': 351.84}, 'sync_k': {'n': 42, 'mean': 64644.68, 'median': 40462.92, 'min': 0.0, 'max': 199568.64}, 'running_notional': {'n': 42, 'mean': 346350.49, 'median': 257269.04, 'min': 200025.04, 'max': 1406451.25}, 'elapsed_since_first_sec': {'n': 42, 'mean': 81.03, 'median': 67.31, 'min': 2.14, 'max': 236.26}}`
  - winner_sessions: `{'ASIA': 15, 'EUROPE': 17, 'OFF': 5, 'US': 21}`
  - loser_sessions: `{'ASIA': 12, 'EUROPE': 9, 'OFF': 8, 'US': 13}`
  - winner_dow: `{'Fri': 11, 'Mon': 12, 'Sat': 4, 'Sun': 5, 'Thu': 8, 'Tue': 12, 'Wed': 6}`
  - loser_dow: `{'Fri': 6, 'Mon': 6, 'Sat': 2, 'Sun': 5, 'Thu': 7, 'Tue': 5, 'Wed': 11}`

## [A10_running_notional_bands]
- 200K_300K: N=57 | WR=52.6% | avg=+12.7 bps | /mo=13.1
- 300K_500K: N=28 | WR=71.4% | avg=+53.0 bps | /mo=6.8
- 500K_1M: N=11 | WR=54.5% | avg=+39.3 bps | /mo=2.9
- 1M_plus: N=4 | WR=50.0% | avg=+2.9 bps | /mo=1.6
