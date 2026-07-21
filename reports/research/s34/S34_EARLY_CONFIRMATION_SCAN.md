# S34 Early Confirmation Scan

Generated: 2026-06-20T09:15:18.979085+00:00

Scope: ETH BUY events. This models delayed entry after observing the first 30/60/120 seconds.

No live runner/config changes. First-window data is not used to pretend signal-time entry; entry is moved to the wait timestamp.

## ALL_200K

- Scope SQL: `cluster_notional >= 200000`
- Events: `450`
- Candidate count after train min-N filter: `180`

### Wait Baselines

| Wait | N | Median | Mean | Cum | WR | Top3 Removed | Positive Days | Exits |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 30s | 450 | -8.00 | +8.44 | +3800.13 | 45.1% | +3567.66 | 43/77 | {'BE': 117, 'SL': 107, 'TIME': 41, 'TP': 185} |
| 60s | 450 | -8.00 | +4.41 | +1982.90 | 41.8% | +1776.68 | 37/77 | {'BE': 110, 'SL': 128, 'TIME': 43, 'TP': 169} |
| 120s | 450 | -8.00 | -1.64 | -736.16 | 36.9% | -935.36 | 25/77 | {'BE': 100, 'SL': 160, 'TIME': 41, 'TP': 149} |

### Top OOS Candidates

| Rank | Candidate | Train N | Train Median | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | wait30_ret>=10_mfe>=20_mae>=-40 | 59 | +52.04 | 25 | -8.00 | -10.62 | -265.53 | -434.64 | 6/14 |
| 2 | wait30_ret>=10_mfe>=20_mae>=-25 | 59 | +52.04 | 25 | -8.00 | -10.62 | -265.53 | -434.64 | 6/14 |
| 3 | wait30_ret>=10_mfe>=20_mae>=-15 | 59 | +52.04 | 25 | -8.00 | -10.62 | -265.53 | -434.64 | 6/14 |
| 4 | wait30_ret>=15_mfe>=20_mae>=-40 | 57 | +52.04 | 23 | -15.90 | -10.85 | -249.53 | -418.64 | 6/14 |
| 5 | wait30_ret>=15_mfe>=20_mae>=-25 | 57 | +52.04 | 23 | -15.90 | -10.85 | -249.53 | -418.64 | 6/14 |
| 6 | wait30_ret>=15_mfe>=20_mae>=-15 | 57 | +52.04 | 23 | -15.90 | -10.85 | -249.53 | -418.64 | 6/14 |
| 7 | wait30_ret>=-5_mfe>=20_mae>=-40 | 61 | +22.98 | 25 | -8.00 | -10.62 | -265.53 | -434.64 | 6/14 |
| 8 | wait30_ret>=-5_mfe>=20_mae>=-25 | 61 | +22.98 | 25 | -8.00 | -10.62 | -265.53 | -434.64 | 6/14 |

### Real-Fill Parity

| Candidate | Total | Real Fill | No Fill | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wait30_ret>=10_mfe>=20_mae>=-40 | 84 | 18 | 66 (78.6%) | 14 | -3.12 | +3.26 | +45.67 | -118.63 | 5/8 |
| wait30_ret>=10_mfe>=20_mae>=-25 | 84 | 18 | 66 (78.6%) | 14 | -3.12 | +3.26 | +45.67 | -118.63 | 5/8 |
| wait30_ret>=10_mfe>=20_mae>=-15 | 84 | 18 | 66 (78.6%) | 14 | -3.12 | +3.26 | +45.67 | -118.63 | 5/8 |
| wait30_ret>=15_mfe>=20_mae>=-40 | 80 | 16 | 64 (80.0%) | 12 | +2.52 | +5.49 | +65.87 | -98.43 | 5/8 |
| wait30_ret>=15_mfe>=20_mae>=-25 | 80 | 16 | 64 (80.0%) | 12 | +2.52 | +5.49 | +65.87 | -98.43 | 5/8 |
| wait30_ret>=15_mfe>=20_mae>=-15 | 80 | 16 | 64 (80.0%) | 12 | +2.52 | +5.49 | +65.87 | -98.43 | 5/8 |
| wait30_ret>=-5_mfe>=20_mae>=-40 | 86 | 18 | 68 (79.1%) | 14 | -3.12 | +3.26 | +45.67 | -118.63 | 5/8 |
| wait30_ret>=-5_mfe>=20_mae>=-25 | 86 | 18 | 68 (79.1%) | 14 | -3.12 | +3.26 | +45.67 | -118.63 | 5/8 |

## 500K_DAYTREND

- Scope SQL: `cluster_notional >= 500000 and day_trend_bps >= 0`
- Events: `97`
- Candidate count after train min-N filter: `165`

### Wait Baselines

| Wait | N | Median | Mean | Cum | WR | Top3 Removed | Positive Days | Exits |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 30s | 97 | +23.87 | +17.28 | +1675.82 | 55.7% | +1478.04 | 24/38 | {'BE': 24, 'SL': 16, 'TIME': 10, 'TP': 47} |
| 60s | 97 | +21.24 | +14.38 | +1394.59 | 52.6% | +1200.44 | 24/38 | {'BE': 19, 'SL': 21, 'TIME': 11, 'TP': 46} |
| 120s | 97 | -8.00 | +8.27 | +802.13 | 45.4% | +614.83 | 20/38 | {'BE': 23, 'SL': 25, 'TIME': 8, 'TP': 41} |

### Top OOS Candidates

| Rank | Candidate | Train N | Train Median | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | wait60_ret>=10_mfe>=15_mae>=-40 | 18 | +52.53 | 21 | -8.00 | +0.28 | +5.86 | -170.26 | 6/11 |
| 2 | wait60_ret>=10_mfe>=15_mae>=-25 | 18 | +52.53 | 21 | -8.00 | +0.28 | +5.86 | -170.26 | 6/11 |
| 3 | wait60_ret>=10_mfe>=15_mae>=-15 | 18 | +52.53 | 21 | -8.00 | +0.28 | +5.86 | -170.26 | 6/11 |
| 4 | wait60_ret>=15_mfe>=0_mae>=-40 | 16 | +52.53 | 20 | -8.00 | -2.34 | -46.72 | -222.84 | 5/10 |
| 5 | wait60_ret>=15_mfe>=0_mae>=-25 | 16 | +52.53 | 20 | -8.00 | -2.34 | -46.72 | -222.84 | 5/10 |
| 6 | wait60_ret>=15_mfe>=0_mae>=-15 | 16 | +52.53 | 20 | -8.00 | -2.34 | -46.72 | -222.84 | 5/10 |
| 7 | wait60_ret>=15_mfe>=10_mae>=-40 | 16 | +52.53 | 20 | -8.00 | -2.34 | -46.72 | -222.84 | 5/10 |
| 8 | wait60_ret>=15_mfe>=10_mae>=-25 | 16 | +52.53 | 20 | -8.00 | -2.34 | -46.72 | -222.84 | 5/10 |

### Real-Fill Parity

| Candidate | Total | Real Fill | No Fill | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wait60_ret>=10_mfe>=15_mae>=-40 | 39 | 15 | 24 (61.5%) | 14 | +4.54 | +0.58 | +8.14 | -171.34 | 4/9 |
| wait60_ret>=10_mfe>=15_mae>=-25 | 39 | 15 | 24 (61.5%) | 14 | +4.54 | +0.58 | +8.14 | -171.34 | 4/9 |
| wait60_ret>=10_mfe>=15_mae>=-15 | 39 | 15 | 24 (61.5%) | 14 | +4.54 | +0.58 | +8.14 | -171.34 | 4/9 |
| wait60_ret>=15_mfe>=0_mae>=-40 | 36 | 13 | 23 (63.9%) | 13 | -8.39 | -4.02 | -52.28 | -226.13 | 3/8 |
| wait60_ret>=15_mfe>=0_mae>=-25 | 36 | 13 | 23 (63.9%) | 13 | -8.39 | -4.02 | -52.28 | -226.13 | 3/8 |
| wait60_ret>=15_mfe>=0_mae>=-15 | 36 | 13 | 23 (63.9%) | 13 | -8.39 | -4.02 | -52.28 | -226.13 | 3/8 |
| wait60_ret>=15_mfe>=10_mae>=-40 | 36 | 13 | 23 (63.9%) | 13 | -8.39 | -4.02 | -52.28 | -226.13 | 3/8 |
| wait60_ret>=15_mfe>=10_mae>=-25 | 36 | 13 | 23 (63.9%) | 13 | -8.39 | -4.02 | -52.28 | -226.13 | 3/8 |

## Read

This is a confirmation-delay research scan. Positives are not immediately live-tradeable because the same surface was swept; they require a separate pre-registered forward rule.
