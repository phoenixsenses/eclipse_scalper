# S54 Entropy + Basis

Entropy dist: `{'min': 0.6930330845273658, 'p25': 1.3643920215464056, 'p50': 1.7223749848376364, 'p75': 2.117586617052527, 'p90': 2.239317918506396}`
Basis dist: `{'min': -0.023645336233497825, 'p25': -0.0022254078092027907, 'p50': 0.0005295786806355256, 'p75': 0.0029973198569489745, 'p90': 0.007736387027134514}`
Pearson r(basis, 60s short return) = `0.3025`

| filter | h | N | WR | mean_ret_bps |
|---|---:|---:|---:|---:|
| entropy < p25 | 60s | 4 | 75.0% | +3.55 |
| entropy < p25 | 120s | 4 | 75.0% | -0.84 |
| entropy < p50 | 60s | 8 | 75.0% | +1.29 |
| entropy < p50 | 120s | 8 | 50.0% | -4.56 |
| entropy > p75 | 60s | 4 | 0.0% | -9.85 |
| entropy > p75 | 120s | 4 | 25.0% | -24.52 |
| basis > 0.5 bps | 60s | 30 | 56.7% | +5.12 |
| basis > 0.5 bps | 120s | 30 | 66.7% | +5.33 |
| basis > 1.0 bps | 60s | 30 | 56.7% | +5.12 |
| basis > 1.0 bps | 120s | 30 | 66.7% | +5.33 |
| basis > 2.0 bps | 60s | 30 | 56.7% | +5.12 |
| basis > 2.0 bps | 120s | 30 | 66.7% | +5.33 |
| basis < 0 | 60s | 21 | 52.4% | -4.60 |
| basis < 0 | 120s | 21 | 23.8% | -15.64 |

## Entropy Cohort @60s

| cohort | N | WR | mean_ret_bps |
|---|---:|---:|---:|
| high | 8 | 37.5% | -1.86 |
| low | 9 | 66.7% | +0.07 |

## Geometry Type @60s

| type | N | WR | mean_ret_bps |
|---|---:|---:|---:|
| type_L | 15 | 53.3% | +0.01 |
| type_S | 14 | 64.3% | +7.14 |
| type_U | 21 | 61.9% | -0.32 |
| type_V | 23 | 60.9% | +8.85 |

## Cluster Label @60s

| cluster | N | WR | mean_ret_bps |
|---|---:|---:|---:|
| cluster_A_high_quality | 5 | 80.0% | +8.91 |
| cluster_B_medium_quality | 45 | 62.2% | +4.82 |
| cluster_C_low_quality | 23 | 52.2% | +1.53 |