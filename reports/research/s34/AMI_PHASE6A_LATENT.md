# AMI Faz 6A — Latent State Discovery

> 2026-07-02 20:51 UTC — prereg `E-LATENT6A-001` hash `a059e89d80175704` (model oncesi frozen). SONUC: **REJECTED**

## features
```json
[
 "ret5m",
 "rv30m",
 "ofi10m",
 "stress10m",
 "buyliq10m",
 "fund_vel_1h",
 "spread5m",
 "trades10m",
 "ret1h"
]
```

## dropped_features
```json
[]
```

## changepoints
```json
{
 "n_cp": 517,
 "per_day": 7.88
}
```

## k_selection
```json
{
 "2": {
  "seed_ari": 0.844,
  "occ_ok": true
 },
 "3": {
  "seed_ari": 0.83,
  "occ_ok": true
 },
 "4": {
  "seed_ari": 0.851,
  "occ_ok": true
 },
 "5": {
  "seed_ari": 0.871,
  "occ_ok": false
 },
 "6": {
  "seed_ari": 0.895,
  "occ_ok": false
 }
}
```

## chosen_k
```json
4
```

## perturb_ari
```json
0.991
```

## hmm_crosscheck_ari
```json
0.578
```

## artifact_hash
```json
"4df1bf45d6dd2bbc"
```

## occupancy_expl
```json
[
 0.603,
 0.273,
 0.071,
 0.053
]
```

## occupancy_val
```json
[
 0.158,
 0.569,
 0.01,
 0.263
]
```

## occ_ratio_val
```json
[
 0.26,
 2.08,
 0.14,
 4.99
]
```

## avg_duration_min_expl
```json
{
 "0": 215.9,
 "1": 107.7,
 "2": 52.8,
 "3": 26.7
}
```

## transition_matrix_expl
```json
[
 [
  0.977,
  0.008,
  0.011,
  0.004
 ],
 [
  0.02,
  0.954,
  0.0,
  0.027
 ],
 [
  0.093,
  0.001,
  0.905,
  0.001
 ],
 [
  0.038,
  0.147,
  0.002,
  0.812
 ]
]
```

## transition_matrix_val
```json
[
 [
  0.517,
  0.254,
  0.011,
  0.218
 ],
 [
  0.074,
  0.792,
  0.009,
  0.124
 ],
 [
  0.304,
  0.261,
  0.109,
  0.326
 ],
 [
  0.117,
  0.287,
  0.007,
  0.589
 ]
]
```

## transition_entropy
```json
0.456
```

## val_transition_corr
```json
0.69
```

## state_profiles
```json
{
 "LS-001": {
  "ret5m": 0.12,
  "rv30m": -0.05,
  "ofi10m": 0.01,
  "stress10m": 0.03,
  "buyliq10m": 0.0,
  "fund_vel_1h": 0.05,
  "spread5m": 0.29,
  "trades10m": -0.08,
  "ret1h": 0.16
 },
 "LS-002": {
  "ret5m": 0.15,
  "rv30m": 0.64,
  "ofi10m": 0.14,
  "stress10m": 3.87,
  "buyliq10m": 6.0,
  "fund_vel_1h": 0.05,
  "spread5m": 0.3,
  "trades10m": 0.97,
  "ret1h": 0.23
 },
 "LS-003": {
  "ret5m": -1.06,
  "rv30m": 1.82,
  "ofi10m": -0.13,
  "stress10m": 0.01,
  "buyliq10m": 0.0,
  "fund_vel_1h": -0.44,
  "spread5m": 0.78,
  "trades10m": 0.49,
  "ret1h": -1.86
 },
 "LS-004": {
  "ret5m": -0.34,
  "rv30m": 0.23,
  "ofi10m": -0.31,
  "stress10m": 5.87,
  "buyliq10m": 0.01,
  "fund_vel_1h": -0.09,
  "spread5m": 0.53,
  "trades10m": 0.42,
  "ret1h": -0.33
 }
}
```

## session_dist
```json
{
 "LS-001": {
  "US": 0.32,
  "EUROPE": 0.27,
  "OFF": 0.42
 },
 "LS-002": {
  "US": 0.37,
  "EUROPE": 0.23,
  "OFF": 0.41
 },
 "LS-003": {
  "US": 0.45,
  "EUROPE": 0.15,
  "OFF": 0.41
 },
 "LS-004": {
  "US": 0.19,
  "EUROPE": 0.27,
  "OFF": 0.53
 }
}
```

## dq_dist
```json
{
 "LS-001": 0.83,
 "LS-002": 0.759,
 "LS-003": 0.807,
 "LS-004": 0.836
}
```

## taxonomy_overlap
```json
{
 "LS-001": {
  "cascade_active_pct": 0.001,
  "downtrend_pct": 0.19
 },
 "LS-002": {
  "cascade_active_pct": 0.12,
  "downtrend_pct": 0.274
 },
 "LS-003": {
  "cascade_active_pct": 0.0,
  "downtrend_pct": 0.9
 },
 "LS-004": {
  "cascade_active_pct": 0.152,
  "downtrend_pct": 0.457
 }
}
```

## unknown_rate
```json
0.0
```

## outcome_eval
```json
{
 "baseline_all": {
  "n": 394,
  "mean_fwd6h": -7.3
 },
 "LS-001": {
  "n": 63,
  "mean_fwd1h": -1.4,
  "mean_fwd6h": -14.7,
  "wr6h": 0.476
 },
 "LS-002": {
  "n": 225,
  "mean_fwd1h": -3.3,
  "mean_fwd6h": -4.2,
  "wr6h": 0.471
 },
 "LS-003": {
  "n": 4,
  "mean_fwd1h": 22.5,
  "mean_fwd6h": 71.6,
  "wr6h": 0.5
 },
 "LS-004": {
  "n": 104,
  "mean_fwd1h": 3.1,
  "mean_fwd6h": -7.3,
  "wr6h": 0.538
 }
}
```

Durust statuler: software-correct ✓ · replay-validated (seed'li) ✓ · latent-state unstable/null · chronological-validation failed/na · alpha-incremental: outcome_eval bolumune bakiniz (ayri katman) · forward-validating ✗ · **operationally FORBIDDEN** (LIVE/SIZING/PORTFOLIO yasak)

*Runner: `python -m ami.latent.discovery`*