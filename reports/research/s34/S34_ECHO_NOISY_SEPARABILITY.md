# Echo — Noisy / Tail Causal Separability (can the lookahead be replaced?)

_2026-07-20T10:05:24.583042+00:00 · READ-ONLY · causal set N=118 · noisy=80 · tail_4h=14_

> DESCRIPTIVE separability / hypothesis-generation. AUC~0.5 => lookahead irreplaceable causally (tail irreducible, matches §162). AUC>>0.5 => FORWARD hypothesis only. NO gate selected, NO threshold, NO return claim. Small-n fragile.

## Predict `noisy` (the T+30m lookahead label)

| feature | AUC | |AUC-.5| | med(label=1) | med(label=0) | n1 | n0 |
|---|---:|---:|---:|---:|---:|---:|
| be_ratio | 0.728 | 0.228 | 0.9868 | 0.4452 | 80 | 38 |
| rv_bps | 0.620 | 0.120 | 16.99 | 14.69 | 52 | 23 |
| bv_bps | 0.609 | 0.109 | 5.881 | 5.193 | 52 | 23 |
| amihud | 0.397 | 0.103 | 0.2978 | 0.4165 | 80 | 38 |
| btc4h | 0.406 | 0.094 | -108.7 | -85.63 | 80 | 38 |
| rn | 0.575 | 0.075 | 2.851e+05 | 2.482e+05 | 80 | 38 |
| btc7d | 0.440 | 0.060 | -51.82 | 0 | 80 | 38 |
| prebuildup | 0.560 | 0.060 | 1 | 1 | 80 | 38 |
| dow | 0.444 | 0.056 | 3 | 4 | 80 | 38 |
| sync_k | 0.549 | 0.049 | 2.818e+05 | 1.802e+05 | 80 | 38 |
| score | 0.548 | 0.048 | 3 | 3 | 80 | 38 |
| vd_now | 0.543 | 0.043 | 3 | 3 | 80 | 38 |
| liq_impact_bps_per_M | 0.466 | 0.034 | 985.9 | 1531 | 80 | 38 |
| jump_frac | 0.467 | 0.033 | 0.8419 | 0.8521 | 52 | 23 |
| hour | 0.472 | 0.028 | 14 | 15 | 80 | 38 |
| kyle_lambda | 0.495 | 0.005 | 2.536 | 2.452 | 80 | 38 |
| btc3d | 0.502 | 0.002 | 58.46 | 38.77 | 80 | 38 |

## Predict `tail_4h` (net_4h < -100, the actual disaster)

| feature | AUC | |AUC-.5| | med(label=1) | med(label=0) | n1 | n0 |
|---|---:|---:|---:|---:|---:|---:|
| be_ratio | 0.729 | 0.229 | 1.614 | 0.6118 | 14 | 104 |
| hour | 0.311 | 0.189 | 9.5 | 14.5 | 14 | 104 |
| jump_frac | 0.374 | 0.126 | 0.8086 | 0.8472 | 9 | 66 |
| bv_bps | 0.584 | 0.084 | 8.025 | 5.503 | 9 | 66 |
| sync_k | 0.417 | 0.083 | 2.205e+05 | 2.214e+05 | 14 | 104 |
| dow | 0.422 | 0.078 | 3 | 4 | 14 | 104 |
| btc4h | 0.566 | 0.066 | -80.52 | -106.6 | 14 | 104 |
| amihud | 0.563 | 0.063 | 0.3707 | 0.3606 | 14 | 104 |
| kyle_lambda | 0.451 | 0.049 | 2.038 | 2.619 | 14 | 104 |
| rv_bps | 0.549 | 0.049 | 17.71 | 16.12 | 9 | 66 |
| score | 0.462 | 0.038 | 3 | 3 | 14 | 104 |
| liq_impact_bps_per_M | 0.467 | 0.033 | 1064 | 1177 | 14 | 104 |
| btc3d | 0.530 | 0.030 | 58.66 | 47.33 | 14 | 104 |
| prebuildup | 0.480 | 0.020 | 1 | 1 | 14 | 104 |
| vd_now | 0.492 | 0.008 | 3 | 3 | 14 | 104 |
| btc7d | 0.497 | 0.003 | -157.1 | -29.43 | 14 | 104 |
| rn | 0.503 | 0.003 | 2.803e+05 | 2.794e+05 | 14 | 104 |

## Read
- Max |AUC-.5| near 0 (~<=0.10) across features => noisy/tail is T0-UNSEPARABLE => the
  lookahead cannot be replaced by any causal parameter here; the frozen edge is not causally
  reproducible and its pristine tail-0 was pure hindsight. Consistent with §162 (tail AUC~0.5).
- A feature with |AUC-.5| clearly elevated is a FORWARD HYPOTHESIS ONLY — candidate causal gate
  to record in the forward ledger and validate post-2026-07-20. It is NOT adopted or thresholded
  on this burned sample. (n is small; a single-sample AUC is not evidence of a real gate.)
