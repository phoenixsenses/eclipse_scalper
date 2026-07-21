# Echo-set Tail Forensics (14 tails, descriptive)

_2026-07-20T10:20:17.036695+00:00 · READ-ONLY · causal N=118 · tails=14_

> DESCRIPTIVE, N=14 fragile. Maps a higher-risk-regime HYPOTHESIS for FORWARD sizing/management. NO threshold, NO edge claim. Refines §162/§163 to the echo set.

## A) Regime / time contrast (tail median vs non-tail median; AUC=separation, ~0.5=none)

| feature | tail med | non-tail med | AUC→tail |
|---|---:|---:|---:|
| hour | 9.5 | 14.5 | 0.311 |
| dow | 3 | 4 | 0.422 |
| btc4h | -80.52 | -106.6 | 0.566 |
| btc7d | -157.1 | -29.43 | 0.497 |
| btc1h_pre | -60.14 | -54.64 | 0.429 |
| btc2h_pre | -88.03 | -82.48 | 0.477 |
| eth1h_pre | -73.34 | -74.37 | 0.477 |
| rv_bps | 17.71 | 16.12 | 0.549 |
| sync_k | 2.205e+05 | 2.214e+05 | 0.417 |
| rn | 2.803e+05 | 2.794e+05 | 0.503 |
| spread_pct | 5.797e-06 | 5.926e-06 | 0.392 |
| bid_depth_usd | 2.422e+05 | 1.519e+05 | 0.576 |
| vol_spike_30m | 0.7923 | 0.8834 | 0.445 |

Tail hours (UTC): [0, 0, 1, 2, 6, 6, 6, 13, 13, 14, 14, 15, 15, 16]
Tail dows (0=Mon): [1, 1, 1, 1, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6]

## B) Clustering (are tails serial?)

- Tail timestamps: 2026-02-28 06:15, 2026-02-28 06:31, 2026-03-13 15:53, 2026-04-02 02:09, 2026-04-19 16:53, 2026-04-23 00:08, 2026-06-09 14:04, 2026-06-16 13:08, 2026-06-18 14:53, 2026-06-18 15:36, 2026-06-23 06:07, 2026-06-25 13:31, 2026-06-30 00:28, 2026-07-17 01:45
- Inter-tail gaps (h): [0.3, 321.4, 466.3, 422.7, 79.3, 1141.9, 167.1, 49.7, 0.7, 110.5, 55.4, 106.9, 409.3]
- Median gap: 110.5 h · within 24h: 2 · within 48h: 2 (of 13 gaps)

## C) Post-tail recovery (do the 4h losses mean-revert?)

- median net: 4h=-144.5 · 6h=-131.4 · 8h=-123.05 bps
- of 14 tails: 4 worse at 8h, 10 better at 8h

## Read
- Any feature with AUC clearly >0.6 or <0.4 = a FORWARD risk-regime hypothesis (size down /
  tighten stop in that regime), NOT a T0 filter (separability already showed none survive).
- Clustering: if tails bunch within 24-48h => a 'tail-density' risk-scaler is worth a FORWARD
  arm (reduce size after a recent tail). Post-tail recovery: if 8h≈4h, holding longer doesn't
  save them (mechanical stop territory, §163). All descriptive, N=14 — forward is the judge.
