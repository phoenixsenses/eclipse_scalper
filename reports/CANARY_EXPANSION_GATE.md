# Canary Expansion Gate

- ts_utc: 2026-03-05T14:24:09Z
- verdict: HOLD
- window_days: 7
- required_max_top_score: 0.500
- days_observed: 1
- coverage_ok: 0
- score_ok: 0

## Daily Top Causes
| date | top_cause | top_score |
|---|---|---:|
| 2026-03-05 | Insufficient/Noisy Evidence | 1.800 |

## Policy
- GO only if all observed days in window have top_score below threshold and coverage is full.
- HOLD otherwise; keep canary only and continue daily calibration.
