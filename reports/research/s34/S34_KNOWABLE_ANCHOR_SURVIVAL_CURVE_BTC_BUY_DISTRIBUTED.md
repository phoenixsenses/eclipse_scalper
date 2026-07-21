# S34 Knowable-Anchor Survival Curve

Generated: `2026-06-28T11:38:01.146213+00:00`

RESEARCH_ONLY. Calibration split only; holdout is frozen separately and not used for selection beyond the deterministic rule encoded in the script.

## Split Freeze

- source_db: `D:\eclipse_scalper\data\microstructure.db`
- source_db_size_bytes: `646983225344`
- holdout_bucket_ids_sha256: `15bf8e8c66a57134582b91fe639583ea3b206110bf260845a2256230e32a1203`
- calibration_buckets: `37`
- holdout_buckets: `16`

## Selected Config

- primary_config_id: `NONE`
- shortlist: `NONE`

## Calibration Grid

| X | Dir | H | Accel | Filled N | No-fill % | Median | Mean | WR | Top3W Removed | Mark CF N | Mark CF Med | Mark CF Mean | CF no-fill med |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000K | LONG | 30 | accelerating | 19 | 36.7% | -5.5 | -8.6 | 21.1% | -188.3 | 30 | -5.3 | -8.8 | -13.0 |
| 1000K | LONG | 30 | all | 23 | 37.8% | -6.6 | -8.5 | 21.7% | -220.0 | 37 | -6.3 | -9.0 | -13.0 |
| 1000K | LONG | 30 | decelerating | 4 | 42.9% | -7.0 | -7.9 | 25.0% | -20.2 | 7 | -7.3 | -10.0 | -20.1 |
| 1000K | LONG | 60 | accelerating | 18 | 40.0% | -7.4 | -10.6 | 16.7% | -233.6 | 30 | -7.4 | -8.5 | -7.9 |
| 1000K | LONG | 60 | all | 22 | 40.5% | -7.7 | -9.9 | 18.2% | -259.6 | 37 | -8.1 | -9.1 | -9.1 |
| 1000K | LONG | 60 | decelerating | 4 | 42.9% | -8.4 | -6.5 | 25.0% | -12.2 | 7 | -9.9 | -11.7 | -21.9 |
| 1000K | LONG | 120 | accelerating | 19 | 36.7% | -7.1 | -8.1 | 31.6% | -220.2 | 30 | -11.3 | -3.7 | -18.5 |
| 1000K | LONG | 120 | all | 23 | 37.8% | -7.1 | -9.5 | 26.1% | -285.2 | 37 | -12.3 | -6.7 | -19.2 |
| 1000K | LONG | 120 | decelerating | 4 | 42.9% | -14.8 | -16.2 | 0.0% | -33.3 | 7 | -24.5 | -19.6 | -25.8 |
| 1000K | SHORT | 30 | accelerating | 19 | 36.7% | -6.7 | -3.6 | 26.3% | -133.0 | 30 | -6.9 | -3.4 | 0.8 |
| 1000K | SHORT | 30 | all | 23 | 37.8% | -5.7 | -3.7 | 26.1% | -150.2 | 37 | -5.9 | -3.2 | 0.8 |
| 1000K | SHORT | 30 | decelerating | 4 | 42.9% | -5.2 | -4.3 | 25.0% | -14.7 | 7 | -4.9 | -2.2 | 7.9 |
| 1000K | SHORT | 60 | accelerating | 18 | 40.0% | -4.8 | -1.6 | 33.3% | -118.7 | 30 | -4.8 | -3.7 | -4.3 |
| 1000K | SHORT | 60 | all | 22 | 40.5% | -4.5 | -2.4 | 27.3% | -141.5 | 37 | -4.1 | -3.1 | -3.1 |
| 1000K | SHORT | 60 | decelerating | 4 | 42.9% | -3.8 | -5.7 | 0.0% | -15.2 | 7 | -2.3 | -0.5 | 9.7 |
| 1000K | SHORT | 120 | accelerating | 19 | 36.7% | -5.1 | -4.1 | 42.1% | -157.8 | 30 | -0.9 | -8.5 | 6.3 |
| 1000K | SHORT | 120 | all | 23 | 37.8% | -5.1 | -2.7 | 43.5% | -152.5 | 37 | 0.1 | -5.5 | 7.0 |
| 1000K | SHORT | 120 | decelerating | 4 | 42.9% | 2.5 | 4.0 | 50.0% | -10.1 | 7 | 12.3 | 7.4 | 13.6 |

## Read

- Anchor is first real-time-knowable crossing of running notional X inside the 300s S34 bucket.
- `accelerating` means current trailing liquidation-rate window exceeds the prior trailing window at the anchor timestamp.
- All entry features are passed through the Feature Availability Contract gate before evaluation.
- A positive calibration row is not a live recommendation; only the frozen holdout pass can produce at most `PAPER_CANDIDATE`.
