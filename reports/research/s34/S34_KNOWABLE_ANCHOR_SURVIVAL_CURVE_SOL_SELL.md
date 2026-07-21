# S34 Knowable-Anchor Survival Curve

Generated: `2026-06-28T11:38:29.726242+00:00`

RESEARCH_ONLY. Calibration split only; holdout is frozen separately and not used for selection beyond the deterministic rule encoded in the script.

## Split Freeze

- source_db: `D:\eclipse_scalper\data\microstructure.db`
- source_db_size_bytes: `646985273344`
- holdout_bucket_ids_sha256: `41458e5a1c8253b5b614853a13035403605cb2df164dae114cf21d9342cc8f0b`
- calibration_buckets: `84`
- holdout_buckets: `36`

## Selected Config

- primary_config_id: `NONE`
- shortlist: `NONE`

## Calibration Grid

| X | Dir | H | Accel | Filled N | No-fill % | Median | Mean | WR | Top3W Removed | Mark CF N | Mark CF Med | Mark CF Mean | CF no-fill med |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100K | LONG | 30 | accelerating | 52 | 26.8% | -6.1 | -7.3 | 23.1% | -426.5 | 71 | -6.1 | -6.9 | -6.3 |
| 100K | LONG | 30 | all | 57 | 26.0% | -6.1 | -8.3 | 22.8% | -521.3 | 77 | -6.2 | -7.8 | -6.7 |
| 100K | LONG | 30 | decelerating | 5 | 16.7% | -12.9 | -19.0 | 20.0% | -81.8 | 6 | -14.1 | -17.6 | -17.0 |
| 100K | LONG | 60 | accelerating | 52 | 26.8% | -8.4 | -8.8 | 25.0% | -534.2 | 71 | -8.3 | -7.7 | -8.9 |
| 100K | LONG | 60 | all | 57 | 26.0% | -8.5 | -10.2 | 24.6% | -659.8 | 77 | -8.9 | -8.9 | -9.2 |
| 100K | LONG | 60 | decelerating | 5 | 16.7% | -25.2 | -25.1 | 20.0% | -88.8 | 6 | -19.2 | -23.0 | -17.5 |
| 100K | LONG | 120 | accelerating | 52 | 26.8% | -7.4 | -9.3 | 25.0% | -594.9 | 71 | -6.2 | -8.3 | -5.7 |
| 100K | LONG | 120 | all | 57 | 26.0% | -7.5 | -10.6 | 22.8% | -710.2 | 77 | -8.6 | -9.1 | -7.3 |
| 100K | LONG | 120 | decelerating | 5 | 16.7% | -16.3 | -23.1 | 0.0% | -84.3 | 6 | -11.7 | -18.3 | -8.9 |
| 100K | SHORT | 30 | accelerating | 52 | 26.8% | -8.8 | -7.5 | 17.3% | -473.5 | 71 | -6.1 | -5.3 | -5.9 |
| 100K | SHORT | 30 | all | 57 | 26.0% | -8.8 | -6.5 | 19.3% | -468.3 | 77 | -6.0 | -4.4 | -5.5 |
| 100K | SHORT | 30 | decelerating | 5 | 16.7% | -2.0 | 3.9 | 40.0% | -30.1 | 6 | 1.9 | 5.4 | 4.8 |
| 100K | SHORT | 60 | accelerating | 52 | 26.8% | -6.1 | -6.0 | 28.8% | -412.2 | 71 | -3.9 | -4.5 | -3.3 |
| 100K | SHORT | 60 | all | 57 | 26.0% | -6.1 | -4.6 | 31.6% | -381.0 | 77 | -3.3 | -3.3 | -3.0 |
| 100K | SHORT | 60 | decelerating | 5 | 16.7% | 10.2 | 10.1 | 60.0% | -18.7 | 6 | 7.0 | 10.8 | 5.3 |
| 100K | SHORT | 120 | accelerating | 52 | 26.8% | -7.5 | -5.5 | 36.5% | -434.7 | 71 | -6.0 | -3.9 | -6.5 |
| 100K | SHORT | 120 | all | 57 | 26.0% | -7.3 | -4.3 | 38.6% | -407.3 | 77 | -3.6 | -3.1 | -4.9 |
| 100K | SHORT | 120 | decelerating | 5 | 16.7% | 1.2 | 8.0 | 60.0% | -15.5 | 6 | -0.5 | 6.1 | -3.3 |
| 200K | LONG | 30 | accelerating | 29 | 29.3% | -8.9 | -10.6 | 20.7% | -342.1 | 41 | -8.1 | -8.9 | -8.7 |
| 200K | LONG | 30 | all | 31 | 27.9% | -8.9 | -10.9 | 19.4% | -372.1 | 43 | -8.1 | -9.1 | -8.7 |
| 200K | LONG | 30 | decelerating | 2 | 0.0% | -15.0 | -15.0 | 0.0% | -30.1 | 2 | -13.6 | -13.6 | None |
| 200K | LONG | 60 | accelerating | 29 | 29.3% | -10.6 | -8.3 | 27.6% | -299.6 | 41 | -9.2 | -7.1 | -9.7 |
| 200K | LONG | 60 | all | 31 | 27.9% | -10.6 | -8.7 | 25.8% | -328.7 | 43 | -9.2 | -7.3 | -9.7 |
| 200K | LONG | 60 | decelerating | 2 | 0.0% | -14.5 | -14.5 | 0.0% | -29.0 | 2 | -11.7 | -11.7 | None |
| 200K | LONG | 120 | accelerating | 29 | 29.3% | -7.5 | -6.0 | 31.0% | -290.3 | 41 | -6.2 | -5.7 | -6.0 |
| 200K | LONG | 120 | all | 31 | 27.9% | -7.5 | -6.7 | 29.0% | -325.2 | 43 | -6.2 | -6.2 | -6.0 |
| 200K | LONG | 120 | decelerating | 2 | 0.0% | -17.5 | -17.5 | 0.0% | -34.9 | 2 | -16.0 | -16.0 | None |
| 200K | SHORT | 30 | accelerating | 29 | 29.3% | -6.1 | -4.3 | 31.0% | -199.8 | 41 | -4.1 | -3.3 | -3.5 |
| 200K | SHORT | 30 | all | 31 | 27.9% | -6.1 | -4.0 | 32.3% | -198.9 | 43 | -4.1 | -3.1 | -3.5 |
| 200K | SHORT | 30 | decelerating | 2 | 0.0% | 0.5 | 0.5 | 50.0% | 1.0 | 2 | 1.4 | 1.4 | None |
| 200K | SHORT | 60 | accelerating | 29 | 29.3% | -4.6 | -6.6 | 27.6% | -245.6 | 41 | -3.0 | -5.1 | -2.5 |
| 200K | SHORT | 60 | all | 31 | 27.9% | -4.6 | -6.2 | 29.0% | -245.6 | 43 | -3.0 | -4.9 | -2.5 |
| 200K | SHORT | 60 | decelerating | 2 | 0.0% | -0.0 | -0.0 | 50.0% | -0.1 | 2 | -0.5 | -0.5 | None |
| 200K | SHORT | 120 | accelerating | 29 | 29.3% | -7.5 | -8.9 | 37.9% | -323.7 | 41 | -6.0 | -6.5 | -6.2 |
| 200K | SHORT | 120 | all | 31 | 27.9% | -7.5 | -8.2 | 38.7% | -317.9 | 43 | -6.0 | -6.0 | -6.2 |
| 200K | SHORT | 120 | decelerating | 2 | 0.0% | 2.9 | 2.9 | 50.0% | 5.8 | 2 | 3.8 | 3.8 | None |

## Read

- Anchor is first real-time-knowable crossing of running notional X inside the 300s S34 bucket.
- `accelerating` means current trailing liquidation-rate window exceeds the prior trailing window at the anchor timestamp.
- All entry features are passed through the Feature Availability Contract gate before evaluation.
- A positive calibration row is not a live recommendation; only the frozen holdout pass can produce at most `PAPER_CANDIDATE`.
