# S34 Knowable-Anchor Survival Curve

Generated: `2026-06-28T11:37:40.480221+00:00`

RESEARCH_ONLY. Calibration split only; holdout is frozen separately and not used for selection beyond the deterministic rule encoded in the script.

## Split Freeze

- source_db: `D:\eclipse_scalper\data\microstructure.db`
- source_db_size_bytes: `646980108288`
- holdout_bucket_ids_sha256: `667db844b7090b4140927364f8c304f5eba25bd7df4bbabe212be669f1b876a6`
- calibration_buckets: `85`
- holdout_buckets: `37`

## Selected Config

- primary_config_id: `NONE`
- shortlist: `NONE`

## Calibration Grid

| X | Dir | H | Accel | Filled N | No-fill % | Median | Mean | WR | Top3W Removed | Mark CF N | Mark CF Med | Mark CF Mean | CF no-fill med |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100K | LONG | 30 | accelerating | 53 | 29.3% | -6.1 | -3.7 | 28.3% | -269.2 | 75 | -6.1 | -2.5 | -7.9 |
| 100K | LONG | 30 | all | 55 | 29.5% | -6.1 | -3.8 | 29.1% | -280.2 | 78 | -5.2 | -2.2 | -7.7 |
| 100K | LONG | 30 | decelerating | 2 | 33.3% | -5.5 | -5.5 | 50.0% | -11.0 | 3 | 2.6 | 3.1 | 2.6 |
| 100K | LONG | 60 | accelerating | 53 | 29.3% | -8.8 | -2.7 | 32.1% | -279.9 | 75 | -5.9 | -2.4 | -9.9 |
| 100K | LONG | 60 | all | 55 | 29.5% | -8.8 | -3.0 | 32.7% | -300.0 | 78 | -6.0 | -2.5 | -10.7 |
| 100K | LONG | 60 | decelerating | 2 | 33.3% | -10.1 | -10.1 | 50.0% | -20.2 | 3 | -10.9 | -5.0 | -10.9 |
| 100K | LONG | 120 | accelerating | 53 | 29.3% | -8.4 | -4.9 | 24.5% | -390.9 | 75 | -8.4 | -3.8 | -13.8 |
| 100K | LONG | 120 | all | 55 | 29.5% | -8.4 | -5.4 | 25.5% | -428.9 | 78 | -8.8 | -4.3 | -13.9 |
| 100K | LONG | 120 | decelerating | 2 | 33.3% | -19.0 | -19.0 | 50.0% | -38.0 | 3 | -22.7 | -16.3 | -22.7 |
| 100K | SHORT | 30 | accelerating | 53 | 29.3% | -8.9 | -11.2 | 5.7% | -597.6 | 75 | -6.1 | -9.7 | -4.3 |
| 100K | SHORT | 30 | all | 55 | 29.5% | -8.9 | -11.1 | 7.3% | -618.8 | 78 | -7.0 | -10.0 | -4.5 |
| 100K | SHORT | 30 | decelerating | 2 | 33.3% | -9.6 | -9.6 | 50.0% | -19.2 | 3 | -14.8 | -15.3 | -14.8 |
| 100K | SHORT | 60 | accelerating | 53 | 29.3% | -6.1 | -12.2 | 15.1% | -657.6 | 75 | -6.3 | -9.8 | -2.3 |
| 100K | SHORT | 60 | all | 55 | 29.5% | -6.1 | -11.9 | 16.4% | -687.4 | 78 | -6.2 | -9.7 | -1.5 |
| 100K | SHORT | 60 | decelerating | 2 | 33.3% | -5.1 | -5.1 | 50.0% | -10.1 | 3 | -1.3 | -7.2 | -1.3 |
| 100K | SHORT | 120 | accelerating | 53 | 29.3% | -6.1 | -10.0 | 30.2% | -575.6 | 75 | -3.8 | -8.4 | 1.6 |
| 100K | SHORT | 120 | all | 55 | 29.5% | -6.1 | -9.5 | 30.9% | -592.4 | 78 | -3.4 | -7.9 | 1.7 |
| 100K | SHORT | 120 | decelerating | 2 | 33.3% | 3.9 | 3.9 | 50.0% | 7.7 | 3 | 10.5 | 4.1 | 10.5 |
| 200K | LONG | 30 | accelerating | 39 | 29.1% | -7.3 | -6.7 | 23.1% | -294.3 | 55 | -7.7 | -5.2 | -8.7 |
| 200K | LONG | 30 | all | 40 | 28.6% | -6.7 | -6.5 | 22.5% | -295.0 | 56 | -7.3 | -5.0 | -8.7 |
| 200K | LONG | 30 | decelerating | 1 | 0.0% | -0.7 | -0.7 | 0.0% | -0.7 | 1 | 4.6 | 4.6 | None |
| 200K | LONG | 60 | accelerating | 39 | 29.1% | -9.6 | -6.4 | 23.1% | -382.6 | 55 | -10.2 | -6.5 | -18.0 |
| 200K | LONG | 60 | all | 40 | 28.6% | -10.1 | -6.7 | 22.5% | -398.2 | 56 | -10.2 | -6.6 | -18.0 |
| 200K | LONG | 60 | decelerating | 1 | 0.0% | -15.5 | -15.5 | 0.0% | -15.5 | 1 | -10.8 | -10.8 | None |
| 200K | LONG | 120 | accelerating | 39 | 29.1% | -13.1 | -6.5 | 17.9% | -422.4 | 55 | -13.4 | -6.8 | -17.8 |
| 200K | LONG | 120 | all | 40 | 28.6% | -13.2 | -7.2 | 17.5% | -455.5 | 56 | -13.8 | -7.2 | -17.8 |
| 200K | LONG | 120 | decelerating | 1 | 0.0% | -33.1 | -33.1 | 0.0% | -33.1 | 1 | -29.7 | -29.7 | None |
| 200K | SHORT | 30 | accelerating | 39 | 29.1% | -7.3 | -8.2 | 15.4% | -346.8 | 55 | -4.5 | -7.0 | -3.5 |
| 200K | SHORT | 30 | all | 40 | 28.6% | -8.0 | -8.3 | 15.0% | -361.0 | 56 | -4.9 | -7.2 | -3.5 |
| 200K | SHORT | 30 | decelerating | 1 | 0.0% | -14.2 | -14.2 | 0.0% | -14.2 | 1 | -16.8 | -16.8 | None |
| 200K | SHORT | 60 | accelerating | 39 | 29.1% | -4.9 | -8.4 | 25.6% | -405.8 | 55 | -2.0 | -5.7 | 5.8 |
| 200K | SHORT | 60 | all | 40 | 28.6% | -4.4 | -8.2 | 27.5% | -405.2 | 56 | -2.0 | -5.6 | 5.8 |
| 200K | SHORT | 60 | decelerating | 1 | 0.0% | 0.6 | 0.6 | 100.0% | 0.6 | 1 | -1.4 | -1.4 | None |
| 200K | SHORT | 120 | accelerating | 39 | 29.1% | -1.8 | -8.4 | 43.6% | -396.6 | 55 | 1.2 | -5.4 | 5.6 |
| 200K | SHORT | 120 | all | 40 | 28.6% | -1.7 | -7.7 | 45.0% | -381.1 | 56 | 1.6 | -5.0 | 5.6 |
| 200K | SHORT | 120 | decelerating | 1 | 0.0% | 18.2 | 18.2 | 100.0% | 18.2 | 1 | 17.5 | 17.5 | None |

## Read

- Anchor is first real-time-knowable crossing of running notional X inside the 300s S34 bucket.
- `accelerating` means current trailing liquidation-rate window exceeds the prior trailing window at the anchor timestamp.
- All entry features are passed through the Feature Availability Contract gate before evaluation.
- A positive calibration row is not a live recommendation; only the frozen holdout pass can produce at most `PAPER_CANDIDATE`.
