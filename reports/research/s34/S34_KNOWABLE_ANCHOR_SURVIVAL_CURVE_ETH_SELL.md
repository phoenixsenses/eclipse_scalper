# S34 Knowable-Anchor Survival Curve

Generated: `2026-06-28T11:38:20.355347+00:00`

RESEARCH_ONLY. Calibration split only; holdout is frozen separately and not used for selection beyond the deterministic rule encoded in the script.

## Split Freeze

- source_db: `D:\eclipse_scalper\data\microstructure.db`
- source_db_size_bytes: `646983225344`
- holdout_bucket_ids_sha256: `02c0016d3db96ea396edf78cdc9b17a5a67adfa13cb49597468955413d0ff292`
- calibration_buckets: `165`
- holdout_buckets: `71`

## Selected Config

- primary_config_id: `NONE`
- shortlist: `NONE`

## Calibration Grid

| X | Dir | H | Accel | Filled N | No-fill % | Median | Mean | WR | Top3W Removed | Mark CF N | Mark CF Med | Mark CF Mean | CF no-fill med |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500K | LONG | 30 | accelerating | 59 | 61.4% | -5.0 | -5.7 | 30.5% | -391.7 | 153 | -5.8 | -6.8 | -5.9 |
| 500K | LONG | 30 | all | 62 | 61.3% | -4.8 | -5.5 | 32.3% | -395.7 | 160 | -5.5 | -6.0 | -5.7 |
| 500K | LONG | 30 | decelerating | 3 | 57.1% | 3.2 | -1.3 | 66.7% | -4.0 | 7 | -4.4 | 11.0 | -0.6 |
| 500K | LONG | 60 | accelerating | 58 | 62.1% | -6.0 | -6.7 | 39.7% | -436.4 | 153 | -5.5 | -7.2 | -5.5 |
| 500K | LONG | 60 | all | 60 | 62.5% | -5.2 | -6.2 | 41.7% | -422.2 | 160 | -5.3 | -6.7 | -5.4 |
| 500K | LONG | 60 | decelerating | 2 | 71.4% | 7.1 | 7.1 | 100.0% | 14.2 | 7 | -3.9 | 4.7 | -5.4 |
| 500K | LONG | 120 | accelerating | 59 | 61.4% | -2.6 | -5.6 | 40.7% | -408.2 | 153 | -4.4 | -6.8 | -4.9 |
| 500K | LONG | 120 | all | 61 | 61.9% | -2.0 | -5.1 | 42.6% | -384.3 | 160 | -4.6 | -6.4 | -5.0 |
| 500K | LONG | 120 | decelerating | 2 | 71.4% | 11.9 | 11.9 | 100.0% | 23.8 | 7 | -12.6 | 1.4 | -12.8 |
| 500K | SHORT | 30 | accelerating | 59 | 61.4% | -7.3 | -6.6 | 25.4% | -472.9 | 153 | -6.4 | -5.4 | -6.3 |
| 500K | SHORT | 30 | all | 62 | 61.3% | -7.5 | -6.9 | 24.2% | -506.6 | 160 | -6.7 | -6.2 | -6.5 |
| 500K | SHORT | 30 | decelerating | 3 | 57.1% | -15.5 | -11.2 | 0.0% | -33.7 | 7 | -7.8 | -23.2 | -11.6 |
| 500K | SHORT | 60 | accelerating | 58 | 62.1% | -6.3 | -5.6 | 25.9% | -430.1 | 153 | -6.7 | -5.0 | -6.7 |
| 500K | SHORT | 60 | all | 60 | 62.5% | -7.1 | -6.1 | 25.0% | -468.9 | 160 | -6.9 | -5.5 | -6.8 |
| 500K | SHORT | 60 | decelerating | 2 | 71.4% | -19.4 | -19.4 | 0.0% | -38.8 | 7 | -8.3 | -16.9 | -6.8 |
| 500K | SHORT | 120 | accelerating | 59 | 61.4% | -9.7 | -6.7 | 27.1% | -539.5 | 153 | -7.8 | -5.4 | -7.3 |
| 500K | SHORT | 120 | all | 61 | 61.9% | -10.3 | -7.3 | 26.2% | -587.9 | 160 | -7.6 | -5.8 | -7.2 |
| 500K | SHORT | 120 | decelerating | 2 | 71.4% | -24.2 | -24.2 | 0.0% | -48.4 | 7 | 0.4 | -13.6 | 0.6 |
| 1000K | LONG | 30 | accelerating | 35 | 50.0% | -6.6 | -8.2 | 25.7% | -331.8 | 70 | -5.5 | -8.1 | -5.4 |
| 1000K | LONG | 30 | all | 37 | 49.3% | -7.5 | -8.7 | 24.3% | -365.3 | 73 | -5.8 | -8.4 | -5.6 |
| 1000K | LONG | 30 | decelerating | 2 | 33.3% | -16.8 | -16.8 | 0.0% | -33.5 | 3 | -12.7 | -14.1 | -6.1 |
| 1000K | LONG | 60 | accelerating | 35 | 50.0% | -4.7 | -5.4 | 40.0% | -240.1 | 70 | -6.8 | -8.5 | -8.8 |
| 1000K | LONG | 60 | all | 37 | 49.3% | -4.7 | -5.1 | 40.5% | -241.3 | 73 | -6.9 | -8.5 | -8.8 |
| 1000K | LONG | 60 | decelerating | 2 | 33.3% | -0.4 | -0.4 | 50.0% | -0.7 | 3 | -15.4 | -7.6 | -18.6 |
| 1000K | LONG | 120 | accelerating | 35 | 50.0% | -1.7 | -4.0 | 45.7% | -223.3 | 70 | -5.6 | -6.6 | -8.1 |
| 1000K | LONG | 120 | all | 37 | 49.3% | -1.7 | -4.3 | 45.9% | -244.2 | 73 | -5.2 | -6.3 | -7.9 |
| 1000K | LONG | 120 | decelerating | 2 | 33.3% | -10.5 | -10.5 | 50.0% | -20.9 | 3 | 8.4 | -0.0 | 19.7 |
| 1000K | SHORT | 30 | accelerating | 35 | 50.0% | -5.7 | -4.1 | 31.4% | -237.5 | 70 | -6.7 | -4.1 | -6.8 |
| 1000K | SHORT | 30 | all | 37 | 49.3% | -4.8 | -3.6 | 35.1% | -228.5 | 73 | -6.4 | -3.8 | -6.6 |
| 1000K | SHORT | 30 | decelerating | 2 | 33.3% | 4.5 | 4.5 | 100.0% | 9.0 | 3 | 0.5 | 1.9 | -6.1 |
| 1000K | SHORT | 60 | accelerating | 35 | 50.0% | -7.6 | -6.9 | 28.6% | -341.3 | 70 | -5.4 | -3.7 | -3.4 |
| 1000K | SHORT | 60 | all | 37 | 49.3% | -7.6 | -7.2 | 29.7% | -365.2 | 73 | -5.3 | -3.7 | -3.4 |
| 1000K | SHORT | 60 | decelerating | 2 | 33.3% | -11.9 | -11.9 | 50.0% | -23.9 | 3 | 3.2 | -4.6 | 6.4 |
| 1000K | SHORT | 120 | accelerating | 35 | 50.0% | -10.6 | -8.3 | 25.7% | -428.2 | 70 | -6.6 | -5.6 | -4.1 |
| 1000K | SHORT | 120 | all | 37 | 49.3% | -10.6 | -8.0 | 27.0% | -431.8 | 73 | -7.0 | -5.9 | -4.3 |
| 1000K | SHORT | 120 | decelerating | 2 | 33.3% | -1.8 | -1.8 | 50.0% | -3.6 | 3 | -20.6 | -12.2 | -31.9 |

## Read

- Anchor is first real-time-knowable crossing of running notional X inside the 300s S34 bucket.
- `accelerating` means current trailing liquidation-rate window exceeds the prior trailing window at the anchor timestamp.
- All entry features are passed through the Feature Availability Contract gate before evaluation.
- A positive calibration row is not a live recommendation; only the frozen holdout pass can produce at most `PAPER_CANDIDATE`.
