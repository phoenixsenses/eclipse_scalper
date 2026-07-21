# S34 Signal Path Quality - 2026-06-07 / 06-11 / 06-14 / 06-15

Scope: read-only signal-path analysis over existing `microstructure.db`. No production runner/config changes.

Model caveat: this measures mark-price path quality after regime-pass ETH BUY liquidation signals. It is not a validation decision and does not include full live paper execution details such as bid/ask fill, adverse selection, max-open interactions, cooldown, and exact runner cursor sequencing.

## Per-Day Path Quality

| day | rule | n | MFE mean | MAE mean | BE hit | TP hit | SL touch | BTC 15m mean | post-5m BUY liq |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-07 | 200K_TP60 | 7 | 150.34 | -28.62 | 100.0% | 85.7% | 28.6% | 37.75 | 9.22M |
| 2026-06-07 | 50K_TP120 | 30 | 90.19 | -49.02 | 66.7% | 16.7% | 46.7% | 12.13 | 2.22M |
| 2026-06-11 | 200K_TP60 | 6 | 79.12 | -49.19 | 66.7% | 33.3% | 50.0% | 11.54 | 1.27M |
| 2026-06-11 | 50K_TP120 | 13 | 61.82 | -45.61 | 61.5% | 7.7% | 53.8% | 6.38 | 0.69M |
| 2026-06-14 | 200K_TP60 | 3 | 105.58 | -31.33 | 66.7% | 66.7% | 33.3% | 40.76 | 2.38M |
| 2026-06-14 | 50K_TP120 | 4 | 87.79 | -43.09 | 50.0% | 50.0% | 50.0% | 31.69 | 0.82M |
| 2026-06-15 | 200K_TP60 | 14 | 151.18 | -20.58 | 85.7% | 71.4% | 21.4% | 14.98 | 2.32M |
| 2026-06-15 | 50K_TP120 | 20 | 121.99 | -31.02 | 70.0% | 40.0% | 30.0% | 7.10 | 2.26M |

## Candidate Separators

| rule | filter | pass/fail n | pass TP | fail TP | pass MFE | fail MFE | pass MAE | fail MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50K_TP120 | MFE >= 60 | 29/38 | 55.2% | 0.0% | 184.15 | 25.26 | -14.04 | -64.46 |
| 50K_TP120 | MAE > -20 | 26/41 | 50.0% | 7.3% | 165.97 | 48.42 | -7.63 | -64.83 |
| 50K_TP120 | BTC 15m >= 0 | 30/37 | 43.3% | 8.1% | 159.50 | 40.96 | -15.96 | -64.26 |
| 50K_TP120 | post-5m BUY liq >= 1M | 17/50 | 41.2% | 18.0% | 157.27 | 72.54 | -22.20 | -49.58 |
| 50K_TP120 | first 5m no deep pullback | 51/16 | 29.4% | 6.2% | 105.99 | 55.93 | -34.16 | -69.64 |
| 200K_TP60 | MFE >= 60 | 20/10 | 100.0% | 0.0% | 185.91 | 24.22 | -9.56 | -68.64 |
| 200K_TP60 | MAE > -20 | 18/12 | 94.4% | 25.0% | 194.91 | 37.66 | -5.04 | -65.57 |
| 200K_TP60 | BTC 15m >= 0 | 18/12 | 94.4% | 25.0% | 193.90 | 39.18 | -11.28 | -56.21 |
| 200K_TP60 | post-5m BUY liq >= 1M | 19/11 | 73.7% | 54.5% | 150.29 | 100.45 | -22.39 | -41.10 |
| 200K_TP60 | first 5m no deep pullback | 24/6 | 75.0% | 33.3% | 154.91 | 40.42 | -19.70 | -67.48 |

## Read

- The bad 06-11 result is not because the signal had no upward move. It had positive MFE, but the path was dirty: high MAE and high SL-touch rate.
- `50K_TP120` needs a cleaner path because TP is far. It is vulnerable when early pullback is deep or continuation is slow.
- `200K_TP60` looks more robust in this small sample because the stronger cluster threshold plus shorter target converts more MFE into TP before path noise takes it back.
- BTC 15m confirmation is a promising separator in this sample, especially for 200K/TP60, but it is partly look-forward as currently measured and must not be used live in this form.
- MFE-based filters are diagnostic only. They cannot be used live because MFE is future information.

## Practical Next Step

Build only no-lookahead candidates from the diagnostic hints:

- stronger entry cluster (`200K`) rather than trying to rescue every `50K` event
- require immediate follow-through proxy available at entry or shortly after, such as post-signal 1m/2m continuation before entry, not future MFE
- test BTC confirmation using only BTC move already known at signal time, or a short confirmation delay explicitly modeled
- compare TP60 vs TP120 under the same no-lookahead filter

Artifact:

- JSON: `reports/research/s34/S34_SIGNAL_PATH_QUALITY_2026-06-07_15.json`
