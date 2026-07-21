# S34 Regime Separation - 2026-06-07 / 06-11 / 06-14 / 06-15

Scope: read-only analysis over existing `microstructure.db`. No production runner/config changes.

Model caveat: this separates market context and fixed-horizon behavior. It is not a validation decision and does not replace live paper results with real bid/ask fills, adverse selection, cooldown, and risk gates.

## Daily Context

| day | label | ETH trend | ETH range | BTC trend | BTC/ETH align | ETH BUY liq | ETH agg trades |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 2026-06-07 | continuation_up | 7.70% | 9.94% | 4.03% | aligned_up | 78.86M | 1,915,841 |
| 2026-06-11 | continuation_up | 3.20% | 4.43% | 3.44% | aligned_up | 15.54M | 1,349,692 |
| 2026-06-14 | continuation_up | 2.64% | 4.54% | 2.01% | aligned_up | 19.86M | 693,841 |
| 2026-06-15 | continuation_up | 4.08% | 8.12% | 0.88% | aligned_up | 70.98M | 1,395,473 |

## Forward Move After Regime-Pass BUY-Liq Signals

| day | rule | n | 5m mean | 15m mean | 30m mean | 60m mean | 15m positive rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-07 | 50K_TP120 | 30 | 11.10 | 18.53 | 13.50 | 20.12 | 43.33% |
| 2026-06-07 | 200K_TP60 | 7 | 70.53 | 62.90 | 57.67 | 45.21 | 71.43% |
| 2026-06-11 | 50K_TP120 | 13 | 18.36 | 19.83 | 13.58 | 9.94 | 38.46% |
| 2026-06-11 | 200K_TP60 | 6 | 21.88 | 25.20 | 33.73 | 17.82 | 50.00% |
| 2026-06-14 | 50K_TP120 | 4 | 13.61 | 38.16 | 38.35 | 39.07 | 50.00% |
| 2026-06-14 | 200K_TP60 | 3 | 14.49 | 46.97 | 37.86 | 59.47 | 66.67% |
| 2026-06-15 | 50K_TP120 | 20 | 18.75 | 36.92 | 58.38 | 65.94 | 60.00% |
| 2026-06-15 | 200K_TP60 | 14 | 31.01 | 60.45 | 74.43 | 86.45 | 71.43% |

## Read

- The simple daily context split is not enough. All four days pass the same broad continuation-up label.
- 06-11 was bad for the 50K route despite positive forward means because the route distribution produced too many SL/BE outcomes. The issue is not "no move"; it is path/stop management.
- 06-15 was strongest because forward response strengthened with horizon, especially 30m/60m, and the larger 200K signals had cleaner follow-through.
- 200K/TP60 looks less fragile in this small slice because it needs fewer clean continuation points than 50K/TP120.

## Next Separator To Test

The next useful separation is not daily trend. It is per-signal path quality:

- pre-entry chase/adverse selection
- max favorable excursion before first adverse drawdown
- time-to-BE and time-to-TP
- local liquidation cluster density in the 5 minutes after entry
- BTC same-window confirmation at 5m/15m

Artifact:

- JSON: `reports/research/s34/S34_REGIME_SEPARATION_2026-06-07_15.json`
