# S34 Conditional Edge Screen

Generated: `2026-06-28T17:28:26.354047+00:00`

Knowable-at-cross feature conditioning on clean per-anchor outcomes. Tercile cuts derived on calibration only, then applied to the chronological holdout. A CANDIDATE bin is positive on BOTH splits with enough filled trades. Many features are screened -> treat any candidate as a lead to re-test as a fresh registered route, not a green light.

## `ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30`  (cal filled=195, hold filled=292)
- baseline net_bps: cal median=-12.1 mean=-11.1 | hold median=-9.8 mean=-0.9
- **No conditioning is positive on both calibration and holdout.**

## `ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30`  (cal filled=112, hold filled=123)
- baseline net_bps: cal median=-9.8 mean=-10.0 | hold median=-9.2 mean=0.5
- **No conditioning is positive on both calibration and holdout.**

## `ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40`  (cal filled=57, hold filled=68)
- baseline net_bps: cal median=-20.8 mean=-10.2 | hold median=-7.7 mean=3.5
- **No conditioning is positive on both calibration and holdout.**

## `SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30`  (cal filled=39, hold filled=23)
- baseline net_bps: cal median=-13.0 mean=-6.9 | hold median=-6.1 mean=1.2
- **No conditioning is positive on both calibration and holdout.**
