# S34 State Machine V9 Profit-Lock Execution Realism

- generated_at_utc: `2026-06-30T19:09:42.926279+00:00`
- research_only: `true`
- live_changes: `none`

## Questions Tested

1. Does profit-lock survive 1/2/5/10/30/60s polling?
2. Does exit delay after lock trigger kill it?
3. Does extra taker/slippage cost kill it?
4. Is partial lock exit better than full lock exit?
5. Is the improvement LONG-side or SHORT-side?
6. Would stop-limit style miss too many exits?
7. Does conservative live-like execution still beat baseline?
8. Does top-3-winner removal still pass?
9. Do folds stay positive?
10. Is this ready for shadow or live?

## Core Results

- baseline: hold `N=30 WR=80.0% sum=3471.4 mean=115.7 med=106.8 T3R=2411.7 maxL=-52.0 DD=52.0`, folds=5/5, hold_top3_removed `N=27 WR=77.8% sum=2411.7 mean=89.3 med=76.7 T3R=1565.9 maxL=-52.0 DD=52.0`, exit_rate=0.0, avg_slip=None
- poll_2s: hold `N=30 WR=86.7% sum=3613.0 mean=120.4 med=106.8 T3R=2553.3 maxL=-52.0 DD=52.0`, folds=5/5, hold_top3_removed `N=27 WR=85.2% sum=2553.3 mean=94.6 med=76.7 T3R=1707.5 maxL=-52.0 DD=52.0`, exit_rate=0.292, avg_slip=0.0
- live_like_conservative_delay2_cost5: hold `N=30 WR=86.7% sum=3597.6 mean=119.9 med=106.8 T3R=2537.9 maxL=-52.0 DD=52.0`, folds=5/5, hold_top3_removed `N=27 WR=85.2% sum=2537.9 mean=94.0 med=76.7 T3R=1692.1 maxL=-52.0 DD=52.0`, exit_rate=0.292, avg_slip=-0.3
- stress_poll5_delay5_cost10: hold `N=30 WR=86.7% sum=3579.4 mean=119.3 med=106.8 T3R=2519.7 maxL=-52.0 DD=52.0`, folds=5/5, hold_top3_removed `N=27 WR=85.2% sum=2519.7 mean=93.3 med=76.7 T3R=1673.9 maxL=-52.0 DD=52.0`, exit_rate=0.306, avg_slip=1.49
- stop_limit_style: hold `N=30 WR=80.0% sum=3471.4 mean=115.7 med=106.8 T3R=2411.7 maxL=-52.0 DD=52.0`, folds=5/5, hold_top3_removed `N=27 WR=77.8% sum=2411.7 mean=89.3 med=76.7 T3R=1565.9 maxL=-52.0 DD=52.0`, exit_rate=0.0, avg_slip=None
- long_only_lock: hold `N=30 WR=80.0% sum=3471.4 mean=115.7 med=106.8 T3R=2411.7 maxL=-52.0 DD=52.0`, folds=5/5, hold_top3_removed `N=27 WR=77.8% sum=2411.7 mean=89.3 med=76.7 T3R=1565.9 maxL=-52.0 DD=52.0`, exit_rate=0.208, avg_slip=0.0
- short_only_lock: hold `N=30 WR=86.7% sum=3613.0 mean=120.4 med=106.8 T3R=2553.3 maxL=-52.0 DD=52.0`, folds=5/5, hold_top3_removed `N=27 WR=85.2% sum=2553.3 mean=94.6 med=76.7 T3R=1707.5 maxL=-52.0 DD=52.0`, exit_rate=0.083, avg_slip=0.0

## Verdict

- pass_shadow: `True`
- pass_live_logic: `False`
- reason: Shadow observer passes if conservative poll/delay/cost remains above baseline; live order-logic still requires forward shadow and operator sign-off.

## Full JSON

- `D:\eclipse_scalper\reports\research\s34\S34_STATE_MACHINE_V9_PROFIT_LOCK_EXECUTION.json`
