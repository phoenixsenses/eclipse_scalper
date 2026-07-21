# S34 State Machine V10 Profit-Lock Follow-Up

- generated_at_utc: `2026-06-30T19:21:56.225290+00:00`
- research_only: `true`
- live_changes: `none`
- rule: `btc1000_dow_score3`

## Core Read

- forward observer: `WAITING_FOR_FORWARD_SAMPLES`, exits=0, delta_sum=0.0 bps
- baseline hold: `N=30 WR=80.0% sum=3471.4 mean=115.7 med=106.8 T3R=2411.7 maxL=-52.0 DD=52.0`
- profit_lock_100_50 hold: `N=30 WR=86.7% sum=3613.0 mean=120.4 med=106.8 T3R=2553.3 maxL=-52.0 DD=52.0`
- short_only_lock hold: `N=30 WR=86.7% sum=3613.0 mean=120.4 med=106.8 T3R=2553.3 maxL=-52.0 DD=52.0`
- long_only_lock hold: `N=30 WR=80.0% sum=3471.4 mean=115.7 med=106.8 T3R=2411.7 maxL=-52.0 DD=52.0`

## Trigger Sensitivity

- trig75_lock25: hold `N=30 WR=86.7% sum=2795.4 mean=93.2 med=27.8 T3R=1791.2 maxL=-52.0 DD=52.0`, exit_rate=0.431
- trig100_lock50: hold `N=30 WR=86.7% sum=3613.0 mean=120.4 med=106.8 T3R=2553.3 maxL=-52.0 DD=52.0`, exit_rate=0.292
- trig125_lock50: hold `N=30 WR=83.3% sum=3561.1 mean=118.7 med=106.8 T3R=2501.4 maxL=-52.0 DD=52.0`, exit_rate=0.125
- trig125_lock75: hold `N=30 WR=83.3% sum=3262.8 mean=108.8 med=69.0 T3R=2203.1 maxL=-52.0 DD=52.0`, exit_rate=0.236
- trig150_lock75: hold `N=30 WR=83.3% sum=3555.7 mean=118.5 med=88.1 T3R=2496.0 maxL=-52.0 DD=52.0`, exit_rate=0.153

## Delay Reality

- delay_0s: hold `N=30 WR=86.7% sum=3613.0 mean=120.4 med=106.8 T3R=2553.3 maxL=-52.0 DD=52.0`, avg_slip=0.0
- delay_2s: hold `N=30 WR=86.7% sum=3612.6 mean=120.4 med=106.8 T3R=2552.9 maxL=-52.0 DD=52.0`, avg_slip=-0.3
- delay_5s: hold `N=30 WR=86.7% sum=3608.9 mean=120.3 med=106.8 T3R=2549.2 maxL=-52.0 DD=52.0`, avg_slip=0.05
- delay_10s: hold `N=30 WR=86.7% sum=3592.3 mean=119.7 med=106.8 T3R=2532.6 maxL=-52.0 DD=52.0`, avg_slip=-1.17
- delay_30s: hold `N=30 WR=86.7% sum=3601.2 mean=120.0 med=106.8 T3R=2541.5 maxL=-52.0 DD=52.0`, avg_slip=-0.4

## False Lock

- exits=21, helped=11, hurt=10
- helped_delta_sum=874.2 bps, hurt_delta_sum=-641.3 bps
- missed_upside_gt_50=5, missed_upside_gt_100=4

## Adverse No-Trigger Case

- triggered hold: `N=20 WR=100.0% sum=3579.5 mean=179.0 med=138.9 T3R=2519.8 maxL=39.9 DD=0.0`
- no_trigger hold: `N=10 WR=60.0% sum=33.5 mean=3.4 med=6.1 T3R=-87.2 maxL=-52.0 DD=52.0`

## Promotion Rule

- status: `SHADOW_RUNNING_NOT_PROMOTABLE`
- reason: Forward sample gate blocks promotion until >=20 shadow exits with positive delta; live logic still needs operator sign-off.

## Full JSON

- `D:\eclipse_scalper\reports\research\s34\S34_STATE_MACHINE_V10_PROFIT_LOCK_FOLLOWUP.json`
