# Funding Extreme Mean-Reversion Research

Generated: `2026-06-29T05:20:21.574686+00:00`

`RESEARCH_ONLY_NO_LIVE_NO_PAPER` - no live or paper state was touched.

## Coverage

- split: `{'cal_months': ['2026-02', '2026-03', '2026-04', '2026-05'], 'hold_months': ['2026-06'], 'rows': 7969, 'cal_rows': 5938, 'hold_rows': 2031}`
- sampling: `{'step_sec': 3600, 'z_lookback': 72, 'fee_bps_side': 3.05, 'symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'], 'horizons_h': [8, 24], 'z_cuts': [1.0, 1.5, 2.0], 'min_n_gate': 40}`

| Table | Symbol | Rows | Start | End |
| --- | --- | ---: | --- | --- |
| `mark_prices` | `BTCUSDT` | 8161835 | 2026-02-15T14:26:28+00:00 | 2026-06-29T05:18:32+00:00 |
| `mark_prices` | `ETHUSDT` | 8161769 | 2026-02-15T14:26:28+00:00 | 2026-06-29T05:18:38+00:00 |
| `mark_prices` | `SOLUSDT` | 2873902 | 2026-04-18T08:41:08+00:00 | 2026-06-29T05:18:43+00:00 |
| `funding_rates` | `BTCUSDT` | 0 | None | None |
| `funding_rates` | `ETHUSDT` | 178 | 2026-02-15T16:00:00+00:00 | 2026-04-13T16:00:00+00:00 |
| `funding_rates` | `SOLUSDT` | 0 | None | None |
| `open_interest` | `BTCUSDT` | 0 | None | None |
| `open_interest` | `ETHUSDT` | 501 | 2026-03-28T12:00:00+00:00 | 2026-04-18T07:53:20.916000+00:00 |
| `open_interest` | `SOLUSDT` | 0 | None | None |
| `spot_prices` | `BTCUSDT` | 57301 | 2026-04-18T08:01:10.083000+00:00 | 2026-06-05T15:59:10.783000+00:00 |
| `spot_prices` | `ETHUSDT` | 58331 | 2026-03-07T16:00:00+00:00 | 2026-06-05T15:59:11.295000+00:00 |
| `spot_prices` | `SOLUSDT` | 57230 | 2026-04-18T08:41:07.354000+00:00 | 2026-06-05T15:59:47.453000+00:00 |

## Consistent Passes

Configs with cal+hold N gate and positive sum/T3R in both splits: `6`

| Config | Cal | Hold |
| --- | --- | --- |
| `funding_abs_z_ge_1_mr_8h` | N=3335 sum=7290.5 med=-2.84 T3R=5371.6 WR=48.8 max_loss=-833.2 | N=928 sum=18319.3 med=9.41 T3R=16412.4 WR=53.0 max_loss=-720.4 |
| `funding_abs_z_ge_1.5_mr_8h` | N=2113 sum=7591.0 med=-0.89 T3R=5672.1 WR=49.7 max_loss=-833.2 | N=622 sum=15350.4 med=13.84 T3R=13443.5 WR=53.9 max_loss=-713.4 |
| `funding_abs_z_ge_2_mr_8h` | N=1020 sum=8117.0 med=1.87 T3R=6319.3 WR=50.6 max_loss=-671.8 | N=333 sum=6942.3 med=13.94 T3R=5281.9 WR=54.1 max_loss=-483.0 |
| `funding_abs_z_ge_1_mr_24h` | N=3335 sum=70359.1 med=13.4 T3R=66672.3 WR=53.1 max_loss=-965.5 | N=926 sum=48462.7 med=54.89 T3R=44971.9 WR=55.8 max_loss=-1016.4 |
| `funding_abs_z_ge_1.5_mr_24h` | N=2113 sum=56467.3 med=22.99 T3R=52780.5 WR=54.9 max_loss=-965.5 | N=622 sum=28990.1 med=72.35 T3R=25499.3 WR=56.6 max_loss=-1016.4 |
| `funding_abs_z_ge_2_mr_24h` | N=1020 sum=38596.7 med=30.44 T3R=34909.9 WR=56.6 max_loss=-965.5 | N=333 sum=14144.2 med=82.3 T3R=10653.3 WR=57.1 max_loss=-1016.4 |

## Ranked Configs

| Rank | Config | Cal | Hold | Positive funding -> SHORT hold | Negative funding -> LONG hold |
| ---: | --- | --- | --- | --- | --- |
| 1 | `funding_abs_z_ge_1_mr_24h` | N=3335 sum=70359.1 med=13.4 T3R=66672.3 WR=53.1 max_loss=-965.5 | N=926 sum=48462.7 med=54.89 T3R=44971.9 WR=55.8 max_loss=-1016.4 | N=505 sum=29217.8 med=60.36 T3R=25779.4 WR=57.2 max_loss=-1016.4 | N=421 sum=19244.9 med=53.12 T3R=15954.7 WR=54.2 max_loss=-978.1 |
| 2 | `funding_abs_z_ge_1.5_mr_24h` | N=2113 sum=56467.3 med=22.99 T3R=52780.5 WR=54.9 max_loss=-965.5 | N=622 sum=28990.1 med=72.35 T3R=25499.3 WR=56.6 max_loss=-1016.4 | N=358 sum=19232.9 med=69.98 T3R=15794.6 WR=56.7 max_loss=-1016.4 | N=264 sum=9757.2 med=77.07 T3R=6467.0 WR=56.4 max_loss=-973.2 |
| 3 | `funding_abs_z_ge_1_mr_8h` | N=3335 sum=7290.5 med=-2.84 T3R=5371.6 WR=48.8 max_loss=-833.2 | N=928 sum=18319.3 med=9.41 T3R=16412.4 WR=53.0 max_loss=-720.4 | N=505 sum=13988.8 med=21.9 T3R=12170.7 WR=56.6 max_loss=-720.4 | N=423 sum=4330.5 med=-4.34 T3R=2524.6 WR=48.7 max_loss=-566.2 |
| 4 | `funding_abs_z_ge_1.5_mr_8h` | N=2113 sum=7591.0 med=-0.89 T3R=5672.1 WR=49.7 max_loss=-833.2 | N=622 sum=15350.4 med=13.84 T3R=13443.5 WR=53.9 max_loss=-713.4 | N=358 sum=10891.0 med=26.06 T3R=9072.8 WR=57.5 max_loss=-713.4 | N=264 sum=4459.5 med=-3.46 T3R=2653.6 WR=48.9 max_loss=-483.0 |
| 5 | `funding_abs_z_ge_2_mr_24h` | N=1020 sum=38596.7 med=30.44 T3R=34909.9 WR=56.6 max_loss=-965.5 | N=333 sum=14144.2 med=82.3 T3R=10653.3 WR=57.1 max_loss=-1016.4 | N=191 sum=5787.0 med=24.77 T3R=2348.7 WR=52.4 max_loss=-1016.4 | N=142 sum=8357.1 med=112.88 T3R=5066.9 WR=63.4 max_loss=-958.4 |
| 6 | `funding_abs_z_ge_2_mr_8h` | N=1020 sum=8117.0 med=1.87 T3R=6319.3 WR=50.6 max_loss=-671.8 | N=333 sum=6942.3 med=13.94 T3R=5281.9 WR=54.1 max_loss=-483.0 | N=191 sum=6018.5 med=26.68 T3R=4375.3 WR=58.6 max_loss=-461.1 | N=142 sum=923.8 med=-6.16 T3R=-563.8 WR=47.9 max_loss=-483.0 |

## Best Holdout By Symbol

Config: `funding_abs_z_ge_1_mr_24h`

| Symbol | Hold metrics |
| --- | --- |
| `BTCUSDT` | N=278 sum=14281.3 med=18.71 T3R=12387.2 WR=52.2 max_loss=-532.1 |
| `ETHUSDT` | N=305 sum=44197.4 med=138.87 T3R=40820.7 WR=70.5 max_loss=-1016.4 |
| `SOLUSDT` | N=343 sum=-10016.0 med=-75.21 T3R=-13454.3 WR=45.8 max_loss=-978.1 |

## Read

- This is a first-pass fresh-signal test, not a promotion.
- Funding is knowable at the snapshot timestamp; forward returns are labels only.
- A pass requires cal+hold consistency, N gate, total/T3R positivity, and later forward shadow.
