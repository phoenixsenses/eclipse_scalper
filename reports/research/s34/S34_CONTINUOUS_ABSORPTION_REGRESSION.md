# S34 Continuous Absorption Regression

Generated: `2026-06-28T23:30:55.934699+00:00`

Research-only. Uses cross-asset pooled rows; no live/paper/executor changes.

## Sample

- Overall: N=541 sum=-2614.6 med=4.4 T3R=-4497.3 max_loss=-507.2 tail<-100=101
- Calibration: N=166 sum=1752.9 med=-6.9 T3R=608.8 max_loss=-285.3 tail<-100=11
- Holdout: N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90

## Raw Feature Correlation vs Net Bps

| Feature | N | Pearson all | Spearman all | Pearson cal | Pearson hold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `book_imbalance` | 541 | -0.052 | -0.042 | -0.005 | -0.072 |
| `bid_depth_usd` | 541 | -0.044 | -0.109 | 0.026 | -0.095 |
| `ask_depth_usd` | 541 | 0.007 | -0.038 | 0.128 | -0.009 |
| `total_top_depth_usd` | 541 | -0.011 | -0.092 | 0.120 | -0.036 |
| `spread_bps` | 541 | 0.025 | 0.030 | -0.041 | 0.054 |
| `vdepth_bps` | 541 | 0.047 | 0.012 | 0.109 | 0.041 |
| `running_notional` | 541 | 0.008 | 0.078 | -0.035 | 0.013 |
| `running_liq_count` | 541 | -0.017 | 0.013 | -0.080 | -0.004 |
| `running_accel` | 541 | 0.003 | 0.057 | -0.061 | 0.011 |

## Route-Normalized Feature Correlation

Each feature is z-scored inside its own `route_id` before correlation. This tests whether a route has unusually high/low absorption relative to its own baseline.

| Feature | N | Pearson all | Spearman all | Pearson cal | Pearson hold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `book_imbalance_route_z` | 537 | -0.052 | -0.051 | -0.011 | -0.070 |
| `bid_depth_usd_route_z` | 537 | -0.100 | -0.138 | -0.060 | -0.127 |
| `ask_depth_usd_route_z` | 537 | -0.042 | -0.070 | -0.047 | -0.038 |
| `total_top_depth_usd_route_z` | 537 | -0.119 | -0.167 | -0.072 | -0.142 |
| `spread_bps_route_z` | 537 | 0.093 | 0.138 | 0.123 | 0.221 |
| `vdepth_bps_route_z` | 537 | -0.015 | -0.028 | 0.062 | -0.034 |
| `running_notional_route_z` | 537 | 0.083 | 0.077 | 0.010 | 0.102 |
| `running_liq_count_route_z` | 537 | -0.054 | -0.017 | -0.196 | -0.021 |
| `running_accel_route_z` | 537 | 0.075 | 0.073 | 0.011 | 0.089 |

## Quartile Extremes

| Feature | Low quartile | High quartile | Delta high-low T3R |
| --- | --- | --- | ---: |
| `running_notional` | N=137 sum=-2893.0 med=-25.5 T3R=-4754.5 max_loss=-507.2 tail<-100=30 | N=136 sum=1477.3 med=25.5 T3R=513.3 max_loss=-417.1 tail<-100=19 | 5267.8 |
| `running_accel` | N=136 sum=-2395.9 med=-9.3 T3R=-4189.5 max_loss=-507.2 tail<-100=27 | N=136 sum=412.2 med=21.3 T3R=-681.2 max_loss=-494.0 tail<-100=23 | 3508.3 |
| `vdepth_bps` | N=137 sum=-2041.1 med=-10.4 T3R=-3879.1 max_loss=-439.8 tail<-100=29 | N=136 sum=1281.9 med=-0.7 T3R=-579.7 max_loss=-348.8 tail<-100=28 | 3299.4 |
| `spread_bps` | N=138 sum=-1470.3 med=9.1 T3R=-2363.4 max_loss=-417.1 tail<-100=20 | N=136 sum=-106.9 med=1.1 T3R=-1968.5 max_loss=-484.2 tail<-100=28 | 394.9 |
| `ask_depth_usd` | N=138 sum=1248.2 med=-0.9 T3R=-46.4 max_loss=-460.6 tail<-100=24 | N=136 sum=1472.9 med=12.4 T3R=-365.1 max_loss=-402.3 tail<-100=23 | -318.7 |
| `book_imbalance` | N=136 sum=897.0 med=6.9 T3R=-985.6 max_loss=-507.2 tail<-100=32 | N=137 sum=-2885.4 med=-6.7 T3R=-3967.8 max_loss=-494.0 tail<-100=32 | -2982.2 |
| `running_liq_count` | N=155 sum=1355.9 med=6.0 T3R=-423.6 max_loss=-494.0 tail<-100=28 | N=139 sum=-3186.3 med=0.9 T3R=-4030.8 max_loss=-439.8 tail<-100=22 | -3607.2 |
| `bid_depth_usd` | N=136 sum=3796.5 med=35.7 T3R=1913.8 max_loss=-507.2 tail<-100=25 | N=136 sum=-2634.5 med=-5.6 T3R=-3674.5 max_loss=-439.8 tail<-100=28 | -5588.3 |
| `total_top_depth_usd` | N=136 sum=5439.6 med=22.9 T3R=3578.1 max_loss=-507.2 tail<-100=17 | N=137 sum=-1486.3 med=0.9 T3R=-3324.3 max_loss=-439.8 tail<-100=25 | -6902.4 |

## Route-Normalized Quartile Extremes

| Feature | Low route-z quartile | High route-z quartile | Delta high-low T3R |
| --- | --- | --- | ---: |
| `running_notional_route_z` | N=135 sum=-4379.3 med=-11.4 T3R=-5344.2 max_loss=-507.2 tail<-100=28 | N=135 sum=2513.2 med=18.1 T3R=1112.3 max_loss=-484.2 tail<-100=19 | 6456.5 |
| `running_accel_route_z` | N=135 sum=-328.5 med=0.9 T3R=-2122.1 max_loss=-507.2 tail<-100=24 | N=135 sum=2882.1 med=20.4 T3R=1481.3 max_loss=-484.2 tail<-100=18 | 3603.4 |
| `spread_bps_route_z` | N=135 sum=-561.2 med=-11.5 T3R=-1705.3 max_loss=-305.8 tail<-100=13 | N=135 sum=3114.7 med=36.3 T3R=1742.2 max_loss=-494.0 tail<-100=24 | 3447.5 |
| `vdepth_bps_route_z` | N=135 sum=1256.8 med=12.0 T3R=138.4 max_loss=-460.6 tail<-100=19 | N=135 sum=1048.7 med=17.1 T3R=-357.5 max_loss=-494.0 tail<-100=25 | -495.9 |
| `running_liq_count_route_z` | N=137 sum=116.8 med=-10.0 T3R=-1207.3 max_loss=-494.0 tail<-100=28 | N=136 sum=-931.6 med=2.3 T3R=-2516.3 max_loss=-484.2 tail<-100=21 | -1309.0 |
| `ask_depth_usd_route_z` | N=135 sum=747.5 med=-2.9 T3R=-547.1 max_loss=-460.6 tail<-100=24 | N=135 sum=-1736.3 med=-10.0 T3R=-3589.6 max_loss=-402.3 tail<-100=31 | -3042.5 |
| `book_imbalance_route_z` | N=135 sum=1809.8 med=20.8 T3R=-72.8 max_loss=-507.2 tail<-100=30 | N=135 sum=-2556.2 med=-10.4 T3R=-3638.7 max_loss=-494.0 tail<-100=32 | -3565.9 |
| `bid_depth_usd_route_z` | N=135 sum=3292.7 med=26.0 T3R=1410.1 max_loss=-507.2 tail<-100=24 | N=135 sum=-5881.1 med=-16.8 T3R=-6896.0 max_loss=-484.2 tail<-100=34 | -8306.1 |
| `total_top_depth_usd_route_z` | N=135 sum=5424.9 med=30.7 T3R=3794.1 max_loss=-507.2 tail<-100=15 | N=135 sum=-3437.1 med=-33.3 T3R=-4958.3 max_loss=-484.2 tail<-100=31 | -8752.4 |

## By Symbol Snapshot

| Symbol | Summary | book_imbalance r | bid_depth r | spread r | vdepth r |
| --- | --- | ---: | ---: | ---: | ---: |
| `BTCUSDT` | N=134 sum=-379.4 med=17.9 T3R=-1272.5 max_loss=-417.1 tail<-100=20 | 0.143 | 0.067 | 0.113 | 0.065 |
| `ETHUSDT` | N=256 sum=-1868.2 med=-0.3 T3R=-3706.2 max_loss=-507.2 tail<-100=53 | -0.117 | -0.113 | 0.152 | 0.045 |
| `SOLUSDT` | N=151 sum=-367.0 med=-13.1 T3R=-2228.6 max_loss=-484.2 tail<-100=28 | -0.082 | -0.227 | 0.148 | 0.037 |

## Read

- Treat this as diagnostics, not model selection. The sample is pooled across route definitions.
- A sign flip between raw and route-normalized features means the pooled binary gate is probably mixing route identity with absorption.
- A feature is interesting only if the sign is directionally stable in calibration and holdout.
