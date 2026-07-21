# S34 V Engine Next Research Results

Generated: `2026-06-29T08:17:52.672199+00:00`

Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`. No live executor, leverage, size, order logic, or .env changes.

## Tail-Injected Compounding

| Scenario | Mode | End Equity | Multiple | Max DD % | Ruined At |
| --- | --- | ---: | ---: | ---: | --- |
| observed_11 | CURRENT_ENV | 636.074 | 18.174 | 0.0 | None |
| observed_11 | STOP_ASSISTED | 39.531 | 1.129 | 0.0 | None |
| observed_11 | BALANCED | 36.8 | 1.051 | 0.0 | None |
| observed_11 | SURVIVAL | 36.207 | 1.034 | 0.0 | None |
| append_minus150 | CURRENT_ENV | 311.676 | 8.905 | 926.85 | None |
| append_minus150 | STOP_ASSISTED | 38.856 | 1.11 | 1.927 | None |
| append_minus150 | BALANCED | 36.543 | 1.044 | 0.735 | None |
| append_minus150 | SURVIVAL | 36.036 | 1.03 | 0.488 | None |
| append_minus300 | CURRENT_ENV | -12.721 | -0.363 | 1853.7 | 12 |
| append_minus300 | STOP_ASSISTED | 38.182 | 1.091 | 3.853 | None |
| append_minus300 | BALANCED | 36.286 | 1.037 | 1.469 | None |
| append_minus300 | SURVIVAL | 35.865 | 1.025 | 0.975 | None |
| append_minus507 | CURRENT_ENV | -460.39 | -13.154 | 3132.754 | 12 |
| append_minus507 | STOP_ASSISTED | 37.252 | 1.064 | 6.512 | None |
| append_minus507 | BALANCED | 35.931 | 1.027 | 2.483 | None |
| append_minus507 | SURVIVAL | 35.63 | 1.018 | 1.648 | None |
| every5_minus150 | CURRENT_ENV | 152.721 | 4.363 | 224.944 | None |
| every5_minus150 | STOP_ASSISTED | 38.194 | 1.091 | 1.831 | None |
| every5_minus150 | BALANCED | 36.288 | 1.037 | 0.719 | None |
| every5_minus150 | SURVIVAL | 35.866 | 1.025 | 0.481 | None |
| every10_minus300 | CURRENT_ENV | -12.721 | -0.363 | 936.481 | 11 |
| every10_minus300 | STOP_ASSISTED | 38.182 | 1.091 | 3.726 | None |
| every10_minus300 | BALANCED | 36.286 | 1.037 | 1.449 | None |
| every10_minus300 | SURVIVAL | 35.865 | 1.025 | 0.966 | None |

## Exit Expansion Top

| Variant | N | Sum bps | Median | Win | T3R | Max loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tp300_sl150_4h | 11 | 1780.1 | 165.6 | 1.0 | 895.1 | 7.3 |
| fixed_4h | 11 | 1740.8 | 165.6 | 1.0 | 822.6 | 7.3 |
| fixed_8h | 11 | 1406.4 | 146.7 | 0.818 | 542.2 | -211.3 |
| trail100_after150_4h | 11 | 1361.0 | 130.9 | 1.0 | 744.5 | 7.3 |
| fixed_2h | 11 | 1089.9 | 46.5 | 1.0 | 406.3 | 12.9 |
| sl150_2h | 11 | 1089.9 | 46.5 | 1.0 | 406.3 | 12.9 |
| partial_tp150_2h | 11 | 1014.4 | 81.2 | 1.0 | 455.0 | 12.9 |

## Bull Pullback Shadow Screen

Definition: Bull regime, ETH SELL-liq running threshold 50/100/150K, shallow vdepth 5-28bps, mark-entry forward label minus 5bps.

Events: `85`

| Cell | N | Sum bps | Median | Win | T3R | Max loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| thr100000_h1800 | 25 | -220.3 | -3.5 | 0.4 | -559.9 | -221.6 |
| thr100000_h3600 | 25 | -576.1 | -18.8 | 0.36 | -852.8 | -205.5 |
| thr100000_h7200 | 25 | -550.9 | -20.6 | 0.48 | -1009.3 | -387.2 |
| thr150000_h1800 | 13 | -24.6 | -3.5 | 0.385 | -109.1 | -64.7 |
| thr150000_h3600 | 13 | -165.8 | -11.9 | 0.385 | -350.6 | -100.3 |
| thr150000_h7200 | 13 | 105.8 | 5.7 | 0.538 | -238.1 | -119.3 |
| thr50000_h1800 | 47 | 277.1 | 7.6 | 0.574 | -96.2 | -221.6 |
| thr50000_h3600 | 47 | -151.4 | -3.9 | 0.468 | -492.1 | -205.5 |
| thr50000_h7200 | 47 | -113.7 | 5.7 | 0.532 | -643.5 | -387.2 |
