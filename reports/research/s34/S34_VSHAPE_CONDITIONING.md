# S34 V-Shape Conditioning (does spike depth predict the reversal?)

Generated: `2026-06-28T18:38:32.656982+00:00`  |  ETHUSDT 200K, cost 6.1bps, holdout 0.3

Fade return binned by knowable V-depth (cascade-direction overshoot at the cross). Terciles from calibration, applied to holdout. Hypothesis holds only if the DEEP bin is net-positive on BOTH splits and beats shallow (monotone depth->reversal). `**` = deep bin stable-positive both splits.

V-depth tercile cuts (bps): low<= 9.9, high> 27.7  |  total events: 1117

## 1h

| Depth bin | cal N | cal net med | cal win | hold N | hold net med | hold win | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| shallow | 261 | 2.0 | 57.1 | 107 | 0.7 | 54.2 | + |
| mid | 260 | -1.7 | 53.5 | 141 | -4.4 | 51.8 |  |
| deep | 260 | 9.2 | 59.2 | 88 | 0.0 | 53.4 |  |

## 4h

| Depth bin | cal N | cal net med | cal win | hold N | hold net med | hold win | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| shallow | 261 | -5.9 | 50.6 | 106 | 4.9 | 57.5 |  |
| mid | 260 | -0.7 | 52.3 | 141 | 18.3 | 56.7 |  |
| deep | 260 | 10.9 | 55.8 | 87 | 13.2 | 55.2 | ** |

## 24h

| Depth bin | cal N | cal net med | cal win | hold N | hold net med | hold win | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| shallow | 261 | -21.3 | 48.7 | 103 | -7.3 | 49.5 |  |
| mid | 260 | 29.8 | 53.8 | 136 | 3.5 | 51.5 | + |
| deep | 260 | -13.4 | 47.3 | 86 | 19.9 | 55.8 |  |
