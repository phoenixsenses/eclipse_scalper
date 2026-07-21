# S34 V02 Propagation Candidate Gauntlet

Generated: `2026-06-30T08:26:09.877218+00:00`

Research-only. No live executor/config/order logic is touched.

## Verdict

- Overall: `SHADOW_CANDIDATES_FOUND_BUT_NO_LIVE_PROMOTION`
- Tags exported: `D:\eclipse_scalper\reports\research\s34\S34_V02_PROPAGATION_NAV_TAGS.jsonl`

## Candidate Leaderboard

| Rank | Candidate | Tau | N | All | Hold | Neg Ctrl | Pass |
| ---: | --- | ---: | ---: | --- | --- | --- | --- |
| 1 | `SELL_SILENCE_FADE_LONG_H4` | 1800 | 345 | sum=9503.0 med=29.5 T3R=8135.0 | sum=4111.6 T3R=2819.5 | sum=-15023.0 T3R=-16174.1 | True |
| 2 | `BUY_PROPAGATION_MOMENTUM_LONG_H1` | 3600 | 194 | sum=7672.0 med=26.6 T3R=6658.1 | sum=2803.9 T3R=2062.4 | sum=-10776.0 T3R=-11623.4 | True |
| 3 | `SELL_PROPAGATION_MOMENTUM_SHORT_H1` | 3600 | 185 | sum=7763.4 med=24.7 T3R=6518.4 | sum=4066.3 T3R=2843.9 | sum=-10723.4 T3R=-11280.2 | True |
| 4 | `SELL_SILENCE_FADE_LONG_H4` | 3600 | 280 | sum=7523.8 med=26.0 T3R=6186.9 | sum=3174.6 T3R=1882.5 | sum=-12003.8 T3R=-13154.9 | True |
| 5 | `SELL_SILENCE_FADE_LONG_H4` | 900 | 405 | sum=6744.0 med=21.5 T3R=5376.0 | sum=3744.3 T3R=2452.2 | sum=-13224.0 T3R=-14539.4 | True |
| 6 | `SELL_SILENCE_FADE_LONG_H4` | 300 | 442 | sum=5479.3 med=19.9 T3R=4111.3 | sum=3230.6 T3R=1915.0 | sum=-12551.3 T3R=-13866.7 | True |
| 7 | `BUY_PROPAGATION_MOMENTUM_LONG_H1` | 1800 | 98 | sum=4974.3 med=39.2 T3R=4103.2 | sum=2059.4 T3R=1320.5 | sum=-6542.3 T3R=-7158.3 | True |
| 8 | `SELL_PROPAGATION_MOMENTUM_SHORT_H1` | 1800 | 93 | sum=4439.8 med=26.2 T3R=3489.1 | sum=2447.4 T3R=1496.7 | sum=-5927.8 T3R=-6423.9 | True |
| 9 | `BUY_SILENCE_FADE_SHORT_H4` | 3600 | 253 | sum=1201.5 med=14.7 T3R=98.0 | sum=1321.4 T3R=534.1 | sum=-5233.5 T3R=-6871.3 | True |
| 10 | `BUY_SILENCE_FADE_SHORT_H4` | 1800 | 321 | sum=-192.9 med=10.5 T3R=-1477.6 | sum=570.4 T3R=-216.9 | sum=-4927.1 T3R=-6564.9 | False |
| 11 | `BUY_SILENCE_FADE_SHORT_H4` | 900 | 386 | sum=-2901.4 med=6.6 T3R=-4186.1 | sum=47.0 T3R=-769.7 | sum=-3258.6 T3R=-4939.6 | False |
| 12 | `BUY_SILENCE_FADE_SHORT_H4` | 300 | 429 | sum=-3809.0 med=5.7 T3R=-5188.1 | sum=456.1 T3R=-569.6 | sum=-3039.0 T3R=-4720.0 | False |
| 13 | `SELL_PROPAGATION_MOMENTUM_SHORT_H1` | 300 | 0 | sum=0.0 med=None T3R=0.0 | sum=0.0 T3R=0.0 | sum=0.0 T3R=0.0 | False |
| 14 | `SELL_PROPAGATION_MOMENTUM_SHORT_H1` | 900 | 0 | sum=0.0 med=None T3R=0.0 | sum=0.0 T3R=0.0 | sum=0.0 T3R=0.0 | False |
| 15 | `BUY_PROPAGATION_MOMENTUM_LONG_H1` | 300 | 0 | sum=0.0 med=None T3R=0.0 | sum=0.0 T3R=0.0 | sum=0.0 T3R=0.0 | False |
| 16 | `BUY_PROPAGATION_MOMENTUM_LONG_H1` | 900 | 0 | sum=0.0 med=None T3R=0.0 | sum=0.0 T3R=0.0 | sum=0.0 T3R=0.0 | False |

## Permutation Max-Stat

```json
{
  "iterations": 500,
  "mc_corrected_p_right": 0.383,
  "null_p95_max_t3r": 10632.9,
  "observed_max_t3r": 8135.0,
  "read": "Coarse max-stat correction across all candidate/tau cells. It is a guardrail, not final proof.",
  "seed": 3403,
  "status": "OK"
}
```

## V02 Compatibility

```json
{
  "all_h4_minus_h2": {
    "max_bps": 223.2,
    "mean_bps": 59.7,
    "median_bps": 83.8,
    "min_bps": -59.2,
    "n": 11,
    "sum_bps": 657.0,
    "t3r_bps": 130.0,
    "tail_lt_-100_n": 0,
    "win_rate": 0.545
  },
  "matched_n": 11,
  "n": 11,
  "not_pressure_high_1800": {
    "max_bps": 223.2,
    "mean_bps": 77.1,
    "median_bps": 89.6,
    "min_bps": -47.3,
    "n": 8,
    "sum_bps": 617.0,
    "t3r_bps": 107.3,
    "tail_lt_-100_n": 0,
    "win_rate": 0.625
  },
  "not_silence_1800": {
    "max_bps": 151.3,
    "mean_bps": 33.9,
    "median_bps": 21.6,
    "min_bps": -59.2,
    "n": 4,
    "sum_bps": 135.4,
    "t3r_bps": -59.2,
    "tail_lt_-100_n": 0,
    "win_rate": 0.5
  },
  "pressure_high_1800": {
    "max_bps": 151.3,
    "mean_bps": 13.3,
    "median_bps": -52.1,
    "min_bps": -59.2,
    "n": 3,
    "sum_bps": 40.0,
    "t3r_bps": 40.0,
    "tail_lt_-100_n": 0,
    "win_rate": 0.333
  },
  "read": "V02 compatibility remains small-N. Tags are management/navigation context, not order logic.",
  "silence_1800": {
    "max_bps": 223.2,
    "mean_bps": 74.5,
    "median_bps": 83.8,
    "min_bps": -47.3,
    "n": 7,
    "sum_bps": 521.6,
    "t3r_bps": 11.9,
    "tail_lt_-100_n": 0,
    "win_rate": 0.571
  }
}
```

## Tag Counts @ 1800s

```json
{
  "BUY_LONG_MOMENTUM_WATCH": 98,
  "BUY_PROPAGATION": 98,
  "BUY_SHORT_FADE_DANGER": 98,
  "BUY_SHORT_FADE_NAV_WATCH": 321,
  "BUY_SILENCE_RECLAIM": 321,
  "FADE_MODE": 666,
  "MOMENTUM_MODE": 191,
  "PROPAGATION_PRESSURE_HIGH": 191,
  "PROPAGATION_PRESSURE_LOW": 32,
  "PROPAGATION_PRESSURE_MID": 253,
  "SELL_FADE_DANGER": 93,
  "SELL_LONG_FADE_NAV_OK": 345,
  "SELL_PROPAGATION": 93,
  "SELL_SHORT_MOMENTUM_WATCH": 93,
  "SELL_SILENCE_RECLAIM": 345,
  "SILENCE_AFTER_SHOCK": 666
}
```

## Read

- The best broad candidates are state labels, not live-ready strategies yet.
- A full pass requires N>=40, positive all/hold sum, positive all/hold T3R, worse negative control, and no top-3 dependency.
- The tag export is meant for chart/navigation: PROPAGATION_PRESSURE_HIGH, SILENCE_AFTER_SHOCK, MOMENTUM_MODE, FADE_MODE.
- Execution is still mark/taker proxy here. Any candidate that looks good must next pass maker/taker live-like fill and forward shadow.
- 9 candidate/tau cells passed the mechanical gauntlet, but they still need execution realism and forward shadow before paper/live.
