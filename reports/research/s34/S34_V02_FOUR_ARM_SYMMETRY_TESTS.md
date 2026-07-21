# S34 V02 Four-Arm Symmetry Tests

Generated: `2026-06-30T08:05:28.760013+00:00`

Research-only. No live executor/config/order logic is touched.

## Verdict

- Overall: `NO_DEPLOYABLE_MIRROR_SHORT`
- Mirror decision: `DO_NOT_ADD_TO_LIVE_OR_PAPER; research/shadow-observe only`

## Arm Summary

| Arm | Role | Eligible | Filled | Fill% | H2 | H4 | H4-H2 | Cross-support H4 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `SELL_LONG_BASELINE` | current_live_family | 11 | 11 | 100.0 | N=11 sum=1048.8 med=43.3 T3R=378.8 | N=11 sum=1705.9 med=161.6 T3R=795.5 | N=11 sum=657.1 med=83.8 | N=10 sum=1679.9 T3R=769.5 |
| `SELL_SHORT_NEG_CONTROL` | same_event_opposite_direction_negative_control | 11 | 11 | 100.0 | N=11 sum=-811.3 med=-32.2 T3R=-933.1 | N=11 sum=-1554.2 med=-114.6 T3R=-1465.3 | N=11 sum=-742.9 med=-93.5 | N=8 sum=-758.5 T3R=-669.6 |
| `BUY_SHORT_MIRROR` | true_mirror_candidate | 17 | 17 | 100.0 | N=17 sum=181.8 med=11.2 T3R=-346.4 | N=17 sum=95.9 med=9.7 T3R=-443.1 | N=17 sum=-85.9 med=-2.8 | N=14 sum=333.0 T3R=-162.3 |
| `BUY_LONG_NEG_CONTROL` | mirror_event_opposite_direction_negative_control | 17 | 13 | 76.5 | N=13 sum=-610.2 med=-2.0 T3R=-700.1 | N=13 sum=-463.8 med=-17.4 T3R=-705.8 | N=13 sum=146.4 med=12.6 | N=10 sum=-234.8 T3R=-393.6 |

## Chronological Holdout

Split: `{'method': 'chronological_month_tail_35pct', 'months': ['2026-04', '2026-06'], 'holdout_months': ['2026-06']}`

| Arm | Cal H4 | Hold H4 |
| --- | --- | --- |
| `SELL_LONG_BASELINE` | N=4 sum=367.4 T3R=3.0 | N=7 sum=1338.5 T3R=428.1 |
| `SELL_SHORT_NEG_CONTROL` | N=4 sum=-331.5 T3R=-154.6 | N=7 sum=-1222.7 T3R=-1012.2 |
| `BUY_SHORT_MIRROR` | N=9 sum=-488.3 T3R=-733.5 | N=8 sum=584.2 T3R=96.6 |
| `BUY_LONG_NEG_CONTROL` | N=6 sum=37.2 T3R=-202.1 | N=7 sum=-501.0 T3R=-483.3 |

## Permutation Max-Stat

```json
{
  "iterations": 500,
  "mc_corrected_p_right": 0.002,
  "null_p95_max_t3r": 227.1,
  "observed_max_t3r": 795.5,
  "observed_t3r": {
    "BUY_LONG_NEG_CONTROL": -705.8,
    "BUY_SHORT_MIRROR": -443.1,
    "SELL_LONG_BASELINE": 795.5,
    "SELL_SHORT_NEG_CONTROL": -1465.3
  },
  "read": "Pass requires observed max T3R > null p95 and low corrected p; this controls the four-arm search at a coarse level.",
  "seed": 3402,
  "status": "OK"
}
```

## Rejection Counts

```json
{
  "BUY": {
    "book": 29,
    "depth": 11,
    "prior": 25,
    "raw": 179,
    "vdepth": 97
  },
  "SELL": {
    "book": 25,
    "depth": 12,
    "prior": 41,
    "raw": 175,
    "vdepth": 86
  }
}
```

## Read

- SELL->LONG baseline remains the reference arm; BUY->SHORT must beat BUY->LONG and survive holdout before it is even shadow-candidate.
- Same-event opposite-direction arms are negative controls. If they are positive, the result is likely regime/fill bias rather than clean direction.
- This suite uses the V02 O20/W300/O5/C1 maker lifecycle and H2/H3/H4 exits, but still uses top-of-book/proxy queue, not full tick queue replay.
- No live or paper bucket is changed by this script.
