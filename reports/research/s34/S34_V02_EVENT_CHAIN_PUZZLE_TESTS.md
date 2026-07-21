# S34 V02 Event-Chain Puzzle Tests

Generated: `2026-06-30T08:12:14.254426+00:00`

Research-only. No live executor/config/order logic is touched.

## Verdict

- Overall: `CHAIN_NAVIGATION_HYPOTHESES_FOUND_NOT_LIVE_RULE`

## 1. Same-Symbol Transition Graph

Current-event H4 fade outcome by transition:

| Transition | Count | Current H4 | Next H4 | Gap sec |
| --- | ---: | --- | --- | --- |
| `BUY->BUY` | 261 | N=261 sum=-20520.3 med=-53.3 T3R=-21425.3 | N=261 sum=-5629.3 med=3.4 T3R=-6866.2 | med=2419.5 |
| `BUY->SELL` | 226 | N=226 sum=6359.0 med=38.7 T3R=4935.2 | N=226 sum=1144.9 med=6.0 T3R=-231.2 | med=2349.7 |
| `SELL->BUY` | 233 | N=233 sum=13589.9 med=53.8 T3R=12186.1 | N=232 sum=-5650.9 med=-17.0 T3R=-7036.0 | med=2673.9 |
| `SELL->SELL` | 283 | N=283 sum=-15052.0 med=-28.5 T3R=-15978.5 | N=283 sum=-132.6 med=17.8 T3R=-1259.2 | med=3032.0 |

## 2. Anchor vs Event-End vs Reclaim

| Side | Anchor H4 | Event-end H4 | Reclaim H4 | Anchor->End delta H4 | Reclaim delay |
| --- | --- | --- | --- | --- | --- |
| `SELL` | N=585 sum=-827.0 med=11.3 T3R=-2230.8 | N=585 sum=5702.0 med=20.1 T3R=4275.2 | N=543 sum=3219.3 med=18.0 T3R=1633.2 | N=585 sum=6529.0 med=4.4 | med=11.1s |
| `BUY` | N=556 sum=-13218.6 med=-3.2 T3R=-14642.4 | N=556 sum=-5379.7 med=8.6 T3R=-6827.5 | N=508 sum=-6295.2 med=0.8 T3R=-7713.1 | N=556 sum=7838.9 med=4.7 | med=10.4s |

## 3. V02 Runner Chain Anatomy

```json
{
  "non_runner": {
    "first_next_gap_sec": {
      "max_bps": 2211.6,
      "mean_bps": 1252.7,
      "median_bps": 1183.7,
      "min_bps": 431.8,
      "n": 4,
      "sum_bps": 5010.8,
      "t3r_bps": 431.8,
      "tail_lt_-100_n": 0,
      "win_rate": 1.0
    },
    "h4_minus_h2": {
      "max_bps": -4.5,
      "mean_bps": -36.6,
      "median_bps": -47.3,
      "min_bps": -59.2,
      "n": 5,
      "sum_bps": -183.2,
      "t3r_bps": -111.3,
      "tail_lt_-100_n": 0,
      "win_rate": 0.0
    },
    "n": 5,
    "next_event_n_60m": {
      "max_bps": 3.0,
      "mean_bps": 1.2,
      "median_bps": 1.0,
      "min_bps": 0.0,
      "n": 5,
      "sum_bps": 6.0,
      "t3r_bps": 1.0,
      "tail_lt_-100_n": 0,
      "win_rate": 0.8
    },
    "opposite_next_60m": {
      "max_bps": 2.0,
      "mean_bps": 0.8,
      "median_bps": 1.0,
      "min_bps": 0.0,
      "n": 5,
      "sum_bps": 4.0,
      "t3r_bps": 0.0,
      "tail_lt_-100_n": 0,
      "win_rate": 0.6
    },
    "prev_event_n_60m": {
      "max_bps": 1.0,
      "mean_bps": 0.2,
      "median_bps": 0.0,
      "min_bps": 0.0,
      "n": 5,
      "sum_bps": 1.0,
      "t3r_bps": 0.0,
      "tail_lt_-100_n": 0,
      "win_rate": 0.2
    },
    "same_side_next_60m": {
      "max_bps": 1.0,
      "mean_bps": 0.4,
      "median_bps": 0.0,
      "min_bps": 0.0,
      "n": 5,
      "sum_bps": 2.0,
      "t3r_bps": 0.0,
      "tail_lt_-100_n": 0,
      "win_rate": 0.4
    }
  },
  "runner": {
    "first_next_gap_sec": {
      "max_bps": 2174.9,
      "mean_bps": 1478.0,
      "median_bps": 1355.9,
      "min_bps": 902.6,
      "n": 5,
      "sum_bps": 7389.9,
      "t3r_bps": 1907.1,
      "tail_lt_-100_n": 0,
      "win_rate": 1.0
    },
    "h4_minus_h2": {
      "max_bps": 223.2,
      "mean_bps": 140.0,
      "median_bps": 142.7,
      "min_bps": 83.8,
      "n": 6,
      "sum_bps": 840.2,
      "t3r_bps": 313.2,
      "tail_lt_-100_n": 0,
      "win_rate": 1.0
    },
    "n": 6,
    "next_event_n_60m": {
      "max_bps": 3.0,
      "mean_bps": 1.7,
      "median_bps": 2.0,
      "min_bps": 0.0,
      "n": 6,
      "sum_bps": 10.0,
      "t3r_bps": 3.0,
      "tail_lt_-100_n": 0,
      "win_rate": 0.833
    },
    "opposite_next_60m": {
      "max_bps": 2.0,
      "mean_bps": 0.8,
      "median_bps": 0.5,
      "min_bps": 0.0,
      "n": 6,
      "sum_bps": 5.0,
      "t3r_bps": 0.0,
      "tail_lt_-100_n": 0,
      "win_rate": 0.5
    },
    "prev_event_n_60m": {
      "max_bps": 2.0,
      "mean_bps": 1.0,
      "median_bps": 1.0,
      "min_bps": 0.0,
      "n": 6,
      "sum_bps": 6.0,
      "t3r_bps": 1.0,
      "tail_lt_-100_n": 0,
      "win_rate": 0.667
    },
    "same_side_next_60m": {
      "max_bps": 2.0,
      "mean_bps": 0.8,
      "median_bps": 1.0,
      "min_bps": 0.0,
      "n": 6,
      "sum_bps": 5.0,
      "t3r_bps": 1.0,
      "tail_lt_-100_n": 0,
      "win_rate": 0.667
    }
  }
}
```

## 4. Cross-Asset Chain

```json
{
  "counter_next_buy": {
    "false": {
      "max_bps": 611.3,
      "mean_bps": -3.1,
      "median_bps": 8.0,
      "min_bps": -540.8,
      "n": 546,
      "sum_bps": -1673.0,
      "t3r_bps": -3049.1,
      "tail_lt_-100_n": 104,
      "win_rate": 0.527
    },
    "true": {
      "max_bps": 405.7,
      "mean_bps": 21.7,
      "median_bps": 51.8,
      "min_bps": -514.1,
      "n": 39,
      "sum_bps": 846.0,
      "t3r_bps": -253.6,
      "tail_lt_-100_n": 10,
      "win_rate": 0.59
    }
  },
  "propagation_next_sell": {
    "false": {
      "max_bps": 611.3,
      "mean_bps": 18.9,
      "median_bps": 20.6,
      "min_bps": -455.3,
      "n": 395,
      "sum_bps": 7456.9,
      "t3r_bps": 6088.9,
      "tail_lt_-100_n": 58,
      "win_rate": 0.577
    },
    "true": {
      "max_bps": 405.7,
      "mean_bps": -43.6,
      "median_bps": -18.3,
      "min_bps": -540.8,
      "n": 190,
      "sum_bps": -8283.9,
      "t3r_bps": -9434.6,
      "tail_lt_-100_n": 56,
      "win_rate": 0.437
    }
  },
  "sync_prev_sell": {
    "false": {
      "max_bps": 611.3,
      "mean_bps": 0.2,
      "median_bps": 7.3,
      "min_bps": -540.8,
      "n": 465,
      "sum_bps": 99.8,
      "t3r_bps": -1276.3,
      "tail_lt_-100_n": 88,
      "win_rate": 0.527
    },
    "true": {
      "max_bps": 405.7,
      "mean_bps": -7.7,
      "median_bps": 28.9,
      "min_bps": -514.1,
      "n": 120,
      "sum_bps": -926.8,
      "t3r_bps": -2056.3,
      "tail_lt_-100_n": 26,
      "win_rate": 0.55
    }
  }
}
```

## 5. Four-Arm Chain Read

```json
{
  "BUY_SHORT_MIRROR": {
    "by_cross_support": {
      "false": {
        "max_bps": 152.0,
        "mean_bps": -79.0,
        "median_bps": -158.4,
        "min_bps": -230.7,
        "n": 3,
        "sum_bps": -237.1,
        "t3r_bps": -237.1,
        "tail_lt_-100_n": 2,
        "win_rate": 0.333
      },
      "true": {
        "max_bps": 278.6,
        "mean_bps": 23.8,
        "median_bps": 19.1,
        "min_bps": -245.4,
        "n": 14,
        "sum_bps": 333.0,
        "t3r_bps": -162.3,
        "tail_lt_-100_n": 1,
        "win_rate": 0.643
      }
    },
    "by_fill_leg": {
      "initial": {
        "max_bps": 278.6,
        "mean_bps": -1.4,
        "median_bps": 28.5,
        "min_bps": -245.4,
        "n": 9,
        "sum_bps": -12.3,
        "t3r_bps": -551.3,
        "tail_lt_-100_n": 3,
        "win_rate": 0.667
      },
      "replacement": {
        "max_bps": 108.3,
        "mean_bps": 13.5,
        "median_bps": 1.9,
        "min_bps": -55.2,
        "n": 8,
        "sum_bps": 108.2,
        "t3r_bps": -95.2,
        "tail_lt_-100_n": 0,
        "win_rate": 0.5
      }
    },
    "h4": {
      "max_bps": 278.6,
      "mean_bps": 5.6,
      "median_bps": 9.7,
      "min_bps": -245.4,
      "n": 17,
      "sum_bps": 95.9,
      "t3r_bps": -443.1,
      "tail_lt_-100_n": 3,
      "win_rate": 0.588
    },
    "h4_minus_h2": {
      "max_bps": 133.7,
      "mean_bps": -5.1,
      "median_bps": -2.8,
      "min_bps": -135.2,
      "n": 17,
      "sum_bps": -85.9,
      "t3r_bps": -385.3,
      "tail_lt_-100_n": 2,
      "win_rate": 0.412
    }
  },
  "SELL_LONG_BASELINE": {
    "by_cross_support": {
      "false": {
        "max_bps": 26.0,
        "mean_bps": 26.0,
        "median_bps": 26.0,
        "min_bps": 26.0,
        "n": 1,
        "sum_bps": 26.0,
        "t3r_bps": 26.0,
        "tail_lt_-100_n": 0,
        "win_rate": 1.0
      },
      "true": {
        "max_bps": 392.1,
        "mean_bps": 168.0,
        "median_bps": 163.2,
        "min_bps": 3.0,
        "n": 10,
        "sum_bps": 1679.9,
        "t3r_bps": 769.5,
        "tail_lt_-100_n": 0,
        "win_rate": 1.0
      }
    },
    "by_fill_leg": {
      "initial": {
        "max_bps": 392.1,
        "mean_bps": 147.2,
        "median_bps": 34.2,
        "min_bps": 3.0,
        "n": 5,
        "sum_bps": 736.2,
        "t3r_bps": 29.0,
        "tail_lt_-100_n": 0,
        "win_rate": 1.0
      },
      "replacement": {
        "max_bps": 237.4,
        "mean_bps": 161.6,
        "median_bps": 163.2,
        "min_bps": 102.0,
        "n": 6,
        "sum_bps": 969.7,
        "t3r_bps": 390.7,
        "tail_lt_-100_n": 0,
        "win_rate": 1.0
      }
    },
    "h4": {
      "max_bps": 392.1,
      "mean_bps": 155.1,
      "median_bps": 161.6,
      "min_bps": 3.0,
      "n": 11,
      "sum_bps": 1705.9,
      "t3r_bps": 795.5,
      "tail_lt_-100_n": 0,
      "win_rate": 1.0
    },
    "h4_minus_h2": {
      "max_bps": 223.2,
      "mean_bps": 59.7,
      "median_bps": 83.8,
      "min_bps": -59.2,
      "n": 11,
      "sum_bps": 657.1,
      "t3r_bps": 129.8,
      "tail_lt_-100_n": 0,
      "win_rate": 0.545
    }
  }
}
```

## Read

- This reframes the V02 alpha as an event lifecycle problem, not a single anchor problem.
- Event-end/reclaim can be better for diagnosis, but if it sacrifices too much entry price it should stay navigation/management-only.
- Same-symbol transition tells whether the next cascade is a continuation, counter-cascade, or silence state.
- Cross-asset chain tests whether BTC/SOL are leading/propagating the ETH event or merely co-moving.
- These are research/navigation outputs only; no live or paper bucket is changed.
- SELL event_end H4 T3R beats anchor H4 T3R in this broad anchor-mark test; this is a management/navigation lead to validate on V02 fills.
