# S34 V02 Candidate Execution Gauntlet

Generated: `2026-06-30T08:34:34.737166+00:00`

Research-only. No live executor/config/order logic is touched.

## Verdict

- `NO_EXECUTION_READY_CANDIDATE`

## Taker Causal Entry Leaderboard

| Candidate | Direction | Tau | Horizon | Taker0 All | Taker0 Hold | Negative0 | Maker best | Stop best |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| `SELL_SILENCE_FADE_LONG_H4` | LONG | 1800 | 14400 | N=131 sum=1160.8 med=15.0 T3R=13.6 | N=81 sum=857.9 T3R=-244.4 | N=131 sum=-3271.8 T3R=-4632.5 | O5.0 N=90 sum=900.1 T3R=44.5 fill=0.261 | SL150.0 sum=1419.7 T3R=271.2 |
| `SELL_PROPAGATION_MOMENTUM_SHORT_H1` | SHORT | 3600 | 3600 | N=98 sum=-736.7 med=-10.2 T3R=-1516.4 | N=76 sum=-710.1 T3R=-1489.8 | N=98 sum=-842.6 T3R=-1344.6 | O10.0 N=72 sum=-4.6 T3R=-738.7 fill=0.389 | SL50.0 sum=-810.2 T3R=-1589.3 |
| `BUY_PROPAGATION_MOMENTUM_LONG_H1` | LONG | 3600 | 3600 | N=99 sum=-1294.3 med=-17.8 T3R=-1835.0 | N=58 sum=-426.3 T3R=-951.9 | N=99 sum=-301.4 T3R=-915.8 | O10.0 N=72 sum=-512.0 T3R=-1027.4 fill=0.371 | SL150.0 sum=-1185.9 T3R=-1728.0 |
| `BUY_SILENCE_FADE_SHORT_H4` | SHORT | 3600 | 14400 | N=83 sum=-492.0 med=-1.7 T3R=-1279.4 | N=39 sum=-146.2 T3R=-823.6 | N=83 sum=-844.5 T3R=-2139.0 | O10.0 N=54 sum=229.4 T3R=-470.6 fill=0.213 | SL150.0 sum=138.2 T3R=-648.9 |

## Delay Sensitivity

```json
{
  "BUY_PROPAGATION_MOMENTUM_LONG_H1": {
    "0": {
      "attempt_n": 194,
      "fill_rate": 0.51,
      "filled_n": 99,
      "max_bps": 234.4,
      "mean_bps": -13.1,
      "median_bps": -17.8,
      "min_bps": -294.0,
      "n": 99,
      "sum_bps": -1294.3,
      "t3r_bps": -1835.0,
      "tail_lt_-100_n": 7,
      "win_rate": 0.354
    },
    "30": {
      "attempt_n": 194,
      "fill_rate": 0.515,
      "filled_n": 100,
      "max_bps": 221.1,
      "mean_bps": -12.6,
      "median_bps": -17.1,
      "min_bps": -275.0,
      "n": 100,
      "sum_bps": -1262.1,
      "t3r_bps": -1783.6,
      "tail_lt_-100_n": 8,
      "win_rate": 0.33
    },
    "300": {
      "attempt_n": 194,
      "fill_rate": 0.521,
      "filled_n": 101,
      "max_bps": 253.1,
      "mean_bps": -10.0,
      "median_bps": -13.3,
      "min_bps": -216.8,
      "n": 101,
      "sum_bps": -1005.8,
      "t3r_bps": -1588.8,
      "tail_lt_-100_n": 6,
      "win_rate": 0.406
    },
    "60": {
      "attempt_n": 194,
      "fill_rate": 0.515,
      "filled_n": 100,
      "max_bps": 228.0,
      "mean_bps": -10.3,
      "median_bps": -15.2,
      "min_bps": -283.2,
      "n": 100,
      "sum_bps": -1025.2,
      "t3r_bps": -1581.8,
      "tail_lt_-100_n": 8,
      "win_rate": 0.38
    }
  },
  "BUY_SILENCE_FADE_SHORT_H4": {
    "0": {
      "attempt_n": 253,
      "fill_rate": 0.328,
      "filled_n": 83,
      "max_bps": 289.7,
      "mean_bps": -5.9,
      "median_bps": -1.7,
      "min_bps": -534.9,
      "n": 83,
      "sum_bps": -492.0,
      "t3r_bps": -1279.4,
      "tail_lt_-100_n": 7,
      "win_rate": 0.494
    },
    "30": {
      "attempt_n": 253,
      "fill_rate": 0.332,
      "filled_n": 84,
      "max_bps": 291.5,
      "mean_bps": -5.4,
      "median_bps": 2.0,
      "min_bps": -540.1,
      "n": 84,
      "sum_bps": -454.1,
      "t3r_bps": -1257.0,
      "tail_lt_-100_n": 7,
      "win_rate": 0.512
    },
    "300": {
      "attempt_n": 253,
      "fill_rate": 0.328,
      "filled_n": 83,
      "max_bps": 295.8,
      "mean_bps": -2.6,
      "median_bps": 3.5,
      "min_bps": -513.2,
      "n": 83,
      "sum_bps": -217.2,
      "t3r_bps": -987.9,
      "tail_lt_-100_n": 7,
      "win_rate": 0.506
    },
    "60": {
      "attempt_n": 253,
      "fill_rate": 0.332,
      "filled_n": 84,
      "max_bps": 298.3,
      "mean_bps": -5.5,
      "median_bps": -0.7,
      "min_bps": -532.8,
      "n": 84,
      "sum_bps": -461.7,
      "t3r_bps": -1281.8,
      "tail_lt_-100_n": 7,
      "win_rate": 0.488
    }
  },
  "SELL_PROPAGATION_MOMENTUM_SHORT_H1": {
    "0": {
      "attempt_n": 185,
      "fill_rate": 0.53,
      "filled_n": 98,
      "max_bps": 347.9,
      "mean_bps": -7.5,
      "median_bps": -10.2,
      "min_bps": -199.4,
      "n": 98,
      "sum_bps": -736.7,
      "t3r_bps": -1516.4,
      "tail_lt_-100_n": 9,
      "win_rate": 0.347
    },
    "30": {
      "attempt_n": 185,
      "fill_rate": 0.53,
      "filled_n": 98,
      "max_bps": 354.8,
      "mean_bps": -9.3,
      "median_bps": -15.9,
      "min_bps": -197.4,
      "n": 98,
      "sum_bps": -908.8,
      "t3r_bps": -1719.6,
      "tail_lt_-100_n": 10,
      "win_rate": 0.367
    },
    "300": {
      "attempt_n": 185,
      "fill_rate": 0.541,
      "filled_n": 100,
      "max_bps": 320.4,
      "mean_bps": -6.8,
      "median_bps": -9.8,
      "min_bps": -188.1,
      "n": 100,
      "sum_bps": -682.6,
      "t3r_bps": -1348.3,
      "tail_lt_-100_n": 9,
      "win_rate": 0.42
    },
    "60": {
      "attempt_n": 185,
      "fill_rate": 0.53,
      "filled_n": 98,
      "max_bps": 334.2,
      "mean_bps": -7.1,
      "median_bps": -13.1,
      "min_bps": -183.0,
      "n": 98,
      "sum_bps": -698.2,
      "t3r_bps": -1455.1,
      "tail_lt_-100_n": 11,
      "win_rate": 0.398
    }
  },
  "SELL_SILENCE_FADE_LONG_H4": {
    "0": {
      "attempt_n": 345,
      "fill_rate": 0.38,
      "filled_n": 131,
      "max_bps": 492.8,
      "mean_bps": 8.9,
      "median_bps": 15.0,
      "min_bps": -494.4,
      "n": 131,
      "sum_bps": 1160.8,
      "t3r_bps": 13.6,
      "tail_lt_-100_n": 15,
      "win_rate": 0.573
    },
    "30": {
      "attempt_n": 345,
      "fill_rate": 0.383,
      "filled_n": 132,
      "max_bps": 489.4,
      "mean_bps": 8.4,
      "median_bps": 9.6,
      "min_bps": -507.7,
      "n": 132,
      "sum_bps": 1106.9,
      "t3r_bps": -39.8,
      "tail_lt_-100_n": 15,
      "win_rate": 0.568
    },
    "300": {
      "attempt_n": 345,
      "fill_rate": 0.383,
      "filled_n": 132,
      "max_bps": 492.9,
      "mean_bps": 8.5,
      "median_bps": 12.9,
      "min_bps": -599.6,
      "n": 132,
      "sum_bps": 1116.7,
      "t3r_bps": -38.8,
      "tail_lt_-100_n": 13,
      "win_rate": 0.561
    },
    "60": {
      "attempt_n": 345,
      "fill_rate": 0.383,
      "filled_n": 132,
      "max_bps": 483.1,
      "mean_bps": 7.9,
      "median_bps": 11.4,
      "min_bps": -503.9,
      "n": 132,
      "sum_bps": 1036.2,
      "t3r_bps": -80.8,
      "tail_lt_-100_n": 14,
      "win_rate": 0.576
    }
  }
}
```

## Negative-Control Anatomy

```json
{
  "BUY_PROPAGATION_MOMENTUM_LONG_H1": {
    "main_minus_negative": {
      "max_bps": 484.9,
      "mean_bps": -10.0,
      "median_bps": -19.5,
      "min_bps": -571.9,
      "n": 99,
      "sum_bps": -992.9,
      "t3r_bps": -2122.6,
      "tail_lt_-100_n": 19,
      "win_rate": 0.384
    },
    "negative_all": {
      "max_bps": 277.9,
      "mean_bps": -3.0,
      "median_bps": 1.7,
      "min_bps": -250.5,
      "n": 99,
      "sum_bps": -301.4,
      "t3r_bps": -915.8,
      "tail_lt_-100_n": 7,
      "win_rate": 0.505
    },
    "negative_by_month": {
      "2026-04": {
        "max_bps": 213.4,
        "mean_bps": 5.1,
        "median_bps": 4.2,
        "min_bps": -148.1,
        "n": 41,
        "sum_bps": 207.3,
        "t3r_bps": -229.7,
        "tail_lt_-100_n": 2,
        "win_rate": 0.537
      },
      "2026-06": {
        "max_bps": 277.9,
        "mean_bps": -8.8,
        "median_bps": -1.2,
        "min_bps": -250.5,
        "n": 58,
        "sum_bps": -508.7,
        "t3r_bps": -1013.6,
        "tail_lt_-100_n": 5,
        "win_rate": 0.483
      }
    },
    "negative_tail_lt_-100_examples": [
      {
        "anchor_utc": "2026-04-16T14:07:24.744000+00:00",
        "main": 100.7,
        "month": "2026-04",
        "neg": -116.8
      },
      {
        "anchor_utc": "2026-04-17T12:47:01.454000+00:00",
        "main": 132.0,
        "month": "2026-04",
        "neg": -148.1
      },
      {
        "anchor_utc": "2026-06-24T18:06:31.487000+00:00",
        "main": 234.4,
        "month": "2026-06",
        "neg": -250.5
      },
      {
        "anchor_utc": "2026-06-25T14:20:22.079000+00:00",
        "main": 116.9,
        "month": "2026-06",
        "neg": -133.1
      },
      {
        "anchor_utc": "2026-06-26T02:17:55.208000+00:00",
        "main": 174.3,
        "month": "2026-06",
        "neg": -190.4
      },
      {
        "anchor_utc": "2026-06-26T13:35:39.424000+00:00",
        "main": 93.6,
        "month": "2026-06",
        "neg": -109.7
      },
      {
        "anchor_utc": "2026-06-26T13:53:37.603000+00:00",
        "main": 89.8,
        "month": "2026-06",
        "neg": -105.9
      }
    ],
    "read": "If negative control is strongly negative at causal entry, the state is directional; if it is tail-only or month-only, treat as regime artefact."
  },
  "BUY_SILENCE_FADE_SHORT_H4": {
    "main_minus_negative": {
      "max_bps": 595.5,
      "mean_bps": 4.2,
      "median_bps": 12.7,
      "min_bps": -1053.7,
      "n": 83,
      "sum_bps": 352.5,
      "t3r_bps": -1270.8,
      "tail_lt_-100_n": 18,
      "win_rate": 0.542
    },
    "negative_all": {
      "max_bps": 518.8,
      "mean_bps": -10.2,
      "median_bps": -14.4,
      "min_bps": -305.8,
      "n": 83,
      "sum_bps": -844.5,
      "t3r_bps": -2139.0,
      "tail_lt_-100_n": 12,
      "win_rate": 0.398
    },
    "negative_by_month": {
      "2026-04": {
        "max_bps": 482.5,
        "mean_bps": -8.2,
        "median_bps": -13.1,
        "min_bps": -305.8,
        "n": 44,
        "sum_bps": -362.2,
        "t3r_bps": -1080.9,
        "tail_lt_-100_n": 5,
        "win_rate": 0.386
      },
      "2026-06": {
        "max_bps": 518.8,
        "mean_bps": -12.4,
        "median_bps": -21.7,
        "min_bps": -300.8,
        "n": 39,
        "sum_bps": -482.3,
        "t3r_bps": -1574.4,
        "tail_lt_-100_n": 7,
        "win_rate": 0.41
      }
    },
    "negative_tail_lt_-100_examples": [
      {
        "anchor_utc": "2026-04-14T14:26:50.093000+00:00",
        "main": 168.1,
        "month": "2026-04",
        "neg": -184.2
      },
      {
        "anchor_utc": "2026-04-16T12:11:20.191000+00:00",
        "main": 93.0,
        "month": "2026-04",
        "neg": -109.1
      },
      {
        "anchor_utc": "2026-04-19T13:21:27.242000+00:00",
        "main": 159.2,
        "month": "2026-04",
        "neg": -175.3
      },
      {
        "anchor_utc": "2026-04-23T05:50:30.442000+00:00",
        "main": 153.3,
        "month": "2026-04",
        "neg": -169.4
      },
      {
        "anchor_utc": "2026-04-27T01:47:32.076000+00:00",
        "main": 289.7,
        "month": "2026-04",
        "neg": -305.8
      },
      {
        "anchor_utc": "2026-06-16T12:31:11.271000+00:00",
        "main": 133.3,
        "month": "2026-06",
        "neg": -149.4
      },
      {
        "anchor_utc": "2026-06-17T03:33:52.548000+00:00",
        "main": 129.0,
        "month": "2026-06",
        "neg": -145.1
      },
      {
        "anchor_utc": "2026-06-18T00:03:43.640000+00:00",
        "main": 179.7,
        "month": "2026-06",
        "neg": -195.8
      },
      {
        "anchor_utc": "2026-06-18T14:19:32.699000+00:00",
        "main": 213.1,
        "month": "2026-06",
        "neg": -229.3
      },
      {
        "anchor_utc": "2026-06-25T21:54:47.503000+00:00",
        "main": 284.6,
        "month": "2026-06",
        "neg": -300.8
      }
    ],
    "read": "If negative control is strongly negative at causal entry, the state is directional; if it is tail-only or month-only, treat as regime artefact."
  },
  "SELL_PROPAGATION_MOMENTUM_SHORT_H1": {
    "main_minus_negative": {
      "max_bps": 712.0,
      "mean_bps": 1.1,
      "median_bps": -4.3,
      "min_bps": -382.7,
      "n": 98,
      "sum_bps": 105.9,
      "t3r_bps": -1501.9,
      "tail_lt_-100_n": 15,
      "win_rate": 0.469
    },
    "negative_all": {
      "max_bps": 183.3,
      "mean_bps": -8.6,
      "median_bps": -5.9,
      "min_bps": -364.1,
      "n": 98,
      "sum_bps": -842.6,
      "t3r_bps": -1344.6,
      "tail_lt_-100_n": 6,
      "win_rate": 0.459
    },
    "negative_by_month": {
      "2026-04": {
        "max_bps": 113.8,
        "mean_bps": -14.9,
        "median_bps": -10.0,
        "min_bps": -99.2,
        "n": 22,
        "sum_bps": -327.6,
        "t3r_bps": -497.8,
        "tail_lt_-100_n": 0,
        "win_rate": 0.318
      },
      "2026-06": {
        "max_bps": 183.3,
        "mean_bps": -6.8,
        "median_bps": -0.2,
        "min_bps": -364.1,
        "n": 76,
        "sum_bps": -515.0,
        "t3r_bps": -1017.0,
        "tail_lt_-100_n": 6,
        "win_rate": 0.5
      }
    },
    "negative_tail_lt_-100_examples": [
      {
        "anchor_utc": "2026-06-16T12:37:04.398000+00:00",
        "main": 97.0,
        "month": "2026-06",
        "neg": -113.1
      },
      {
        "anchor_utc": "2026-06-17T18:00:30.110000+00:00",
        "main": 210.6,
        "month": "2026-06",
        "neg": -226.7
      },
      {
        "anchor_utc": "2026-06-24T14:02:04.207000+00:00",
        "main": 148.3,
        "month": "2026-06",
        "neg": -164.5
      },
      {
        "anchor_utc": "2026-06-24T15:33:20.159000+00:00",
        "main": 347.9,
        "month": "2026-06",
        "neg": -364.1
      },
      {
        "anchor_utc": "2026-06-24T15:59:34.193000+00:00",
        "main": 177.1,
        "month": "2026-06",
        "neg": -193.3
      },
      {
        "anchor_utc": "2026-06-26T01:06:01.792000+00:00",
        "main": 221.2,
        "month": "2026-06",
        "neg": -237.3
      }
    ],
    "read": "If negative control is strongly negative at causal entry, the state is directional; if it is tail-only or month-only, treat as regime artefact."
  },
  "SELL_SILENCE_FADE_LONG_H4": {
    "main_minus_negative": {
      "max_bps": 1001.7,
      "mean_bps": 33.8,
      "median_bps": 46.2,
      "min_bps": -972.7,
      "n": 131,
      "sum_bps": 4432.6,
      "t3r_bps": 2089.9,
      "tail_lt_-100_n": 27,
      "win_rate": 0.611
    },
    "negative_all": {
      "max_bps": 478.3,
      "mean_bps": -25.0,
      "median_bps": -31.2,
      "min_bps": -508.9,
      "n": 131,
      "sum_bps": -3271.8,
      "t3r_bps": -4632.5,
      "tail_lt_-100_n": 23,
      "win_rate": 0.351
    },
    "negative_by_month": {
      "2026-04": {
        "max_bps": 217.4,
        "mean_bps": -22.2,
        "median_bps": -13.0,
        "min_bps": -365.0,
        "n": 50,
        "sum_bps": -1108.0,
        "t3r_bps": -1692.1,
        "tail_lt_-100_n": 9,
        "win_rate": 0.4
      },
      "2026-06": {
        "max_bps": 478.3,
        "mean_bps": -26.7,
        "median_bps": -34.4,
        "min_bps": -508.9,
        "n": 81,
        "sum_bps": -2163.8,
        "t3r_bps": -3524.5,
        "tail_lt_-100_n": 14,
        "win_rate": 0.321
      }
    },
    "negative_tail_lt_-100_examples": [
      {
        "anchor_utc": "2026-04-13T12:42:33.054000+00:00",
        "main": 133.3,
        "month": "2026-04",
        "neg": -149.4
      },
      {
        "anchor_utc": "2026-04-13T15:30:57.123000+00:00",
        "main": 229.3,
        "month": "2026-04",
        "neg": -245.4
      },
      {
        "anchor_utc": "2026-04-16T14:16:36.805000+00:00",
        "main": 125.7,
        "month": "2026-04",
        "neg": -141.8
      },
      {
        "anchor_utc": "2026-04-16T16:52:48.346000+00:00",
        "main": 93.8,
        "month": "2026-04",
        "neg": -109.9
      },
      {
        "anchor_utc": "2026-04-20T05:10:09.389000+00:00",
        "main": 186.9,
        "month": "2026-04",
        "neg": -203.6
      },
      {
        "anchor_utc": "2026-04-21T22:23:13.928000+00:00",
        "main": 231.3,
        "month": "2026-04",
        "neg": -247.4
      },
      {
        "anchor_utc": "2026-04-22T01:19:12.462000+00:00",
        "main": 348.9,
        "month": "2026-04",
        "neg": -365.0
      },
      {
        "anchor_utc": "2026-04-26T21:33:07.357000+00:00",
        "main": 87.2,
        "month": "2026-04",
        "neg": -103.3
      },
      {
        "anchor_utc": "2026-04-26T22:34:24.614000+00:00",
        "main": 135.9,
        "month": "2026-04",
        "neg": -151.9
      },
      {
        "anchor_utc": "2026-06-11T14:11:15.226000+00:00",
        "main": 204.6,
        "month": "2026-06",
        "neg": -220.7
      }
    ],
    "read": "If negative control is strongly negative at causal entry, the state is directional; if it is tail-only or month-only, treat as regime artefact."
  }
}
```

## Read

- All entries are re-anchored to the first causal detection time: anchor + tau. This is stricter than the broad anchor-mark gauntlet.
- Taker0 is the cleanest executable proxy; maker rows are pullback-fill proxies and still need real queue replay.
- If a candidate dies after causal re-anchoring, the broad result was mostly an early-label/entry-price effect.
- Negative-control anatomy checks whether the large negative control is broad directional evidence or just regime/tail artefact.
- No candidate clears causal execution gates. Keep as navigation until stronger evidence appears.
