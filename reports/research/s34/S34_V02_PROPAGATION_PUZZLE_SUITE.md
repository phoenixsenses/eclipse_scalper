# S34 V02 Propagation Puzzle Suite

Generated: `2026-06-30T08:18:38.535057+00:00`

Research-only. No live executor/config/order logic is touched.

## Verdict

- `PROPAGATION_STATE_IS_REAL_NAVIGATION_NOT_LIVE_ALPHA_YET`

## 1. Propagation Predictor

Overall next-same-side rate: `0.342`

## 2. Momentum vs Fade Alpha

```json
{
  "BUY": {
    "all_momentum_h1": {
      "max_bps": 365.8,
      "mean_bps": -6.2,
      "median_bps": -13.3,
      "min_bps": -402.8,
      "n": 557,
      "sum_bps": -3479.0,
      "t3r_bps": -4492.9,
      "tail_lt_-100_n": 65,
      "win_rate": 0.406
    },
    "all_momentum_h4": {
      "max_bps": 655.3,
      "mean_bps": 7.8,
      "median_bps": -12.9,
      "min_bps": -521.6,
      "n": 556,
      "sum_bps": 4322.6,
      "t3r_bps": 2523.6,
      "tail_lt_-100_n": 114,
      "win_rate": 0.453
    },
    "no_propagation_fade_h4": {
      "max_bps": 505.6,
      "mean_bps": -3.5,
      "median_bps": 12.0,
      "min_bps": -671.3,
      "n": 279,
      "sum_bps": -969.5,
      "t3r_bps": -2073.0,
      "tail_lt_-100_n": 47,
      "win_rate": 0.566
    },
    "no_propagation_n": 280,
    "propagation_fade_h4": {
      "max_bps": 460.4,
      "mean_bps": -44.2,
      "median_bps": -29.1,
      "min_bps": -591.7,
      "n": 277,
      "sum_bps": -12249.1,
      "t3r_bps": -13586.4,
      "tail_lt_-100_n": 83,
      "win_rate": 0.415
    },
    "propagation_momentum_h1": {
      "max_bps": 365.8,
      "mean_bps": 21.4,
      "median_bps": 8.6,
      "min_bps": -402.8,
      "n": 277,
      "sum_bps": 5934.5,
      "t3r_bps": 4920.6,
      "tail_lt_-100_n": 26,
      "win_rate": 0.542
    },
    "propagation_momentum_h4": {
      "max_bps": 575.7,
      "mean_bps": 28.2,
      "median_bps": 13.1,
      "min_bps": -476.4,
      "n": 277,
      "sum_bps": 7817.1,
      "t3r_bps": 6215.7,
      "tail_lt_-100_n": 50,
      "win_rate": 0.549
    },
    "propagation_n": 277
  },
  "SELL": {
    "all_momentum_h1": {
      "max_bps": 516.1,
      "mean_bps": -10.7,
      "median_bps": -17.9,
      "min_bps": -500.7,
      "n": 585,
      "sum_bps": -6248.8,
      "t3r_bps": -7493.8,
      "tail_lt_-100_n": 55,
      "win_rate": 0.356
    },
    "all_momentum_h4": {
      "max_bps": 524.8,
      "mean_bps": -14.6,
      "median_bps": -27.3,
      "min_bps": -627.3,
      "n": 585,
      "sum_bps": -8533.0,
      "t3r_bps": -10039.8,
      "tail_lt_-100_n": 139,
      "win_rate": 0.398
    },
    "no_propagation_fade_h4": {
      "max_bps": 611.3,
      "mean_bps": 22.4,
      "median_bps": 20.6,
      "min_bps": -455.3,
      "n": 303,
      "sum_bps": 6785.2,
      "t3r_bps": 5448.3,
      "tail_lt_-100_n": 40,
      "win_rate": 0.587
    },
    "no_propagation_n": 303,
    "propagation_fade_h4": {
      "max_bps": 405.7,
      "mean_bps": -27.0,
      "median_bps": -6.0,
      "min_bps": -540.8,
      "n": 282,
      "sum_bps": -7612.2,
      "t3r_bps": -8782.7,
      "tail_lt_-100_n": 74,
      "win_rate": 0.472
    },
    "propagation_momentum_h1": {
      "max_bps": 516.1,
      "mean_bps": 20.9,
      "median_bps": 1.6,
      "min_bps": -290.0,
      "n": 282,
      "sum_bps": 5895.2,
      "t3r_bps": 4650.2,
      "tail_lt_-100_n": 14,
      "win_rate": 0.514
    },
    "propagation_momentum_h4": {
      "max_bps": 524.8,
      "mean_bps": 11.0,
      "median_bps": -10.0,
      "min_bps": -421.7,
      "n": 282,
      "sum_bps": 3100.2,
      "t3r_bps": 1593.4,
      "tail_lt_-100_n": 57,
      "win_rate": 0.461
    },
    "propagation_n": 282
  }
}
```

## 3. First / Second / Third Chain Rank

```json
{
  "BUY": {
    "1": {
      "fade_h4": {
        "max_bps": 505.6,
        "mean_bps": -23.8,
        "median_bps": -3.2,
        "min_bps": -671.3,
        "n": 556,
        "sum_bps": -13218.6,
        "t3r_bps": -14642.4,
        "tail_lt_-100_n": 130,
        "win_rate": 0.491
      },
      "momentum_h1": {
        "max_bps": 365.8,
        "mean_bps": -6.2,
        "median_bps": -13.3,
        "min_bps": -402.8,
        "n": 557,
        "sum_bps": -3479.0,
        "t3r_bps": -4492.9,
        "tail_lt_-100_n": 65,
        "win_rate": 0.406
      },
      "momentum_h4": {
        "max_bps": 655.3,
        "mean_bps": 7.8,
        "median_bps": -12.9,
        "min_bps": -521.6,
        "n": 556,
        "sum_bps": 4322.6,
        "t3r_bps": 2523.6,
        "tail_lt_-100_n": 114,
        "win_rate": 0.453
      },
      "n": 557,
      "next_same_rate": 0.357
    },
    "2": {
      "fade_h4": {
        "max_bps": null,
        "mean_bps": null,
        "median_bps": null,
        "min_bps": null,
        "n": 0,
        "sum_bps": 0.0,
        "t3r_bps": 0.0,
        "tail_lt_-100_n": 0,
        "win_rate": null
      },
      "momentum_h1": {
        "max_bps": null,
        "mean_bps": null,
        "median_bps": null,
        "min_bps": null,
        "n": 0,
        "sum_bps": 0.0,
        "t3r_bps": 0.0,
        "tail_lt_-100_n": 0,
        "win_rate": null
      },
      "momentum_h4": {
        "max_bps": null,
        "mean_bps": null,
        "median_bps": null,
        "min_bps": null,
        "n": 0,
        "sum_bps": 0.0,
        "t3r_bps": 0.0,
        "tail_lt_-100_n": 0,
        "win_rate": null
      },
      "n": 0,
      "next_same_rate": null
    },
    "3+": {
      "fade_h4": {
        "max_bps": null,
        "mean_bps": null,
        "median_bps": null,
        "min_bps": null,
        "n": 0,
        "sum_bps": 0.0,
        "t3r_bps": 0.0,
        "tail_lt_-100_n": 0,
        "win_rate": null
      },
      "momentum_h1": {
        "max_bps": null,
        "mean_bps": null,
        "median_bps": null,
        "min_bps": null,
        "n": 0,
        "sum_bps": 0.0,
        "t3r_bps": 0.0,
        "tail_lt_-100_n": 0,
        "win_rate": null
      },
      "momentum_h4": {
        "max_bps": null,
        "mean_bps": null,
        "median_bps": null,
        "min_bps": null,
        "n": 0,
        "sum_bps": 0.0,
        "t3r_bps": 0.0,
        "tail_lt_-100_n": 0,
        "win_rate": null
      },
      "n": 0,
      "next_same_rate": null
    }
  },
  "SELL": {
    "1": {
      "fade_h4": {
        "max_bps": 611.3,
        "mean_bps": -1.4,
        "median_bps": 11.3,
        "min_bps": -540.8,
        "n": 585,
        "sum_bps": -827.0,
        "t3r_bps": -2230.8,
        "tail_lt_-100_n": 114,
        "win_rate": 0.532
      },
      "momentum_h1": {
        "max_bps": 516.1,
        "mean_bps": -10.7,
        "median_bps": -17.9,
        "min_bps": -500.7,
        "n": 585,
        "sum_bps": -6248.8,
        "t3r_bps": -7493.8,
        "tail_lt_-100_n": 55,
        "win_rate": 0.356
      },
      "momentum_h4": {
        "max_bps": 524.8,
        "mean_bps": -14.6,
        "median_bps": -27.3,
        "min_bps": -627.3,
        "n": 585,
        "sum_bps": -8533.0,
        "t3r_bps": -10039.8,
        "tail_lt_-100_n": 139,
        "win_rate": 0.398
      },
      "n": 585,
      "next_same_rate": 0.326
    },
    "2": {
      "fade_h4": {
        "max_bps": null,
        "mean_bps": null,
        "median_bps": null,
        "min_bps": null,
        "n": 0,
        "sum_bps": 0.0,
        "t3r_bps": 0.0,
        "tail_lt_-100_n": 0,
        "win_rate": null
      },
      "momentum_h1": {
        "max_bps": null,
        "mean_bps": null,
        "median_bps": null,
        "min_bps": null,
        "n": 0,
        "sum_bps": 0.0,
        "t3r_bps": 0.0,
        "tail_lt_-100_n": 0,
        "win_rate": null
      },
      "momentum_h4": {
        "max_bps": null,
        "mean_bps": null,
        "median_bps": null,
        "min_bps": null,
        "n": 0,
        "sum_bps": 0.0,
        "t3r_bps": 0.0,
        "tail_lt_-100_n": 0,
        "win_rate": null
      },
      "n": 0,
      "next_same_rate": null
    },
    "3+": {
      "fade_h4": {
        "max_bps": null,
        "mean_bps": null,
        "median_bps": null,
        "min_bps": null,
        "n": 0,
        "sum_bps": 0.0,
        "t3r_bps": 0.0,
        "tail_lt_-100_n": 0,
        "win_rate": null
      },
      "momentum_h1": {
        "max_bps": null,
        "mean_bps": null,
        "median_bps": null,
        "min_bps": null,
        "n": 0,
        "sum_bps": 0.0,
        "t3r_bps": 0.0,
        "tail_lt_-100_n": 0,
        "win_rate": null
      },
      "momentum_h4": {
        "max_bps": null,
        "mean_bps": null,
        "median_bps": null,
        "min_bps": null,
        "n": 0,
        "sum_bps": 0.0,
        "t3r_bps": 0.0,
        "tail_lt_-100_n": 0,
        "win_rate": null
      },
      "n": 0,
      "next_same_rate": null
    }
  }
}
```

## 4. Cross-Asset Propagation Timing

```json
{
  "BUY": {
    "cross_next_same_1800s": {
      "false_fade_h4": {
        "max_bps": 505.6,
        "mean_bps": -8.3,
        "median_bps": 7.0,
        "min_bps": -671.3,
        "n": 372,
        "sum_bps": -3079.4,
        "t3r_bps": -4364.1,
        "tail_lt_-100_n": 71,
        "win_rate": 0.54
      },
      "false_momentum_h1": {
        "max_bps": 365.8,
        "mean_bps": -19.9,
        "median_bps": -21.6,
        "min_bps": -402.8,
        "n": 373,
        "sum_bps": -7410.7,
        "t3r_bps": -8404.7,
        "tail_lt_-100_n": 50,
        "win_rate": 0.34
      },
      "true_fade_h4": {
        "max_bps": 460.4,
        "mean_bps": -55.1,
        "median_bps": -45.0,
        "min_bps": -591.7,
        "n": 184,
        "sum_bps": -10139.2,
        "t3r_bps": -11476.5,
        "tail_lt_-100_n": 59,
        "win_rate": 0.391
      },
      "true_momentum_h1": {
        "max_bps": 326.8,
        "mean_bps": 21.4,
        "median_bps": 7.1,
        "min_bps": -276.3,
        "n": 184,
        "sum_bps": 3931.7,
        "t3r_bps": 3040.7,
        "tail_lt_-100_n": 15,
        "win_rate": 0.538
      }
    },
    "cross_next_same_300s": {
      "false_fade_h4": {
        "max_bps": 505.6,
        "mean_bps": -16.9,
        "median_bps": 2.4,
        "min_bps": -671.3,
        "n": 458,
        "sum_bps": -7725.1,
        "t3r_bps": -9104.2,
        "tail_lt_-100_n": 97,
        "win_rate": 0.515
      },
      "false_momentum_h1": {
        "max_bps": 365.8,
        "mean_bps": -12.1,
        "median_bps": -16.7,
        "min_bps": -402.8,
        "n": 459,
        "sum_bps": -5567.1,
        "t3r_bps": -6581.0,
        "tail_lt_-100_n": 56,
        "win_rate": 0.379
      },
      "true_fade_h4": {
        "max_bps": 457.8,
        "mean_bps": -56.1,
        "median_bps": -33.9,
        "min_bps": -591.7,
        "n": 98,
        "sum_bps": -5493.5,
        "t3r_bps": -6703.3,
        "tail_lt_-100_n": 33,
        "win_rate": 0.378
      },
      "true_momentum_h1": {
        "max_bps": 289.8,
        "mean_bps": 21.3,
        "median_bps": 7.6,
        "min_bps": -276.3,
        "n": 98,
        "sum_bps": 2088.1,
        "t3r_bps": 1270.0,
        "tail_lt_-100_n": 9,
        "win_rate": 0.531
      }
    },
    "cross_next_same_3600s": {
      "false_fade_h4": {
        "max_bps": 505.6,
        "mean_bps": -3.2,
        "median_bps": 10.5,
        "min_bps": -671.3,
        "n": 334,
        "sum_bps": -1056.4,
        "t3r_bps": -2239.0,
        "tail_lt_-100_n": 57,
        "win_rate": 0.56
      },
      "false_momentum_h1": {
        "max_bps": 321.3,
        "mean_bps": -26.7,
        "median_bps": -24.9,
        "min_bps": -402.8,
        "n": 335,
        "sum_bps": -8934.7,
        "t3r_bps": -9781.9,
        "tail_lt_-100_n": 46,
        "win_rate": 0.31
      },
      "true_fade_h4": {
        "max_bps": 460.4,
        "mean_bps": -54.8,
        "median_bps": -43.6,
        "min_bps": -591.7,
        "n": 222,
        "sum_bps": -12162.2,
        "t3r_bps": -13499.5,
        "tail_lt_-100_n": 73,
        "win_rate": 0.387
      },
      "true_momentum_h1": {
        "max_bps": 365.8,
        "mean_bps": 24.6,
        "median_bps": 8.6,
        "min_bps": -276.3,
        "n": 222,
        "sum_bps": 5455.7,
        "t3r_bps": 4473.3,
        "tail_lt_-100_n": 19,
        "win_rate": 0.55
      }
    },
    "cross_next_same_900s": {
      "false_fade_h4": {
        "max_bps": 505.6,
        "mean_bps": -13.8,
        "median_bps": 3.3,
        "min_bps": -671.3,
        "n": 410,
        "sum_bps": -5672.3,
        "t3r_bps": -6957.0,
        "tail_lt_-100_n": 85,
        "win_rate": 0.524
      },
      "false_momentum_h1": {
        "max_bps": 365.8,
        "mean_bps": -14.9,
        "median_bps": -18.6,
        "min_bps": -402.8,
        "n": 411,
        "sum_bps": -6127.7,
        "t3r_bps": -7121.7,
        "tail_lt_-100_n": 53,
        "win_rate": 0.372
      },
      "true_fade_h4": {
        "max_bps": 460.4,
        "mean_bps": -51.7,
        "median_bps": -39.1,
        "min_bps": -591.7,
        "n": 146,
        "sum_bps": -7546.3,
        "t3r_bps": -8883.6,
        "tail_lt_-100_n": 45,
        "win_rate": 0.397
      },
      "true_momentum_h1": {
        "max_bps": 326.8,
        "mean_bps": 18.1,
        "median_bps": -0.4,
        "min_bps": -276.3,
        "n": 146,
        "sum_bps": 2648.7,
        "t3r_bps": 1757.7,
        "tail_lt_-100_n": 12,
        "win_rate": 0.5
      }
    }
  },
  "SELL": {
    "cross_next_same_1800s": {
      "false_fade_h4": {
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
      "false_momentum_h1": {
        "max_bps": 325.7,
        "mean_bps": -27.0,
        "median_bps": -28.1,
        "min_bps": -500.7,
        "n": 395,
        "sum_bps": -10652.8,
        "t3r_bps": -11449.8,
        "tail_lt_-100_n": 46,
        "win_rate": 0.284
      },
      "true_fade_h4": {
        "max_bps": 405.7,
        "mean_bps": -43.6,
        "median_bps": -18.3,
        "min_bps": -540.8,
        "n": 190,
        "sum_bps": -8283.9,
        "t3r_bps": -9434.6,
        "tail_lt_-100_n": 56,
        "win_rate": 0.437
      },
      "true_momentum_h1": {
        "max_bps": 516.1,
        "mean_bps": 23.2,
        "median_bps": 0.2,
        "min_bps": -290.0,
        "n": 190,
        "sum_bps": 4404.0,
        "t3r_bps": 3159.0,
        "tail_lt_-100_n": 9,
        "win_rate": 0.505
      }
    },
    "cross_next_same_300s": {
      "false_fade_h4": {
        "max_bps": 611.3,
        "mean_bps": 6.2,
        "median_bps": 15.2,
        "min_bps": -540.8,
        "n": 464,
        "sum_bps": 2894.5,
        "t3r_bps": 1526.5,
        "tail_lt_-100_n": 82,
        "win_rate": 0.554
      },
      "false_momentum_h1": {
        "max_bps": 380.6,
        "mean_bps": -15.5,
        "median_bps": -22.9,
        "min_bps": -500.7,
        "n": 464,
        "sum_bps": -7169.3,
        "t3r_bps": -8223.9,
        "tail_lt_-100_n": 47,
        "win_rate": 0.334
      },
      "true_fade_h4": {
        "max_bps": 405.7,
        "mean_bps": -30.8,
        "median_bps": -7.1,
        "min_bps": -514.1,
        "n": 121,
        "sum_bps": -3721.5,
        "t3r_bps": -4872.2,
        "tail_lt_-100_n": 32,
        "win_rate": 0.446
      },
      "true_momentum_h1": {
        "max_bps": 516.1,
        "mean_bps": 7.6,
        "median_bps": -5.0,
        "min_bps": -290.0,
        "n": 121,
        "sum_bps": 920.5,
        "t3r_bps": -71.4,
        "tail_lt_-100_n": 8,
        "win_rate": 0.438
      }
    },
    "cross_next_same_3600s": {
      "false_fade_h4": {
        "max_bps": 611.3,
        "mean_bps": 25.1,
        "median_bps": 27.4,
        "min_bps": -455.3,
        "n": 362,
        "sum_bps": 9071.9,
        "t3r_bps": 7703.9,
        "tail_lt_-100_n": 47,
        "win_rate": 0.597
      },
      "false_momentum_h1": {
        "max_bps": 253.0,
        "mean_bps": -34.4,
        "median_bps": -31.6,
        "min_bps": -500.7,
        "n": 362,
        "sum_bps": -12445.5,
        "t3r_bps": -13071.4,
        "tail_lt_-100_n": 45,
        "win_rate": 0.246
      },
      "true_fade_h4": {
        "max_bps": 405.7,
        "mean_bps": -44.4,
        "median_bps": -18.7,
        "min_bps": -540.8,
        "n": 223,
        "sum_bps": -9898.9,
        "t3r_bps": -11049.6,
        "tail_lt_-100_n": 67,
        "win_rate": 0.426
      },
      "true_momentum_h1": {
        "max_bps": 516.1,
        "mean_bps": 27.8,
        "median_bps": 2.8,
        "min_bps": -290.0,
        "n": 223,
        "sum_bps": 6196.7,
        "t3r_bps": 4951.7,
        "tail_lt_-100_n": 10,
        "win_rate": 0.534
      }
    },
    "cross_next_same_900s": {
      "false_fade_h4": {
        "max_bps": 611.3,
        "mean_bps": 11.3,
        "median_bps": 16.2,
        "min_bps": -495.8,
        "n": 425,
        "sum_bps": 4806.5,
        "t3r_bps": 3438.5,
        "tail_lt_-100_n": 70,
        "win_rate": 0.558
      },
      "false_momentum_h1": {
        "max_bps": 348.3,
        "mean_bps": -20.5,
        "median_bps": -24.4,
        "min_bps": -500.7,
        "n": 425,
        "sum_bps": -8697.4,
        "t3r_bps": -9624.4,
        "tail_lt_-100_n": 47,
        "win_rate": 0.318
      },
      "true_fade_h4": {
        "max_bps": 405.7,
        "mean_bps": -35.2,
        "median_bps": -6.5,
        "min_bps": -540.8,
        "n": 160,
        "sum_bps": -5633.5,
        "t3r_bps": -6784.2,
        "tail_lt_-100_n": 44,
        "win_rate": 0.463
      },
      "true_momentum_h1": {
        "max_bps": 516.1,
        "mean_bps": 15.3,
        "median_bps": -3.9,
        "min_bps": -290.0,
        "n": 160,
        "sum_bps": 2448.6,
        "t3r_bps": 1271.2,
        "tail_lt_-100_n": 8,
        "win_rate": 0.456
      }
    }
  }
}
```

## 5. Silence After Shock

```json
{
  "BUY": {
    "noisy_fade_h4": {
      "max_bps": 460.4,
      "mean_bps": -48.4,
      "median_bps": -29.5,
      "min_bps": -591.7,
      "n": 297,
      "sum_bps": -14376.8,
      "t3r_bps": -15714.1,
      "tail_lt_-100_n": 91,
      "win_rate": 0.411
    },
    "noisy_momentum_h1": {
      "max_bps": 365.8,
      "mean_bps": 23.5,
      "median_bps": 9.9,
      "min_bps": -402.8,
      "n": 297,
      "sum_bps": 6992.3,
      "t3r_bps": 5978.4,
      "tail_lt_-100_n": 27,
      "win_rate": 0.562
    },
    "noisy_n": 297,
    "silence_fade_h4": {
      "max_bps": 505.6,
      "mean_bps": 4.5,
      "median_bps": 14.5,
      "min_bps": -671.3,
      "n": 259,
      "sum_bps": 1158.2,
      "t3r_bps": 54.7,
      "tail_lt_-100_n": 39,
      "win_rate": 0.583
    },
    "silence_momentum_h1": {
      "max_bps": 155.8,
      "mean_bps": -40.3,
      "median_bps": -31.2,
      "min_bps": -296.4,
      "n": 260,
      "sum_bps": -10471.3,
      "t3r_bps": -10834.7,
      "tail_lt_-100_n": 38,
      "win_rate": 0.227
    },
    "silence_n": 260
  },
  "SELL": {
    "noisy_fade_h4": {
      "max_bps": 405.7,
      "mean_bps": -28.8,
      "median_bps": -7.3,
      "min_bps": -540.8,
      "n": 298,
      "sum_bps": -8570.1,
      "t3r_bps": -9740.6,
      "tail_lt_-100_n": 79,
      "win_rate": 0.463
    },
    "noisy_momentum_h1": {
      "max_bps": 516.1,
      "mean_bps": 22.5,
      "median_bps": 3.6,
      "min_bps": -290.0,
      "n": 298,
      "sum_bps": 6709.0,
      "t3r_bps": 5464.0,
      "tail_lt_-100_n": 14,
      "win_rate": 0.534
    },
    "noisy_n": 298,
    "silence_fade_h4": {
      "max_bps": 611.3,
      "mean_bps": 27.0,
      "median_bps": 24.2,
      "min_bps": -455.3,
      "n": 287,
      "sum_bps": 7743.1,
      "t3r_bps": 6406.2,
      "tail_lt_-100_n": 35,
      "win_rate": 0.603
    },
    "silence_momentum_h1": {
      "max_bps": 117.2,
      "mean_bps": -45.1,
      "median_bps": -39.4,
      "min_bps": -500.7,
      "n": 287,
      "sum_bps": -12957.8,
      "t3r_bps": -13260.7,
      "tail_lt_-100_n": 41,
      "win_rate": 0.171
    },
    "silence_n": 287
  }
}
```

## 6. BUY Side Diagnosis

```json
{
  "buy_buy_propagation_long_h1": {
    "max_bps": 365.8,
    "mean_bps": 38.6,
    "median_bps": 26.6,
    "min_bps": -402.8,
    "n": 199,
    "sum_bps": 7674.0,
    "t3r_bps": 6660.1,
    "tail_lt_-100_n": 16,
    "win_rate": 0.628
  },
  "buy_buy_propagation_short_fade_h4": {
    "max_bps": 457.8,
    "mean_bps": -61.7,
    "median_bps": -44.5,
    "min_bps": -591.7,
    "n": 199,
    "sum_bps": -12283.2,
    "t3r_bps": -13520.1,
    "tail_lt_-100_n": 67,
    "win_rate": 0.372
  },
  "buy_continuation_long_h1": {
    "max_bps": 365.8,
    "mean_bps": -6.2,
    "median_bps": -13.3,
    "min_bps": -402.8,
    "n": 557,
    "sum_bps": -3479.0,
    "t3r_bps": -4492.9,
    "tail_lt_-100_n": 65,
    "win_rate": 0.406
  },
  "buy_continuation_long_h4": {
    "max_bps": 655.3,
    "mean_bps": 7.8,
    "median_bps": -12.9,
    "min_bps": -521.6,
    "n": 556,
    "sum_bps": 4322.6,
    "t3r_bps": 2523.6,
    "tail_lt_-100_n": 114,
    "win_rate": 0.453
  },
  "buy_fade_short_h4": {
    "max_bps": 505.6,
    "mean_bps": -23.8,
    "median_bps": -3.2,
    "min_bps": -671.3,
    "n": 556,
    "sum_bps": -13218.6,
    "t3r_bps": -14642.4,
    "tail_lt_-100_n": 130,
    "win_rate": 0.491
  },
  "buy_silence_short_fade_h4": {
    "max_bps": 505.6,
    "mean_bps": 4.5,
    "median_bps": 14.5,
    "min_bps": -671.3,
    "n": 259,
    "sum_bps": 1158.2,
    "t3r_bps": 54.7,
    "tail_lt_-100_n": 39,
    "win_rate": 0.583
  }
}
```

## 7. Composite Propagation Indicator

```json
{
  "BUY": {
    "HIGH_4_PLUS": {
      "fade_h4": {
        "max_bps": 457.8,
        "mean_bps": -46.6,
        "median_bps": -29.1,
        "min_bps": -591.7,
        "n": 273,
        "sum_bps": -12714.2,
        "t3r_bps": -14004.2,
        "tail_lt_-100_n": 83,
        "win_rate": 0.414
      },
      "momentum_h1": {
        "max_bps": 365.8,
        "mean_bps": 21.8,
        "median_bps": 9.4,
        "min_bps": -402.8,
        "n": 273,
        "sum_bps": 5955.2,
        "t3r_bps": 4941.3,
        "tail_lt_-100_n": 26,
        "win_rate": 0.542
      },
      "n": 273,
      "next_same_rate": 0.722
    },
    "LOW_0_1": {
      "fade_h4": {
        "max_bps": 311.0,
        "mean_bps": 12.4,
        "median_bps": 14.6,
        "min_bps": -388.8,
        "n": 82,
        "sum_bps": 1017.0,
        "t3r_bps": 318.3,
        "tail_lt_-100_n": 9,
        "win_rate": 0.561
      },
      "momentum_h1": {
        "max_bps": 108.6,
        "mean_bps": -35.9,
        "median_bps": -33.6,
        "min_bps": -223.1,
        "n": 83,
        "sum_bps": -2975.8,
        "t3r_bps": -3198.6,
        "tail_lt_-100_n": 7,
        "win_rate": 0.229
      },
      "n": 83,
      "next_same_rate": 0.0
    },
    "MID_2_3": {
      "fade_h4": {
        "max_bps": 505.6,
        "mean_bps": -7.6,
        "median_bps": 10.2,
        "min_bps": -671.3,
        "n": 201,
        "sum_bps": -1521.4,
        "t3r_bps": -2774.3,
        "tail_lt_-100_n": 38,
        "win_rate": 0.567
      },
      "momentum_h1": {
        "max_bps": 199.6,
        "mean_bps": -32.1,
        "median_bps": -27.1,
        "min_bps": -296.4,
        "n": 201,
        "sum_bps": -6458.4,
        "t3r_bps": -6976.0,
        "tail_lt_-100_n": 32,
        "win_rate": 0.294
      },
      "n": 201,
      "next_same_rate": 0.01
    }
  },
  "SELL": {
    "HIGH_4_PLUS": {
      "fade_h4": {
        "max_bps": 405.7,
        "mean_bps": -28.4,
        "median_bps": -7.2,
        "min_bps": -540.8,
        "n": 276,
        "sum_bps": -7832.4,
        "t3r_bps": -9002.9,
        "tail_lt_-100_n": 73,
        "win_rate": 0.464
      },
      "momentum_h1": {
        "max_bps": 516.1,
        "mean_bps": 21.5,
        "median_bps": 1.9,
        "min_bps": -290.0,
        "n": 276,
        "sum_bps": 5943.6,
        "t3r_bps": 4698.6,
        "tail_lt_-100_n": 14,
        "win_rate": 0.522
      },
      "n": 276,
      "next_same_rate": 0.678
    },
    "LOW_0_1": {
      "fade_h4": {
        "max_bps": 355.7,
        "mean_bps": 30.9,
        "median_bps": 31.1,
        "min_bps": -331.5,
        "n": 97,
        "sum_bps": 2993.0,
        "t3r_bps": 1958.8,
        "tail_lt_-100_n": 11,
        "win_rate": 0.619
      },
      "momentum_h1": {
        "max_bps": 78.0,
        "mean_bps": -46.0,
        "median_bps": -36.1,
        "min_bps": -500.7,
        "n": 97,
        "sum_bps": -4464.7,
        "t3r_bps": -4684.7,
        "tail_lt_-100_n": 13,
        "win_rate": 0.196
      },
      "n": 97,
      "next_same_rate": 0.0
    },
    "MID_2_3": {
      "fade_h4": {
        "max_bps": 611.3,
        "mean_bps": 18.9,
        "median_bps": 16.1,
        "min_bps": -455.3,
        "n": 212,
        "sum_bps": 4012.4,
        "t3r_bps": 2677.3,
        "tail_lt_-100_n": 30,
        "win_rate": 0.58
      },
      "momentum_h1": {
        "max_bps": 253.0,
        "mean_bps": -36.5,
        "median_bps": -33.0,
        "min_bps": -341.6,
        "n": 212,
        "sum_bps": -7727.7,
        "t3r_bps": -8313.9,
        "tail_lt_-100_n": 28,
        "win_rate": 0.212
      },
      "n": 212,
      "next_same_rate": 0.019
    }
  }
}
```

## 8. Transition Matrix Navigation

```json
{
  "BUY->BUY": {
    "fade_h4": {
      "max_bps": 413.1,
      "mean_bps": -78.6,
      "median_bps": -53.3,
      "min_bps": -671.3,
      "n": 261,
      "sum_bps": -20520.3,
      "t3r_bps": -21425.3,
      "tail_lt_-100_n": 93,
      "win_rate": 0.314
    },
    "momentum_h1": {
      "max_bps": 365.8,
      "mean_bps": 29.7,
      "median_bps": 9.6,
      "min_bps": -181.9,
      "n": 261,
      "sum_bps": 7762.7,
      "t3r_bps": 6748.8,
      "tail_lt_-100_n": 16,
      "win_rate": 0.582
    },
    "n": 261,
    "pressure_score": {
      "max_bps": 8.0,
      "mean_bps": 4.7,
      "median_bps": 5.0,
      "min_bps": 0.0,
      "n": 261,
      "sum_bps": 1214.0,
      "t3r_bps": 1190.0,
      "tail_lt_-100_n": 0,
      "win_rate": 0.981
    }
  },
  "BUY->SELL": {
    "fade_h4": {
      "max_bps": 505.6,
      "mean_bps": 28.1,
      "median_bps": 38.7,
      "min_bps": -591.7,
      "n": 226,
      "sum_bps": 6359.0,
      "t3r_bps": 4935.2,
      "tail_lt_-100_n": 36,
      "win_rate": 0.677
    },
    "momentum_h1": {
      "max_bps": 289.8,
      "mean_bps": -41.8,
      "median_bps": -36.6,
      "min_bps": -402.8,
      "n": 226,
      "sum_bps": -9441.3,
      "t3r_bps": -10201.8,
      "tail_lt_-100_n": 46,
      "win_rate": 0.248
    },
    "n": 226,
    "pressure_score": {
      "max_bps": 8.0,
      "mean_bps": 3.0,
      "median_bps": 2.0,
      "min_bps": 0.0,
      "n": 226,
      "sum_bps": 675.0,
      "t3r_bps": 651.0,
      "tail_lt_-100_n": 0,
      "win_rate": 0.965
    }
  },
  "SELL->BUY": {
    "fade_h4": {
      "max_bps": 611.3,
      "mean_bps": 58.3,
      "median_bps": 53.8,
      "min_bps": -514.1,
      "n": 233,
      "sum_bps": 13589.9,
      "t3r_bps": 12186.1,
      "tail_lt_-100_n": 24,
      "win_rate": 0.704
    },
    "momentum_h1": {
      "max_bps": 325.7,
      "mean_bps": -38.6,
      "median_bps": -39.4,
      "min_bps": -500.7,
      "n": 233,
      "sum_bps": -9004.9,
      "t3r_bps": -9803.1,
      "tail_lt_-100_n": 43,
      "win_rate": 0.24
    },
    "n": 233,
    "pressure_score": {
      "max_bps": 8.0,
      "mean_bps": 2.9,
      "median_bps": 2.0,
      "min_bps": 0.0,
      "n": 233,
      "sum_bps": 677.0,
      "t3r_bps": 653.0,
      "tail_lt_-100_n": 0,
      "win_rate": 0.966
    }
  },
  "SELL->SELL": {
    "fade_h4": {
      "max_bps": 350.4,
      "mean_bps": -53.2,
      "median_bps": -28.5,
      "min_bps": -540.8,
      "n": 283,
      "sum_bps": -15052.0,
      "t3r_bps": -15978.5,
      "tail_lt_-100_n": 84,
      "win_rate": 0.371
    },
    "momentum_h1": {
      "max_bps": 516.1,
      "mean_bps": 14.0,
      "median_bps": -3.7,
      "min_bps": -290.0,
      "n": 283,
      "sum_bps": 3967.7,
      "t3r_bps": 2722.7,
      "tail_lt_-100_n": 10,
      "win_rate": 0.466
    },
    "n": 283,
    "pressure_score": {
      "max_bps": 8.0,
      "mean_bps": 4.5,
      "median_bps": 5.0,
      "min_bps": 0.0,
      "n": 283,
      "sum_bps": 1265.0,
      "t3r_bps": 1241.0,
      "tail_lt_-100_n": 0,
      "win_rate": 0.986
    }
  }
}
```

## 9. V02 H4 Hold Decision

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
  "pressure_high": {
    "max_bps": 152.5,
    "mean_bps": 72.2,
    "median_bps": 95.4,
    "min_bps": -59.2,
    "n": 7,
    "sum_bps": 505.7,
    "t3r_bps": 67.9,
    "tail_lt_-100_n": 0,
    "win_rate": 0.714
  },
  "pressure_low_mid": {
    "max_bps": 223.2,
    "mean_bps": 37.8,
    "median_bps": -12.3,
    "min_bps": -47.3,
    "n": 4,
    "sum_bps": 151.3,
    "t3r_bps": -47.3,
    "tail_lt_-100_n": 0,
    "win_rate": 0.25
  },
  "same_side_next_false": {
    "max_bps": 223.2,
    "mean_bps": 49.3,
    "median_bps": -4.5,
    "min_bps": -47.3,
    "n": 5,
    "sum_bps": 246.7,
    "t3r_bps": -67.4,
    "tail_lt_-100_n": 0,
    "win_rate": 0.4
  },
  "same_side_next_true": {
    "max_bps": 152.5,
    "mean_bps": 68.4,
    "median_bps": 108.9,
    "min_bps": -59.2,
    "n": 6,
    "sum_bps": 410.3,
    "t3r_bps": -27.5,
    "tail_lt_-100_n": 0,
    "win_rate": 0.667
  },
  "silence_false": {
    "max_bps": 152.5,
    "mean_bps": 72.2,
    "median_bps": 95.4,
    "min_bps": -59.2,
    "n": 7,
    "sum_bps": 505.7,
    "t3r_bps": 67.9,
    "tail_lt_-100_n": 0,
    "win_rate": 0.714
  },
  "silence_true": {
    "max_bps": 223.2,
    "mean_bps": 37.8,
    "median_bps": -12.3,
    "min_bps": -47.3,
    "n": 4,
    "sum_bps": 151.3,
    "t3r_bps": -47.3,
    "tail_lt_-100_n": 0,
    "win_rate": 0.25
  }
}
```

## Read

- The negative SELL->SELL and BUY->BUY fade cells behave like same-side propagation/runaway states.
- The key practical split is not BUY vs SELL alone; it is propagation pressure versus silence/reclaim.
- Momentum tests here are mark-entry broad-event tests, not live-ready maker execution.
- For current V02, use these as navigation/management observers until forward filled N is large enough.
