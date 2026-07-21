# S34 V02 Next Navigation Tests

Generated: `2026-06-30T08:40:52.950480+00:00`

Research-only. No live executor/config/order logic is touched.

## Verdict

- `NAVIGATION_VALUE_CONFIRMED_EXECUTION_ALPHA_NOT_CONFIRMED`

## 1. Early Tau Sweep

```json
{
  "120": {
    "BUY": {
      "pressure_fade": {
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
      "pressure_momentum": {
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
      "silence_fade": {
        "max_bps": 451.0,
        "mean_bps": -27.3,
        "median_bps": -5.8,
        "min_bps": -541.3,
        "n": 118,
        "sum_bps": -3222.1,
        "t3r_bps": -4211.6,
        "tail_lt_-100_n": 22,
        "win_rate": 0.492
      }
    },
    "SELL": {
      "pressure_fade": {
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
      "pressure_momentum": {
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
      "silence_fade": {
        "max_bps": 535.8,
        "mean_bps": 15.6,
        "median_bps": 15.3,
        "min_bps": -436.6,
        "n": 128,
        "sum_bps": 2003.1,
        "t3r_bps": 836.8,
        "tail_lt_-100_n": 17,
        "win_rate": 0.594
      }
    },
    "state_counts": {
      "PRESSURE_HIGH": 0,
      "PRESSURE_LOW": 358,
      "PRESSURE_MID": 182,
      "SILENCE_RECLAIM": 603
    }
  },
  "30": {
    "BUY": {
      "pressure_fade": {
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
      "pressure_momentum": {
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
      "silence_fade": {
        "max_bps": 468.9,
        "mean_bps": -22.6,
        "median_bps": -2.3,
        "min_bps": -539.5,
        "n": 79,
        "sum_bps": -1783.0,
        "t3r_bps": -2803.3,
        "tail_lt_-100_n": 14,
        "win_rate": 0.494
      }
    },
    "SELL": {
      "pressure_fade": {
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
      "pressure_momentum": {
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
      "silence_fade": {
        "max_bps": 596.9,
        "mean_bps": 36.2,
        "median_bps": 29.9,
        "min_bps": -453.0,
        "n": 78,
        "sum_bps": 2827.3,
        "t3r_bps": 1586.3,
        "tail_lt_-100_n": 9,
        "win_rate": 0.667
      }
    },
    "state_counts": {
      "PRESSURE_HIGH": 0,
      "PRESSURE_LOW": 637,
      "PRESSURE_MID": 123,
      "SILENCE_RECLAIM": 383
    }
  },
  "300": {
    "BUY": {
      "pressure_fade": {
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
      "pressure_momentum": {
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
      "silence_fade": {
        "max_bps": 451.9,
        "mean_bps": -24.1,
        "median_bps": -4.0,
        "min_bps": -573.4,
        "n": 149,
        "sum_bps": -3590.2,
        "t3r_bps": -4647.3,
        "tail_lt_-100_n": 28,
        "win_rate": 0.483
      }
    },
    "SELL": {
      "pressure_fade": {
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
      "pressure_momentum": {
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
      "silence_fade": {
        "max_bps": 556.0,
        "mean_bps": 7.6,
        "median_bps": 10.9,
        "min_bps": -441.8,
        "n": 164,
        "sum_bps": 1238.3,
        "t3r_bps": 23.6,
        "tail_lt_-100_n": 26,
        "win_rate": 0.549
      }
    },
    "state_counts": {
      "PRESSURE_HIGH": 0,
      "PRESSURE_LOW": 148,
      "PRESSURE_MID": 220,
      "SILENCE_RECLAIM": 775
    }
  },
  "60": {
    "BUY": {
      "pressure_fade": {
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
      "pressure_momentum": {
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
      "silence_fade": {
        "max_bps": 463.8,
        "mean_bps": -20.0,
        "median_bps": -2.8,
        "min_bps": -533.9,
        "n": 94,
        "sum_bps": -1877.7,
        "t3r_bps": -2878.4,
        "tail_lt_-100_n": 17,
        "win_rate": 0.489
      }
    },
    "SELL": {
      "pressure_fade": {
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
      "pressure_momentum": {
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
      "silence_fade": {
        "max_bps": 564.9,
        "mean_bps": 28.1,
        "median_bps": 22.8,
        "min_bps": -434.3,
        "n": 101,
        "sum_bps": 2834.1,
        "t3r_bps": 1634.2,
        "tail_lt_-100_n": 12,
        "win_rate": 0.634
      }
    },
    "state_counts": {
      "PRESSURE_HIGH": 0,
      "PRESSURE_LOW": 518,
      "PRESSURE_MID": 158,
      "SILENCE_RECLAIM": 467
    }
  },
  "600": {
    "BUY": {
      "pressure_fade": {
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
      "pressure_momentum": {
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
      "silence_fade": {
        "max_bps": 261.2,
        "mean_bps": -19.5,
        "median_bps": -0.7,
        "min_bps": -634.6,
        "n": 143,
        "sum_bps": -2785.7,
        "t3r_bps": -3555.5,
        "tail_lt_-100_n": 23,
        "win_rate": 0.497
      }
    },
    "SELL": {
      "pressure_fade": {
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
      "pressure_momentum": {
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
      "silence_fade": {
        "max_bps": 580.6,
        "mean_bps": 10.7,
        "median_bps": 6.3,
        "min_bps": -477.2,
        "n": 160,
        "sum_bps": 1718.5,
        "t3r_bps": 475.7,
        "tail_lt_-100_n": 25,
        "win_rate": 0.537
      }
    },
    "state_counts": {
      "PRESSURE_HIGH": 0,
      "PRESSURE_LOW": 94,
      "PRESSURE_MID": 265,
      "SILENCE_RECLAIM": 784
    }
  },
  "900": {
    "BUY": {
      "pressure_fade": {
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
      "pressure_momentum": {
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
      "silence_fade": {
        "max_bps": 296.5,
        "mean_bps": -25.4,
        "median_bps": -6.2,
        "min_bps": -588.9,
        "n": 134,
        "sum_bps": -3405.8,
        "t3r_bps": -4231.9,
        "tail_lt_-100_n": 24,
        "win_rate": 0.463
      }
    },
    "SELL": {
      "pressure_fade": {
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
      "pressure_momentum": {
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
      "silence_fade": {
        "max_bps": 547.9,
        "mean_bps": 10.7,
        "median_bps": 2.8,
        "min_bps": -479.2,
        "n": 152,
        "sum_bps": 1625.2,
        "t3r_bps": 341.6,
        "tail_lt_-100_n": 18,
        "win_rate": 0.533
      }
    },
    "state_counts": {
      "PRESSURE_HIGH": 0,
      "PRESSURE_LOW": 78,
      "PRESSURE_MID": 307,
      "SILENCE_RECLAIM": 758
    }
  }
}
```

## 2. SELL Event-End / Reclaim Entry

```json
{
  "event_end_plus_0s": {
    "all": {
      "max_bps": 603.3,
      "mean_bps": 6.2,
      "median_bps": 18.1,
      "min_bps": -496.6,
      "n": 270,
      "sum_bps": 1669.6,
      "t3r_bps": 248.7,
      "tail_lt_-100_n": 46,
      "win_rate": 0.552
    },
    "fill_rate": 0.461,
    "state1800_silence": {
      "max_bps": 603.3,
      "mean_bps": 40.6,
      "median_bps": 33.7,
      "min_bps": -457.7,
      "n": 129,
      "sum_bps": 5241.5,
      "t3r_bps": 3902.8,
      "tail_lt_-100_n": 11,
      "win_rate": 0.651
    },
    "state900_silence": {
      "max_bps": 603.3,
      "mean_bps": 32.3,
      "median_bps": 27.9,
      "min_bps": -457.7,
      "n": 153,
      "sum_bps": 4935.0,
      "t3r_bps": 3596.3,
      "tail_lt_-100_n": 16,
      "win_rate": 0.621
    }
  },
  "event_end_plus_300s": {
    "all": {
      "max_bps": 554.8,
      "mean_bps": -3.3,
      "median_bps": 7.6,
      "min_bps": -513.9,
      "n": 266,
      "sum_bps": -873.5,
      "t3r_bps": -2261.1,
      "tail_lt_-100_n": 53,
      "win_rate": 0.534
    },
    "fill_rate": 0.454,
    "state1800_silence": {
      "max_bps": 554.8,
      "mean_bps": 24.3,
      "median_bps": 11.2,
      "min_bps": -438.7,
      "n": 125,
      "sum_bps": 3043.1,
      "t3r_bps": 1832.2,
      "tail_lt_-100_n": 16,
      "win_rate": 0.608
    },
    "state900_silence": {
      "max_bps": 554.8,
      "mean_bps": 15.9,
      "median_bps": 10.8,
      "min_bps": -438.7,
      "n": 149,
      "sum_bps": 2361.9,
      "t3r_bps": 1151.0,
      "tail_lt_-100_n": 23,
      "win_rate": 0.577
    }
  },
  "event_end_plus_60s": {
    "all": {
      "max_bps": 557.6,
      "mean_bps": -1.8,
      "median_bps": 9.6,
      "min_bps": -522.0,
      "n": 270,
      "sum_bps": -474.1,
      "t3r_bps": -1738.7,
      "tail_lt_-100_n": 45,
      "win_rate": 0.541
    },
    "fill_rate": 0.461,
    "state1800_silence": {
      "max_bps": 557.6,
      "mean_bps": 33.4,
      "median_bps": 24.7,
      "min_bps": -436.6,
      "n": 129,
      "sum_bps": 4306.7,
      "t3r_bps": 3062.1,
      "tail_lt_-100_n": 11,
      "win_rate": 0.636
    },
    "state900_silence": {
      "max_bps": 557.6,
      "mean_bps": 23.6,
      "median_bps": 21.0,
      "min_bps": -436.6,
      "n": 154,
      "sum_bps": 3641.5,
      "t3r_bps": 2396.9,
      "tail_lt_-100_n": 16,
      "win_rate": 0.597
    }
  },
  "event_end_plus_900s": {
    "all": {
      "max_bps": 547.1,
      "mean_bps": -3.8,
      "median_bps": 3.4,
      "min_bps": -523.4,
      "n": 269,
      "sum_bps": -1016.2,
      "t3r_bps": -2422.6,
      "tail_lt_-100_n": 48,
      "win_rate": 0.524
    },
    "fill_rate": 0.459,
    "state1800_silence": {
      "max_bps": 547.1,
      "mean_bps": 19.8,
      "median_bps": 12.0,
      "min_bps": -475.9,
      "n": 129,
      "sum_bps": 2549.8,
      "t3r_bps": 1249.7,
      "tail_lt_-100_n": 13,
      "win_rate": 0.558
    },
    "state900_silence": {
      "max_bps": 547.1,
      "mean_bps": 13.4,
      "median_bps": 10.1,
      "min_bps": -475.9,
      "n": 153,
      "sum_bps": 2045.8,
      "t3r_bps": 745.7,
      "tail_lt_-100_n": 18,
      "win_rate": 0.549
    }
  },
  "reclaim_entry": {
    "all": {
      "max_bps": 604.9,
      "mean_bps": 1.5,
      "median_bps": 14.4,
      "min_bps": -552.7,
      "n": 246,
      "sum_bps": 371.5,
      "t3r_bps": -1000.8,
      "tail_lt_-100_n": 43,
      "win_rate": 0.541
    },
    "fill_rate": 0.452,
    "state1800_silence": {
      "max_bps": 604.9,
      "mean_bps": 33.6,
      "median_bps": 26.7,
      "min_bps": -458.0,
      "n": 129,
      "sum_bps": 4339.6,
      "t3r_bps": 3060.5,
      "tail_lt_-100_n": 13,
      "win_rate": 0.628
    },
    "state900_silence": {
      "max_bps": 604.9,
      "mean_bps": 26.3,
      "median_bps": 23.1,
      "min_bps": -458.0,
      "n": 153,
      "sum_bps": 4028.5,
      "t3r_bps": 2749.4,
      "tail_lt_-100_n": 17,
      "win_rate": 0.595
    }
  }
}
```

## 3. BUY Propagation Scalp Horizons

```json
{
  "120": {
    "horizons": {
      "180": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "1800": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "300": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "3600": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "60": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "900": {
        "attempt_n": 0,
        "fill_rate": null,
        "max_bps": null,
        "mean_bps": null,
        "median_bps": null,
        "min_bps": null,
        "n": 0,
        "sum_bps": 0.0,
        "t3r_bps": 0.0,
        "tail_lt_-100_n": 0,
        "win_rate": null
      }
    },
    "selected_n": 0
  },
  "30": {
    "horizons": {
      "180": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "1800": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "300": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "3600": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "60": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "900": {
        "attempt_n": 0,
        "fill_rate": null,
        "max_bps": null,
        "mean_bps": null,
        "median_bps": null,
        "min_bps": null,
        "n": 0,
        "sum_bps": 0.0,
        "t3r_bps": 0.0,
        "tail_lt_-100_n": 0,
        "win_rate": null
      }
    },
    "selected_n": 0
  },
  "300": {
    "horizons": {
      "180": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "1800": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "300": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "3600": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "60": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "900": {
        "attempt_n": 0,
        "fill_rate": null,
        "max_bps": null,
        "mean_bps": null,
        "median_bps": null,
        "min_bps": null,
        "n": 0,
        "sum_bps": 0.0,
        "t3r_bps": 0.0,
        "tail_lt_-100_n": 0,
        "win_rate": null
      }
    },
    "selected_n": 0
  },
  "60": {
    "horizons": {
      "180": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "1800": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "300": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "3600": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "60": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "900": {
        "attempt_n": 0,
        "fill_rate": null,
        "max_bps": null,
        "mean_bps": null,
        "median_bps": null,
        "min_bps": null,
        "n": 0,
        "sum_bps": 0.0,
        "t3r_bps": 0.0,
        "tail_lt_-100_n": 0,
        "win_rate": null
      }
    },
    "selected_n": 0
  },
  "600": {
    "horizons": {
      "180": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "1800": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "300": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "3600": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "60": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "900": {
        "attempt_n": 0,
        "fill_rate": null,
        "max_bps": null,
        "mean_bps": null,
        "median_bps": null,
        "min_bps": null,
        "n": 0,
        "sum_bps": 0.0,
        "t3r_bps": 0.0,
        "tail_lt_-100_n": 0,
        "win_rate": null
      }
    },
    "selected_n": 0
  },
  "900": {
    "horizons": {
      "180": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "1800": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "300": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "3600": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "60": {
        "attempt_n": 0,
        "fill_rate": null,
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
      "900": {
        "attempt_n": 0,
        "fill_rate": null,
        "max_bps": null,
        "mean_bps": null,
        "median_bps": null,
        "min_bps": null,
        "n": 0,
        "sum_bps": 0.0,
        "t3r_bps": 0.0,
        "tail_lt_-100_n": 0,
        "win_rate": null
      }
    },
    "selected_n": 0
  }
}
```

## 4. Tag Sequence Model

```json
{
  "BUY": {
    "L>L>L>L>L>L>H": {
      "fade_h4": {
        "max_bps": 193.6,
        "mean_bps": 74.5,
        "median_bps": 38.8,
        "min_bps": -6.2,
        "n": 7,
        "sum_bps": 521.2,
        "t3r_bps": 70.8,
        "tail_lt_-100_n": 0,
        "win_rate": 0.857
      },
      "momentum_h1": {
        "max_bps": 63.3,
        "mean_bps": -41.0,
        "median_bps": -30.1,
        "min_bps": -165.3,
        "n": 7,
        "sum_bps": -287.2,
        "t3r_bps": -324.9,
        "tail_lt_-100_n": 1,
        "win_rate": 0.286
      },
      "n": 11
    },
    "L>L>L>L>L>L>L": {
      "fade_h4": {
        "max_bps": 125.0,
        "mean_bps": -12.8,
        "median_bps": 10.6,
        "min_bps": -447.1,
        "n": 8,
        "sum_bps": -102.6,
        "t3r_bps": -472.4,
        "tail_lt_-100_n": 1,
        "win_rate": 0.5
      },
      "momentum_h1": {
        "max_bps": 50.3,
        "mean_bps": -19.4,
        "median_bps": -26.4,
        "min_bps": -74.1,
        "n": 8,
        "sum_bps": -154.9,
        "t3r_bps": -204.0,
        "tail_lt_-100_n": 0,
        "win_rate": 0.25
      },
      "n": 24
    },
    "L>L>L>L>S>S>S": {
      "fade_h4": {
        "max_bps": 192.8,
        "mean_bps": 6.3,
        "median_bps": 0.9,
        "min_bps": -135.0,
        "n": 5,
        "sum_bps": 31.7,
        "t3r_bps": -213.7,
        "tail_lt_-100_n": 1,
        "win_rate": 0.6
      },
      "momentum_h1": {
        "max_bps": 172.3,
        "mean_bps": 31.8,
        "median_bps": 50.5,
        "min_bps": -90.1,
        "n": 6,
        "sum_bps": 190.9,
        "t3r_bps": -101.1,
        "tail_lt_-100_n": 0,
        "win_rate": 0.667
      },
      "n": 21
    },
    "L>L>L>S>S>S>H": {
      "fade_h4": {
        "max_bps": 279.0,
        "mean_bps": 76.2,
        "median_bps": 77.4,
        "min_bps": -127.8,
        "n": 3,
        "sum_bps": 228.6,
        "t3r_bps": 228.6,
        "tail_lt_-100_n": 1,
        "win_rate": 0.667
      },
      "momentum_h1": {
        "max_bps": -42.1,
        "mean_bps": -84.7,
        "median_bps": -71.4,
        "min_bps": -140.7,
        "n": 3,
        "sum_bps": -254.2,
        "t3r_bps": -254.2,
        "tail_lt_-100_n": 1,
        "win_rate": 0.0
      },
      "n": 8
    },
    "L>L>L>S>S>S>S": {
      "fade_h4": {
        "max_bps": 277.9,
        "mean_bps": -6.1,
        "median_bps": -9.4,
        "min_bps": -290.5,
        "n": 24,
        "sum_bps": -147.0,
        "t3r_bps": -775.0,
        "tail_lt_-100_n": 5,
        "win_rate": 0.458
      },
      "momentum_h1": {
        "max_bps": 282.5,
        "mean_bps": 19.1,
        "median_bps": 4.2,
        "min_bps": -244.8,
        "n": 23,
        "sum_bps": 440.4,
        "t3r_bps": -151.5,
        "tail_lt_-100_n": 1,
        "win_rate": 0.565
      },
      "n": 72
    },
    "L>L>S>S>S>S>S": {
      "fade_h4": {
        "max_bps": 120.9,
        "mean_bps": -50.0,
        "median_bps": -11.5,
        "min_bps": -523.1,
        "n": 17,
        "sum_bps": -849.2,
        "t3r_bps": -1123.2,
        "tail_lt_-100_n": 2,
        "win_rate": 0.471
      },
      "momentum_h1": {
        "max_bps": 121.8,
        "mean_bps": 6.9,
        "median_bps": 3.8,
        "min_bps": -71.2,
        "n": 17,
        "sum_bps": 116.7,
        "t3r_bps": -133.2,
        "tail_lt_-100_n": 0,
        "win_rate": 0.529
      },
      "n": 49
    },
    "L>S>S>S>S>S>S": {
      "fade_h4": {
        "max_bps": 198.8,
        "mean_bps": -26.7,
        "median_bps": -30.4,
        "min_bps": -288.1,
        "n": 10,
        "sum_bps": -267.3,
        "t3r_bps": -578.1,
        "tail_lt_-100_n": 1,
        "win_rate": 0.5
      },
      "momentum_h1": {
        "max_bps": 123.4,
        "mean_bps": 3.5,
        "median_bps": 9.7,
        "min_bps": -105.5,
        "n": 11,
        "sum_bps": 38.9,
        "t3r_bps": -207.5,
        "tail_lt_-100_n": 1,
        "win_rate": 0.545
      },
      "n": 26
    },
    "M>M>M>M>M>M>H": {
      "fade_h4": {
        "max_bps": 49.5,
        "mean_bps": -69.3,
        "median_bps": -79.7,
        "min_bps": -354.7,
        "n": 11,
        "sum_bps": -761.9,
        "t3r_bps": -883.9,
        "tail_lt_-100_n": 2,
        "win_rate": 0.273
      },
      "momentum_h1": {
        "max_bps": 159.2,
        "mean_bps": 4.4,
        "median_bps": -21.7,
        "min_bps": -112.0,
        "n": 11,
        "sum_bps": 48.9,
        "t3r_bps": -208.7,
        "tail_lt_-100_n": 1,
        "win_rate": 0.455
      },
      "n": 15
    },
    "M>M>M>M>M>M>M": {
      "fade_h4": {
        "max_bps": 453.9,
        "mean_bps": 18.8,
        "median_bps": 27.4,
        "min_bps": -329.3,
        "n": 35,
        "sum_bps": 658.8,
        "t3r_bps": -233.8,
        "tail_lt_-100_n": 4,
        "win_rate": 0.6
      },
      "momentum_h1": {
        "max_bps": 40.0,
        "mean_bps": -43.3,
        "median_bps": -24.4,
        "min_bps": -539.3,
        "n": 35,
        "sum_bps": -1516.0,
        "t3r_bps": -1611.2,
        "tail_lt_-100_n": 3,
        "win_rate": 0.229
      },
      "n": 46
    },
    "S>S>S>S>S>S>H": {
      "fade_h4": {
        "max_bps": 191.8,
        "mean_bps": -111.9,
        "median_bps": -14.2,
        "min_bps": -516.7,
        "n": 8,
        "sum_bps": -894.9,
        "t3r_bps": -1325.8,
        "tail_lt_-100_n": 3,
        "win_rate": 0.5
      },
      "momentum_h1": {
        "max_bps": 44.8,
        "mean_bps": -18.3,
        "median_bps": -17.1,
        "min_bps": -97.6,
        "n": 8,
        "sum_bps": -146.7,
        "t3r_bps": -188.2,
        "tail_lt_-100_n": 0,
        "win_rate": 0.25
      },
      "n": 22
    },
    "S>S>S>S>S>S>M": {
      "fade_h4": {
        "max_bps": 53.9,
        "mean_bps": 27.3,
        "median_bps": 36.0,
        "min_bps": -16.9,
        "n": 4,
        "sum_bps": 109.1,
        "t3r_bps": -16.9,
        "tail_lt_-100_n": 0,
        "win_rate": 0.75
      },
      "momentum_h1": {
        "max_bps": 29.5,
        "mean_bps": -25.2,
        "median_bps": -18.1,
        "min_bps": -94.1,
        "n": 4,
        "sum_bps": -100.9,
        "t3r_bps": -94.1,
        "tail_lt_-100_n": 0,
        "win_rate": 0.25
      },
      "n": 9
    },
    "S>S>S>S>S>S>S": {
      "fade_h4": {
        "max_bps": 318.2,
        "mean_bps": -29.4,
        "median_bps": -18.4,
        "min_bps": -593.1,
        "n": 52,
        "sum_bps": -1529.6,
        "t3r_bps": -2257.9,
        "tail_lt_-100_n": 9,
        "win_rate": 0.404
      },
      "momentum_h1": {
        "max_bps": 67.0,
        "mean_bps": -12.8,
        "median_bps": -4.8,
        "min_bps": -274.7,
        "n": 53,
        "sum_bps": -676.6,
        "t3r_bps": -871.4,
        "tail_lt_-100_n": 3,
        "win_rate": 0.472
      },
      "n": 135
    }
  },
  "SELL": {
    "L>L>L>L>L>L>L": {
      "fade_h4": {
        "max_bps": 118.6,
        "mean_bps": -59.7,
        "median_bps": -110.5,
        "min_bps": -186.5,
        "n": 7,
        "sum_bps": -417.9,
        "t3r_bps": -615.9,
        "tail_lt_-100_n": 4,
        "win_rate": 0.286
      },
      "momentum_h1": {
        "max_bps": 113.3,
        "mean_bps": 11.9,
        "median_bps": -15.1,
        "min_bps": -26.8,
        "n": 7,
        "sum_bps": 83.4,
        "t3r_bps": -76.8,
        "tail_lt_-100_n": 0,
        "win_rate": 0.429
      },
      "n": 18
    },
    "L>L>L>L>S>S>S": {
      "fade_h4": {
        "max_bps": 18.5,
        "mean_bps": -22.6,
        "median_bps": -19.1,
        "min_bps": -71.0,
        "n": 4,
        "sum_bps": -90.6,
        "t3r_bps": -71.0,
        "tail_lt_-100_n": 0,
        "win_rate": 0.25
      },
      "momentum_h1": {
        "max_bps": -9.4,
        "mean_bps": -28.3,
        "median_bps": -30.7,
        "min_bps": -42.3,
        "n": 4,
        "sum_bps": -113.1,
        "t3r_bps": -42.3,
        "tail_lt_-100_n": 0,
        "win_rate": 0.0
      },
      "n": 14
    },
    "L>L>L>M>M>M>M": {
      "fade_h4": {
        "max_bps": 77.0,
        "mean_bps": 18.8,
        "median_bps": 39.1,
        "min_bps": -49.2,
        "n": 9,
        "sum_bps": 169.0,
        "t3r_bps": -17.4,
        "tail_lt_-100_n": 0,
        "win_rate": 0.556
      },
      "momentum_h1": {
        "max_bps": 80.7,
        "mean_bps": -9.3,
        "median_bps": -17.3,
        "min_bps": -106.0,
        "n": 9,
        "sum_bps": -83.3,
        "t3r_bps": -215.8,
        "tail_lt_-100_n": 1,
        "win_rate": 0.333
      },
      "n": 14
    },
    "L>L>L>S>S>S>H": {
      "fade_h4": {
        "max_bps": 130.5,
        "mean_bps": -7.1,
        "median_bps": 23.9,
        "min_bps": -187.4,
        "n": 9,
        "sum_bps": -63.5,
        "t3r_bps": -264.5,
        "tail_lt_-100_n": 1,
        "win_rate": 0.556
      },
      "momentum_h1": {
        "max_bps": 195.4,
        "mean_bps": 33.1,
        "median_bps": 32.8,
        "min_bps": -47.5,
        "n": 9,
        "sum_bps": 297.8,
        "t3r_bps": -50.4,
        "tail_lt_-100_n": 0,
        "win_rate": 0.667
      },
      "n": 15
    },
    "L>L>L>S>S>S>S": {
      "fade_h4": {
        "max_bps": 304.0,
        "mean_bps": 2.4,
        "median_bps": -0.7,
        "min_bps": -226.4,
        "n": 30,
        "sum_bps": 73.1,
        "t3r_bps": -647.1,
        "tail_lt_-100_n": 3,
        "win_rate": 0.5
      },
      "momentum_h1": {
        "max_bps": 91.1,
        "mean_bps": -10.3,
        "median_bps": -6.8,
        "min_bps": -164.6,
        "n": 31,
        "sum_bps": -318.8,
        "t3r_bps": -547.4,
        "tail_lt_-100_n": 1,
        "win_rate": 0.419
      },
      "n": 69
    },
    "L>L>S>S>S>S>S": {
      "fade_h4": {
        "max_bps": 305.5,
        "mean_bps": -16.4,
        "median_bps": 10.4,
        "min_bps": -471.0,
        "n": 18,
        "sum_bps": -295.2,
        "t3r_bps": -804.0,
        "tail_lt_-100_n": 3,
        "win_rate": 0.556
      },
      "momentum_h1": {
        "max_bps": 118.0,
        "mean_bps": 0.3,
        "median_bps": 6.8,
        "min_bps": -187.7,
        "n": 18,
        "sum_bps": 5.5,
        "t3r_bps": -234.8,
        "tail_lt_-100_n": 1,
        "win_rate": 0.611
      },
      "n": 55
    },
    "L>M>M>M>M>M>M": {
      "fade_h4": {
        "max_bps": 248.8,
        "mean_bps": -30.2,
        "median_bps": -38.7,
        "min_bps": -229.3,
        "n": 7,
        "sum_bps": -211.1,
        "t3r_bps": -488.2,
        "tail_lt_-100_n": 2,
        "win_rate": 0.286
      },
      "momentum_h1": {
        "max_bps": 43.7,
        "mean_bps": -27.7,
        "median_bps": -21.5,
        "min_bps": -146.8,
        "n": 9,
        "sum_bps": -249.3,
        "t3r_bps": -307.0,
        "tail_lt_-100_n": 1,
        "win_rate": 0.333
      },
      "n": 16
    },
    "L>S>S>S>S>S>S": {
      "fade_h4": {
        "max_bps": 199.3,
        "mean_bps": 26.5,
        "median_bps": 19.5,
        "min_bps": -65.5,
        "n": 13,
        "sum_bps": 343.9,
        "t3r_bps": -5.0,
        "tail_lt_-100_n": 0,
        "win_rate": 0.692
      },
      "momentum_h1": {
        "max_bps": 106.6,
        "mean_bps": -25.2,
        "median_bps": -18.4,
        "min_bps": -212.9,
        "n": 14,
        "sum_bps": -352.3,
        "t3r_bps": -491.6,
        "tail_lt_-100_n": 1,
        "win_rate": 0.214
      },
      "n": 32
    },
    "M>M>M>M>M>M>H": {
      "fade_h4": {
        "max_bps": 152.8,
        "mean_bps": -174.3,
        "median_bps": -86.0,
        "min_bps": -553.4,
        "n": 12,
        "sum_bps": -2091.3,
        "t3r_bps": -2433.8,
        "tail_lt_-100_n": 6,
        "win_rate": 0.417
      },
      "momentum_h1": {
        "max_bps": 212.8,
        "mean_bps": 28.4,
        "median_bps": 19.2,
        "min_bps": -151.7,
        "n": 12,
        "sum_bps": 341.4,
        "t3r_bps": -96.3,
        "tail_lt_-100_n": 1,
        "win_rate": 0.667
      },
      "n": 15
    },
    "M>M>M>M>M>M>M": {
      "fade_h4": {
        "max_bps": 438.1,
        "mean_bps": -1.0,
        "median_bps": -11.8,
        "min_bps": -397.2,
        "n": 33,
        "sum_bps": -33.0,
        "t3r_bps": -1103.2,
        "tail_lt_-100_n": 8,
        "win_rate": 0.455
      },
      "momentum_h1": {
        "max_bps": 439.2,
        "mean_bps": 15.4,
        "median_bps": 1.4,
        "min_bps": -222.9,
        "n": 32,
        "sum_bps": 494.3,
        "t3r_bps": -307.5,
        "tail_lt_-100_n": 2,
        "win_rate": 0.5
      },
      "n": 47
    },
    "S>S>S>S>S>S>H": {
      "fade_h4": {
        "max_bps": 186.2,
        "mean_bps": 61.4,
        "median_bps": 47.7,
        "min_bps": -39.0,
        "n": 5,
        "sum_bps": 306.8,
        "t3r_bps": -29.4,
        "tail_lt_-100_n": 0,
        "win_rate": 0.8
      },
      "momentum_h1": {
        "max_bps": 76.5,
        "mean_bps": -21.1,
        "median_bps": -18.6,
        "min_bps": -111.5,
        "n": 5,
        "sum_bps": -105.3,
        "t3r_bps": -201.5,
        "tail_lt_-100_n": 1,
        "win_rate": 0.4
      },
      "n": 16
    },
    "S>S>S>S>S>S>S": {
      "fade_h4": {
        "max_bps": 492.8,
        "mean_bps": 20.5,
        "median_bps": 18.3,
        "min_bps": -494.4,
        "n": 61,
        "sum_bps": 1248.6,
        "t3r_bps": 165.8,
        "tail_lt_-100_n": 7,
        "win_rate": 0.607
      },
      "momentum_h1": {
        "max_bps": 334.8,
        "mean_bps": -14.3,
        "median_bps": -18.3,
        "min_bps": -217.5,
        "n": 61,
        "sum_bps": -871.0,
        "t3r_bps": -1482.6,
        "tail_lt_-100_n": 5,
        "win_rate": 0.377
      },
      "n": 153
    }
  }
}
```

## 5. V02 Management Compatibility

```json
{
  "all": {
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
  "state_1800": {
    "PRESSURE_HIGH": {
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
    "PRESSURE_MID": {
      "max_bps": 95.4,
      "mean_bps": 95.4,
      "median_bps": 95.4,
      "min_bps": 95.4,
      "n": 1,
      "sum_bps": 95.4,
      "t3r_bps": 95.4,
      "tail_lt_-100_n": 0,
      "win_rate": 1.0
    },
    "SILENCE_RECLAIM": {
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
  },
  "state_300": {
    "PRESSURE_LOW": {
      "max_bps": -4.5,
      "mean_bps": -4.5,
      "median_bps": -4.5,
      "min_bps": -4.5,
      "n": 1,
      "sum_bps": -4.5,
      "t3r_bps": -4.5,
      "tail_lt_-100_n": 0,
      "win_rate": 0.0
    },
    "PRESSURE_MID": {
      "max_bps": 95.4,
      "mean_bps": 21.7,
      "median_bps": 21.6,
      "min_bps": -52.1,
      "n": 2,
      "sum_bps": 43.3,
      "t3r_bps": 43.3,
      "tail_lt_-100_n": 0,
      "win_rate": 0.5
    },
    "SILENCE_RECLAIM": {
      "max_bps": 223.2,
      "mean_bps": 77.3,
      "median_bps": 108.9,
      "min_bps": -59.2,
      "n": 8,
      "sum_bps": 618.2,
      "t3r_bps": 91.2,
      "tail_lt_-100_n": 0,
      "win_rate": 0.625
    }
  },
  "state_900": {
    "PRESSURE_MID": {
      "max_bps": 95.4,
      "mean_bps": 21.7,
      "median_bps": 21.6,
      "min_bps": -52.1,
      "n": 2,
      "sum_bps": 43.3,
      "t3r_bps": 43.3,
      "tail_lt_-100_n": 0,
      "win_rate": 0.5
    },
    "SILENCE_RECLAIM": {
      "max_bps": 223.2,
      "mean_bps": 68.2,
      "median_bps": 83.8,
      "min_bps": -59.2,
      "n": 9,
      "sum_bps": 613.7,
      "t3r_bps": 86.7,
      "tail_lt_-100_n": 0,
      "win_rate": 0.556
    }
  }
}
```

## Read

- Early tau labels are now strictly causal: only states known by anchor+tau are used.
- The tests evaluate navigation/management value; they do not change live execution.
- A candidate needs positive causal execution, holdout/T3R robustness, and V02 compatibility before paper/live.
