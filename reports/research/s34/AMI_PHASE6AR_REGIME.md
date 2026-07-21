# AMI Faz 6A-R — Regime-Conditioned Latent + Drift

> 2026-07-03 07:31 UTC — prereg `E-LATENT6AR-001` hash `1b6d0b2b6ef42581`. VERDICT: **PASS**

## regime_thresholds
```json
{
 "trend_up": 100.0,
 "trend_dn": -100.0,
 "vol_lo": 0.002406306595262669,
 "vol_hi": 0.0029201003325009583,
 "stress_hi": 21875210.589069977,
 "lev_pos": 4.540000000000001e-06,
 "lev_neg": -4.1e-06
}
```

## drift_attribution
```json
{
 "ret5m": {
  "psi": 0.043,
  "js": 0.006,
  "std_mean_shift": 0.0,
  "miss_a": 0.0,
  "miss_b": 0.0,
  "source": "NO_DRIFT",
  "confidence": "HIGH"
 },
 "rv30m": {
  "psi": 0.155,
  "js": 0.03,
  "std_mean_shift": 0.24,
  "miss_a": 0.0,
  "miss_b": 0.0,
  "source": "MILD_MARKET_SHIFT",
  "confidence": "MEDIUM"
 },
 "ofi10m": {
  "psi": 0.004,
  "js": 0.001,
  "std_mean_shift": 0.01,
  "miss_a": 0.257,
  "miss_b": 0.0,
  "source": "NO_DRIFT",
  "confidence": "HIGH"
 },
 "stress10m": {
  "psi": 0.547,
  "js": 0.027,
  "std_mean_shift": 0.56,
  "miss_a": 0.0,
  "miss_b": 0.0,
  "source": "MARKET_SHIFT",
  "confidence": "MEDIUM"
 },
 "buyliq10m": {
  "psi": 0.362,
  "js": 0.017,
  "std_mean_shift": 0.1,
  "miss_a": 0.0,
  "miss_b": 0.0,
  "source": "MARKET_SHIFT",
  "confidence": "MEDIUM"
 },
 "fund_vel_1h": {
  "psi": 0.06,
  "js": 0.01,
  "std_mean_shift": 0.02,
  "miss_a": 0.0,
  "miss_b": 0.0,
  "source": "NO_DRIFT",
  "confidence": "HIGH"
 },
 "spread5m": {
  "psi": 7.381,
  "js": 0.778,
  "std_mean_shift": 0.79,
  "miss_a": 0.19,
  "miss_b": 0.001,
  "source": "MARKET_SHIFT",
  "confidence": "LOW"
 },
 "trades10m": {
  "psi": 0.89,
  "js": 0.109,
  "std_mean_shift": 0.4,
  "miss_a": 0.0,
  "miss_b": 0.0,
  "source": "MARKET_SHIFT",
  "confidence": "MEDIUM"
 },
 "ret1h": {
  "psi": 0.043,
  "js": 0.006,
  "std_mean_shift": 0.0,
  "miss_a": 0.0,
  "miss_b": 0.0,
  "source": "NO_DRIFT",
  "confidence": "HIGH"
 },
 "regime:trend": {
  "expl": {
   "DOWN": 0.353,
   "RANGE": 0.385,
   "UP": 0.262
  },
  "val": {
   "DOWN": 0.367,
   "RANGE": 0.37,
   "UP": 0.263
  }
 },
 "regime:vol": {
  "expl": {
   "HIGH": 0.436,
   "LOW": 0.289,
   "NORMAL": 0.275
  },
  "val": {
   "HIGH": 0.773,
   "LOW": 0.055,
   "NORMAL": 0.172
  }
 },
 "regime:stress": {
  "expl": {
   "NORMAL": 0.735,
   "STRESSED": 0.265
  },
  "val": {
   "NORMAL": 0.128,
   "STRESSED": 0.872
  }
 },
 "regime:leverage": {
  "expl": {
   "CONTRACTING": 0.335,
   "EXPANDING": 0.325,
   "NEUTRAL": 0.34
  },
  "val": {
   "CONTRACTING": 0.336,
   "EXPANDING": 0.373,
   "NEUTRAL": 0.291
  }
 },
 "regime:session": {
  "expl": {
   "EUROPE": 0.249,
   "OFF": 0.419,
   "US": 0.332
  },
  "val": {
   "EUROPE": 0.252,
   "OFF": 0.406,
   "US": 0.342
  }
 }
}
```

## per_regime
```json
{
 "trend=DOWN": {
  "n": 8400,
  "k": 2,
  "seed_ari": 1.0,
  "perturb_ari": 1.0,
  "occ_train": [
   0.767,
   0.233
  ],
  "occ_val": [
   0.292,
   0.708
  ],
  "occ_ratio": [
   0.38,
   3.04
  ],
  "trans_corr": 0.975,
  "trans_entropy": 0.045,
  "avg_dur_min": {
   "0": 2050.0,
   "1": 622.7
  },
  "chrono_stable": false,
  "profiles": {
   "S0": {
    "rv30m": 0.68,
    "spread5m": 1.34
   },
   "S1": {
    "stress10m": 5.2,
    "buyliq10m": 4.16,
    "trades10m": 0.7
   }
  }
 },
 "trend=RANGE": {
  "n": 9036,
  "k": 2,
  "seed_ari": 0.813,
  "perturb_ari": 0.813,
  "occ_train": [
   0.859,
   0.141
  ],
  "occ_val": [
   0.306,
   0.694
  ],
  "occ_ratio": [
   0.36,
   4.94
  ],
  "trans_corr": 0.913,
  "trans_entropy": 0.178,
  "avg_dur_min": {
   "0": 604.0,
   "1": 98.8
  },
  "chrono_stable": false,
  "profiles": {
   "S0": {
    "spread5m": 0.57
   },
   "S1": {
    "stress10m": 5.98,
    "buyliq10m": 3.66,
    "trades10m": 0.73
   }
  }
 },
 "trend=UP": {
  "n": 6199,
  "k": 2,
  "seed_ari": 0.646,
  "perturb_ari": 1.0,
  "occ_train": [
   0.572,
   0.428
  ],
  "occ_val": [
   0.387,
   0.613
  ],
  "occ_ratio": [
   0.68,
   1.43
  ],
  "trans_corr": 0.976,
  "trans_entropy": 0.198,
  "avg_dur_min": {
   "0": 188.1,
   "1": 140.6
  },
  "chrono_stable": true,
  "profiles": {
   "S0": {
    "stress10m": 1.05
   },
   "S1": {
    "rv30m": 1.12,
    "stress10m": 4.63,
    "buyliq10m": 5.99,
    "trades10m": 1.34
   }
  }
 },
 "vol=HIGH": {
  "n": 11903,
  "k": 5,
  "seed_ari": 0.742,
  "perturb_ari": 0.996,
  "occ_train": [
   0.135,
   0.06,
   0.128,
   0.356,
   0.322
  ],
  "occ_val": [
   0.311,
   0.158,
   0.094,
   0.435,
   0.001
  ],
  "occ_ratio": [
   2.3,
   2.65,
   0.73,
   1.22,
   0.0
  ],
  "trans_corr": 0.708,
  "trans_entropy": 0.55,
  "avg_dur_min": {
   "0": 30.1,
   "1": 23.7,
   "2": 127.0,
   "3": 68.3,
   "4": 248.1
  },
  "chrono_stable": false,
  "profiles": {
   "S0": {
    "rv30m": 0.65,
    "stress10m": 6.0,
    "spread5m": 1.0,
    "trades10m": 0.61
   },
   "S1": {
    "rv30m": 0.7,
    "buyliq10m": 6.0,
    "spread5m": 0.69,
    "trades10m": 0.75,
    "ret1h": 0.51
   },
   "S2": {
    "rv30m": 2.53,
    "spread5m": 3.57,
    "ret1h": -1.19
   },
   "S3": {
    "rv30m": 1.25,
    "stress10m": 6.0,
    "buyliq10m": 6.0,
    "spread5m": 0.63,
    "trades10m": 1.4
   },
   "S4": {
    "spread5m": 0.72
   }
  }
 },
 "vol=LOW": {
  "n": 5722,
  "k": 2,
  "seed_ari": 0.781,
  "perturb_ari": 0.74,
  "occ_train": [
   0.812,
   0.188
  ],
  "occ_val": [
   0.685,
   0.315
  ],
  "occ_ratio": [
   0.84,
   1.67
  ],
  "trans_corr": 0.998,
  "trans_entropy": 0.136,
  "avg_dur_min": {
   "0": 650.2,
   "1": 150.8
  },
  "chrono_stable": true,
  "profiles": {
   "S0": {},
   "S1": {
    "stress10m": 4.59,
    "buyliq10m": 3.6
   }
  }
 },
 "vol=NORMAL": {
  "n": 6010,
  "k": 2,
  "seed_ari": 0.852,
  "perturb_ari": 0.704,
  "occ_train": [
   0.836,
   0.164
  ],
  "occ_val": [
   0.607,
   0.393
  ],
  "occ_ratio": [
   0.73,
   2.39
  ],
  "trans_corr": 0.998,
  "trans_entropy": 0.124,
  "avg_dur_min": {
   "0": 837.4,
   "1": 164.3
  },
  "chrono_stable": true,
  "profiles": {
   "S0": {},
   "S1": {
    "stress10m": 6.0,
    "buyliq10m": 3.92,
    "trades10m": 0.66
   }
  }
 },
 "stress=NORMAL": {
  "n": 14493,
  "k": 2,
  "seed_ari": 0.7,
  "perturb_ari": 0.663,
  "occ_train": [
   0.912,
   0.088
  ],
  "occ_val": [
   0.863,
   0.137
  ],
  "occ_ratio": [
   0.95,
   1.56
  ],
  "trans_corr": 0.98,
  "trans_entropy": 0.198,
  "avg_dur_min": {
   "0": 811.5,
   "1": 78.4
  },
  "chrono_stable": true,
  "profiles": {
   "S0": {},
   "S1": {
    "rv30m": 0.53,
    "stress10m": 4.24,
    "buyliq10m": 5.92,
    "trades10m": 0.84
   }
  }
 },
 "stress=STRESSED": {
  "n": 9142,
  "k": 5,
  "seed_ari": 0.807,
  "perturb_ari": 1.0,
  "occ_train": [
   0.114,
   0.196,
   0.177,
   0.36,
   0.153
  ],
  "occ_val": [
   0.152,
   0.001,
   0.415,
   0.001,
   0.431
  ],
  "occ_ratio": [
   1.33,
   0.01,
   2.34,
   0.0,
   2.81
  ],
  "trans_corr": 0.504,
  "trans_entropy": 0.699,
  "avg_dur_min": {
   "0": 24.7,
   "1": 40.2,
   "2": 44.3,
   "3": 63.6,
   "4": 37.2
  },
  "chrono_stable": false,
  "profiles": {
   "S0": {
    "rv30m": 0.62,
    "buyliq10m": 5.99,
    "spread5m": 2.02,
    "trades10m": 0.61
   },
   "S1": {
    "stress10m": 4.22
   },
   "S2": {
    "stress10m": 4.5,
    "spread5m": 4.93
   },
   "S3": {
    "rv30m": 1.13,
    "stress10m": 6.0,
    "buyliq10m": 6.0,
    "trades10m": 1.33
   },
   "S4": {
    "rv30m": 1.27,
    "stress10m": 6.0,
    "buyliq10m": 6.0,
    "spread5m": 4.84,
    "trades10m": 1.15
   }
  }
 }
}
```

## walk_forward
```json
{
 "ALL": {
  "folds": [
   {
    "val_win": [
     0.4,
     0.55
    ],
    "occ_ratio": [
     0.0,
     1.7,
     0.0,
     1.73
    ],
    "band_ok": false,
    "center_cos_prev": null
   },
   {
    "val_win": [
     0.55,
     0.7
    ],
    "occ_ratio": [
     1.24,
     0.4,
     0.39,
     0.49
    ],
    "band_ok": true,
    "center_cos_prev": 0.785
   },
   {
    "val_win": [
     0.7,
     0.85
    ],
    "occ_ratio": [
     0.14,
     4.56,
     5.07,
     2.88
    ],
    "band_ok": false,
    "center_cos_prev": 0.998
   },
   {
    "val_win": [
     0.85,
     1.0
    ],
    "occ_ratio": [
     0.18,
     2.98,
     1.81,
     3.57
    ],
    "band_ok": false,
    "center_cos_prev": 0.975
   }
  ],
  "band_ok_folds": 1,
  "cos_ok_folds": 2,
  "persistent": false
 },
 "trend=UP": {
  "folds": [
   {
    "val_win": [
     0.4,
     0.55
    ],
    "occ_ratio": [
     2.46,
     0.0
    ],
    "band_ok": false,
    "center_cos_prev": null
   },
   {
    "val_win": [
     0.55,
     0.7
    ],
    "occ_ratio": [
     0.34,
     1.86
    ],
    "band_ok": true,
    "center_cos_prev": 0.809
   },
   {
    "val_win": [
     0.7,
     0.85
    ],
    "occ_ratio": [
     0.7,
     1.41
    ],
    "band_ok": true,
    "center_cos_prev": 0.834
   },
   {
    "val_win": [
     0.85,
     1.0
    ],
    "occ_ratio": [
     0.3,
     1.48
    ],
    "band_ok": true,
    "center_cos_prev": 0.669
   }
  ],
  "band_ok_folds": 3,
  "cos_ok_folds": 2,
  "persistent": true
 },
 "vol=LOW": {
  "folds": [
   {
    "val_win": [
     0.4,
     0.55
    ],
    "occ_ratio": [
     1.09,
     0.0
    ],
    "band_ok": false,
    "center_cos_prev": null
   },
   {
    "val_win": [
     0.55,
     0.7
    ],
    "occ_ratio": [
     1.14,
     0.0
    ],
    "band_ok": false,
    "center_cos_prev": 0.919
   },
   {
    "val_win": [
     0.7,
     0.85
    ],
    "occ_ratio": [
     1.23,
     0.0
    ],
    "band_ok": false,
    "center_cos_prev": 0.527
   },
   {
    "val_win": [
     0.85,
     1.0
    ],
    "occ_ratio": [
     0.54,
     4.69
    ],
    "band_ok": false,
    "center_cos_prev": 0.819
   }
  ],
  "band_ok_folds": 0,
  "cos_ok_folds": 2,
  "persistent": false
 },
 "vol=NORMAL": {
  "folds": [
   {
    "val_win": [
     0.4,
     0.55
    ],
    "occ_ratio": [
     1.29,
     0.0
    ],
    "band_ok": false,
    "center_cos_prev": null
   },
   {
    "val_win": [
     0.55,
     0.7
    ],
    "occ_ratio": [
     1.26,
     0.0
    ],
    "band_ok": false,
    "center_cos_prev": 0.72
   },
   {
    "val_win": [
     0.7,
     0.85
    ],
    "occ_ratio": [
     1.04,
     0.78
    ],
    "band_ok": true,
    "center_cos_prev": 0.871
   },
   {
    "val_win": [
     0.85,
     1.0
    ],
    "occ_ratio": [
     0.58,
     3.98
    ],
    "band_ok": false,
    "center_cos_prev": 0.846
   }
  ],
  "band_ok_folds": 1,
  "cos_ok_folds": 2,
  "persistent": false
 },
 "stress=NORMAL": {
  "folds": [
   {
    "val_win": [
     0.4,
     0.55
    ],
    "occ_ratio": [
     1.3,
     0.0
    ],
    "band_ok": false,
    "center_cos_prev": null
   },
   {
    "val_win": [
     0.55,
     0.7
    ],
    "occ_ratio": [
     1.15,
     0.0
    ],
    "band_ok": false,
    "center_cos_prev": 0.771
   },
   {
    "val_win": [
     0.7,
     0.85
    ],
    "occ_ratio": [
     1.1,
     0.0
    ],
    "band_ok": false,
    "center_cos_prev": 0.73
   },
   {
    "val_win": [
     0.85,
     1.0
    ],
    "occ_ratio": [
     0.67,
     4.5
    ],
    "band_ok": false,
    "center_cos_prev": 0.84
   }
  ],
  "band_ok_folds": 0,
  "cos_ok_folds": 1,
  "persistent": false
 }
}
```

## dq_explanation_ruled_out
```json
true
```

## alpha_eval
```json
{
 "1_baseline_all": {
  "n": 50,
  "trade_per_day": 4.06,
  "wr": 46.0,
  "median": -5.5,
  "mean": -7.0,
  "cum": -349.0,
  "top3_removed": -1335.0,
  "pf": 0.89,
  "mdd": -1363.0
 },
 "2_baseline+regime": {
  "n": 24,
  "trade_per_day": 1.95,
  "wr": 41.7,
  "median": -23.3,
  "mean": 7.1,
  "cum": 171.0,
  "top3_removed": -620.0,
  "pf": 1.14,
  "mdd": -686.1
 },
 "3_baseline+latent(calm)": {
  "n": 41,
  "trade_per_day": 3.33,
  "wr": 51.2,
  "median": 4.0,
  "mean": -23.4,
  "cum": -959.0,
  "top3_removed": -1980.0,
  "pf": 0.65,
  "mdd": -1806.5
 },
 "4_regime+latent": {
  "n": 14,
  "trade_per_day": 1.14,
  "wr": 50.0,
  "median": 0.5,
  "mean": 16.2,
  "cum": 227.0,
  "top3_removed": -458.0,
  "pf": 1.41,
  "mdd": -416.1
 },
 "5_latent_only_states": {
  "LS-001": {
   "n": 41,
   "trade_per_day": 3.33,
   "wr": 51.2,
   "median": 4.0,
   "mean": -23.4,
   "cum": -959.0,
   "top3_removed": -1980.0,
   "pf": 0.65,
   "mdd": -1806.5
  },
  "LS-002": {
   "n": 44,
   "trade_per_day": 3.57,
   "wr": 45.5,
   "median": -9.0,
   "mean": -16.5,
   "cum": -728.0,
   "top3_removed": -1862.0,
   "pf": 0.76,
   "mdd": -1733.3
  },
  "LS-003": {
   "n": 48,
   "trade_per_day": 3.9,
   "wr": 52.1,
   "median": 2.0,
   "mean": -1.9,
   "cum": -89.0,
   "top3_removed": -1109.0,
   "pf": 0.97,
   "mdd": -1145.6
  },
  "LS-004": {
   "n": 47,
   "trade_per_day": 3.82,
   "wr": 48.9,
   "median": -0.2,
   "mean": -6.3,
   "cum": -296.0,
   "top3_removed": -1075.0,
   "pf": 0.89,
   "mdd": -1064.7
  }
 }
}
```

Durust statuler: software-correct ✓ · drift-attributed (bkz. drift_attribution) · regime-conditioned stable · walk-forward passed · alpha: alpha_eval bolumune bakiniz (ayri untouched katman) · forward-validating ✗ · **operationally FORBIDDEN**

*Runner: `python -m ami.latent.regime`*