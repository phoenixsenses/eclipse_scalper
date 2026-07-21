# AMI Faz 6A-R2 — Risk and Applicability Validation

> 2026-07-03 08:04 UTC — prereg `E-RISKAPP-6AR2-001` hash `db07a7371cc279c8`. OUTCOME: **FALSIFIES / INSUFFICIENT_SAMPLE**

- Trade populasyonu: 328 no-overlap 6h LONG grid trade (veto yorumu)
- Toplam degerlendirilebilir aday N: 9
- Fold verdict: {"evaluable_folds": 1, "passed_folds": 0, "majority_pass": false, "all_folds_reported": 4}
- dq elendi: True · top-winner siralama stabil: True
- Alarm: lead_days=0.0 leading=True fp_suspension=0.69 blocked_good=145 blocked_bad=180

## Fold 0 val=[0.4, 0.55]
```json
{
 "val_win": [
  0.4,
  0.55
 ],
 "contaminated": false,
 "calm_state": 1,
 "n_base": 49,
 "n_cand": 9,
 "baseline": {
  "n": 49,
  "trade_per_day": 3.98,
  "exposure_hours": 294.0,
  "cum": -1265.3,
  "mean": -25.82,
  "median": -5.0,
  "per_active_hour": -4.304,
  "top1_removed": -1501.0,
  "top3_removed": -1818.2,
  "top5_removed": -2079.8,
  "pf": 0.5,
  "mdd": -1337.1,
  "avg_dd": -779.1,
  "dd_duration_max_trades": 48,
  "recovery_trades_mean": null,
  "worst": -376.7,
  "bottom3_cum": -898.4,
  "cvar5": -299.5,
  "mae_mean": -82.3,
  "mae_p10": -213.0,
  "loss_rate": 0.531,
  "max_consec_loss": 7,
  "ret_vol": 112.8,
  "ret_to_dd": -0.95,
  "downside_dev": 100.3,
  "session_conc": {
   "EUROPE": 0.24,
   "OFF": 0.49,
   "US": 0.27
  },
  "regime_conc": {
   "DOWN": 0.37,
   "RANGE": 0.45,
   "UP": 0.18
  }
 },
 "candidate": {
  "n": 9,
  "trade_per_day": 0.73,
  "exposure_hours": 54.0,
  "cum": -515.8,
  "mean": -57.32,
  "median": -51.65,
  "per_active_hour": -9.553,
  "top1_removed": -536.1,
  "top3_removed": -535.7,
  "top5_removed": -479.0,
  "pf": 0.05,
  "mdd": -418.9,
  "avg_dd": -232.5,
  "dd_duration_max_trades": 8,
  "recovery_trades_mean": null,
  "worst": -175.2,
  "bottom3_cum": -382.1,
  "cvar5": -175.2,
  "mae_mean": -99.9,
  "mae_p10": -171.2,
  "loss_rate": 0.778,
  "max_consec_loss": 4,
  "ret_vol": 62.4,
  "ret_to_dd": -1.23,
  "downside_dev": 84.5,
  "session_conc": {
   "EUROPE": 0.22,
   "OFF": 0.33,
   "US": 0.44
  },
  "regime_conc": {
   "UP": 1.0
  }
 },
 "regime_only": {
  "n": 9,
  "trade_per_day": 0.73,
  "exposure_hours": 54.0,
  "cum": -515.8,
  "mean": -57.32,
  "median": -51.65,
  "per_active_hour": -9.553,
  "top1_removed": -536.1,
  "top3_removed": -535.7,
  "top5_removed": -479.0,
  "pf": 0.05,
  "mdd": -418.9,
  "avg_dd": -232.5,
  "dd_duration_max_trades": 8,
  "recovery_trades_mean": null,
  "worst": -175.2,
  "bottom3_cum": -382.1,
  "cvar5": -175.2,
  "mae_mean": -99.9,
  "mae_p10": -171.2,
  "loss_rate": 0.778,
  "max_consec_loss": 4,
  "ret_vol": 62.4,
  "ret_to_dd": -1.23,
  "downside_dev": 84.5,
  "session_conc": {
   "EUROPE": 0.22,
   "OFF": 0.33,
   "US": 0.44
  },
  "regime_conc": {
   "UP": 1.0
  }
 },
 "latent_only": {
  "n": 45,
  "trade_per_day": 3.66,
  "exposure_hours": 270.0,
  "cum": -816.1,
  "mean": -18.13,
  "median": 0.73,
  "per_active_hour": -3.022,
  "top1_removed": -1051.8,
  "top3_removed": -1369.1,
  "top5_removed": -1630.7,
  "pf": 0.61,
  "mdd": -1086.0,
  "avg_dd": -588.2,
  "dd_duration_max_trades": 44,
  "recovery_trades_mean": null,
  "worst": -376.7,
  "bottom3_cum": -898.4,
  "cvar5": -299.5,
  "mae_mean": -76.1,
  "mae_p10": -193.7,
  "loss_rate": 0.489,
  "max_consec_loss": 6,
  "ret_vol": 111.5,
  "ret_to_dd": -0.75,
  "downside_dev": 95.6,
  "session_conc": {
   "EUROPE": 0.22,
   "OFF": 0.49,
   "US": 0.29
  },
  "regime_conc": {
   "DOWN": 0.33,
   "RANGE": 0.47,
   "UP": 0.2
  }
 },
 "evaluable": true,
 "sample_status": "OK",
 "matched_count": {
  "n_draws": 2000,
  "block_len": 5,
  "boot_median": {
   "cvar5": -223.1,
   "mdd": -294.0,
   "downside_dev": 84.4,
   "mean": -21.7,
   "worst": -223.1
  },
  "cand_percentile": {
   "cvar5": 0.644,
   "mdd": 0.287,
   "downside_dev": 0.506,
   "mean": 0.139,
   "worst": 0.644
  }
 },
 "random_veto": {
  "n_draws": 2000,
  "rv_median": {
   "cvar5": -202.0,
   "mdd": -298.7,
   "downside_dev": 89.2,
   "mean": -25.9,
   "worst": -202.0
  },
  "cand_percentile": {
   "cvar5": 0.65,
   "mdd": 0.3,
   "downside_dev": 0.462,
   "mean": 0.184,
   "worst": 0.65
  }
 },
 "session_match_tv": 0.179,
 "loss_concentration": {
  "bad_class_sizes": {
   "worst5": 3,
   "worst10": 5,
   "sl": 7,
   "high_mae": 10,
   "giveback": 1,
   "dd_start": 38
  },
  "veto_recall_by_class": {
   "worst5": 1.0,
   "worst10": 1.0,
   "sl": 0.857,
   "high_mae": 0.7,
   "giveback": 1.0,
   "dd_start": 0.789
  },
  "bad_trade_recall": 0.7,
  "bad_trade_precision": 0.175,
  "good_trade_retention": 0.087,
  "selection_rate": 0.184,
  "winner_sacrifice_bps": 1255.5,
  "loss_avoided_bps": 2004.9,
  "profit_sacrificed_bps": 1255.5,
  "net_economic_value_bps": 749.4,
  "retention_ratio": 0.47
 },
 "checks": {
  "a_tail_vs_matched": false,
  "b_beats_random_veto": false,
  "c_incremental_over_regime": false,
  "d_retention": false,
  "e_min_n": true
 },
 "fold_pass": false
}
```

## Fold 1 val=[0.55, 0.7]
```json
{
 "val_win": [
  0.55,
  0.7
 ],
 "contaminated": false,
 "calm_state": 0,
 "n_base": 49,
 "n_cand": 1,
 "baseline": {
  "n": 49,
  "trade_per_day": 3.98,
  "exposure_hours": 294.0,
  "cum": -2430.8,
  "mean": -49.61,
  "median": -30.11,
  "per_active_hour": -8.268,
  "top1_removed": -2753.2,
  "top3_removed": -3112.9,
  "top5_removed": -3360.4,
  "pf": 0.39,
  "mdd": -3087.8,
  "avg_dd": -1206.2,
  "dd_duration_max_trades": 48,
  "recovery_trades_mean": null,
  "worst": -447.2,
  "bottom3_cum": -1215.2,
  "cvar5": -405.1,
  "mae_mean": -150.6,
  "mae_p10": -395.8,
  "loss_rate": 0.612,
  "max_consec_loss": 5,
  "ret_vol": 146.3,
  "ret_to_dd": -0.79,
  "downside_dev": 138.0,
  "session_conc": {
   "EUROPE": 0.24,
   "OFF": 0.51,
   "US": 0.24
  },
  "regime_conc": {
   "DOWN": 0.59,
   "RANGE": 0.31,
   "UP": 0.1
  }
 },
 "candidate": {
  "n": 1,
  "trade_per_day": 0.08,
  "exposure_hours": 6.0,
  "cum": -51.7,
  "mean": -51.72,
  "median": -51.72,
  "per_active_hour": -8.621,
  "top1_removed": null,
  "top3_removed": null,
  "top5_removed": null,
  "pf": 0.0,
  "mdd": 0.0,
  "avg_dd": 0.0,
  "dd_duration_max_trades": 0,
  "recovery_trades_mean": null,
  "worst": -51.7,
  "bottom3_cum": -51.7,
  "cvar5": -51.7,
  "mae_mean": -48.8,
  "mae_p10": -48.8,
  "loss_rate": 1.0,
  "max_consec_loss": 1,
  "ret_vol": 0.0,
  "ret_to_dd": null,
  "downside_dev": 51.7,
  "session_conc": {
   "OFF": 1.0
  },
  "regime_conc": {
   "UP": 1.0
  }
 },
 "regime_only": {
  "n": 5,
  "trade_per_day": 0.41,
  "exposure_hours": 30.0,
  "cum": 304.6,
  "mean": 60.91,
  "median": 29.55,
  "per_active_hour": 10.152,
  "top1_removed": -17.8,
  "top3_removed": -204.5,
  "top5_removed": null,
  "pf": 2.49,
  "mdd": -152.8,
  "avg_dd": -30.6,
  "dd_duration_max_trades": 1,
  "recovery_trades_mean": null,
  "worst": -152.8,
  "bottom3_cum": -175.0,
  "cvar5": -152.8,
  "mae_mean": -104.4,
  "mae_p10": -195.4,
  "loss_rate": 0.4,
  "max_consec_loss": 1,
  "ret_vol": 165.5,
  "ret_to_dd": 1.99,
  "downside_dev": 72.2,
  "session_conc": {
   "EUROPE": 0.2,
   "OFF": 0.6,
   "US": 0.2
  },
  "regime_conc": {
   "UP": 1.0
  }
 },
 "latent_only": {
  "n": 43,
  "trade_per_day": 3.49,
  "exposure_hours": 258.0,
  "cum": -3088.5,
  "mean": -71.83,
  "median": -43.65,
  "per_active_hour": -11.971,
  "top1_removed": -3212.9,
  "top3_removed": -3427.3,
  "top5_removed": -3563.0,
  "pf": 0.19,
  "mdd": -3087.8,
  "avg_dd": -1009.1,
  "dd_duration_max_trades": 42,
  "recovery_trades_mean": null,
  "worst": -447.2,
  "bottom3_cum": -1215.2,
  "cvar5": -405.1,
  "mae_mean": -160.1,
  "mae_p10": -415.3,
  "loss_rate": 0.674,
  "max_consec_loss": 5,
  "ret_vol": 131.6,
  "ret_to_dd": -1.0,
  "downside_dev": 145.4,
  "session_conc": {
   "EUROPE": 0.26,
   "OFF": 0.51,
   "US": 0.23
  },
  "regime_conc": {
   "DOWN": 0.63,
   "RANGE": 0.35,
   "UP": 0.02
  }
 },
 "evaluable": false,
 "sample_status": "INSUFFICIENT_SAMPLE",
 "matched_count": {
  "skipped": "n_cand<min"
 },
 "random_veto": {
  "skipped": "n_cand<min"
 }
}
```

## Fold 2 val=[0.7, 0.85]
```json
{
 "val_win": [
  0.7,
  0.85
 ],
 "contaminated": false,
 "calm_state": 0,
 "n_base": 50,
 "n_cand": 2,
 "baseline": {
  "n": 50,
  "trade_per_day": 4.06,
  "exposure_hours": 300.0,
  "cum": 188.9,
  "mean": 3.78,
  "median": -11.66,
  "per_active_hour": 0.63,
  "top1_removed": -203.6,
  "top3_removed": -823.3,
  "top5_removed": -1210.8,
  "pf": 1.09,
  "mdd": -893.7,
  "avg_dd": -311.5,
  "dd_duration_max_trades": 22,
  "recovery_trades_mean": 8.0,
  "worst": -359.7,
  "bottom3_cum": -766.7,
  "cvar5": -255.6,
  "mae_mean": -87.8,
  "mae_p10": -197.4,
  "loss_rate": 0.56,
  "max_consec_loss": 7,
  "ret_vol": 131.5,
  "ret_to_dd": 0.21,
  "downside_dev": 83.9,
  "session_conc": {
   "EUROPE": 0.26,
   "OFF": 0.5,
   "US": 0.24
  },
  "regime_conc": {
   "DOWN": 0.3,
   "RANGE": 0.3,
   "UP": 0.4
  }
 },
 "candidate": {
  "n": 2,
  "trade_per_day": 0.16,
  "exposure_hours": 12.0,
  "cum": 226.5,
  "mean": 113.27,
  "median": 113.27,
  "per_active_hour": 18.879,
  "top1_removed": -14.2,
  "top3_removed": null,
  "top5_removed": null,
  "pf": 16.94,
  "mdd": -14.2,
  "avg_dd": -7.1,
  "dd_duration_max_trades": 1,
  "recovery_trades_mean": null,
  "worst": -14.2,
  "bottom3_cum": 226.5,
  "cvar5": -14.2,
  "mae_mean": -50.6,
  "mae_p10": -102.2,
  "loss_rate": 0.5,
  "max_consec_loss": 1,
  "ret_vol": 127.5,
  "ret_to_dd": 15.94,
  "downside_dev": 10.0,
  "session_conc": {
   "EUROPE": 0.5,
   "OFF": 0.5
  },
  "regime_conc": {
   "UP": 1.0
  }
 },
 "regime_only": {
  "n": 20,
  "trade_per_day": 1.63,
  "exposure_hours": 120.0,
  "cum": 486.2,
  "mean": 24.31,
  "median": 8.52,
  "per_active_hour": 4.051,
  "top1_removed": 107.2,
  "top3_removed": -361.4,
  "top5_removed": -554.7,
  "pf": 1.63,
  "mdd": -383.9,
  "avg_dd": -119.3,
  "dd_duration_max_trades": 8,
  "recovery_trades_mean": 3.0,
  "worst": -195.4,
  "bottom3_cum": -561.8,
  "cvar5": -195.4,
  "mae_mean": -82.7,
  "mae_p10": -200.6,
  "loss_rate": 0.5,
  "max_consec_loss": 2,
  "ret_vol": 139.4,
  "ret_to_dd": 1.27,
  "downside_dev": 76.5,
  "session_conc": {
   "EUROPE": 0.25,
   "OFF": 0.55,
   "US": 0.2
  },
  "regime_conc": {
   "UP": 1.0
  }
 },
 "latent_only": {
  "n": 5,
  "trade_per_day": 0.41,
  "exposure_hours": 30.0,
  "cum": 298.7,
  "mean": 59.75,
  "median": 3.99,
  "per_active_hour": 9.958,
  "top1_removed": 58.0,
  "top3_removed": -24.1,
  "top5_removed": null,
  "pf": 13.41,
  "mdd": -14.2,
  "avg_dd": -4.8,
  "dd_duration_max_trades": 1,
  "recovery_trades_mean": 1.0,
  "worst": -14.2,
  "bottom3_cum": -20.1,
  "cvar5": -14.2,
  "mae_mean": -47.8,
  "mae_p10": -97.4,
  "loss_rate": 0.4,
  "max_consec_loss": 1,
  "ret_vol": 96.5,
  "ret_to_dd": 21.03,
  "downside_dev": 7.7,
  "session_conc": {
   "EUROPE": 0.2,
   "OFF": 0.8
  },
  "regime_conc": {
   "RANGE": 0.6,
   "UP": 0.4
  }
 },
 "evaluable": false,
 "sample_status": "INSUFFICIENT_SAMPLE",
 "matched_count": {
  "skipped": "n_cand<min"
 },
 "random_veto": {
  "skipped": "n_cand<min"
 }
}
```

## Fold 3 val=[0.85, 1.0] **[CONTAMINATED]**
```json
{
 "val_win": [
  0.85,
  1.0
 ],
 "contaminated": true,
 "calm_state": 0,
 "n_base": 48,
 "n_cand": 1,
 "baseline": {
  "n": 48,
  "trade_per_day": 3.9,
  "exposure_hours": 288.0,
  "cum": -321.9,
  "mean": -6.71,
  "median": -17.24,
  "per_active_hour": -1.118,
  "top1_removed": -675.4,
  "top3_removed": -1308.5,
  "top5_removed": -1769.6,
  "pf": 0.89,
  "mdd": -1359.6,
  "avg_dd": -843.8,
  "dd_duration_max_trades": 41,
  "recovery_trades_mean": 5.0,
  "worst": -657.5,
  "bottom3_cum": -1290.0,
  "cvar5": -430.0,
  "mae_mean": -112.4,
  "mae_p10": -184.6,
  "loss_rate": 0.542,
  "max_consec_loss": 5,
  "ret_vol": 169.5,
  "ret_to_dd": -0.24,
  "downside_dev": 130.0,
  "session_conc": {
   "EUROPE": 0.25,
   "OFF": 0.5,
   "US": 0.25
  },
  "regime_conc": {
   "DOWN": 0.33,
   "RANGE": 0.38,
   "UP": 0.29
  }
 },
 "candidate": {
  "n": 1,
  "trade_per_day": 0.08,
  "exposure_hours": 6.0,
  "cum": 53.5,
  "mean": 53.49,
  "median": 53.49,
  "per_active_hour": 8.914,
  "top1_removed": null,
  "top3_removed": null,
  "top5_removed": null,
  "pf": null,
  "mdd": 0.0,
  "avg_dd": 0.0,
  "dd_duration_max_trades": 0,
  "recovery_trades_mean": null,
  "worst": 53.5,
  "bottom3_cum": 53.5,
  "cvar5": 53.5,
  "mae_mean": -4.5,
  "mae_p10": -4.5,
  "loss_rate": 0.0,
  "max_consec_loss": 0,
  "ret_vol": 0.0,
  "ret_to_dd": null,
  "downside_dev": 0.0,
  "session_conc": {
   "EUROPE": 1.0
  },
  "regime_conc": {
   "UP": 1.0
  }
 },
 "regime_only": {
  "n": 14,
  "trade_per_day": 1.14,
  "exposure_hours": 84.0,
  "cum": -514.1,
  "mean": -36.72,
  "median": -2.92,
  "per_active_hour": -6.12,
  "top1_removed": -822.4,
  "top3_removed": -1080.5,
  "top5_removed": -1189.7,
  "pf": 0.59,
  "mdd": -1136.2,
  "avg_dd": -675.2,
  "dd_duration_max_trades": 13,
  "recovery_trades_mean": null,
  "worst": -657.5,
  "bottom3_cum": -917.9,
  "cvar5": -657.5,
  "mae_mean": -120.1,
  "mae_p10": -179.8,
  "loss_rate": 0.5,
  "max_consec_loss": 4,
  "ret_vol": 209.9,
  "ret_to_dd": -0.45,
  "downside_dev": 188.6,
  "session_conc": {
   "EUROPE": 0.29,
   "OFF": 0.43,
   "US": 0.29
  },
  "regime_conc": {
   "UP": 1.0
  }
 },
 "latent_only": {
  "n": 8,
  "trade_per_day": 0.65,
  "exposure_hours": 48.0,
  "cum": 16.7,
  "mean": 2.09,
  "median": -12.54,
  "per_active_hour": 0.348,
  "top1_removed": -148.0,
  "top3_removed": -280.1,
  "top5_removed": -255.1,
  "pf": 1.06,
  "mdd": -161.2,
  "avg_dd": -74.6,
  "dd_duration_max_trades": 3,
  "recovery_trades_mean": 3.0,
  "worst": -101.2,
  "bottom3_cum": -255.1,
  "cvar5": -101.2,
  "mae_mean": -55.9,
  "mae_p10": -106.9,
  "loss_rate": 0.5,
  "max_consec_loss": 2,
  "ret_vol": 86.3,
  "ret_to_dd": 0.1,
  "downside_dev": 54.1,
  "session_conc": {
   "EUROPE": 0.38,
   "OFF": 0.5,
   "US": 0.12
  },
  "regime_conc": {
   "DOWN": 0.38,
   "RANGE": 0.5,
   "UP": 0.12
  }
 },
 "evaluable": false,
 "sample_status": "INSUFFICIENT_SAMPLE",
 "matched_count": {
  "skipped": "n_cand<min"
 },
 "random_veto": {
  "skipped": "n_cand<min"
 }
}
```

## Alarm lead/lag
```json
{
 "train_mean": -10.0,
 "train_std": 113.0,
 "deterioration_threshold": -66.5,
 "n_windows": 13,
 "windows": [
  {
   "end_idx": 16197,
   "end_utc": "06-06",
   "status": "UNUSABLE",
   "back_mean": -94.4,
   "fwd_mean": 20.1,
   "n_back": 28,
   "n_fwd": 28
  },
  {
   "end_idx": 16773,
   "end_utc": "06-08",
   "status": "UNUSABLE",
   "back_mean": -64.8,
   "fwd_mean": 18.3,
   "n_back": 28,
   "n_fwd": 28
  },
  {
   "end_idx": 17349,
   "end_utc": "06-10",
   "status": "UNUSABLE",
   "back_mean": -43.6,
   "fwd_mean": 23.4,
   "n_back": 28,
   "n_fwd": 28
  },
  {
   "end_idx": 17925,
   "end_utc": "06-12",
   "status": "UNUSABLE",
   "back_mean": 10.3,
   "fwd_mean": 5.8,
   "n_back": 28,
   "n_fwd": 28
  },
  {
   "end_idx": 18501,
   "end_utc": "06-14",
   "status": "UNUSABLE",
   "back_mean": 4.4,
   "fwd_mean": -7.0,
   "n_back": 28,
   "n_fwd": 28
  },
  {
   "end_idx": 19077,
   "end_utc": "06-16",
   "status": "UNUSABLE",
   "back_mean": 28.0,
   "fwd_mean": -30.2,
   "n_back": 28,
   "n_fwd": 28
  },
  {
   "end_idx": 19653,
   "end_utc": "06-18",
   "status": "UNUSABLE",
   "back_mean": 3.0,
   "fwd_mean": -33.9,
   "n_back": 28,
   "n_fwd": 28
  },
  {
   "end_idx": 20229,
   "end_utc": "06-20",
   "status": "UNUSABLE",
   "back_mean": 8.6,
   "fwd_mean": -38.3,
   "n_back": 28,
   "n_fwd": 28
  },
  {
   "end_idx": 20805,
   "end_utc": "06-22",
   "status": "UNUSABLE",
   "back_mean": -18.6,
   "fwd_mean": -27.7,
   "n_back": 28,
   "n_fwd": 28
  },
  {
   "end_idx": 21381,
   "end_utc": "06-24",
   "status": "UNUSABLE",
   "back_mean": -30.7,
   "fwd_mean": -7.9,
   "n_back": 28,
   "n_fwd": 28
  },
  {
   "end_idx": 21957,
   "end_utc": "06-26",
   "status": "UNUSABLE",
   "back_mean": -31.6,
   "fwd_mean": 28.0,
   "n_back": 28,
   "n_fwd": 23
  },
  {
   "end_idx": 22533,
   "end_utc": "06-28",
   "status": "UNUSABLE",
   "back_mean": -35.1,
   "fwd_mean": 51.3,
   "n_back": 28,
   "n_fwd": 15
  },
  {
   "end_idx": 23109,
   "end_utc": "06-30",
   "status": "UNUSABLE",
   "back_mean": -24.4,
   "fwd_mean": 108.4,
   "n_back": 28,
   "n_fwd": 7
  }
 ],
 "first_alarm_window": 0,
 "first_deterioration_window": 0,
 "alarm_lead_days": 0.0,
 "alarm_leading": true,
 "false_positive_suspension_rate": 0.69,
 "blocked_good_trades_after_alarm": 145,
 "blocked_bad_trades_after_alarm": 180
}
```

Durust statuler: software-correct ✓ · frequency-normalized ✓ (matched-count + random-veto esit-N) · risk-non-incremental · applicability-leading · walk-forward failed · forward-not-validating (N=0) · **operationally FORBIDDEN** (max: RESEARCH/BACKTEST/SHADOW + SHADOW_SUSPEND_SUGGESTION)

*Runner: `python -m ami.latent.risk_applicability`*
## Yorum (post-hoc; frozen kriterlere DOKUNULMADI)

1. **Seçim çöküşü bulgunun kendisidir:** aday veto (trend=UP ∧ latent-calm) fold 1-3'te
   yalnız 1-2 trade seçti — 6A'da ölçülen rejim kayması altında "calm" state'in kimliği
   per-fold refit'te değişiyor (fold0 calm=1, sonra calm=0) ve occupancy çöküyor.
   Applicability overlay'i devreye giremiyorsa risk avantajı iddiası test bile edilemez.
2. **Tek değerlendirilebilir fold'da (fold0) aday, regime-only ile BİREBİR AYNI set** —
   latent katmanın incremental katkısı sıfır; matched-count/random-veto percentile ~0.65
   (<0.75) ve retention_ratio 0.47 (kazananları orantısız atıyor).
3. **6A-R'deki N=14 mdd −416 farkının kaynağı:** ALL-era-fit artifact + hipotez-kaynağı
   pencere. Per-fold dürüst refit altında aynı aday YENİDEN ÜRETİLEMEDİ.
4. **Alarm SATÜRE:** validation erasında 13/13 pencere UNUSABLE. "Lead ≥ 0" frozen kriteri
   teknik olarak sağlandı ama sürekli-açık alarmın ayırt edici değeri yoktur;
   false-positive suspension 0.69. Dürüst statü: applicability-**degenerate/saturated**
   (leading iddiası kanıt sayılmaz).
5. Frozen öncelik sırasına göre sınıf: **INSUFFICIENT_SAMPLE** (toplam aday N=9 < 40,
   değerlendirilebilir fold 1 < 2). Retry koşulu: forward shadow verisi birikince
   (≥6 ay) YENİ prereg; kriter gevşetme YASAK.
