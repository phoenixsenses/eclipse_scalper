# Micro Edge Signal V2 (Passive Alpha)

## Purpose
`micro_edge_v2_passive_alpha` upgrades v1 directional spikes into a fillability-aware microstructure score for `exec_model=passive_realistic`.

Success criterion:
- `pass_rate >= 0.50` at `maker_fee_bps=1.0`
- Survives adverse stress `passive_adverse_mult in {1.0, 1.2}`

## Feature Definitions
All features are computed from existing bucketed series (no L2 assumptions):

- `v2_delta_spread = spread_t - spread_{t-1}`
- `v2_spread_recompress = (spread_{t-1} - spread_t) / spread_{t-1}`
- `v2_d_intensity = intensity_t - intensity_{t-1}`
- `v2_intensity_decay = (max_recent_intensity - intensity_t) / max_recent_intensity`
- `v2_vol_change = vol_t - vol_{t-1}`
- `v2_vol_stabilize = -v2_vol_change`
- `v2_imbalance_ema = EMA(imbalance)`
- `v2_streak_norm = signed return streak / long_window`
- `v2_imbalance_persist = 0.5*v2_imbalance_ema + 0.5*v2_streak_norm`
- `v2_flip_rate = sign-flip rate over short window`
- `v2_meanrev_prob = blend(flip_rate, |ret_1| burst indicator)`

Score and confidence:
- `toxicity = max(0, v2_d_intensity) + max(0, v2_vol_change) + 0.25*v2_meanrev_prob`
- `fill_quality = 0.45*v2_spread_recompress + 0.35*v2_intensity_decay + 0.20*v2_vol_stabilize`
- `core = |v2_imbalance_persist|*fill_quality - 0.50*toxicity`
- `v2_score = sign(v2_imbalance_persist) * core`
- `v2_confidence = sigmoid(|core|*40)`

Rule fires when:
- `|v2_score| >= v2_min_score` (default train q80)
- `|v2_imbalance_persist| >= v2_min_persistence` (default train q60)
- `v2_confidence >= v2_min_confidence` (default 0.50)

Side (`--side auto`):
- LONG if `v2_side_signal > 0`
- SHORT if `v2_side_signal < 0`

## Why This Reduces Adverse Selection
- Penalizes entries during volatility/intensity bursts (`toxicity` term).
- Prefers post-spike normalization (`spread_recompress`, `intensity_decay`, `vol_stabilize`) where passive fills are more likely and less toxic.
- Requires persistence, not one-tick imbalance noise (`imbalance_ema + streak`).

## Runbook (PowerShell)

Compile changed modules:
```powershell
python -m py_compile tools/micro_edge_signal_v2.py tools/micro_edge_lib.py tools/micro_edge_backtest.py tools/sweep_passive_realistic_filters.py tools/validate_passive_pocket_forward.py tools/rank_passive_pockets_forward.py
```

Targeted tests:
```powershell
pytest -q tests/test_micro_edge_signal_v2.py tests/test_validate_pocket_forward_api.py tests/test_rank_passive_pockets_forward.py tests/test_passive_adverse_mult.py
```

Baseline v1 vs v2 backtest (same config):
```powershell
python -m tools.micro_edge_backtest --db data/microstructure.db --symbols BTCUSDT,ETHUSDT --lookback-min 1440 --bucket-sec 1 --horizon-sec 60 --rule intensity_spike_imbalance_cont --side auto --exec-model passive_realistic --maker-fee-bps 1.0 --passive-adverse-mult 1.0
python -m tools.micro_edge_backtest --db data/microstructure.db --symbols BTCUSDT,ETHUSDT --lookback-min 1440 --bucket-sec 1 --horizon-sec 60 --rule micro_edge_v2_passive_alpha --side auto --exec-model passive_realistic --maker-fee-bps 1.0 --passive-adverse-mult 1.0 --v2-min-confidence 0.5
```

Sweep v2 pockets (low-DOF knobs):
```powershell
python -m tools.sweep_passive_realistic_filters --db data/microstructure.db --symbols BTCUSDT,ETHUSDT --lookback-min 1440 --bucket-sec 1 --horizon-grid 30,60,120 --rule micro_edge_v2_passive_alpha --side auto --maker-fee-bps 1.0 --passive-adverse-mult 1.0 --v2-min-score-grid 0.0,0.0005,0.0010 --v2-min-persistence-grid 0.2,0.3,0.4 --out-md reports/FILTER_SWEEP_PASSIVE_REALISTIC_V2.md
```

Forward validate one pocket:
```powershell
python -m tools.validate_passive_pocket_forward --db data/microstructure.db --symbol ETHUSDT --lookback-min 1440 --bucket-sec 1 --horizon-sec 60 --rule micro_edge_v2_passive_alpha --side auto --min-imbalance 0.3 --min-trade-intensity 2500 --max-spread 0.00025 --v2-min-score 0.0005 --v2-min-persistence 0.3 --v2-min-confidence 0.5 --splits 4 --seeds 11,22,33,44,55 --min-n 50 --min-n-frac 0.0 --maker-fee-bps 1.0 --passive-adverse-mult 1.2 --out-md reports/PASSIVE_POCKET_FORWARD_VALIDATION_V2.md
```

Rank all candidate pockets (v1+v2):
```powershell
python -m tools.rank_passive_pockets_forward --db data/microstructure.db --lookback-min 1440 --bucket-sec 1 --rules intensity_spike_imbalance_cont,micro_edge_v2_passive_alpha --side auto --candidates-md reports/FILTER_SWEEP_PASSIVE_REALISTIC_ETH.md,reports/FILTER_SWEEP_PASSIVE_REALISTIC_BTC.md --splits 4 --seeds 11,22,33,44,55 --min-n 50 --min-n-frac 0.0 --maker-fee-bps-grid 0.5,1.0,1.5 --passive-adverse-mult-grid 0.8,1.0,1.2 --v2-min-score 0.0005 --v2-min-persistence 0.3 --v2-min-confidence 0.5 --out-md reports/PASSIVE_POCKET_RANKING.md --out-json reports/PASSIVE_POCKET_RANKING.json
```

## Notes
- Research-only. No live execution wiring is changed.
- Deterministic behavior comes from fixed seed lists + deterministic passive simulator hashing.
