# Passive Scratch And Horizon Override

## Purpose
- Add optional post-fill scratch/escape behavior for passive simulations.
- Add optional horizon override in rank sweeps without changing candidate files.

## New Rank CLI options
- `--horizon-sec`: override candidate `horizon_sec` (default `0` keeps candidate horizon).
- `--scratch-bps`: adverse move threshold in bps for scratch trigger.
- `--scratch-window-sec`: seconds after fill to monitor scratch trigger.
- `--scratch-taker-fee-bps`: one-way taker fee bps added on scratch exit.
- `--scratch-slippage-bps`: one-way extra slippage bps added on scratch exit.
- `--mitigation-profile anti_adverse_v4`: applies `anti_adverse_v3` volatility quantile gate plus conservative scratch defaults:
  - `scratch_bps=4.0`
  - `scratch_window_sec=10`
  - `scratch_taker_fee_bps=1.0`
  - `scratch_slippage_bps=0.5`
  - `vol_quantile_reject` from CLI (default `0.01`)

Behavior is unchanged unless these options are set or `anti_adverse_v4` is selected.

## Repro (PowerShell)
```powershell
python -m tools.rank_passive_pockets_forward `
  --db data/microstructure.db `
  --lookback-min 30240 `
  --bucket-sec 1 `
  --rule micro_edge_v3_passive_alpha `
  --side sell `
  --candidates-md reports/FILTER_SWEEP_V3_21D_ETH_h120_ADV1p2.md `
  --splits 3 `
  --seeds 7,11,22,33,44,55,66,77,88 `
  --min-n 20 `
  --min-n-frac 0.00010 `
  --maker-fee-bps-grid 0.8,0.9,1.0 `
  --passive-adverse-mult-grid 1.0,1.2,1.5 `
  --mitigation-profile anti_adverse_v4 `
  --vol-quantile-reject 0.02 `
  --horizon-sec 120 `
  --scratch-bps 4.0 `
  --scratch-window-sec 10 `
  --scratch-taker-fee-bps 1.0 `
  --scratch-slippage-bps 0.5 `
  --out-md reports/RANK_V3_V4_SCRATCH.md `
  --out-json reports/RANK_V3_V4_SCRATCH.json
```

