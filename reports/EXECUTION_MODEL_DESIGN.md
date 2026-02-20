# EXECUTION_MODEL_DESIGN

## Objective
Replace static maker-penalty assumptions with a deterministic, data-conditioned passive execution simulator.

## Module
`execution/passive_execution_simulator.py`

## Inputs
- Event: `symbol`, `event_id`, `side`, `entry_price`, `future_mids`
- Features: `spread`, `trade_intensity`, `vol_proxy`, `imbalance_for_fill`
- Params (calibrated):
  - `base_touch`
  - `base_full_cond_touch`
  - `base_adverse_bps`
  - per-feature tertile edges and rates
  - `partial_fill_fraction`, `depth_full_threshold`, `touch_without_cross_factor`

## Fill-state model
1. Compute passive touch feasibility from future path and passive limit offset `0.5*spread`.
2. Compute calibrated touch probability `p_touch` by blending base + feature-bin rates.
3. Draw deterministic uniform `u_touch = H(seed,event_id,"touch")`; fill attempt if `u_touch < p_touch`.
4. Conditional on fill, compute `p_full` from calibrated full-fill rates + depth adjustment.
5. Draw deterministic `u_full = H(seed,event_id,"full")`; full vs partial fill.
6. If no fill -> event is dropped (missed fill).

## Cost model (passive_realistic)
- Effective cost (bps) = `2*maker_fee_bps + adverse_selection_bps`
- Adverse selection is calibrated from empirical post-touch movement proxies and feature bins.
- Execution price adjustment approximates passive half-spread improvement, scaled by fill fraction.

## Determinism
- Randomness is hash-based (`sha256(seed|event_id|tag)`) so identical input dataset + seed yields identical results.

## Backtest integration
- New exec model: `--exec-model passive_realistic`
- Backtest debug rows now include:
  - `filled_flag`
  - `fill_fraction`
  - `effective_cost_bps`
  - `adverse_selection_bps`
  - `execution_price_adjustment`
- Summary includes passive attempts, fills, unfilled, partial, and attempt fill-rate.
