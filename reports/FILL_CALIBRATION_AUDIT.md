# FILL CALIBRATION AUDIT -- ETHUSDT


**Generated:** 2026-02-28 23:21 UTC


## Context

Prior analysis (NPA_DECOMPOSITION_REGIME.md) showed:
- Sell NPA = 6.3e-05 vs Buy NPA = 7.0e-06  (9:1 ratio)
- Path excursions are SYMMETRIC (sell_adverse=10.21 bps vs buy_adverse=10.15 bps)
- NPA gap is NOT from directional hit rates (both ~55-56%)

This report isolates the gap by auditing the passive fill probability model:
  - touch_rate: does price reach the passive limit within the 120s hold?
  - full_proxy_rate: is depth >= 0.5 when touched (proxy for queue priority)?
  - adverse_bps: 1-step reversal immediately after fill
  - conditional return: trade P&L at horizon given fill

Passive fill mechanics:
  SHORT limit = ep*(1+0.5*spread)  -- fills when price goes UP
  LONG  limit = ep*(1-0.5*spread)  -- fills when price goes DOWN

SELL signals: imbalance <= -0.5  |  BUY signals: imbalance >= +0.5
Top pocket: trade_intensity>=3500, spread<=0.0003, horizon=120s


## 1. Basic Fill Statistics

| Metric | SELL (SHORT) | BUY (LONG) | Sell / Buy ratio |
|                      ---|           ---|           ---|             ---|
| n signals | 17563 | 20872 | — |
| n touched | 13455 | 16474 | — |
| n full_proxy | 6965 | 7997 | — |
| touch_rate | 0.7661 | 0.7893 | 0.97 |
| full_rate | touch | 0.5177 | 0.4854 | 1.07 |
| fill_rate (joint) | 0.3966 | 0.3831 | 1.04 |
| depth_mean | touch | 3.0544 | 3.2709 | 0.93 |
| depth_p50 | touch | 0.5335 | 0.4765 | 1.12 |
| spread_mean | 0.000152 | 0.000134 | — |
| spread_p50 | 0.000152 | 0.000129 | — |


## 2. adverse_bps Distribution (1-step reversal after touch)


> adverse_bps is computed at the moment of fill (touch), not over the full hold.
> Formula:
>   SHORT: max(0, (p_touch - p_next) / p_touch * 10000)
>   LONG:  max(0, (p_next - p_touch) / p_touch * 10000)
> This measures immediate 1-second reversal AFTER the passive limit is touched.

| Stat | SELL (SHORT) | BUY (LONG) | Sell - Buy |
|                      ---|           ---|           ---|         ---|
| adverse_bps mean (all) | 0.2658 | 0.2014 | 0.0644 |
| adverse_bps mean (touched) | 0.3469 | 0.2551 | 0.0918 |
| adverse_bps p50 (touched) | 0.0000 | 0.0000 | 0.0000 |
| adverse_bps p90 (touched) | 1.2898 | 0.8678 | 0.4220 |


## 3. Touch Timing Distribution

Mean seconds to first touch (within 120s hold), conditional on touching.

| Stat | SELL (SHORT) | BUY (LONG) |
|                      ---|           ---|           ---|
| touch_idx mean (s) | 22.2279 | 19.9662 |
| touch_idx p25 (s) | 3.0000 | 3.0000 |
| touch_idx p50 (s) | 10.0000 | 9.0000 |
| touch_idx p75 (s) | 30.0000 | 26.0000 |


## 4. Conditional Trade Returns (given touched at limit)


Return computed from the FILL PRICE to the 120s horizon exit price.
  SHORT: (p_touch - p_exit) / p_touch * 10000 bps
  LONG:  (p_exit - p_touch) / p_touch * 10000 bps

| Stat | SELL (SHORT) | BUY (LONG) | Sell - Buy |
|                      ---|           ---|           ---|         ---|
| cond_ret mean (bps) | -0.4776 | -1.0479 | 0.5703 |
| cond_ret p25 (bps) | -8.7675 | -9.3382 | 0.5708 |
| cond_ret p50 (bps) | -0.4518 | -0.8145 | 0.3627 |
| cond_ret p75 (bps) | 7.6508 | 7.5315 | 0.1194 |


## 5. Touch Rate by Spread Tertile

Touch rate within each spread tercile (T1=tight, T2=mid, T3=wide).

| Tertile | SELL n | SELL touch_rate | SELL fill_rate | BUY n | BUY touch_rate | BUY fill_rate | spread_mean |
|         ---|      ---|            ---|           ---|      ---|            ---|           ---|         ---|
| T1_tight | 5849 | 0.8345 | 0.6288 | 6951 | 0.8347 | 0.6188 | 0.00006 |
| T2_mid | 5865 | 0.7596 | 0.3478 | 6970 | 0.7901 | 0.3364 | 0.00015 |
| T3_wide | 5849 | 0.7042 | 0.2132 | 6951 | 0.7431 | 0.1944 | 0.00024 |


## 6. Calibrated Model Parameters


Parameters from inline replication of calibrate_passive_model (same algorithm as
execution/passive_execution_simulator.py). These are the base rates the simulator
blends against per-feature tertile corrections.

| Parameter | SELL | BUY |
|                           ---|         ---|         ---|
| n samples | 17563 | 20872 |
| n touched | 13455 | 16474 |
| base_touch | 0.7661 | 0.7893 |
| base_full_cond_touch | 0.5177 | 0.4854 |
| implied_fill_rate = base_touch x base_full | 0.3966 | 0.3831 |
| base_adverse_bps | 0.2658 | 0.2014 |
| depth_full_threshold | 0.5335 | 0.4765 |


## 7. NPA Decomposition and Counterfactual


NPA estimate (bps per signal):
  NPA = fill_rate_joint * (cond_ret_mean - adverse_bps_touched_mean - 2*maker_fee)
  maker_fee = 2.0 bps one-way

Note: This is an approximation. The actual validate_pocket_forward NPA uses the
full simulation (including queue competition penalty and partial fill logic).

| Component | SELL | BUY | Sell / Buy |
|                                ---|         ---|         ---|         ---|
| fill_rate_joint | 0.3966 | 0.3831 | 1.04 |
| cond_ret_mean (bps) | -0.4776 | -1.0479 | 0.46 |
| NPA estimate (bps) | -1.9133 | -2.0318 | 0.94 |

**Counterfactual analysis** (swapping SELL parameters to BUY values):

| Scenario | SELL NPA (bps) | vs actual SELL | Interpretation |
|                                     ---|            ---|            ---|                                ---|
| Actual SELL | -1.9133 | — | Baseline |
| CF1: use BUY fill_rate, keep SELL cond_ret | -1.8485 | 0.0648 | Isolate fill_rate effect |
| CF2: keep SELL fill_rate, use BUY cond_ret | -2.1394 | -0.2262 | Isolate return effect |
| CF3: use BUY fill_rate + BUY cond_ret | -2.0318 | -0.1186 | Full convergence = actual BUY |
| Actual BUY | -2.0318 | — | Reference |


## 8. Verdict: Source of 9:1 NPA Gap


**Fill rate ratio (SELL/BUY):** 1.04x
**Conditional return ratio (SELL/BUY):** 0.46x

| Factor | Value | Contribution to gap |
|---|---:|---|

| fill_rate | 1.04x | secondary |

| cond_ret  | 0.46x | PRIMARY |


The NPA gap is primarily driven by ASYMMETRIC CONDITIONAL RETURNS. SELL signals, once filled, achieve better 120s returns than BUY signals. This suggests the sell-side alpha (mean-reversion after momentum) has stronger post-fill directional persistence than the buy-side alpha.


---
*Generated by `tools/audit_fill_calibration.py` -- self-contained.*
