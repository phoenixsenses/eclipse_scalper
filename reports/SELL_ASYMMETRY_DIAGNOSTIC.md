# SELL ASYMMETRY DIAGNOSTIC -- ETHUSDT

**Generated:** 2026-02-28 22:35 UTC

## Overview

Investigates whether the sell-side passive pocket edge on ETHUSDT is genuine microstructure alpha or a directional artifact from a bearish data window.

**Pocket filter (top pocket from sweep):**
- `abs(imbalance) >= 0.5`
- `trade_intensity >= 3500`  (trades/min equivalent at 1-sec buckets)
- `spread_proxy <= 0.000300`  (VWAP deviation from mark price)
- Horizon: 120 s
- Rule: `micro_edge_v3_passive_alpha` direction logic -- sell when imbalance <= -0.5

> **Spread note:** The original `build_bucket_features` computes spread as the per-trade average of `|price - mark_price| / mark_price`. This diagnostic uses `|VWAP - mark_price| / mark_price` as a bucket-level approximation. For high-liquidity ETH futures this is a close proxy; the threshold is loose enough that conclusions are not sensitive to the approximation method.

---
## 1. Data Coverage & Price Regime

| Item | Value |
|---|---|
| Symbol | ETHUSDT |
| Data window | 2026-02-15 14:26 UTC -> 2026-02-28 22:35 UTC |
| Duration | 13.3 days |
| Start price | $2,001.61 |
| End price | $1,974.12 |
| **Total drift** | **-1.37% (-137 bps)** |

![Price trajectory](plots\01_price_trajectory.png)

---
## 2. Sub-period Stability Test

The data window is split into 4 equal sub-periods. For each, the raw signal hit rates are computed independently to detect whether the asymmetry is concentrated in one regime slice or consistent throughout.

| Period | Date Range | Drift (bps/bkt) | v Baseline | Sell n | Sell HR | Buy n | Buy HR | Sell - Buy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | 02-15 14:26 -> 02-18 21:50 | -0.133 | 49.9% | 4570 | 55.7% | 5701 | 55.3% | +0.4% |
| P2 | 02-18 21:50 -> 02-22 08:01 | +0.048 | 49.6% | 2768 | 56.6% | 3171 | 57.4% | -0.8% |
| P3 | 02-22 08:01 -> 02-25 15:45 | +0.106 | 50.0% | 3958 | 55.1% | 4766 | 56.5% | -1.4% |
| P4 | 02-25 15:45 -> 02-28 22:33 | -0.116 | 50.4% | 6188 | 55.1% | 7147 | 56.7% | -1.6% |

![Sub-period hit rates](plots\02_subperiod_hit_rates.png)

**Sell HR** = fraction of sell-pocket signals where price falls within 120 s (favorable for SHORT).  
**v Baseline** = unconditional fraction of 120-s windows that are down in that sub-period.  
A sell HR close to the baseline suggests no microstructure alpha beyond trend following.

---
## 3. Regime-Conditioned Analysis

Each 1-second bucket is labelled **UP** or **DOWN** based on the rolling 1-hour log-return of the mark price (UP if ret >= 0, DOWN otherwise). Signal performance is then measured separately within each regime label.

| Regime | Buckets | % of time | Drift (bps/bkt) | v Baseline | Sell n | Sell HR | Sell edge (bps) | Buy n | Buy HR | Buy edge (bps) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **UP** | 567,913 | 50.6% | +0.5064 | 50.4% | 8036 | 56.7% | 1.215 | 9990 | 55.8% | 3.422 |
| **DOWN** | 553,886 | 49.4% | -0.5675 | 49.6% | 9448 | 54.4% | 3.087 | 10795 | 56.9% | 1.424 |
| **ALL** | 1,121,799 | 100.0% | -0.0238 | 50.0% | 17484 | 55.5% | 2.226 | 20785 | 56.4% | 2.384 |

![Regime breakdown](plots\03_regime_breakdown.png)

![Signal feature space](plots\04_signal_feature_space.png)

---
## 4. Key Diagnostic Statistics

| Metric | Value |
|---|---|
| Sell hit-rate (all regimes) | 55.49% |
| Buy  hit-rate (all regimes) | 56.37% |
| Sell HR in UP regime        | 56.72% |
| Sell HR in DOWN regime      | 54.43% |
| DOWN - UP sell HR gap       | -2.28% |
| Sell edge above baseline (all) | 5.50% |
| Sell edge above baseline (UP)  | 6.33% |
| Sell edge above baseline (DOWN) | 4.87% |
| Sell HR std (across sub-periods) | 0.0062 |
| Sell HR range (max-min)          | 1.50% |
| Sell signals in UP regime   | 8036 |
| Sell signals in DOWN regime | 9448 |

---
## 5. Verdict

### [OK] LIKELY_STRUCTURAL

The sell-side edge appears **structurally robust**. It persists across sub-periods and is not confined to DOWN regimes, suggesting genuine microstructure alpha above the directional baseline. However, always validate on fresh data before going live.

### Evidence flags

- **NEUTRAL_WINDOW**: Total price drift = -1.4% -- roughly balanced window.
- **REGIME_ROBUST_SELL**: Sell hit-rate gap between DOWN vs UP regime = -2.3%. Signal is relatively regime-robust.
- **EDGE_OVER_BASELINE**: Sell signal adds 5.50% above the unconditional down-fraction (50.0% -> 55.5%).
- **CONSISTENT_SUBPERIODS**: Sell hit-rate std across sub-periods = 0.006 -- reasonably stable across time.

### Interpretation guide

- If sell HR >> buy HR across **all regimes** (UP and DOWN) and consistently across sub-periods -> the asymmetry is likely structural microstructure alpha.
- If sell HR is elevated **only in DOWN regime** and approximately equals the unconditional down-fraction -> the signal is merely a proxy for trend, not edge.
- If the sell HR gap (DOWN - UP) > 5 pp the signal has non-trivial regime sensitivity. Consider regime-filtering or collecting data across a bull run to confirm robustness.

---
*Generated by `tools/diagnose_sell_asymmetry.py` -- self-contained, no dependency on tools/ modules.*
