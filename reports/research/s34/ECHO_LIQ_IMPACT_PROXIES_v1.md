# Echo Liq/Impact Proxies — Descriptive Characterization (v1)

_generated 2026-07-20T10:00:21.273939+00:00 · READ-ONLY · OD-029 SAFE (no outcome mining)_

DESCRIPTIVE characterization only. NO outcome/net_bps conditioning. Predictive claims are FORWARD-ONLY (OD-029). Anchor population identical to echo_forward_ledger frozen params.

- Anchors: **695** (ETHUSDT SELL, frozen echo params) · regime-gate ON: 553 · window 1771166036254–1784541146341

## Proxy distributions & coverage

| proxy | cover% | p10 | p25 | p50 | p75 | p90 | mean |
|---|---|---|---|---|---|---|---|
| kyle_lambda | 100.0 | -0.1478 | 1.348 | 2.392 | 4.08 | 8.055 | 0.8174 |
| amihud | 100.0 | 0.1188 | 0.2364 | 0.3929 | 0.5749 | 0.7574 | 0.4267 |
| rv_bps | 55.5 | 8.851 | 11.39 | 16.13 | 22.82 | 34.64 | 20.03 |
| bv_bps | 55.5 | 3.159 | 4.111 | 6.082 | 9.652 | 15.97 | 8.675 |
| jump_frac | 55.5 | 0.6811 | 0.7865 | 0.8455 | 0.8821 | 0.9148 | 0.8225 |
| liq_impact_bps_per_M | 100.0 | 332.9 | 628.4 | 1355 | 2660 | 4624 | 2137 |

## Mutual rank-correlation (redundancy check)

| pair | Spearman rho | n |
|---|---|---|
| rv_bps~bv_bps | 0.919 | 386 |
| bv_bps~jump_frac | -0.59 | 386 |
| amihud~rv_bps | -0.517 | 386 |
| kyle_lambda~amihud | 0.508 | 695 |
| amihud~bv_bps | -0.413 | 386 |
| kyle_lambda~rv_bps | -0.307 | 386 |
| kyle_lambda~bv_bps | -0.291 | 386 |
| rv_bps~jump_frac | -0.266 | 386 |
| bv_bps~liq_impact_bps_per_M | -0.154 | 386 |
| amihud~regime_int | 0.153 | 695 |
| jump_frac~liq_impact_bps_per_M | 0.144 | 386 |
| kyle_lambda~jump_frac | 0.14 | 386 |
| rv_bps~liq_impact_bps_per_M | -0.133 | 386 |
| kyle_lambda~regime_int | 0.129 | 695 |
| rv_bps~regime_int | -0.107 | 386 |
| bv_bps~regime_int | -0.085 | 386 |
| kyle_lambda~liq_impact_bps_per_M | 0.069 | 695 |
| amihud~liq_impact_bps_per_M | -0.026 | 695 |
| jump_frac~regime_int | 0.018 | 386 |
| amihud~jump_frac | 0.011 | 386 |
| liq_impact_bps_per_M~regime_int | 0.001 | 695 |

## Regime-gate cross-tab (T0-knowable split, NOT outcome)

| proxy | regime ON median (n) | regime OFF median (n) |
|---|---|---|
| kyle_lambda | 2.592 (553) | 1.951 (142) |
| amihud | 0.4049 (553) | 0.3264 (142) |
| rv_bps | 15.69 (300) | 17.47 (86) |
| bv_bps | 5.95 (300) | 6.597 (86) |
| jump_frac | 0.8445 (300) | 0.8479 (86) |
| liq_impact_bps_per_M | 1339 (553) | 1458 (142) |

## Boundary (do not cross without forward data)
- This is structure/feasibility only. Whether any proxy improves the echo lead's NET is a
  FORWARD question: wire the surviving proxies into research_s34_echo_forward_ledger.py's
  indicator_snapshot() as new causal fields (dev-list #15/#19 `o candidate` -> `captured`),
  then accumulate post-2026-07-20 anchors. No threshold is selected here.
