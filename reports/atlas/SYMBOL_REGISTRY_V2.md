# SYMBOL REGISTRY V2 — exponent family plus design family

`DESIGN_FAMILY_ADDED_AND_A_K_IS_C_TAIL_INDEX` · built 2026-08-27T03:37:10Z · lane C, `C-T26`

Supersedes `EXPONENT_SYMBOL_REGISTRY_V1` by containing it verbatim; V1 is **not withdrawn**.

**13 letters carry 25 objects** (15 exponent, 10 design); **8 letters are overloaded**.

---

## 1. A non-finding, recorded as one — `NO_DEFECT_THE_FILE_RESOLVES_ITS_OWN_CONVENTION`

**Checked:** whether the frozen prereg leaves the fee convention ambiguous, since h* goes as c^2 and A's own S5 trigger says a 2x cost change moves h* by 4x.

**Result:** line 56 states it -- 'section 460's frontier is single-leg at c = 10 bps' -- and line 45 contrasts the two-leg pairs case at 20 bps.

*The check was run and came back clean, which is worth as much as a finding and is usually not written down*

## 2. 🔴 The bridge — `A_K_IS_C_TAIL_INDEX`

A's `k = E|r|/σ` is frozen at **0.6966**. For a Gaussian that ratio is **0.7979**, so A sits **-12.69% below** it — a fat-tail signature, since the ratio falls monotonically as the tail thickens. Inverting on a standardised Student-t:

| statistic | source | tail index |
|---|---|--:|
| `k = E\|r\|/σ` → ν | A-S45 (bulk shape) | **3.765** |
| Hill on 60-min moves | §478 (order statistic) | **2.33–3.83** |
| universality | Bouchaud | ~3.0 |

**They agree: True.** The two statistics share no machinery: a mean-absolute-to-sigma ratio computed over a whole distribution, and an order-statistic estimator using only the largest observations.

> k is not a free constant. It is a function of the tail index, and the tail index is the quantity Bouchaud reports as near-universal.

### A free check for A's freeze

The freeze block says *"k = 0.6966 (§467; **re-measure at freeze**)"*. Since `k` is a function of the tail index, the re-measurement has a **predicted band**:

> **nu in [3, 5]  <->  k in [0.6366, 0.7351]**

- **inside** → tail regime unchanged, nothing to revisit
- **outside** → either the tail regime moved or the estimator broke; since N_required goes as k^-2, the cost of not noticing is a mis-sized sample
- **cost:** zero -- A already plans to re-measure k at freeze

## 3. The cost basis, measured where A assumed

A: *"Spread = one tick (section 452, 12 of 15 symbols)"*. C: holds on 3 of 3 majors at 97.7% / 98.8% / 99.9% of quotes -- the assumption is stronger than A recorded, not weaker.

| symbol | spread bps | c bps | spread / c | h\* factor |
|---|--:|--:|--:|--:|
| BTCUSDT | 0.0154 | 10.0154 | **0.154%** | **1.0031×** |
| ETHUSDT | 0.053 | 10.053 | **0.527%** | **1.0106×** |
| SOLUSDT | 1.3148 | 11.3148 | **11.620%** | **1.2802×** |

> c is FEE-DOMINATED on the small-tick majors: the spread moves the design horizon by 0.3% and 1.1%. On SOL it is 11.6% of cost and lifts h* by 28%. Cross-symbol variation in h* comes from sigma_d, not the spread -- except on the large-tick symbol.

## 4. Design-family collisions

- **`k`** — two objects -- a dimensionless distribution ratio and a dimensional impact prefactor
- **`h`** — THREE units across lanes: days (A's frontier), trades (C-T23's lag grid), minutes (the episode work). This already cost C-T23 an inference, where A-S40's h grid could not be placed on either scale from its text.
- **`f`** — NOT a collision -- A's capture scalar and A-S40's f(h) are the same object at one horizon, verified consistent

---

## The full registry

| family | letter | object | definition | measured | owner | shares letter with |
|---|---|---|---|---|---|---|
| exponent | `zeta` | **ZETA_WINDOW_IMBALANCE** | outer-region exponent of R against |dV| over windows of T trades | 0.416 / 0.439 / 0.495 | A-S30 | ZETA_SINGLE_ORDER_SIZE, ZETA_RETURN_TAIL |
| exponent | `zeta` | **ZETA_SINGLE_ORDER_SIZE** | R(v,1) = A (v/V_best)^zeta <s>, Bouchaud Eq. 11.7 | 0.166 / 0.230 / 0.262 at 600 s; 0.63-0.72 at lag-1 and mechanical | C-T20 | ZETA_WINDOW_IMBALANCE, ZETA_RETURN_TAIL |
| exponent | `zeta` | **ZETA_RETURN_TAIL** | tail exponent of the unconditional return distribution, P(|r|>x) ~ x^-zeta | Hill 2.33-3.83 across k on 60-minute moves; Bouchaud reports ~3 universally | section 478 | ZETA_WINDOW_IMBALANCE, ZETA_SINGLE_ORDER_SIZE |
| exponent | `gamma` | **GAMMA_METAORDER_IMPACT** | concavity of price response in metaorder size, I ~ Q^gamma | NOT IDENTIFIABLE on anonymised data (Bouchaud 12.2) | C-T24 | GAMMA_LMF_SIGN_MEMORY |
| exponent | `gamma` | **GAMMA_LMF_SIGN_MEMORY** | decay of the order-sign autocorrelation, C(l) ~ l^-gamma_LMF | 0.7746 / 0.7892 / 0.2092 (SOL an aggregation artefact) | C-T24 | GAMMA_METAORDER_IMPACT |
| exponent | `delta` | **DELTA_CASCADE_IMPACT** | dP = k Q^delta on cascade episodes | 0.684 / 0.666 / 0.696 | C-T20 | DELTA_QUOTE_DEPTH |
| exponent | `delta` | **DELTA_QUOTE_DEPTH** | depth in bps from the mid, the abscissa of Cartea Eq. (8.1) exp(-kappa*delta) | A-S45 fitted kappa ~ 0.0097/bps over an hour | A-S45 | DELTA_CASCADE_IMPACT |
| exponent | `kappa` | **KAPPA_RESPONSE_T_EXPONENT** | R(dV,T) = R(1) T^kappa F(dV/(V_D T^chi)) -- prefactor exponent in the collapsed scaling form | not reported alone; enters as kappa-chi = 0.25-0.30 | A-S30, C-T21 | KAPPA_UNCONDITIONAL_RESPONSE, KAPPA_FILL_DECAY_RATE |
| exponent | `kappa` | **KAPPA_UNCONDITIONAL_RESPONSE** | d log R / d log T with no scaling collapse | 0.6507 / 0.5782 / 0.5209 | C-T23 | KAPPA_RESPONSE_T_EXPONENT, KAPPA_FILL_DECAY_RATE |
| exponent | `kappa` | **KAPPA_FILL_DECAY_RATE** | rate constant in Cartea Eq. (8.1), P(fill) = exp(-kappa*delta) | 0.0097/bps (A-S45); 0.00956 re-derived (C-T22) | A-S45 | KAPPA_RESPONSE_T_EXPONENT, KAPPA_UNCONDITIONAL_RESPONSE |
| exponent | `chi` | **CHI_VOLUME_SCALE** | exponent of the volume normaliser inside F(dV/(V_D T^chi)) | 0.6498 / 0.6817 / 0.5902 as sd(dV) ~ T^chi | C-T23 | — |
| exponent | `p` | **P_PREDICTOR_CAPTURE_DECAY** | f(h) = R(h)/E|r|(h) ~ h^p for a real predictor | -0.409 / -0.495 / -0.508 | A-S40 | P_CONTEMPORANEOUS_FLOW_RATIO, P_LAGGED_FLOW_RATIO |
| exponent | `p` | **P_CONTEMPORANEOUS_FLOW_RATIO** | same formula, flow and return measured over the SAME window | -0.026 / -0.014 / +0.000 | C-T23 | P_PREDICTOR_CAPTURE_DECAY, P_LAGGED_FLOW_RATIO |
| exponent | `p` | **P_LAGGED_FLOW_RATIO** | same formula, signal from a prior window, response measured forward | +0.215/+0.009 at h<=16; -0.721/-0.785 at h>=256 | C-T23 | P_PREDICTOR_CAPTURE_DECAY, P_CONTEMPORANEOUS_FLOW_RATIO |
| exponent | `alpha` | **ALPHA_METAORDER_SIZE_TAIL** | Pareto tail exponent of the metaorder size distribution | 1.775 / 1.789 as UPPER BOUNDS via LMF; book: equities ~1.5, Bitcoin ~1.10 | C-T24 | — |
| design | `k` | **K_MEAN_ABS_OVER_SIGMA** | k = E|r|/sigma, the mean-absolute-to-sigma ratio of returns | 0.6966 (A-S45 freeze block, from section 467) | A-S45 | K_IMPACT_PREFACTOR |
| design | `k` | **K_IMPACT_PREFACTOR** | prefactor in dP = k Q^delta | not reported separately from delta | C-T6 | K_MEAN_ABS_OVER_SIGMA |
| design | `f` | **F_CAPTURE** | the capture fraction, f = R/E|r| evaluated at the design horizon | the estimand -- NOT preregistered; f_design = 0.010 enters only through power | A prereg section 3 | — |
| design | `c` | **C_SINGLE_LEG_COST** | c = spread + fee, single-leg round trip | 10 bps (prereg line 56); two-leg pairs case is 20 bps (line 45) | A prereg | — |
| design | `h` | **H_HORIZON_DAYS** | design horizon h* = [2c/(k f sigma_d)]^2 | derived per symbol | A prereg | H_LAG_IN_TRADES, H_HORIZON_MINUTES |
| design | `h` | **H_LAG_IN_TRADES** | lag in trades in the lagged capture curve f(h) | grid 1..4096 | C-T23 | H_HORIZON_DAYS, H_HORIZON_MINUTES |
| design | `h` | **H_HORIZON_MINUTES** | holding horizon of the episode outcome imp_H | grid 1..360 | episode work | H_HORIZON_DAYS, H_LAG_IN_TRADES |
| design | `sigma` | **SIGMA_DAILY** | daily volatility entering h* | per symbol at freeze | A prereg | SIGMA_REALISED_30MIN |
| design | `sigma` | **SIGMA_REALISED_30MIN** | realised volatility over the prior 30 minutes | 4.64 vs 3.36 bps, tail vs rest | C episode work | SIGMA_DAILY |
| design | `N_eff` | **N_EFF_EFFECTIVE_BETS** | effective independent observation count | 3.27, not the 8 first assumed | A prereg section 8b | — |

## What is NOT claimed

- That A's k is wrong. It is measured; this only shows what it implies and gives a band to check it against at freeze.
- That the Student-t inversion identifies the true distribution. It is one parametric family; the agreement with an independent order-statistic estimator is the evidence, not the model.
- That the spread finding changes the design. It does not: A pools symbols and the spread is negligible on two of three. It is recorded so the third is not read as the same case.

```verdict
DESIGN_FAMILY_ADDED_AND_A_K_IS_C_TAIL_INDEX
NO_DEFECT_THE_FILE_RESOLVES_ITS_OWN_CONVENTION
A_K_IMPLIES_NU_3_765_AGREEING_WITH_C_HILL_2_33_TO_3_83
K_IS_NOT_A_FREE_CONSTANT_IT_IS_A_FUNCTION_OF_THE_TAIL_INDEX
FREEZE_DATE_K_HAS_A_PREDICTED_BAND_0_6366_TO_0_7351
SPREAD_IS_ONE_TICK_ON_THREE_OF_THREE_MAJORS
COST_IS_FEE_DOMINATED_EXCEPT_ON_THE_LARGE_TICK_SYMBOL
H_IS_USED_WITH_THREE_DIFFERENT_UNITS_ACROSS_LANES
F_IS_NOT_A_COLLISION_VERIFIED_CONSISTENT
```
