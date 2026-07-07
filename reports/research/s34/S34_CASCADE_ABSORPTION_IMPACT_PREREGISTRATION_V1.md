# S34_CASCADE_ABSORPTION_IMPACT_PREREGISTRATION_V1

**Gate:** BATCH-CASCADE-ABSORPTION-IMPACT-PREREGISTRATION-V1
**Status:** PREREGISTERED, NOT EXECUTED. No TEST outcome has been read. No `experiment_registry` row exists yet.
**Date:** 2026-07-07 · **Author:** Sonnet 5

This document is binding. Any deviation (re-tuning, peeking at TEST, changing population/model/controls after this document is written) voids this preregistration and requires a new, versioned one before any TEST access.

---

## 0. Research question

Does the pre-birth, signal-relative, 300-second price-impact-per-signed-notional (aggressive-flow absorption capacity) preceding a LONG-direction signal contain continuous incremental predictive information for the already-accepted primary LONG reversal outcome (the continuous endpoint-return quantity underlying the existing `REVERSAL`/`CONTINUATION`/`CHOP` path classification), controlling for event notional, session, and day_trend_bps?

This is a descriptive/inferential association question. **No entry rule, no threshold, no economic/fee claim, no route or bucket promotion decision is made by this preregistration or by the eventual TEST result alone.**

---

## 1. Window ruling (operator-approved, frozen before this document)

> **OPERATOR RULING — FAM_CASCADE_ABSORPTION_IMPACT PRIMARY WINDOW V1.** The primary predictor window for the first preregistration of `FAM_CASCADE_ABSORPTION_IMPACT` is **W300**, representing the 300 seconds immediately preceding signal birth. W300 was selected before outcome access on mechanistic and data-quality grounds: it provides a defensible balance between short-horizon microstructure sensitivity and longer-horizon temporal dilution; it has complete 324/324 `EXACT_RECONSTRUCTABLE` coverage; it satisfies the same frozen known-at contract as the other candidate windows; and it matches the previously governed W300 CVD lane, improving cross-mechanism comparability while eliminating window choice as an additional researcher degree of freedom. Only W300 may be joined to outcomes in this first experiment. W60, W600, W1800 and W3600 may not be used for outcome-linked diagnostics, alternative fits or verdict modification. They remain available only for future independently preregistered hypotheses. W3600 additionally retains its frozen one-row `SOURCE_GAPPED` exclusion. This ruling applies only to the first preregistration of `FAM_CASCADE_ABSORPTION_IMPACT` and does not establish a universal window rule for other mechanism families.

Two mandatory amendments were applied to the original outcome-blind proposal before approval:

1. **Removed unverified print-count claim.** The proposal's claim that W60 "may contain only 1-2 prints" is not frozen or repeated — it was never verified from an outcome-blind source-distribution audit. Replaced with the narrower, defensible statement: *W60 carries greater sensitivity to short-horizon microstructure noise, bid-ask effects and transient price response than W300.*
2. **No outcome-linked alternative-window diagnostics.** W300 is the only window permitted to be joined to the outcome or included in the scientific model in this first experiment. W60/W600/W1800/W3600 must not be evaluated against TRAIN or TEST outcomes in any form — not as robustness coefficients, diagnostic outcome tables, subgroup results, or alternative model fits. They remain canonical data products only, available for future independently preregistered hypotheses. This prevents the first experiment from becoming an implicit five-window scan.

## 2. Graveyard gate (checked first, per repository discipline)

`match_graveyard()` run against this specification's full spec_text (`question_ids` + `hypothesis_id` + `frozen_population` + `frozen_features` + `frozen_target` + `frozen_thresholds` + `frozen_splits`) against the real, curated 31-fingerprint list in `data/ami/knowledge.sqlite`: **0 hits.**

**Not a graveyard retest of:**
- `S34_ORDERFLOW_LEAD` (graveyarded standalone all-timestamp OFI-quantile momentum) — different causal channel (price-impact-per-notional/absorption capacity, not net order-flow imbalance momentum), different anchor population (event-anchored, not all-timestamp), no threshold/entry rule.
- The closed CVD test (`E-CVD-PRIMARY-LONG-W300-PREREG-001` → `NO_RELIABLE_ASSOCIATION`) — a genuinely different causal channel (how much price moves per unit of aggressive flow, i.e. market-depth/absorption capacity, not how much net flow occurred), a different source table lineage (`ami_absorption_impact_windowed_flow`'s `agg_trades`-derived price-impact reconstruction, not `ami_cvd_windowed_flow`'s net taker-flow notional), and its own independent `family_id`/nullifier.

**`NOT_A_GRAVEYARD_RETEST: CONFIRMED`** by this analysis; operator may independently re-verify before authorizing execution.

## 3. Identity resolution

| # | Element | Resolution | Source |
|---|---|---|---|
| 1 | Primary LONG reversal outcome ID | Reuses `endpoint_return_bps@swing_24h` **verbatim, not redefined** — identical outcome identity to the closed CVD preregistration/test, itself accepted and reused across `E-W4`→`E-W10A`. No new outcome is proposed. | `ami/research/w4_post_event_path_taxonomy.py:classify_path`; `S34_CVD_PRIMARY_LONG_PREREGISTRATION_V1.md` |
| 2 | Exact outcome definition | `classify_path(endpoint_return_bps)`: `REVERSAL` iff `endpoint_return_bps >= +CLASSIFICATION_BAND_BPS`, `CONTINUATION` iff `<= -CLASSIFICATION_BAND_BPS`, else `CHOP`. **Continuous source value = `endpoint_return_bps`** (`ami_lifecycle_path_observations.endpoint_return_bps`, effective/corrected selection), `(last_close - reference_price)/reference_price*1e4` at horizon end, **not direction-flipped** (absolute price-return sign). | `ami/lifecycle/path_metrics.py` |
| 3 | Fee/slippage assumptions | **None** — `endpoint_return_bps` is a pure descriptive path metric, no execution model, no fee, identical to the CVD prereg's own treatment. | `ami/lifecycle/path_metrics.py` |
| 4 | Signal/event universe identity | `ami_signal_lifecycle` (324 total: 220 `LONG`/104 `SHORT`) | `ami_signal_lifecycle` schema |
| 5 | Independent-cycle representative rule | **Identical rule to the closed CVD preregistration, reused not redefined**: representative = the eligible LONG signal with the earliest `signal_birth_ts` within each `independent_cycle_id` (eligible = swing_24h `observation_status='OK'`, decided independent of the outcome's value). | `ami/research/w8_short_expanded_baseline.py:compute_global_cycle_split` (is-identity convention) |
| 6 | TRAIN/TEST split identity/version | Cycle-grouped chronological split, is-identity reuse of `compute_global_cycle_split`'s algorithm — **freshly computed `split_version` for this family** (`SPLITv1:16ea98c239034593`, different wording of the `frozen_splits` description than CVD's own `SPLITv1:0a1b96fd74dd281e` — the two adapters hash literal description text, a documented residual property of `resolve_split_version`, not a defect). The underlying TRAIN/TEST cycle **sets** are nonetheless proven byte-identical to CVD's own (§5), since eligibility depends only on direction + outcome-availability, never on which feature is under test. | `ami/governance/epistemic_gates.py:resolve_split_version`, computed this session |
| 7 | Eligible LONG representative count | 220 LONG signals → 194 eligible (swing_24h `observation_status='OK'`) → **131 representative cycles** → **TRAIN=91/TEST=40** — reproduced independently this session, not copied from the CVD prereg | computed this session, real DB, read-only |
| 8 | Exact absorption/impact feature column | `ami_absorption_impact_windowed_flow.price_response_per_signed_notional` WHERE `window_id='W300'` — **324/324 rows `EXACT_RECONSTRUCTABLE`, 0 `SOURCE_GAPPED` for this specific window** (the family's sole exclusion, 1 signal, applies only to `W3600`); all 220 LONG signals present | `ami_absorption_impact_windowed_flow`/`ami_absorption_impact_window_quality_v1`, queried this session |
| 9 | Feature-availability / known-at contract | `window_start_ts_ms = signal_birth_ts - 300_000`, `window_end_ts_ms = signal_birth_ts`, `feature_available_ts_ms = signal_birth_ts`, `known_at_classification='KNOWN_AT_SAFE'` for all 324 rows (verified 0/324 violations at W300) | `ami_absorption_impact_windowed_flow`, queried this session |
| 10 | Session / day_trend_bps definitions | Session: `ami/chart/level_registry.py:_session_of_hour` — `ASIA[0,7)/EUROPE[7,13)/US[13,21)/OFF[21,24)` UTC. `day_trend_bps`: `ami/research/w6rs_confirmation.py:compute_day_trend_bps` — identical to the CVD prereg's own controls, reused verbatim | both modules, read this session |

## 4. Frozen population

- **Base universe:** `ami_signal_lifecycle`, `direction='LONG'` (220 of 324 total signals).
- **Absorption/impact eligibility:** `window_id='W300'`, `quality_status='EXACT_RECONSTRUCTABLE'` via `ami_absorption_impact_window_quality_v1` (**324/324 — no exclusion needed for this window**). No proxy rows used (no proxy layer exists for this family at all, per the frozen contract/rehearsal/freeze — book-depth proxying was ruled `LOW_FIDELITY_PROXY_ONLY` and never constructed).
- **W3600 exclusion:** the family's sole exclusion (`SIG-e03382b4d82720185dfc870a`, LONG, `CONFIRMED_GAP_OVERLAP`) applies **only to W3600** — not applicable to this W300 population, noted for completeness per Amendment 2's requirement to keep the excluded window's status visible even though it is never joined here.
- **Outcome eligibility:** swing_24h `observation_status='OK'` via effective/corrected path-observation selection — **194 of 220** LONG signals eligible (23 `MISSING_INTERNAL_GAP`, 3 `EXCLUDED_NO_HORIZON_DATA`). A data-availability gate computed independent of the outcome's value.
- **Cycle deduplication:** one representative per `independent_cycle_id` → **131 representative cycles**.
- **No post-outcome eligibility filtering:** confirmed — every exclusion above is either population membership (direction), a data-quality/coverage gate (absorption/impact quality, path-observation maturity), or the cycle-representative rule; none depends on the sign or magnitude of `endpoint_return_bps`.

**TRAIN = 91 cycles, TEST = 40 cycles** (cycle-grouped chronological 70/30, cut by count, no straddling — TRAIN's latest representative signal precedes TEST's earliest). **Both cycle-set hashes (`61486bc6…`/`98174ed3…`) are byte-identical to the closed CVD preregistration's own recorded hashes** — independently reproduced this session, not copied — proving the population truly is unchanged by the choice of feature under test, exactly as the original hypothesis-selection artifact (`0c976e21`) predicted it would be.

## 5. Primary predictor (frozen)

```
price_response_per_signed_notional_w300 =
    mark_price_return_bps([T-300s, T]) / max(|signed_notional_w300| / 1e6, FLOOR_USD_M)
```

- `FLOOR_USD_M = 0.01` (frozen, unchanged from the accepted rehearsal/freeze; never bound on this experiment's TRAIN population either — 0/91 floor-applied rows, checked this session).
- Units: bps of mark-price return per $1,000,000 of net signed aggressive notional.
- Sign: not direction-flipped — positive means price rose over `[T-300s,T]`, matching the outcome's own absolute-sign convention. No transform needed.
- Evidence layer: `EXACT` only (schema `CHECK`-enforced).
- Continuous, no threshold, no binning.
- **Scaling:** none. TRAIN-only distribution (outcome-blind, n=91): min=−93.206, max=0.541, mean=−4.898, median=−2.308, stdev=10.701 — already well-scaled for OLS coefficient interpretation, unlike CVD's raw multi-million-dollar notional (which needed a `/1e6` divisor). No winsorization, log, or clipping applied.

## 6. Controls (reused verbatim, same frozen set as CVD)

| Control | Column | Type | Scaling |
|---|---|---|---|
| `event_notional` | `ami_events.notional` | continuous | ÷ 100,000 |
| `session` | `_session_of_hour(signal_birth_ts)` | categorical, reference=ASIA | see §7 |
| `day_trend_bps` | `compute_day_trend_bps(...)` | continuous | none |

## 7. Zero-variance / rank-deficiency policy (frozen before TEST, per operator requirement)

The prior CVD execution exposed a zero-variance session-control ambiguity. This population (identical LONG/eligible set) carries the **same structural fact**, confirmed outcome-blind this session:

| Level | TRAIN (n=91) | TEST (n=40) |
|---|---|---|
| ASIA | 21 | 12 |
| EUROPE | **0** | **0** |
| US | 62 | 24 |
| OFF | 8 | 4 |

**Frozen policy:**
- **Detection:** after computing TRAIN-only session assignments (outcome-blind), any candidate level among `{ASIA, EUROPE, US, OFF}` with zero TRAIN observations is deterministically **dropped** from the design matrix before fitting — never imputed, never forced.
- **This experiment:** `EUROPE` is dropped (0/91 TRAIN). Retained dummy columns: `session_US`, `session_OFF`. Reference category: `ASIA` (never its own column).
- **Degrees of freedom:** design matrix = intercept + predictor + `event_notional_per_100k` + `day_trend_bps` + `session_US` + `session_OFF` = 6 parameters against TRAIN n=91 / TEST n=40 — comfortably identified. An explicit matrix-rank check (rank must equal column count) is a mandatory pre-fit validation, not assumed from parameter count alone.
- **TRAIN/TEST design-rank mismatch policy:** TEST's design matrix uses only the column set frozen from TRAIN. If TEST alone additionally lacked a level present in TRAIN, that dummy would simply evaluate to all-zero for the affected TEST rows — valid, not rank-deficient (the column's existence is a TRAIN-only decision, applied deterministically forward). This scenario does not arise here (0 TEST `EUROPE` rows either), verified outcome-blind, but the policy is recorded regardless of whether it binds.
- **Any other unanticipated rank deficiency at fit time** → verdict is `PROTOCOL_OR_DATA_QUALITY_INVALIDATED`. No ad hoc post-hoc fix is permitted.
- **Collinearity (VIF):** applies only to the three continuous regressors (predictor, `event_notional`, `day_trend_bps`); if any VIF>10, drop `day_trend_bps` first, then `event_notional`. Session dummies are never dropped via VIF — only via the zero-variance rule above, applied first.

## 8. Model specification

```
endpoint_return_bps ~ price_response_per_signed_notional_w300 + event_notional_per_100k
                       + session_US + session_OFF + day_trend_bps
```

- **Family:** OLS linear regression.
- **Standard errors:** cluster-robust (CR1), clustered by `independent_cycle_id`.
- **Significance:** two-sided p<0.05, 95% CI.
- **Minimum sample:** n≥20.
- **Effect-size relevance floor:** `|coefficient × 10.70108397867223| ≥ 5 bps` — a one-TRAIN-standard-deviation move in the predictor must imply at least 5bps of expected outcome change. This is a standard, outcome-blind, standardized-effect-size convention: since the predictor is already a ratio (not a raw notional like CVD's), a fixed round-dollar reference move (CVD's own "$10M" convention) does not transfer; anchoring the floor to the TRAIN predictor's own dispersion is the direct outcome-blind analogue.
- **Missing values:** listwise deletion, count reported (0 observed in TRAIN this session).

## 9. TRAIN policy

TRAIN may be used only for: data integrity, implementation validation, distribution sanity checks (§5's scaling basis, §7's zero-variance detection), deterministic scaling, rank/collinearity checks, numerical-stability validation. TRAIN may **not** be used for window/threshold/direction/subgroup/outcome/control selection, interaction/nonlinear search, or judging whether the hypothesis "looks promising."

## 10. TEST policy

One TEST authorization, one primary model, one primary verdict. No second holdout pass, no subgroup rescue, no alternate direction/window/threshold/proxy/outcome, no model replacement after TEST. Permitted diagnostics (below) are `NON_PROMOTABLE_DIAGNOSTIC` — they may not alter the primary verdict.

**Permitted diagnostics:**
- Same model with `mfe_bps@swing_24h` as outcome.
- TRAIN-side coefficient sign/magnitude (descriptive only).
- VIF/collinearity diagnostic.
- TRAIN-only predictor distribution diagnostics.

**Forbidden analyses:** threshold scan, quantile bucket sweep, subgroup rescue, session-specific reruns, alternative outcome swap after TEST, interaction search, nonlinear/spline search, pooling exact and proxy absorption-impact evidence, **joining/evaluating/reporting W60/W600/W1800/W3600 against TRAIN or TEST outcomes in any form** (per Amendment 2), short-horizon/alternative-window substitution without a new preregistration, any economic/fee-based route or bucket promotion claim from this result alone.

## 11. Verdict rule

| Disposition | Condition |
|---|---|
| `SUPPORTS_INCREMENTAL_ABSORPTION_IMPACT_ASSOCIATION` | TEST 95% CI excludes 0 AND p<0.05 AND \|coefficient × 10.70108397867223\| ≥ 5bps AND n≥20 AND no invalidation |
| `NO_RELIABLE_INCREMENTAL_ASSOCIATION` | CI includes 0, OR p≥0.05, OR below the relevance floor |
| `UNDERPOWERED_OR_INCONCLUSIVE` | n<20 OR CI half-width > 2× the relevance floor |
| `PROTOCOL_OR_DATA_QUALITY_INVALIDATED` | any required validation fails |

A favorable coefficient sign alone is insufficient — all conditions above are required jointly. **This experiment may create a candidate mechanism result only; it may not directly create or promote a trading route or bucket.**

## 12. Nullifier and gate

| Field | Value |
|---|---|
| `canonical_family_id` | `FAMv1:3e2dfe63f9e271bf` |
| `experiment_id` | `E-CASCADE-ABSORPTION-IMPACT-LONG-W300-PREREG-001` |
| `specification_hash_sha256` | `531b16232a88d5a6c692055bd00fa59bd508b7b69cd7fd45cf8e666772fb6608` |
| `outcome_id` | `endpoint_return_bps@swing_24h` |
| `split_version` | `SPLITv1:16ea98c239034593` |
| Ordered TRAIN cycle-set hash | `61486bc62392eed7b7fc038715f2cd9775e270a568e5c1f728dc2d60417671a5` |
| Ordered TEST cycle-set hash | `98174ed356826b15bd8513584015447b68d18718bb933d75380a4d6b2c4f7b04` |
| `test_nullifier_sha256` | `4e3d1229edc04a946ef29994f1562444fd7c9e77b6ff3ecf3004677f919df7d4` |
| Nullifier prior consumption count | 0 |
| Nullifier consumed by this batch | **false** |
| `gate_receipt_hash` | `6dbe0f59416977fce75b20a13876ff4d54dddae171d1fa8b07613135550e06e4` |
| `registry_result` | `PREREGISTERED_NOT_EXECUTED` |
| Graveyard decision | `CLEAN`, 0 hits |
| Authorization state | `NOT_REQUIRED_CLEAN_GRAVEYARD` |

## 13. Input manifest

| Field | Value |
|---|---|
| `canonical.sqlite` sha256 (at preregistration time) | `a229d4b0a7ed82c0ec8411f767a3cba031414e61e32b42ace3e7f6ef390aaaf7` |
| `canonical.sqlite` schema_version | 13 |
| Migration commit | `8808ada8` |
| Waiver commit | `5ab89f63` |
| Readiness/contract commit | `fc1321f5` |
| Rehearsal commit | `fc43e972` |
| Row-accounting freeze commit | `931cd3dd` |
| Selection artifact commit | `0c976e21` |

## Execution stop conditions

Do not access TEST outcomes, execute the model, consume the nullifier, or promote any route/bucket without new, separate operator instruction. Any amendment to this document (population, model, controls, window, outcome) requires a new versioned file and a new gate cycle before TEST access — this document may not be silently patched.

## Amendment policy

Immutable once committed; any change requires a new versioned file and a new gate cycle before TEST access.

---

## Status

**`PREREGISTERED_NOT_EXECUTED`**
