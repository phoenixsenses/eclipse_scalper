# FAM_BOOK_SPREAD_DYNAMICS_PRIMARY_DEFINITION_V1

**Family:** `FAM_BOOK_SPREAD_DYNAMICS`
**Status:** Operator ruling, recorded before any outcome access. Resolves the `SPREAD_EXPANSION_COMPRESSION_DEFINITION_AMBIGUOUS` readiness verdict (commit `f115b9c1`).
**Date:** 2026-07-07 · **Recorded by:** Sonnet 5 (operator ruling verbatim)

---

## Operator ruling

The following definition is approved for the **first child hypothesis** of `FAM_BOOK_SPREAD_DYNAMICS`, before any TRAIN/TEST outcome access.

### Primary window — W300

W300 represents the change between the exact spread state at signal birth and the exact spread state 300 seconds before signal birth. Selected **outcome-blind**, on:

- a defensible short-horizon liquidity-withdrawal / liquidity-normalization mechanism timescale;
- identical source coverage across all audited candidate windows (no candidate receives a coverage-derived advantage — confirmed window-invariant in the readiness audit: 97 independent cycles at every window);
- reduced sensitivity to very-short-horizon quote noise vs. W60;
- lower temporal-dilution risk than W600/W1800/W3600;
- consistency with the already-governed W300 CVD and absorption mechanism lanes (simpler cross-mechanism comparison);
- reduced researcher degrees of freedom;
- exact live computability from the accepted L1 source.

**Scope:** this ruling applies **only** to the first child hypothesis of `FAM_BOOK_SPREAD_DYNAMICS`. It does **not** establish W300 as a universal window for other families or for later independently-preregistered spread hypotheses.

### Primary feature form — signed additive difference in spread bps

```
mid_price(t)  = (best_ask(t) + best_bid(t)) / 2
spread_bps(t) = 10,000 × (best_ask(t) - best_bid(t)) / mid_price(t)
spread_change_bps_w300 = spread_bps(t0) - spread_bps(t0 - 300s)
```

**Sign interpretation:** positive = spread **expansion** (widening) over W300; negative = spread **compression** (narrowing); zero = no change between the two governed reference points. **Units:** basis points of spread change.

**Why the additive bps difference** (operator rationale): `spread_bps` is already price-normalized; the result is directly interpretable as widening/narrowing; it avoids division by zero/near-zero baseline spreads; it avoids multiplicative instability from ratios/log-ratios; it avoids sample-dependent scaling and the extra policy choices a z-score requires; a spread level alone does not represent expansion/compression; and it minimizes researcher degrees of freedom.

### Reference-point contract

- **Current reference:** target `t0` (canonical signal-birth timestamp); selected quote = latest valid **exact** quote at or before `t0`.
- **Historical reference:** target `t0 − 300s` exactly; selected quote = latest valid **exact** quote at or before `t0 − 300s`.
- Uses the **accepted deterministic quote-selection contract from commit `f115b9c1`**: receipt-time known-at safety; never a quote received after its target; duplicate `ts_ms` tie-broken by **`id DESC`**; never a post-target nearest-neighbour; no interpolation; no future backfill; no forward-fill beyond the accepted staleness tolerance; never cross signal birth while selecting/repairing.
- Uses the exact maximum-staleness (5 min), locked-book, crossed-book, zero/negative-price, source-gap, symbol, venue and market-segment rules frozen in the accepted readiness artifacts. No accepted source-quality policy is silently reinterpreted.

### Alternative-form and alternative-window prohibition (first child)

- **W300 is the only permitted window.** W60/W600/W1800/W3600 must not be joined to outcomes, evaluated against TRAIN/TEST outcomes, built as alternative rehearsal predictors, placed in robustness tables or alternative model fits, or allowed to influence the verdict.
- **Additive `spread_change_bps_w300` is the only permitted primary feature.** Prohibited alternative forms: spread level as a predictor, ratio, percentage ratio, log-ratio, z-score (rolling/expanding), quantile, bin, threshold, sign-only indicator, clipped/winsorized/smoothed difference, path max/min/mean/median/volatility, persistence/reversion score, nonlinear transform, interaction.
- The two endpoint spread levels may be retained **only as deterministic provenance fields** to reproduce the approved difference — never presented or tested as competing scientific predictors.

## Amendment policy

Immutable once committed. Any change to the window, feature form, endpoint contract, or scope requires a new versioned ruling before any outcome access — this document is not silently patched.
