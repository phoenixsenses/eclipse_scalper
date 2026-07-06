# S34_CVD_PRIMARY_LONG_PREREGISTRATION_V1

**Gate:** G2-CVD-PRIMARY-LONG-PREREGISTRATION-V1
**Status:** PREREGISTERED, NOT EXECUTED. No TEST outcome has been read. No experiment_registry row exists yet.
**Date:** 2026-07-06 · **Author:** Sonnet 5

This document is binding. Any deviation (re-tuning, peeking at TEST, changing population/model/controls after this document is written) voids this preregistration and requires a new, versioned one before any TEST access.

---

## 0. Research question

Does the exact, signal-relative, 300-second net taker-flow notional preceding a LONG-direction signal contain continuous incremental predictive information for the already-accepted primary LONG reversal outcome (the continuous endpoint-return quantity underlying the existing `REVERSAL`/`CONTINUATION`/`CHOP` path classification), controlling for event notional, session, and day_trend_bps?

This is a descriptive/inferential association question. **No entry rule, no threshold, no economic/fee claim, no route or promotion decision is made by this preregistration or by the eventual TEST result alone.**

## 1. Graveyard gate (checked first, per repository discipline)

`match_graveyard()` run against this specification's full spec_text (question_ids + hypothesis_id + frozen_population + frozen_features + frozen_target + frozen_thresholds) against the real, curated 31-fingerprint list in `data/ami/knowledge.sqlite`: **0 hits.** This is not a retest of `S34_ORDERFLOW_LEAD` (graveyarded standalone all-timestamp OFI-quantile momentum, 20-30s horizons, net-of-cost economic claim, dead on all 3 symbols) — this experiment differs on every axis that mattered there: event/signal-anchored population (not all-timestamp), pre-birth bounded window `[T-300s, T]` (no post-anchor flow), outcome = the existing frozen lifecycle path-observation endpoint return (not a fee-net entry claim), no threshold, no entry rule. **NOT_A_GRAVEYARD_RETEST: CONFIRMED** by this analysis; operator may independently re-verify before authorizing execution.

## 2. Identity resolution (do-not-invent discipline)

Every identity element below was located in canonical repository state, not invented. Where no dedicated pre-existing function covered a step exactly, that gap is marked explicitly (not silently papered over).

| # | Element | Resolution | Source |
|---|---|---|---|
| 1 | Primary LONG reversal outcome ID | **No single named `experiment_id` exists for this yet** — this preregistration is the first to freeze one. The underlying *classification* (`REVERSAL`/`CONTINUATION`/`CHOP`) is already accepted and reused verbatim across `E-W4-POST-EVENT-PATH-TAXONOMY-001` → `E-W5A-MORPHOLOGY-GRAMMAR`, `E-W6-COMPRESSION-RS-SESSION`, `E-W6RS-CONFIRMATION`, `E-W6RS-CONFOUND-RESOLUTION`, `E-W7A-STATE-STRUCTURE-AGING-MARKET-CLOCKS`, `E-W10A-MULTI-TF-STRUCTURAL-CONFLICT` (all six say "reused, not redefined"). This preregistration reuses the identical classification's continuous source value (below), not a new outcome. | `ami/research/w4_post_event_path_taxonomy.py:classify_path`; 6 downstream experiments' `frozen_target` text |
| 2 | Exact outcome definition | `classify_path(endpoint_return_bps)`: `REVERSAL` iff `endpoint_return_bps >= +CLASSIFICATION_BAND_BPS`, `CONTINUATION` iff `<= -CLASSIFICATION_BAND_BPS`, else `CHOP`. **Continuous source value = `endpoint_return_bps`** (`ami_lifecycle_path_observations.endpoint_return_bps`, effective/corrected selection), computed as `(last_close - reference_price) / reference_price * 1e4` at the horizon end, **not direction-flipped** (absolute price-return sign; `REVERSAL` for a LONG signal means price moved favorably/up). | `ami/lifecycle/path_metrics.py` lines 281-284 |
| 3 | Fee/slippage assumptions | **None frozen into this outcome** — `endpoint_return_bps` is a pure descriptive path metric, no execution model, no fee. Matches `S34_CVD_NEXT_BATCHES_PLAN_2026-07-06.md`'s explicit "No FEE/economic gate (descriptive/inferential stratification only; any later economic claim = a NEW prereg)." | `ami/lifecycle/path_metrics.py`; plan doc BATCH-CVD-B |
| 4 | Signal/event universe identity | `ami_signal_lifecycle` (324 total: 220 `LONG` / 104 `SHORT`), accessed via `ami.research.feature_gateway.fetch_lifecycle_signals` (exposure-logged, standard gateway path) | `ami_signal_lifecycle` schema; `ami/research/feature_gateway.py` |
| 5 | Independent-cycle representative rule | **No dedicated pre-existing function does this exact dedup.** 220 LONG signals map to only 142 distinct `independent_cycle_id`s (up to 5 signals per cycle — the `LONG_SILENCE` setup can re-trigger within one cycle). This preregistration defines the rule fresh, consistent with the one existing convention that touches this question (`ami.research.w8_short_expanded_baseline.compute_global_cycle_split` orders cycles "by the EARLIEST signal_birth_ts among each cycle's member rows"): **representative = the eligible LONG signal with the earliest `signal_birth_ts` within each `independent_cycle_id`** (eligible = swing_24h `observation_status='OK'`, decided independent of the outcome's value — see §3). Flagged here explicitly as a new-but-minimal, non-arbitrary rule for operator review, not a claimed pre-existing one. | `ami/research/w8_short_expanded_baseline.py:compute_global_cycle_split` (is-identity convention, not is-identity code) |
| 6 | TRAIN/TEST split identity/version | Cycle-grouped chronological split, **is-identity reuse** of `compute_global_cycle_split`'s algorithm (order representative cycles by signal_birth_ts, cut at `TRAIN_FRACTION=0.7` by cycle COUNT, never row count) — applied fresh to this family's own 131-cycle population (this exact split has never been frozen before; it is not a reuse of the 167-cycle W1 split or any other family's split). `split_version = SPLITv1:0a1b96fd74dd281e` (via `ami.governance.epistemic_gates.resolve_split_version`). | `ami/research/w8_short_expanded_baseline.py`; computed this session |
| 7 | Eligible LONG representative count | 220 LONG signals → 194 eligible (swing_24h `observation_status='OK'`, effective/corrected path-observation selection) → **131 representative cycles** (one per `independent_cycle_id`) → **TRAIN=91 / TEST=40** | computed this session, real DB, read/exposure-logged only |
| 8 | Exact CVD feature column | `ami_cvd_windowed_flow.cvd_notional` WHERE `window_id='W300'` — confirmed real, existing column/window (300 seconds), **324/324 rows `EXACT_RECONSTRUCTABLE`, 0 `SOURCE_GAPPED` for this specific window** (the 12 SOURCE_GAPPED rows in the CVD contract belong only to the `BUCKET` window, not `W300`) | `ami_cvd_windowed_flow` / `ami_cvd_window_quality_v1` schema, queried this session |
| 9 | Feature-availability / known-at contract | `window_start_ts_ms = signal_birth_ts - 300_000`, `window_end_ts_ms = signal_birth_ts` (pre-birth, zero post-anchor rows by construction — verified `0/324` rows with `window_end_ts_ms > signal_birth_ts`), `feature_available_ts_ms = signal_birth_ts` for all rows, `known_at_classification='KNOWN_AT_SAFE'` for all 324 rows (verified, not assumed) | `ami_cvd_windowed_flow`, queried this session |
| 10 | Session / day_trend_bps definitions | Session: `ami/chart/level_registry.py:_session_of_hour` — `ASIA[0,7) / EUROPE[7,13) / US[13,21) / OFF[21,24)` UTC. `day_trend_bps`: `ami/research/w6rs_confirmation.py:compute_day_trend_bps` = `(mark_now - day_open)/day_open*1e4`, itself reused from `tools/research_s34_btc_sell_anomaly_audit.py` (is-identity, not reinvented) | both modules, read this session |

**Residual identity risk, disclosed:** item 1 and item 5 above are the two places where "already accepted" required interpretation rather than a literal pre-existing artifact. If the operator has a different specific "primary LONG reversal outcome" in mind (e.g. `mfe_bps` as primary rather than as the secondary check below, or a different horizon), this preregistration must be superseded by a new version before TEST access — it must not be silently patched.

## 3. Frozen population

- **Base universe:** `ami_signal_lifecycle`, `direction='LONG'` (220 of 324 total signals).
- **CVD eligibility:** `window_id='W300'`, `quality_status='EXACT_RECONSTRUCTABLE'` via `ami_cvd_window_quality_v1` (**324/324 — no exclusion needed for this window**). No proxy rows used (`ami_cvd_windowed_flow_proxy` never opened by this preregistration or its future execution code).
- **BUCKET exclusions:** the 104-row `ami_cvd_bucket_exclusions` table is **100% SHORT-direction** (verified: 0 LONG signals excluded) — not applicable to this population, noted for completeness.
- **Outcome eligibility:** swing_24h `observation_status='OK'` via `fetch_effective_path_observations` (corrected-version-preferred, single row per signal×horizon) — **194 of 220** LONG signals eligible (23 `MISSING_INTERNAL_GAP`, 3 `EXCLUDED_NO_HORIZON_DATA`). This is a data-availability gate computed independent of the outcome's value (whether price went up or down), not a post-outcome filter.
- **Cycle deduplication:** one representative per `independent_cycle_id` (§2 item 5) → **131 representative cycles**.
- **No post-outcome eligibility filtering:** confirmed — every exclusion above is either population membership (direction), a data-quality/coverage gate (CVD quality, path-observation maturity), or the cycle-representative rule; none depends on the sign or magnitude of `endpoint_return_bps`.

**TRAIN = 91 cycles, TEST = 40 cycles** (cycle-grouped chronological 70/30, cut by count). TRAIN's latest representative signal_birth_ts (`1781271076544`) precedes TEST's earliest (`1781279802488`) — confirmed no straddling.

**TRAIN cycle-set hash:** `61486bc62392eed7b7fc038715f2cd9775e270a568e5c1f728dc2d60417671a5`
**TEST cycle-set hash:** `98174ed356826b15bd8513584015447b68d18718bb933d75380a4d6b2c4f7b04`

**TEST outcomes have not been read.** Only cycle counts, hashes, and CVD/quality/availability metadata (never `endpoint_return_bps` values) were inspected to construct this table.

## 4. Predictor

**Primary:** `ami_cvd_windowed_flow.cvd_notional` WHERE `window_id='W300'`, **raw, continuous, signal-relative net taker-flow notional** (taker-buy minus taker-sell notional in the 300 seconds immediately preceding `signal_birth_ts`).

- No binning, no sign threshold, no quantile threshold, no optimization.
- **Fixed scaling (numerical stability/interpretability only, TRAIN-distribution-informed, outcome-blind):** divide by `1,000,000` (report coefficients as bps-per-$1,000,000 net notional). TRAIN-only distribution (n=91, outcome never inspected): min=-114.5M, max=+20.2M, mean=-23.0M, median=-16.2M, stdev=23.4M (USD). No winsorization — no evidence of encoding-error outliers found in this range; all values are physically plausible net notional flows for ETHUSDT.
- No sign transformation, no log transformation (raw values include negatives and cross zero, so signed log is not applicable without an arbitrary offset — deliberately not used to avoid a hidden researcher-degree-of-freedom).

## 5. Controls (frozen)

| Control | Canonical column | Encoding | Scaling | Missing policy |
|---|---|---|---|---|
| Event notional | `ami_events.notional` (via `source_event_id`) | continuous | divide by 100,000 (bps-per-$100k) | listwise deletion (report count dropped) |
| Session | `_session_of_hour(signal_birth_ts)` (`ami/chart/level_registry.py`) | categorical, dummy-coded, **reference = ASIA** | n/a | listwise deletion (should not occur — session is always computable) |
| day_trend_bps | `compute_day_trend_bps` (`ami/research/w6rs_confirmation.py`) | continuous, already bps-scaled | none (matches outcome's own bps unit) | listwise deletion (report count dropped; can be `None` if day-open mark price is unavailable) |

**Collinearity policy (TRAIN-only, prespecified):** compute VIF for all predictors+controls on TRAIN. If any VIF > 10, drop the lowest-priority control in this fixed order: `day_trend_bps` first, then `event_notional` (session is never dropped). This check and any resulting drop must be finalized before TEST is touched, and is reported in the results regardless of outcome.

**No control may be added or removed after TEST is seen.**

## 6. Analysis model (frozen)

- **Model family:** Ordinary Least Squares linear regression. `endpoint_return_bps ~ cvd_notional_w300_per_1M + event_notional_per_100k + session_EUROPE + session_US + session_OFF + day_trend_bps`.
- **Coefficient interpretation:** bps change in `endpoint_return_bps` (swing_24h) per 1 unit of each scaled predictor/control, holding others fixed.
- **Standard-error method:** cluster-robust (CR1), clustered by `independent_cycle_id` — this population, unlike the strict one-representative-per-cycle design, still has other signals from the SAME cycle's neighbourhood correlated through shared market conditions even after dedup; cluster-robust SEs are the established convention in this codebase (`w6rs_confound_resolution`, `w7a`, `w8_hold_baseline`) and are used here even though the representative count already equals the cycle count, as the conservative default.
- **Treatment of independent cycles:** one row per representative cycle (§3); clustering variable retained even though currently 1:1 with rows, as a structural/documentation safeguard against any future population change.
- **Predictor/control scaling:** fixed, listed in §4-§5, decided from TRAIN-only distributions before any outcome access.
- **Significance criterion:** two-sided p < 0.05 on the primary CVD coefficient.
- **Confidence interval level:** 95%.
- **Minimum sample requirement:** `MIN_BUCKET_N = 20` (established convention); TEST n=40 clears this.
- **Failure conditions:** TEST n < 20 → `INSUFFICIENT_SAMPLE` (does not occur here, recorded as a rule regardless); any control's VIF > 10 unresolved after the prescribed drop order → `PROTOCOL_INVALIDATION`; any known-at violation discovered at execution time → `PROTOCOL_INVALIDATION`.

**No automated model selection. No interaction search. No nonlinear/spline search. No threshold scan. No subgroup rescue.**

### Permitted secondary checks (non-promotable; must not alter the primary verdict)

1. Same model specification with `mfe_bps` (swing_24h) substituted as the outcome — reported alongside, never replacing, the primary `endpoint_return_bps` result.
2. TRAIN-side coefficient sign/magnitude, reported for descriptive continuity only (not a selection step, not compared to TEST to "choose" anything).
3. VIF/collinearity diagnostic (§5).
4. TRAIN-only predictor-distribution diagnostics (already reported in §4).

## 7. TRAIN/TEST protocol

TRAIN may be used only for: data-integrity validation, predictor-distribution inspection (done, §4), prespecified numerical-stability decisions (scaling units, §4-§5), model implementation validation (e.g. running the regression code on TRAIN to confirm it executes and produces sane output — never to select anything).

TRAIN must not be used to choose a threshold, select subgroups/sessions, select an alternative outcome, choose a transformation based on apparent edge, or change the primary hypothesis. **None of these were done.**

TEST rules: one authorized holdout pass; one frozen primary specification (this document); no reopening after seeing the result; no subgroup rescue; no alternative threshold; no replacement outcome; no proxy substitution; no adding/removing controls after TEST.

## 8. Preregistered verdict rule (frozen before TEST execution)

The primary verdict must be exactly one of:

1. **`EVIDENCE_SUPPORTS_INCREMENTAL_ASSOCIATION`** — TEST 95% CI on the primary CVD coefficient excludes 0 AND two-sided p<0.05 AND the point estimate's magnitude implies at least a 5 bps swing_24h `endpoint_return_bps` change per $10,000,000 net notional shift (`|coefficient × 10| >= 5`, a fixed, outcome-blind relevance floor set now, not tuned to the result) AND n>=20 AND no protocol/quality invalidation.
2. **`NO_RELIABLE_ASSOCIATION`** — TEST CI includes 0, OR p>=0.05, OR the point estimate is below the 5bps/$10M relevance floor even if nominally significant.
3. **`INSUFFICIENT_SAMPLE_OR_INCONCLUSIVE`** — TEST n<20, or the CI half-width exceeds 2× the 5bps/$10M relevance floor (too wide to distinguish a relevant effect from noise either way).
4. **`PROTOCOL_OR_DATA_QUALITY_INVALIDATION`** — any of §9's required validations fails, or the frozen VIF-drop procedure cannot resolve collinearity.

**A positive/significant coefficient alone, without clearing the effect-size floor, is `NO_RELIABLE_ASSOCIATION`, not support.** No trading route, entry rule, or promotion decision is defined by any of these four outcomes — that would require a separate, later, economic preregistration.

## 9. Required validations (to be proven at execution time, before verdict is read)

- Exact and proxy CVD populations were not pooled (structural: only `ami_cvd_windowed_flow`/`W300` opened; `ami_cvd_windowed_flow_proxy` never imported/queried by the execution code).
- No TEST outcome was read during this preregistration (true — only cycle counts/hashes and CVD/quality/availability metadata were inspected; verified no `endpoint_return_bps`/`mfe_bps` value was printed or read at any point in this session).
- No experiment_registry row / experiment_results row was written by this preregistration (verified: 22/323, unchanged before/after).
- No threshold scan was performed (true — no threshold exists in this design).
- No subgroup was selected (true — full eligible LONG population used, no session/route/setup subsetting).
- Known-at violations = 0 (verified: 0/324 W300 rows with `window_end_ts_ms > signal_birth_ts`).
- Cycle representatives are unique (verified: 131 representative cycles = 131 distinct `independent_cycle_id` values, one signal each).
- Split identity is frozen (`SPLITv1:0a1b96fd74dd281e`, this document).
- Outcome identity is pre-existing and unchanged (`classify_path`/`endpoint_return_bps`, `ami/lifecycle/path_metrics.py`, untouched by this batch).
- Canonical `schema_version` remains 12; canonical hash proof in the transition proof.
- Protected runtime/risk/execution delta = 0 (no file under `execution/`, `risk/`, `brain/`, `.env`, `tools/s34_state_machine_live_executor.py` touched).

## 10. Pre-cascade legacy incident — explicit non-interaction statement

`PRE_CASCADE_DIP_RECOVERY_NO_EDGE_LEGACY_BYPASS_RECORDED` (SYSTEM_STATE.md §96/§97, `failure_archive` id=22) is a **separate, already-closed, unrelated finding** (SELL-side pre-cascade dip-recovery timing hypothesis, NO_EDGE, recorded via a different ad-hoc script family). This preregistration:
- does not merge it into the `FAM_CVD_PRIMARY_LONG_REVERSAL` family,
- does not reuse its identity or experiment/family IDs,
- does not reinterpret or touch its `failure_archive` record,
- was not used to shape any decision in this document.

## 11. Nullifier and gate enforcement (first real consumer of the M-0033/M-0034 mechanism)

| Field | Value |
|---|---|
| Canonical family ID | `FAMv1:bec99d8d36f7d6a1` |
| Experiment ID | `E-CVD-PRIMARY-LONG-W300-PREREG-001` |
| Specification hash (spec_text sha256) | `a2fd9e5b08ed2a716ac0c1cae0658740f24b48024d5b7524eb843e4441940b57` |
| Outcome ID | `endpoint_return_bps@swing_24h` (via `ami_lifecycle_path_observations`, effective selection) |
| Split version | `SPLITv1:0a1b96fd74dd281e` |
| Ordered TRAIN cycle-set hash | `61486bc62392eed7b7fc038715f2cd9775e270a568e5c1f728dc2d60417671a5` |
| Ordered TEST cycle-set hash | `98174ed356826b15bd8513584015447b68d18718bb933d75380a4d6b2c4f7b04` |
| TEST-evidence nullifier | `085397f31c199c1d0c1d5ce647af4d1aa311166c63199f92872e089db8e72a7a` |
| Nullifier prior consumption | 0 (confirmed, real `epistemic_test_nullifiers` table) — **NOT consumed by this preregistration**; consumption is deferred to actual TEST execution via `register_experiment_with_gates`, matching item 11 of the required sequence |
| Gate receipt ID / hash | `d46f7e2c6b3621215e13eed136f1e22aec2531549769c96da694a407855f7e5c` (`experiment_gate_receipts`, real `data/ami/knowledge.sqlite`, `registry_result='PREREGISTERED_NOT_EXECUTED'`) |
| Graveyard decision | CLEAN (0 hits against the real 31-fingerprint curated list) |
| Authorization state | N/A — no retry/supersession token needed (clean graveyard, brand-new family) |
| Input manifest root | `data/ami/canonical.sqlite` sha256 `458bc07ca5b436041e59c781a26cf502779d5dc2751a3be8a0c1cddb93e84d49` (unchanged before/after this preregistration except accepted `researcher_exposure_ledger` appends) |
| Code/version commitment | commit `09104298` (current HEAD at preregistration time) |

## 12. Amendment policy

This document, once committed, is **immutable**. Any change to population, predictor, controls, model, split, or verdict rule requires a new file (`..._V2.md` or later) and a new preregistration/gate cycle before any TEST access under the new terms. This V1 document is never edited in place after commit.

## 13. Stop conditions

Execution (a future, separately-authorized batch) must stop and report `PROTOCOL_OR_DATA_QUALITY_INVALIDATION` rather than proceed if, at execution time: the real population/split/hashes recomputed from canonical.sqlite differ from §3/§11 above; any exact/proxy pooling is detected; any known-at violation is detected; the nullifier described above is found already consumed by a different experiment_id without a valid supersession authorization.
