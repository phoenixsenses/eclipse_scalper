# Alpha Candidate Availability Audit V1

BATCH-ALPHA-CANDIDATE-AVAILABILITY-AUDIT-V1.

This gate does not search for alpha, does not optimize candidate performance,
does not invent new strategies, and is not a migration. It codifies, as an
enforceable contract, an invariant that `DISPOSABLE_ALPHA_CANDIDATE_PROMOTION_
REHEARSAL_V1` (2026-07-09) found violated by every candidate it rehearsed.

Tool: `tools/alpha_candidate_availability_audit.py`. Tests:
`tests/test_alpha_candidate_availability_audit.py`. Auto-generated per-run
family record: `ALPHA_CANDIDATE_AVAILABILITY_AUDIT_V1_FAMILY_RECORD.md` +
`ALPHA_CANDIDATE_AVAILABILITY_AUDIT_V1.json` (this directory, regenerate with
`python tools/alpha_candidate_availability_audit.py`).

Companion module: `tools/s34_feature_availability.py` (per-*feature*
`FeatureClass`/`knowable_at_ts` checks, already used by
`tools/s34_shadow_paper_runner.py` and
`tools/research_s34_knowable_anchor_continuation.py`). This gate adds a
candidate-*level* disposition layer on top — it does not replace or duplicate
that module's per-feature registry.

## 1. Summary of the disposable rehearsal finding

`DISPOSABLE_ALPHA_CANDIDATE_PROMOTION_REHEARSAL_V1` rehearsed five
`FAM_ETH_BUY_LIQ_CONTINUATION` candidates (ETH BUY liquidation-cluster
continuation, delay-0 entry, TP40/SL50/BE20) in an isolated, read-only,
disposable worktree. It found:

- `event_ts_ms == cluster_start_ts_ms` for all 182 universe events.
- Predicate features (`cluster_notional`, `cluster_liq_count`) finalize only
  at `cluster_end_ts_ms` — median **173 s** after `cluster_start_ts_ms`
  (range 4–298 s).
- A delay-0 entry conditioned on `cluster_notional >= 500K` therefore uses
  information that does not exist yet at the entry timestamp.
- Honest no-lookahead control (enter at the timestamp the running cluster
  notional actually first crosses 500K — median delay 81–159 s): mark median
  **≈ −8 bps**, real-fill median **≈ −5.6 bps** (C1) / **≈ +3.4 bps ≈ zero**
  (C3, +count22).
- Predicate nullifiers (day_trend<0, shuffled day labels, 400–500K fake
  split) matched the lookahead candidates (~+33 bps) → the conditioning
  itself was spurious.
- Timing nullifiers (randomized entry offset, random in-cluster entry)
  matched the honest control (~−8 bps) → the apparent edge was the
  intra-cluster price move, harvested retroactively.

**Current ETH BUY continuation candidate family is not eligible for the
canonical alpha gate.**

## 2. The invariant

```
entry_decision_ts_ms >= feature_available_ts_ms
```

A candidate's entry decision may only condition on a feature value once that
value is actually knowable. This must hold for every feature in a
candidate's predicate, not just the union.

## 3. Definitions

| term | definition |
|---|---|
| `event_ts_ms` | The timestamp a candidate's feature-store row is stamped with. For `FAM_ETH_BUY_LIQ_CONTINUATION` this equals `cluster_start_ts_ms` — **not** the moment the predicate became true. |
| `cluster_start_ts_ms` | Timestamp of the first liquidation print belonging to a cluster. |
| `cluster_end_ts_ms` | Timestamp of the last liquidation print belonging to a cluster — the point at which cluster-aggregate features finalize. |
| `threshold_cross_ts_ms` | Timestamp of the first print at which a *running* cumulative predicate (e.g. cumulative notional, running count) first satisfies its threshold. Computed by `compute_threshold_cross_ts` / `compute_count_cross_ts`. |
| `feature_available_ts_ms` | The earliest timestamp at which a given predicate feature is actually knowable, per the availability rules below. |
| `entry_decision_ts_ms` | The timestamp at which the candidate's entry logic evaluates its predicate and decides to enter. |
| `entry_fill_ts_ms` | The timestamp of the actual (or simulated real-fill) execution following the entry decision; always `>= entry_decision_ts_ms`. |
| `outcome_start_ts_ms` | The timestamp from which post-entry outcome tracking (MFE/MAE/exit) begins; `== entry_fill_ts_ms` in this repo's simulators. |
| `outcome_end_ts_ms` | The timestamp the trade closes (TP/SL/BE/TIME). |
| `predicate_feature` | A feature value the entry logic conditions on. Must satisfy the invariant. |
| `outcome_feature` | A feature describing the *result* of a trade already entered (net_bps, exit_reason, MFE/MAE). Never a predicate. |
| `post_entry_diagnostic_feature` | A feature computed from data strictly after `entry_decision_ts_ms` (tempo/outcome states, `time_to_MFE`, `first_5m_net_bps`, ...). Explanation-only; never a predicate, by rule 5, independent of any timestamp bookkeeping. |

## 4. Availability rules

1. **Entry predicate invariant**: `entry_decision_ts_ms >= feature_available_ts_ms`.
2. **Completed-cluster aggregates** (final `cluster_notional`, final
   `cluster_liq_count`, final `cluster_duration_s`, final
   `max_single_liq_share`, `frontloaded_ratio`/`backloaded_ratio`, geometry/
   shape fields, or any feature requiring full cluster membership):
   `feature_available_ts_ms = cluster_end_ts_ms`.
3. **Running threshold-crossing predicates** (e.g.
   `running_cluster_notional >= 500K`): `feature_available_ts_ms =
   threshold_cross_ts_ms` — the first timestamp the running cumulative value
   actually crosses the threshold, not the cluster's start or end.
4. **Pre-event-context predicates** (`day_trend`, `prior15_buy_liq_notional`,
   `pre_event_mark_velocity`, computed strictly from data before the
   event/entry): `feature_available_ts_ms <= entry_decision_ts_ms` — legal by
   construction as long as the underlying window is strictly historical.
5. **Post-entry outcome/tempo features cannot be entry predicates**, ever —
   `time_to_MFE`, `time_to_MAE`, `first_Nm_net_bps`, `time_to_TP/SL/BE`,
   `outcome_tempo_state`, etc. Diagnostics only.
6. If feature availability cannot be proven: `BLOCKED_BY_AVAILABILITY_UNKNOWN`.
7. If `entry_decision_ts_ms < feature_available_ts_ms`: `REJECT_ENTRY_PREDICATE_LOOKAHEAD`.
8. If `event_ts_ms == cluster_start_ts_ms` but the predicate requires final
   `cluster_notional` or final `cluster_liq_count` (or any completed-cluster
   aggregate): `REJECT_ENTRY_PREDICATE_LOOKAHEAD` — this is the exact
   structural failure the rehearsal found, called out explicitly so it is
   caught even before any timestamp arithmetic is attempted.
9. A candidate that uses threshold-cross (or otherwise passes the
   availability audit) but shows weak or no performance is **not** promoted
   by this gate — the correct disposition is
   `REJECT_NO_EDGE_AFTER_AVAILABILITY_CORRECTION` or
   `HOLD_FOR_NEW_MECHANISM_CLAIM`, never `PROMOTE`. This gate never emits a
   promotion verdict; it only certifies or rejects availability.

## 5. Disposition table

| disposition | meaning |
|---|---|
| `PASS_AVAILABILITY_AUDIT` | Every predicate feature is knowable at `entry_decision_ts_ms`. |
| `BLOCKED_BY_AVAILABILITY_UNKNOWN` | `feature_available_ts_ms` cannot be determined; audit cannot certify either way. |
| `REJECT_ENTRY_PREDICATE_LOOKAHEAD` | `entry_decision_ts_ms < feature_available_ts_ms` (directly, or via rule 5/8's structural shortcuts). |
| `REJECT_NO_EDGE_AFTER_AVAILABILITY_CORRECTION` | Availability corrected (e.g. re-anchored to threshold-cross) but the resulting edge is not there. |
| `HOLD_FOR_NEW_MECHANISM_CLAIM` | Availability-clean reformulation is plausible but requires a fresh, explicit mechanism hypothesis before testing. |
| `HOLD_FOR_FORWARD_PREREGISTRATION` | Availability-clean and mechanism stated; needs forward (not retrospective) evidence before further evaluation. |
| `OBSERVATION_ONLY` | Availability-clean but explicitly not a promotion candidate (e.g. SELL-side gate rules, or decisively negative honest result kept only for reference). |
| `NOT_AN_ALPHA_CANDIDATE` | Does not meet the structural definition of a tradeable candidate at all. |

This gate never emits `PROMOTE_*` — see rule 9.

## 6. Current family audit record — `FAM_ETH_BUY_LIQ_CONTINUATION`

Candidates: `CAND_ETH_BUY_CONT_500K_DAYTREND_D0_TP40_SL50_BE20`,
`CAND_ETH_BUY_CONT_1M_DAYTREND_D0_TP40_SL50_BE20`,
`CAND_ETH_BUY_CONT_500K_GEOM_COUNT22_D0_TP40_SL50_BE20`,
`CAND_ETH_BUY_CONT_500K_CASCADE_P15_109K_D0_TP40_SL50_BE20`,
`CAND_ETH_BUY_CONT_500K_DAYTREND_GEOM_CASCADE_D0_TP40_SL50_BE20`.

- `family_disposition` = `REJECT_ENTRY_PREDICATE_LOOKAHEAD` (all 5/5, per rule 8)
- `promotion_disposition` = `REJECT_SPURIOUS_OR_DATA_PATH_SUSPECT` (from the rehearsal gate's own vocabulary)
- reason:
  - `event_ts_ms == cluster_start_ts_ms` for every universe event
  - predicate uses completed-cluster aggregate(s) / running-threshold facts
    not available until `cluster_end_ts_ms` / `threshold_cross_ts_ms`
  - honest threshold-cross control loses the edge
    (`~-8 bps` mark median vs `~+33 bps` lookahead-entry median)
- canonical alpha gate eligibility: **false**
- permitted future work: **only** a new preregistered mechanism claim using
  `feature_available_ts_ms` / `threshold_cross_ts_ms` as the entry anchor —
  see `tools/research_s34_knowable_anchor_continuation.py` for this repo's
  established knowable-anchor pattern (same failure class was found and
  fixed once before, per the June-2026 clean-recheck: "+1805 bps paper was
  lookahead; clean test is the alpha bar").
- banned future work: reuse of cluster-start (delay-0) entry combined with a
  completed-cluster predicate, for this or any structurally identical
  family.

Machine-readable version: `ALPHA_CANDIDATE_AVAILABILITY_AUDIT_V1.json`
(generated by `python tools/alpha_candidate_availability_audit.py`).

## 7. Future candidate gate checklist

Before any candidate is rehearsed or promoted, answer all of the following:

1. What is the predicate, exactly (every feature name it touches)?
2. When is *every* predicate feature actually known — not "typically", the
   real timestamp?
3. What is `feature_available_ts_ms` for the predicate as a whole (the max
   over all its features' individual availability)?
4. Is `entry_decision_ts_ms >= feature_available_ts_ms`?
5. Are post-entry diagnostics (tempo/outcome states, `time_to_*`,
   `first_Nm_net_bps`, MFE/MAE) excluded from the predicate entirely?
6. If the predicate uses a *final* cluster aggregate, is it only evaluated
   at or after `cluster_end_ts_ms`?
7. If the predicate is threshold-based, is entry anchored to
   `threshold_cross_ts_ms` (not `cluster_start_ts_ms`, not `cluster_end_ts_ms`)?

If any answer is "no" or "unknown", the candidate does not proceed to
performance testing — file it as `BLOCKED_BY_AVAILABILITY_UNKNOWN` or
`REJECT_ENTRY_PREDICATE_LOOKAHEAD` first.

## 8. Next allowed research direction

- `HONEST_THRESHOLD_CROSS_REFORMULATION` — re-anchor entry to
  `threshold_cross_ts_ms`, forward-preregister, then test. (Rehearsal's own
  honest control already shows the ETH BUY 500K/count22 reformulation
  nets ≈ 0 at real fill — a new mechanism claim is required, not a rerun of
  the same predicate.)
- `PRE_THRESHOLD_EARLY_WARNING` — only using features that are knowable
  *before* `threshold_cross_ts_ms` (pre-event context per rule 4). Must be
  forward-preregistered before any performance evaluation.
- Both directions require a preregistered mechanism claim
  (`HOLD_FOR_NEW_MECHANISM_CLAIM` → `HOLD_FOR_FORWARD_PREREGISTRATION`)
  before any backtest-style performance testing begins.

## 9. Explicit forbidden pattern

**Cluster-start entry using a completed-cluster predicate.** I.e. setting
`entry_decision_ts_ms = event_ts_ms = cluster_start_ts_ms` while the
predicate depends on `cluster_notional`, `cluster_liq_count`, or any other
field that only finalizes at `cluster_end_ts_ms`. This is the exact pattern
that produced the false `~+33 bps` edge in
`DISPOSABLE_ALPHA_CANDIDATE_PROMOTION_REHEARSAL_V1` and is banned outright by
rule 8, independent of any other analysis.

## 10. Scope note

This gate does not touch `tools/s34_state_machine_live_executor.py`, `.env`,
`execution/`, `risk/`, `brain/`, leverage/sizing, range-read/ASOF consumer
migrations, the archive/catalog, or `shadow_runner`/dashboard runtime state.
No live/paper/scheduler activation. No canonical alpha acceptance. This gate
only codifies and enforces the availability invariant.
