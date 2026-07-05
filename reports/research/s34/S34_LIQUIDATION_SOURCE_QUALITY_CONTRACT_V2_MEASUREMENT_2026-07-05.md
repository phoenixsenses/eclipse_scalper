# LIQUIDATION SOURCE-QUALITY CONTRACT V2 — FROZEN, FIELD-LEVEL MEASUREMENT (2026-07-05)

**Contract version:** `liquidation-source-quality-contract-v2` (operator-approved, authoritative).
**Code:** `ami/geometry/liquidation_source_quality_contract_v2.py` (schema/classification/append-only backfill) — reused by `tools/research_s34_contract_v2_measurement.py` (deterministic; regenerates this report).
**Tests:** `tests/test_ami_geometry_liquidation_source_quality_contract_v2.py` (15/15 ✓, covers all 12 operator-required proofs).
**Mode:** READ-ONLY measurement (no canonical write, no outcome read).

## Contract semantics (as implemented, field-level)

- `ALL_MARKET_TRANSITION_TS_MS = 1780767832123` (2026-06-06 17:43:52.123 UTC) — measured as the first liquidations row following the 40.14-day blackout (2026-04-27 14:27:26.345 → 2026-06-06 17:43:52.123); 171 distinct symbols in the following hour confirms the `!forceOrder@arr` all-market transition (vs. 2-3 symbols throughout Feb–Apr).
- `CRITICAL_GAP_MS = 300_000` — the original collector's own frozen stream-specific-reconnect threshold (recovered from git stash `07e1a1f9`), not fit to this population.
- Each of the 8 RUNNING_CLUSTER fields gets its OWN required source window (Goal D):
  - `running_notional` / `running_liq_count` / `max_single_notional` / `running_single_liq_dominance` / `running_rate` / `elapsed_since_first_sec`: `[bucket_start_ts_ms, anchor_ts_ms]`.
  - `running_accel`: `[anchor_ts_ms − 2×ACCEL_WIN_SEC×1000, anchor_ts_ms]` (its own frozen two-window definition; ACCEL_WIN_SEC=30s → 60s span), independent of `bucket_start`.
  - `inter_cluster_gap_sec`: `[previous_accepted_anchor_ts_ms (or earliest available liquidation ts if this is the first anchor ever), anchor_ts_ms]` — never inherits the current bucket's own completeness.
- Row-level status (when one filter value is needed) = **worst** of the 8 field statuses (GAPPED > UNRESOLVED > COMPLETE) — never an independent assessment.
- Quality assessments are stored append-only in `ami_birth_truncated_geometry_field_quality_v2` keyed by `(feature_id, field_name, coverage_assessment_version)`; a differing re-assessment under an EXISTING version fails closed (`ImmutableFieldQualityConflict`); a genuine re-assessment must use a new `coverage_assessment_version`.

## Measured result (real `data/microstructure.db` + real `data/ami/canonical.sqlite`, mode=ro / disposable copy only)

**Per-field status counts (220 LONG signals):**

| Field | COMPLETE | GAPPED | UNRESOLVED |
|---|---|---|---|
| running_notional / running_liq_count / max_single_notional / running_single_liq_dominance / running_rate / elapsed_since_first_sec (identical bucket window) | 94 | 0 | 126 |
| running_accel | 94 | 0 | 126 |
| **inter_cluster_gap_sec** | **87** | **6** | **127** |

`inter_cluster_gap_sec` is strictly the limiting field: its own (previous-anchor → anchor) window is longer/different from the bucket window and picks up 6 resolved-gap overlaps the other 7 fields never see, plus 1 additional signal where its window straddles into the unresolved zone while the bucket window itself is fully post-transition.

**Row-level worst-case status (what a consumer filtering `SOURCE_COMPLETE_ONLY` actually gets):**

| Status | N |
|---|---|
| SOURCE_COMPLETE | **87** |
| SOURCE_GAPPED | 6 |
| SOURCE_COVERAGE_UNRESOLVED | 127 |

(Total 220. Limiting-field breakdown: 87 rows all-8-fields-COMPLETE; 7 rows limited solely by `inter_cluster_gap_sec` UNRESOLVED; 126 rows limited by ALL 8 fields simultaneously UNRESOLVED — i.e. pre-transition signals fail every field at once, as expected.)

**SOURCE_COMPLETE_ONLY population (87 signals):**

- source events: 87, independent cycles: **51**, TRAIN=35 / **TEST=16**
- monthly distribution: 2026-06 = 87 (100% — every COMPLETE signal is post-transition, as the contract requires by construction)
- setup composition: 100% `LONG_SILENCE`
- **MIN_BUCKET_N=20 verdict: `INSUFFICIENT_SAMPLE`** (TEST=16 < 20)

This is measured, not forced — it differs from the operator's approximate expectation (93 signals / 54 cycles / 37-17) by exactly the `inter_cluster_gap_sec`-driven demotions (94→87 row-level, cycles 54→51-ish once the field-level worst-case rule is applied). The qualitative conclusion is unchanged either way: **TEST cycle N stays below MIN_BUCKET_N=20 under contract-v2**, same as it did under every previously-examined candidate except the invalid ones (METHOD_B absence-of-evidence, cross-stream health).

## Verdicts

- **Source-quality contract:** `LIQUIDATION_SOURCE_QUALITY_CONTRACT_V2_FROZEN`
- **Research readiness:** `GEOMETRY_INFERENTIAL_RESEARCH_BLOCKED_BY_SOURCE_QUALITY` (SOURCE_COMPLETE_ONLY TEST cycle N = 16 < MIN_BUCKET_N = 20)
- Canonical storage of correctly field-flagged geometry rows is **not** blocked by this — only inferential-research eligibility is.

## Integrity

`data/ami/canonical.sqlite` sha256/mtime unchanged throughout measurement (verified before/after); `data/microstructure.db` opened mode=ro only; no outcome column read; no experiment write; protected delta ZERO.
