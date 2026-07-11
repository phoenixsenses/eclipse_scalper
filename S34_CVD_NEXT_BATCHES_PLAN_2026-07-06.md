# S34 CVD — POST-MIGRATION NEXT BATCHES PLAN (2026-07-06)

**Status: PLAN ONLY. Nothing below is implemented. Prepared for Sonnet implementation
after operator relay. Each batch requires its own operator approval to start.**

Context anchor (do not re-derive): schema **12** is live on `data/ami/canonical.sqlite`
(M-0031, `CVD_REPAIR_REHEARSAL_CANONICAL_MIGRATION_APPLIED`). Canonical CVD tables:
`ami_agg_trades_repaired` (40,934), `ami_cvd_repair_batch_ledger` (8),
`ami_cvd_windowed_flow` (1,840 exact), `ami_cvd_windowed_flow_proxy` (1,840),
`ami_cvd_bucket_exclusions` (104), `ami_cvd_window_quality_v1` (1,840;
EXACT_RECONSTRUCTABLE=1,828 / SOURCE_GAPPED=12) + 2 effective views.
Ground truth **852/852**; frozen regression command unchanged. Contract phases
D1–D5 are ALL complete. Detail: SYSTEM_STATE.md §93, MIGRATION_LOG.md M-0031.

---

## BATCH-CVD-A — Research access enablement (gateway extension, no research)

**Goal:** the canonical CVD exact layer becomes reachable ONLY through
`ami/research/feature_gateway.py`, quality-gated, with the proxy layer
structurally unreachable. No outcome read, no experiment.

**Scope (all additive):**
1. `KNOWN_FEATURE_TABLES` += `ami_cvd_windowed_flow`, `ami_cvd_window_quality_v1`,
   `ami_cvd_bucket_exclusions`. These require **dedicated fetch functions**
   (lifecycle-table precedent) — `fetch_chart_feature()` must explicitly reject them.
2. New `fetch_cvd_windowed_flow(conn, window_id=None)`:
   - INNER JOIN `ami_cvd_window_quality_v1_effective` on (signal_id, window_id),
     returning **only** `quality_status='EXACT_RECONSTRUCTABLE'` rows (1,828).
     No parameter may relax this to include SOURCE_GAPPED (fail-closed, not opt-in).
   - Pins `feature_definition_version='s34-cvd-windowed-taker-flow-v1-birth-truncated'`.
   - Reuses `ami.cvd.windowed_taker_flow.assert_not_pooled` (is-identity import).
   - Exposure logging preserved (`_record_exposure`, existing pattern).
3. New `fetch_cvd_bucket_exclusions(conn)` (audit/accounting reads only).
4. **`ami_cvd_windowed_flow_proxy` is NOT added to the allowlist.** Proxy stays
   outside the gateway entirely in this batch (descriptive-only law). A static
   test locks this (table name absent from `KNOWN_FEATURE_TABLES`).

**Tests (new file `tests/test_ami_research_feature_gateway_cvd.py`, ~8-10):**
eligible-row count = 1,828 on real DB (read-only); SOURCE_GAPPED rows never
returned; proxy table rejected as unknown; `fetch_chart_feature` rejects the 3
new tables; version pin enforced; exposure row written; pooling guard fires on
a synthetic mixed population; no outcome table opened (static guard reusing
`FORBIDDEN_TABLES` discipline from `ami/cvd/cvd_rehearsal.py`).

**DoD:** frozen regression green at new ground truth (852 + new tests, count
recorded); canonical.sqlite hash unchanged except `researcher_exposure_ledger`
append rows (known by-design exception); protected delta ZERO.

---

## BATCH-CVD-B — Preregistration ONLY: `E-CVD-WINDOWED-FLOW-001` (no computation)

**Goal:** a frozen preregistration document. NO code that reads outcomes runs in
this batch. Output = one prereg MD in `reports/research/s34/` + stop.

**⚠️ GRAVEYARD GATE (mandatory first section of the prereg):**
`S34_ORDERFLOW_LEAD.md` (graveyarded) tested **standalone, all-timestamp
OFI-quantile momentum entries** at 20–30s horizons with a net-of-cost economic
claim — dead on all 3 symbols, must never be re-tested. This experiment differs
on every axis and the prereg must state it explicitly for operator sign-off:
event-anchored population (324 canonical cascade signals only), **pre-birth**
bounded windows `[T−W, T]` (no post-anchor flow), outcomes = the EXISTING frozen
`ami_lifecycle_path_observations` path-v2 rows at 30m–24h horizons (reused, not
recomputed), stratification question (no entry rule, no cost/net claim, no
short-horizon momentum continuation claim). **Operator must answer
`NOT_A_GRAVEYARD_RETEST: CONFIRMED` before BATCH-CVD-C may run.**

**Frozen spec to write into the prereg:**
- Population: the 1,828 EXACT_RECONSTRUCTABLE rows via `fetch_cvd_windowed_flow`
  (BATCH-CVD-A is a dependency), 167 cycles, cycle-grouped split TRAIN=116/TEST=51
  (verbatim `w8_short_expanded_baseline` machinery, is-identity), MIN_BUCKET_N=20.
- Feature: `normalized_cvd` (primary, already bounded [−1,1]); bucket rule =
  sign OR TRAIN-median split — **pick ONE in the prereg, no sweep**
  (recommendation: sign — parameter-free, not fitted at all).
- Outcome: `mfe_bps` / `mae_bps` per horizon from effective path-v2 rows
  (`fetch_path_observations`, version pinned).
- **Primary cells (Holm family declared upfront, keep small):**
  recommendation = 2 windows (`W600`, `BUCKET`) × 2 metrics × 1 horizon
  (`swing_24h`) = **4 primary cells**; all other window/horizon combinations
  DESCRIPTIVE-ONLY (reported, never counted as confirmatory). Operator may
  swap the window pair — decision point, recorded in the prereg before any run.
- Stats: W8 discipline verbatim — cluster bootstrap risk/median difference +
  two-sided label permutation + Holm across the declared primary cells only;
  `INSUFFICIENT_SAMPLE` honesty; no threshold re-selection after TEST is seen.
- Direction handling: LONG (220) and SHORT (104) analyzed **separately**
  (never pooled — a signed flow feature has opposite priors by direction);
  SHORT likely INSUFFICIENT at cell level → say so upfront.
- No FEE/economic gate (descriptive/inferential stratification only; any later
  economic claim = a NEW prereg).

**DoD:** prereg MD exists, registered nowhere in SQL yet (experiment_registry
row is created by CVD-C at run time, W8 precedent); NO_UPDATE_REQUIRED on test
ground truth; stop at WAIT_FOR_OPERATOR_APPROVAL.

---

## BATCH-CVD-C — Run `E-CVD-WINDOWED-FLOW-001` (only after B sign-off)

**Scope:** new `ami/research/cvd_windowed_flow_001.py` following the W8-004
module pattern exactly (is-identity reuse: split machinery, experiment_ledger
immutable writers, Holm/bootstrap/permutation helpers from
`w8_hold_baseline`/`w7a`); `freeze_and_record()` single entry; idempotent rerun;
supersedes nothing. Population/cells/stats exactly as the signed prereg — any
deviation is a hard stop, not an adjustment.

**Tests:** ~10-12 (module math on synthetic data, known-at guard [feature rows
already carry `feature_available_ts_ms = signal_birth_ts` — assert no row with
later availability enters], real-data smoke + idempotency + prior-experiments-
untouched, insufficient-sample branches).

**DoD:** experiment_registry +1 / experiment_results +k; protected tables
unchanged; regression green at new ground truth; results reported with
verdict vocabulary (`ANSWERED_*` / `INSUFFICIENT_SAMPLE` / regime labels);
findings → Knowledge Object discipline; if null → graveyard entry, no rescue
sweeps.

---

## BATCH-HK — Housekeeping (small, can run before or parallel to CVD-A)

1. **Canonicalize the paired regression runner:** promote the rehearsal's
   `run_regression.sh` (≤2 files/call, paired subprocesses) to
   `tools/run_ami_regression.sh` with a parametrized `--basetemp`. Motivation:
   this session proved a single mega-invocation produces 13 false cross-file
   failures — the guardrail needs to live in `tools/`, not in a disposable folder.
2. **Frozen-source retention lock:** `data/ami/cvd_rehearsal_disposable_20260705/
   cvd_rehearsal_disposable.sqlite` is now the applied migration's frozen source
   package (M-0031 idempotency depends on it). Record its sha256 as an M-0031
   addendum in MIGRATION_LOG.md; folder must NOT be deleted. (Replay/scan
   sqlite side-files in the same folder MAY be pruned — operator decision.)
3. **Whitepaper living-document duty (working contract):** the v12 canonical CVD
   schema is an AMI architecture change → DR-XXXX in `docs/ami/AMI_DECISION_RECORDS/`
   + `AMI_CHANGELOG.md` entry + **PATCH-XXXX in whitepaper Appendix H**
   (operator preference 2026-07-02). Content: 6 tables + 2 views, exact/proxy
   physical separation law, window-level-quality-is-field-level statement.
4. **Operator-gated (not for Sonnet to decide):** git commit of the branch
   (large untracked set); C: drive old-scratchpad cleanup (142GB, operator
   wanted to inspect first).

---

## Explicit operator decision points (answer before/at relay)

| # | Decision | Recommendation |
|---|---|---|
| 1 | Graveyard gate for CVD-B/C | Confirm distinction as written above |
| 2 | Primary cells for prereg | W600+BUCKET × mfe/mae × swing_24h (4 cells) |
| 3 | Bucket rule | sign(normalized_cvd) — parameter-free |
| 4 | Proxy gateway exposure | NO (keep outside gateway indefinitely) |
| 5 | Disposable-folder side-file pruning | Keep main sqlite, prune replay_*.sqlite ok |
| 6 | Batch order | HK → CVD-A → CVD-B (stop) → CVD-C |

## Guardrails that bind every batch above (unchanged)

Protected paths untouched (`execution/`, `risk/`, `brain/`, live executor,
`.env`); no parallel Python processes; pytest ≤2 files/call + scratchpad
basetemp + `-p no:cacheprovider`; canonical DB writes only via approved
migration/experiment writers (immutable, fail-closed); `microstructure.db`
strictly `mode=ro`; no lookahead; thresholds TRAIN-only; graveyard never
re-tested; MD updates only SYSTEM_STATE + PROGRESS_LEDGER (+TEST_STATUS) at
batch end.
