# Honest Ranking Update — Capacity-Aware, Non-Degenerate Scores

Date: 2026-02-20
Branch: feat/reliability-gate-automation

## Problem

Ranking was degenerate:
- `score` was clamped to `0` when `robust_core` or `robust_stress` failed the old
  pass-rate gate, so all failing pockets were indistinguishable.
- Per-fill metrics (`filled_avg_net`, `filled_p90_net`) measure edge conditional on
  filling, ignoring how rarely fills occur — an optimistic view of alpha.
- No capacity metrics exposed in JSON or markdown output.

## Changes

### `tools/validate_passive_pocket_forward.py`

**Additive fields in every `per_combo` row** (backward-compatible; no renames):

| Field | Source | Determinism note |
|---|---|---|
| `val_attempts` | `attempt_level_metrics.n_attempts` | Derived from sim output |
| `val_filled` | same as `filled_n` | — |
| `net_per_attempt` | `attempt_level_metrics.net_per_attempt` | Seed-deterministic |
| `attempts_per_min` | `val_attempts / (val_rows * bucket_sec / 60)` | No wall-clock; uses input params |

`attempts_per_min` uses only `val_rows` (integer count of validation buckets) and
`bucket_sec` (function parameter). No `datetime.now()` or `time.time()` dependency.
Satisfies DAT-03 (deterministic per seed+input).

**`_aggregate_per_split`** gains `net_per_attempt_mean` and `attempts_per_min_mean`.

**Markdown per_combo table** now includes `net_per_attempt` (scientific) and
`attempts_per_min`; `filled_avg_net`/`filled_p90_net` formatted to 8 decimal places.

**Markdown per_split table** gains `net_per_attempt_mean` and `attempts_per_min_mean`.

### `tools/rank_passive_pockets_forward.py`

**`_aggregate_eval`** now returns three additional fields (additive, no renames):
- `median_net_per_attempt`: median NPA across all `per_combo` rows
- `attempt_fill_rate_median`: median attempt fill rate
- `attempts_per_min_median`: median attempts per minute

**New CLI flags** (safe defaults, backward-compatible):
- `--min-attempt-fill-rate` (default `0.10`): skip pockets whose core-eval
  `attempt_fill_rate_median` is below this.
- `--max-insufficient-fill-rate` (default `0.50`): skip pockets whose core-eval
  `insufficient_fill_rate` exceeds this. (Pre-existing guard, now gated.)

**Score rewrite** — primary score is now NPA-based, not `_fee_score`-based:
```
fee_one  = value in fee_grid closest to 1.0
adv_one  = value in adverse_grid closest to 1.0
fee_max  = max(fee_grid)
adv_max  = max(adverse_grid)

core_npa      = get_eval(fee_one, adv_one).median_net_per_attempt
stress_npa    = get_eval(fee_one, adv_max).median_net_per_attempt
npa_bps       = core_npa * 10_000
stab_bps      = core_eval.stability_std * 10_000
base_score    = max(0, npa_bps) / (1 + max(0, stab_bps))
capacity_pen  = 1 + 0.5 * insufficient_fill_rate
score         = base_score / capacity_pen  if (core_npa > 0 and stress_npa > 0)  else 0
```

**Robustness gates** now use `median_net_per_attempt > 0` (not pass_rate thresholds).
A pocket must have positive NPA at both core and adverse conditions to get a non-zero score.

**Score raw fields** (always populated, even when `score == 0`):
- `score_raw_core`: `median_filled_avg_net` at `(fee_min, adv_one)`
- `score_raw_stress`: `median_filled_avg_net` at `(fee_max, adv_max)`
- `score_raw_min`: `min(median_filled_avg_net)` across all eval grid points

**Sort key** changed to `(score, score_raw_core)` descending so that zero-gated
pockets are still ordered by raw edge rather than all lumped together.

**Markdown table** now includes: `robust_core`, `robust_stress`, `net_per_attempt`,
`attempt_fill_rate`, `attempts_per_min`, `score_raw_core`, `score_raw_stress`,
`score_raw_min`. Floats use `{:.6e}` or `{:+.8f}` for full precision.

## Invariant Analysis

| Invariant | Impact |
|---|---|
| EXE-01 through EXE-05 | Not affected — research-only change |
| DAT-01 (no lookahead) | Not affected — `attempts_per_min` derived from count/param |
| DAT-02 (timing alignment) | Not affected |
| DAT-03 (determinism) | Preserved — `attempts_per_min` uses no wall-clock |
| DAT-04 (cost units) | Not affected |
| DAT-05 (JSONL schema stability) | Additive only — no renames |
| VAL-01 (true forward splits) | Not affected |
| VAL-02 (ranking reproducibility) | Preserved — same seed → same NPA → same rank |
| VAL-03 (candidate parsing integrity) | Not affected |
| SAF-01 / SAF-02 | Not affected |

## Test Coverage

| Test | File | What it covers |
|---|---|---|
| `test_net_per_attempt_deterministic` | `tests/test_validate_pocket_forward_api.py` | DAT-03: same seed → same `net_per_attempt` |
| `test_ranker_filters_low_capacity_pocket` | `tests/test_rank_passive_pockets_forward.py` | `attempt_fill_rate < threshold` → excluded from ranking |
| `test_score_raw_fields_present_in_output` | `tests/test_rank_passive_pockets_forward.py` | `score_raw_*` populated; `score_raw_core > 0`, `score_raw_stress < 0` match fee sweep |
| `test_ranking_fee_priority_and_stability` (updated) | `tests/test_rank_passive_pockets_forward.py` | ETHUSDT still ranks above BTCUSDT under NPA-based scoring |

All 77 tests pass: `pytest -q` → 77 passed.

## Repro Commands

```powershell
# Validate a single pocket (shows new per_combo and per_split fields):
python -m tools.validate_passive_pocket_forward `
  --db data/microstructure.db --symbol ETHUSDT `
  --lookback-min 1440 --bucket-sec 1 --horizon-sec 120 `
  --rule intensity_spike_imbalance_cont --side auto `
  --min-imbalance 0.5 --min-trade-intensity 2500 --max-spread 0.0005 `
  --splits 4 --seeds 11,22,33,44,55 --min-n 30 --maker-fee-bps 0.25

# Rank with capacity filter and new score fields:
python -m tools.rank_passive_pockets_forward `
  --candidates-md reports/FILTER_SWEEP_PASSIVE_REALISTIC_ETH.md,reports/FILTER_SWEEP_PASSIVE_REALISTIC_BTC.md `
  --db data/microstructure.db --lookback-min 1440 --bucket-sec 1 `
  --rule intensity_spike_imbalance_cont --side auto `
  --splits 4 --seeds 11,22,33,44,55 --min-n 50 `
  --maker-fee-bps-grid 0.5,1.0,1.5 --passive-adverse-mult-grid 0.8,1.0,1.2 `
  --min-attempt-fill-rate 0.10 --max-insufficient-fill-rate 0.50 `
  --out-md reports/PASSIVE_POCKET_RANKING.md --out-json reports/PASSIVE_POCKET_RANKING.json

# Run tests:
pytest tests/test_rank_passive_pockets_forward.py tests/test_validate_pocket_forward_api.py -v
pytest -q
```
