## Track
- [x] research
- [ ] runtime
- [ ] shared

## Summary
This PR narrows the `passive_then_taker` promotion claim for ETH 60s pockets.
Research still supports `passive_then_taker = experimental_on`, but only for a tighter ETH 60s subfamily, not the full family.
The updated docs separate real execution flips from fillability-only improvements and make the rollout boundary explicit.
Expected outcome: PR reviewers can approve a narrower, defensible research claim without implying broad ETH or BTC readiness.

## Ownership
- Owner: `research`
- Branch: `codex/research/pocket-promotion-checklist`
- Affected paths:
  - `docs/PASSIVE_THEN_TAKER_DECISION.md`
  - `docs/PASSIVE_THEN_TAKER_ETH60_FAMILY_MAP.md`
  - `docs/PASSIVE_THEN_TAKER_PR_SUMMARY.md`

## Validation
- [ ] targeted tests ran
- [ ] smoke or equivalent ran
- [x] docs updated if behavior changed

Commands:
```powershell
# no runtime or test commands ran
# docs-only research update based on existing report artifacts
```

## Artifacts
- Output files:
  - `docs/PASSIVE_THEN_TAKER_ETH60_FAMILY_MAP.md`
  - `docs/PASSIVE_THEN_TAKER_PR_SUMMARY.md`
- Report paths:
  - `reports/ETH_POCKET_B_7D_BASELINE_SPLIT2.json`
  - `reports/ETH_POCKET_B_7D_PASSIVE_THEN_TAKER.json`
  - `reports/ETH_POCKET_C_7D_PASSIVE_THEN_TAKER.json`
  - `reports/ETH_POCKET_SOFT_7D_BASELINE.json`
  - `reports/ETH_POCKET_SOFT_7D_PASSIVE_THEN_TAKER.json`
  - `reports/ETH_POCKET_MID_7D_BASELINE.json`
  - `reports/ETH_POCKET_MID_7D_PASSIVE_THEN_TAKER.json`
  - `reports/ETH_POCKET_TIGHTMID_7D_BASELINE.json`
  - `reports/ETH_POCKET_TIGHTMID_7D_PASSIVE_THEN_TAKER.json`
- Registry or state touched:
  - none

## Risk Check
- Runtime impact:
  - none, docs-only
- Config/env impact:
  - none
- Rollback path:
  - revert the docs changes or restore the previous broader wording

## Merge Checklist
- [ ] no unrelated files included
- [x] shared paths reviewed if touched
- [x] branch matches ownership lane
- [x] report paths do not overwrite active runs

## Reviewer Focus
- Confirm the promotion language stays narrow: `ETHUSDT`, `micro_edge_v3_passive_alpha`, `h=60`, tighter pockets only.
- Confirm `Soft` is treated as `observe_only` and `Mid` is not presented as promotable.
- Confirm the PR does not imply BTC validation, family-wide ETH validation, or default-execution readiness.
