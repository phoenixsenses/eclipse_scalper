# RESEARCH LIQUIDATION REVERSAL E2E PLAN

## Goal

Build a single research chain for `high_liq_reversal_regime`:

1. measure raw coverage
2. generate a rule-specific candidate surface
3. rank that surface under baseline passive execution
4. rank the same surface under event-driven passive wait profile
5. produce a single summary artifact with the decision

## Why

The repository already proved:

- raw liquidation reversal coverage exists
- standard passive pocket search does not yield tradeable pockets
- extra passive wait plus scratch still does not open tradeable coverage

That means the next phase must be run as a controlled end-to-end chain, not as isolated one-off scripts.

## Stages

### Stage 1: Coverage

Tool:
- `tools/liquidation_rule_coverage.py`

Question:
- does the rule fire often enough across real lookbacks?

Success condition:
- stable, non-zero `rule_fire_count`

### Stage 2: Candidate Surface

Tool:
- `tools/generate_liq_reversal_candidates.py`

Question:
- what pocket search space is appropriate for this rule?

Success condition:
- deterministic candidate markdown + JSON

### Stage 3: Baseline Rank

Tool:
- `tools/rank_passive_pockets_forward.py`

Profile:
- `baseline`

Question:
- does the rule survive current passive execution assumptions?

Success condition:
- at least one ranked pocket with non-zero count

### Stage 4: Event-Driven Rank

Tool:
- `tools/rank_passive_pockets_forward.py`

Profile:
- `anti_adverse_v5`

Question:
- does extra passive wait plus conservative scratch improve tradeable coverage?

Success condition:
- ranked pocket count improves versus baseline

### Stage 5: Decision

Tool:
- `tools/run_liq_reversal_e2e.py`

Question:
- should the next research move stay inside passive execution, or switch execution style?

Current decision rule:
- if baseline and `v5` both produce zero ranked pockets, next step is `change_execution_style`

## Current State

Current evidence says:

- coverage exists
- passive pocket count is zero
- event-driven passive wait profile is also zero

So the active next step is:

- `change_execution_style`

## Practical Next Build

After this end-to-end chain, the next implementation should be one of:

1. semi-passive / maker-then-scratch evaluator
2. liquidation-event alerting layer without direct execution
3. longer-horizon liquidation reversal evaluator
