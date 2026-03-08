# Research Fill Toxicity Lane

## Why This Lane

`fill_toxicity_regime` is an execution-quality lane derived from realized trade outcomes.

It is useful when:

- adverse selection rises
- passive fills become systematically toxic
- realized pnl deteriorates under the same fill conditions

## First Deliverable

Tool:

- `python -m tools.fill_toxicity_state`

This first version is intentionally narrow:

- reads trade logs / paper trades
- computes realized toxicity by side
- emits a runtime-ready state/card payload
- produces JSON/MD + `run_summary`

## Expected Use

Short term:

- monitoring annotation
- dashboard caution card
- operator signal for execution quality drift

Not yet:

- direct trade trigger
- automatic runtime mutation without separate validation
