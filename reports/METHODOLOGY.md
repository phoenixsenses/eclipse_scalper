# Backtest Methodology

## Statistical Robustness

- Primary score is `net_per_attempt` (NPA), not raw fill-only return.
- Fold structure: `seeds x splits` (default splits can be overridden; use `splits=5` for 60-day retest).
- Robustness gate requires positive edge under both:
1. Core condition (`fee~1.0`, `adverse~1.0`)
2. Stress condition (same fee, highest adverse multiplier)

## Bootstrap Confidence Intervals

- Enable with `--bootstrap-ci`.
- For each pocket, bootstrap resamples fold-level `net_per_attempt`.
- Reported:
1. 95% CI (`bootstrap_ci_low`, `bootstrap_ci_high`)
2. One-sided p-value (`bootstrap_p_value`, H0: mean NPA <= 0)

## Multiple Testing Correction

- Supported methods:
1. `none`
2. `bh` (Benjamini-Hochberg FDR control)
3. `bonferroni` (family-wise error control)
- Select via `--mtc-method`.
- Legacy `--bh-correction` maps to `bh`.
- Significance decision uses corrected q/p-value versus `--alpha`.

## Capacity and Fill Reality

- Capacity filters reject pockets with poor practical throughput:
1. low attempt fill-rate
2. high insufficient-fill-rate
- Cost decomposition is reported as:
1. gross edge
2. fee cost
3. adverse cost
4. scratch cost
5. residual

## Fee/Execution Notes

- Maker/taker costs and scratch slippage are modeled explicitly in validator/backtest args.
- Scratch path should include both taker fee and slippage assumptions.

