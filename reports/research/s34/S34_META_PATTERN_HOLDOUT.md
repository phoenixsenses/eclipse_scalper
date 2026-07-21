# S34 Meta-Pattern Holdout + MC-Corrected Permutation Null

Generated: `2026-06-30T09:52:22.596292+00:00`

Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`

Cal N=1404, Hold N=602  |  Cal date range: 2026-02-15T18:32:18Z to 2026-06-08T01:05:38Z

Hold date range: 2026-06-08T01:24:48Z -> 2026-06-29T08:28:10Z

## Permutation Null (MC-corrected, cal split)

Candidates tested: 6  |  Permutations: 1000  |  Seed: 42

Real max T3R (best candidate): **6404.6**
Null p95 max T3R: **1261.5**
p-right (real >= null): **0.0**

**MC-corrected verdict: PASS_MC_CORRECTED (p-right=0.0, best=k5_CLEAN_NORMAL)**

## Per-Candidate Results

| Candidate | In-sample T3R | Cal N | Cal T3R | Cal median | Cal win | Cal maxL | Hold N | Hold T3R | Hold median | Hold win | Hold maxL | Holdout verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| k5_CLEAN_NORMAL | 7876.5 | 140 | 6404.6 | 36.0 | 0.757 | -199.8 | 84 | -4563.9 | -23.9 | 0.381 | -412.4 | **HOLD_NEGATIVE** |
| k5_DANGER_REVERSE | 7453.9 | 226 | 2092.1 | -6.8 | 0.465 | -685.9 | 42 | -1481.4 | -29.4 | 0.238 | -167.3 | **HOLD_NEGATIVE** |
| k8_DANGER_REVERSE | 6551.6 | 392 | -672.2 | -9.1 | 0.444 | -728.1 | 80 | -325.8 | -16.7 | 0.388 | -260.8 | **HOLD_NEGATIVE** |
| k5_CLEAN_k20_DANGER_NORMAL | 3143.4 | 61 | 2896.0 | 43.7 | 0.77 | -179.2 | 27 | None | -23.9 | 0.37 | -364.0 | **HOLD_SMALL_N(27)** |
| danger_count_0_NORMAL | 2417.9 | 66 | 2854.2 | 35.5 | 0.773 | -167.7 | 40 | -2122.7 | -65.2 | 0.425 | -405.4 | **HOLD_NEGATIVE** |
| k5_CLEAN_k8_CLEAN_NORMAL | None | 85 | 3792.1 | 35.5 | 0.776 | -167.7 | 49 | -3173.7 | -87.9 | 0.388 | -412.4 | **HOLD_NEGATIVE** |

## Interpretation

- **MC-corrected permutation null**: because ~20 patterns were scanned in-sample, the null
  tracks max T3R across all candidates per shuffle. The corrected p-right tests whether the
  BEST candidate beats the null 95th percentile under the MC-corrected threshold.
  If p-right > 0.05 -> the entire family is an artifact (consistent with 0 PASS verdict).

- **Holdout T3R**: independent OOS check. Labels computed from cal neighbors only (no leakage).
  HOLD_POSITIVE required for any live/shadow promotion consideration.

- **Max loss -685 bps** on k5=DANGER REVERSE is a hard veto on live promotion regardless of T3R.
  Any promotion requires a TP/SL sweep showing tail-budget compatibility.

All results: RESEARCH_ONLY. Permutation-null is the definitive discipline test.
