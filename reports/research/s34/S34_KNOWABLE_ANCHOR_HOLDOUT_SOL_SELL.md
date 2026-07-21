# S34 Knowable-Anchor Holdout

Generated: `2026-06-28T11:38:29.726242+00:00`

RESEARCH_ONLY. This is the single holdout pass for the frozen knowable-anchor configuration selected on calibration.

## Frozen Config

- primary_config_id: `NONE`
- config_sha256: `NONE`
- holdout_bucket_ids_sha256: `41458e5a1c8253b5b614853a13035403605cb2df164dae114cf21d9342cc8f0b`

## Holdout Result

- verdict: `BLOCKED`
- n: `0`
- median_net_bps: `None`
- mean_net_bps: `None`
- win_rate: `None`
- top3_winner_removed_cum_bps: `0.0`
- no_fill_n: `0`
- no_fill_rate: `None`
- no_fill_counterfactual_median_bps: `None`

## Shortlist Holdout Diagnostics

| Config | Verdict | N | Median | Mean | WR | Top3W Removed | No-fill % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |

## Verdict Taxonomy

- `BLOCKED`: insufficient holdout N or no calibration-selected config.
- `RESEARCH_ONLY`: H0 not rejected on the frozen holdout.
- `PAPER_CANDIDATE`: holdout mean/median net positive and top-3-winner-removed cumulative positive. This authorizes only a fresh paper pre-registration from zero.
