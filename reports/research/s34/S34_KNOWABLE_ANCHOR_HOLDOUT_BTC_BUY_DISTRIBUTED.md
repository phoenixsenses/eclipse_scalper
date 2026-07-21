# S34 Knowable-Anchor Holdout

Generated: `2026-06-28T11:38:01.146213+00:00`

RESEARCH_ONLY. This is the single holdout pass for the frozen knowable-anchor configuration selected on calibration.

## Frozen Config

- primary_config_id: `NONE`
- config_sha256: `NONE`
- holdout_bucket_ids_sha256: `15bf8e8c66a57134582b91fe639583ea3b206110bf260845a2256230e32a1203`

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
