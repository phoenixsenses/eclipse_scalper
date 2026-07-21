# S34 No-Lookahead Filter Sweep - 2026-06-07 / 06-11 / 06-14 / 06-15

Scope: read-only research sweep over existing `microstructure.db`. No production runner/config changes.

Model caveat: simplified mark-price replay with flat 8 bps fee. This does not include real bid/ask fill, adverse selection, max-open timing, cooldown, or exact live runner cursor behavior. These are candidate filters for future forward testing, not proof.

## Top Eligible Candidates

Eligibility for this table: at least 12 routed signals and at least 3 active days.

| rank | threshold | TP | entry delay | filter | n | days | mean net | median net | cum net | WR | exits | day counts |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 200K | 120 | 60s | BTC pre-15m >= 0 | 17 | 4 | 65.32 | 112.12 | 1110.50 | 70.6% | BE 3 / TP 9 / TIME 4 / SL 1 | 6 / 1 / 2 / 8 |
| 2 | 200K | 120 | 0s | BTC pre-15m >= 0 | 17 | 4 | 59.19 | 93.56 | 1006.15 | 64.7% | BE 4 / TP 8 / SL 2 / TIME 3 | 6 / 1 / 2 / 8 |
| 3 | 200K | 120 | 120s | BTC pre-15m >= 0 | 17 | 4 | 56.88 | 70.91 | 966.88 | 64.7% | SL 2 / TIME 5 / BE 2 / TP 8 | 6 / 1 / 2 / 8 |
| 4 | 100K | 120 | 0s | 2m ETH confirmation >= 8 bps | 14 | 3 | 56.49 | 72.50 | 790.83 | 57.1% | BE 6 / TP 6 / TIME 2 | 5 / 4 / 0 / 5 |
| 5 | 100K | 120 | 60s | 2m ETH confirmation >= 8 bps | 14 | 3 | 51.95 | 81.84 | 727.26 | 57.1% | BE 3 / TP 7 / SL 2 / TIME 2 | 5 / 4 / 0 / 5 |
| 6 | 200K | 80 | 60s | BTC pre-15m >= 0 | 17 | 4 | 51.80 | 73.19 | 880.67 | 76.5% | BE 2 / TP 12 / TIME 2 / SL 1 | 6 / 1 / 2 / 8 |
| 7 | 200K | 80 | 120s | BTC pre-15m >= 0 | 17 | 4 | 50.35 | 72.43 | 855.91 | 76.5% | SL 2 / TP 13 / BE 1 / TIME 1 | 6 / 1 / 2 / 8 |
| 8 | 50K | 120 | 0s | 2m ETH confirmation >= 8 bps | 21 | 3 | 49.11 | 51.44 | 1031.25 | 52.4% | TIME 3 / BE 9 / TP 8 / SL 1 | 7 / 5 / 0 / 9 |
| 9 | 200K | 120 | 60s | ETH pre-5m >= 0 | 20 | 4 | 48.17 | 61.61 | 963.38 | 60.0% | BE 3 / TP 9 / SL 4 / TIME 4 | 7 / 2 / 2 / 9 |
| 10 | 50K | 120 | 60s | 2m ETH confirmation >= 8 bps | 21 | 3 | 46.44 | 51.56 | 975.25 | 52.4% | TIME 3 / BE 6 / TP 9 / SL 3 | 7 / 5 / 0 / 9 |

Day counts are ordered: 2026-06-07 / 2026-06-11 / 2026-06-14 / 2026-06-15.

## Read

- The cleanest no-lookahead candidate is `200K threshold + BTC pre-15m >= 0`.
- Entry delay of 60s was best in this simplified replay, but 0s and 120s were also positive. That means the filter is not only a one-timestamp artifact.
- `TP80` improves win rate but lowers mean/cum versus `TP120` in the top BTC-filtered family.
- ETH confirmation-delay filters also look promising, but they intentionally wait 1-2 minutes after signal; these must be modeled as delayed entries, not retroactive entries.
- `50K` can be rescued by a confirmation-delay filter, but it becomes less pure: many weak early signals are skipped.

## Candidate For Forward Test, Not Current Pre-Reg

Do not mutate the current pre-registered S34 validation mid-sample.

Candidate research variant:

`ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60`

Definition:

- ETH BUY liquidation cluster >= 200K in 1m bucket
- existing day-so-far regime filter passes
- BTC mark return over the 15 minutes before signal >= 0 bps
- enter after 60 seconds, not immediately
- TP 120 bps / SL 40 bps / BE 30 bps
- use the same real bid/ask fill and cost attribution as the live paper runner

Lower-variance sibling:

`ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP80_SL40_BE30_DELAY60`

This had lower mean but higher win rate in the simplified replay.

## Discipline

This is research output only. It should become a separate exploratory paper variant or a future pre-registration. It should not be mixed into the existing `50K_TP120` validation sample.

Artifact:

- JSON: `reports/research/s34/S34_NO_LOOKAHEAD_FILTER_SWEEP_2026-06-07_15.json`
