# S34 Knowable-Anchor Alpha Family Recheck

Generated: 2026-06-28

Scope: offline/read-only recheck of the non-ETH-BUY S34 candidates using the same knowable-anchor protocol. This is not a live recommendation and does not modify executor parameters.

## Method

- Entry anchor: first real-time-knowable crossing of running liquidation notional.
- Features at entry: running cluster state only, passed through the Feature Availability Contract gate.
- Outcomes: fixed forward horizons 30s/60s/120s from the knowable anchor.
- Fill model: executable bid/ask when available, 3.05 bps per side.
- Diagnostic: mark-price counterfactual over all anchors, to separate directional signal from fill coverage.
- Selection rule: calibration only; requires filled N >= 20, positive mean, positive median, and positive top-3-winner-removed cumulative. Holdout is touched only if a primary config is selected.

## Results

| Family | Target Dir | Anchors | Primary | Verdict | Best target-direction read |
| --- | --- | ---: | --- | --- | --- |
| SOL BUY 100K/200K | LONG | 191 | NONE | BLOCKED | Best large-N rows are negative: 100K 30s all exec median -6.1, mark CF median -5.2 |
| BTC BUY 1M distributed | LONG | 53 | NONE | BLOCKED | 1M 30s accelerating exec median -5.5, mark CF median -5.3 |
| ETH SELL 500K/1M | SHORT | 342 | NONE | BLOCKED | 1M 30s all exec median -4.8, mark CF median -6.4; positive decel pocket N=2 only |
| SOL SELL 100K/200K | SHORT | 172 | NONE | BLOCKED | 100K 60s decel positive but N=5 and top3-removed negative; 200K large-N rows negative |

## Candidate Details

### SOL BUY

Target alpha was BUY liquidation -> LONG continuation.

| X | H | Accel | Filled N | No-fill | Exec Median | Exec Mean | Top3W Removed | Mark CF Median | Mark CF Mean |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100K | 30 | all | 55 | 29.5% | -6.1 | -3.8 | -280.2 | -5.2 | -2.2 |
| 100K | 30 | accelerating | 53 | 29.3% | -6.1 | -3.7 | -269.2 | -6.1 | -2.5 |
| 200K | 30 | all | 40 | 28.6% | -6.7 | -6.5 | -295.0 | -7.3 | -5.0 |

Read: no knowable-anchor LONG continuation survived. This weakens the older SOL BUY TP/SL sweep as an alpha claim until retested with route exits under the same anchor contract.

### BTC BUY Distributed

Target alpha was BTC BUY liquidation -> LONG continuation with running single-liq dominance <= 50%.

| X | H | Accel | Filled N | No-fill | Exec Median | Exec Mean | Top3W Removed | Mark CF Median | Mark CF Mean |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1M | 30 | accelerating | 19 | 36.7% | -5.5 | -8.6 | -188.3 | -5.3 | -8.8 |
| 1M | 30 | all | 23 | 37.8% | -6.6 | -8.5 | -220.0 | -6.3 | -9.0 |
| 1M | 120 | all | 23 | 37.8% | -7.1 | -9.5 | -285.2 | -12.3 | -6.7 |

Read: the distributed filter did not rescue fixed-horizon knowable-anchor continuation.

### ETH SELL

Target alpha was SELL liquidation -> SHORT continuation.

| X | H | Accel | Filled N | No-fill | Exec Median | Exec Mean | Top3W Removed | Mark CF Median | Mark CF Mean |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1M | 30 | all | 37 | 49.3% | -4.8 | -3.6 | -228.5 | -6.4 | -3.8 |
| 1M | 30 | accelerating | 35 | 50.0% | -5.7 | -4.1 | -237.5 | -6.7 | -4.1 |
| 500K | 60 | all | 60 | 62.5% | -7.1 | -6.1 | -468.9 | -6.9 | -5.5 |
| 1M | 30 | decelerating | 2 | 33.3% | +4.5 | +4.5 | +9.0 | +0.5 | +1.9 |

Read: the positive ETH SELL decelerating row is N=2, not a candidate. Large-N rows are negative.

### SOL SELL

Target alpha was SELL liquidation -> SHORT continuation.

| X | H | Accel | Filled N | No-fill | Exec Median | Exec Mean | Top3W Removed | Mark CF Median | Mark CF Mean |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100K | 60 | decelerating | 5 | 16.7% | +10.2 | +10.1 | -18.7 | +7.0 | +10.8 |
| 100K | 120 | decelerating | 5 | 16.7% | +1.2 | +8.0 | -15.5 | -0.5 | +6.1 |
| 200K | 60 | all | 31 | 27.9% | -4.6 | -6.2 | -245.6 | -3.0 | -4.9 |
| 200K | 60 | accelerating | 29 | 29.3% | -4.6 | -6.6 | -245.6 | -3.0 | -5.1 |

Read: there is a small SOL SELL decelerating hint, but it fails N and robustness. The 200K larger-N rows are negative.

## Verdict

No non-ETH-BUY S34 family produced a clean knowable-anchor `PAPER_CANDIDATE` under this fixed-horizon protocol.

This does not prove every old TP/SL route sweep was false, because TP/SL/BE path exits are not identical to fixed-horizon returns. It does mean the claimed alpha is not established until those route exits are rerun with the same knowable-anchor reconstruction and feature availability gate.

Next research step: rerun the exact frozen TP/SL/BE route definitions for SOL BUY, BTC BUY distributed, ETH SELL, and SOL SELL using the same reconstructed knowable anchors. If those also fail, the prior positive route sweeps were likely terminal-cluster/anchor artifacts or exit-parameter artifacts.
