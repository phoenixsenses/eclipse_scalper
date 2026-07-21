# S34 Guardrail V4 Audit

Generated at: `2026-06-24T12:09:26.133378+00:00`

Guardrail: `guardrail_v4_50k_warning_lt200k`

Definition: `rule=50K/TP120 AND model_guardrail=warning AND cluster_notional < 200K`

Scope: shadow-only. No runner/config/live reject was changed by this audit.

## Result

| Scenario | N | Cum | Mean | Median | WR % | Extra |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_all_closed | 74 | 1167.85 | 15.78 | 20.75 | 51.35 |  |
| would_block_closed | 15 | -397.96 | -26.53 | -46.3 | 13.33 | signals 175 |
| kept_after_block | 59 | 1565.82 | 26.54 | 49.35 | 61.02 | delta 397.97 |

## Blocked By Rule

| Rule | N | Cum | Mean | Median | WR % |
| --- | --- | --- | --- | --- | --- |
| ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 15 | -397.96 | -26.53 | -46.3 | 13.33 |

## Blocked Examples

| Trade | Exit | Net | Cluster | Count | Signal UTC |
| --- | --- | --- | --- | --- | --- |
| P150 | SL | -55.59 | 76855.88 | 3 | 2026-06-15T13:32:04.231000+00:00 |
| P056 | SL | -53.45 | 146467.59 | 5 | 2026-06-11T13:45:22.120000+00:00 |
| P418 | SL | -53.37 | 90925.93 | 5 | 2026-06-22T16:10:22.356000+00:00 |
| P169 | SL | -51.87 | 101877.85 | 2 | 2026-06-15T22:35:35.460000+00:00 |
| P419 | SL | -49.36 | 151053.75 | 5 | 2026-06-22T23:36:41.186000+00:00 |
| P416 | SL | -48.03 | 154940.25 | 2 | 2026-06-22T13:44:10.354000+00:00 |
| P063 | SL | -47.76 | 153126.29 | 17 | 2026-06-11T18:00:24.402000+00:00 |
| P149 | SL | -46.3 | 185562.3 | 14 | 2026-06-15T13:25:20.594000+00:00 |
| P116 | SL | -45.82 | 58150.63 | 2 | 2026-06-14T22:44:24.574000+00:00 |
| P361 | TIME | -43.52 | 57157.49 | 1 | 2026-06-20T23:18:48.431000+00:00 |
| P394 | BE | -11.77 | 177012.84 | 11 | 2026-06-22T11:05:18.417000+00:00 |
| P058 | BE | -9.83 | 135373.6 | 5 | 2026-06-11T15:07:45.145000+00:00 |
| P391 | BE | -7.71 | 55976.47 | 7 | 2026-06-22T07:11:07.643000+00:00 |
| P353 | TIME | 10.4 | 92406.31 | 5 | 2026-06-20T22:12:43.788000+00:00 |
| P111 | TP | 116.02 | 55220.25 | 4 | 2026-06-14T21:35:16.845000+00:00 |

## Verdict

Strong in-sample shadow reject candidate for 50K weak clusters. Still shadow-only; promotion requires forward confirmation.
