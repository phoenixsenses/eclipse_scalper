# S34 Research Summary - 2026-06-16

Scope: read-only research over existing `data/microstructure.db`. No production runner/config changes are implied by this report.

Important model caveat: most replay numbers below use simplified mark-price fills and a flat 8 bps round-trip cost. They are route-discovery tools, not final validation. Live paper remains the authority for executable bid/ask fills, adverse selection, rule-scoped risk gates, cooldowns, and cursor sequencing.

## Executive Summary

Today produced a useful separation of S34 ideas:

| Idea | Status | Reason |
|---|---|---|
| BUY liquidation -> LONG momentum | Best current family | Strongest and cleanest evidence, especially the 200K/BTC-pre variant |
| BUY liquidation -> SHORT reversal | Killed for now | Best candidate still negative after costs |
| SELL liquidation -> SHORT continuation | Research-only, weak | Mild positive raw pocket but median-negative and TIME-heavy |
| SELL liquidation -> delayed LONG reversal | Research-only, interesting | Positive pocket exists, but fails badly on 2026-06-07 and needs a no-lookahead regime discriminator |

The cleanest forward-test candidate discovered today remains:

`ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60`

That should stay an exploratory paper variant separate from the pre-registered `50K/TP120` validation sample.

## 1. BUY-Liq Multi-Day Replay

Source: `S34_MULTI_DAY_REPLAY_2026-06-07_15.md`

The first replay compared the existing BUY-liq LONG family across four liquidation-data days: 2026-06-07, 06-11, 06-14, and 06-15.

Key results:

| Day | Rule | Regime-Pass Signals | Mean Net | Median Net | Cum Net | Read |
|---|---|---:|---:|---:|---:|---|
| 2026-06-07 | 50K/TP120 | 30 | -0.67 | -9.49 | -20.05 | noisy, weak |
| 2026-06-07 | 200K/TP60 | 7 | +31.60 | +53.31 | +221.23 | strong |
| 2026-06-11 | 50K/TP120 | 13 | -16.95 | -10.07 | -220.34 | bad |
| 2026-06-11 | 200K/TP60 | 6 | +1.08 | -8.31 | +6.48 | flat |
| 2026-06-14 | 50K/TP120 | 4 | +3.30 | -29.44 | +13.20 | mixed |
| 2026-06-14 | 200K/TP60 | 3 | -0.61 | -10.49 | -1.83 | flat |
| 2026-06-15 | 50K/TP120 | 20 | +44.98 | +61.84 | +899.67 | very strong |
| 2026-06-15 | 200K/TP60 | 14 | +30.60 | +52.52 | +428.46 | strong |

Main interpretation:

The 50K/TP120 pre-reg route is more fragile. It can do very well on a strong day, but it also admits many noisy lower-quality events. The 200K family is cleaner because the larger liquidation threshold filters out weak events before risk management has to deal with them.

## 2. Daily Regime Separation

Source: `S34_REGIME_SEPARATION_2026-06-07_15.md`

All four days were broadly classified as continuation-up:

| Day | ETH Trend | ETH Range | BTC Trend | ETH BUY Liq | ETH Agg Trades |
|---|---:|---:|---:|---:|---:|
| 2026-06-07 | +7.70% | 9.94% | +4.03% | 78.86M | 1,915,841 |
| 2026-06-11 | +3.20% | 4.43% | +3.44% | 15.54M | 1,349,692 |
| 2026-06-14 | +2.64% | 4.54% | +2.01% | 19.86M | 693,841 |
| 2026-06-15 | +4.08% | 8.12% | +0.88% | 70.98M | 1,395,473 |

This was important because a simple daily label did not explain performance. 06-11 had positive forward movement but still produced bad route outcomes for 50K/TP120. The problem was not "no movement"; it was path quality: drawdown, stop touches, and failure to convert MFE into TP.

## 3. BUY-Liq Signal Path Quality

Source: `S34_SIGNAL_PATH_QUALITY_2026-06-07_15.md`

Path quality explained the earlier contradiction. Some signals moved in the right direction eventually but first moved against the trade enough to trigger SL/BE.

Key path table:

| Day | Rule | N | MFE Mean | MAE Mean | BE Hit | TP Hit | SL Touch | BTC 15m Mean | Post-5m BUY Liq |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-06-07 | 200K/TP60 | 7 | +150.34 | -28.62 | 100.0% | 85.7% | 28.6% | +37.75 | 9.22M |
| 2026-06-07 | 50K/TP120 | 30 | +90.19 | -49.02 | 66.7% | 16.7% | 46.7% | +12.13 | 2.22M |
| 2026-06-11 | 200K/TP60 | 6 | +79.12 | -49.19 | 66.7% | 33.3% | 50.0% | +11.54 | 1.27M |
| 2026-06-11 | 50K/TP120 | 13 | +61.82 | -45.61 | 61.5% | 7.7% | 53.8% | +6.38 | 0.69M |
| 2026-06-15 | 200K/TP60 | 14 | +151.18 | -20.58 | 85.7% | 71.4% | 21.4% | +14.98 | 2.32M |
| 2026-06-15 | 50K/TP120 | 20 | +121.99 | -31.02 | 70.0% | 40.0% | 30.0% | +7.10 | 2.26M |

Main interpretation:

The larger 200K signal has better path geometry. It produces larger MFE, lower MAE, and higher TP-hit rates. The 50K signal has more false starts and deeper early pullbacks. This supports engineering stronger entry quality rather than trying to rescue every weak 50K event.

## 4. BUY-Liq No-Lookahead Filter Sweep

Source: `S34_NO_LOOKAHEAD_FILTER_SWEEP_2026-06-07_15.md`

The best no-lookahead candidate was:

`ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60`

Rules:

- ETH BUY liquidation cluster >= 200K in 1m bucket
- day-so-far regime filter passes
- BTC 15-minute pre-signal return >= 0 bps
- enter after 60 seconds
- TP120 / SL40 / BE30

Top result:

| Threshold | TP | Entry Delay | Filter | N | Days | Mean Net | Median Net | Cum Net | WR | Exits |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 200K | 120 | 60s | BTC pre-15m >= 0 | 17 | 4 | +65.32 | +112.12 | +1110.50 | 70.6% | BE 3 / TP 9 / TIME 4 / SL 1 |

This is the strongest research result of the day. It is not proof, because the replay is still simplified, but it is structurally coherent:

- stronger liquidation threshold;
- BTC confirmation available before the ETH signal;
- explicit entry delay;
- positive across four active days;
- TP120 survives better than TP80 in mean/cum.

Decision: keep it as an exploratory paper variant, not mixed into the pre-registered 50K/TP120 sample.

## 5. SELL-Liq -> SHORT Continuation

Sources:

- `S34_SELL_LIQ_REPLAY_2026-06-07_15.md`
- `S34_SELL_LIQ_FILTER_SWEEP_2026-06-07_15.md`
- `S34_SELL_LIQ_PATH_QUALITY_2026-06-07_15.md`

Raw SELL->SHORT had one weak pocket:

| Candidate | N | Days | Mean Net | Median Net | Cum Net | WR | Exits |
|---|---:|---:|---:|---:|---:|---:|---|
| SELL->SHORT 200K TP80 | 20 | 4 | +5.04 | -5.53 | +100.83 | 45.0% | TP 3 / TIME 12 / BE 3 / SL 2 |

Path-quality detail:

| Candidate | N | Mean MFE | Mean MAE | TP Touch | SL Touch | BE Hit |
|---|---:|---:|---:|---:|---:|---:|
| REPLAY_CLUSTER 200K TP80 | 20 | +44.20 | -18.75 | 20.0% | 30.0% | 70.0% |
| REPLAY_CLUSTER 200K TP60 | 20 | +40.79 | -18.75 | 30.0% | 30.0% | 70.0% |

This is not clean enough. Median is negative and TIME exits dominate. The positive mean is too thin to justify a live paper rule.

The filter sweep initially found positive-looking rows using 1m/2m short confirmation. But once confirmation was made deployable by entering after the confirmation, the edge disappeared. That means those apparent pockets were at least partly timing/lookahead artifacts.

Decision: SELL->SHORT continuation remains research-only.

## 6. SELL-Liq -> Delayed LONG Reversal

Sources:

- `S34_SELL_LIQ_REVERSAL_LONG_2026-06-07_15.md`
- `S34_SELL_REVERSAL_FILTER_SWEEP_2026-06-07_15.md`

This was the most interesting SELL-side result.

Main candidate:

`SELL_REVERSAL_LONG 200K TP40 DELAY300s`

Meaning:

- ETH SELL liquidation >= 200K;
- wait 300 seconds;
- enter LONG;
- TP40 / SL40 / BE30.

Result:

| N | Days | Mean Net | Median Net | Cum Net | WR | Mean MFE | Mean MAE | Exits |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 33 | 4 | +9.24 | +32.33 | +305.06 | 66.7% | +32.15 | -20.90 | TP 22 / BE 1 / SL 8 / TIME 2 |

Day split:

| Day | N | Cum Net | Mean Net | Median Net |
|---|---:|---:|---:|---:|
| 2026-06-07 | 10 | -165.76 | -16.58 | -48.47 |
| 2026-06-11 | 5 | +165.01 | +33.00 | +33.04 |
| 2026-06-14 | 5 | +81.57 | +16.31 | +32.02 |
| 2026-06-15 | 13 | +224.24 | +17.25 | +32.43 |

Outlier check: full cum was +305.06; removing the top 3 winners still left +202.70. So the positive result was not only one or two prints.

The problem is regime dependency. It worked on 06-11, 06-14, and 06-15, but failed badly on 06-07. That means the idea is not "SELL liq always means bounce." It is "SELL liq sometimes creates delayed bounce, but in stronger continuation regimes it catches a falling knife."

Filter sweep found some high-looking rows, for example:

| Filter | N | Days | Mean | Median | Cum | WR |
|---|---:|---:|---:|---:|---:|---:|
| eth_wait5_bps <= 20 AND btc_wait5_bps >= 0 | 8 | 4 | +22.59 | +32.83 | +180.70 | 87.5% |
| eth_pre5_bps >= -40 AND day_sell_liq_m <= 5 | 8 | 4 | +22.85 | +32.81 | +182.79 | 87.5% |
| btc_pre15_bps >= -60 AND btc_wait5_bps >= 0 | 15 | 4 | +21.92 | +32.63 | +328.79 | 86.7% |

These are too small to promote. Some have one trade on a day. The correct discipline is to keep this idea alive as research-only until more liquidation days exist.

Decision: SELL delayed reversal LONG is promising but not ready for active runner.

## 7. BUY-Liq -> SHORT Reversal

Source: `S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15.md`

This tested the opposite of the current BUY-liq LONG thesis: after very large BUY liquidations, does exhaustion produce a good SHORT?

Best result was still negative:

| Candidate | N | Days | Mean Net | Median Net | Cum Net | WR |
|---|---:|---:|---:|---:|---:|---:|
| BUY_REVERSAL_SHORT 500K TP40 DELAY600s | 59 | 4 | -7.77 | -9.40 | -458.64 | 33.9% |
| BUY_REVERSAL_SHORT 300K TP40 DELAY600s | 101 | 4 | -8.32 | -9.43 | -840.75 | 34.7% |
| BUY_REVERSAL_SHORT 200K TP40 DELAY600s | 152 | 4 | -9.20 | -9.46 | -1398.61 | 34.2% |

The result is useful because it supports the existing architecture. In this data window, BUY liquidation behaves like a momentum/continuation signal, not an exhaustion short signal. Waiting 10 minutes helps reduce immediate squeeze damage but still does not overcome costs.

Decision: kill BUY-liq reversal SHORT for now.

## 8. What We Learned Today

The strongest conclusion is not just "which strategy won." It is the engineering frame:

1. Liquidation side is not symmetric.

BUY liquidation and SELL liquidation do not mirror each other cleanly. BUY liquidation supports momentum LONG. SELL liquidation does not cleanly support immediate SHORT; it may sometimes support delayed LONG reversal.

2. Threshold quality matters.

The 200K threshold repeatedly improved quality versus 50K or 100K. Larger liquidation clusters are less frequent but have cleaner MFE/MAE geometry.

3. Path quality matters more than daily trend.

All four studied days looked like broad continuation-up days, but trade outcomes differed sharply. The useful separator is not simply "trend day." It is whether the specific signal path reaches favorable excursion before adverse stop pressure.

4. No-lookahead discipline changed conclusions.

Some filters looked attractive until entry was moved after the confirmation. Once the timing was made deployable, the edge disappeared. This is exactly why the research pipeline must enforce no-lookahead timing automatically.

5. The active BUY-liq LONG thesis became stronger, not weaker.

Testing the inverse idea, BUY-liq SHORT, failed. Testing SELL-liq SHORT also looked weak. The cleanest family remains BUY-liq LONG, especially with stronger threshold and BTC pre-confirmation.

## 9. Current Candidate Ranking

| Rank | Candidate | Action |
|---:|---|---|
| 1 | `ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60` | Keep exploratory paper / forward test |
| 2 | `ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30` | Useful benchmark/exploratory variant |
| 3 | `ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30` | Continue pre-registered validation only; do not judge early |
| 4 | `ETH_SELL_LIQ_LONG_200K_TP40_DELAY300_SL40_BE30` | Research-only, needs more days and filter |
| 5 | `ETH_SELL_LIQ_SHORT_200K_TP80_SL40_BE30` | Weak research-only |
| 6 | `ETH_BUY_LIQ_SHORT_*` | Killed for now |

## 10. Recommended Next Step

Build the S34 Feature Factory before adding more live rules.

The factory should produce one feature row per liquidation event or cluster:

- symbol;
- liquidation side;
- cluster notional;
- BTC pre-return windows;
- ETH pre-return windows;
- day-so-far trend/range/liquidation/agg counts;
- post-signal path metrics for research only;
- MFE/MAE;
- TP/SL/BE outcome under route templates;
- no-lookahead eligibility flags;
- live-feasible entry timestamp.

Then every hypothesis becomes a systematic query:

- BUY liq -> LONG continuation;
- BUY liq -> SHORT exhaustion;
- SELL liq -> SHORT continuation;
- SELL liq -> LONG reversal;
- cross-symbol versions;
- threshold and delay families.

This avoids hand-tuning. It lets us ask: which side/direction/threshold/delay survives costs, no-lookahead, day spread, median, and path quality?

## Final Decision

Do not add a SELL rule to the live runner yet.

Continue active BUY-liq paper collection.

Use today's work as the blueprint for a feature-factory script, then use that script to systematically engineer new S34 variants from existing liquidation data.
