# S34 Live vs Replay Parity Audit

Scope: replay selected closed live paper trades through the current runner evaluation path using the stored live signal/trade snapshot.

- Audited trades: `10`
- Exact parity: `10/10`

| Trade | Rule | Live Exit | Replay Exit | Live Net | Replay Net | Net Diff | Verdict | Reasons |
|---|---|---|---|---:|---:|---:|---|---|
| P187 | ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30 | TP 2026-06-16T12:31:15.005000+00:00 | TP 2026-06-16T12:31:15.005000+00:00 | +52.469373 | +52.469373 | +0.000000 | OK |  |
| P188 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | TP 2026-06-16T12:33:48+00:00 | TP 2026-06-16T12:33:48+00:00 | +125.727163 | +125.727163 | +0.000000 | OK |  |
| P189 | ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60 | BE 2026-06-16T12:40:38.007000+00:00 | BE 2026-06-16T12:40:38.007000+00:00 | -18.717407 | -18.717407 | +0.000000 | OK |  |
| P191 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | TP 2026-06-16T12:33:30.005000+00:00 | TP 2026-06-16T12:33:30.005000+00:00 | +49.860268 | +49.860268 | +0.000000 | OK |  |
| P192 | ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30 | TP 2026-06-16T12:33:30.005000+00:00 | TP 2026-06-16T12:33:30.005000+00:00 | +49.860268 | +49.860268 | +0.000000 | OK |  |
| P206 | ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30 | TP 2026-06-16T18:03:43.006000+00:00 | TP 2026-06-16T18:03:43.006000+00:00 | +49.351582 | +49.351582 | +0.000000 | OK |  |
| P217 | ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30 | SL 2026-06-17T02:34:16.001000+00:00 | SL 2026-06-17T02:34:16.001000+00:00 | -48.848344 | -48.848344 | +0.000000 | OK |  |
| P326 | SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | TP 2026-06-20T05:07:36.003000+00:00 | TP 2026-06-20T05:07:36.003000+00:00 | +46.999295 | +46.999295 | +0.000000 | OK |  |
| P328 | SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | TP 2026-06-20T05:10:15+00:00 | TP 2026-06-20T05:10:15+00:00 | +77.434174 | +77.434174 | +0.000000 | OK |  |
| P330 | ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30 | TP 2026-06-20T05:10:16+00:00 | TP 2026-06-20T05:10:16+00:00 | +72.693052 | +72.693052 | +0.000000 | OK |  |

## Read

A mismatch here means the live journal and current replay path do not agree for the same stored signal/trade snapshot. That does not automatically mean the trade PnL is wrong, but it blocks using research replay and live paper as interchangeable evidence until explained.
