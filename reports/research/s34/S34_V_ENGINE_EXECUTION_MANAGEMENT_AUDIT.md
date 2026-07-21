# S34 V Engine Execution Management Audit

Mode: `READ_ONLY_RESEARCH_NO_LIVE_CHANGE`

## Live Env Snapshot (Read-Only)

- env planned notional/margin: `$1190.0` / `$29.8`
- tail-budget notional/margin: `$11.0` / `$0.3`
- leverage: `40.0x`
- configured stop: `150.0 bps`
- poll: `2.0s`
- kill switch file: `runtime/KILL_SWITCH`

## Stop Sweep

- baseline: N=23 sum=1112.6 med=37.0 T3R=433.6 maxL=-144.4
- current configured stop: `fixed_sl_150` -> N=23 sum=1081.3 med=37.0 T3R=402.3 maxL=-175.7, exit rate `0.043`
- best T3R stop: `fixed_sl_150` -> N=23 sum=1081.3 med=37.0 T3R=402.3 maxL=-175.7
- read: 150 bps is least destructive among tested hard stops, but it does not replace sizing

## Stop Budget Math

- research realized max loss at current stop: `-175.7 bps`
- current env notional loss at that stop: `$20.9` = `59.7%` equity
- tail-budget notional loss at that stop: `$0.2` = `0.6%` equity
- 40x liquidation adverse move approx: `250.0 bps`
- stress tail: `634.0 bps`

## Gap-Through / Realization

- nominal stop: `150.0 bps`
- observed worst realized stop loss: `-175.7 bps`
- gap+fee beyond nominal: `25.7 bps`
- read: stop is not guaranteed at nominal bps; book/taker exit can realize worse than trigger

## Atomicity / Kill Switch

- exchange-native stop-market: `True`
- reduce-only: `True`
- mark-price trigger: `True`
- stop repair path: `True`
- orphan emergency stop path: `True`
- kill switch blocks new entries: `True`
- finding: `NOT_ATOMIC_ENTRY_THEN_STOP_AFTER_FILL_DETECTION`
- read: entry limit is placed first; protective stop is submitted only after position detection in a later manage_active cycle

## Tail Frequency

- large-loss rate: `18.7%` (101/539)
- at least one tail probabilities: `{'at_least_one_tail_in_1_trades': 18.7, 'at_least_one_tail_in_3_trades': 46.3, 'at_least_one_tail_in_5_trades': 64.6, 'at_least_one_tail_in_10_trades': 87.4, 'at_least_one_tail_in_20_trades': 98.4}`

## Recommendations

- Do not change live logic automatically.
- Operator should reduce size to tail-budget or disarm before relying on any stop.
- Keep exchange-native STOP_MARKET; process-only exits are not outage-safe.
- Treat the entry-fill-to-stop-placement window as real atomicity risk.
- Kill switch should be operator-actionable: create runtime/KILL_SWITCH to block new entries.
