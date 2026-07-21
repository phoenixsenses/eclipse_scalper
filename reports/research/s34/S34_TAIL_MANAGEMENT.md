# S34 Tail MANAGEMENT — does cutting the tail beat the chop?

_ETHUSDT SELL-flush -> LONG fade, >= 200K, 4h, n=676, fee 3.05/side._

## Rule comparison (ALL / TRAIN / TEST net bps)

| rule | n | mean | median | WR | MAX_LOSS | T3R | tr.mean | te.mean | tr.maxL | te.maxL | chop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HOLD_4h | 676 | 1.1 | 8.5 | 0.536 | -539.8 | -1.0 | 2.7 | -2.5 | -539.8 | -513.2 |  |
| TIME_STOP_30m | 676 | -1.8 | 3.6 | 0.527 | -461.6 | -3.0 | -3.6 | 2.4 | -280.5 | -461.6 |  |
| TIME_STOP_60m | 676 | -1.6 | 5.4 | 0.541 | -530.2 | -3.3 | -2.7 | 1.1 | -362.4 | -530.2 |  |
| TIME_STOP_120m | 676 | -0.7 | 6.3 | 0.531 | -439.5 | -2.4 | -1.7 | 1.7 | -350.6 | -439.5 |  |
| PRICE_STOP_80bps | 676 | 0.7 | -26.6 | 0.422 | -86.1 | -1.4 | 0.3 | 1.7 | -86.1 | -86.1 | w77/give263.6 |
| PRICE_STOP_120bps ⭐ | 676 | 3.8 | -0.9 | 0.496 | -126.1 | 1.7 | 3.1 | 5.4 | -126.1 | -126.1 | w27/give-1815.6 |
| PRICE_STOP_150bps ⭐ | 676 | 3.7 | 3.1 | 0.516 | -156.1 | 1.6 | 3.4 | 4.3 | -156.1 | -156.1 | w13/give-1715.9 |
| REACTIVE_cvd<0.0_exit5m | 676 | -7.0 | -6.6 | 0.396 | -495.0 | -9.1 | -7.9 | -5.0 | -495.0 | -440.4 | w241/give5499.2 |
| REACTIVE_cvd<0.0_exit15m | 676 | -7.1 | -2.8 | 0.472 | -495.0 | -9.1 | -8.6 | -3.5 | -495.0 | -440.4 | w241/give5527.2 |
| REACTIVE_cvd<-1.0_exit5m | 676 | -5.7 | -7.1 | 0.404 | -495.8 | -7.8 | -7.3 | -2.1 | -495.0 | -495.8 | w192/give4640.9 |
| REACTIVE_cvd<-1.0_exit15m | 676 | -6.3 | -3.8 | 0.47 | -495.8 | -8.3 | -8.4 | -1.3 | -495.0 | -495.8 | w192/give4982.2 |
| COMBINED_reactive+pstop150 | 676 | -30.3 | -20.9 | 0.367 | -322.0 | -32.5 | -32.2 | -25.8 | -322.0 | -253.2 | w229/give30038.0 |

⭐ = improves MAX_LOSS vs baseline AND keeps mean>=baseline AND train&test mean>=0.

**Chop ledger** `wN` = would-be winners (full-hold net>=0) chopped out; `giveX` = total net bps given up vs holding. A cut that saves the tail but has a large chop bill is net-negative.

_Read-only management diagnostic. Price-stop fills assume exit AT the stop level (optimistic; real stops slip). No edge claim._
