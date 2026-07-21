# S34 BUY-Side State-Machine Symmetry Gauntlet

Generated: `2026-07-01T08:19:11.191123+00:00`

Research-only. No live executor, `.env`, order logic, leverage, or sizing was changed.

## Side Semantics
- `ETH SELL liquidation` = long liquidation / forced sell.
- `ETH BUY liquidation` = short liquidation / forced buy.
- BUY-side tested directions:
  - `ETH BUY -> SHORT` = mean-reversion / fade after short squeeze.
  - `ETH BUY -> LONG` = continuation after short squeeze.

## Dataset
- ETH BUY 200K knowable anchors: `563`
- Date range: `2026-02-15T22:47:14.217000+00:00` -> `2026-07-01T04:45:52.552000+00:00`
- Candidate cells searched: `76`

## Top Candidates
| Candidate | Family | Dir | N | WR | Avg | Sum | T3R | Holdout N | Holdout Avg | Holdout T3R | No-overlap N | No-overlap T3R | Folds +sum/+t3r | Worst | TailN | Readiness |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| F_silence_short_h60 | BUY_TO_SHORT_FADE | SHORT | 184 | 67.9% | +22.7 | +4183.1 | +3426.3 | 55 | +20.4 | +638.4 | 171 | +3140.4 | 5/5 | -334.3 | 5 | SHADOW_ONLY_TAIL |
| F_silence_short_h120 | BUY_TO_SHORT_FADE | SHORT | 184 | 63.6% | +18.0 | +3307.8 | +2380.4 | 55 | +14.5 | +214.3 | 153 | +2264.5 | 5/3 | -548.1 | 16 | SHADOW_ONLY_TAIL |
| F_score3_silence_short_h60 | BUY_TO_SHORT_FADE | SHORT | 151 | 66.2% | +19.2 | +2906.6 | +2255.3 | 45 | +24.5 | +618.4 | 141 | +2038.6 | 5/4 | -334.3 | 5 | SHADOW_ONLY_TAIL |
| F_score3_silence_short_h120 | BUY_TO_SHORT_FADE | SHORT | 151 | 60.9% | +15.6 | +2348.8 | +1552.5 | 45 | +27.7 | +708.5 | 125 | +1747.0 | 4/3 | -548.1 | 13 | SHADOW_ONLY_TAIL |
| F_prebuild2_silence_short_h60 | BUY_TO_SHORT_FADE | SHORT | 54 | 70.4% | +33.4 | +1805.4 | +1286.8 | 16 | +41.9 | +388.0 | 54 | +1286.8 | 5/3 | -206.6 | 2 | SHADOW_ONLY_TAIL |
| F_echo45_120_silence_short_h240 | BUY_TO_SHORT_FADE | SHORT | 76 | 61.8% | +23.7 | +1798.1 | +797.7 | 23 | +59.3 | +661.2 | 55 | +262.2 | 3/3 | -466.6 | 7 | SHADOW_ONLY_TAIL |
| F_echo45_120_silence_short_h60 | BUY_TO_SHORT_FADE | SHORT | 77 | 70.1% | +21.9 | +1685.3 | +1040.1 | 23 | +37.9 | +390.7 | 74 | +971.2 | 4/2 | -334.3 | 5 | RESEARCH_ONLY_FOLD_WEAK |
| F_sync_lt200_regime_short_h60 | BUY_TO_SHORT_FADE | SHORT | 103 | 63.1% | +17.1 | +1765.2 | +1008.4 | 31 | +12.1 | -93.9 | 99 | +898.2 | 4/2 | -334.3 | 5 | RESEARCH_ONLY_HOLDOUT_WEAK |
| F_prebuild2_silence_short_h120 | BUY_TO_SHORT_FADE | SHORT | 54 | 70.4% | +23.2 | +1254.5 | +524.2 | 16 | +54.2 | +474.1 | 52 | +520.4 | 4/2 | -548.1 | 6 | RESEARCH_ONLY_FOLD_WEAK |
| F_sync_lt200_regime_short_h120 | BUY_TO_SHORT_FADE | SHORT | 103 | 59.2% | +11.1 | +1148.3 | +396.5 | 31 | +15.5 | -34.3 | 94 | +620.3 | 4/0 | -548.1 | 12 | RESEARCH_ONLY_HOLDOUT_WEAK |
| F_echo45_120_silence_short_h120 | BUY_TO_SHORT_FADE | SHORT | 77 | 61.0% | +14.8 | +1140.8 | +344.4 | 23 | +38.5 | +348.5 | 60 | -248.2 | 3/2 | -548.1 | 10 | RESEARCH_ONLY_FOLD_WEAK |
| C_score4_btc2000k_delay10_long_h120 | BUY_TO_LONG_CONT | LONG | 4 | 50.0% | -27.1 | -108.4 | -116.7 | 1 | -116.7 | -116.7 | 3 | -152.1 | 2/2 | -116.7 | 2 | LOW_N_RESEARCH_ONLY |
| C_score4_btc2000k_delay10_long_h60 | BUY_TO_LONG_CONT | LONG | 4 | 50.0% | -19.6 | -78.2 | -119.9 | 1 | -119.9 | -119.9 | 3 | -95.9 | 2/2 | -119.9 | 1 | LOW_N_RESEARCH_ONLY |
| C_score3_btc2000k_delay10_long_h240 | BUY_TO_LONG_CONT | LONG | 8 | 75.0% | +50.4 | +403.4 | -192.0 | 2 | -44.0 | -87.9 | 6 | -140.1 | 3/3 | -225.0 | 1 | LOW_N_RESEARCH_ONLY |
| C_score3_btc2000k_delay5_long_h120 | BUY_TO_LONG_CONT | LONG | 9 | 55.6% | +14.2 | +127.9 | -201.7 | 3 | -46.7 | -140.2 | 7 | -257.0 | 3/3 | -116.7 | 2 | LOW_N_RESEARCH_ONLY |
| C_score4_btc2000k_delay10_long_h180 | BUY_TO_LONG_CONT | LONG | 4 | 25.0% | -37.6 | -150.3 | -208.8 | 1 | -208.8 | -208.8 | 3 | -126.0 | 1/1 | -208.8 | 1 | LOW_N_RESEARCH_ONLY |
| C_score4_btc2000k_delay5_long_h60 | BUY_TO_LONG_CONT | LONG | 5 | 40.0% | -20.2 | -100.9 | -211.0 | 1 | -119.9 | -119.9 | 3 | -95.9 | 2/2 | -119.9 | 1 | LOW_N_RESEARCH_ONLY |
| F_silence_short_h240 | BUY_TO_SHORT_FADE | SHORT | 183 | 57.4% | +6.1 | +1124.5 | -213.9 | 55 | +20.6 | +598.0 | 125 | +435.6 | 4/1 | -668.3 | 22 | REJECT_T3R |
| C_score3_btc2000k_delay10_long_h180 | BUY_TO_LONG_CONT | LONG | 8 | 62.5% | +49.3 | +394.3 | -214.5 | 2 | -76.9 | -153.8 | 6 | -206.5 | 3/3 | -208.8 | 1 | LOW_N_RESEARCH_ONLY |
| C_score4_btc2000k_delay5_long_h120 | BUY_TO_LONG_CONT | LONG | 5 | 60.0% | -11.6 | -58.2 | -221.0 | 1 | -116.7 | -116.7 | 3 | -152.1 | 3/3 | -116.7 | 2 | LOW_N_RESEARCH_ONLY |
| C_score4_btc2000k_delay10_long_h240 | BUY_TO_LONG_CONT | LONG | 4 | 50.0% | -52.3 | -209.1 | -225.0 | 1 | -225.0 | -225.0 | 3 | -112.9 | 2/2 | -225.0 | 1 | LOW_N_RESEARCH_ONLY |
| C_score3_btc2000k_delay10_long_h120 | BUY_TO_LONG_CONT | LONG | 8 | 50.0% | +9.7 | +77.8 | -245.4 | 2 | -75.9 | -151.9 | 7 | -257.0 | 3/3 | -116.7 | 2 | LOW_N_RESEARCH_ONLY |
| C_score4_btc2000k_delay5_long_h180 | BUY_TO_LONG_CONT | LONG | 5 | 20.0% | -42.4 | -212.2 | -270.7 | 1 | -208.8 | -208.8 | 3 | -126.0 | 1/1 | -208.8 | 1 | LOW_N_RESEARCH_ONLY |
| C_score3_btc2000k_delay5_long_h180 | BUY_TO_LONG_CONT | LONG | 9 | 55.6% | +36.9 | +332.4 | -276.3 | 3 | -27.0 | -81.1 | 6 | -206.5 | 3/3 | -208.8 | 1 | LOW_N_RESEARCH_ONLY |
| C_score3_btc2000k_delay5_long_h240 | BUY_TO_LONG_CONT | LONG | 9 | 66.7% | +35.1 | +315.9 | -279.5 | 3 | -12.4 | -37.1 | 6 | -140.1 | 2/2 | -225.0 | 1 | LOW_N_RESEARCH_ONLY |
| C_score3_btc2000k_delay10_long_h60 | BUY_TO_LONG_CONT | LONG | 8 | 50.0% | -3.3 | -26.6 | -281.1 | 2 | -98.8 | -197.6 | 7 | -298.8 | 2/2 | -119.9 | 1 | LOW_N_RESEARCH_ONLY |
| C_score3_btc2000k_delay5_long_h60 | BUY_TO_LONG_CONT | LONG | 9 | 44.4% | -5.5 | -49.2 | -303.8 | 3 | -69.2 | -207.6 | 7 | -298.8 | 3/3 | -119.9 | 1 | LOW_N_RESEARCH_ONLY |
| C_score4_btc2000k_delay5_long_h240 | BUY_TO_LONG_CONT | LONG | 5 | 40.0% | -59.3 | -296.6 | -321.2 | 1 | -225.0 | -225.0 | 3 | -112.9 | 2/2 | -225.0 | 1 | LOW_N_RESEARCH_ONLY |
| F_echo45_120_silence_short_h180 | BUY_TO_SHORT_FADE | SHORT | 77 | 63.6% | +4.8 | +369.0 | -353.6 | 23 | +46.8 | +567.6 | 56 | -835.3 | 3/2 | -460.1 | 9 | REJECT_T3R |
| F_silence_short_h180 | BUY_TO_SHORT_FADE | SHORT | 184 | 59.2% | +2.7 | +491.4 | -400.8 | 55 | +17.4 | +445.8 | 133 | +256.2 | 2/1 | -460.1 | 19 | REJECT_T3R |

## Family Summary
| Family | Best candidate | N | WR | Avg | T3R | Readiness |
|---|---|---:|---:|---:|---:|---|
| BUY_TO_LONG_CONT | C_score4_btc2000k_delay10_long_h120 | 4 | 50.0% | -27.1 | -116.7 | LOW_N_RESEARCH_ONLY |
| BUY_TO_LONG_SAME_PROP | C_same_side_follow_long_h60 | 142 | 37.3% | -9.3 | -2176.7 | REJECT_T3R |
| BUY_TO_SHORT_FADE | F_silence_short_h60 | 184 | 67.9% | +22.7 | +3426.3 | SHADOW_ONLY_TAIL |

## Multiple-Comparison Permutation
```json
{
  "iterations": 500,
  "note": "max-stat permutation across searched candidate cells; conservative artifact check",
  "null_p95_max_t3r": 2197.2,
  "observed_max_t3r": 3426.3,
  "p_right": 0.01
}
```

## Direct Answers
- ETH BUY -> LONG continuation best: `C_score4_btc2000k_delay10_long_h120` LOW_N_RESEARCH_ONLY N=4 avg=-27.1 T3R=-116.7.
- ETH BUY -> LONG same-side propagation best: `C_same_side_follow_long_h60` REJECT_T3R N=142 avg=-9.3 T3R=-2176.7.
- ETH BUY -> SHORT mean-reversion best: `F_silence_short_h60` SHADOW_ONLY_TAIL N=184 avg=22.7 T3R=3426.3.
- No BUY-side cell reached PAPER_CANDIDATE under holdout + folds + no-overlap gates.
- Max-stat permutation says at least one searched BUY-side cell exceeds the 95% null; still needs forward shadow before live.

## State Diagnostics
### BUY_anchor_SHORT_4h_by_state
- `NOISY`: N=316 WR=45.3% avg=-36.6 T3R=-12870.8
- `SILENCE`: N=246 WR=55.7% avg=+0.0 T3R=-1327.9

### BUY_anchor_SHORT_4h_by_session
- `ASIA`: N=114 WR=49.1% avg=-8.8 T3R=-2038.9
- `EUROPE`: N=102 WR=49.0% avg=-37.2 T3R=-4808.8
- `OFF`: N=55 WR=45.5% avg=-54.7 T3R=-3530.1
- `US`: N=291 WR=51.2% avg=-12.9 T3R=-5087.5

### BUY_anchor_SHORT_4h_by_dow
- `Fri`: N=67 WR=56.7% avg=-0.6 T3R=-1175.0
- `Mon`: N=110 WR=45.5% avg=-63.1 T3R=-7663.9
- `Sat`: N=39 WR=46.2% avg=-15.3 T3R=-1210.4
- `Sun`: N=54 WR=38.9% avg=-45.1 T3R=-3119.0
- `Thu`: N=85 WR=60.0% avg=+23.2 T3R=+730.8
- `Tue`: N=88 WR=43.2% avg=-22.2 T3R=-2688.4
- `Wed`: N=119 WR=53.8% avg=-13.3 T3R=-2746.7
